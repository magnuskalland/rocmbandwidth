#include "common.h"
#include "kernels.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

/* Util kernels */

static __global__ void spinKernelDevice(int dev, volatile int* latch, const uint64_t timeout)
{
    uint64_t endTime = clock64() + timeout;
    while (!*latch)
    {
        if (timeout != ~0ULL && clock64() > endTime)
        {
            printf("error: spin kernel timed out on device %d\n", dev);
            break;
        }
    }
}

auto launchSpinKernel(int dev, volatile int* latch, hipStream_t stream, uint64_t timeoutMs) -> int
{
    DeviceGuard guard;
    int         clocksPerMs = 0;
    HIP_CHECK(hipDeviceGetAttribute(&clocksPerMs, hipDeviceAttributeClockRate, dev));
    uint64_t timeout = static_cast<uint64_t>(clocksPerMs) * timeoutMs;

    HIP_CHECK(hipSetDevice(dev));
    spinKernelDevice<<<1, 1, 0, stream>>>(dev, latch, timeout);

    HIP_CHECK(hipGetLastError());
    return hipSuccess;
}

static __global__ void memsetKernel(uint64_t* buf, uint64_t pattern, uint64_t count)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        buf[idx] = pattern;
    }
}

void launchMemsetKernel(uint64_t* buf, uint64_t pattern, uint64_t count)
{
    if (count == 0)
    {
        return; /* A zero-block launch is an invalid configuration */
    }
    const int blockSize = 256;
    const int blocks    = (count + blockSize - 1) / blockSize;
    memsetKernel<<<blocks, blockSize>>>(buf, pattern, count);
}

static __global__ void verifyKernel(const uint64_t* buf, uint64_t pattern, uint64_t count, bool* ok)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && buf[idx] != pattern)
    {
        *ok = false;
    }
}

void launchVerifyKernel(const uint64_t* buf, uint64_t pattern, uint64_t count, bool* ok)
{
    if (count == 0)
    {
        return; /* A zero-block launch is an invalid configuration */
    }
    const int blockSize = 256;
    const int blocks    = (count + blockSize - 1) / blockSize;
    verifyKernel<<<blocks, blockSize>>>(buf, pattern, count, ok);
}

/* Work kernels */

template<int LD_ST_PIPELINES>
static __global__ void copyKernel(const float4* src,
                                  float4*       dst,
                                  uint64_t      perThreadChunkCount,
                                  uint64_t      stride,
                                  uint64_t      tailStart,
                                  uint64_t      tailCount,
                                  uint64_t      iterations)
{
    uint64_t      offset    = blockDim.x * blockIdx.x + threadIdx.x;
    const float4* offsetSrc = src + offset;
    float4*       offsetDst = dst + offset;
    float4*       pipelinedLoadEnd
        = (offsetDst + ((perThreadChunkCount / LD_ST_PIPELINES) * LD_ST_PIPELINES) * stride);
    float4* nonPipelinedLoadEnd = offsetDst + perThreadChunkCount * stride;

    for (uint64_t iter = 0; iter < iterations; iter++)
    {
        const float4* loadAddress  = offsetSrc;
        float4*       storeAddress = offsetDst;
        float4        registers[LD_ST_PIPELINES];

        /* Do pipelined load/stores */
        while (storeAddress < pipelinedLoadEnd)
        {
#pragma unroll
            for (int r = 0; r < LD_ST_PIPELINES; r++)
            {
                registers[r] = *loadAddress;
                loadAddress  = loadAddress + stride;
            }
#pragma unroll
            for (int r = 0; r < LD_ST_PIPELINES; r++)
            {
                *storeAddress = registers[r];
                storeAddress  = storeAddress + stride;
            }
        }

        /* Do the last load/stores that didn't fully fit in the pipeline slots */
        while (storeAddress < nonPipelinedLoadEnd)
        {
            *storeAddress = *loadAddress;
            storeAddress  = storeAddress + stride;
            loadAddress   = loadAddress + stride;
        }

        /* Chunks left over when the buffer doesn't divide evenly between all threads */
        if (offset < tailCount)
        {
            dst[tailStart + offset] = src[tailStart + offset];
        }
    }
}

/* Launch shape of the CU copy kernel. Measured on 2x MI210 (n016), one block per CU with no
 * load/store pipelining was best everywhere, but the best block size depends on the path:
 *   - both buffers in the executing GPU's own memory: 1024 threads (local copy ~615 GB/s,
 *     vs ~223 GB/s with 64 threads)
 *   - either buffer across PCIe (host or peer GPU): 64 threads (device->host ~9 GB/s and
 *     peer read ~18 GB/s, vs ~2-3 and ~13 GB/s with 1024 threads)
 * ROCMBW_CU_THREADS, ROCMBW_CU_BLOCKS_PER_CU and ROCMBW_CU_PIPELINES (1, 2, 4 or 8) override
 * the corresponding value for every path, for tuning on other hardware. */
struct CopyLaunchShape
{
    int threadsPerBlock;
    int blocksPerCU;
    int pipelines;
};

struct CopyLaunchShapes
{
    CopyLaunchShape local  = {1024, 1, 1};
    CopyLaunchShape remote = {64, 1, 1};
};

static auto envInt(const char* name, int min, int max) -> int
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return 0;
    }
    char* end = nullptr;
    long  v   = std::strtol(value, &end, 10);
    if (*end != '\0' || v < min || v > max)
    {
        fprintf(stderr, "error: %s=%s is not in [%d, %d]\n", name, value, min, max);
        exit(EXIT_FAILURE);
    }
    return static_cast<int>(v);
}

static auto copyLaunchShapes() -> const CopyLaunchShapes&
{
    static const CopyLaunchShapes shapes = []
    {
        CopyLaunchShapes s;
        int threads = envInt("ROCMBW_CU_THREADS", 1, 1024);
        int blocks  = envInt("ROCMBW_CU_BLOCKS_PER_CU", 1, 64);
        int pipes   = envInt("ROCMBW_CU_PIPELINES", 1, 8);
        if (pipes != 0 && (pipes & (pipes - 1)) != 0)
        {
            fprintf(stderr, "error: ROCMBW_CU_PIPELINES must be 1, 2, 4 or 8\n");
            exit(EXIT_FAILURE);
        }
        for (CopyLaunchShape* shape : {&s.local, &s.remote})
        {
            shape->threadsPerBlock = threads != 0 ? threads : shape->threadsPerBlock;
            shape->blocksPerCU     = blocks != 0 ? blocks : shape->blocksPerCU;
            shape->pipelines       = pipes != 0 ? pipes : shape->pipelines;
        }
        if (threads != 0 || blocks != 0 || pipes != 0)
        {
            fprintf(stderr,
                    "note: CU copy kernel launch overridden: local %d threads/block, %d blocks/CU, "
                    "%d pipelines; remote %d threads/block, %d blocks/CU, %d pipelines\n",
                    s.local.threadsPerBlock,
                    s.local.blocksPerCU,
                    s.local.pipelines,
                    s.remote.threadsPerBlock,
                    s.remote.blocksPerCU,
                    s.remote.pipelines);
        }
        return s;
    }();
    return shapes;
}

static auto isLocalDeviceMemory(const void* ptr, int dev) -> bool
{
    hipPointerAttribute_t attrs;
    if (hipPointerGetAttributes(&attrs, ptr) != hipSuccess)
    {
        (void)hipGetLastError();
        return false;
    }
    return attrs.type == hipMemoryTypeDevice && attrs.device == dev;
}

auto launchCopyKernel(
    const void* src, void* dst, size_t size, uint64_t iterations, hipStream_t stream) -> uint64_t
{
    hipDeviceProp_t p;
    int             dev;

    DeviceGuard guard;
    HIP_CHECK(hipStreamGetDevice(stream, &dev));
    HIP_CHECK(hipGetDeviceProperties(&p, dev));
    HIP_CHECK(hipSetDevice(dev));

    const bool             local = isLocalDeviceMemory(src, dev) && isLocalDeviceMemory(dst, dev);
    const CopyLaunchShape& shape = local ? copyLaunchShapes().local : copyLaunchShapes().remote;

    const int      blocks              = p.multiProcessorCount * shape.blocksPerCU;
    const uint64_t totalThreads        = static_cast<uint64_t>(blocks) * shape.threadsPerBlock;
    const uint64_t totalChunkCount     = size / sizeof(float4);
    const uint64_t realChunkCount      = totalThreads * (totalChunkCount / totalThreads);
    const uint64_t perThreadChunkCount = realChunkCount / totalThreads;

    auto s = reinterpret_cast<const float4*>(src);
    auto d = reinterpret_cast<float4*>(dst);
    auto t = totalChunkCount - realChunkCount;
    switch (shape.pipelines)
    {
        case 1:
            copyKernel<1><<<blocks, shape.threadsPerBlock, 0, stream>>>(
                s, d, perThreadChunkCount, totalThreads, realChunkCount, t, iterations);
            break;
        case 2:
            copyKernel<2><<<blocks, shape.threadsPerBlock, 0, stream>>>(
                s, d, perThreadChunkCount, totalThreads, realChunkCount, t, iterations);
            break;
        case 4:
            copyKernel<4><<<blocks, shape.threadsPerBlock, 0, stream>>>(
                s, d, perThreadChunkCount, totalThreads, realChunkCount, t, iterations);
            break;
        default:
            copyKernel<8><<<blocks, shape.threadsPerBlock, 0, stream>>>(
                s, d, perThreadChunkCount, totalThreads, realChunkCount, t, iterations);
            break;
    }

    return totalChunkCount * sizeof(float4);
}

static __global__ void ptrChaseKernel(const node** position, uint32_t accesses, int agent)
{
    if (blockIdx.x != agent)
    {
        return;
    }

    /* Continue from where the previous launch stopped, so successive launches walk new parts
     * of the chain instead of re-reading lines the previous launch just pulled into cache */
    volatile const node* n = *position;
    for (uint32_t i = 0; i < accesses; i++)
    {
        n = n->next;
    }
    *position = const_cast<const node*>(n);
}

auto launchPtrChaseKernel(const void* data,
                          size_t      size,
                          uint32_t    accesses,
                          hipStream_t stream) -> uint64_t
{
    (void)size;
    hipEvent_t      start, end;
    int             dev;
    float           totalSecs = 0.0;
    int             numComputeUnits;
    hipDeviceProp_t p;
    const node**    position;

    DeviceGuard guard;
    HIP_CHECK(hipStreamGetDevice(stream, &dev));
    HIP_CHECK(hipGetDeviceProperties(&p, dev));
    numComputeUnits = p.multiProcessorCount;

    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&position), sizeof(*position)));
    const node* first = reinterpret_cast<const node*>(data);
    HIP_CHECK(hipMemcpy(position, &first, sizeof(first), hipMemcpyHostToDevice));

    const uint32_t perLaunch = accesses / numComputeUnits;
    for (int cu = 0; cu < numComputeUnits; cu++)
    {
        float ms;
        HIP_CHECK(hipEventRecord(start, stream));
        ptrChaseKernel<<<numComputeUnits, 1, 0, stream>>>(position, perLaunch, cu);
        HIP_CHECK(hipEventRecord(end, stream));

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipEventElapsedTime(&ms, start, end));

        totalSecs += ms / 1000.0;
    }

    /* Divide by the accesses actually performed (accesses may not divide evenly) */
    uint64_t performed            = static_cast<uint64_t>(perLaunch) * numComputeUnits;
    uint64_t nanoSecondsPerAccess = performed ? (totalSecs * 1.0E9) / performed : 0;

    HIP_CHECK(hipFree(position));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(end));

    return nanoSecondsPerAccess;
}
