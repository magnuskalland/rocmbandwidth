#include "common.h"
#include "kernels.hpp"
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
    }
}

auto launchCopyKernel(
    const void* src, void* dst, size_t size, uint64_t iterations, hipStream_t stream) -> uint64_t
{
    hipDeviceProp_t p;
    int             dev;
    int             numComputeUnits;
    int             threadsPerBlock;
    int             totalThreads;
    uint64_t        totalChunkCount;
    uint64_t        realChunkCount;
    uint64_t        perThreadChunkCount;

    DeviceGuard guard;
    HIP_CHECK(hipStreamGetDevice(stream, &dev));
    HIP_CHECK(hipGetDeviceProperties(&p, dev));
    HIP_CHECK(hipSetDevice(dev));

    numComputeUnits     = p.multiProcessorCount;
    threadsPerBlock     = 8;
    totalThreads        = numComputeUnits * threadsPerBlock;
    totalChunkCount     = size / sizeof(float4);
    realChunkCount      = totalThreads * (totalChunkCount / totalThreads);
    perThreadChunkCount = realChunkCount / totalThreads;

    copyKernel<1>
        <<<numComputeUnits, threadsPerBlock, 0, stream>>>(reinterpret_cast<const float4*>(src),
                                                          reinterpret_cast<float4*>(dst),
                                                          perThreadChunkCount,
                                                          totalThreads,
                                                          iterations);

    return realChunkCount * sizeof(float4);
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
