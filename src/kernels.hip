#include "common.h"
#include "kernels.hpp"
#include <hip/hip_runtime.h>

/* Util kernels */

__global__ void spinKernelDevice(volatile int* latch, const uint64_t timeout)
{
    uint64_t endTime = clock64() + timeout;
    while (!*latch)
    {
        if (timeout != ~0ULL && clock64() > endTime)
        {
            printf("error: spin kernel timed out\n");
            break;
        }
    }
}

auto launchSpinKernel(int dev, volatile int* latch, hipStream_t stream, uint64_t timeoutMs) -> int
{
    DeviceGuard guard;
    int clocksPerMs = 0;
    HIP_CHECK(hipDeviceGetAttribute(&clocksPerMs, hipDeviceAttributeClockRate, dev));
    uint64_t timeout = static_cast<uint64_t>(clocksPerMs) * timeoutMs;

    HIP_CHECK(hipSetDevice(dev));
    spinKernelDevice<<<1, 1, 0, stream>>>(latch, timeout);

    HIP_CHECK(hipGetLastError());
    return hipSuccess;
}

__global__ void memsetKernel(uint64_t* buf, uint64_t pattern, uint64_t count)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        buf[idx] = pattern;
    }
}

void launchMemsetKernel(uint64_t* buf, uint64_t pattern, uint64_t count)
{
    const int blockSize = 256;
    const int blocks    = (count + blockSize - 1) / blockSize;
    memsetKernel<<<blocks, blockSize>>>(buf, pattern, count);
}

__global__ void verifyKernel(const uint64_t* buf, uint64_t pattern, uint64_t count, bool* ok)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && buf[idx] != pattern)
    {
        *ok = false;
    }
}

void launchVerifyKernel(const uint64_t* buf, uint64_t pattern, uint64_t count, bool* ok)
{
    const int blockSize = 256;
    const int blocks    = (count + blockSize - 1) / blockSize;
    verifyKernel<<<blocks, blockSize>>>(buf, pattern, count, ok);
}

/* Work kernels */