#pragma once
#include <cstdint>

#include "hip/hip_runtime.h"

struct node
{
    struct node* next;
};

/* Util kernels */

auto launchSpinKernel(int           dev,
                      volatile int* latch,
                      volatile int* timedOut,
                      hipStream_t   stream,
                      uint64_t      timeoutMs) -> int;
void launchMemsetKernel(uint64_t* buf, uint64_t pattern, uint64_t count);
void launchVerifyKernel(const uint64_t* buf, uint64_t pattern, uint64_t count, bool* ok);

/* Work kernels */

auto launchCopyKernel(
    const void* src, void* dst, size_t size, uint64_t iterations, hipStream_t stream) -> uint64_t;

auto launchPtrChaseKernel(const void* data,
                          size_t      size,
                          uint32_t    accesses,
                          hipStream_t stream) -> uint64_t;