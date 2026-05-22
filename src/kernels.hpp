#pragma once
#include <cstdint>

#include "hip/hip_runtime.h"

/* Util kernels */

auto launchSpinKernel(int dev, volatile int* latch, hipStream_t stream, uint64_t timeoutMs) -> int;
void launchMemsetKernel(uint64_t* buf, uint64_t pattern, uint64_t count);
void launchVerifyKernel(const uint64_t* buf, uint64_t pattern, uint64_t count, bool* ok);

/* Work kernels */