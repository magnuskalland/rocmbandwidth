#pragma once

#include <cstdint>
#include <iostream>

#include "hip/hip_runtime.h"

#define PRINT_ERR()                                                                          \
    std::cerr << "HIP Error: " << hipGetErrorString(hipGetLastError()) << " in " << __FILE__ \
              << " at line " << __LINE__ << std::endl;

#define HIP_EXIT_ON_FAILURE(command) \
    if ((command) != hipSuccess)     \
    {                                \
        PRINT_ERR()                  \
        exit(1);                     \
    }

#define HIP_RETURN_DOUBLE_ON_FAILURE(command) \
    if ((command) != hipSuccess)              \
    {                                         \
        PRINT_ERR()                           \
        return 0.0;                           \
    }

#define HIP_RETURN_VOID_ON_FAILURE(command) \
    if ((command) != hipSuccess)            \
    {                                       \
        PRINT_ERR()                         \
        return;                             \
    }

#define HIP_CHECK(command)       \
    if ((command) != hipSuccess) \
    {                            \
        PRINT_ERR()              \
    }

#define calculate_bandwidth(size, count, ms) \
    (std::round(((size * count) / 1000000000.0) / (ms / 1000.0) * 100.0) / 100.0)

enum class CopyAgentPolicy
{
    HOST,
    SOURCE,
    DESTINATION
};

struct config
{
    int      device_count;
    uint32_t iterations = (1 << 4);
    uint32_t bw_bytes   = (512 << 20);
    uint32_t lat_bytes  = sizeof(void*);
};

struct DeviceGuard
{
    int previous;
    DeviceGuard()
    {
        HIP_EXIT_ON_FAILURE(hipGetDevice(&previous));
    }
    ~DeviceGuard()
    {
        HIP_EXIT_ON_FAILURE(hipSetDevice(previous));
    }
};

inline auto supportsUnifiedAddressing(int dev) -> bool
{
    if (dev == -1)
    {
        return false;
    }
    hipDeviceProp_t p;
    HIP_EXIT_ON_FAILURE(hipGetDeviceProperties(&p, dev));
    return p.unifiedAddressing;
}
