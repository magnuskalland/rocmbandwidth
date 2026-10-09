#pragma once

#include <cstdint>
#include <iostream>

#include "hip/hip_runtime.h"

#define PRINT_ERR(e)                                                                        \
    std::cerr << "HIP Error: " << hipGetErrorString(e) << " in " << __FILE__ << " at line " \
              << __LINE__ << std::endl;

#define HIP_EXIT_ON_FAILURE(command) \
    do                               \
    {                                \
        hipError_t _e = (command);   \
        if (_e != hipSuccess)        \
        {                            \
            PRINT_ERR(_e)            \
            exit(1);                 \
        }                            \
    }                                \
    while (0)

#define HIP_RETURN_DOUBLE_ON_FAILURE(command) \
    do                                        \
    {                                         \
        hipError_t _e = (command);            \
        if (_e != hipSuccess)                 \
        {                                     \
            PRINT_ERR(_e)                     \
            return 0.0;                       \
        }                                     \
    }                                         \
    while (0)

#define HIP_RETURN_VOID_ON_FAILURE(command) \
    do                                      \
    {                                       \
        hipError_t _e = (command);          \
        if (_e != hipSuccess)               \
        {                                   \
            PRINT_ERR(_e)                   \
            return;                         \
        }                                   \
    }                                       \
    while (0)

#define HIP_CHECK(command)         \
    do                             \
    {                              \
        hipError_t _e = (command); \
        if (_e != hipSuccess)      \
        {                          \
            PRINT_ERR(_e)          \
            std::abort();          \
        }                          \
    }                              \
    while (0)

#define calculate_bandwidth(size, count, ms) \
    (std::round(((size * count) / 1000000000.0) / (ms / 1000.0) * 100.0) / 100.0)

enum class CopyAgentPolicy
{
    HOST,
    SOURCE,
    DESTINATION,
    SOURCE_AND_HOST,
    HOST_AND_DESTINATION,
    SOURCE_AND_DESTINATION,
    DESTINATION_AND_SOURCE,
};

struct config
{
    int      device_count = 0;
    uint32_t iterations   = (16 << 0);
    uint64_t bw_bufsize   = (512ULL << 20);
    uint64_t lat_bufsize  = (2ULL << 20);
    uint32_t lat_accesses = 1000000;
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
