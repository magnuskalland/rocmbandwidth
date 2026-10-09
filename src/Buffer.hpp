#pragma once

#include <cstdint>

#include "hip/hip_runtime.h"

#include "common.h"

class Buffer
{
public:
    explicit Buffer(int device, uint64_t size) : _device(device), _size(size) {}
    virtual ~Buffer() = default;

    Buffer(const Buffer&)                    = delete;
    auto operator=(const Buffer&) -> Buffer& = delete;
    Buffer(Buffer&&)                         = delete;
    auto operator=(Buffer&&) -> Buffer&      = delete;

    [[nodiscard]] auto getDevice() const -> int
    {
        return _device;
    }

    [[nodiscard]] auto getBuffer() const -> void*
    {
        return _buffer;
    }

    [[nodiscard]] auto getSize() const -> uint64_t
    {
        return _size;
    }

protected:
    int      _device;
    void*    _buffer = nullptr;
    uint64_t _size;
};

class HostBuffer : public Buffer
{
public:
    HostBuffer(int targetDevice, uint64_t size) : Buffer(-1, size)
    {
        DeviceGuard guard;
        HIP_EXIT_ON_FAILURE(hipSetDevice(targetDevice));
        HIP_EXIT_ON_FAILURE(hipHostMalloc(&_buffer, _size, hipHostMallocMapped));
    }

    ~HostBuffer() override
    {
        if (_buffer != nullptr)
        {
            HIP_CHECK(hipHostFree(_buffer));
        }
    }
};

class DeviceBuffer : public Buffer
{
public:
    DeviceBuffer(int device, uint64_t size) : Buffer(device, size)
    {
        DeviceGuard guard;
        HIP_EXIT_ON_FAILURE(hipSetDevice(device));
        HIP_EXIT_ON_FAILURE(hipMalloc(&_buffer, size));
    }

    ~DeviceBuffer() override
    {
        if (_buffer != nullptr)
        {
            DeviceGuard guard;
            HIP_CHECK(hipSetDevice(_device));
            HIP_CHECK(hipFree(_buffer));
        }
    }
};
