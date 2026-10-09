#pragma once

#include <cstdint>

#include "hip/hip_runtime.h"
#include <vector>

#include "Buffer.hpp"
#include "common.h"

class TransferExecutor
{
public:
    TransferExecutor(const TransferExecutor&)                    = delete;
    auto operator=(const TransferExecutor&) -> TransferExecutor& = delete;

    explicit TransferExecutor(uint64_t pattern) : _sourcePattern(pattern)
    {
        HIP_CHECK(hipHostMalloc((void**)&_latch, sizeof(int), hipHostMallocDefault));
    }

    virtual ~TransferExecutor()
    {
        HIP_CHECK(hipHostFree((void*)_latch));
    }

    virtual auto execute(const void* src,
                         void*       dst,
                         uint64_t    size,
                         uint64_t    iterations,
                         hipStream_t stream) -> uint64_t = 0;

    virtual auto resetSourceBuffer(void* buf, uint64_t size) -> void;

    auto verify(void* buf, uint64_t size) -> bool;
    auto getAgent(const Buffer& src, const Buffer& dst, CopyAgentPolicy agent) -> int;
    auto resetStreams() -> void;
    auto blockStreams(std::vector<hipStream_t> streams) -> void;
    auto unblockStreams() -> void;
    auto resetDestBuffer(void* buf, uint64_t size) -> void;

protected:
    uint64_t      _sourcePattern;
    uint64_t      _clearPattern = 0ULL;
    volatile int* _latch;

private:
    auto _memset(void* buf, uint64_t size, uint64_t pattern) -> void;
};

class LatencyExecutor : public TransferExecutor
{
public:
    explicit LatencyExecutor(uint64_t pattern) : TransferExecutor(pattern) {}
    auto execute(const void* dev, void* data, uint64_t size, uint64_t accesses, hipStream_t stream)
        -> uint64_t override;
    auto resetSourceBuffer(void* buf, uint64_t size) -> void override;
};

class SystemDMAExecutor : public TransferExecutor
{
public:
    explicit SystemDMAExecutor(uint64_t pattern) : TransferExecutor(pattern) {}
    auto execute(const void* src, void* dst, uint64_t size, uint64_t iterations, hipStream_t stream)
        -> uint64_t override;
};

class ComputeUnitExecutor : public TransferExecutor
{
public:
    explicit ComputeUnitExecutor(uint64_t pattern) : TransferExecutor(pattern) {}
    auto execute(const void* src, void* dst, uint64_t size, uint64_t iterations, hipStream_t stream)
        -> uint64_t override;
};