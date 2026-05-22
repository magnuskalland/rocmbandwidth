
#include <algorithm>
#include <cstdint>

#include "hip/hip_runtime.h"
#include <boost/assert.hpp>
#include <vector>

#include "Buffer.hpp"
#include "common.h"

class TransferExecutor
{
public:
    explicit TransferExecutor(uint64_t pattern) : _sourcePattern(pattern)
    {
        HIP_CHECK(hipHostMalloc((void**)&_latch, sizeof(int), hipHostMallocDefault));
    }

    ~TransferExecutor()
    {
        HIP_CHECK(hipHostFree((void*)_latch));
    }

    virtual auto execute(const void* src,
                         void*       dst,
                         uint64_t    size,
                         uint64_t    iterations,
                         hipStream_t stream) -> void = 0;

    auto getAgent(const Buffer& src, const Buffer& dst, CopyAgentPolicy agent) -> int;
    auto resetStreams() -> void;
    auto blockStreams(std::vector<hipStream_t> streams) -> void;
    auto unblockStreams() -> void;
    auto resetSourceBuffer(void* buf, uint64_t size) -> void;
    auto resetDestBuffer(void* buf, uint64_t size) -> void;

    [[nodiscard]] auto verify(void* buf, uint64_t size) -> bool;

protected:
    uint64_t      _sourcePattern;
    uint64_t      _clearPattern = 0ULL;
    volatile int* _latch;

private:
    auto _memset(void* buf, uint64_t size, uint64_t pattern) -> void;
};

class SystemDMAExecutor : public TransferExecutor
{
public:
    explicit SystemDMAExecutor(uint64_t pattern) : TransferExecutor(pattern) {}
    auto execute(const void* src, void* dst, uint64_t size, uint64_t iterations, hipStream_t stream)
        -> void override;
};