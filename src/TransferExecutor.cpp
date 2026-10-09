
#include "TransferExecutor.hpp"
#include "common.h"
#include "kernels.hpp"
#include <algorithm>
#include <random>

auto TransferExecutor::getAgent(const Buffer& src, const Buffer& dst, CopyAgentPolicy agent) -> int
{
    /* Host memory can't issue a transfer, so a host-side buffer always falls back to the
     * GPU at the other end. Bidirectional tests swap source/destination for the reverse
     * direction, so the compound policies reduce to "issue from the source" (write/push)
     * or "issue from the destination" (read/pull) of each individual transfer. */
    bool pull = false;
    switch (agent)
    {
        case CopyAgentPolicy::DESTINATION:
        case CopyAgentPolicy::DESTINATION_AND_SOURCE: pull = true; break;
        case CopyAgentPolicy::HOST:
        case CopyAgentPolicy::SOURCE:
        case CopyAgentPolicy::SOURCE_AND_HOST:
        case CopyAgentPolicy::HOST_AND_DESTINATION:
        case CopyAgentPolicy::SOURCE_AND_DESTINATION: pull = false; break;
    }

    const Buffer& preferred = pull ? dst : src;
    const Buffer& fallback  = pull ? src : dst;
    return preferred.getDevice() != -1 ? preferred.getDevice() : fallback.getDevice();
}

auto TransferExecutor::resetStreams() -> void
{
    /* Closed latch */
    *_latch = 0;
}

auto TransferExecutor::blockStreams(std::vector<hipStream_t> streams) -> void
{
    for (auto stream : streams)
    {
        int dev;
        HIP_CHECK(hipStreamGetDevice(stream, &dev));
        launchSpinKernel(dev, _latch, stream, 1000);
    }
}

auto TransferExecutor::unblockStreams() -> void
{
    /* Open latch */
    *_latch = 1;
}

auto TransferExecutor::resetSourceBuffer(void* buf, uint64_t size) -> void
{
    _memset(buf, size, _sourcePattern);
}

auto TransferExecutor::resetDestBuffer(void* buf, uint64_t size) -> void
{
    _memset(buf, size, _clearPattern);
}

[[nodiscard]] auto TransferExecutor::verify(void* buf, uint64_t size) -> bool
{
    uint64_t    count = size / sizeof(uint64_t);
    DeviceGuard guard;

    hipPointerAttribute_t attrs;
    HIP_CHECK(hipPointerGetAttributes(&attrs, buf));

    if (attrs.type == hipMemoryTypeHost)
    {
        auto* p = reinterpret_cast<uint64_t*>(buf);
        return std::all_of(p, p + count, [&](uint64_t v) -> bool { return v == _sourcePattern; });
    }
    else
    {
        bool  hostOk = true;
        bool* deviceOk;
        HIP_CHECK(hipSetDevice(attrs.device));
        HIP_CHECK(hipMalloc(&deviceOk, sizeof(bool)));
        HIP_CHECK(hipMemcpy(deviceOk, &hostOk, sizeof(bool), hipMemcpyHostToDevice));
        launchVerifyKernel(reinterpret_cast<uint64_t*>(buf), _sourcePattern, count, deviceOk);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(&hostOk, deviceOk, sizeof(bool), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(deviceOk));
        return hostOk;
    }
}

auto TransferExecutor::_memset(void* buf, uint64_t size, uint64_t pattern) -> void
{
    uint64_t              count = size / sizeof(pattern);
    hipPointerAttribute_t attrs;
    DeviceGuard           guard;
    HIP_CHECK(hipPointerGetAttributes(&attrs, buf));

    if (attrs.type == hipMemoryTypeHost)
    {
        std::fill_n(reinterpret_cast<uint64_t*>(buf), count, pattern);
    }
    else
    {
        HIP_CHECK(hipSetDevice(attrs.device));
        launchMemsetKernel(reinterpret_cast<uint64_t*>(buf), pattern, count);
        HIP_CHECK(hipDeviceSynchronize());
    }
}

auto LatencyExecutor::resetSourceBuffer(void* buf, uint64_t size) -> void
{
    const int        stride = 16; /* GPU cache line size in words */
    const int        nptrs  = size / stride;
    node*            nodes  = reinterpret_cast<node*>(buf);
    std::vector<int> perm(nptrs);
    for (int i = 0; i < nptrs; i++)
    {
        perm[i] = i;
    }

    std::mt19937 rng(12345);
    std::shuffle(perm.begin(), perm.end(), rng);

    for (int i = 0; i < nptrs; i++)
    {
        int next_i          = perm[(i + 1) % nptrs];
        nodes[perm[i]].next = &nodes[next_i];
    }
}

auto LatencyExecutor::execute(
    const void* src, void* dst, uint64_t size, uint64_t accesses, hipStream_t stream) -> uint64_t
{
    (void)dst;
    uint64_t lat = launchPtrChaseKernel(src, size, accesses, stream);
    return lat;
}

auto SystemDMAExecutor::execute(
    const void* src, void* dst, uint64_t size, uint64_t iterations, hipStream_t stream) -> uint64_t
{
    for (uint64_t i = 0; i < iterations; i++)
    {
        HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDefault, stream));
    }
    return size;
}

auto ComputeUnitExecutor::execute(
    const void* src, void* dst, uint64_t size, uint64_t iterations, hipStream_t stream) -> uint64_t
{
    return launchCopyKernel(src, dst, size, iterations, stream);
}