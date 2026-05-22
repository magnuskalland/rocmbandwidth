
#include "TransferExecutor.hpp"
#include "common.h"
#include "kernels.hpp"

auto TransferExecutor::getAgent(const Buffer& src, const Buffer& dst, CopyAgentPolicy agent) -> int
{
    BOOST_ASSERT(!(src.getDevice() == -1 && dst.getDevice() == -1));

    if (agent == CopyAgentPolicy::SOURCE)
    {
        return src.getDevice();
    }

    if (agent == CopyAgentPolicy::DESTINATION)
    {
        return dst.getDevice();
    }

    if (src.getDevice() == -1)
    {
        return dst.getDevice();
    }

    return src.getDevice();
}

auto TransferExecutor::resetStreams() -> void
{
    /* Open latch */
    *_latch = 0;
}

auto TransferExecutor::unblockStreams() -> void
{
    /* Closed latch */
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
        return std::all_of(p, p + count, [&](uint64_t v) { return v == _sourcePattern; });
    }
    else
    {
        bool  hostOk = true;
        bool* deviceOk;
        HIP_CHECK(hipSetDevice(attrs.device));
        HIP_CHECK(hipMalloc(&deviceOk, sizeof(bool)));
        HIP_CHECK(hipMemcpy(deviceOk, &hostOk, sizeof(bool), hipMemcpyHostToDevice));
        launchVerifyKernel(reinterpret_cast<uint64_t*>(buf), _sourcePattern, count, deviceOk);
        HIP_CHECK(hipGetLastError()); // ← catch launch failures
        HIP_CHECK(hipDeviceSynchronize()); // ← ensure kernel completes before memcpy back
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

auto SystemDMAExecutor::execute(
    const void* src, void* dst, uint64_t size, uint64_t iterations, hipStream_t stream) -> void
{
    for (uint64_t i = 0; i < iterations; i++)
    {
        HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDefault, stream));
    }
}

auto SystemDMAExecutor::blockStream(int dev, hipStream_t stream) -> void
{
    HIP_CHECK(launchSpinKernel(dev, _latch, stream, 10));
}