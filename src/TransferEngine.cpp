
#include "TransferEngine.hpp"
#include "Buffer.hpp"
#include "common.h"

#include <boost/assert.hpp>
#include <cmath>

auto TransferEngine::sequentialUnidirectional(const Buffer&   buffer,
                                              const Buffer&   peerBuffer,
                                              CopyAgentPolicy agent) -> double
{
    return sequentialBirectional({&buffer}, {&peerBuffer}, agent);
}

auto TransferEngine::concurrentUnidirectional(const std::vector<const Buffer*>& buffers,
                                              const std::vector<const Buffer*>& peerBuffers,
                                              CopyAgentPolicy                   agent,
                                              std::vector<double>*              bws) -> void
{
    concurrentBidirectional(buffers, peerBuffers, agent, bws);
}

auto TransferEngine::_prepareSourceBuffers(const std::vector<const Buffer*>& buffers) -> void
{
    /* Memset a pattern into the source buffers */
    for (auto buf : buffers)
    {
        _executor->resetSourceBuffer(buf->getBuffer(), buf->getSize());
    }
}

auto TransferEngine::_prepareTransfers(const std::vector<const Buffer*>& buffers,
                                       const std::vector<const Buffer*>& peerBuffers,
                                       CopyAgentPolicy                   policy,
                                       std::vector<hipStream_t>&         streams,
                                       std::vector<hipEvent_t>&          startEvents,
                                       std::vector<hipEvent_t>&          endEvents) -> void
{
    auto             nBuffers = buffers.size();
    std::vector<int> agents(nBuffers);

    for (decltype(nBuffers) i = 0; i < nBuffers; i++)
    {
        DeviceGuard guard;
        agents[i] = _executor->getAgent(*buffers.at(i), *peerBuffers.at(i), policy);
        HIP_RETURN_VOID_ON_FAILURE(hipSetDevice(agents[i]));
        HIP_RETURN_VOID_ON_FAILURE(hipStreamCreate(&streams.at(i)));
        HIP_RETURN_VOID_ON_FAILURE(hipEventCreate(&startEvents.at(i)));
        HIP_RETURN_VOID_ON_FAILURE(hipEventCreate(&endEvents.at(i)));
    }

    _executor->resetStreams();

    for (decltype(nBuffers) i = 0; i < nBuffers; i++)
    {
        _executor->blockStream(agents[i], streams.at(i));
    }

    for (decltype(nBuffers) i = 0; i < nBuffers; i++)
    {
        if (i == 0)
        {
            HIP_RETURN_VOID_ON_FAILURE(hipEventRecord(startEvents[0], streams[0]));
        }
        else
        {
            HIP_RETURN_VOID_ON_FAILURE(hipStreamWaitEvent(streams[i], startEvents[0], 0));
            HIP_RETURN_VOID_ON_FAILURE(hipEventRecord(startEvents[i], streams[i]));
        }
    }

    for (decltype(nBuffers) i = 0; i < nBuffers; i++)
    {
        void*       src      = buffers.at(i)->getBuffer();
        void*       dst      = peerBuffers.at(i)->getBuffer();
        uint64_t    size     = buffers.at(i)->getSize();
        hipStream_t stream   = streams.at(i);
        hipEvent_t  endEvent = endEvents.at(i);

        BOOST_ASSERT(buffers.at(i)->getSize() == peerBuffers.at(i)->getSize());
        _executor->execute(src, dst, size, _iterations, stream);
        HIP_RETURN_VOID_ON_FAILURE(hipEventRecord(endEvent, stream));
    }

    for (decltype(nBuffers) i = 0; i < nBuffers; i++)
    {
        HIP_RETURN_VOID_ON_FAILURE(hipStreamWaitEvent(streams[0], endEvents[i]));
    }
}

auto TransferEngine::_prepareDestinationBuffers(const std::vector<const Buffer*>& buffers) -> void
{
    /* Zero out the destination buffers */
    for (auto buf : buffers)
    {
        _executor->resetDestBuffer(buf->getBuffer(), buf->getSize());
    }
}

auto TransferEngine::_cleanupTransfers(std::vector<hipStream_t>& streams,
                                       std::vector<hipEvent_t>&  startEvents,
                                       std::vector<hipEvent_t>&  endEvents) -> void
{
    auto nBuffers = streams.size();
    for (uint64_t i = 0; i < nBuffers; i++)
    {
        HIP_CHECK(hipStreamDestroy(streams.at(i)));
        HIP_CHECK(hipEventDestroy(startEvents.at(i)));
        HIP_CHECK(hipEventDestroy(endEvents.at(i)));
    }
}

auto TransferEngine::_verify(const std::vector<const Buffer*>& buffers) -> bool
{
    for (auto buf : buffers)
    {
        bool ok = _executor->verify(buf->getBuffer(), buf->getSize());
        if (!ok)
        {
            fprintf(stderr, "error: destination buffer corrupt\n");
            return false;
        }
    }
    return true;
}

auto SystemDMAEngine::sequentialBirectional(const std::vector<const Buffer*>& buffers,
                                            const std::vector<const Buffer*>& peerBuffers,
                                            CopyAgentPolicy                   policy) -> double
{
    if ((policy == CopyAgentPolicy::DESTINATION
         && !(supportsUnifiedAddressing(peerBuffers.at(0)->getDevice()))))
    {
        return 0.0;
    }

    BOOST_ASSERT(buffers.size() == peerBuffers.size());
    BOOST_ASSERT(buffers.size() > 0);

    auto                     nBuffers = buffers.size();
    std::vector<hipStream_t> streams(nBuffers);
    std::vector<hipEvent_t>  startEvents(nBuffers);
    std::vector<hipEvent_t>  endEvents(nBuffers);
    std::vector<float>       times(nBuffers);
    double                   bandwidth = 0.0;
    bool                     ok;

    _prepareSourceBuffers(buffers);
    _prepareDestinationBuffers(peerBuffers);
    _prepareTransfers(buffers, peerBuffers, policy, streams, startEvents, endEvents);

    /* Unblock spin kernels, start measurements */
    _executor->unblockStreams();

    /* Wait for completions */
    for (decltype(nBuffers) i = 0; i < nBuffers; i++)
    {
        if (!buffers[i])
        {
            continue;
        }
        HIP_RETURN_DOUBLE_ON_FAILURE(hipStreamSynchronize(streams[i]));
    }

    /* Verify desination buffers */
    ok = _verify(peerBuffers);
    if (!ok)
    {
        return 0.0;
    }

    for (decltype(nBuffers) i = 0; i < nBuffers; i++)
    {
        if (!buffers[i])
        {
            continue;
        }
        HIP_RETURN_DOUBLE_ON_FAILURE(
            hipEventElapsedTime(&times.at(i), startEvents.at(i), endEvents.at(i)))
        bandwidth += calculate_bandwidth(buffers.at(i)->getSize(), _iterations, times.at(i));
    }

    _cleanupTransfers(streams, startEvents, endEvents);

    return bandwidth;
}

auto SystemDMAEngine::concurrentBidirectional(const std::vector<const Buffer*>& buffers,
                                              const std::vector<const Buffer*>& peerBuffers,
                                              CopyAgentPolicy                   policy,
                                              std::vector<double>*              bandwidths) -> void
{

    if ((policy == CopyAgentPolicy::DESTINATION
         && !(supportsUnifiedAddressing(peerBuffers.at(0)->getDevice()))))
    {
        return;
    }

    BOOST_ASSERT(buffers.size() == peerBuffers.size());
    BOOST_ASSERT(buffers.size() > 0);

    auto                     nBuffers = buffers.size();
    std::vector<hipStream_t> streams(nBuffers);
    std::vector<hipEvent_t>  startEvents(nBuffers);
    std::vector<hipEvent_t>  endEvents(nBuffers);
    std::vector<float>       times(nBuffers);
    bool                     ok;

    _prepareSourceBuffers(buffers);
    _prepareDestinationBuffers(peerBuffers);
    _prepareTransfers(buffers, peerBuffers, policy, streams, startEvents, endEvents);

    /* Unblock spin kernels, start measurements */
    _executor->unblockStreams();

    /* Wait for completions */
    for (decltype(nBuffers) i = 0; i < nBuffers; i++)
    {
        HIP_RETURN_VOID_ON_FAILURE(hipStreamSynchronize(streams[i]));
    }

    /* Verify desination buffers */
    ok = _verify(peerBuffers);
    if (!ok)
    {
        return;
    }

    for (decltype(nBuffers) i = 0; i < nBuffers; i++)
    {
        HIP_RETURN_VOID_ON_FAILURE(
            hipEventElapsedTime(&times.at(i), startEvents.at(i), endEvents.at(i)))
        (*bandwidths)[i] = calculate_bandwidth(buffers.at(i)->getSize(), _iterations, times.at(i));
    }

    _cleanupTransfers(streams, startEvents, endEvents);
}
