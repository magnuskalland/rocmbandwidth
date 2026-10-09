
#include "TransferEngine.hpp"
#include "Buffer.hpp"
#include "common.h"

#include <algorithm>
#include <cmath>

auto TransferEngine::_prepareSourceBuffers(const std::vector<const Buffer*>& buffers) -> void
{
    /* Memset a pattern into the source buffers */
    for (auto buf : buffers)
    {
        _executor->resetSourceBuffer(buf->getBuffer(), buf->getSize());
    }
}

auto TransferEngine::_createStreamsAndEvents(const std::vector<int>&   agents,
                                             std::vector<hipStream_t>& streams,
                                             std::vector<hipEvent_t>&  startEvents,
                                             std::vector<hipEvent_t>&  endEvents) -> void
{
    for (int i = 0; i < static_cast<int>(streams.size()); i++)
    {
        DeviceGuard guard;
        HIP_RETURN_VOID_ON_FAILURE(hipSetDevice(agents[i]));
        HIP_RETURN_VOID_ON_FAILURE(hipStreamCreate(&streams[i]));
        HIP_RETURN_VOID_ON_FAILURE(hipEventCreate(&startEvents[i]));
        HIP_RETURN_VOID_ON_FAILURE(hipEventCreate(&endEvents[i]));
    }
}

auto TransferEngine::_enqueueTransfers(const std::vector<const Buffer*>& sources,
                                       const std::vector<const Buffer*>& destinations,
                                       std::vector<hipStream_t>&         streams,
                                       std::vector<hipEvent_t>&          startEvents,
                                       std::vector<hipEvent_t>&          endEvents,
                                       std::vector<uint64_t>&            bytesCopied) -> void
{
    auto nBuffers = sources.size();

    /* Warmup */
    for (size_t i = 0; i < nBuffers; i++)
    {
        _executor->execute(sources[i]->getBuffer(),
                           destinations[i]->getBuffer(),
                           sources[i]->getSize(),
                           4,
                           streams[i]);
    }

    /* Barrier: no stream starts its timed copies until every stream has finished its warmup.
     * Otherwise a stream whose warmup is queued behind another stream's timed copies (e.g. on
     * a shared SDMA engine) starts its clock late, and transfers that actually ran one after
     * the other are each timed as if they had the link to themselves. */
    std::vector<hipEvent_t> warmupDone(nBuffers);
    for (size_t i = 0; i < nBuffers; i++)
    {
        DeviceGuard guard;
        int         dev;
        HIP_RETURN_VOID_ON_FAILURE(hipStreamGetDevice(streams[i], &dev));
        HIP_RETURN_VOID_ON_FAILURE(hipSetDevice(dev));
        HIP_RETURN_VOID_ON_FAILURE(hipEventCreateWithFlags(&warmupDone[i], hipEventDisableTiming));
        HIP_RETURN_VOID_ON_FAILURE(hipEventRecord(warmupDone[i], streams[i]));
    }
    for (size_t i = 0; i < nBuffers; i++)
    {
        for (size_t j = 0; j < nBuffers; j++)
        {
            if (i != j)
            {
                HIP_RETURN_VOID_ON_FAILURE(hipStreamWaitEvent(streams[i], warmupDone[j], 0));
            }
        }
        HIP_RETURN_VOID_ON_FAILURE(hipEventRecord(startEvents[i], streams[i]));
    }
    for (auto event : warmupDone)
    {
        /* Safe while still pending: resources are released once the event completes */
        HIP_RETURN_VOID_ON_FAILURE(hipEventDestroy(event));
    }

    for (size_t i = 0; i < nBuffers; i++)
    {
        auto src       = sources[i]->getBuffer();
        auto dst       = destinations[i]->getBuffer();
        auto size      = sources[i]->getSize();
        auto stream    = streams[i];
        bytesCopied[i] = _executor->execute(src, dst, size, _iterations, stream);
        HIP_RETURN_VOID_ON_FAILURE(hipEventRecord(endEvents[i], stream));
    }

    for (size_t i = 0; i < nBuffers; i++)
    {
        HIP_RETURN_VOID_ON_FAILURE(hipStreamWaitEvent(streams[0], endEvents[i], 0));
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

auto TransferEngine::_verify(const std::vector<const Buffer*>& buffers,
                             std::vector<uint64_t>             bytesCopied) -> bool
{
    for (uint64_t i = 0; i < buffers.size(); i++)
    {
        const Buffer* buf   = buffers.at(i);
        uint64_t      bytes = bytesCopied.at(i);
        bool          ok    = _executor->verify(buf->getBuffer(), bytes);

        if (!ok)
        {
            fprintf(stderr, "error: destination buffer corrupt\n");
            return false;
        }
    }
    return true;
}

auto TransferEngine::sequentialUnidirectional(const Buffer&   buffer,
                                              const Buffer&   peerBuffer,
                                              CopyAgentPolicy agent) -> double
{
    return sequentialBidirectional({&buffer}, {&peerBuffer}, agent);
}

auto TransferEngine::concurrentUnidirectional(const std::vector<const Buffer*>& buffers,
                                              const std::vector<const Buffer*>& peerBuffers,
                                              CopyAgentPolicy                   agent,
                                              std::vector<double>*              bws,
                                              double*                           aggregate) -> void
{
    concurrentBidirectional({buffers}, {peerBuffers}, agent, bws, aggregate);
}

auto LatencyTransferEngine::sequentialUnidirectional(const Buffer&   buffer,
                                                     const Buffer&   peerBuffer,
                                                     CopyAgentPolicy policy) -> double
{
    hipStream_t stream;
    uint64_t    lat;
    int         agent = _executor->getAgent(buffer, peerBuffer, policy);

    _prepareSourceBuffers({&buffer});

    DeviceGuard guard;
    HIP_CHECK(hipSetDevice(agent));
    HIP_CHECK(hipStreamCreate(&stream));
    lat = _executor->execute(buffer.getBuffer(),
                             peerBuffer.getBuffer(),
                             buffer.getSize(),
                             _iterations,
                             stream);

    HIP_CHECK(hipStreamDestroy(stream));
    return lat;
};

auto TransferEngine::sequentialBidirectional(const std::vector<const Buffer*>& buffers,
                                             const std::vector<const Buffer*>& peerBuffers,
                                             CopyAgentPolicy                   policy) -> double
{
    auto                       nBuffers = static_cast<int>(buffers.size());
    std::vector<const Buffer*> sourceBuffers(buffers.size());
    std::vector<const Buffer*> destinationBuffers(buffers.size());
    std::vector<uint64_t>      bytesCopied(nBuffers);
    std::vector<hipStream_t>   streams(nBuffers);
    std::vector<hipEvent_t>    startEvents(nBuffers);
    std::vector<hipEvent_t>    endEvents(nBuffers);
    std::vector<float>         times(nBuffers);
    std::vector<int>           agents(nBuffers);
    double                     totalBandwidth = 0.0;
    bool                       ok;

    for (int i = 0; i < nBuffers; i++)
    {
        if (i % 2 == 0)
        {
            sourceBuffers[i]      = buffers.at(i);
            destinationBuffers[i] = peerBuffers.at(i);
        }
        else
        {
            sourceBuffers[i]      = peerBuffers.at(i);
            destinationBuffers[i] = buffers.at(i);
        }

        DeviceGuard guard;
        agents[i] = _executor->getAgent(*sourceBuffers.at(i), *destinationBuffers.at(i), policy);
    }

    _prepareSourceBuffers(sourceBuffers);
    _prepareDestinationBuffers(destinationBuffers);
    _createStreamsAndEvents(agents, streams, startEvents, endEvents);
    _executor->resetStreams();
    _executor->blockStreams(streams);
    _enqueueTransfers(sourceBuffers,
                      destinationBuffers,
                      streams,
                      startEvents,
                      endEvents,
                      bytesCopied);

    /* Unblock spin kernels, start measurements */
    _executor->unblockStreams();

    /* Wait for completions */
    for (int i = 0; i < nBuffers; i++)
    {
        HIP_RETURN_DOUBLE_ON_FAILURE(hipStreamSynchronize(streams[i]));
    }

    /* If a spin kernel gave up before the host released it, the streams didn't start together
     * (usually because enqueueing the transfers blocked on the host), so the timing is invalid */
    if (_executor->blockTimedOut())
    {
        fprintf(stderr, "error: streams were released by the spin-kernel timeout; result invalid\n");
        _cleanupTransfers(streams, startEvents, endEvents);
        return 0.0;
    }

    /* Verify destination buffers */
    ok = _verify(destinationBuffers, bytesCopied);
    if (!ok)
    {
        _cleanupTransfers(streams, startEvents, endEvents);
        return 0.0;
    }

    /* All transfers start at a common barrier, so the combined bandwidth is the total number of
     * bytes over the time until the last transfer finished. Summing per-transfer bandwidths
     * instead would over-count whenever the transfers don't finish at the same time. */
    uint64_t totalBytes = 0;
    float    maxTime    = 0.0f;
    for (int i = 0; i < nBuffers; i++)
    {
        HIP_RETURN_DOUBLE_ON_FAILURE(
            hipEventElapsedTime(&times.at(i), startEvents.at(i), endEvents.at(i)));
        totalBytes += bytesCopied[i];
        maxTime = std::max(maxTime, times.at(i));
    }
    totalBandwidth = calculate_bandwidth(totalBytes, _iterations, maxTime);

    _cleanupTransfers(streams, startEvents, endEvents);

    return totalBandwidth;
}

auto TransferEngine::concurrentBidirectional(
    const std::vector<std::vector<const Buffer*>>& bufferPairs,
    const std::vector<std::vector<const Buffer*>>& peerBufferPairs,
    CopyAgentPolicy                                policy,
    std::vector<double>*                           bandwidths,
    double*                                        aggregate) -> void
{
    /* Pair size is 1 if called from concurrentUnidirectional and 2 if called raw */
    int pairSize = static_cast<int>(bufferPairs.size());
    int nBuffers = static_cast<int>(bufferPairs.at(0).size());

    std::vector<std::vector<const Buffer*>> sourceBufferPairs(pairSize);
    std::vector<std::vector<const Buffer*>> destinationBufferPairs(pairSize);
    std::vector<uint64_t>                   bytesCopied(nBuffers);
    std::vector<std::vector<hipStream_t>>   streamPairs(pairSize);
    std::vector<std::vector<hipEvent_t>>    startEventPairs(pairSize);
    std::vector<std::vector<hipEvent_t>>    endEventPairs(pairSize);
    std::vector<std::vector<int>>           agentPairs(pairSize);
    bool                                    ok;

    _executor->resetStreams();

    for (int i = 0; i < pairSize; i++)
    {
        sourceBufferPairs.at(i).resize(nBuffers);
        destinationBufferPairs.at(i).resize(nBuffers);
        streamPairs.at(i).resize(nBuffers);
        startEventPairs.at(i).resize(nBuffers);
        endEventPairs.at(i).resize(nBuffers);
        agentPairs.at(i).resize(nBuffers);

        for (int j = 0; j < nBuffers; j++)
        {
            if (i % 2 == 0)
            {
                sourceBufferPairs[i][j]      = bufferPairs[i][j];
                destinationBufferPairs[i][j] = peerBufferPairs[i][j];
            }
            else
            {
                sourceBufferPairs[i][j]      = peerBufferPairs[i][j];
                destinationBufferPairs[i][j] = bufferPairs[i][j];
            }

            DeviceGuard guard;
            agentPairs[i][j] = _executor->getAgent(*sourceBufferPairs[i][j],
                                                   *destinationBufferPairs[i][j],
                                                   policy);
        }

        _prepareSourceBuffers(sourceBufferPairs.at(i));
        _prepareDestinationBuffers(destinationBufferPairs.at(i));
        _createStreamsAndEvents(agentPairs.at(i),
                                streamPairs.at(i),
                                startEventPairs.at(i),
                                endEventPairs.at(i));
    }

    for (int i = 0; i < pairSize; i++)
    {
        _executor->blockStreams(streamPairs.at(i));
    }

    /* Enqueue both directions in one call so all transfers share a common start barrier */
    std::vector<const Buffer*> flatSources;
    std::vector<const Buffer*> flatDestinations;
    std::vector<hipStream_t>   flatStreams;
    std::vector<hipEvent_t>    flatStartEvents;
    std::vector<hipEvent_t>    flatEndEvents;
    std::vector<uint64_t>      flatBytesCopied(static_cast<size_t>(pairSize) * nBuffers);
    for (int i = 0; i < pairSize; i++)
    {
        auto append
            = [](auto& to, const auto& from) { to.insert(to.end(), from.begin(), from.end()); };
        append(flatSources, sourceBufferPairs.at(i));
        append(flatDestinations, destinationBufferPairs.at(i));
        append(flatStreams, streamPairs.at(i));
        append(flatStartEvents, startEventPairs.at(i));
        append(flatEndEvents, endEventPairs.at(i));
    }
    _enqueueTransfers(flatSources,
                      flatDestinations,
                      flatStreams,
                      flatStartEvents,
                      flatEndEvents,
                      flatBytesCopied);
    for (int j = 0; j < nBuffers; j++)
    {
        bytesCopied[j] = flatBytesCopied[j];
    }

    /* Unblock spin kernels, start measurements */
    _executor->unblockStreams();

    /* Wait for completions */
    for (int i = 0; i < pairSize; i++)
    {
        for (int j = 0; j < nBuffers; j++)
        {
            HIP_RETURN_VOID_ON_FAILURE(hipStreamSynchronize(streamPairs.at(i).at(j)));
        }
    }

    /* See sequentialBidirectional(): a spin-kernel timeout invalidates the timing */
    if (_executor->blockTimedOut())
    {
        fprintf(stderr, "error: streams were released by the spin-kernel timeout; result invalid\n");
        for (int i = 0; i < pairSize; i++)
        {
            _cleanupTransfers(streamPairs.at(i), startEventPairs.at(i), endEventPairs.at(i));
        }
        return;
    }

    /* Verify destination buffers */
    for (int i = 0; i < pairSize; i++)
    {
        ok = _verify(destinationBufferPairs.at(i), bytesCopied);
        if (!ok)
        {
            for (int j = 0; j < pairSize; j++)
            {
                _cleanupTransfers(streamPairs.at(j), startEventPairs.at(j), endEventPairs.at(j));
            }
            return;
        }
    }

    /* Every transfer starts at the common barrier, so a link's bandwidth is its total bytes over
     * the time until its slowest direction finished, and the aggregate is all bytes over the
     * time until the very last transfer finished. Summing per-transfer bandwidths would
     * over-count whenever transfers sharing a link or copy engine finish at different times. */
    uint64_t allBytes   = 0;
    float    allMaxTime = 0.0f;
    for (int i = 0; i < nBuffers; i++)
    {
        uint64_t linkBytes   = 0;
        float    linkMaxTime = 0.0f;
        for (int j = 0; j < pairSize; j++)
        {
            float time;
            HIP_RETURN_VOID_ON_FAILURE(
                hipEventElapsedTime(&time, startEventPairs.at(j).at(i), endEventPairs.at(j).at(i)));
            linkBytes += bytesCopied[i];
            linkMaxTime = std::max(linkMaxTime, time);
        }
        (*bandwidths)[i] = calculate_bandwidth(linkBytes, _iterations, linkMaxTime);
        allBytes += linkBytes;
        allMaxTime = std::max(allMaxTime, linkMaxTime);
    }
    if (aggregate != nullptr)
    {
        *aggregate = calculate_bandwidth(allBytes, _iterations, allMaxTime);
    }

    for (int i = 0; i < pairSize; i++)
    {
        _cleanupTransfers(streamPairs.at(i), startEventPairs.at(i), endEventPairs.at(i));
    }
}
