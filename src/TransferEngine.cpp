
#include "TransferEngine.hpp"
#include "Buffer.hpp"
#include "common.h"

#include <cmath>

auto TransferEngine::sequentialUnidirectional(const Buffer&   buffer,
                                              const Buffer&   peerBuffer,
                                              CopyAgentPolicy agent) -> double
{
    return sequentialBidirectional({&buffer}, {&peerBuffer}, agent);
}

auto TransferEngine::concurrentUnidirectional(const std::vector<const Buffer*>& buffers,
                                              const std::vector<const Buffer*>& peerBuffers,
                                              CopyAgentPolicy                   agent,
                                              std::vector<double>*              bws) -> void
{
    concurrentBidirectional({buffers}, {peerBuffers}, agent, bws);
}

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
                                       std::vector<hipEvent_t>&          endEvents) -> void
{
    auto nBuffers = sources.size();

    HIP_RETURN_VOID_ON_FAILURE(hipEventRecord(startEvents[0], streams[0]));
    for (size_t i = 1; i < nBuffers; i++)
    {
        HIP_RETURN_VOID_ON_FAILURE(hipStreamWaitEvent(streams[i], startEvents[0], 0));
        HIP_RETURN_VOID_ON_FAILURE(hipEventRecord(startEvents[i], streams[i]));
    }

    for (size_t i = 0; i < nBuffers; i++)
    {
        _executor->execute(sources[i]->getBuffer(),
                           destinations[i]->getBuffer(),
                           sources[i]->getSize(),
                           _iterations,
                           streams[i]);
        HIP_RETURN_VOID_ON_FAILURE(hipEventRecord(endEvents[i], streams[i]));
    }

    for (size_t i = 0; i < nBuffers; i++)
        HIP_RETURN_VOID_ON_FAILURE(hipStreamWaitEvent(streams[0], endEvents[i]));
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

auto SystemDMAEngine::sequentialBidirectional(const std::vector<const Buffer*>& buffers,
                                              const std::vector<const Buffer*>& peerBuffers,
                                              CopyAgentPolicy                   policy) -> double
{
    // if ((policy == CopyAgentPolicy::DESTINATION
    //      && !(supportsUnifiedAddressing(peerBuffers.at(0)->getDevice()))))
    // {
    //     return 0.0;
    // }

    auto                       nBuffers = static_cast<int>(buffers.size());
    std::vector<const Buffer*> sourceBuffers(buffers.size());
    std::vector<const Buffer*> destinationBuffers(buffers.size());
    std::vector<hipStream_t>   streams(nBuffers);
    std::vector<hipEvent_t>    startEvents(nBuffers);
    std::vector<hipEvent_t>    endEvents(nBuffers);
    std::vector<float>         times(nBuffers);
    std::vector<int>           agents(nBuffers);
    double                     totalBandwidth = 0.0;
    bool                       ok;

    for (int i = 0; i < nBuffers; i++)
    {
    }

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
    _enqueueTransfers(sourceBuffers, destinationBuffers, streams, startEvents, endEvents);

    /* Unblock spin kernels, start measurements */
    _executor->unblockStreams();

    /* Wait for completions */
    for (int i = 0; i < nBuffers; i++)
    {
        HIP_RETURN_DOUBLE_ON_FAILURE(hipStreamSynchronize(streams[i]));
    }

    /* Verify destination buffers */
    ok = _verify(destinationBuffers);
    if (!ok)
    {
        return 0.0;
    }

    for (int i = 0; i < nBuffers; i++)
    {
        HIP_RETURN_DOUBLE_ON_FAILURE(
            hipEventElapsedTime(&times.at(i), startEvents.at(i), endEvents.at(i)))
        double bandwidth
            = calculate_bandwidth(sourceBuffers.at(i)->getSize(), _iterations, times.at(i));
        totalBandwidth += bandwidth;
    }

    _cleanupTransfers(streams, startEvents, endEvents);

    return totalBandwidth;
}

auto SystemDMAEngine::concurrentBidirectional(
    const std::vector<std::vector<const Buffer*>>& bufferPairs,
    const std::vector<std::vector<const Buffer*>>& peerBufferPairs,
    CopyAgentPolicy                                policy,
    std::vector<double>*                           bandwidths) -> void
{

    // if ((policy == CopyAgentPolicy::DESTINATION
    //      && !(supportsUnifiedAddressing(peerBufferPairs.at(0).at(0)->getDevice()))))
    // {
    //     return;
    // }

    /* Pair size is 1 if called from concurrentUnidirectional and 2 if called raw */
    int pairSize = static_cast<int>(bufferPairs.size());
    int nBuffers = static_cast<int>(bufferPairs.at(0).size());

    std::vector<std::vector<const Buffer*>> sourceBufferPairs(pairSize);
    std::vector<std::vector<const Buffer*>> destinationBufferPairs(pairSize);
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

    for (int i = 0; i < pairSize; i++)
    {
        _enqueueTransfers(sourceBufferPairs.at(i),
                          destinationBufferPairs.at(i),
                          streamPairs.at(i),
                          startEventPairs.at(i),
                          endEventPairs.at(i));
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

    /* Verify destination buffers */
    for (int i = 0; i < pairSize; i++)
    {
        ok = _verify(destinationBufferPairs.at(i));
        if (!ok)
        {
            return;
        }
    }

    for (int i = 0; i < nBuffers; i++)
    {
        (*bandwidths)[i] = 0.0;
        for (int j = 0; j < pairSize; j++)
        {
            float  time;
            double bandwidth;

            HIP_RETURN_VOID_ON_FAILURE(
                hipEventElapsedTime(&time, startEventPairs.at(j).at(i), endEventPairs.at(j).at(i)))

            bandwidth = calculate_bandwidth(bufferPairs.at(j).at(i)->getSize(), _iterations, time);

            (*bandwidths)[i] += bandwidth;
        }
    }

    for (int i = 0; i < pairSize; i++)
    {
        _cleanupTransfers(streamPairs.at(i), startEventPairs.at(i), endEventPairs.at(i));
    }
}
