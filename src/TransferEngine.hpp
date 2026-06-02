#pragma once

#include "Buffer.hpp"
#include "TransferExecutor.hpp"
#include "common.h"
#include <memory>
#include <vector>

class TransferEngine
{
public:
    explicit TransferEngine(uint64_t iterations) : _iterations(iterations) {}
    virtual ~TransferEngine() = default;

    // virtual auto computeExecutionOutput(float time, ) -> double = 0;

    auto sequentialBidirectional(const std::vector<const Buffer*>& buffers,
                                 const std::vector<const Buffer*>& peerBuffers,
                                 CopyAgentPolicy                   agent) -> double;

    auto concurrentBidirectional(const std::vector<std::vector<const Buffer*>>& bufferPairs,
                                 const std::vector<std::vector<const Buffer*>>& peerBufferPairs,
                                 CopyAgentPolicy                                agent,
                                 std::vector<double>*                           bws) -> void;

    virtual auto sequentialUnidirectional(const Buffer&   buffer,
                                          const Buffer&   peerBuffer,
                                          CopyAgentPolicy agent) -> double;

    auto concurrentUnidirectional(const std::vector<const Buffer*>& buffers,
                                  const std::vector<const Buffer*>& peerBuffers,
                                  CopyAgentPolicy                   agent,
                                  std::vector<double>*              bws) -> void;

protected:
    std::unique_ptr<TransferExecutor> _executor;
    uint64_t                          _iterations;

    auto _prepareSourceBuffers(const std::vector<const Buffer*>& buffers) -> void;

    auto _prepareDestinationBuffers(const std::vector<const Buffer*>& buffers) -> void;

    auto _createStreamsAndEvents(const std::vector<int>&   agents,
                                 std::vector<hipStream_t>& streams,
                                 std::vector<hipEvent_t>&  startEvents,
                                 std::vector<hipEvent_t>&  endEvents) -> void;

    auto _enqueueTransfers(const std::vector<const Buffer*>& sources,
                           const std::vector<const Buffer*>& destinations,
                           std::vector<hipStream_t>&         streams,
                           std::vector<hipEvent_t>&          startEvents,
                           std::vector<hipEvent_t>&          endEvents,
                           std::vector<uint64_t>&            bytesCopied) -> void;

    auto _cleanupTransfers(std::vector<hipStream_t>& streams,
                           std::vector<hipEvent_t>&  startEvents,
                           std::vector<hipEvent_t>&  endEvents) -> void;

    auto _verify(const std::vector<const Buffer*>& buffers,
                 std::vector<uint64_t>             bytesCopied) -> bool;

private:
};

class LatencyTransferEngine : public TransferEngine
{
public:
    explicit LatencyTransferEngine(uint64_t iterations) : TransferEngine(iterations)
    {
        _executor = std::make_unique<LatencyExecutor>(0xDeadBeefCafeBabeULL);
    }
    ~LatencyTransferEngine() override = default;

    auto sequentialUnidirectional(const Buffer&   buffer,
                                  const Buffer&   peerBuffer,
                                  CopyAgentPolicy agent) -> double override;
};

class BandwidthTransferEngine : public TransferEngine
{
public:
    explicit BandwidthTransferEngine(uint64_t iterations) : TransferEngine(iterations) {}
    ~BandwidthTransferEngine() override = default;
};

class SystemDMATransferEngine : public BandwidthTransferEngine
{
public:
    explicit SystemDMATransferEngine(uint64_t iterations) : BandwidthTransferEngine(iterations)
    {
        _executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    };
};

class ComputeUnitTransferEngine : public BandwidthTransferEngine
{
public:
    explicit ComputeUnitTransferEngine(uint64_t iterations) : BandwidthTransferEngine(iterations)
    {
        _executor = std::make_unique<ComputeUnitExecutor>(0xDeadBeefCafeBabeULL);
    };
};
