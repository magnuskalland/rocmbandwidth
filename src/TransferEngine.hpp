#pragma once

#include "Buffer.hpp"
#include "TransferExecutor.hpp"
#include "common.h"
#include <memory>
#include <vector>

class TransferEngine
{
public:
    TransferEngine(std::unique_ptr<TransferExecutor> executor, uint64_t iterations)
        : _executor(std::move(executor)), _iterations(iterations)
    {}

    virtual auto sequentialBirectional(const std::vector<const Buffer*>& buffers,
                                       const std::vector<const Buffer*>& peerBuffers,
                                       CopyAgentPolicy                   agent) -> double = 0;

    virtual auto concurrentBidirectional(const std::vector<const Buffer*>& buffers,
                                         const std::vector<const Buffer*>& peerBuffers,
                                         CopyAgentPolicy                   agent,
                                         std::vector<double>*              bws) -> void = 0;

    auto sequentialUnidirectional(const Buffer&   buffer,
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

    auto _prepareTransfers(const std::vector<const Buffer*>& buffers,
                           const std::vector<const Buffer*>& peerBuffers,
                           CopyAgentPolicy                   policy,
                           std::vector<hipStream_t>&         streams,
                           std::vector<hipEvent_t>&          startEvents,
                           std::vector<hipEvent_t>&          endEvents) -> void;

    auto _cleanupTransfers(std::vector<hipStream_t>& streams,
                           std::vector<hipEvent_t>&  startEvents,
                           std::vector<hipEvent_t>&  endEvents) -> void;

    auto _verify(const std::vector<const Buffer*>& buffers) -> bool;
};

class SystemDMAEngine : public TransferEngine
{
public:
    using TransferEngine::concurrentUnidirectional;
    using TransferEngine::sequentialUnidirectional;
    SystemDMAEngine(std::unique_ptr<TransferExecutor> executor, uint64_t iterations)
        : TransferEngine(std::move(executor), iterations)
    {}

    auto sequentialBirectional(const std::vector<const Buffer*>& buffers,
                               const std::vector<const Buffer*>& peerBuffers,
                               CopyAgentPolicy                   agent) -> double override;

    auto concurrentBidirectional(const std::vector<const Buffer*>& buffers,
                                 const std::vector<const Buffer*>& peerBuffers,
                                 CopyAgentPolicy                   agent,
                                 std::vector<double>*              bws) -> void override;
};
