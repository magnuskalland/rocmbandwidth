#pragma once

#include "TransferEngine.hpp"
#include "common.h"
#include <sstream>
#include <string>
#include <utility>

class Testcase
{
public:
    Testcase(const struct config&            conf,
             std::unique_ptr<TransferEngine> engine,
             std::string                     key,
             std::string                     desc)
        : _conf(conf), _engine(std::move(engine)), _key(std::move(key)), _desc(std::move(desc))
    {}

    virtual void execute() = 0;

    auto getKey() -> const char*
    {
        return _key.c_str();
    }

    auto getDesc() -> const char*
    {
        return _desc.c_str();
    }

    auto getResult() -> std::string
    {
        return _output.str();
    }

protected:
    virtual void startEngine(CopyAgentPolicy policy) = 0;

    const struct config&            _conf;
    std::unique_ptr<TransferEngine> _engine;
    std::stringstream               _output;

private:
    std::string _key;
    std::string _desc;
};

/* Sequential/Concurrent, Unidirectional/Bidirectional, Latency */

class SequentialUnidirectionalTestcase : public Testcase
{
public:
    SequentialUnidirectionalTestcase(const struct config&                     conf,
                                     std::unique_ptr<BandwidthTransferEngine> engine,
                                     std::string                              key,
                                     std::string                              desc)
        : Testcase(conf, std::move(engine), key, desc)
    {}
    void runEngine(const Buffer& buffer, const Buffer& peerBuffer, CopyAgentPolicy policy);
};

class SequentialBidirectionalTestcase : public Testcase
{
public:
    SequentialBidirectionalTestcase(const struct config&                     conf,
                                    std::unique_ptr<BandwidthTransferEngine> engine,
                                    std::string                              key,
                                    std::string                              desc)
        : Testcase(conf, std::move(engine), key, desc)
    {}
    void runEngine(const std::vector<const Buffer*>& buffers,
                   const std::vector<const Buffer*>& peerBuffers,
                   CopyAgentPolicy                   policy);
};

class ConcurrentUnidirectionalTestcase : public Testcase
{
public:
    ConcurrentUnidirectionalTestcase(const struct config&                     conf,
                                     std::unique_ptr<BandwidthTransferEngine> engine,
                                     std::string                              key,
                                     std::string                              desc)
        : Testcase(conf, std::move(engine), key, desc)
    {}
    void runEngine(const std::vector<const Buffer*>& buffers,
                   const std::vector<const Buffer*>& peerBuffers,
                   CopyAgentPolicy                   policy);
};

class ConcurrentBidirectionalTestcase : public Testcase
{
public:
    ConcurrentBidirectionalTestcase(const struct config&                     conf,
                                    std::unique_ptr<BandwidthTransferEngine> engine,
                                    std::string                              key,
                                    std::string                              desc)
        : Testcase(conf, std::move(engine), key, desc)
    {}
    void runEngine(const std::vector<std::vector<const Buffer*>>& bufferPairs,
                   const std::vector<std::vector<const Buffer*>>& peerBufferPairs,
                   CopyAgentPolicy                                agent);
};

class LatencyTestCase : public Testcase
{
public:
    LatencyTestCase(const struct config&                   conf,
                    std::unique_ptr<LatencyTransferEngine> engine,
                    std::string                            key,
                    std::string                            desc)
        : Testcase(conf, std::move(engine), key, desc)
    {}
    virtual void runEngine(const Buffer& buffer, const Buffer& peerBuffer, CopyAgentPolicy policy);
};

/* LatencyHostToDevice */

class LatencyHostToDevice : public LatencyTestCase
{
public:
    explicit LatencyHostToDevice(const struct config& conf)
        : LatencyTestCase(conf,
                          std::make_unique<LatencyTransferEngine>(conf.lat_accesses),
                          "latency_host_to_device",
                          "")
    {}
    void startEngine(CopyAgentPolicy policy) override;
    void execute() override;
};

/* LatencyDeviceToDevice */

class LatencyDeviceToDevice : public LatencyTestCase
{
public:
    explicit LatencyDeviceToDevice(const struct config& conf)
        : LatencyTestCase(conf,
                          std::make_unique<LatencyTransferEngine>(conf.lat_accesses),
                          "latency_device_to_device",
                          "")
    {}
    void startEngine(CopyAgentPolicy policy) override;
    void execute() override;
};

/* BandwidthHostToDevice */

class BandwidthHostToDeviceUnidir : public SequentialUnidirectionalTestcase
{
public:
    BandwidthHostToDeviceUnidir(const struct config&                     conf,
                                std::unique_ptr<BandwidthTransferEngine> engine,
                                std::string                              key,
                                std::string                              desc)
        : SequentialUnidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthHostToDeviceUnidir : public BandwidthHostToDeviceUnidir
{
public:
    explicit SDMABandwidthHostToDeviceUnidir(const struct config& conf)
        : BandwidthHostToDeviceUnidir(conf,
                                      std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                      "sdma_bw_host_to_device_unidir",
                                      ""){};
    void execute() override;
};

class CUBandwidthHostToDeviceUnidir : public BandwidthHostToDeviceUnidir
{
public:
    explicit CUBandwidthHostToDeviceUnidir(const struct config& conf)
        : BandwidthHostToDeviceUnidir(conf,
                                      std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                      "cu_bw_host_to_device_unidir",
                                      ""){};
    void execute() override;
};

class BandwidthHostToDeviceBidir : public SequentialBidirectionalTestcase
{
public:
    BandwidthHostToDeviceBidir(const struct config&                     conf,
                               std::unique_ptr<BandwidthTransferEngine> engine,
                               std::string                              key,
                               std::string                              desc)
        : SequentialBidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthHostToDeviceBidir : public BandwidthHostToDeviceBidir
{
public:
    explicit SDMABandwidthHostToDeviceBidir(const struct config& conf)
        : BandwidthHostToDeviceBidir(conf,
                                     std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                     "sdma_bw_host_to_device_bidir",
                                     ""){};
    void execute() override;
};

class CUBandwidthHostToDeviceBidir : public BandwidthHostToDeviceBidir
{
public:
    explicit CUBandwidthHostToDeviceBidir(const struct config& conf)
        : BandwidthHostToDeviceBidir(conf,
                                     std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                     "cu_bw_host_to_device_bidir",
                                     ""){};
    void execute() override;
};

/* BandwidthDeviceToHost */

class BandwidthDeviceToHostUnidir : public SequentialUnidirectionalTestcase
{
public:
    BandwidthDeviceToHostUnidir(const struct config&                     conf,
                                std::unique_ptr<BandwidthTransferEngine> engine,
                                std::string                              key,
                                std::string                              desc)
        : SequentialUnidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthDeviceToHostUnidir : public BandwidthDeviceToHostUnidir
{
public:
    explicit SDMABandwidthDeviceToHostUnidir(const struct config& conf)
        : BandwidthDeviceToHostUnidir(conf,
                                      std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                      "sdma_bw_device_to_host_unidir",
                                      ""){};
    void execute() override;
};

class CUBandwidthDeviceToHostUnidir : public BandwidthDeviceToHostUnidir
{
public:
    explicit CUBandwidthDeviceToHostUnidir(const struct config& conf)
        : BandwidthDeviceToHostUnidir(conf,
                                      std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                      "cu_bw_device_to_host_unidir",
                                      ""){};
    void execute() override;
};

class BandwidthDeviceToHostBidir : public SequentialBidirectionalTestcase
{
public:
    BandwidthDeviceToHostBidir(const struct config&                     conf,
                               std::unique_ptr<BandwidthTransferEngine> engine,
                               std::string                              key,
                               std::string                              desc)
        : SequentialBidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthDeviceToHostBidir : public BandwidthDeviceToHostBidir
{
public:
    explicit SDMABandwidthDeviceToHostBidir(const struct config& conf)
        : BandwidthDeviceToHostBidir(conf,
                                     std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                     "sdma_bw_device_to_host_bidir",
                                     ""){};
    void execute() override;
};

class CUBandwidthDeviceToHostBidir : public BandwidthDeviceToHostBidir
{
public:
    explicit CUBandwidthDeviceToHostBidir(const struct config& conf)
        : BandwidthDeviceToHostBidir(conf,
                                     std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                     "cu_bw_device_to_host_bidir",
                                     ""){};
    void execute() override;
};

/* BandwidthDeviceLocalCopy */

class BandwidthDeviceLocalCopy : public SequentialUnidirectionalTestcase
{
public:
    BandwidthDeviceLocalCopy(const struct config&                     conf,
                             std::unique_ptr<BandwidthTransferEngine> engine,
                             std::string                              key,
                             std::string                              desc)
        : SequentialUnidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthDeviceLocalCopy : public BandwidthDeviceLocalCopy
{
public:
    explicit SDMABandwidthDeviceLocalCopy(const struct config& conf)
        : BandwidthDeviceLocalCopy(conf,
                                   std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                   "sdma_bw_device_local_copy",
                                   ""){};
    void execute() override;
};

class CUBandwidthDeviceLocalCopy : public BandwidthDeviceLocalCopy
{
public:
    explicit CUBandwidthDeviceLocalCopy(const struct config& conf)
        : BandwidthDeviceLocalCopy(conf,
                                   std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                   "cu_bw_device_local_copy",
                                   ""){};
    void execute() override;
};

/* BandwidthDeviceToDevice */

class BandwidthDeviceToDeviceUnidir : public SequentialUnidirectionalTestcase
{
public:
    BandwidthDeviceToDeviceUnidir(const struct config&                     conf,
                                  std::unique_ptr<BandwidthTransferEngine> engine,
                                  std::string                              key,
                                  std::string                              desc)
        : SequentialUnidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthDeviceToDeviceUnidirWrite : public BandwidthDeviceToDeviceUnidir
{
public:
    explicit SDMABandwidthDeviceToDeviceUnidirWrite(const struct config& conf)
        : BandwidthDeviceToDeviceUnidir(conf,
                                        std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                        "sdma_bw_device_to_device_unidir_write",
                                        ""){};
    void execute() override;
};

class CUBandwidthDeviceToDeviceUnidirWrite : public BandwidthDeviceToDeviceUnidir
{
public:
    explicit CUBandwidthDeviceToDeviceUnidirWrite(const struct config& conf)
        : BandwidthDeviceToDeviceUnidir(
              conf,
              std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
              "cu_bw_device_to_device_unidir_write",
              ""){};
    void execute() override;
};

class SDMABandwidthDeviceToDeviceUnidirRead : public BandwidthDeviceToDeviceUnidir
{
public:
    explicit SDMABandwidthDeviceToDeviceUnidirRead(const struct config& conf)
        : BandwidthDeviceToDeviceUnidir(conf,
                                        std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                        "sdma_bw_device_to_device_unidir_read",
                                        ""){};
    void execute() override;
};

class CUBandwidthDeviceToDeviceUnidirRead : public BandwidthDeviceToDeviceUnidir
{
public:
    explicit CUBandwidthDeviceToDeviceUnidirRead(const struct config& conf)
        : BandwidthDeviceToDeviceUnidir(
              conf,
              std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
              "cu_bw_device_to_device_unidir_read",
              ""){};
    void execute() override;
};

class BandwidthDeviceToDeviceBidir : public SequentialBidirectionalTestcase
{
public:
    BandwidthDeviceToDeviceBidir(const struct config&                     conf,
                                 std::unique_ptr<BandwidthTransferEngine> engine,
                                 std::string                              key,
                                 std::string                              desc)
        : SequentialBidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthDeviceToDeviceBidirWrite : public BandwidthDeviceToDeviceBidir
{
public:
    explicit SDMABandwidthDeviceToDeviceBidirWrite(const struct config& conf)
        : BandwidthDeviceToDeviceBidir(conf,
                                       std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                       "sdma_bw_device_to_device_bidir_write",
                                       ""){};
    void execute() override;
};

class CUBandwidthDeviceToDeviceBidirWrite : public BandwidthDeviceToDeviceBidir
{
public:
    explicit CUBandwidthDeviceToDeviceBidirWrite(const struct config& conf)
        : BandwidthDeviceToDeviceBidir(conf,
                                       std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                       "cu_bw_device_to_device_bidir_write",
                                       ""){};
    void execute() override;
};

class SDMABandwidthDeviceToDeviceBidirRead : public BandwidthDeviceToDeviceBidir
{
public:
    explicit SDMABandwidthDeviceToDeviceBidirRead(const struct config& conf)
        : BandwidthDeviceToDeviceBidir(conf,
                                       std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                       "sdma_bw_device_to_device_bidir_read",
                                       ""){};
    void execute() override;
};

class CUBandwidthDeviceToDeviceBidirRead : public BandwidthDeviceToDeviceBidir
{
public:
    explicit CUBandwidthDeviceToDeviceBidirRead(const struct config& conf)
        : BandwidthDeviceToDeviceBidir(conf,
                                       std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                       "cu_bw_device_to_device_bidir_read",
                                       ""){};
    void execute() override;
};

/* BandwidthHostToAll */

class BandwidthHostToAllUnidir : public ConcurrentUnidirectionalTestcase
{
public:
    explicit BandwidthHostToAllUnidir(const struct config&                     conf,
                                      std::unique_ptr<BandwidthTransferEngine> engine,
                                      std::string                              key,
                                      std::string                              desc)
        : ConcurrentUnidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthHostToAllUnidir : public BandwidthHostToAllUnidir
{
public:
    explicit SDMABandwidthHostToAllUnidir(const struct config& conf)
        : BandwidthHostToAllUnidir(conf,
                                   std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                   "sdma_bw_host_to_all_unidir",
                                   "")
    {}
    void execute() override;
};

class CUBandwidthHostToAllUnidir : public BandwidthHostToAllUnidir
{
public:
    explicit CUBandwidthHostToAllUnidir(const struct config& conf)
        : BandwidthHostToAllUnidir(conf,
                                   std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                   "cu_bw_host_to_all_unidir",
                                   "")
    {}
    void execute() override;
};

class BandwidthHostToAllBidir : public ConcurrentBidirectionalTestcase
{
public:
    explicit BandwidthHostToAllBidir(const struct config&                     conf,
                                     std::unique_ptr<BandwidthTransferEngine> engine,
                                     std::string                              key,
                                     std::string                              desc)
        : ConcurrentBidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthHostToAllBidir : public BandwidthHostToAllBidir
{
public:
    explicit SDMABandwidthHostToAllBidir(const struct config& conf)
        : BandwidthHostToAllBidir(conf,
                                  std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                  "sdma_bw_host_to_all_bidir",
                                  "")
    {}
    void execute() override;
};

class CUBandwidthHostToAllBidir : public BandwidthHostToAllBidir
{
public:
    explicit CUBandwidthHostToAllBidir(const struct config& conf)
        : BandwidthHostToAllBidir(conf,
                                  std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                  "cu_bw_host_to_all_bidir",
                                  "")
    {}
    void execute() override;
};

/* BandwidthAllToHost */

class BandwidthAllToHostUnidir : public ConcurrentUnidirectionalTestcase
{
public:
    explicit BandwidthAllToHostUnidir(const struct config&                     conf,
                                      std::unique_ptr<BandwidthTransferEngine> engine,
                                      std::string                              key,
                                      std::string                              desc)
        : ConcurrentUnidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthAllToHostUnidir : public BandwidthAllToHostUnidir
{
public:
    explicit SDMABandwidthAllToHostUnidir(const struct config& conf)
        : BandwidthAllToHostUnidir(conf,
                                   std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                   "sdma_bw_all_to_host_unidir",
                                   "")
    {}
    void execute() override;
};

class CUBandwidthAllToHostUnidir : public BandwidthAllToHostUnidir
{
public:
    explicit CUBandwidthAllToHostUnidir(const struct config& conf)
        : BandwidthAllToHostUnidir(conf,
                                   std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                   "cu_bw_all_to_host_unidir",
                                   "")
    {}
    void execute() override;
};

class BandwidthAllToHostBidir : public ConcurrentBidirectionalTestcase
{
public:
    explicit BandwidthAllToHostBidir(const struct config&                     conf,
                                     std::unique_ptr<BandwidthTransferEngine> engine,
                                     std::string                              key,
                                     std::string                              desc)
        : ConcurrentBidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthAllToHostBidir : public BandwidthAllToHostBidir
{
public:
    explicit SDMABandwidthAllToHostBidir(const struct config& conf)
        : BandwidthAllToHostBidir(conf,
                                  std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                  "sdma_bw_all_to_host_bidir",
                                  "")
    {}
    void execute() override;
};

class CUBandwidthAllToHostBidir : public BandwidthAllToHostBidir
{
public:
    explicit CUBandwidthAllToHostBidir(const struct config& conf)
        : BandwidthAllToHostBidir(conf,
                                  std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                  "cu_bw_all_to_host_bidir",
                                  "")
    {}
    void execute() override;
};

/* BandwidthDeviceToAll */

class BandwidthDeviceToAll : public ConcurrentUnidirectionalTestcase
{
public:
    BandwidthDeviceToAll(const struct config&                     conf,
                         std::unique_ptr<BandwidthTransferEngine> engine,
                         std::string                              key,
                         std::string                              desc)
        : ConcurrentUnidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthDeviceToAllWrite : public BandwidthDeviceToAll
{
public:
    explicit SDMABandwidthDeviceToAllWrite(const struct config& conf)
        : BandwidthDeviceToAll(conf,
                               std::make_unique<SystemDMATransferEngine>(conf.iterations),
                               "sdma_bw_device_to_all_write",
                               ""){};
    void execute() override;
};

class CUBandwidthDeviceToAllWrite : public BandwidthDeviceToAll
{
public:
    explicit CUBandwidthDeviceToAllWrite(const struct config& conf)
        : BandwidthDeviceToAll(conf,
                               std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                               "cu_bw_device_to_all_write",
                               ""){};
    void execute() override;
};

class SDMABandwidthDeviceToAllRead : public BandwidthDeviceToAll
{
public:
    explicit SDMABandwidthDeviceToAllRead(const struct config& conf)
        : BandwidthDeviceToAll(conf,
                               std::make_unique<SystemDMATransferEngine>(conf.iterations),
                               "sdma_bw_device_to_all_read",
                               ""){};
    void execute() override;
};

class CUBandwidthDeviceToAllRead : public BandwidthDeviceToAll
{
public:
    explicit CUBandwidthDeviceToAllRead(const struct config& conf)
        : BandwidthDeviceToAll(conf,
                               std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                               "cu_bw_device_to_all_read",
                               ""){};
    void execute() override;
};

/* BandwidthDeviceRing */

class BandwidthDeviceRingUnidir : public ConcurrentUnidirectionalTestcase
{
public:
    BandwidthDeviceRingUnidir(const struct config&                     conf,
                              std::unique_ptr<BandwidthTransferEngine> engine,
                              std::string                              key,
                              std::string                              desc)
        : ConcurrentUnidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthDeviceRingUnidirWrite : public BandwidthDeviceRingUnidir
{
public:
    explicit SDMABandwidthDeviceRingUnidirWrite(const struct config& conf)
        : BandwidthDeviceRingUnidir(conf,
                                    std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                    "sdma_bw_device_ring_unidir_write",
                                    ""){};
    void execute() override;
};

class CUBandwidthDeviceRingUnidirWrite : public BandwidthDeviceRingUnidir
{
public:
    explicit CUBandwidthDeviceRingUnidirWrite(const struct config& conf)
        : BandwidthDeviceRingUnidir(conf,
                                    std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                    "cu_bw_device_ring_unidir_write",
                                    ""){};
    void execute() override;
};

class SDMABandwidthDeviceRingUnidirRead : public BandwidthDeviceRingUnidir
{
public:
    explicit SDMABandwidthDeviceRingUnidirRead(const struct config& conf)
        : BandwidthDeviceRingUnidir(conf,
                                    std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                    "sdma_bw_device_ring_unidir_read",
                                    ""){};
    void execute() override;
};

class CUBandwidthDeviceRingUnidirRead : public BandwidthDeviceRingUnidir
{
public:
    explicit CUBandwidthDeviceRingUnidirRead(const struct config& conf)
        : BandwidthDeviceRingUnidir(conf,
                                    std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                    "cu_bw_device_ring_unidir_read",
                                    ""){};
    void execute() override;
};

class BandwidthDeviceRingBidir : public ConcurrentBidirectionalTestcase
{
public:
    BandwidthDeviceRingBidir(const struct config&                     conf,
                             std::unique_ptr<BandwidthTransferEngine> engine,
                             std::string                              key,
                             std::string                              desc)
        : ConcurrentBidirectionalTestcase(conf, std::move(engine), key, desc)
    {}
    void startEngine(CopyAgentPolicy policy) override;
};

class SDMABandwidthDeviceRingBidirWrite : public BandwidthDeviceRingBidir
{
public:
    explicit SDMABandwidthDeviceRingBidirWrite(const struct config& conf)
        : BandwidthDeviceRingBidir(conf,
                                   std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                   "sdma_bw_device_ring_bidir_write",
                                   ""){};
    void execute() override;
};

class CUBandwidthDeviceRingBidirWrite : public BandwidthDeviceRingBidir
{
public:
    explicit CUBandwidthDeviceRingBidirWrite(const struct config& conf)
        : BandwidthDeviceRingBidir(conf,
                                   std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                   "cu_bw_device_ring_bidir_write",
                                   ""){};
    void execute() override;
};

class SDMABandwidthDeviceRingBidirRead : public BandwidthDeviceRingBidir
{
public:
    explicit SDMABandwidthDeviceRingBidirRead(const struct config& conf)
        : BandwidthDeviceRingBidir(conf,
                                   std::make_unique<SystemDMATransferEngine>(conf.iterations),
                                   "sdma_bw_device_ring_bidir_read",
                                   ""){};
    void execute() override;
};

class CUBandwidthDeviceRingBidirRead : public BandwidthDeviceRingBidir
{
public:
    explicit CUBandwidthDeviceRingBidirRead(const struct config& conf)
        : BandwidthDeviceRingBidir(conf,
                                   std::make_unique<ComputeUnitTransferEngine>(conf.iterations),
                                   "cu_bw_device_ring_bidir_read",
                                   ""){};
    void execute() override;
};
