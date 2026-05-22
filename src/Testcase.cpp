
#include "Testcase.hpp"
#include "Buffer.hpp"
#include "TransferEngine.hpp"
#include "common.h"

/* BandwidthHostToDevice */

void BandwidthHostToDeviceUnidir::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);

    for (int dev = 0; dev < conf.device_count; dev++)
    {
        HostBuffer   buffer(dev, conf.bw_bytes);
        DeviceBuffer peerBuffer(dev, conf.bw_bytes);

        double bandwidth = engine.sequentialUnidirectional(buffer, peerBuffer, policy);
        _output << "\tCPU -> GPU " << dev << ": " << bandwidth << " GB/s\n";
    }
}

void BandwidthHostToDeviceUnidir::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::HOST);
}

void BandwidthHostToDeviceBidir::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);

    for (int dev = 0; dev < conf.device_count; dev++)
    {
        HostBuffer   hostBuffer1(dev, conf.bw_bytes), hostBuffer2(dev, conf.bw_bytes);
        DeviceBuffer deviceBuffer1(dev, conf.bw_bytes), deviceBuffer2(dev, conf.bw_bytes);

        double bandwidth = engine.sequentialBidirectional({&hostBuffer1, &hostBuffer2},
                                                          {&deviceBuffer1, &deviceBuffer2},
                                                          policy);
        _output << "\tCPU <-> GPU " << dev << ": " << bandwidth << " GB/s\n";
    }
}

void BandwidthHostToDeviceBidir::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::HOST_AND_DESTINATION);
}

/* BandwidthDeviceToHost */

void BandwidthDeviceToHostUnidir::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);

    for (int dev = 0; dev < conf.device_count; dev++)
    {
        DeviceBuffer deviceBuffer(dev, conf.bw_bytes);
        HostBuffer   hostBuffer(dev, conf.bw_bytes);

        double bandwidth = engine.sequentialUnidirectional(deviceBuffer, hostBuffer, policy);
        _output << "\tGPU " << dev << " -> CPU: " << bandwidth << " GB/s\n";
    }
}

void BandwidthDeviceToHostUnidir::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE);
}

void BandwidthDeviceToHostBidir::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);

    for (int dev = 0; dev < conf.device_count; dev++)
    {
        DeviceBuffer deviceBuffer1(dev, conf.bw_bytes), deviceBuffer2(dev, conf.bw_bytes);
        HostBuffer   hostBuffer1(dev, conf.bw_bytes), hostBuffer2(dev, conf.bw_bytes);

        double bandwidth = engine.sequentialBidirectional({&deviceBuffer1, &deviceBuffer2},
                                                          {&hostBuffer1, &hostBuffer2},
                                                          policy);
        _output << "\tGPU " << dev << " <-> CPU: " << bandwidth << " GB/s\n";
    }
}

void BandwidthDeviceToHostBidir::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::HOST_AND_DESTINATION);
}

/* BandwidthDeviceLocalCopy */

void BandwidthDeviceLocalCopy::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);

    for (int src = 0; src < conf.device_count; src++)
    {
        DeviceBuffer sourceBuffer(src, conf.bw_bytes);
        DeviceBuffer destinationBuffer(src, conf.bw_bytes);
        double bandwidth = engine.sequentialUnidirectional(sourceBuffer, destinationBuffer, policy);
        _output << "\tGPU " << src << " -> GPU " << src << ": " << bandwidth << " GB/s\n";
    }
}

void BandwidthDeviceLocalCopy::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE);
}

/* BandwidthDeviceToDevice */

void BandwidthDeviceToDeviceUnidir::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);

    for (int src = 0; src < conf.device_count; src++)
    {
        HIP_EXIT_ON_FAILURE(hipSetDevice(src));
        DeviceBuffer sourceBuffer(src, conf.bw_bytes);
        for (int dst = 0; dst < conf.device_count; dst++)
        {
            if (src == dst)
            {
                continue;
            }
            DeviceBuffer destinationBuffer(dst, conf.bw_bytes);
            double       bandwidth
                = engine.sequentialUnidirectional(sourceBuffer, destinationBuffer, policy);
            _output << "\tGPU " << src << " -> GPU " << dst << ": " << bandwidth << " GB/s\n";
        }
    }
}

void BandwidthDeviceToDeviceUnidirWrite::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE);
}

void BandwidthDeviceToDeviceUnidirRead::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::DESTINATION);
}

void BandwidthDeviceToDeviceBidir::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);

    for (int dev = 0; dev < conf.device_count; dev++)
    {
        HIP_EXIT_ON_FAILURE(hipSetDevice(dev));
        DeviceBuffer buffer1(dev, conf.bw_bytes), buffer2(dev, conf.bw_bytes);
        for (int peer = 0; peer < conf.device_count; peer++)
        {
            if (dev == peer)
            {
                continue;
            }
            DeviceBuffer peerBuffer1(peer, conf.bw_bytes), peerBuffer2(peer, conf.bw_bytes);
            double       bandwidth = engine.sequentialBidirectional({&buffer1, &buffer2},
                                                                    {&peerBuffer1, &peerBuffer2},
                                                              policy);
            _output << "\tGPU " << dev << " <-> GPU " << peer << ": " << bandwidth << " GB/s\n";
        }
    }
}

void BandwidthDeviceToDeviceBidirWrite::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE_AND_DESTINATION);
}

void BandwidthDeviceToDeviceBidirRead::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE_AND_DESTINATION);
}

/* BandwidthHostToAll */

void BandwidthHostToAllUnidir::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);
    int             devcount = conf.device_count;
    std::vector<const Buffer*> hostBuffers(devcount);
    std::vector<const Buffer*> deviceBuffers(devcount);
    std::vector<double>        bandwidths(devcount);
    double                     sum = 0.0;

    for (int dev = 0; dev < devcount; dev++)
    {
        HIP_EXIT_ON_FAILURE(hipSetDevice(dev));
        hostBuffers[dev]   = new HostBuffer(dev, conf.bw_bytes);
        deviceBuffers[dev] = new DeviceBuffer(dev, conf.bw_bytes);
    }

    engine.concurrentUnidirectional(hostBuffers, deviceBuffers, policy, &bandwidths);

    for (int dev = 0; dev < devcount; dev++)
    {
        _output << "\tCPU -> GPU " << dev << ": " << bandwidths.at(dev) << " GB/s\n";
        sum += bandwidths.at(dev);
    }
    _output << "\tSum: " << sum << " GB/s\n";
}

void BandwidthHostToAllUnidir::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::HOST);
}

void BandwidthHostToAllBidir::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);
    int             devcount = conf.device_count;
    std::vector<std::vector<const Buffer*>> hostBufferPairs(2);
    std::vector<std::vector<const Buffer*>> deviceBufferPairs(2);
    std::vector<double>                     bandwidths(devcount);
    double                                  sum = 0.0;

    for (int i = 0; i < 2; i++)
    {
        hostBufferPairs[i].resize(devcount);
        deviceBufferPairs[i].resize(devcount);

        for (int dev = 0; dev < devcount; dev++)
        {
            HIP_EXIT_ON_FAILURE(hipSetDevice(dev));
            hostBufferPairs[i][dev]   = new HostBuffer(dev, conf.bw_bytes);
            deviceBufferPairs[i][dev] = new DeviceBuffer(dev, conf.bw_bytes);
        }
    }

    engine.concurrentBidirectional(hostBufferPairs, deviceBufferPairs, policy, &bandwidths);

    for (int dev = 0; dev < devcount; dev++)
    {
        _output << "\tCPU <-> GPU " << dev << ": " << bandwidths.at(dev) << " GB/s\n";
        sum += bandwidths.at(dev);

        for (int i = 0; i < 2; i++)
        {
            delete hostBufferPairs[i][dev];
            delete deviceBufferPairs[i][dev];
        }
    }
    _output << "\tSum: " << sum << " GB/s\n";
}

void BandwidthHostToAllBidir::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::HOST_AND_DESTINATION);
}

/* BandwidthAllToHost */

void BandwidthAllToHostUnidir::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);
    int             devcount = conf.device_count;
    std::vector<const Buffer*> deviceBuffers(devcount);
    std::vector<const Buffer*> hostBuffers(devcount);
    std::vector<double>        bandwidths(devcount);
    double                     sum = 0.0;

    for (int dev = 0; dev < devcount; dev++)
    {
        HIP_EXIT_ON_FAILURE(hipSetDevice(dev));
        deviceBuffers[dev] = new DeviceBuffer(dev, conf.bw_bytes);
        hostBuffers[dev]   = new HostBuffer(dev, conf.bw_bytes);
    }

    engine.concurrentUnidirectional(deviceBuffers, hostBuffers, policy, &bandwidths);

    for (int dev = 0; dev < devcount; dev++)
    {
        _output << "\tGPU " << dev << " -> CPU" << ": " << bandwidths.at(dev) << " GB/s\n";
        sum += bandwidths.at(dev);

        delete deviceBuffers[dev];
        delete hostBuffers[dev];
    }
    _output << "\tSum: " << sum << " GB/s\n";
}

void BandwidthAllToHostUnidir::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE);
}

void BandwidthAllToHostBidir::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);
    int             devcount = conf.device_count;
    std::vector<std::vector<const Buffer*>> deviceBufferPairs(2);
    std::vector<std::vector<const Buffer*>> hostBufferPairs(2);
    std::vector<double>                     bandwidths(devcount);
    double                                  sum = 0.0;

    for (int i = 0; i < 2; i++)
    {
        hostBufferPairs[i].resize(devcount);
        deviceBufferPairs[i].resize(devcount);

        for (int dev = 0; dev < devcount; dev++)
        {
            HIP_EXIT_ON_FAILURE(hipSetDevice(dev));
            deviceBufferPairs[i][dev] = new DeviceBuffer(dev, conf.bw_bytes);
            hostBufferPairs[i][dev]   = new HostBuffer(dev, conf.bw_bytes);
        }
    }

    engine.concurrentBidirectional(deviceBufferPairs, hostBufferPairs, policy, &bandwidths);

    for (int dev = 0; dev < devcount; dev++)
    {
        _output << "\tGPU " << dev << " -> CPU" << ": " << bandwidths.at(dev) << " GB/s\n";
        sum += bandwidths.at(dev);

        for (int i = 0; i < 2; i++)
        {
            delete hostBufferPairs[i][dev];
            delete deviceBufferPairs[i][dev];
        }
    }
    _output << "\tSum: " << sum << " GB/s\n";
}

void BandwidthAllToHostBidir::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE_AND_HOST);
}

/* BandwidthDeviceToAll */

void BandwidthDeviceToAll::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);
    int             devcount = conf.device_count;
    std::vector<const Buffer*> sourceBuffers(devcount);
    std::vector<const Buffer*> destinationBuffers(devcount);
    std::vector<double>        bandwidths(devcount);

    for (int src = 0; src < devcount; src++)
    {
        double sum = 0.0;
        for (int dst = 0; dst < devcount; dst++)
        {
            sourceBuffers[dst] = new DeviceBuffer(src, conf.bw_bytes);
            if (src == dst)
            {
                destinationBuffers[dst] = new HostBuffer(dst, conf.bw_bytes);
            }
            else
            {
                destinationBuffers[dst] = new DeviceBuffer(dst, conf.bw_bytes);
            }
        }

        engine.concurrentUnidirectional(sourceBuffers, destinationBuffers, policy, &bandwidths);

        for (int dev = 0; dev < devcount; dev++)
        {
            if (src == dev)
            {
                _output << "\tGPU " << src << " -> CPU  : " << bandwidths.at(dev) << " GB/s\n";
            }
            else
            {
                _output << "\tGPU " << src << " -> GPU " << dev << ": " << bandwidths.at(dev)
                        << " GB/s\n";
            }
            sum += bandwidths.at(dev);

            delete sourceBuffers[dev];
            delete destinationBuffers[dev];

            sourceBuffers[dev]      = nullptr;
            destinationBuffers[dev] = nullptr;
        }
        _output << "\tSum: " << sum << " GB/s\n";
    }
}

void BandwidthDeviceToAllWrite::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE);
}

void BandwidthDeviceToAllRead::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::DESTINATION);
}

/* BandwidthDeviceRing */

void BandwidthDeviceRing::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);
    int             devcount = conf.device_count;
    std::vector<const Buffer*> sourceBuffers(devcount);
    std::vector<const Buffer*> destinationBuffers(devcount);
    std::vector<double>        bandwidths(devcount);
    double                     sum = 0.0;

    for (int dev = 0; dev < devcount; dev++)
    {
        int dst                 = (dev + 1) % devcount;
        sourceBuffers[dev]      = new DeviceBuffer(dev, conf.bw_bytes);
        destinationBuffers[dev] = new DeviceBuffer(dst, conf.bw_bytes);
    }

    engine.concurrentUnidirectional(sourceBuffers, destinationBuffers, policy, &bandwidths);
    for (int src = 0; src < devcount; src++)
    {
        int    dst = (src + 1) % devcount;
        double bw  = bandwidths.at(src);
        sum += bw;
        _output << "\tGPU " << src << " -> GPU " << dst << ": " << bw << " GB/s\n";

        delete sourceBuffers[src];
        delete destinationBuffers[src];
    }
    _output << "\tSum: " << sum << " GB/s\n";
}

void BandwidthDeviceRingWrite::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE);
}

void BandwidthDeviceRingRead::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::DESTINATION);
}