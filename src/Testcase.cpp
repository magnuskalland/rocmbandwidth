
#include "Testcase.hpp"
#include "Buffer.hpp"
#include "TransferEngine.hpp"
#include "common.h"

/* BandwidthHostToDevice */

void BandwidthHostToDevice::doExecute(const struct config& conf, CopyAgentPolicy policy)
{
    auto            executor = std::make_unique<SystemDMAExecutor>(0xDeadBeefCafeBabeULL);
    SystemDMAEngine engine(std::move(executor), conf.iterations);

    for (int dev = 0; dev < conf.device_count; dev++)
    {
        HostBuffer   hostBuffer(dev, conf.bw_bytes);
        DeviceBuffer deviceBuffer(dev, conf.bw_bytes);

        double bandwidth = engine.sequentialUnidirectional(hostBuffer, deviceBuffer, policy);
        _output << "\tCPU -> GPU " << dev << ": " << bandwidth << " GB/s\n";
    }
}

void BandwidthHostToDevice::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::HOST);
}

/* BandwidthDeviceToHost */

void BandwidthDeviceToHost::doExecute(const struct config& conf, CopyAgentPolicy policy)
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

void BandwidthDeviceToHost::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE);
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

void BandwidthDeviceToDevice::doExecute(const struct config& conf, CopyAgentPolicy policy)
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

void BandwidthDeviceToDeviceWrite::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE);
}

void BandwidthDeviceToDeviceRead::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::DESTINATION);
}

/* BandwidthHostToAll */

void BandwidthHostToAll::doExecute(const struct config& conf, CopyAgentPolicy policy)
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

void BandwidthHostToAll::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::HOST);
}

/* BandwidthAllToHost */

void BandwidthAllToHost::doExecute(const struct config& conf, CopyAgentPolicy policy)
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

void BandwidthAllToHost::execute(const struct config& conf)
{
    doExecute(conf, CopyAgentPolicy::SOURCE);
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