
#include "Testcase.hpp"
#include "Buffer.hpp"
#include "TransferEngine.hpp"
#include "common.h"

void addToOutputLatency(std::stringstream& output,
                        const Buffer&      buffer,
                        const Buffer&      peerBuffer,
                        double             latency)
{
    int dev1 = buffer.getDevice();
    int dev2 = peerBuffer.getDevice();
    if (dev1 == -1)
    {
        output << "\tCPU -> GPU " << dev2 << ": " << latency << " ns\n";
    }
    else if (dev2 == -1)
    {
        output << "\tGPU " << dev1 << " -> CPU  : " << latency << " ns\n";
    }
    else
    {
        output << "\tGPU " << dev1 << " -> GPU " << dev2 << ": " << latency << " ns\n";
    }
}

void addToOutput(std::stringstream& output,
                 const Buffer&      buffer,
                 const Buffer&      peerBuffer,
                 double             bandwidth)
{
    int dev1 = buffer.getDevice();
    int dev2 = peerBuffer.getDevice();
    if (dev1 == -1)
    {
        output << "\tCPU -> GPU " << dev2 << ": " << bandwidth << " GB/s\n";
    }
    else if (dev2 == -1)
    {
        output << "\tGPU " << dev1 << " -> CPU  : " << bandwidth << " GB/s\n";
    }
    else
    {
        output << "\tGPU " << dev1 << " -> GPU " << dev2 << ": " << bandwidth << " GB/s\n";
    }
}

void addToOutput(std::stringstream&                output,
                 const std::vector<const Buffer*>& buffers,
                 const std::vector<const Buffer*>& peerBuffers,
                 double                            bandwidth)
{
    for (size_t i = 0; i < buffers.size() / 2; i++)
    {
        int dev1 = buffers.at(i)->getDevice();
        int dev2 = peerBuffers.at(i)->getDevice();
        if (dev1 == -1)
        {
            output << "\tCPU <-> GPU " << dev2 << ": " << bandwidth << " GB/s\n";
        }
        else if (dev2 == -1)
        {
            output << "\tGPU " << dev1 << " <-> CPU  : " << bandwidth << " GB/s\n";
        }
        else
        {
            output << "\tGPU " << dev1 << " <-> GPU " << dev2 << ": " << bandwidth << " GB/s\n";
        }
    }
}

void addToOutput(std::stringstream&                output,
                 const std::vector<const Buffer*>& buffers,
                 const std::vector<const Buffer*>& peerBuffers,
                 std::vector<double>               bandwidths)
{
    double sum = 0.0;
    for (size_t i = 0; i < buffers.size(); i++)
    {
        int    dev1 = buffers.at(i)->getDevice();
        int    dev2 = peerBuffers.at(i)->getDevice();
        double bw   = bandwidths.at(i);
        if (dev1 == -1)
        {
            output << "\tCPU -> GPU " << dev2 << ": " << bw << " GB/s\n";
        }
        else if (dev2 == -1)
        {
            output << "\tGPU " << dev1 << " -> CPU  : " << bw << " GB/s\n";
        }
        else
        {
            output << "\tGPU " << dev1 << " -> GPU " << dev2 << ": " << bw << " GB/s\n";
        }
        sum += bw;
    }
    output << "\tSum: " << sum << " GB/s\n";
}

void addToOutput(std::stringstream&                             output,
                 const std::vector<std::vector<const Buffer*>>& bufferPairs,
                 const std::vector<std::vector<const Buffer*>>& peerBufferPairs,
                 std::vector<double>*                           bandwidths)
{
    double sum = 0.0;
    for (size_t dev = 0; dev < bufferPairs.at(0).size(); dev++)
    {
        int    dev1 = bufferPairs.at(0).at(dev)->getDevice();
        int    dev2 = peerBufferPairs.at(0).at(dev)->getDevice();
        double bw   = bandwidths->at(dev);
        if (dev1 == -1)
        {
            output << "\tCPU <-> GPU " << dev2 << ": " << bw << " GB/s\n";
        }
        else if (dev2 == -1)
        {
            output << "\tGPU " << dev1 << " <-> CPU  : " << bw << " GB/s\n";
        }
        else
        {
            output << "\tGPU " << dev1 << " <-> GPU " << dev2 << ": " << bw << " GB/s\n";
        }
        sum += bandwidths->at(dev);
    }
    output << "\tSum: " << sum << " GB/s\n";
}

void SequentialUnidirectionalTestcase::runEngine(const Buffer&   buffer,
                                                 const Buffer&   peerBuffer,
                                                 CopyAgentPolicy policy)
{
    double bandwidth = _engine->sequentialUnidirectional(buffer, peerBuffer, policy);
    addToOutput(_output, buffer, peerBuffer, bandwidth);
};

void SequentialBidirectionalTestcase::runEngine(const std::vector<const Buffer*>& buffers,
                                                const std::vector<const Buffer*>& peerBuffers,
                                                CopyAgentPolicy                   policy)
{
    double bandwidth = _engine->sequentialBidirectional(buffers, peerBuffers, policy);
    addToOutput(_output, buffers, peerBuffers, bandwidth);
};

void ConcurrentUnidirectionalTestcase::runEngine(const std::vector<const Buffer*>& buffers,
                                                 const std::vector<const Buffer*>& peerBuffers,
                                                 CopyAgentPolicy                   policy)
{
    std::vector<double> bandwidths(buffers.size());
    _engine->concurrentUnidirectional(buffers, peerBuffers, policy, &bandwidths);
    addToOutput(_output, buffers, peerBuffers, bandwidths);
};

void ConcurrentBidirectionalTestcase::runEngine(
    const std::vector<std::vector<const Buffer*>>& bufferPairs,
    const std::vector<std::vector<const Buffer*>>& peerBufferPairs,
    CopyAgentPolicy                                policy)
{
    std::vector<double> bandwidths(bufferPairs.at(0).size());
    _engine->concurrentBidirectional(bufferPairs, peerBufferPairs, policy, &bandwidths);
    addToOutput(_output, bufferPairs, peerBufferPairs, &bandwidths);
};

void LatencyTestCase::runEngine(const Buffer&   buffer,
                                const Buffer&   peerBuffer,
                                CopyAgentPolicy policy)
{
    double latency = _engine->sequentialUnidirectional(buffer, peerBuffer, policy);
    addToOutputLatency(_output, buffer, peerBuffer, latency);
};

/* LatencyHostToDevice */

void LatencyHostToDevice::startEngine(CopyAgentPolicy policy)
{
    for (int dev = 0; dev < _conf.device_count; dev++)
    {
        HostBuffer   hostBuffer(dev, _conf.lat_bufsize);
        DeviceBuffer deviceBuffer(dev, _conf.lat_bufsize);

        runEngine(hostBuffer, deviceBuffer, policy);
    }
}

void LatencyHostToDevice::execute()
{
    startEngine(CopyAgentPolicy::HOST);
}

/* LatencyDeviceToDevice */

void LatencyDeviceToDevice::startEngine(CopyAgentPolicy policy)
{
    for (int src = 0; src < _conf.device_count; src++)
    {
        DeviceBuffer buffer(src, _conf.lat_bufsize);
        for (int dst = 0; dst < _conf.device_count; dst++)
        {
            if (src == dst)
            {
                continue;
            }
            DeviceBuffer peerBuffer(dst, _conf.lat_bufsize);

            runEngine(peerBuffer, buffer, policy);
        }
    }
}

void LatencyDeviceToDevice::execute()
{
    startEngine(CopyAgentPolicy::DESTINATION);
}

/* BandwidthHostToDevice */

void BandwidthHostToDeviceUnidir::startEngine(CopyAgentPolicy policy)
{
    for (int dev = 0; dev < _conf.device_count; dev++)
    {
        HostBuffer   buffer(dev, _conf.bw_bufsize);
        DeviceBuffer peerBuffer(dev, _conf.bw_bufsize);

        runEngine(buffer, peerBuffer, policy);
    }
}

void SDMABandwidthHostToDeviceUnidir::execute()
{
    startEngine(CopyAgentPolicy::HOST);
}

void CUBandwidthHostToDeviceUnidir::execute()
{
    startEngine(CopyAgentPolicy::HOST);
}

void BandwidthHostToDeviceBidir::startEngine(CopyAgentPolicy policy)
{
    for (int dev = 0; dev < _conf.device_count; dev++)
    {
        HostBuffer   buffer1(dev, _conf.bw_bufsize), buffer2(dev, _conf.bw_bufsize);
        DeviceBuffer peerBuffer1(dev, _conf.bw_bufsize), peerBuffer2(dev, _conf.bw_bufsize);

        runEngine({&buffer1, &buffer2}, {&peerBuffer1, &peerBuffer2}, policy);
    }
}

void SDMABandwidthHostToDeviceBidir::execute()
{
    startEngine(CopyAgentPolicy::HOST_AND_DESTINATION);
}

void CUBandwidthHostToDeviceBidir::execute()
{
    startEngine(CopyAgentPolicy::HOST_AND_DESTINATION);
}

/* BandwidthDeviceToHost */

void BandwidthDeviceToHostUnidir::startEngine(CopyAgentPolicy policy)
{
    for (int dev = 0; dev < _conf.device_count; dev++)
    {
        DeviceBuffer buffer(dev, _conf.bw_bufsize);
        HostBuffer   peerBuffer(dev, _conf.bw_bufsize);

        runEngine(buffer, peerBuffer, policy);
    }
}

void SDMABandwidthDeviceToHostUnidir::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void CUBandwidthDeviceToHostUnidir::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void BandwidthDeviceToHostBidir::startEngine(CopyAgentPolicy policy)
{
    for (int dev = 0; dev < _conf.device_count; dev++)
    {
        HostBuffer   buffer1(dev, _conf.bw_bufsize), buffer2(dev, _conf.bw_bufsize);
        DeviceBuffer peerBuffer1(dev, _conf.bw_bufsize), peerBuffer2(dev, _conf.bw_bufsize);

        runEngine({&buffer1, &buffer2}, {&peerBuffer1, &peerBuffer2}, policy);
    }
}

void SDMABandwidthDeviceToHostBidir::execute()
{
    startEngine(CopyAgentPolicy::HOST_AND_DESTINATION);
}

void CUBandwidthDeviceToHostBidir::execute()
{
    startEngine(CopyAgentPolicy::HOST_AND_DESTINATION);
}

/* BandwidthDeviceLocalCopy */

void BandwidthDeviceLocalCopy::startEngine(CopyAgentPolicy policy)
{
    for (int dev = 0; dev < _conf.device_count; dev++)
    {
        DeviceBuffer src(dev, _conf.bw_bufsize);
        DeviceBuffer dst(dev, _conf.bw_bufsize);

        runEngine(src, dst, policy);
    }
}

void SDMABandwidthDeviceLocalCopy::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void CUBandwidthDeviceLocalCopy::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

/* BandwidthDeviceToDevice */

void BandwidthDeviceToDeviceUnidir::startEngine(CopyAgentPolicy policy)
{
    for (int src = 0; src < _conf.device_count; src++)
    {
        DeviceBuffer buffer(src, _conf.bw_bufsize);
        for (int dst = 0; dst < _conf.device_count; dst++)
        {
            if (src == dst)
            {
                continue;
            }
            DeviceBuffer peerBuffer(dst, _conf.bw_bufsize);

            runEngine(buffer, peerBuffer, policy);
        }
    }
}

void SDMABandwidthDeviceToDeviceUnidirWrite::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void CUBandwidthDeviceToDeviceUnidirWrite::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void SDMABandwidthDeviceToDeviceUnidirRead::execute()
{
    startEngine(CopyAgentPolicy::DESTINATION);
}

void CUBandwidthDeviceToDeviceUnidirRead::execute()
{
    startEngine(CopyAgentPolicy::DESTINATION);
}

void BandwidthDeviceToDeviceBidir::startEngine(CopyAgentPolicy policy)
{
    for (int dev = 0; dev < _conf.device_count; dev++)
    {
        DeviceBuffer buffer1(dev, _conf.bw_bufsize), buffer2(dev, _conf.bw_bufsize);
        for (int peer = 0; peer < _conf.device_count; peer++)
        {
            if (dev == peer)
            {
                continue;
            }
            DeviceBuffer peerBuffer1(peer, _conf.bw_bufsize), peerBuffer2(peer, _conf.bw_bufsize);

            runEngine({&buffer1, &buffer2}, {&peerBuffer1, &peerBuffer2}, policy);
        }
    }
}

void SDMABandwidthDeviceToDeviceBidirWrite::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void CUBandwidthDeviceToDeviceBidirWrite::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void SDMABandwidthDeviceToDeviceBidirRead::execute()
{
    startEngine(CopyAgentPolicy::DESTINATION);
}

void CUBandwidthDeviceToDeviceBidirRead::execute()
{
    startEngine(CopyAgentPolicy::DESTINATION);
}

/* BandwidthHostToAll */

void BandwidthHostToAllUnidir::startEngine(CopyAgentPolicy policy)
{
    int                        devcount = _conf.device_count;
    std::vector<const Buffer*> hostBuffers(devcount);
    std::vector<const Buffer*> peerBuffers(devcount);
    std::vector<double>        bandwidths(devcount);

    for (int dev = 0; dev < devcount; dev++)
    {
        hostBuffers[dev] = new HostBuffer(dev, _conf.bw_bufsize);
        peerBuffers[dev] = new DeviceBuffer(dev, _conf.bw_bufsize);
    }

    runEngine(hostBuffers, peerBuffers, policy);

    for (int dev = 0; dev < devcount; dev++)
    {
        delete hostBuffers[dev];
        delete peerBuffers[dev];
    }
}

void SDMABandwidthHostToAllUnidir::execute()
{
    startEngine(CopyAgentPolicy::HOST);
}

void CUBandwidthHostToAllUnidir::execute()
{
    startEngine(CopyAgentPolicy::HOST);
}

void BandwidthHostToAllBidir::startEngine(CopyAgentPolicy policy)
{
    int                                     devcount = _conf.device_count;
    std::vector<std::vector<const Buffer*>> hostBufferPairs(2);
    std::vector<std::vector<const Buffer*>> deviceBufferPairs(2);

    for (int i = 0; i < 2; i++)
    {
        hostBufferPairs[i].resize(devcount);
        deviceBufferPairs[i].resize(devcount);

        for (int dev = 0; dev < devcount; dev++)
        {
            hostBufferPairs[i][dev]   = new HostBuffer(dev, _conf.bw_bufsize);
            deviceBufferPairs[i][dev] = new DeviceBuffer(dev, _conf.bw_bufsize);
        }
    }

    runEngine(hostBufferPairs, deviceBufferPairs, policy);

    for (int i = 0; i < 2; i++)
    {
        for (int dev = 0; dev < devcount; dev++)
        {
            delete hostBufferPairs[i][dev];
            delete deviceBufferPairs[i][dev];
        }
    }
}

void SDMABandwidthHostToAllBidir::execute()
{
    startEngine(CopyAgentPolicy::HOST_AND_DESTINATION);
}

void CUBandwidthHostToAllBidir::execute()
{
    startEngine(CopyAgentPolicy::HOST_AND_DESTINATION);
}

/* BandwidthAllToHost */

void BandwidthAllToHostUnidir::startEngine(CopyAgentPolicy policy)
{
    int                        devcount = _conf.device_count;
    std::vector<const Buffer*> deviceBuffers(devcount);
    std::vector<const Buffer*> hostBuffers(devcount);

    for (int dev = 0; dev < devcount; dev++)
    {
        deviceBuffers[dev] = new DeviceBuffer(dev, _conf.bw_bufsize);
        hostBuffers[dev]   = new HostBuffer(dev, _conf.bw_bufsize);
    }

    runEngine(deviceBuffers, hostBuffers, policy);

    for (int dev = 0; dev < devcount; dev++)
    {
        delete deviceBuffers[dev];
        delete hostBuffers[dev];
    }
}

void SDMABandwidthAllToHostUnidir::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void CUBandwidthAllToHostUnidir::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void BandwidthAllToHostBidir::startEngine(CopyAgentPolicy policy)
{
    int                                     devcount = _conf.device_count;
    std::vector<std::vector<const Buffer*>> deviceBufferPairs(2);
    std::vector<std::vector<const Buffer*>> hostBufferPairs(2);

    for (int i = 0; i < 2; i++)
    {
        deviceBufferPairs[i].resize(devcount);
        hostBufferPairs[i].resize(devcount);

        for (int dev = 0; dev < devcount; dev++)
        {
            deviceBufferPairs[i][dev] = new DeviceBuffer(dev, _conf.bw_bufsize);
            hostBufferPairs[i][dev]   = new HostBuffer(dev, _conf.bw_bufsize);
        }
    }

    runEngine(deviceBufferPairs, hostBufferPairs, policy);

    for (int i = 0; i < 2; i++)
    {
        for (int dev = 0; dev < devcount; dev++)
        {
            delete deviceBufferPairs[i][dev];
            delete hostBufferPairs[i][dev];
        }
    }
}

void SDMABandwidthAllToHostBidir::execute()
{
    startEngine(CopyAgentPolicy::SOURCE_AND_HOST);
}

void CUBandwidthAllToHostBidir::execute()
{
    startEngine(CopyAgentPolicy::SOURCE_AND_HOST);
}

/* BandwidthDeviceToAll */

void BandwidthDeviceToAll::startEngine(CopyAgentPolicy policy)
{
    int                        devcount = _conf.device_count;
    std::vector<const Buffer*> sourceBuffers(devcount);
    std::vector<const Buffer*> destinationBuffers(devcount);

    for (int src = 0; src < devcount; src++)
    {
        for (int dst = 0; dst < devcount; dst++)
        {
            sourceBuffers[dst] = new DeviceBuffer(src, _conf.bw_bufsize);
            if (src == dst)
            {
                destinationBuffers[dst] = new HostBuffer(dst, _conf.bw_bufsize);
            }
            else
            {
                destinationBuffers[dst] = new DeviceBuffer(dst, _conf.bw_bufsize);
            }
        }

        runEngine(sourceBuffers, destinationBuffers, policy);

        for (int dev = 0; dev < devcount; dev++)
        {
            delete sourceBuffers[dev];
            delete destinationBuffers[dev];
            sourceBuffers[dev]      = nullptr;
            destinationBuffers[dev] = nullptr;
        }
    }
}

void SDMABandwidthDeviceToAllWrite::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void CUBandwidthDeviceToAllWrite::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void SDMABandwidthDeviceToAllRead::execute()
{
    startEngine(CopyAgentPolicy::DESTINATION);
}

void CUBandwidthDeviceToAllRead::execute()
{
    startEngine(CopyAgentPolicy::DESTINATION);
}

/* BandwidthDeviceRing */

void BandwidthDeviceRingUnidir::startEngine(CopyAgentPolicy policy)
{
    int                        devcount = _conf.device_count;
    std::vector<const Buffer*> sourceBuffers(devcount);
    std::vector<const Buffer*> destinationBuffers(devcount);

    for (int dev = 0; dev < devcount; dev++)
    {
        int dst                 = (dev + 1) % devcount;
        sourceBuffers[dev]      = new DeviceBuffer(dev, _conf.bw_bufsize);
        destinationBuffers[dev] = new DeviceBuffer(dst, _conf.bw_bufsize);
    }

    runEngine(sourceBuffers, destinationBuffers, policy);

    for (int src = 0; src < devcount; src++)
    {
        delete sourceBuffers[src];
        delete destinationBuffers[src];
    }
}

void SDMABandwidthDeviceRingUnidirWrite::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void CUBandwidthDeviceRingUnidirWrite::execute()
{
    startEngine(CopyAgentPolicy::SOURCE);
}

void SDMABandwidthDeviceRingUnidirRead::execute()
{
    startEngine(CopyAgentPolicy::DESTINATION);
}

void CUBandwidthDeviceRingUnidirRead::execute()
{
    startEngine(CopyAgentPolicy::DESTINATION);
}

void BandwidthDeviceRingBidir::startEngine(CopyAgentPolicy policy)
{
    int                                     devcount = _conf.device_count;
    std::vector<std::vector<const Buffer*>> bufferPairs(2);
    std::vector<std::vector<const Buffer*>> peerBufferPairs(2);

    for (int i = 0; i < 2; i++)
    {
        peerBufferPairs[i].resize(devcount);
        bufferPairs[i].resize(devcount);

        for (int dev = 0; dev < devcount; dev++)
        {
            int dst                 = (dev + 1) % devcount;
            bufferPairs[i][dev]     = new DeviceBuffer(dev, _conf.bw_bufsize);
            peerBufferPairs[i][dev] = new DeviceBuffer(dst, _conf.bw_bufsize);
        }
    }

    runEngine(bufferPairs, peerBufferPairs, policy);

    for (int i = 0; i < 2; i++)
    {
        for (int dev = 0; dev < devcount; dev++)
        {
            delete bufferPairs[i][dev];
            delete peerBufferPairs[i][dev];
        }
    }
}

void SDMABandwidthDeviceRingBidirWrite::execute()
{
    startEngine(CopyAgentPolicy::SOURCE_AND_DESTINATION);
}

void CUBandwidthDeviceRingBidirWrite::execute()
{
    startEngine(CopyAgentPolicy::SOURCE_AND_DESTINATION);
}

void SDMABandwidthDeviceRingBidirRead::execute()
{
    startEngine(CopyAgentPolicy::DESTINATION_AND_SOURCE);
}

void CUBandwidthDeviceRingBidirRead::execute()
{
    startEngine(CopyAgentPolicy::DESTINATION_AND_SOURCE);
}
