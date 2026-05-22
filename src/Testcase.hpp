#pragma once

#include "common.h"
#include <sstream>
#include <string>
#include <utility>

class Testcase
{
public:
    Testcase(std::string key, std::string desc) : _key(std::move(key)), _desc(std::move(desc)) {}

    virtual void execute(const struct config& conf) = 0;

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
    virtual void      doExecute(const struct config& conf, CopyAgentPolicy policy) = 0;
    std::stringstream _output;

private:
    std::string _key;
    std::string _desc;
};

/* BandwidthHostToDevice */

class BandwidthHostToDevice : public Testcase
{
public:
    BandwidthHostToDevice() : Testcase("bw_host_to_device", "") {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
    void execute(const struct config& cfg) override;
};

/* BandwidthDeviceToHost */

class BandwidthDeviceToHost : public Testcase
{
public:
    BandwidthDeviceToHost() : Testcase("bw_device_to_host", "") {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
    void execute(const struct config& cfg) override;
};

/* BandwidthDeviceLocalCopy */

class BandwidthDeviceLocalCopy : public Testcase
{
public:
    BandwidthDeviceLocalCopy() : Testcase("bw_device_local_copy", "") {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
    void execute(const struct config& cfg) override;
};

/* BandwidthDeviceToDevice */

class BandwidthDeviceToDevice : public Testcase
{
public:
    BandwidthDeviceToDevice(std::string key, std::string desc) : Testcase(key, desc) {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
};

class BandwidthDeviceToDeviceWrite : public BandwidthDeviceToDevice
{
public:
    BandwidthDeviceToDeviceWrite() : BandwidthDeviceToDevice("bw_device_to_device_write", "") {}
    void execute(const struct config& cfg) override;
};

class BandwidthDeviceToDeviceRead : public BandwidthDeviceToDevice
{
public:
    BandwidthDeviceToDeviceRead() : BandwidthDeviceToDevice("bw_device_to_device_read", "") {}
    void execute(const struct config& cfg) override;
};

/* BandwidthHostToAll */

class BandwidthHostToAll : public Testcase
{
public:
    BandwidthHostToAll() : Testcase("bw_host_to_all", "") {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
    void execute(const struct config& cfg) override;
};

/* BandwidthAllToHost */

class BandwidthAllToHost : public Testcase
{
public:
    BandwidthAllToHost() : Testcase("bw_all_to_host", "") {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
    void execute(const struct config& cfg) override;
};

/* BandwidthDeviceToAll */

class BandwidthDeviceToAll : public Testcase
{
public:
    BandwidthDeviceToAll(std::string key, std::string desc) : Testcase(key, desc) {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
};

class BandwidthDeviceToAllWrite : public BandwidthDeviceToAll
{
public:
    BandwidthDeviceToAllWrite() : BandwidthDeviceToAll("bw_device_to_all_write", "") {}
    void execute(const struct config& cfg) override;
};

class BandwidthDeviceToAllRead : public BandwidthDeviceToAll
{
public:
    BandwidthDeviceToAllRead() : BandwidthDeviceToAll("bw_device_to_all_read", "") {}
    void execute(const struct config& cfg) override;
};

/* BandwidthDeviceRing */

class BandwidthDeviceRing : public Testcase
{
public:
    BandwidthDeviceRing(std::string key, std::string desc) : Testcase(key, desc) {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
};

class BandwidthDeviceRingWrite : public BandwidthDeviceRing
{
public:
    BandwidthDeviceRingWrite() : BandwidthDeviceRing("bw_device_to_all_ring_write", "") {}
    void execute(const struct config& cfg) override;
};

class BandwidthDeviceRingRead : public BandwidthDeviceRing
{
public:
    BandwidthDeviceRingRead() : BandwidthDeviceRing("bw_device_to_all_ring_read", "") {}
    void execute(const struct config& cfg) override;
};
