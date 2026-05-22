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

class BandwidthHostToDeviceUnidir : public Testcase
{
public:
    BandwidthHostToDeviceUnidir() : Testcase("bw_host_to_device_unidir", "") {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
    void execute(const struct config& cfg) override;
};

class BandwidthHostToDeviceBidir : public Testcase
{
public:
    BandwidthHostToDeviceBidir() : Testcase("bw_host_to_device_bidir", "") {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
    void execute(const struct config& cfg) override;
};

/* BandwidthDeviceToHost */

class BandwidthDeviceToHostUnidir : public Testcase
{
public:
    BandwidthDeviceToHostUnidir() : Testcase("bw_device_to_host_unidir", "") {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
    void execute(const struct config& cfg) override;
};

class BandwidthDeviceToHostBidir : public Testcase
{
public:
    BandwidthDeviceToHostBidir() : Testcase("bw_device_to_host_bidir", "") {}
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

class BandwidthDeviceToDeviceUnidir : public Testcase
{
public:
    BandwidthDeviceToDeviceUnidir(std::string key, std::string desc) : Testcase(key, desc) {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
};

class BandwidthDeviceToDeviceUnidirWrite : public BandwidthDeviceToDeviceUnidir
{
public:
    BandwidthDeviceToDeviceUnidirWrite()
        : BandwidthDeviceToDeviceUnidir("bw_device_to_device_unidir_write", "")
    {}
    void execute(const struct config& cfg) override;
};

class BandwidthDeviceToDeviceUnidirRead : public BandwidthDeviceToDeviceUnidir
{
public:
    BandwidthDeviceToDeviceUnidirRead()
        : BandwidthDeviceToDeviceUnidir("bw_device_to_device_unidir_read", "")
    {}
    void execute(const struct config& cfg) override;
};

class BandwidthDeviceToDeviceBidir : public Testcase
{
public:
    BandwidthDeviceToDeviceBidir(std::string key, std::string desc) : Testcase(key, desc) {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
};

class BandwidthDeviceToDeviceBidirWrite : public BandwidthDeviceToDeviceBidir
{
public:
    BandwidthDeviceToDeviceBidirWrite()
        : BandwidthDeviceToDeviceBidir("bw_device_to_device_bidir_write", "")
    {}
    void execute(const struct config& cfg) override;
};

class BandwidthDeviceToDeviceBidirRead : public BandwidthDeviceToDeviceBidir
{
public:
    BandwidthDeviceToDeviceBidirRead()
        : BandwidthDeviceToDeviceBidir("bw_device_to_device_bidir_read", "")
    {}
    void execute(const struct config& cfg) override;
};

/* BandwidthHostToAll */

class BandwidthHostToAllUnidir : public Testcase
{
public:
    BandwidthHostToAllUnidir() : Testcase("bw_host_to_all_unidir", "") {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
    void execute(const struct config& cfg) override;
};

class BandwidthHostToAllBidir : public Testcase
{
public:
    BandwidthHostToAllBidir() : Testcase("bw_host_to_all_bidir", "") {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
    void execute(const struct config& cfg) override;
};

/* BandwidthAllToHost */

class BandwidthAllToHostUnidir : public Testcase
{
public:
    BandwidthAllToHostUnidir() : Testcase("bw_all_to_host_unidir", "") {}
    void doExecute(const struct config& conf, CopyAgentPolicy policy) override;
    void execute(const struct config& cfg) override;
};

class BandwidthAllToHostBidir : public Testcase
{
public:
    BandwidthAllToHostBidir() : Testcase("bw_all_to_host_bidir", "") {}
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
