#include <boost/assign/std/vector.hpp>
#include <boost/program_options.hpp>
#include <cstdlib>
#include <iostream>

#include "Testcase.hpp"
#include "common.h"
#include "hip/hip_runtime.h"

namespace po = boost::program_options;
using namespace boost::assign;

struct config conf;

void print_device_info()
{
    printf("Using devices:\n");
    for (int i = 0; i < conf.device_count; i++)
    {
        hipDeviceProp_t p;
        HIP_EXIT_ON_FAILURE(hipGetDeviceProperties(&p, i));
        printf("\t%d: %04x:%02x:%02d.0 %-25s %8ldM\n",
               i,
               p.pciDomainID,
               p.pciBusID,
               p.pciDeviceID,
               p.name,
               p.totalGlobalMem >> 20);
    }
    printf("\n");
}

void print_config(const config& cfg)
{
    printf("Configuration:\n");
    printf("\tMeasurement iterations:           %-4d\n", cfg.iterations);
    printf("\tBandwidth-bound transfer size:    %-4dM\n", cfg.bw_bytes >> 20);
    // printf("\tLatency-bound transfer size:      %-4d\n", cfg.lat_bytes);
}

auto create_config(int argc, char** argv) -> int
{
    po::options_description desc("rocmbandwidth options");
    po::variables_map       vm;

    desc.add_options()
        /* --help / -h: conventional flag; no value, just a boolean presence. */
        ("help,h", "Print this help message and exit")

        /* --size / -s: bandwidth transfer size in MBs. */
        ("size,s",
         po::value<uint64_t>()->default_value(conf.bw_bytes),
         ("Transfer size for bandwidth measurements in MB (default: "
          + std::to_string(conf.bw_bytes >> 20) + ")")
             .c_str())

        /* --iterations / -n: how many timed repetitions to run. */
        ("iterations,n",
         po::value<uint64_t>()->default_value(conf.iterations),
         ("Number of timed iterations (default: " + std::to_string(conf.iterations) + ")").c_str());

    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (const po::error& e)
    {
        fprintf(stderr, "Argument error: %s\n\n", e.what());
        std::cerr << desc << "\n";
        exit(EXIT_FAILURE);
    }

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        exit(EXIT_SUCCESS);
    }

    HIP_EXIT_ON_FAILURE(hipGetDeviceCount(&conf.device_count));
    return 0;
}

void enable_p2p()
{
    DeviceGuard guard;
    for (int dev = 0; dev < conf.device_count; dev++)
    {
        HIP_CHECK(hipSetDevice(dev));
        for (int peer = 0; peer < conf.device_count; peer++)
        {
            int can_access;
            if (dev == peer)
            {
                continue;
            }

            HIP_CHECK(hipDeviceCanAccessPeer(&can_access, dev, peer));
            if (!can_access)
            {
                printf("Device %d cannot P2P access %d, transfers will be host staged\n",
                       dev,
                       peer);
                continue;
            }

            hipError_t err = hipDeviceEnablePeerAccess(peer, 0);
            // hipError_t err = hipDeviceDisablePeerAccess(peer);
            if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled)
            {
                printf("Failed to enable P2P from device %d to %d: %s\n",
                       dev,
                       peer,
                       hipGetErrorString(err));
            }
            printf("Enabled P2P from GPU %d to GPU %d\n", dev, peer);
        }
    }
}

auto main(int argc, char** argv) -> int
{
    std::vector<std::unique_ptr<Testcase>> bandwidthTestcases;

    create_config(argc, argv);
    print_config(conf);
    print_device_info();
    enable_p2p();

    // bandwidthTestcases += std::make_unique<BandwidthHostToDeviceUnidir>();
    // bandwidthTestcases += std::make_unique<BandwidthHostToDeviceBidir>();
    // bandwidthTestcases += std::make_unique<BandwidthDeviceToHostUnidir>();
    // bandwidthTestcases += std::make_unique<BandwidthDeviceToHostBidir>();
    // bandwidthTestcases += std::make_unique<BandwidthDeviceLocalCopy>();
    // bandwidthTestcases += std::make_unique<BandwidthDeviceToDeviceUnidirWrite>();
    // bandwidthTestcases += std::make_unique<BandwidthDeviceToDeviceBidirWrite>();
    // bandwidthTestcases += std::make_unique<BandwidthDeviceToDeviceUnidirRead>();
    // bandwidthTestcases += std::make_unique<BandwidthDeviceToDeviceBidirRead>();
    // bandwidthTestcases += std::make_unique<BandwidthHostToAllUnidir>();
    // bandwidthTestcases += std::make_unique<BandwidthHostToAllBidir>();
    // bandwidthTestcases += std::make_unique<BandwidthAllToHostUnidir>();
    // bandwidthTestcases += std::make_unique<BandwidthAllToHostBidir>();
    // bandwidthTestcases += std::make_unique<BandwidthDeviceToAllWrite>();
    bandwidthTestcases += std::make_unique<BandwidthDeviceToAllRead>();
    // bandwidthTestcases += std::make_unique<BandwidthDeviceRingWrite>();
    // bandwidthTestcases += std::make_unique<BandwidthDeviceRingRead>();

    for (auto& tc : bandwidthTestcases)
    {
        printf("\nRunning %s\n", tc->getKey());
        tc->execute(conf);
        printf("%s", tc->getResult().c_str());
    }

    return EXIT_SUCCESS;
}