#include <cstdlib>
#include <iostream>

#include "Testcase.hpp"
#include "common.h"
#include "hip/hip_runtime.h"

struct config conf;

void print_device_info()
{
    printf("Using devices:\n");
    for (int i = 0; i < conf.device_count; i++)
    {
        hipDeviceProp_t p;
        HIP_EXIT_ON_FAILURE(hipGetDeviceProperties(&p, i));
        printf("\t%d: %04x:%02x:%02x.0 %-25s (%zuM memory, %d CUs)\n",
               i,
               p.pciDomainID,
               p.pciBusID,
               p.pciDeviceID,
               p.name,
               p.totalGlobalMem >> 20,
               p.multiProcessorCount);
    }
    printf("\n");
}

void print_config(const config& cfg)
{
    printf("Configuration:\n");
    printf("\tMeasurement iterations:           %u\n", cfg.iterations);
    printf("\tBandwidth-bound transfer size:    %uM\n", cfg.bw_bufsize >> 20);
    printf("\tLatency-bound transfer size:      %uK\n", cfg.lat_bufsize >> 10);
    printf("\tLatency accesses:                 %u\n", cfg.lat_accesses);
}

auto create_config(int argc, char** argv) -> int
{
    // po::options_description desc("rocmbandwidth options");
    // po::variables_map       vm;

    // desc.add_options()
    //     /* --help / -h: conventional flag; no value, just a boolean presence. */
    //     ("help,h", "Print this help message and exit")

    //     /* --size / -s: bandwidth transfer size in MBs. */
    //     ("size,s",
    //      po::value<uint64_t>()->default_value(conf.bw_bufsize),
    //      ("Transfer size for bandwidth measurements in MB (default: "
    //       + std::to_string(conf.bw_bufsize >> 20) + ")")
    //          .c_str())

    //     /* --iterations / -n: how many timed repetitions to run. */
    //     ("iterations,n",
    //      po::value<uint64_t>()->default_value(conf.iterations),
    //      ("Number of timed iterations (default: " + std::to_string(conf.iterations) + ")").c_str());

    // try
    // {
    //     po::store(po::parse_command_line(argc, argv, desc), vm);
    //     po::notify(vm);
    // }
    // catch (const po::error& e)
    // {
    //     fprintf(stderr, "Argument error: %s\n\n", e.what());
    //     std::cerr << desc << "\n";
    //     exit(EXIT_FAILURE);
    // }

    // if (vm.count("help"))
    // {
    //     std::cout << desc << "\n";
    //     exit(EXIT_SUCCESS);
    // }

    (void)argc;
    (void)argv;

    HIP_EXIT_ON_FAILURE(hipGetDeviceCount(&conf.device_count));
    if (conf.device_count <= 0)
    {
        fprintf(stderr, "error: no HIP devices found\n");
        exit(EXIT_FAILURE);
    }
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
            if (err == hipErrorPeerAccessAlreadyEnabled)
            {
                /* Clear the sticky error so later hipGetLastError() checks don't trip on it */
                (void)hipGetLastError();
            }
            if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled)
            {
                printf("Failed to enable P2P from device %d to %d: %s\n",
                       dev,
                       peer,
                       hipGetErrorString(err));
                continue;
            }
            printf("Enabled P2P from GPU %d to GPU %d\n", dev, peer);
        }
    }
}

auto main(int argc, char** argv) -> int
{
    std::vector<std::unique_ptr<Testcase>> testcases;

    create_config(argc, argv);
    print_config(conf);
    print_device_info();
    enable_p2p();

    testcases.push_back(std::make_unique<LatencyHostToDevice>(conf));
    // testcases.push_back(std::make_unique<LatencyDeviceToDevice>(conf));

    testcases.push_back(std::make_unique<SDMABandwidthHostToDeviceUnidir>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthHostToDeviceBidir>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthDeviceToHostUnidir>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthDeviceToHostBidir>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthDeviceLocalCopy>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthDeviceToDeviceUnidirWrite>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthDeviceToDeviceUnidirRead>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthDeviceToDeviceBidirWrite>(conf));
    testcases.push_back(
        std::make_unique<SDMABandwidthDeviceToDeviceBidirRead>(conf)); /* TODO: remove */
    testcases.push_back(std::make_unique<SDMABandwidthHostToAllUnidir>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthHostToAllBidir>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthAllToHostUnidir>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthAllToHostBidir>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthDeviceToAllWrite>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthDeviceToAllRead>(conf));
    // testcases.push_back(std::make_unique<SDMABandwidthAllToDeviceWrite>(conf)); /* TODO: implement */
    // testcases.push_back(std::make_unique<SDMABandwidthAllToDeviceRead>(conf)); /* TODO: implement */
    testcases.push_back(std::make_unique<SDMABandwidthDeviceRingUnidirWrite>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthDeviceRingUnidirRead>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthDeviceRingBidirWrite>(conf));
    testcases.push_back(std::make_unique<SDMABandwidthDeviceRingBidirRead>(conf));

    testcases.push_back(std::make_unique<CUBandwidthHostToDeviceUnidir>(conf));
    testcases.push_back(std::make_unique<CUBandwidthHostToDeviceBidir>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceToHostUnidir>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceToHostBidir>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceLocalCopy>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceToDeviceUnidirWrite>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceToDeviceUnidirRead>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceToDeviceBidirWrite>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceToDeviceBidirRead>(conf));
    testcases.push_back(std::make_unique<CUBandwidthHostToAllUnidir>(conf));
    testcases.push_back(std::make_unique<CUBandwidthHostToAllBidir>(conf));
    testcases.push_back(std::make_unique<CUBandwidthAllToHostUnidir>(conf));
    testcases.push_back(std::make_unique<CUBandwidthAllToHostBidir>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceToAllWrite>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceToAllRead>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceRingUnidirWrite>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceRingUnidirRead>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceRingBidirWrite>(conf));
    testcases.push_back(std::make_unique<CUBandwidthDeviceRingBidirRead>(conf));

    for (auto& tc : testcases)
    {
        printf("\nRunning %s\n", tc->getKey());
        /* Flush so the test name comes before any errors the test prints on stderr, even when
         * stdout is a pipe or file (fully buffered) */
        fflush(stdout);
        tc->execute();
        printf("%s", tc->getResult().c_str());
        fflush(stdout);
    }

    return EXIT_SUCCESS;
}
