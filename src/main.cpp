#include <algorithm>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
    printf("\tBandwidth-bound transfer size:    %luM\n",
           static_cast<unsigned long>(cfg.bw_bufsize >> 20));
    printf("\tLatency-bound transfer size:      %luK\n",
           static_cast<unsigned long>(cfg.lat_bufsize >> 10));
    printf("\tLatency accesses:                 %u\n", cfg.lat_accesses);
}

static void print_usage(const char* prog)
{
    printf("Usage: %s [options]\n"
           "\n"
           "Options:\n"
           "  -h, --help              Print this help message and exit\n"
           "  -s, --size SIZE         Transfer size for bandwidth measurements (default: %luM).\n"
           "                          Plain numbers are MiB; K, M and G suffixes are accepted.\n"
           "  -n, --iterations N      Number of timed iterations (default: %u)\n"
           "  -t, --tests PATTERNS    Only run tests whose key contains one of the\n"
           "                          comma-separated PATTERNS (e.g. -t sdma_bw,latency)\n"
           "  -l, --list              List all test keys and exit\n",
           prog,
           static_cast<unsigned long>(conf.bw_bufsize >> 20),
           conf.iterations);
}

/* Parses "512", "512M", "64K", "2G" (plain numbers are MiB). Returns 0 on error. */
static auto parse_size(const char* arg) -> uint64_t
{
    char*              end   = nullptr;
    unsigned long long value = std::strtoull(arg, &end, 10);
    if (end == arg || value == 0)
    {
        return 0;
    }

    uint64_t shift = 20;
    if (*end != '\0')
    {
        switch (*end)
        {
            case 'k':
            case 'K': shift = 10; break;
            case 'm':
            case 'M': shift = 20; break;
            case 'g':
            case 'G': shift = 30; break;
            default: return 0;
        }
        if (end[1] != '\0')
        {
            return 0;
        }
    }

    if (value > (UINT64_MAX >> shift))
    {
        return 0;
    }
    return static_cast<uint64_t>(value) << shift;
}

static auto split_patterns(const std::string& arg) -> std::vector<std::string>
{
    std::vector<std::string> patterns;
    std::stringstream        ss(arg);
    std::string              item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty())
        {
            patterns.push_back(item);
        }
    }
    return patterns;
}

struct cli_options
{
    std::vector<std::string> test_patterns;
    bool                     list_tests = false;
};

auto create_config(int argc, char** argv, cli_options& opts) -> int
{
    static const struct option long_options[] = {
        {      "help",       no_argument, nullptr, 'h'},
        {      "size", required_argument, nullptr, 's'},
        {"iterations", required_argument, nullptr, 'n'},
        {     "tests", required_argument, nullptr, 't'},
        {      "list",       no_argument, nullptr, 'l'},
        {     nullptr,                 0, nullptr,   0},
    };

    int c;
    while ((c = getopt_long(argc, argv, "hs:n:t:l", long_options, nullptr)) != -1)
    {
        switch (c)
        {
            case 'h': print_usage(argv[0]); exit(EXIT_SUCCESS);
            case 's':
                {
                    uint64_t size = parse_size(optarg);
                    /* Copies are done in 16-byte (float4) chunks and verified as uint64_t words */
                    if (size == 0 || size % 16 != 0)
                    {
                        fprintf(stderr, "Argument error: invalid size '%s'\n\n", optarg);
                        print_usage(argv[0]);
                        exit(EXIT_FAILURE);
                    }
                    conf.bw_bufsize = size;
                    break;
                }
            case 'n':
                {
                    char*         end = nullptr;
                    unsigned long n   = std::strtoul(optarg, &end, 10);
                    if (end == optarg || *end != '\0' || n == 0 || n > UINT32_MAX)
                    {
                        fprintf(stderr, "Argument error: invalid iteration count '%s'\n\n", optarg);
                        print_usage(argv[0]);
                        exit(EXIT_FAILURE);
                    }
                    conf.iterations = static_cast<uint32_t>(n);
                    break;
                }
            case 't': opts.test_patterns = split_patterns(optarg); break;
            case 'l': opts.list_tests = true; break;
            default:
                fprintf(stderr, "\n");
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (optind < argc)
    {
        fprintf(stderr, "Argument error: unexpected argument '%s'\n\n", argv[optind]);
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    HIP_EXIT_ON_FAILURE(hipGetDeviceCount(&conf.device_count));
    if (conf.device_count <= 0)
    {
        fprintf(stderr, "error: no HIP devices found\n");
        exit(EXIT_FAILURE);
    }
    return 0;
}

static auto test_selected(const char* key, const std::vector<std::string>& patterns) -> bool
{
    if (patterns.empty())
    {
        return true;
    }
    return std::any_of(patterns.begin(),
                       patterns.end(),
                       [key](const std::string& p)
                       { return std::string(key).find(p) != std::string::npos; });
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
    cli_options                            opts;

    create_config(argc, argv, opts);

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

    if (opts.list_tests)
    {
        for (auto& tc : testcases)
        {
            printf("%s\n", tc->getKey());
        }
        return EXIT_SUCCESS;
    }

    std::erase_if(testcases,
                  [&opts](const std::unique_ptr<Testcase>& tc)
                  { return !test_selected(tc->getKey(), opts.test_patterns); });
    if (testcases.empty())
    {
        fprintf(stderr, "error: no tests match the given --tests patterns (see --list)\n");
        return EXIT_FAILURE;
    }

    print_config(conf);
    print_device_info();
    enable_p2p();

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
