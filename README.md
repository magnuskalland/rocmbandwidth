# rocmbandwidth

`rocmbandwidth` is a ROCm-native benchmarking tool for measuring memory-copy
bandwidth and latency between the host and one or more AMD GPUs, and between
GPUs themselves. It runs a fixed suite of ~39 micro-benchmarks that cover
host↔device, device↔device, and multi-GPU "all links at once" transfer
patterns, using both the ROCm runtime's own copy path (`hipMemcpyAsync`,
generally backed by the GPU's DMA/SDMA copy engines) and a hand-written HIP
kernel that drives the copy from the GPU's compute units directly.

> **Note on scope:** this README documents the tool as it exists in this
> source tree, including a few rough edges (disabled CLI parsing, one
> unimplemented test category) that are called out explicitly below so you
> don't go looking for flags or tests that aren't actually wired up yet.

## Contents

- [rocmbandwidth](#rocmbandwidth)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Building](#building)
    - [CMake configuration options](#cmake-configuration-options)
  - [Running](#running)
  - [Configuring a run (no CLI flags yet)](#configuring-a-run-no-cli-flags-yet)
  - [What gets measured](#what-gets-measured)
    - [Test catalog](#test-catalog)
  - [Reading the output](#reading-the-output)
  - [Peer-to-peer (P2P) access](#peer-to-peer-p2p-access)
  - [Known limitations](#known-limitations)
  - [Development notes](#development-notes)

## Prerequisites

| Requirement | Notes |
|---|---|
| Linux | ROCm's supported platform; the build enforces nothing OS-specific beyond what ROCm itself requires. |
| [ROCm](https://rocm.docs.amd.com/) with HIP | Provides the HIP compiler/runtime and headers. Built and last compiled against ROCm 7.2.2, per the checked-in `compile_commands.json`; any reasonably recent ROCm release with HIP + `hip-config.cmake` should work. |
| CMake ≥ 3.21 | Required for CMake's first-class `LANGUAGES ... HIP` support (`enable_language(HIP)` without a separate `FindHIP` module). |
| A C++20-capable host compiler | Used for the `.cpp` translation units (`g++`/`clang++`); picked up via the usual `CXX`/`CMAKE_CXX_COMPILER` mechanisms. |
| ROCm's HIP compiler (`hipcc`/`amdclang++`) | Used for the `.hip` device/kernel translation unit; located automatically by CMake once it finds your ROCm install. |
| `make` (or another CMake generator) | The examples below use the default "Unix Makefiles" generator; pass `-G Ninja` if you'd rather use Ninja. |
| One or more AMD GPUs with the `amdgpu` driver loaded, and permission to access them (typically membership in the `render`/`video` groups) | Only needed to *run* the benchmark — building doesn't require a physical GPU, since HIP device code is cross-compiled for whatever `--offload-arch` values you request. |

**Not required:** despite a couple of leftover comments in the CMake files
that mention Boost (`Boost::filesystem`, `Boost::system`, "modern Boost
headers"), the project does **not** currently call `find_package(Boost)`
anywhere and does not link against Boost. You do not need Boost installed to
build this tree today — see [Known limitations](#known-limitations) for
context on why those comments exist.

## Building

```bash
cmake -S . -B build
make -C build
```

The resulting binary is `build/rocmbandwidth`.

In-source builds (`cmake .`) are explicitly rejected — the root
`CMakeLists.txt` checks `CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR` and
exits with an error telling you to use `-B build` instead, so the project
root stays clean.

### CMake configuration options

| Option | Default | Description |
|---|---|---|
| `ROCM_PATH` | `/opt/rocm` | Path to your ROCm install. Appended to `CMAKE_PREFIX_PATH` so `find_package(hip)` can locate `hip-config.cmake` without extra environment setup. |
| `GPU_TARGETS` | `gfx1200` | Semicolon-separated list of GPU ISAs to compile device code for; turned into one `--offload-arch=<gfx>` flag per entry. |
| `CMAKE_BUILD_TYPE` | `Debug` | Standard CMake build types (`Debug`/`Release`/`RelWithDebInfo`/`MinSizeRel`). The `Debug` default additionally appends `--save-temps` to both the C++ and HIP compile flags, which leaves preprocessed/intermediate files next to each object file under `build/` — handy for inspecting generated device code, but adds clutter. Use `Release` for anything perf-sensitive. |

Examples:

```bash
# Build for a single target architecture
cmake -S . -B build -DGPU_TARGETS=gfx90a
make -C build

# Multiple architectures, a non-default ROCm install, optimized build
cmake -S . -B build \
  -DGPU_TARGETS="gfx90a;gfx1100" \
  -DROCM_PATH=/opt/rocm-7.2.2 \
  -DCMAKE_BUILD_TYPE=Release
make -C build -j"$(nproc)"
```

If you don't know your GPU's target ISA, `rocminfo | grep gfx` (look for
the `Name:` field of each GPU agent) will print it.

`CMAKE_CXX_STANDARD` (20) and `CMAKE_HIP_STANDARD` (17) are also cached
variables, but they're set by the project to match what the code actually
uses and aren't meant to be overridden.

## Running

```bash
./build/rocmbandwidth
```

No arguments are needed (and, currently, none are read — see the next
section). On startup the tool:

1. Prints the active configuration (iteration count, transfer sizes).
2. Enumerates and prints every visible GPU (`hipGetDeviceCount` /
   `hipGetDeviceProperties`).
3. Attempts to enable peer-to-peer access between every ordered pair of
   GPUs, printing a line for any pair that doesn't support it.
4. Runs the full built-in suite of ~39 test cases in a fixed order,
   printing each test's key followed by its result(s) as it completes.

To restrict which GPUs are visible to a run without rebuilding, use HIP's
standard environment variable, e.g.:

```bash
HIP_VISIBLE_DEVICES=0,1 ./build/rocmbandwidth
```

## Configuring a run (no CLI flags yet)

`main.cpp` contains a `create_config()` function with a complete
Boost.Program_options-based parser sketched out — `--help`/`-h`,
`--size`/`-s`, `--iterations`/`-n` — but the entire block is commented out,
and the function otherwise ignores `argc`/`argv` entirely. **No
command-line flags currently work**; the binary always runs with the
compiled-in defaults below.

To change these, edit the `config` struct's default member initializers in
`src/common.h` and rebuild:

| Field | Default | Meaning |
|---|---|---|
| `iterations` | `16` | Timed copy repetitions per bandwidth measurement (plus 4 untimed warm-up copies issued beforehand on the same stream). |
| `bw_bufsize` | `512 << 20` (512 MiB) | Buffer size used for every bandwidth test. |
| `lat_bufsize` | `2 << 20` (2 MiB) | Buffer size used for the latency (pointer-chase) test. |
| `lat_accesses` | `1,000,000` | Number of chained pointer dereferences performed per latency measurement. |

If you want working CLI flags, the scaffolding in `main.cpp` is a
reasonable starting point, but note that `find_package(Boost ...)` and
linking `Boost::program_options` aren't set up anywhere in the current
`CMakeLists.txt` files — that would need to be added first.

## What gets measured

Every bandwidth topology is exercised by two independent copy engines, and
each therefore shows up twice in the output under a distinct key prefix:

- **`sdma_bw_*`** — the **S**ystem **DMA** engine. Issues `iterations`
  repetitions of `hipMemcpyAsync()` on a stream; this exercises the ROCm
  runtime's own copy path, generally backed by the GPU's dedicated DMA/SDMA
  copy engines (or a host `memcpy` for host-only transfers).
- **`cu_bw_*`** — the **C**ompute **U**nit engine. A custom HIP kernel
  (one thread-block per compute unit, vectorized `float4` loads/stores)
  performs the copy directly on the GPU's shader/compute units, bypassing
  the DMA engines. The block size depends on the path: 1024 threads when
  both buffers are in the executing GPU's own memory, 64 threads when either
  side is across PCIe (host memory or a peer GPU). Those were the best
  settings measured on 2× MI210. To tune for other hardware, override them
  with the environment variables `ROCMBW_CU_THREADS`,
  `ROCMBW_CU_BLOCKS_PER_CU` and `ROCMBW_CU_PIPELINES` (1, 2, 4 or 8), which
  apply to every path.

The latency test (`latency_host_to_device`) uses a third executor that
times a pointer-chase through a randomly-shuffled linked list built inside
the buffer, reporting nanoseconds per access.

### Test catalog

For device-to-device and multi-GPU patterns, `write` means the transfer is
issued from the **source** GPU's context (a push), and `read` means it's
issued from the **destination** GPU's context (a pull) — see
`CopyAgentPolicy` in `src/common.h` / `TransferExecutor::getAgent()` if you
want the exact rules. "Sequential" tests measure one link per invocation;
"Concurrent" tests launch every listed link at once on separate streams and
report a per-link breakdown plus an `Aggregate:` line. All transfers in a test
start from a common barrier; a link's figure is its total bytes over the time
until its slowest direction finished, and the aggregate is all bytes over the
time until the last transfer finished (not the sum of the per-link numbers).

| Category | Pattern | Mode | Keys run (`sdma_bw_…` / `cu_bw_…`) |
|---|---|---|---|
| Latency | Host ↔ one GPU, per GPU | Sequential | `latency_host_to_device` *(latency executor only; no sdma/cu split)* |
| Host↔Device | Host ↔ one GPU, per GPU, one direction | Sequential, unidir | `host_to_device_unidir` |
| Host↔Device | Host ↔ one GPU, per GPU, both directions at once | Sequential, bidir | `host_to_device_bidir` |
| Device↔Host | One GPU ↔ Host, per GPU, one direction | Sequential, unidir | `device_to_host_unidir` |
| Device↔Host | One GPU ↔ Host, per GPU, both directions at once | Sequential, bidir | `device_to_host_bidir` |
| Local copy | One GPU, buffer → buffer on itself | Sequential | `device_local_copy` |
| Device↔Device | Every ordered GPU pair, one direction | Sequential, unidir | `device_to_device_unidir_write`, `device_to_device_unidir_read` |
| Device↔Device | Every ordered GPU pair, both directions at once | Sequential, bidir | `device_to_device_bidir_write`, `device_to_device_bidir_read` |
| Host→All | One host buffer per GPU, all GPUs copy concurrently | Concurrent, unidir | `host_to_all_unidir` |
| Host↔All | Same, but each link runs both directions at once | Concurrent, bidir | `host_to_all_bidir` |
| All→Host | Every GPU copies to its own host buffer concurrently | Concurrent, unidir | `all_to_host_unidir` |
| All↔Host | Same, both directions at once per link | Concurrent, bidir | `all_to_host_bidir` |
| Device→All | Each GPU in turn broadcasts to every other GPU concurrently (one run per "hub" GPU) | Concurrent, unidir | `device_to_all_write`, `device_to_all_read` |
| Device ring | Every GPU *i* ↔ GPU *(i+1) mod N*, all links concurrent | Concurrent, unidir | `device_ring_unidir_write`, `device_ring_unidir_read` |
| Device ring | Same ring topology, both directions at once per link | Concurrent, bidir | `device_ring_bidir_write`, `device_ring_bidir_read` |

## Reading the output

Bandwidth is reported in **decimal** GB/s (bytes × 1e-9 / seconds); latency
in nanoseconds. Every destination buffer is verified against the expected
pattern after each transfer; if verification fails you'll see
`error: destination buffer corrupt` on stderr and a `0.0` result for that
test instead of a bogus number — treat that as a correctness bug to
investigate, not just "slow."

Illustrative shape of the output on a 2-GPU system (numbers are made up —
run the tool on your own hardware for real ones):

```
Configuration:
        Measurement iterations:           16
        Bandwidth-bound transfer size:    512M
        Latency-bound transfer size:      1000000
Using devices:
        0: 0000:0c:00.0 AMD Instinct MI210          (65536M memory, 104 CUs)
        1: 0000:22:00.0 AMD Instinct MI210          (65536M memory, 104 CUs)

Enabled P2P from GPU 0 to GPU 1
Enabled P2P from GPU 1 to GPU 0

Running latency_host_to_device
        CPU -> GPU 0: 812 ns
        CPU -> GPU 1: 799 ns

Running sdma_bw_host_to_device_unidir
        CPU -> GPU 0: 24.31 GB/s
        CPU -> GPU 1: 24.28 GB/s

Running cu_bw_device_to_all_write
        GPU 0 -> GPU 0: 0.00 GB/s
        GPU 0 -> GPU 1: 41.72 GB/s
        Aggregate: 41.72 GB/s
...
```

## Peer-to-peer (P2P) access

`enable_p2p()` runs once at startup and tries `hipDeviceEnablePeerAccess`
for every ordered pair of visible GPUs. If a pair doesn't support P2P, a
line like `Device 0 cannot P2P access 1, transfers will be host staged` is
printed and the process moves on — device-to-device transfers for that pair
will then be routed through host memory by the ROCm runtime, which can
substantially lower the reported bandwidth. Check the startup log when
interpreting `device_to_device_*` / `device_ring_*` results.

## Known limitations

- **No working CLI flags** — see
  [Configuring a run](#configuring-a-run-no-cli-flags-yet) above.
- **`latency_device_to_device` is implemented but not run.** The
  `LatencyDeviceToDevice` test class exists in `src/Testcase.hpp`/`.cpp` and
  works the same way as the host↔device latency test, but its
  `push_back` in `main.cpp` is commented out, so only host↔device latency
  is currently exercised by default.
- **`*_all_to_device_write` / `*_all_to_device_read` are not implemented.**
  Unlike the item above, these have no backing classes at all — `main.cpp`
  only has a `/* TODO: implement */` comment where they'd go.
- **`sdma_bw_device_to_device_bidir_read` is flagged by the author.** Its
  `push_back` in `main.cpp` carries a `/* TODO: remove */` comment, so
  treat that particular result with a bit of caution — it may be dropped or
  reworked in a future revision.
- **`src/Output.hpp` is currently empty.** It looks like a placeholder for
  a future dedicated output/formatting module; today all result formatting
  happens ad hoc via `std::stringstream` in `Testcase.cpp`.
- **No `LICENSE` file** is currently included in this repository.
- **The checked-in `compile_commands.json`** is a stale snapshot from the
  author's own machine (paths under `/home/magnus/...`, and GPU targets
  that don't match the current `GPU_TARGETS` default), not something
  generated fresh by this tree. If you want one for editor tooling
  (clangd, etc.), generate your own — the top-level `CMakeLists.txt` does
  not currently set `CMAKE_EXPORT_COMPILE_COMMANDS`, so pass it explicitly:
  `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` and point your
  tooling at `build/compile_commands.json`.

## Development notes

- **Code style:** `.clang-format` (WebKit-based, rocBLAS-derived: 4-space
  indent, 100-column limit) and `.clang-tidy` are included at the repo
  root; run them before submitting changes.
- **Adding a new test case:** subclass one of the four shapes in
  `src/Testcase.hpp` (`Sequential`/`Concurrent` × `Unidir`/`Bidir`, or
  `LatencyTestCase`), give it an SDMA- and/or CU-flavored leaf class that
  picks a `SystemDMATransferEngine` or `ComputeUnitTransferEngine`, and
  `push_back` an instance of it in `main()` (`src/main.cpp`).
- Source layout:

  ```
  CMakeLists.txt          Top-level build config (ROCm/HIP discovery, GPU_TARGETS, etc.)
  src/
    CMakeLists.txt        Target definition, compiler flags, link libraries
    main.cpp               Entry point; device discovery, P2P setup, test list, driver loop
    common.h                config struct, HIP error-check macros, CopyAgentPolicy
    Buffer.hpp              HostBuffer / DeviceBuffer allocation wrappers
    TransferEngine.{hpp,cpp}    Orchestrates one measurement (streams, events, timing)
    TransferExecutor.{hpp,cpp}  The actual copy/latency implementations (SDMA/CU/Latency)
    Testcase.{hpp,cpp}      All concrete benchmark scenarios and output formatting
    kernels.{hpp,hip}       Device-side kernels: copy, pointer-chase, memset, verify, spin
    Output.hpp               Currently empty
  ```