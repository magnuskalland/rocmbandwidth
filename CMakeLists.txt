# ==============================================================================
# Minimum CMake version.
# 3.21+ is required for first-class HIP language support (enable_language(HIP)
# without needing a separate FindHIP module). If your distro ships an older
# CMake you can install a newer one from https://cmake.org/download/.
# ==============================================================================
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# ==============================================================================
# Project declaration.
#
# LANGUAGES tells CMake which compilers to locate up front.
#   CXX  – standard C++ compiler (g++ / clang++)
#   HIP  – AMD HIP compiler (hipcc, backed by clang)
#
# Listing HIP here triggers CMake's built-in ROCm/HIP toolchain detection;
# it sets CMAKE_HIP_COMPILER and related variables automatically when it
# finds a ROCm installation (usually under /opt/rocm).
# ==============================================================================
project(rocmbandwidth
    VERSION 0.0.1
    LANGUAGES CXX HIP
)

# ==============================================================================
# C++ / HIP standard.
# C++17 is the practical minimum for modern Boost headers and ROCm.
# EXTENSIONS OFF keeps it to -std=c++17 rather than -std=gnu++17.
# ==============================================================================
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ Standard in use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# ==============================================================================
# Build type.
# Defaults to RelWithDebInfo (optimised + debug symbols) when the user has not
# passed -DCMAKE_BUILD_TYPE=... on the command line.
# ==============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
        "Build type (Debug|Release|RelWithDebInfo|MinSizeRel)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS}     --save-temps")
    set(CMAKE_HIP_FLAGS     "${CMAKE_HIP_FLAGS}     --save-temps")
endif()

# ==============================================================================
# ROCm / HIP configuration.
#
# ROCM_PATH defaults to /opt/rocm but can be overridden:
#   cmake -DROCM_PATH=/custom/rocm/path .
#
# We append the ROCm CMake config directories to CMAKE_PREFIX_PATH so that
# find_package() can locate hip-config.cmake (and rocblas, etc.) without
# the user needing to set environment variables.
# ==============================================================================
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm installation")
endif()
message(STATUS "ROCm path: ${ROCM_PATH}")

list(APPEND CMAKE_PREFIX_PATH
    "${ROCM_PATH}"
    "${ROCM_PATH}/lib/cmake"
    "${ROCM_PATH}/hip/lib/cmake/hip"
)

# ==============================================================================
# GPU architecture targets.
#
# Controls which GCN/RDNA ISAs are compiled via --offload-arch.
# Common values:
#   gfx900  – Vega 10   (RX Vega 56/64, Instinct MI25)
#   gfx906  – Vega 20   (Radeon VII, Instinct MI50/60)
#   gfx908  – CDNA 1    (Instinct MI100)
#   gfx90a  – CDNA 2    (Instinct MI200/250)
#   gfx1030 – RDNA 2    (RX 6800/6900 XT)
#   gfx1100 – RDNA 3    (RX 7900 XT/XTX)
#
# Override from the command line:
#   cmake -DGPU_TARGETS="gfx90a;gfx1100" .
# ==============================================================================
if(NOT DEFINED GPU_TARGETS)
    set(GPU_TARGETS "gfx1200" CACHE STRING
        "Semicolon-separated list of GPU architectures to target")
endif()
message(STATUS "GPU targets: ${GPU_TARGETS}")

# Build a list of --offload-arch=<gfx> flags from the semicolon-separated list.
# This variable is consumed by src/CMakeLists.txt.
set(HIP_ARCH_FLAGS "")
foreach(arch IN LISTS GPU_TARGETS)
    list(APPEND HIP_ARCH_FLAGS "--offload-arch=${arch}")
endforeach()

# ==============================================================================
# Find HIP.
#
# Provides the imported targets:
#   hip::host   – ROCm runtime + HIP headers for CPU-side calls
#   hip::device – compile flags / headers for GPU device code
# ==============================================================================
find_package(hip REQUIRED CONFIG)
message(STATUS "Found HIP: ${hip_VERSION}")

# ==============================================================================
# Enforce out-of-source build.
#
# Running `cmake .` (in-source) would scatter Makefiles and CMake cache files
# all over the project root, making the tree hard to read and .gitignore hard
# to maintain. We abort with a helpful message so the developer uses the
# correct invocation instead:
#
#   cmake -S . -B build && make -C build
#
# CMAKE_SOURCE_DIR – the root where this CMakeLists.txt lives.
# CMAKE_BINARY_DIR – where CMake writes its generated files (set by -B build).
# When the two are equal the user ran cmake from inside the source tree.
# ==============================================================================
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "\n"
        "  In-source builds are not allowed.\n"
        "  Please use an out-of-source build instead:\n"
        "\n"
        "    cmake -S . -B build && make -C build\n"
        "\n"
        "  This keeps generated Makefiles and cache files inside build/,\n"
        "  leaving the project root clean with only /src and /build.\n"
    )
endif()

# ==============================================================================
# Global output directory.
#
# CMAKE_RUNTIME_OUTPUT_DIRECTORY tells CMake where to place compiled
# executables for ALL targets defined anywhere in the project (root or any
# subdirectory). Setting it here once means src/CMakeLists.txt does not need
# per-target set_target_properties() calls.
#
# CMAKE_BINARY_DIR is the build/ directory passed via -B build, so this
# expands to <project_root>/build/ — exactly where we want the binary.
# ==============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# ==============================================================================
# Descend into src/.
#
# add_subdirectory() processes src/CMakeLists.txt, which defines the actual
# executable target using the variables and imported targets set up above.
# This keeps project-wide configuration (finding packages, setting standards)
# cleanly separated from target-level concerns (sources, flags, linking).
# ==============================================================================
add_subdirectory(src)

# ==============================================================================
# Summary banner – printed at the end of every cmake run.
# ==============================================================================
message(STATUS "--------------------------------------------------------------")
message(STATUS "  Project   : ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ std   : ${CMAKE_CXX_STANDARD}")
message(STATUS "  GPU arches: ${GPU_TARGETS}")
message(STATUS "  ROCm path : ${ROCM_PATH}")
message(STATUS "  Binary dir: ${CMAKE_BINARY_DIR}")
message(STATUS "--------------------------------------------------------------")