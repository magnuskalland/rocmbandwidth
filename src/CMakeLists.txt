# ==============================================================================
# src/CMakeLists.txt
#
# This file is processed by add_subdirectory(src) in the root CMakeLists.txt.
# By the time CMake reaches this file, the following are already available:
#
#   Variables (set in root CMakeLists.txt):
#     PROJECT_NAME       – e.g. "MyROCmProject"
#     HIP_ARCH_FLAGS     – list of --offload-arch=<gfx> flags
#     CMAKE_SOURCE_DIR   – absolute path to the project root
#     CMAKE_CURRENT_SOURCE_DIR – absolute path to this src/ directory
#
#   Imported targets (from find_package calls in root CMakeLists.txt):
#     hip::host          – ROCm runtime + HIP CPU-side headers
#
# This file's sole job is to define the build target (the executable) and
# describe its sources, include paths, compile flags, and link dependencies.
# ==============================================================================

# ==============================================================================
# Collect source files.
#
# GLOB_RECURSE walks this src/ directory and collects every file whose name
# matches the given patterns:
#   *.cpp – host-side C++ translation units
#   *.hip – HIP device/kernel translation units (compiled by hipcc/clang)
#   *.hpp – headers (listed so IDEs show them in the project tree; they are
#           not compiled directly)
#
# CMAKE_CURRENT_SOURCE_DIR always points to the directory that owns this
# CMakeLists.txt (i.e. src/), regardless of where cmake was invoked from.
#
# !! IMPORTANT: CMake does NOT re-run automatically when you add or remove
# files with GLOB_RECURSE. Run `cmake .` again from the project root any
# time you create or delete a source file.
#
# Alternative – explicit file list (re-run not needed, but more maintenance):
#   set(PROJECT_SOURCES
#       main.cpp
#       bandwidth.hip
#       utils/timer.cpp
#   )
# ==============================================================================
file(GLOB_RECURSE PROJECT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hip"
)

file(GLOB_RECURSE PROJECT_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

if(NOT PROJECT_SOURCES)
    message(WARNING
        "No *.cpp or *.hip source files found under src/. "
        "Add source files there and re-run cmake.")
endif()

# ==============================================================================
# Executable target.
#
# add_executable() registers a build target named PROJECT_NAME (inherited from
# the root CMakeLists.txt, e.g. "MyROCmProject") and associates it with the
# collected source files.
#
# CMake automatically detects the language of each file by extension:
#   .cpp  → compiled with the CXX compiler (g++/clang++)
#   .hip  → compiled with the HIP compiler (hipcc/clang with ROCm headers)
#
# Headers are listed for IDE visibility only; CMake does not compile them.
# ==============================================================================
add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
    ${PROJECT_HEADERS}
)

# ==============================================================================
# Include directories.
#
# target_include_directories() adds -I flags so the compiler can find headers.
#
# PRIVATE means these paths apply only to this target, not to any library that
# links against it (relevant if you later refactor into a library + executable).
#
# CMAKE_CURRENT_SOURCE_DIR (src/) lets translation units include sibling
# headers with paths relative to src/, e.g.:
#   #include "utils/timer.hpp"    // resolves to src/utils/timer.hpp
# ==============================================================================
target_include_directories(${PROJECT_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

# ==============================================================================
# Compiler flags.
#
# target_compile_options() appends extra flags on top of what CMake injects.
# Generator expressions ($<$<COMPILE_LANGUAGE:...>:...>) ensure that each
# set of flags is applied ONLY to the appropriate language:
#
#   HIP files get:
#     --offload-arch=<gfx>  – target GPU ISA(s), built in root CMakeLists.txt
#     -ffast-math           – allow reassociation / reciprocal optimisations
#                             (safe for most GPU kernels; remove if you need
#                             strict IEEE 754 compliance)
#
#   C++ files get:
#     -Wall -Wextra         – broad diagnostic warnings on host code
# ==============================================================================
target_compile_options(${PROJECT_NAME}
    PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:
            ${HIP_ARCH_FLAGS}
            -ffast-math
        >
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wall
            -Wextra
        >
)

# ==============================================================================
# Link libraries.
#
# target_link_libraries() declares the runtime dependencies of the executable.
#
#   hip::host        – links libamdhip64.so and provides the HIP runtime API
#                      (hipMalloc, hipMemcpy, hipLaunchKernelGGL, etc.) on the
#                      host side. This is all you need for a single-binary HIP
#                      project; hip::device is handled internally by the HIP
#                      language toolchain during device-code linking.
#
#                    The project has no other external dependencies; command-line
#                    parsing uses POSIX getopt_long().
#
# Add further ROCm libraries here as your project grows, e.g.:
#   roc::rocblas     – rocBLAS (BLAS on GPU)
#   roc::rocsolver   – rocSOLVER (LAPACK on GPU)
# ==============================================================================
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        hip::host
)

# ==============================================================================
# Output directory.
#
# No override is needed here. The root CMakeLists.txt sets:
#   CMAKE_RUNTIME_OUTPUT_DIRECTORY = ${CMAKE_BINARY_DIR}
# which expands to <project_root>/build/ when cmake is invoked as:
#   cmake -S . -B build
# All targets in the project inherit this automatically.
# ==============================================================================