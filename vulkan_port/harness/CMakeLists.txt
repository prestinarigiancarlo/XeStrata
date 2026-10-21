# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# harness/CMakeLists.txt - build the minimal Vulkan harness on macOS.
cmake_minimum_required(VERSION 3.20)
project(xestrata_vulkan_harness CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan: LunarG SDK on macOS provides the loader + headers.  Point CMAKE_PREFIX_PATH at the SDK install.
find_package(Vulkan REQUIRED)

add_executable(vulkan_harness main.cpp)
target_include_directories(vulkan_harness PRIVATE ../reference)
target_link_libraries(vulkan_harness PRIVATE Vulkan::Vulkan)

# Tests: smoke test on lavapipe and MoltenVK (see run_smoke.sh).
enable_testing()
add_test(NAME smoke_lavapipe COMMAND vulkan_harness)
set_tests_properties(smoke_lavapipe PROPERTIES
    ENVIRONMENT "STRATA_VULKAN_ICD=lavapipe")
add_test(NAME smoke_moltenvk COMMAND vulkan_harness)
set_tests_properties(smoke_moltenvk PROPERTIES
    ENVIRONMENT "STRATA_VULKAN_ICD=moltenvk")
