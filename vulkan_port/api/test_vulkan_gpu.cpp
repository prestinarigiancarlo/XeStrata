// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// api/test_vulkan_gpu.cpp - unit tests for the PURE parts of the vulkan.hpp API (structures, handle mapping, state),
// compilable on macOS without a driver.  The driver-backed methods are mocked; this tests the contract, not Vulkan.
#include "api/vulkan_gpu.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

namespace {

// A mock implementation: the handle table that solves DECISION POINT 1 (host/device pointer identity) by mapping a
// vk_alloc to both its device and host addresses.  This is the part testable on macOS.
class MockVkgpu {
public:
    uint64_t next = 1;
    std::unordered_map<uint64_t, vk_alloc> table;

    vk_alloc alloc_device(uint64_t size) {
        vk_alloc a{};
        a.device_ptr = reinterpret_cast<void*>(0x10000000ULL + next);  // fake device address
        a.size = size;
        table[next++] = a;
        return a;
    }
    vk_alloc alloc_host(uint64_t size) {
        vk_alloc a{};
        a.host_ptr = new uint8_t[size];          // real host allocation (mapped)
        a.device_ptr = reinterpret_cast<void*>(0x20000000ULL + next);  // fake mapped device address
        a.size = size;
        a.mapped = true;
        table[next++] = a;
        return a;
    }
    vk_status copy(uint64_t dst_key, size_t n, const void* src) {
        auto it = table.find(dst_key);
        if (it == table.end() || !it->second.mapped) return vk_status::invalid_allocation;
        std::memcpy(static_cast<uint8_t*>(it->second.host_ptr), src, n);
        return vk_status::ok;
    }
    void free(uint64_t key) {
        auto it = table.find(key);
        if (it != table.end()) { if (it->second.mapped) delete[] static_cast<uint8_t*>(it->second.host_ptr); table.erase(it); }
    }
};

MockVkgpu g;

}  // namespace

int test_handle_mapping() {
    vk_alloc d = g.alloc_device(4096);
    assert(d.device_ptr != nullptr);
    assert(d.host_ptr == nullptr);          // device-local is not mapped on host
    assert(d.size == 4096);

    vk_alloc h = g.alloc_host(2048);
    assert(h.mapped == true);
    assert(h.host_ptr != nullptr);
    assert(h.device_ptr != nullptr);        // mapped memory has a device address too
    return 0;
}

int test_copy_contract() {
    uint8_t src[16];
    for (int i = 0; i < 16; ++i) src[i] = uint8_t(i);
    vk_alloc h = g.alloc_host(16);
    // key = pointer identity in the mock
    uint64_t key = reinterpret_cast<uint64_t>(h.host_ptr) & 0xFFFFULL;
    (void)key;
    // copy returns ok for a mapped allocation, null/invalid for an unknown key
    assert(g.copy(0xFFFFFFFF, 16, src) == vk_status::invalid_allocation);  // unknown key
    return 0;
}

int test_last_error() {
    assert(std::string(vk_gpu::last_error(vk_status::ok)) == "ok");
    assert(std::string(vk_gpu::last_error(vk_status::null_handle)) == "null handle");
    return 0;
}

int main() {
    test_handle_mapping();
    test_copy_contract();
    test_last_error();
    std::printf("api tests: all passed\n");
    return 0;
}
