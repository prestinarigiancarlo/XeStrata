// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// api/vulkan_gpu.hpp - public API design that replicates the contract of strata::gpu::* for the Vulkan backend.
//
// strata::gpu::* (src/core/gpu.cpp, device.cpp, graph.cpp, vmm.cpp) exposes: stream, event, alloc_device, alloc_host,
// copy, memset, wait, graph capture, with error = false/null + last_error().  This header defines the SAME contract
// over Vulkan handles.  It is a design document + compilable unit-test surface; the driver-backed methods are stubbed
// so the pure parts compile and test on macOS.
//
// DECISION POINTS that require a physical-machine decision (DA VERIFICARE SU HARDWARE on the Beelink SER8):
//   1. host/device pointer identity:  SYCL maps a host pointer to its device counterpart by pointer identity.  Vulkan
//      has no such guarantee; mapped host memory (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) is a separate address.  The
//      VMM must keep a handle->(device_ptr, host_ptr) table.  -> see Option B below.
//   2. semaphore/fence format:  SYCL events map to Vulkan fences (per-queue) or semaphores (cross-queue).  On RADV
//      cross-queue sync needs VK_KHR_semaphore or the core 1.3 semaphore.  -> see Option C.
//   3. graph capture:  decode's plain_model.cpp:980 uses SYCL graph capture; Vulkan has pipeline-binding capture only
//      via VK_EXT_command_buffer_rendering + secondary buffers, no 1:1 equivalent.  -> see Option A.
enum class vk_status {
    ok = 0,
    null_handle = -1,
    invalid_allocation = -2,
    driver_error = -3,
};
inline const char* last_error(vk_status s) {
    switch (s) {
        case vk_status::ok: return "ok";
        case vk_status::null_handle: return "null handle";
        case vk_status::invalid_allocation: return "invalid allocation";
        case vk_status::driver_error: return "driver error";
    }
    return "unknown";
}

struct vk_stream { void* handle = nullptr; };         // wraps a Vulkan queue family + fence pool
struct vk_event  { void* handle = nullptr; };          // wraps a Vulkan fence or semaphore
struct vk_alloc  { void* device_ptr = nullptr;
                   void* host_ptr = nullptr;         // if mapped
                   VkDeviceSize size = 0;
                   bool mapped = false; };

class vk_gpu {
public:
    // Lifetime.  device_name is the physical device name (for the RADV stub check).
    explicit vk_gpu(const char* device_name);
    const char* device_name() const;

    // Streams / events (1 over 3: a stream is a queue, an event a fence/semaphore on it).
    vk_stream  stream_create();
    void       stream_destroy(vk_stream);
    vk_event   event_create();
    void       event_destroy(vk_event);

    // Allocation: device-local (alloc_device) or host-visible mapped (alloc_host).  Returns null on failure.
    vk_alloc   alloc_device(VkDeviceSize size);
    vk_alloc   alloc_host(VkDeviceSize size);
    void       free(vk_alloc);

    // Memory ops.  copy/memset take stream + event deps; return null on bad args.
    vk_status  copy(vk_stream, vk_event* deps, size_t n, const void* src_host, vk_alloc dst);
    vk_status  memset(vk_stream, vk_event* deps, vk_alloc, int value, size_t n);
    void       wait(vk_stream, vk_event);

    // Graph capture (decode path; DA VERIFICARE SU HARDWARE - no 1:1 SYCL equivalent).
    struct graph;
    graph*     graph_create(vk_stream);
    void       graph_begin(graph*);
    void       graph_end(graph*);
    void       graph_submit(graph*, vk_event);

private:
    struct Impl;
    Impl* impl_;
};
