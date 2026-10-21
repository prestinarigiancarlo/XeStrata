# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# docs/PROMPT_FOR_LLM_macos_HARDWARE_CHECKLIST.md - verification checklist for the Beelink SER8 (gfx1103) + RADV.
#
# Every item below is DA VERIFICARE SU HARDWARE on macOS.  Run these on the Beelink SER8 with RADV installed.  For
# each: exact command, expected output, go/no-go criterion.  Do NOT fill this in from macOS assumptions.

## 1. Driver & device enumeration
- Command: `vulkaninfo --summary`
- Expected: a device named "Radeon RX 780M" / "RADV gfx1103", apiVersion >= 1.3.
- Go: device listed; No-go: absent or RADV not present.

## 2. Subgroup matrix support (the core of the port)
- Command: `vulkaninfo --summary | grep -i -A2 subgroup`  and check `VK_EXT_subgroup_matrix` in the extensions list.
- Expected: `VK_EXT_subgroup_matrix` advertised; subgroupSize 32; SubgroupSupportedOps includes matrix multiply.
- Go: extension present, subgroup layout num_rows=16/num_columns=16/nesting=32 supported (query
  `SubgroupMatrixPropertiesEXT`).  No-go: extension missing -> wmma_subgroup.comp cannot run, only the FMA fallback.

## 3. VK_AMD_device_host_access (mapped memory visibility)
- Command: `vulkaninfo --summary | grep -i device_host_access`
- Expected: `VK_AMD_device_host_access` advertised (allows device access to host-mapped memory).
- Go: present.  No-go: absent -> the mapped-host readback pattern must use explicit transfer (staging buffers).

## 4. Host/device pointer identity
- Command: allocate a mapped buffer, write on host, read via a kernel; compare.  (The harness's RADV stub path
  skips this on macOS.)
- Expected: device sees the host write without a transfer IF device_host_access is present.
- Go: match without staging.  No-go: mismatch -> use the handle->(device,host) table from api/vulkan_gpu.hpp Option B.

## 5. Graph capture (decode path, plain_model.cpp:980)
- Command: build a secondary command buffer with pipeline bind + dispatch, record, execute, wait; compare to
  equivalent immediate submits.
- Expected: identical output.
- Go: capture reproduces immediate results.  No-go: no 1:1 SYCL equivalent -> implement capture in api/vulkan_gpu.hpp
  Option A (secondary buffers) or fall back to immediate submit.

## 6. Bit-exact parity vs the SYCL kernel
- Command: run the SYCL `amd_wmma` kernel and the GLSL `wmma_subgroup.comp` on the same 16x16x16 tile; compare with
  scripts/compare.py --dtype float.
- Expected: 0 mismatches, max_ulp=0.
- Go: exact.  No-go: investigate rounding order (WMMA accumulates in a fixed order vs the FMA loop).

## 7. Performance (deferred from macOS)
- Command: time prefill/decode token/s, latency, VRAM via `radeontop` / `vulkaninfo`.
- Expected: numbers recorded (not asserted here).
- Go/No-go: recorded in records/ (LOCAL.md), not in git.
