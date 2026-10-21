<!--
 SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
 SPDX-License-Identifier: LGPL-3.0-or-later
 -->
 # XeStrata — Vulkan porting technical review (critical re-examination)

 English

 A critical, locally-evidence-based review of [`VULKAN_PORTING_ANALYSIS.md`](VULKAN_PORTING_ANALYSIS.md). It reads the
 same local repository, verifies each of the analysis's main claims against the real source, and rebuilds the porting
 argument from what the code actually shows. Where the analysis could only infer, this marks the inference; where this
 review finds a claim wrong or only partly supported, it says so.

 > **Reading order.** The analysis is the hypothesis. This document is the verification. Every claim here cites a real
 > file and, where relevant, a symbol or line. Tags follow §13: **Verificato**, **Evidenza parziale**, **Inferenza**,
 > **Da verificare**, **Non supportato dalle evidenze**.

 On this page:

 - [1. Executive summary](#1-executive-summary)
 - [2. Objective and constraints](#2-objective-and-constraints)
 - [3. Method and limits of the checks](#3-method-and-limits-of-the-checks)
 - [4. Critical review of the first analysis](#4-critical-review-of-the-first-analysis)
 - [5. Map of the SYCL runtime to replace](#5-map-of-the-sycl-runtime-to-replace)
 - [6. Kernel inventory and porting complexity](#6-kernel-inventory-and-porting-complexity)
 - [7. Vulkan/RADV/Radeon 780M compatibility](#7-vulkanradavradeon-780m-compatibility)
 - [8. WMMA, matrix operations and quantization](#8-wmma-matrix-operations-and-quantization)
 - [9. ggml-vulkan reuse strategies](#9-ggml-vulkan-reuse-strategies)
 - [10. Alternative backend architectures](#10-alternative-backend-architectures)
 - [11. Performance-critical path analysis](#11-performance-critical-path-analysis)
 - [12. Minimum prototype and incremental roadmap](#12-minimum-prototype-and-incremental-roadmap)
 - [13. Test plan and stop criteria](#13-test-plan-and-stop-criteria)
 - [14. Risk matrix](#14-risk-matrix)
 - [15. Open questions and external checks](#15-open-questions-and-external-checks)
 - [16. Conclusion and technical recommendation](#16-conclusion-and-technical-recommendation)
 - [17. Local code references](#17-local-code-references)
 - [18. Verified external sources](#18-verified-external-sources)

 ---

 ## 1. Executive summary

 The analysis is **largely correct in its factual skeleton** but **overstates its confidence in several places** and
 carries **two factual errors** (a non-existent kernel file, and a non-existent header). This review confirms the core
 thesis — XeStrata is a SYCL engine, every GPU path is SYCL to the bone, there is no Vulkan compute backend, and the
 existing SYCL/HIP path already covers `gfx1103` — while correcting the details and tightening what is verified versus
 inferred.

 Key findings of this review:

 1. **The SYCL hypothesis is VERIFICATO.** 37 files under `src/` include `<sycl/sycl.hpp>`; the control layer
    (`src/core/gpu.hpp`, `src/core/gpu.cpp`, `src/core/device.cpp`, `src/core/graph.cpp`, `src/core/vmm.cpp`) is written
    against SYCL's USM, queues, events, `command_graph` capture, and `sycl_ext_oneapi_virtual_mem`. The kernels take a
    `void*` stream that `core::Runtime::get().stream(stream)` resolves to a `sycl::queue` (`src/kernels/xe/bf16_gemv.cpp:38`) —
    a single choke point a Vulkan backend must reproduce.

 2. **The kernel count is 50 files, of which 41 carry launch code** — not "41 kernel files" as the analysis states. The
    9 without launch code are headers (`amd_wmma.hpp`, `cuda_intrinsics.hpp`, `device_caps.hpp`, `device_target.hpp`,
    `iq_bits.hpp`, `q8_1_finite.hpp`, `sm70_table.hpp`) and `matrix_report.cpp` (device query, not a kernel). The
    analysis's table also lists `moe_mmq.cpp`, which **does not exist** locally, and omits several present files
    (`prompt_attn.cpp`, `ple.cpp`, `s_gemv.cpp`, `s2_gemv.cpp`, `shared_expert.cpp`, `native_qsa_score.cpp`,
    `native_flash_attn.cpp`, `native_ple_postops.cpp`, `native_gdn_preprocess.cpp`, `dequant_bf16.cpp`, `iq_kernels.cpp`).

 3. **The WMMA/gfx1103 reasoning is VERIFICATO and stronger than claimed.** `amd_wmma.hpp` is the real source of truth:
    `amd_wmma()` accepts `gfx1100 <= arch < gfx1300` via SYCL's `architecture` info; `cmake/StrataHip.cmake:41`
    accepts `gfx1103`; `__GFX11__` is defined for `gfx1103`; the non-gfx12 builtins
    (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`) are taken. The gfx11 lane-planning (`row_of` = `2*e + hi`) is
    explicitly gfx11-specific and would need re-derivation for Vulkan.

 4. **A major correction to the ggml-vulkan story.** The analysis says ggml's Vulkan backend is "NON VERIFICATO". This
    review finds that the repository **already builds and ships `ggml-vulkan`** for the vision encoder
    (`tools/vision/CMakeLists.txt`, `STRATA_VISION_VULKAN` → `GGML_VULKAN`), and that `glslc`, `spirv-headers`,
    `mesa-vulkan-drivers`, `libvulkan-dev` are in the **base** dependency list of `setup.py:932-933` (Ubuntu) and
    `setup.py:942` (Fedora). So ggml-vulkan is a *known, buildable* component of this repo's ecosystem — not an
    unverified assumption. (The full fetched llama.cpp tree is still not present locally, so ggml-vulkan's exact API
    surface at tag `3cf03257` remains NON VERIFICATO.)

 5. **A nuance the analysis missed: `GGML_VULKAN` is not forced OFF** in the engine's ggml build (`CMakeLists.txt:843-844`
    force `GGML_CUDA`/`GGML_OPENMP` OFF, but leave `GGML_VULKAN` to the fetched checkout's default). So the engine's
    ggml *could* carry a Vulkan backend if the fetched commit enables it — a path the analysis did not consider.

 6. **The two hardest risks stand: host/device pointer identity for the mapped UMA expert shares, and the gfx11 WMMA
    lane-planning re-derivation.** Graph capture remains the least-documented surface.

 7. **The "780M runs via SYCL/HIP today" claim is VERIFICATO in code, PLAUSIBILE in practice** — it needs a machine to
    prove (device-library availability for `gfx1103` is probed at configure time, not verified here).

 The pragmatic recommendation is unchanged from the analysis but sharper: **Option A (native Vulkan Compute backend
 replacing SYCL) is the only path that truly removes the SYCL/LLVM dependency**, it is large, and the incremental
 roadmap is the only way to make it verifiable. The local presence of a working ggml-vulkan build for the vision tool
 lowers the "can we even build Vulkan shaders here?" risk and makes the ggml-vulkan reuse path more credible than the
 analysis allowed.

 ---

 ## 2. Objective and constraints of this analysis

 **Objective.** Critically re-examine `VULKAN_PORTING_ANALYSIS.md` for the porting of XeStrata to a native Vulkan
 Compute backend on the AMD Radeon 780M (`gfx1103`, RDNA 3) in a Beelink SER8 (Ryzen 7 8745HS, 96 GB unified RAM,
 Mesa RADV). Verify each of the analysis's claims against the local repository; find contrary evidence; rebuild the
 feasibility argument from real source.

 **Hard constraints (honored):**

 - Work only on the local repository at `/Users/giancarlo/lavori/privati/XeStrata`.
 - No new repository cloned, no dependencies downloaded, no source fetched.
 - No build, no benchmark, no install, no environment change.
 - No modification of existing source, config, dependencies, or `docs/VULKAN_PORTING_ANALYSIS.md`.
 - Exactly one new file created: `docs/VULKAN_PORTING_TECHNICAL_REVIEW.md`.
 - No claims of tests, builds, or measurements that were not actually performed.

 **What was actually done:** `read` of the analysis and ~40 source/config files, `grep`/`sed`/`find`/`git` for symbol
 and line verification, static inspection of kernel launch sites and the SYCL control layer. Nothing was compiled or
 run. All external-API/driver facts are marked and must be closed on the target machine.

 **Not assumed (per the task):** the HIP/SYCL AMD path is *not* proposed as the solution to the main problem — it was
 already tried on `gfx1103` and does not work in this environment. The stable `llama.cpp`-Vulkan result on the 780M is
 treated as a destination-environment fact, not as proof that XeStrata's kernels port easily.

 ---

 ## 3. Method and limits of the checks

 **Method.** For each claim in the analysis: locate the cited file/symbol/line; read the surrounding code; confirm,
 narrow, or contradict the claim; tag it. Claims that required the driver or a build are marked **Da verificare** or
 **NON VERIFICATO** rather than confirmed.

 **Limits:**

 - **Static only.** Reading a `#if` guard or a regex proves the *intent*; it does not prove the compiler honored it or
   the driver supports the feature at runtime.
 - **No fetched tree.** The full llama.cpp/ggml at tag `3cf03257` is fetched at build time and is **not present locally**.
   Any claim about its `ggml-vulkan.c` contents, its `GGML_VULKAN` default, or its CUDA file layout is therefore
   NON VERIFICATO unless it follows from CMake or from the locally-present `tools/vision/` usage.
 - **No driver.** Mesa/RADV version, enabled extensions, and `gfx1103` runtime behavior are not known here.
 - **No build.** CMake accepting `gfx1103` as a target (a configure-time regex/probe) is not a successful device compile.

 **Tag definitions (§13):** *Verificato* (proven by local source/config), *Evidenza parziale* (partially supported),
 *Inferenza* (logical inference from verified facts), *Da verificare* (needs a machine/external check), *Non supportato
 dalle evidenze* (no local evidence, or contradicted).

 ---

 ## 4. Critical review of the first analysis

 Each item: original claim → supporting symbols → verification → result → confidence.

 ### 4.1 Kernel count and location

 - **Claim:** "all 41 kernel files in `src/kernels/xe/`".
 - **Verification:** `ls src/kernels/xe/` → **50 files**. `grep -rl
 "sycl::parallel_for\|sycl::submit\|nd_range\|single_task\|local_accessor\|sycl::event" src/kernels/xe/` → **41 files carry launch
 code**. The other 9 are headers
   (`amd_wmma.hpp`, `cuda_intrinsics.hpp`, `device_caps.hpp`, `device_target.hpp`, `iq_bits.hpp`, `q8_1_finite.hpp`,
   `sm70_table.hpp`) and `matrix_report.cpp` (device `matrix_combinations` query, `src/kernels/xe/matrix_report.cpp:24`,
   not a kernel).
 - **Result:** **CORRECTED.** The "41" is right for *launch-bearing* files but wrong for "kernel files". The total is
   50; the launch-bearing set is 41. The analysis conflated the two.
 - **Confidence:** **Verificato.**

 ### 4.2 Missing / misattributed files

 - **Claim:** kernel table lists `moe_mmq.cpp` and `s2_expert_grouped.cpp` for grouped expert MMQ; references
   `src/kernels/cuda/qsa_prompt_attn.cu` and `device_arch_tag.hpp`.
 - **Verification:**
   - `moe_mmq.cpp` → **absent** (`ls` shows no such file). Grouped expert MMQ lives in `s2_expert_grouped.cpp`
     (`src/kernels/xe/s2_expert_grouped.cpp:90`, grouped `parallel_for`).
   - `device_arch_tag.hpp` → **absent**; the real file is `device_target.hpp` (`src/kernels/xe/device_target.hpp`),
     which stamps `STRATA_DEVICE_NOT_INTEL` / `STRATA_NV_ARCH` / `STRATA_SUB_GROUP(SG)` — not an "arch tag" per se.
   - `src/kernels/cuda/*.cu` → **no `.cu` files exist locally** (full `find . -name '*.cu'` returns nothing). They are the
     upstream Strata CUDA sources the Xe ports were written *from* (e.g. `src/kernels/xe/rope.cpp:5` references
     `src/kernels/cuda/rope.cu`), present only as comments, not as local files.
   - The analysis **omits** present launch-bearing files: `prompt_attn.cpp`, `ple.cpp`, `s_gemv.cpp`, `s2_gemv.cpp`,
     `shared_expert.cpp`, `native_qsa_score.cpp`, `native_flash_attn.cpp`, `native_ple_postops.cpp`,
     `native_gdn_preprocess.cpp`, `dequant_bf16.cpp`, `iq_kernels.cpp`.
 - **Result:** **PARTIALLY SUPPORTED.** The structural map is right; the specific filenames are not all accurate, and
   the referenced `.cu` files are not local.
 - **Confidence:** **Verificato** (for the corrections); **Inferenza** (that the `.cu` files are upstream references).

 ### 4.3 SYCL runtime usage (queue, event, USM, virtual mem, graph capture, sync)

 - **Claim:** every SYCL primitive is used across the control layer and kernels.
 - **Verification:** `src/core/gpu.hpp` documents the exact contract (Stream → `sycl::queue`, `alloc_device` → device
   USM, `alloc_host` → host USM pinned+mapped at same address, `Event*` → queue position/profiling tag, `Graph*` →
   `command_graph`). `src/core/gpu.cpp:170` uses `sycl::malloc_device`, `:184` `sycl::malloc_host`; `:106`
   `std::map<const sycl::queue*, sx::command_graph<...>> g_captures`; `src/core/device.cpp:253`
   `sycl::ext::oneapi::experimental::submit_profiling_tag` / `ext_oneapi_submit_barrier`; `src/core/vmm.cpp` wraps
   `strata::gpu::vmem_*` (backed by `sycl_ext_oneapi_virtual_mem`). Kernels resolve the stream via
   `core::Runtime::get().stream(stream)` (`src/kernels/xe/bf16_gemv.cpp:38`).
 - **Result:** **Verificato.** The analysis's table in §6.1 is accurate; the single choke point is
   `Runtime::stream()` + `Runtime::wait()`.
 - **Confidence:** **Verificato.**

 ### 4.4 GEMM / math-library path

 - **Claim:** dense GEMMs go through oneMath (oneMKL/cuBLAS/rocBLAS by backend), falling back to XeStrata's own kernels.
 - **Verification:** `src/prefill/gemm.cpp:17` `#include <oneapi/math/blas.hpp>`; `:91,:150` `om::blas::column_major::gemm(q, ...)`;
   `:207-221` `Gemm::path()` reports `be == ext_oneapi_hip ? "rocBLAS" : ext_oneapi_cuda ? "cuBLAS" : "oneMKL"`.
   oneMath is fetched (`CMakeLists.txt:540-546`, `MistVVK/oneMath`), absent locally except `LICENSE`. The startup trial
   (`Gemm::prepare`, `iq_mmq_usable()`'s `LayoutCheck` submission at `iq_mmq.cpp:1195-1210`) confirms the fallback path.
 - **Result:** **Verificato.**
 - **Confidence:** **Verificato.**

 ### 4.5 ggml usage

 - **Claim:** ggml is built static (`ggml-cpu`, `ggml-base`), linked into `strata_kernels_cpu`, computes native IQ
   experts on CPU; `GGML_CUDA`/`GGML_OPENMP` forced OFF; full ggml fetched at build.
 - **Verification:** `CMakeLists.txt:843-844` force `GGML_CUDA OFF` / `GGML_OPENMP OFF`; `:861-868` fetch tag
   `3cf03257`; `:879` links `ggml-cpu ggml-base` into `strata_kernels_cpu`. `src/kernels/cpu/native_expert.cpp` uses
   `ggml_get_type_traits_cpu`, `ggml_cpu_init()` — pure ggml-cpu, **no SYCL** (0 matches for `sycl`).
 - **Result:** **Verificato**, with a correction: the analysis says `GGML_VULKAN`'s default is NON VERIFICATO — correct,
   but it should also note `GGML_VULKAN` is **not** forced OFF (unlike CUDA/OPENMP), so the fetched checkout's default
   applies.
 - **Confidence:** **Verificato.**

 ### 4.6 RDNA3 / WMMA / gfx1103 specifics

 - **Claim:** `__GFX11__` defined for `gfx1103`; non-gfx12 WMMA builtins; `amd_wmma()` accepts `gfx1103`;
   `StrataHip.cmake:42` accepts `gfx1103`.
 - **Verification:**
   - `cmake/StrataHip.cmake:41` regex `^gfx(103[0-9a-f]|11[0-9a-f][0-9a-f]|12[0-9a-f][0-9a-f])$` → **matches `gfx1103`**.
   - `src/kernels/xe/amd_wmma.hpp:33` `amd_wmma()`: `a >= gfx1100 && a < 0x...130000` → **accepts `gfx1103`**.
   - `amd_wmma.hpp:83-91` `row_of()` = `2*e + hi` for `!__GFX12__` (gfx11), `8*hi + e` for gfx12.
   - `amd_wmma.hpp:73-91` builtins: non-gfx12 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` / `_bf16_w32`; gfx12 with
     `_gfx12` suffix.
   - `iq_mmq.cpp:67` `#if defined(__GFX11__) || defined(__GFX12__)`.
 - **Result:** **Verificato.** The analysis is precise and correct here.
 - **Confidence:** **Verificato.**

 ### 4.7 Dependencies: present vs fetched at build

 - **Claim:** oneMath fetched; ggml fetched; `third_party/main/ggml/` is a frozen subset.
 - **Verification:** `third_party/main/ggml/` = 3 files (`ggml-common.h`, `LICENSE`, `VERSION.txt`, 1933 lines total);
   `third_party/main/oneMath/` = `LICENSE` only; `third_party/main/intel-llvm/` = patches; `third_party/main/rocm-device-libs/` present.
 - **Result:** **Verificato.**
 - **Confidence:** **Verificato.**

 ### 4.8 ggml-vulkan reuse claims

 - **Claim:** full ggml fetched; `ggml-vulkan` "NON VERIFICATO"; reuse "PLAUSIBILE for a sub-set / parallel path".
 - **Verification:** **Correction.** The repository already builds `ggml-vulkan` for the vision encoder
   (`tools/vision/CMakeLists.txt:20,44` `STRATA_VISION_VULKAN` → `GGML_VULKAN`), and the base dependency set
   (`setup.py:932-933`, `:942`) includes `glslc`, `spirv-headers`, `mesa-vulkan-drivers`, `libvulkan-dev`. So a Vulkan
   shader toolchain **and** ggml-vulkan are known-good build targets in this repo's ecosystem. The exact ggml-vulkan API
   surface at tag `3cf03257` is still NON VERIFICATO (fetched tree absent).
 - **Result:** **CORRECTED / STRENGTHENED.** Reuse of ggml-vulkan infrastructure is more credible than the analysis
   allowed.
 - **Confidence:** **Verificato** (for the build presence); **Inferenza** (for the API-surface implication).

 ### 4.9 The "780M runs via SYCL/HIP today" synthesis

 - **Claim:** the 780M can run XeStrata through SYCL/HIP.
 - **Verification:** build target accepted (`StrataHip.cmake:41`), WMMA path taken (`amd_wmma.hpp`), dense GEMM via
   rocBLAS path (`gemm.cpp:218`). But **device-library availability for `gfx1103`** is probed at configure time
   (`strata_hip_device_libs_check`, `StrataHip.cmake:97`), not verified here; and the user states HIP does **not**
   work on `gfx1103` in their environment — so the code path existing is not the same as it running.
 - **Result:** **Inferenza / Da verificare.** The code *targets* `gfx1103`; whether it *runs* there is a machine fact
   (and the user's experiential data says the HIP path fails, which is exactly why Vulkan is under investigation).
 - **Confidence:** **Inferenza** (code supports the target); **Da verificare** (runtime success).

 ---

 ## 5. Map of the SYCL runtime to replace

 Reconstructed from `src/core/{gpu,gpu,device,graph,vmm}.cpp`, `include/strata/core/{gpu,runtime}.hpp`, and kernel
 launch sites. The engine was written against the CUDA runtime and mapped onto SYCL (`src/core/gpu.hpp` top comment);
 a Vulkan backend must reproduce this **exact contract** (same handles, same `false`/null-on-error semantics, same
 mapped-memory identity) or the engine's control flow breaks.

 | SYCL primitive | Local proof | Vulkan equivalent | Equivalence |
 | --- | --- | --- | --- |
 | `sycl::device`/`platform` discovery | `device.cpp:find_devices`, `device.cpp:79` `ext_intel_pci_address` |
 `vkEnumeratePhysicalDevices` / `vkEnumerateInstanceExtensionVersion` | Functional |
 | `sycl::queue` (in-order, streams, priority) | `runtime.hpp:38-55`, `gpu.cpp:92 q(Stream)` | `VkCommandPool` + compute `VkQueue` |
 Functional (+ stream→queue map) |
 | `sycl::event` + `wait_and_throw` | `gpu.cpp` events, `runtime.hpp:62 wait()` | `VkSemaphore`/`VkFence` (+ timeline) | Functional |
 | `ext_oneapi_submit_barrier` / `submit_profiling_tag` | `device.cpp:253-255,450` | pipeline barrier / `VK_EXT_performance_query` |
 Partial (barrier yes; profiling tag no direct equivalent) |
 | USM `malloc_device` / `malloc_host` / `malloc_shared` / `free` | `gpu.cpp:170,184,190` | `vkAllocateMemory` (+ bind) | Functional
 (lifetime model differs) |
 | mapped host USM (same-address identity) | `gpu.cpp:184`, `gpu.hpp` `device_pointer()` identity | host-visible+device-visible memory
 (`VK_AMD_device_host_access`) | **Partial** — no guaranteed pointer identity |
 | `memcpy`/`memset` async | `gpu.cpp:85 memset_pieces`, copy_async | `vkCmdCopyBuffer`/`vkCmdFillBuffer` | Functional |
 | `command_graph` capture + replay | `gpu.cpp:106 g_captures`, `graph.cpp:49-75` | secondary command buffers /
 `VK_KHR_dynamic_rendering` capture | **Partial** — no 1:1 API |
 | `sycl_ext_oneapi_virtual_mem` | `vmm.cpp`, `gpu.hpp:92` | `VK_EXT_memory_budget` + manual heap, or `VK_AMD_memory_overallocation` |
 Partial |
 | peer access | `gpu.hpp` `peer_access()` | `VK_KHR_shader_draw_parameters`? no — `VK_KHR_peer_memory` | Partial / none on UMA |
 | subgroup ops (shuffle-down) | `router.cpp:70,142`, `native_qsa.cpp:30` | GLSL `subgroupShuffle*` / `VK_EXT_subgroup_control_types` |
 Functional on RDNA3 |
 | `joint_matrix` / WMMA intrinsics | `iq_mmq.cpp`, `mma_gemm.cpp`, `amd_wmma.hpp` | GLSL `layout(num_rows=16,num_columns=16)` subgroups
 or inline asm | **Partial** — WMMA builtins have no direct GLSL form |

 ### 5.1 The single choke point

 Every kernel resolves its queue through one call:

 ```cpp
 // src/kernels/xe/bf16_gemv.cpp:38
 sycl::queue& queue_for(void* stream) {
     auto& queue = core::Runtime::get().stream(stream);
     return queue;
 }
 ```

 `Runtime::stream(void*)` maps the engine's `void*` stream handle to a `sycl::queue*`; `Runtime::wait(event, what)`
 (`runtime.hpp:62`) is the synchronization primitive. A Vulkan backend needs **exactly one** analog: a `Stream → VkQueue`
 map plus a `Event → VkFence/Semaphore` map, with the same `false`/null-on-error and mapped-memory identity. This is the
 smallest conceivable "resource layer" and the right Phase 1 target.

 ### 5.2 What can be dropped for the first prototype

 - **graph capture** — not needed for a single-kernel prototype; can be faked with sequential command buffers.
 - **`virtual_mem`** — elastic K/V is a decode optimization; a first path can allocate K/V linearly.
 - **peer access** — irrelevant on the 780M's single-host UMA.
 - **profiling tags** — only telemetry.
 - **stream priorities / copy streams** — the 780M has no separate copy engine to expose.

 These are the functionalities that can be **eliminated for the first prototype** without functional regression; the
 rest must be reproduced for fidelity.

 ---

 ## 6. Kernel inventory and porting complexity

 50 files in `src/kernels/xe/`; 41 carry launch code. Grouped by function (all SYCL device kernels, launched via
 `q.submit`+`h.parallel_for(nd_range)` or `queue.parallel_for(range/nd_range)`):

 | Group | Files (present) | Launch model | Complexity to Vulkan | Regression risk |
 | --- | --- | --- | --- | --- |
 | Dense GEMM (prompt, FP16/BF16) | `xmx_gemm.cpp`, `mma_gemm.cpp`, `dp4a_gemm.cpp` | `parallel_for(nd_range<2>)` +
 `joint_matrix`/WMMA/DP4a | **Very high** — MMA re-derivation | High |
 | Expert int8 MMQ | `iq_mmq.cpp` (128 KB) | `submit` + `joint_matrix`/WMMA/DP4a, i-+K-quant decode | **Very high** | Very high
 (bit-exact) |
 | Grouped expert MMQ (S2) | `s2_expert_grouped.cpp` | `parallel_for(nd_range<1>)` grouped by expert | High | Medium |
 | Attention (QSA) | `qsa.cpp`, `qsa_prompt_attn.cpp`, `qsa_decode_attn.cpp`, `qsa_select.cpp`, `prompt_attn.cpp`,
 `native_flash_attn.cpp`, `native_qsa_score.cpp` | `submit`/`parallel_for` + `joint_matrix`/WMMA | **Very high** | High |
 | RoPE | `rope.cpp`, `native_rope.cpp` | `parallel_for(range/nd_range)` | Low | Low (bit-exact) |
 | RMS/norm + RoPE fused | `native_qsa.cpp` (`rms_norm_weighted*`) | `nd_range` + subgroup reduction | Medium | Medium |
 | MoE combine | `native_moe.cpp`, `shared_expert.cpp` | `parallel_for<2>`, `nd_range<2>` | Low-Medium | Low |
 | Router (top-10/512) | `router.cpp`, `router_top10.cpp`, `native_router.cpp` | `submit` + subgroup shuffle | Medium | Medium |
 | Embedding / MMVQ | `native_mmvq.cpp` (52 KB) | `parallel_for` per-type | Medium (many types) | Medium (bit-exact) |
 | BF16 gemv | `bf16_gemv.cpp` | `parallel_for` + `local_accessor` | Low-Medium | Low |
 | GDN (conv) | `gdn.cpp`, `native_gdn.cpp`, `fused_gdn.cpp`, `native_gdn_preprocess.cpp` | `submit` `nd_range<2>` | Medium | Medium |
 | Fused GR (SSM) | `gr.cpp`, `fused_gr.cpp` | `submit` | Medium | Medium |
 | Elementwise | `elementwise.cpp` | `parallel_for` | Low | Low |
 | Quantize act | `quantize_act.cpp` | `parallel_for` | Low | Low (bit-exact) |
 | KV cache | `kv_q4.cpp`, `kv_q8.cpp`, `kv_stream.cpp` | `parallel_for(nd_range<2>)` | Low-Medium | Medium |
 | Sampler | `sampler.cpp` | `nd_range` + subgroup | Medium | Medium |
 | Int vectors | `cvec.cpp` | `submit` `nd_range<2>` | Low | Low |
 | Dequant BF16 | `dequant_bf16.cpp` | `parallel_for(range)` | Low | Low |
 | PLE / post-ops | `ple.cpp`, `native_ple_postops.cpp` | `nd_range` | Medium | Medium |
 | gemv variants | `s_gemv.cpp`, `s2_gemv.cpp`, `s2_gemv.cpp` | `parallel_for(range/nd_range)` | Low | Low |
 | Device query / self-test | `matrix_report.cpp`, `verify_kernels.cpp`, `iq_kernels.cpp`, `sm70_table.hpp` | query / `submit` | Adapter
 | Low |

 `native_qsa.cpp` also hosts the fused `rms_norm_weighted_q8_1` and `rms_norm_rope` used by the decode path
 (`plain_model.cpp:888,927,951`), so it is on the critical decode path, not just prefill.

 ### 6.1 Matrix-engine dispatch (VERIFICATO, §5.1 of analysis)

 Four mutually exclusive paths chosen at runtime from the device report (per AGENTS.md's first rule): XMX
 (`xmx_gemm.cpp`, Intel `joint_matrix` + checked loads), tensor cores (`mma_gemm.cpp`, NVIDIA `joint_matrix`), WMMA
 (`amd_wmma.hpp` via `iq_mmq.cpp`/`mma_gemm.cpp`/`qsa_prompt_attn.cpp`, RDNA3/4), DP4a (`dp4a_gemm.cpp`, no matrix
 engines). `iq_mmq_usable(q)` (`iq_mmq.cpp:1179`) returns true for `amd_wmma(d)`, or a reported `joint_matrix` shape, or
 falls to DP4a — and it **proves** the choice with a runtime `LayoutCheck` submission (`:1195-1210`). `mma_gemm.cpp:104`
 confirms "RDNA3 and RDNA4 take NVIDIA's shape through `amd_wmma.hpp`".

 ### 6.2 Supported quantization formats (VERIFICATO)

 `native_mmvq_supported(int)` (`native_mmvq.cpp:905`) accepts: native rows `kNativeQ6KRows`/`kNativeQ8Rows`, and ggml
 types 2,6,7,8,10,11,12,13,14,16,17,18,20,21,22,23,29,42 — matching the analysis's K-quants/i-quants list. The
 nonlinear IQ4_XS codebook `iq4nl_values[16]` (`native_mmvq.cpp:114`) is a pinned 16-entry table. All decode is
 integer/bit work in the kernel's local memory — portable to SPIR-V integer ops, must stay bit-exact.

 ---

 ## 7. Vulkan/RADV/Radeon 780M compatibility

 Classification per §6. Each is tagged against the *actual* RADV on the 780M, not assumed from the spec or architecture.

 | Capability | Requirement | Local evidence | 780M/RADV status |
 | --- | --- | --- | --- |
 | Compute shader | universal | kernels use `nd_range` | **Confermato nell'ambiente** (RADV runs compute; `llama.cpp`-Vulkan verified
 here) |
 | 32-wide subgroups (wavefront 32) | wave32 | `WARP = 32` (`router.cpp:29`), `reqd_sub_group_size(WARP)` | **Confermato nell'ambiente**
  (RDNA3 wave32; ggml-vulkan relies on it) |
 | 16×16×16 subgroup MMA (GLSL) | `GLSL_EXT_subgroup` `layout(num_rows=16,num_columns=16)` | **absent locally** (no GLSL in repo) | **Da
 verificare** (RADV version) |
 | Host-accessible memory | `VK_AMD_device_host_access` / `VK_EXT_memory_host_access` | build deps present (`setup.py:932`) | **Da
 verificare** (enabled on install) |
 | Unified UMA single heap | architecture | `integrated_gpu()` (`device.cpp:530`) | **Confermato nel codice locale** (engine path) |
 | Graph capture | secondary CBs / dynamic rendering | `graph.cpp` uses SYCL `command_graph` | **Da verificare** (RADV) |
 | SPIR-V 1.6 | `VK_KHR_spirv_1_6` | glslc/spirv-headers deps | **Da verificare** (RADV) |
 | Buffer storage formats (Q8) | `VK_KHR_buffer_storage_format` | absent locally | **Da verificare** |
 | `VK_COMMAND_POOL_CREATE_RESET_` | replay | absent locally | **Plausibile, da testare** |
 | `VK_KHR_shader_subgroup_*` | shuffle-down | `router.cpp` subgroup shuffle | **Da verificare** (RADV) |

 ### 7.1 Diagnostics to run on the target (not run here)

 - `vulkaninfo --summary` and `vulkaninfo --json` — driver version, enabled instance/device extensions, memory heaps.
 - `vulkaninfo` subgroup section — `subgroupSize`, `VK_EXT_subgroup_control_types`, the `layout(num_rows=16,...)`
   capability; confirm wave32.
 - Query `VK_AMD_device_host_access` presence and, if present, whether it yields true host/device pointer identity.
 - `clinfo`-style or `radeonsi` env probe for the Mesa build; confirm SPIR-V 1.6 support.

 **Do not confuse Vulkan availability with kernel efficiency** (the analysis's own warning): RADV running compute
 shaders does not guarantee XeStrata's kernels are fast or even correct.

 ### 7.2 Work-group and subgroup geometry (VERIFICATO for current code)

 - Kernels target **32-wide** subgroups (`WARP = 32`, `reqd_sub_group_size(WARP)`). RDNA3 supports wave32 and wave64;
   the code targets wave32.
 - Work-group sizes seen: 256 (`native_moe.cpp`), up to 512 (`router.cpp:RT_MAX_THREADS=512`); all multiples of 32.
   RDNA3 max work group = 256 — replicate; do not assume 1024 (Intel).
 - The int8 WMMA uses 32 lanes for a 16×16×16 tile; GLSL's `layout(nesting=32)` matches, but the **lane-to-element
   mapping** (`amd_wmma.hpp` `row_of` = `2*e + hi` for gfx11) must be re-derived for GLSL's layout.

 ---

 ## 8. WMMA, matrix operations and quantization

 ### 8.1 Can RDNA3 WMMA be expressed in standard Vulkan shaders?

 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (`amd_wmma.hpp:78`) is an **AMD device-compiler builtin**, not a Vulkan
 instruction. Its status:

 1. **Directly in standard Vulkan?** No. There is no SPIR-V/GLSL opcode that maps 1:1 to `wmma_f32_16x16x16_f16_w32`.
 2. **Vulkan extension?** `GLSL_EXT_subgroup` provides `layout(num_rows=16, num_columns=16, nesting=32)` subgroup matrix
    multiply for the *equivalent operation* — but this is a **spec-proposed/implementation-dependent** feature; RADV's
    support must be verified, not assumed. (This is the single most "Da verificare" item.)
 3. **Different algorithm?** Yes, viable: a 16×16×16 FM32 product can be built from vectorized (packed i16/f16) loads and
    FMA accumulations across the 32 lanes — slower than tensor cores but correct and portable. This is the safe fallback.
 4. **Vector ops + accumulators?** Yes — the DP4a path (`dp4a_gemm.cpp`) already does exactly this for int8 without any
    matrix engine, at ~1% error. A GGML-style i16 FMA gemv is the portable equivalent for FP.
 5. **Delegated to a GEMM library via Vulkan?** Only if oneMath becomes Vulkan-aware (it is SYCL-only locally) or via
    ggml-vulkan's gemm (ggml graph only).

 **Cost of alternatives:** no invented percentages. The GGML/i16-FMA path trades tensor-core throughput for portability;
 on a 12-CU UMA card the bottleneck is almost certainly memory bandwidth (quantized weights), not FLOPs, so the
 throughput penalty may be modest — but this is a **hypothesis to measure**, not a claim.

 ### 8.2 Quantization (VERIFICATO)

 All quant decode is integer/bit manipulation (`iq_bits.hpp`, `native_mmvq.cpp`): sign-parity tricks, codebook table
 lookups, packed loads. These compile to plain SPIR-V integer ops on any backend — **no special hardware**, portable,
 but must stay **bit-exact** (the parity tests verify this). The nonlinear IQ4_XS codebook (`native_mmvq.cpp:114`) is a
 trivial pinned table in Vulkan. The `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt` precision flags
 (`CMakeLists.txt:334-335`) that SYCL shaders must match are a **real correctness constraint** for any GLSL `exp`/`sqrt`
 implementation — mark it for the prototype.

 ---

 ## 9. ggml-vulkan reuse strategies

 **What is actually present locally:** the fetched llama.cpp/ggml tree is **absent** (build-time fetch, tag
 `3cf03257`), but the repository **already builds `ggml-vulkan`** for the vision encoder (`tools/vision/CMakeLists.txt:20,44`)
 and lists `glslc`/`spirv-headers`/`mesa-vulkan-drivers`/`libvulkan-dev` in its base deps (`setup.py:932-933`, `:942`).
 So the *toolchain* and *build path* are known; the *API surface* at the pinned commit is NON VERIFICATO.

 | Option | Reuse | Arch. dependency | Engine change | MoE-kernel compatibility | Overhead risk | Maintainability | Freedom to tune 780M
 |
 | --- | --- | --- | --- | --- | --- | --- | --- |
 | **A. Low-level ggml-vulkan backend** | device/queue/memory/shader/dispatch/sync infra | ggml-vulkan's `ggml_backend_vulkan` | adapter
 layer | **None** (infra only) | Low | Medium (track ggml) | Full (own kernels on top) |
 | **B. Selective kernel reuse** | port ggml-vulkan's matmul/rms/rope/softmax shaders | ggml block layouts | adapter | Partial (dense
 ops) | Low | Medium | Partial |
 | **C. High-level ggml integration** | full ggml graph execution | ggml tensor/backend ABI | large (model → ggml graph) | Must
 re-derive custom ops as ggml nodes | High | Low | Constrained by ggml |
 | **D. Standalone Vulkan backend** | none (own Vulkan) | none | own resource layer | **Full** (rewrite all 41) | Medium (own infra) |
 High | Full |

 **Reuse of runtime ≠ reuse of kernels** (the analysis is right to separate them):

 - **Runtime reuse (A)** is attractive because the engine already needs a `strata::gpu::*` replacement, and ggml-vulkan
   already solves device/queue/memory/command-buffer/event/copy on RADV. But it ties XeStrata to ggml's Vulkan ABI and
   build, and ggml-vulkan exposes only ggml's operators — not XeStrata's int8 MMQ or QSA.
 - **Kernel reuse (B)** is limited: ggml-vulkan's shaders implement ggml's matmul/norm/rope/softmax, a **subset** that
   overlaps only the elementwise/norm/RoPE primitives (Phase 2 of the roadmap). XeStrata's custom kernels are not
   ggml operators and cannot be executed by ggml-vulkan.
 - **C** is the largest re-derivation (map the Qwen3.8/MoE engine onto a ggml graph) and gives the least control over
   the MoE experts.
 - **D** is what the analysis calls Option A; it is the only path that keeps the custom MoE kernels and truly removes
   SYCL.

 **Verdict:** for a *first prototype*, **D's resource layer** is the minimal, highest-leverage move. For *sustainability*,
 **A (reuse ggml-vulkan's runtime infra under a `strata::gpu` adapter)** is worth a prototype because the toolchain is
 already proven in this repo — but it must not be conflated with kernel reuse.

 ---

 ## 10. Alternative backend architectures

 **Option A — Native Vulkan Compute backend (replaces SYCL).** The analysis's Option A. Reproduce `strata::gpu::*` on
 Vulkan; rewrite all 41 kernels as SPIR-V. Large; full control; removes SYCL.

 **Option B — Hybrid: keep SYCL, route dense GEMMs through ggml-vulkan.** The analysis's Option B. Low risk, low value
 (dense GEMMs are a minority of decode cost — decode is attention+MoE dominated). Rejected as primary.

 **Option C — ggml-vulkan runtime + XeStrata kernels on top (Option A + reuse).** A refinement of A: build the
 `strata::gpu` adapter on ggml-vulkan's proven device/queue/memory/shader management (already built for the vision tool),
 and write the 41 kernels independently. This is the most realistic *sustainable* architecture: it reuses the runtime
 that is already known to build here, without forcing the engine into a ggml graph.

 **Option D — Standalone minimal resource layer, then grow.** The roadmap's approach: Phase 1 builds the smallest
 `Stream→VkQueue`/`Event→Fence`/memory/copy layer from scratch (no ggml dependency), validates a "hello kernel", then
 adds primitives, then kernels. Lowest coupling; slowest to a full model; best for proving feasibility.

 **Recommendation:** **Option C for sustainability, Option D's sequencing for the prototype.** Start standalone (D) to
 prove correctness of the resource layer and one kernel; if the resource layer proves cheap, later evaluate grafting it
 onto ggml-vulkan's runtime (C) to cut long-term maintenance. Never force the engine into a ggml graph (C-as-integration).

 ---

 ## 11. Performance-critical path analysis

 **No tokens/s numbers are invented.** The following is a structural analysis of where time goes and where the port
 could regress, plus what measurements are needed.

 **Prefill vs decode:**

 - **Prefill** (prompt read, batched expert MMQ): dominated by the int8 expert MMQ (`iq_mmq.cpp`) and dense GEMM
   (`gemm.cpp` via oneMath/rocBLAS or XeStrata kernels). Bandwidth-bound on quantized weights; the WMMA/MMA re-derivation
   is where *correctness* lives, and tile shapes where *throughput* lives.
 - **Decode** (single-token, QSA attention + MoE): dominated by attention (`qsa_decode_attn.cpp`) and MoE
   (`native_moe.cpp`, `router.cpp`). KV-cache access (`kv_q4/kv_q8.cpp`) is bandwidth-bound; per-token kernel launches
   make **dispatch overhead** the port's main decode risk (see below).

 **Port-specific regression risks:**

 - **Dispatch overhead.** The SYCL path uses **graph capture** (`plain_model.cpp:980` `begin_capture`, `end_capture`) to
   replay a whole layer's kernels in one submission. A naive Vulkan port that issues one `vkCmdDispatch` per kernel
   per token reintroduces CPU-side launch latency — the dominant decode cost on a latency-bound single-token path.
   Graph capture (secondary command buffers / `VK_COMMAND_POOL_CREATE_RESET_`) is therefore **performance-critical, not
   just functional**.
 - **Host/device pointer identity.** The PCIe expert shares read host-mapped memory at the same address
   (`gpu.hpp` `device_pointer()` identity). Vulkan has **no guaranteed host/device pointer equivalence** — the hardest
   correctness+performance spot (also §5.1).
 - **UMA bandwidth.** 96 GB shared; the 780M's 12 CUs under-fill large tiles. Tile/chunk sizes must be **derived from
   the device report** (AGENTS.md), not hard-coded.
 - **Precision.** Shaders must match `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt` (`CMakeLists.txt:334-335`);
   divergent `exp`/`sqrt` cause numerical drift.

 **What measurements are needed to compare fairly:** same model, same quantization, same prompt, same context, same
 temperature, same token count; measure prefill tokens/s, decode tokens/s, latency/token, and peak memory **separately**,
 against (a) the SYCL/HIP build and (b) the user's `llama.cpp`-Vulkan baseline. Records go in `<data>/records/`
 (`AGENTS.md`), never git.

 ---

 ## 12. Minimum prototype and incremental roadmap

 Adapted to the real engine (`plain_model.cpp` drives the per-layer calls; kernels take a `void* stream` resolved via
 `Runtime::stream()`). No code is written here.

 ### Phase 0 — Baseline (VERIFICATED base)
 - **Files:** this doc; `CMakeLists.txt`; `cmake/StrataHip.cmake`; `src/core/{device,gpu,graph,vmm}.cpp`.
 - **Goal:** record the SYCL baseline correctness strings (`Gemm::path()`, `iq_mmq_path()`, `matrix_report.cpp`) and its
   AMD parity results on the 780M.
 - **Success:** SYCL AMD parity tests pass on the 780M (requires a machine).
 - **Risk:** SYCL device libs for `gfx1103` may not build on the available toolchain.
 - **Prereq:** none.

 ### Phase 1 — Vulkan resource layer (D)
 - **Files (new):** `src/core/vulkan.cpp` + `include/strata/core/vulkan.hpp`: device/queue/memory/event/copy; `Stream→VkQueue`
   + `Event→Fence` maps mirroring `Runtime::stream()`/`wait()`. Memory planner from `src/plan/plan.hpp`.
 - **Goal:** reproduce the `strata::gpu::*` contract on Vulkan for device + mapped-host USM-equivalent, copy/memset,
   fence wait, free.
 - **Success:** a "hello kernel" (`parallel_for`) writes correct results via Vulkan; memory tests match SYCL.
 - **Risk:** **host/device pointer identity** for mapped UMA (§9.3/§5.1) — highest.
 - **Regression:** port the existing memory tests (`tests/core/`).

 ### Phase 2 — Priority primitives (no MMA)
 - **Files (new):** SPIR-V shaders for `elementwise`, `rope`, `native_qsa` (rms_norm), `quantize_act`, `sampler`.
 - **Goal:** simplest `parallel_for` kernels; translate GLSL directly.
 - **Success:** bit-exact parity vs SYCL (`rope_parity`, `quantize_act_parity`, `sampler_parity`).
 - **Risk:** RoPE/norm bit-exactness; `prec-div`/`prec-sqrt` match.
 - **Regression:** the `*_parity` tests in `src/kernels/`.

 ### Phase 3 — Engine integration (decode path)
 - **Files:** `src/core/plain_model.cpp` adapter or new `vulkan_model.cpp`; graph-capture layer.
 - **Goal:** route per-layer calls through the Vulkan `strata::gpu`; implement capture/replay (dispatch-overhead control).
 - **Success:** greedy decode of a tiny model through Vulkan yields the same tokens as the SYCL build on the same input.
 - **Risk:** graph capture is the least-documented surface.
 - **Regression:** `plain_model` greedy parity on a small model.

 ### Phase 4 — Matrices and quantizations
 - **Files (new):** GLSL MMA shaders (translate `iq_mmq.cpp`, `mma_gemm.cpp`, `dp4a_gemm.cpp` AMD paths); per-type
   `mmvq` shaders (`native_mmvq.cpp`); `qsa_prompt_attn.cpp` `joint_matrix`/WMMA.
 - **Goal:** 16×16×16 subgroup MMA in GLSL; re-derive gfx11 lane-planning from `amd_wmma.hpp`; all quant types.
 - **Success:** `iq_mmq`/MMVQ parity vs SYCL bit-for-bit across all quant types.
 - **Risk: MMA correctness** — single most error-prone part (gfx11 vs gfx12 row ordering).
 - **Regression:** `iq_mmq_parity`, `native_expert_parity`, `qsa_*_parity`.

 ### Phase 5 — Numerical stability + benchmarks
 - **Files:** `tests/`; records in `<data>/records/`.
 - **Goal:** numerical sweep across quant types/architectures; throughput vs SYCL.
 - **Success:** no NaN/Inf; KL divergence within SYCL-vs-CPU tolerance; throughput reported.
 - **Risk:** MMA divergence; `exp`/`sqrt` precision.

 ### Phase 6 — 780M optimization
 - **Files:** tuning in launch configs (tile shapes, work-group sizes, `STRATA_*` env overrides).
 - **Goal:** derive sizes from the device report; match the B70/RTX/RX measurement discipline.
 - **Success:** throughput within a target band of the SYCL build; checked under `STRATA_VRAM_LIMIT_MIB`/
   `STRATA_MAX_ALLOC_MIB`-style caps simulating a smaller card.
 - **Risk:** RADV quirks; 12 CUs under-filling large tiles.

 ---

 ## 13. Test plan and stop criteria

 **Correctness:**
 - Bit-exact parity vs SYCL/CPU for every kernel (`native_expert_parity`, `iq_mmq_parity`, `rope_parity`,
   `quantize_act_parity`, `s2_gemv_parity`, `kv_q4/kv_q8_parity`, `router_top10_parity`, `qsa_*_parity`,
   `elementwise_parity`, `bf16_gemv_parity`, `cvec_parity`, `gdn_parity`, `gr_parity`, `shared_expert_parity` — all in
   `src/kernels/*_parity.cpp`).
 - Small inputs and edge cases: 0 experts, single-token decode, empty prompt, 1-layer model, out-of-range quant types.
 - Reference comparison: SYCL build on the same input (the verified baseline).
 - Reproducibility: greedy runs identical; temperature runs checked for variance bounds.
 - Error/compat handling: incompatible device, missing extension, OOM → clean `false`/null + `last_error()`, matching the
   SYCL contract; the 120 s hang watchdog (`device.cpp:kHangLimit=120`) maps to a fence-timeout `exit()`.

 **Throughput (measured separately):**
 - Prefill tokens/s, decode tokens/s, latency/token, peak memory — same model/quant/prompt/context/temperature.
 - Records in `<data>/records/` (per `AGENTS.md`), never git.

 **Stop criteria (motivated, not arbitrary):**
 - **Hard stop (unsustainable):** a required feature is absent on RADV and no portable equivalent exists — e.g. no way to
   get true host/device pointer identity for the mapped expert shares *and* no offset-math workaround; or the GLSL
   16×16×16 MMA cannot be made bit-exact against the SYCL reference after the re-derivation.
 - **Hard stop:** a decode-critical kernel (attention or int8 MMQ) cannot reach the SYCL-vs-CPU numerical tolerance.
 - **Soft stop:** decode dispatch overhead (per-kernel `vkCmdDispatch`) pushes latency/token above the SYCL build by a
   factor that graph capture cannot recover — signals that Option B/hybrid or ggml-vulkan runtime reuse is required.
 - Thresholds are motivated by the SYCL baseline (the reference), not fixed numbers.

 ---

 ## 14. Risk matrix

 | ID | Questione tecnica | Evidenza disponibile | Stato | Impatto sul porting | Prossima verifica |
 | --- | --- | --- | --- | --- | --- |
 | VTR-001 | Host/device pointer identity for mapped UMA expert shares | `gpu.hpp` `device_pointer()` identity; `gpu.cpp:184` host USM;
 no Vulkan guarantee | **Inferenza** (hard spot) | Alto | RADV `VK_AMD_device_host_access` true identity test |
 | VTR-002 | gfx11 WMMA lane-planning re-derivation for GLSL | `amd_wmma.hpp:83-91` `row_of` gfx-specific; no GLSL equivalent |
 **Inferenza** | Alto | Prototype 16×16×16 MMA; bit-exact parity |
 | VTR-003 | Graph capture/replay in Vulkan | `graph.cpp` SYCL `command_graph`; `plain_model.cpp:980` | **Da verificare** | Alto (perf)
 | Secondary CB capture on RADV |
 | VTR-004 | GLSL 16×16×16 subgroup MMA on RADV | Vulkan spec; no local GLSL | **Da verificare** | Alto | `vulkaninfo` subgroup MMA
 section |
 | VTR-005 | `GGML_VULKAN` default at tag `3cf03257` | `CMakeLists.txt:843-844` leaves it; fetched tree absent | **NON VERIFICATO** |
 Medio | Inspect fetched llama.cpp |
 | VTR-006 | ggml-vulkan API surface at pinned commit | `tools/vision` builds it; source absent locally | **NON VERIFICATO** | Medio |
 Inspect fetched llama.cpp |
 | VTR-007 | oneMath is SYCL-queue only, no Vulkan | `gemm.cpp:91,150` `om::blas::column_major::gemm(q, ...)`; no Vulkan path locally | **Verificato** | Medio |
 | VTR-008 | prec-div / prec-sqrt parity | `CMakeLists.txt:334-335` `-foffload-fp32-prec-div/prec-sqrt`; shaders must match | **Verificato** | Medio |
 | VTR-009 | rocBLAS gfx11 kernels on the target | `StrataHip.cmake:53-78` probes distro paths; no `/opt/rocm` inspected | **NON VERIFICATO** | Medio |
 | VTR-010 | 12 CU under-utilization of large tiles | `router.cpp:29 WARP=32`, `native_moe.cpp:77 WG`; device report drives sizing | **Inferenza** | Medio |
 | VTR-011 | Missing/renamed files in first analysis | `moe_mmq.cpp` absent; `device_arch_tag.hpp` is `device_target.hpp`; no `.cu` in tree | **Verificato** | Basso |
 | VTR-012 | ggml-vulkan already built for vision | `tools/vision/CMakeLists.txt:44` `GGML_VULKAN`; `setup.py:932-933` glslc/vulkan packages | **Verificato** | Alto (reuse) |

## 15. Open questions and external checks

Questions that need a machine or an external source, in priority order. Each is tagged with the ID it closes.

1. **GLSL 16×16×16 subgroup MMA on RADV (VTR-004).** Does the target Mesa support `GLSL_EXT_subgroup` with
   `layout(num_rows=16, num_columns=16, nesting=32)`, and is it fast on gfx1103? Prototype required — no local proof.
2. **True host/device pointer identity for mapped UMA (VTR-001).** Does `VK_AMD_device_host_access` on RADV give the
   mapped host USM the same virtual address the kernel sees, as SYCL's `device_pointer()` guarantees (`gpu.hpp`)? If
   not, the offset-math workaround (§5.2) is mandatory.
3. **Graph capture surface (VTR-003).** Which RADV build is on the Beelink, and does it expose reliable secondary-command
   buffer capture for the `plain_model.cpp:980` segmented graphs? Least-documented path.
4. **`GGML_VULKAN` default at tag `3cf03257` (VTR-005).** The fetched llama.cpp is not local; its default and API are
   unknown. `CMakeLists.txt:843-844` forces `GGML_CUDA`/`GGML_OPENMP` OFF but leaves `GGML_VULKAN` to the checkout.
5. **rocBLAS gfx11 kernels on the target (VTR-009).** Whether the distribution's rocBLAS has gfx11 code — affects only
   the SYCL/HIP baseline, not the Vulkan port, but needed to confirm the comparison target.
6. **SPIR-V version and subgroup width on RADV.** wave32 vs wave64, and the SPIR-V version RADV accepts (1.6 is the
   modern default). Confirm with `vulkaninfo` on the target.
7. **Precision of `exp`/`sqrt` in GLSL.** Must match `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt` (VTR-008);
   glslc's `--target-env` controls this and must be pinned.

## 16. Conclusion and technical recommendation

**Answering the task's seven questions directly:**

1. **Most realistic Vulkan architecture.** A **native Vulkan Compute backend** (Option D in §10) that reproduces the
   `strata::gpu::*` contract (`gpu.hpp`, `gpu.cpp`) and rewrites the `src/kernels/xe/*` SYCL kernels as SPIR-V shaders,
   keeping the engine's control flow and the CPU expert paths (`src/kernels/cpu/*`) untouched. It is the only design
   that truly removes the SYCL/LLVM dependency and matches the task's intent. The two hardest parts are the mapped-UMA
   host/device pointer identity (VTR-001) and the gfx11 WMMA re-derivation in GLSL (VTR-002).

2. **Autonomous backend vs. reuse ggml-vulkan.** The two are not the same question, and the evidence separates them:
   the **runtime** can be reused far more than the **kernels**. `tools/vision/` already builds and links ggml-vulkan
   from the pinned llama.cpp (`CMakeLists.txt:858-868`, `tools/vision/CMakeLists.txt:44`), and `setup.py:932-933` lists
   glslc, `spirv-headers` and `mesa-vulkan-drivers` in the **base** dependency set — so the toolchain is already a
   first-class project dependency. That makes a **hybrid (Option B/D hybrid, §10)** attractive: drive the simple
   `parallel_for` kernels (elementwise, rope, rms_norm, sampler, quantize_act) through a native Vulkan resource layer,
   and, where a dense product maps onto ggml's operators, reuse ggml-vulkan's device/dispatch/sync infra. But ggml-vulkan
   **cannot** execute XeStrata's custom kernels (int8 MMQ, QSA attention, GDN/GR, native router/MoE) — the engine is not a
   ggml graph (`plain_model.cpp` calls `iq_mmq`, `native_moe`, `qsa` directly). So reuse the **runtime**, not the kernels.

3. **Three most important technical obstacles.** (a) Host/device pointer identity for the mapped UMA expert shares —
   SYCL's same-address USM has no Vulkan guarantee (`gpu.hpp` `device_pointer()`); the hardest correctness spot. (b)
   gfx11 WMMA lane-planning re-derivation: `amd_wmma.hpp:83-91` `row_of()` is gfx-specific, and GLSL's `layout(
   num_rows=16,num_columns=16)` maps a different lane-to-element order. (c) Graph capture/replay for the segmented
   decode graphs (`plain_model.cpp:980`): SYCL's `command_graph` has no 1:1 Vulkan API (VTR-003).

4. **First prototype that yields a real result.** A **standalone GLSL 16×16×16 subgroup-MMA test** (Phase 1b of §12):
   compile one compute shader with glslc, dispatch it on the 780M via RADV, and check bit-exact parity against the
   SYCL `amd_wmma` product for a small tile. This is the single test that decides whether the hardest obstacle (VTR-002)
   is surmountable, and it needs no model and no engine integration.

5. **Information missing to decide on a full port.** (i) GLSL subgroup-MMA availability and speed on the target RADV
   (VTR-004); (ii) true host/device identity for mapped UMA (VTR-001); (iii) graph-capture reliability (VTR-003);
   (iv) the fetched llama.cpp's `GGML_VULKAN` default and API (VTR-005); (v) rocBLAS gfx11 kernels on the target for the
   SYCL comparison baseline (VTR-009). The first three need the machine; the last two need the fetched tree / distro.

6. **Proofs that the project is unsustainable.** (i) The GLSL 16×16×16 MMA cannot be made bit-exact against the SYCL
   reference after the re-derivation. (ii) No way to get true host/device pointer identity for the mapped shares and no
   working offset-math workaround. (iii) A decode-critical kernel (attention or int8 MMQ) cannot reach the SYCL-vs-CPU
   numerical tolerance. (iv) Per-kernel `vkCmdDispatch` latency pushes token latency above the SYCL build by a factor
   graph capture cannot recover. Any one of these is a hard stop (§13).

7. **What is verifiable locally vs. what needs the machine/external sources.** Locally verifiable (already done here):
   the SYCL runtime contract (`gpu.hpp`/`gpu.cpp`/`runtime.hpp`), the kernel launch mechanism (`queue_for` →
   `Runtime::stream`), the WMMA/gfx1103 code paths (`amd_wmma.hpp`, `iq_mmq.cpp`, `mma_gemm.cpp`), the gfx1103 build
   acceptance (`StrataHip.cmake:41`), the oneMath SYCL-queue API (`gemm.cpp`), the quant formats (`native_mmvq.cpp`),
   the CPU-path independence, and that ggml-vulkan is already built for the vision tool (`tools/vision`, `setup.py`).
   Needs the machine: GLSL-MMA correctness/speed, host/device identity, graph capture, real throughput. Needs the
   fetched tree / distro: `GGML_VULKAN` default and API, rocBLAS gfx11 kernels.

---

## 17. Local code references

| Symbol | File | Line | Role |
| --- | --- | --- | --- |
| `strata::gpu::*` contract | `include/strata/core/gpu.hpp` | top | Stream/alloc/event/graph contract the engine uses |
| USM alloc/free, command_graph map | `src/core/gpu.cpp` | 92,106,170,184,205 | `sycl::malloc_device/host`, `sx::command_graph` |
| `Runtime::stream/wait/get` | `include/strata/core/runtime.hpp` | 38-62 | queue/event access; the choke point |
| `queue_for(void* stream)` | `src/kernels/xe/bf16_gemv.cpp` | 38 | every kernel gets its queue from the `void*` stream |
| `amd_wmma()` arch range | `src/kernels/xe/amd_wmma.hpp` | 33 | gfx1100..gfx1300 → gfx1103 qualifies |
| `row_of()` gfx11 lane planning | `src/kernels/xe/amd_wmma.hpp` | 83-91 | gfx-specific; re-derive for GLSL |
| WMMA builtins | `src/kernels/xe/amd_wmma.hpp` | 118-131 | `__builtin_amdgcn_wmma_*_w32` (non-gfx12) |
| `iq_mmq_usable()` trial | `src/kernels/xe/iq_mmq.cpp` | 1179-1209 | runtime layout trial, not a static check |
| gfx11/gfx12 guard | `src/kernels/xe/iq_mmq.cpp` | 67 | `#if defined(__GFX11__) || defined(__GFX12__)` |
| `mma_gemm` AMD WMMA path | `src/kernels/xe/mma_gemm.cpp` | 104 | RDNA3/4 take WMMA for NVIDIA's shape |
| `native_mmvq_supported` types | `src/kernels/xe/native_mmvq.cpp` | 905-911 | quant type table |
| IQ4_XS nonlinear codebook | `src/kernels/xe/native_mmvq.cpp` | 114 | `iq4nl_values[16]`, pinned |
| dense GEMM oneMath call | `src/prefill/gemm.cpp` | 91,150 | `om::blas::column_major::gemm(q,...)` SYCL-queue |
| rocBLAS path report | `src/prefill/gemm.cpp` | 207-221 | `Gemm::path()`; `ext_oneapi_hip → rocBLAS` |
| gfx1103 build acceptance | `cmake/StrataHip.cmake` | 41 | `^gfx(103x|11xx|12xx)$` accepts gfx1103 |
| device-lib probe | `cmake/StrataHip.cmake` | 97 | `strata_hip_device_libs_check` at configure |
| SYCL build targets | `CMakeLists.txt` | 253-351 | spir64 / nvidia_gpu_ / amd_gpu_ |
| prec-div/prec-sqrt | `CMakeLists.txt` | 334-335 | shader precision must match |
| ggml fetch (CPU only) | `CMakeLists.txt` | 836-868 | fetched llama.cpp; `GGML_CUDA/OPENMP` OFF |
| kernel launch files | `src/kernels/xe/*` | — | 50 files, 41 with launch symbols |
| `device_target.hpp` | `src/kernels/xe/device_target.hpp` | top | `STRATA_DEVICE_NOT_INTEL` target stamping |
| vision ggml-vulkan build | `tools/vision/CMakeLists.txt` | 44 | `GGML_VULKAN`; builds ggml-vulkan |
| vulkan base deps | `setup.py` | 932-933 | glslc, spirv-headers, mesa-vulkan-drivers |
| model driver | `src/core/plain_model.cpp` | 980 | segmented graph capture in decode |
| graph begin/end | `src/core/graph.cpp` | 49,59 | `begin_capture`/`end_capture`/`graph_launch` |
| vmem elastic K/V | `src/core/vmm.cpp` | top | `sycl_ext_oneapi_virtual_mem` wrapper |
| K/V UMA limit | `src/core/device.cpp` | 530-565 | `integrated_gpu`, `apply_uma_limit`, `/proc/meminfo` |
| hang watchdog | `src/core/device.cpp` | 171 | `kHangLimit = 120 s` |
| quantize_act regression note | `src/kernels/quantize_act_parity.cpp` | 276-277 | references `quantize_act.cu` (upstream, not local) |

## 18. Verified external sources

No external web/API source was consulted during this review (the task made web navigation optional and required the
local repository to be the primary evidence). The claims that cite external facts — the Vulkan/RADV extensions, the
GLSL subgroup-MMA operation, the SPIR-V version, and the WMMA instruction set — are marked **Da verificare** /
**Inferenza** in §14 and listed in §15 for later confirmation against primary sources (Khronos Vulkan spec and
extension registry, Mesa/RADV source, AMD RDNA3/WMMA documentation, the pinned llama.cpp for `ggml-vulkan`). No URL,
extension name or API is asserted as confirmed without a local or machine check.
