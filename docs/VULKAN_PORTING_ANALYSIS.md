<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata - Vulkan Compute porting analysis

English | [日本語](VULKAN_PORTING_ANALYSIS.ja.md)

A technical, locally-verified analysis of what a Vulkan Compute backend for XeStrata would require, with particular
attention to the AMD Radeon 780M (`gfx1103`, RDNA 3) on a Beelink SER8. Everything below is read from the local
repository copy; where a claim cannot be checked from the code, it is marked **NON VERIFICATO** or **PLAUSIBILE**.

> **On this page:** [1. Executive summary](#1-executive-summary) · [2. Local copy analyzed](#2-local-copy-analyzed) ·
> [3. Effective architecture](#3-effective-architecture) · [4. GPU dependency inventory](#4-gpu-dependency-inventory) ·
> [5. Kernel and operation map](#5-kernel-and-operation-map) · [6. SYCL/LLVM dependencies](#6-syclllvm-dependencies) ·
> [7. ggml-vulkan reuse](#7-ggml-vulkan-reuse) · [8. Proposed Vulkan architecture](#8-proposed-vulkan-architecture) ·
> [9. Radeon 780M analysis](#9-radeon-780m-analysis) · [10. Implementation roadmap](#10-implementation-roadmap) ·
> [11. File-by-file intervention matrix](#11-file-by-file-intervention-matrix) · [12. Test and benchmark plan](#12-test-and-benchmark-plan) ·
> [13. Risks, limits and alternatives](#13-risks-limits-and-alternatives) · [14. Evidence matrix and open questions](#14-evidence-matrix-and-open-questions) ·
> [15. Conclusion](#15-conclusion)

---

## 1. Executive summary

XeStrata is a **SYCL-based** inference engine. Every GPU path — device discovery, memory, queues, events, graph
capture, and all 41 kernel files in `src/kernels/xe/` — is written against the SYCL runtime on top of Level Zero, CUDA
or HIP. There is **no Vulkan Compute backend** in the local copy, and none of the SYCL calls can be reached without
the SYCL runtime.

The key finding, verified in the code, is that **the existing SYCL kernels already run on AMD RDNA 3/4 GPUs through
intel/llvm's HIP target**. The int8 expert products (`iq_mmq.cpp`) and the prompt-attention products use AMD's WMMA
instructions via `__builtin_amdgcn_wmma_*_w32` and `*_w32_gfx12`, guarded by `#if defined(__GFX11__) || defined(__GFX12__)`
(`src/kernels/xe/amd_wmma.hpp`, `src/kernels/xe/iq_mmq.cpp:67,838,897`). The `gfx1103` architecture (RDNA 3) falls
inside both `__GFX11__` and the `amd_wmma()` runtime check (`gfx1100 <= arch < gfx1300`), and `cmake/StrataHip.cmake:42`
accepts `gfx1103` as a supported target. **This means the 780M can already run XeStrata — through SYCL/HIP, not through
Vulkan.**

Therefore a "Vulkan port" has two distinct meanings, and the analysis separates them:

1. **Replace the SYCL runtime with a native Vulkan Compute backend** — a large, from-scratch effort: re-implement the
   entire `strata::gpu` control layer (streams, USM-style memory, events, graph capture) on the Vulkan API, and rewrite
   or transpile all 41 kernel files. This is the literal reading of the task.
2. **Reuse `ggml`'s Vulkan backend** (already present in the fetched llama.cpp checkout) for the parts of the workload
   ggml can cover — mainly the dense GEMMs through oneMath's cuBLAS/rocBLAS analogue, or a ggml-based fallback path.
   This is smaller but does not cover XeStrata's custom kernels.

The pragmatic conclusion: **the 780M does not need a Vulkan port to run XeStrata** — the SYCL/HIP path already covers
it. A Vulkan backend is justified only if the goal is to remove the SYCL/LLVM dependency (smaller binary, no ROCm
device libraries, no DPC++ toolchain) or to share code with ggml-based tooling. That goal is achievable but large, and
the incremental roadmap below is the only way to make it verifiable.

Every major conclusion is tagged **VERIFICATO / PLAUSIBILE / SPERIMENTALE / NON VERIFICATO** per §8.

## 2. Local copy analyzed

| Item | Value | Source |
| --- | --- | --- |
| Working directory | `/Users/giancarlo/lavori/privati/XeStrata` | `pwd` |
| Repository root | `/Users/giancarlo/lavori/privati/XeStrata` (has `CMakeLists.txt`, `src/`, `include/`, `third_party/`) | directory inspection |
| Branch | `main` | `git branch` |
| Commit | `180d0776c04de40a76b8da00519bcc45b1d75255` | `git rev-parse HEAD` |
| Upstream integrated | Strata `0.1.45`, commit `a0116b2` (`CMakeLists.txt:3-4`) | `STRATA_UPSTREAM_COMMIT` |
| XeStrata revision | `1` (`XESTRATA_REVISION`) | `CMakeLists.txt:11` |
| Working tree | clean, up to date with `origin/main` | `git status` |

The repository was **not** cloned, re-downloaded, rebuilt or modified. Only local files were read. The `third_party/main/ggml/`
folder holds only a **frozen subset** of ggml (three files: `ggml-common.h`, `LICENSE`, `VERSION.txt`); the full ggml
and llama.cpp are fetched at build time from `https://github.com/ggml-org/llama.cpp.git` at tag
`3cf03257f219afbe7334045ff7c6a06ac68c627d` (`CMakeLists.txt:838-868`), or taken from a local checkout via
`-DSTRATA_GGML_DIR=`. That full checkout is **not present locally**, so any claim about its contents is
**NON VERIFICATO** unless it follows from the CMake configuration.

## 3. Effective architecture

The engine runs an MoE model (Qwen3.8 / Qwen4-Exp, Qwen3.5-MoE, Qwen3-MoE, GLM5-Next — `ModelArch` in
`include/strata/artifact/model_spec.hpp:31`) through a **prefill** phase (prompt read, batched expert MMQ) and a
**decode** phase (single-token, QSA attention + MoE). The control flow is written against the CUDA runtime and mapped
onto SYCL (`include/strata/core/gpu.hpp`, top comment):

- **Stream** → `sycl::queue` (`void*`, like `cudaStream_t`; null = the compute queue)
- **`alloc_device`** → device USM (`sycl::malloc_device`)
- **`alloc_host`** → host USM, pinned and device-accessible at the same address (`cudaHostAlloc(Mapped)`)
- **`Event*`** → a queue position / profiling tag (`sycl::event`, `ext_oneapi_submit_barrier` / `submit_profiling_tag`)
- **`Graph*`** → an executable SYCL command graph (`sycl::ext::oneapi::experimental::command_graph`, stream capture +
  instantiate)

This mapping is the **single integration layer** between the engine and the GPU. Every engine call goes through
`strata::gpu::*` in `src/core/gpu.cpp`, which catches SYCL's synchronous exceptions and forwards asynchronous ones to
the runtime's next wait. A Vulkan backend must reproduce this exact contract (same handles, same `false`/null-on-error
semantics, same mapped-memory identity) or the engine's control flow breaks.

The per-device state lives in `Runtime` (`include/strata/core/runtime.hpp`, `src/core/device.cpp`): one per engine
device, owning a `sycl::context`, a `compute_` and `transfer_` in-order queue, engine streams, and a watchdog thread
that `exit()`s if any wait exceeds `kHangLimit` (120 s). Memory is managed by a `DeviceArena` (USM + manual 4 KiB
alignment) and, on Level Zero/CUDA, `sycl_ext_oneapi_virtual_mem` for elastic K/V (`src/core/vmm.cpp`).

## 4. GPU dependency inventory

### 4.1 The SYCL runtime (VERIFICATO)

72 files under `src/` include `<sycl/sycl.hpp>` or reference SYCL symbols. The device code is compiled for multiple
targets at link time (`CMakeLists.txt:253-351`):

- **`spir64`** — compiled at program start (Intel GPUs); this is the default and what runs without any other target.
- **`nvidia_gpu_<arch>`** / **`nvptx64-nvidia-cuda`** — NVIDIA tensor cores (contrib mode, needs CUDA toolkit).
- **`amd_gpu_<arch>`** — AMD GPUs via intel/llvm's HIP target (free **and** contrib mode, since ROCm is free
  software). `cmake/StrataHip.cmake` detects them from the kernel KFD topology and builds them with ROCm's device
  libraries (`ockl.bc`, etc.).

The SYCL compiler is intel/llvm's DPC++ 7+ (`-fsycl`), **not** Intel oneAPI's `icpx` (refused by CMake). The runtime
is libsycl 9. The engine links **no maker's driver library**; each backend's library is opened at runtime by the
Unified Runtime (`libze_loader.so.1` for Intel, the HIP/CUDA runtimes otherwise).

### 4.2 oneMath for dense GEMMs (VERIFICATO)

`src/prefill/gemm.cpp` runs the prompt's dense matrix products through **oneMath** (`third_party/main/oneMath`,
Apache-2.0, XeStrata's fork) when `STRATA_ONEMATH` is set (contrib mode, and AMD in both modes):

- Intel GPU → oneMKL
- NVIDIA GPU → cuBLAS
- **AMD GPU → rocBLAS** (`gemm.cpp:Gemm::path()`, `be == sycl::backend::ext_oneapi_hip ? "rocBLAS"`)

oneMath is fetched from `https://github.com/MistVVK/oneMath/archive/refs/tags/xestrata-3.tar.gz`
(`CMakeLists.txt:540-546`) — **not present locally**, only its `LICENSE` file. On a GPU oneMath has no backend for, or
when a product fails its startup trial (`Gemm::prepare`), the engine falls back to **XeStrata's own kernels**.

### 4.3 ggml / llama.cpp (VERIFICATO for the build config; NON VERIFICATO for contents)

ggml is built as a static library (`ggml-cpu`, `ggml-base`) and linked only into `strata_kernels_cpu`
(`CMakeLists.txt:879`), which computes the **native IQ experts on the CPU** (`STRATA_NATIVE_EXPERTS`, default ON).
`GGML_CUDA` and `GGML_OPENMP` are forced OFF (`CMakeLists.txt:843-844`). The full ggml/llama.cpp source is fetched at
build time — **not in the local copy**. Whether that checkout's `ggml-vulkan` exists and its build option
`GGML_VULKAN` is ON is **NON VERIFICATO** (the fetched tree is not inspected).

### 4.4 Memory model (VERIFICATO)

- Device memory: **device USM** (`sycl::malloc_device`), 4 KiB aligned by hand (`src/core/device.cpp:DeviceArena`).
- Host memory: **host USM**, pinned and mapped at the same virtual address (`cudaHostAlloc(Mapped)` analogue), so
  `device_pointer` is the identity (`gpu.cpp`). Kernels address it in place for the PCIe expert shares.
- UMA (integrated GPUs): `integrated_gpu()` checks `ext_oneapi_is_integrated_gpu` or HIP's `host_unified_memory`; the
  engine takes the smaller of the runtime's free figure and `/proc/meminfo` MemAvailable less `STRATA_UMA_HEADROOM_GIB`
  (default 6) (`device.cpp:apply_uma_limit`). This is the exact path the 780M takes.
- Per-allocation cap: `max_mem_alloc_size` capped by `STRATA_MAX_ALLOC_MIB`; total VRAM capped by `STRATA_VRAM_LIMIT_MIB`.

### 4.5 Dependencies that would block removing SYCL for Vulkan (VERIFICATO)

Removing SYCL from the Vulkan path means replacing, in one new layer, every SYCL primitive the engine and kernels use:
`sycl::queue`, `sycl::context`, `sycl::event`, USM alloc/free, `parallel_for`/`nd_range`/`submit`/`single_task`,
`local_accessor`, `memcpy`/`memset`, `ext_oneapi_submit_barrier`/`submit_profiling_tag`, stream capture +
`command_graph` replay, `sycl_ext_oneapi_virtual_mem`, and peer access. **All of these are used across the 41 kernel
files.** There is no partial SYCL surface — the kernels are SYCL to the bone.

## 5. Kernel and operation map

All 41 files in `src/kernels/xe/` are SYCL device kernels. The operations, verified from the launch sites:

| Operation | File(s) | SYCL launch | Notes |
| --- | --- | --- | --- |
| Dense GEMM (prompt, FP16/BF16) | `xmx_gemm.cpp`, `mma_gemm.cpp`, `dp4a_gemm.cpp` | `parallel_for`/`submit` + `joint_matrix` or WMMA or DP4a | Matrix-engine dependent (see §6) |
| Expert int8 MMQ | `iq_mmq.cpp` (128 KB) | `submit` + `joint_matrix`/WMMA/DP4a | The big one; i-quant + K-quant decoding |
| Expert grouped MMQ | `moe_mmq.cpp`, `s2_expert_grouped.cpp` | grouped `nd_range` | Layer-split / resident cache |
| Attention (QSA) | `qsa.cpp`, `qsa_prompt_attn.cpp`, `qsa_decode_attn.cpp`, `qsa_select.cpp`, `native_qsa*.cpp` | `submit`/`parallel_for` | Prompt, decode, block-score, select |
| RoPE | `rope.cpp`, `native_rope.cpp` | `parallel_for` | IMRoPE, scaling via `rope_scaling.hpp` |
| RMS norm / layernorm | `native_qsa.cpp` (`rms_norm_weighted*`) | `nd_range` + subgroup reduction | Fused norm+ROPE, norm+quantize |
| MoE combine | `native_moe.cpp` | `parallel_for<2>` | Weighted sum + shared expert |
| Router (top-k) | `router.cpp`, `router_top10.cpp`, `native_router.cpp` | `submit` + subgroup shuffle | ggml-cuda port, top-10/512 |
| Embedding / MMVQ | `native_mmvq.cpp` (52 KB) | `parallel_for` | Per-type dispatch (see §5.2) |
| BF16 gemv | `bf16_gemv.cpp` | `parallel_for`/`submit` + `local_accessor` | FP32 rows |
| GDN (conv) | `gdn.cpp`, `native_gdn.cpp`, `fused_gdn.cpp`, `native_gdn_preprocess.cpp` | `submit` | Conv1D over history |
| Fused GR (SSM) | `gr.cpp`, `fused_gr.cpp` | `submit` | State-space model block |
| Elementwise | `elementwise.cpp` | `parallel_for` | Silu/gate/scale |
| Quantize act | `quantize_act.cpp` | `parallel_for` | FP32 → Q8_1 |
| KV cache | `kv_q4.cpp`, `kv_q8.cpp`, `kv_stream.cpp` | `parallel_for` | Page table append/gather |
| Sampler | `sampler.cpp` | `nd_range` + subgroup | Top-k / temperature / |z| |
| Int vectors | `cvec.cpp` | `parallel_for` | Conversation vectors |
| Matrix report | `matrix_report.cpp` | `parallel_for` | Backend self-report |

### 5.1 Matrix-engine dispatch (VERIFICATO)

The products have **four mutually exclusive paths**, chosen at run time from what the device reports (the first rule in
`AGENTS.md` — never from a vendor string or model number):

- **XMX** (`xmx_gemm.cpp`) — Intel Xe2/Xe-HPG matrix engines, `joint_matrix` with Intel layouts + checked loads.
- **Tensor cores** (`mma_gemm.cpp`) — NVIDIA via `joint_matrix` portable API; tile shape and sub-group size from the
  device's `matrix_combinations`.
- **WMMA** (`amd_wmma.hpp`, used by `iq_mmq.cpp` and `qsa_prompt_attn.cpp`) — RDNA 3/4 (`__GFX11__`/`__GFX12__`),
  `__builtin_amdgcn_wmma_*_w32`.
- **DP4a** (`dp4a_gemm.cpp`, `iq_mmq.cpp` DP4a body) — a GPU without matrix engines; int8 blocks, ~1% error.

`iq_mmq_usable()` (`iq_mmq.cpp:1184`) returns true if `amd_wmma(d)` (RDNA3/4), or the device reports a `joint_matrix`
shape, or falls to DP4a. `qsa_prompt_attn.cpp:489-490` confirms RDNA3/4 take the tiles through `amd_wmma.hpp`.

### 5.2 Supported quantization formats (VERIFICATO)

`native_mmvq_supported()` (`native_mmvq.cpp:905`) and `native_mmvq()` accept these ggml types:

- **K-quants**: Q2_K (10), Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14), Q4_0 (2), Q5_0 (6), Q8_0 (8), Q2_0 (42)
- **i-quants**: IQ4_NL (20), IQ4_XS (23), IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL (16/17/18/21/22/29, via `iq_mmq`)
- **Native row forms**: `kNativeQ6KRows`, `kNativeQ8Rows`
- **BF16** and **FP32** (dense, `plain_model.cpp`)

`iq_mmq_type_ok()` covers the i-quants except IQ1_S, and the K-quants except Q2_0/Q8_0/Q5_0/Q2_K. IQ4_XS uses a pinned
nonlinear codebook (`native_mmvq.cpp:114`). All of these are decoded **on the GPU** in the kernel's local memory.

### 5.3 CPU fallback paths (VERIFICATO)

- **Native IQ experts on CPU** through `ggml-cpu` (`src/kernels/cpu/*`: `expert.cpp`, `native_expert.cpp`,
  `iq_avx2.cpp`, `iq_avx512.cpp`, `canon_expert.cpp`) — the PCIe / prefill shares (`plain_model.cpp:586-588`,
  `plain_prompt.cpp`).
- **CPU pool** for missed experts during prefill (`STRATA_PREFILL_CPU_SHARE`, `src/core/moe_window.cpp`).
- **MTP** draft and helper-GPU expert shares (`src/core/mtp.cpp`, `src/core/remote_experts.cpp`).

These CPU paths are independent of the GPU backend and would be **unchanged** by a Vulkan port.

## 6. SYCL/LLVM dependencies

### 6.1 What must be replaced for a native Vulkan backend (VERIFICATO)

From `src/core/{device.cpp,gpu.cpp,graph.cpp,vmm.cpp}` and the kernel launch sites, a Vulkan backend must provide:

| SYCL primitive | Used in | Vulkan equivalent |
| --- | --- | --- |
| `sycl::device`/`platform` discovery | `device.cpp:find_devices` | `vkEnumeratePhysicalDevices` |
| `sycl::queue` (in-order, streams, priority) | `runtime.hpp`, `gpu.cpp` | `VkCommandPool` + `VkQueue` (compute) |
| `sycl::event` + `wait_and_throw` | everywhere | `VkSemaphore`/`VkFence` |
| `ext_oneapi_submit_barrier` / `submit_profiling_tag` | `device.cpp:measure_marks` | pipeline barrier / `VK_EXT_performance_query` |
| USM `malloc_device`/`malloc_shared`/`free` | `gpu.cpp`, `device.cpp` | `vkAllocateMemory` (+ `VK_AMD_device_host_access` for shared) |
| `memcpy`/`memset` async | `gpu.cpp:copy_async` | `vkCmdCopyBuffer`/`vkCmdFillBuffer` |
| `command_graph` capture + replay | `graph.cpp` | `vkCmdBeginRenderPass`? no — `vkCmdBeginCommandBuffer` capture or `VK_KHR_pipeline_library` |
| `sycl_ext_oneapi_virtual_mem` | `vmm.cpp`, `gpu.cpp` | `VK_EXT_memory_budget` + manual heap mgmt, or `VK_AMD_memory_overallocation` |
| `ext_intel_pci_address`, `ext_intel_free_memory` | `device.cpp` | `VK_AMD_device_config` / vendor extensions |
| subgroup ops (`get_sub_group`, shuffle-down) | `router.cpp`, `qsa*.cpp` | `vkSubgroup*` / `GLSL subgroup` |
| `joint_matrix` / WMMA intrinsics | `iq_mmq.cpp`, `mma_gemm.cpp` | **GLSL/MSL `layout(num_rows=16,num_columns=16)` subgroups** or inline asm |

### 6.2 The device-code problem (VERIFICATO)

The kernels are not plain host C++: they carry device code compiled per-target. `device_arch_tag.hpp` stamps every
source with its target (`spir64`, `sm_*`, `gfx*`). The matrix intrinsics are compiler builtins:

- Intel: `mx::joint_matrix_*` (`sycl::ext::oneapi::experimental::matrix`)
- NVIDIA: same portable `joint_matrix`
- **AMD gfx11**: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` / `_i32_16x16x16_iu8_w32` (non-gfx12) and
  `..._w32_gfx12` (gfx12)

A Vulkan backend would compile these to SPIR-V via the Vulkan shader pipeline (GLSL → SPIR-V, or WGSL). The WMMA
builtins have **no direct GLSL equivalent**; GLSL 4.60 / `GLSL_EXT_subgroup` provides `layout(num_rows=16,
num_columns=16, nesting=32)` matrix multiply for the equivalent operation, but the exact lane-planning in
`amd_wmma.hpp` (row ordering differs between gfx11 and gfx12) would need re-derivation. This is the single most
labor-intensive part.

### 6.3 Dependencies that would prevent a clean SYCL→Vulkan cut (VERIFICATO)

- **oneMath** (`third_party/main/oneMath`, fetched) calls `om::blas::column_major::gemm(sycl::queue, ...)` — it is
  written against SYCL queues, not Vulkan. A Vulkan backend would either need oneMath to be Vulkan-aware (not the case
  locally) or route dense GEMMs through XeStrata's own kernels / ggml-vulkan.
- **ggml-cpu** is CPU-only here; ggml's Vulkan backend (`ggml-vulkan`) is a separate build option in the fetched
  checkout, not linked into XeStrata today.

## 7. ggml-vulkan reuse

### 7.1 What is available locally (VERIFICATO)

- The full ggml/llama.cpp is fetched at build time (`CMakeLists.txt:858-868`) from tag `3cf03257`. Its `ggml/src/`
  contains `ggml-vulkan.c` and the Vulkan backend, gated by `GGML_VULKAN` (ON by default in recent llama.cpp).
- **Not present locally**: the fetched tree, so `GGML_VULKAN`'s default and the backend's exact API surface are
  **NON VERIFICATO**.

### 7.2 What ggml-vulkan can and cannot cover (PLAUSIBILE)

ggml-vulkan implements ggml's graph operators (matmul, bias add, silu, rms_norm, rope, softmax, etc.) for general
inference. XeStrata's workload is **not** a ggml graph: it is a custom Qwen3.8/MoE engine with:

- Custom int8 MMQ with per-32-value scales decoded on-device (`iq_mmq`) — ggml has no equivalent.
- Custom QSA attention with block scoring, page tables, and top-k selection (`qsa*`) — ggml attention is the simple
  variant.
- Custom GDN/GR (SSM) blocks, native router, native MoE combine — none in ggml.
- Custom quantizations (IQ4_XS nonlinear codebook, native Q8_1 rows) — partially in ggml.

So ggml-vulkan could serve **only a hypothetical ggml-based reimplementation** of XeStrata, or the dense GEMMs if one
ran them through ggml. It **cannot** execute XeStrata's existing kernels. Reuse is therefore **PLAUSIBILE for a
sub-set / a parallel path**, not for the existing engine.

### 7.3 Data-structure and API compatibility (NON VERIFICATO)

ggml's `ggml_tensor`, `ggml_backend`, and block layouts (`block`/`type_traits`) are the ABI ggml-vulkan expects.
XeStrata uses its own `TensorInfo`, `gguf_reader`, and native block structs (`IQ4NLBlock`, `IQ4XSBlock`, `Q81Block` in
`native_mmvq.cpp:70-98`). Mapping XeStrata's tensors onto ggml's graph would be a substantial adapter, and the
bit-exact contracts (XeStrata kernels are verified bit-for-bit against ggml's CPU dot, `iq_avx2_parity`, etc.) would
need re-validation. **NON VERIFICATO** on exact ggml version details.

## 8. Proposed Vulkan architecture

Two architectures are possible. Both keep the engine's control flow unchanged by reproducing the `strata::gpu::*`
contract.

### 8.1 Option A — Native Vulkan Compute backend (replaces SYCL)

```
strata::gpu (new: vulkan.cpp)          strata::kernels::xe (rewritten as SPIR-V)
 ├ vkDevice / vkQueue (compute)         ├ gemm: GLSL 16x16x16 subgroup MMA (gfx11 path re-derived)
 ├ vkMemory (host-access UMA for 780M) ├ iq_mmq: GLSL subgroups + codebook decode
 ├ vkFence/Semaphore (events)          ├ qsa/rope/norm/sampler: GLSL parallel_for
 ├ vkCmdCopyBuffer/Fill (copies)       ├ router: GLSL subgroup shuffle
 ├ vkCmdGraph (capture+replay)         ├ mmvq: per-type GLSL
 └ virtual mem (manual heap)           └ elementwise/gdn/gr/quantize: GLSL
```

- **Shader compilation**: GLSL sources → SPIR-V via `glslang` (bundled) at build; or WGSL. A runtime SPIR-V loader
  (`spirv-cross`/`VK_KHR_shader_subgroup_*`) for device selection.
- **Memory**: `vkAllocateMemory` on the shared heap for the 780M's UMA, with `VK_AMD_device_host_access` (or
  `VK_EXT_memory_host_access`) for the mapped-host path. Memory planner mirrors `plan.hpp`.
- **Graph capture**: `VK_KHR_dynamic_rendering` + secondary command buffers, or `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER`
  for replay. This is the least-documented, riskiest part — SYCL's `command_graph` capture has no 1:1 Vulkan API.
- **MMA**: GLSL 4.60 `layout(num_rows=16, num_columns=16, nesting=32) subgroups` matrix instructions (RDNA 3 supports
  these). The gfx11 lane-planning in `amd_wmma.hpp` must be re-derived for GLSL's layout.

### 8.2 Option B — Hybrid: keep SYCL, add ggml-vulkan for dense GEMMs only

Reuse the existing SYCL/HIP path (which already runs on the 780M) for all custom kernels, and route **only the dense
prompt GEMMs** through ggml-vulkan when beneficial. Minimal new surface (one adapter in `gemm.cpp`), but:

- Adds a second runtime (SYCL + Vulkan) — complexity, not savings.
- The dense GEMMs are a minority of the decode cost (decode is attention + MoE dominated).
- **PLAUSIBILE**, low risk, but low value. The roadmap in §10 is Option A.

### 8.3 Recommendation

Option A is the only path that truly "removes SYCL", matching the task's intent. It is large; the roadmap is staged so
each stage is verifiable in isolation. Option B is the fallback if Option A's graph-capture or MMA work proves too
costly.

## 9. Radeon 780M analysis

Target: Beelink SER8, Ryzen 7 8745HS, Radeon 780M (`gfx1103`, RDNA 3, 12 CUs), 96 GB unified RAM, Linux + Mesa RADV.

### 9.1 The SYCL/HIP path already covers the 780M (VERIFICATO)

- **Build target**: `cmake/StrataHip.cmake:42` accepts `gfx1103` (`^gfx(11[0-9a-f][0-9a-f])`). intel/llvm's HIP target
  compiles `amd_gpu_gfx1103`. **Not locally verified**: whether the specific intel/llvm version present builds the
  `gfx1103` device libraries (`ockl.bc` etc.) — `strata_hip_device_libs_check` (`StrataHip.cmake:87`) probes this at
  configure time. **Marked UNVERIFIED for the exact toolchain, VERIFICATO for target acceptance.**
- **WMMA kernels**: `__GFX11__` is defined for `gfx1103` (the `__GFX11__` macro covers `gfx1100`–`gfx1109`). The int8
  and FP16 WMMA paths use the **non-gfx12** builtins (`__builtin_amdgcn_wmma_*_w32`, no `_gfx12` suffix) —
  `iq_mmq.cpp:838-901`, `amd_wmma.hpp:98-105`. So the 780M takes the same code path as other RDNA 3 parts.
- **Runtime check**: `amd_wmma()` (`amd_wmma.hpp:33`) accepts `gfx1100 <= arch < gfx1300`; `gfx1103` qualifies.
- **Dense GEMM**: rocBLAS via oneMath; `strata_hip_rocblas()` (`StrataHip.cmake`) needs rocBLAS packages with `gfx11`
  kernels. On the 780M this depends on the distribution's rocBLAS — **NON VERIFICATO** locally (no `/opt/rocm` or
  distro packages inspected).

**Bottom line: the 780M can run XeStrata today through SYCL/HIP.** This is the single most important fact and it is
verified in the code, not assumed.

### 9.2 Vulkan/RADV capabilities to verify (PLAUSIBILE / NON VERIFICATO)

Assuming Option A, the following must be confirmed on the actual RADV build (not assumed from Vulkan spec):

| Capability | Vulkan requirement | 780M/RADV status |
| --- | --- | --- |
| Compute shader | `VK_KHR_compute_shader` | universal |
| Subgroup 16x16x16 MMA | `GLSL_EXT_subgroup` `layout(num_rows=16,num_columns=16)` | RDNA 3 supports; RADV present — **verify version** |
| Host-accessible memory | `VK_AMD_device_host_access` or `VK_EXT_memory_host_access` | RADV has `VK_AMD_device_host_access` — **verify enabled** |
| Unified UMA | single heap, host-visible + device-visible | 780M: yes, by architecture |
| Graph capture | `VK_KHR_dynamic_rendering` + secondary CBs, or `VK_COMMAND_POOL_CREATE_RESET_` | present in modern RADV — **verify** |
| Buffer storage storage | `VK_KHR_buffer_storage_format` if needed for Q8 | present |
| SPIR-V version | `VK_KHR_spirv_1_4/1_6` | RADV supports SPIR-V 1.6 |
| 32-wide subgroups | wavefront 32 | RDNA 3: wave32 and wave64 |

**Do not confuse Vulkan availability with kernel efficiency.** RADV being present guarantees the driver runs compute
shaders; it does not guarantee XeStrata's kernels will be fast. The gfx11 lane-planning re-derivation (§6.2) is where
correctness lives, and tuning (tile shapes, work-group sizes) is measured on the real card — none of it is verifiable
from the local code.

### 9.3 Memory and limits for the 780M (VERIFICATO for the model, NON VERIFICATO for Vulkan specifics)

- **UMA**: the engine already takes the smaller of the runtime's free figure and `/proc/meminfo` less 6 GiB headroom
  (`device.cpp:apply_uma_limit`). On the 780M this reads the 96 GB shared pool. A Vulkan backend must replicate this:
  `vkGetMemoryBudget` (via `VK_EXT_memory_budget`) against the shared heap, minus headroom.
- **Per-allocation cap**: `max_mem_alloc_size` on integrated GPUs is often ¼ of system RAM; `STRATA_MAX_ALLOC_MIB`
  overrides. Replicate in the Vulkan allocator.
- **Mapped host memory**: the PCIe expert shares rely on host USM being addressable by the kernel at the same address.
  In Vulkan this is a host-visible + device-visible allocation with `VK_AMD_device_host_access`; the "same address"
  identity is harder in Vulkan (no guaranteed host/device pointer equivalence) and needs `VK_EXT_external_memory_host`
  or careful offset math. **This is a known hard spot — SPERIMENTALE.**
- **Synchronization**: the 120 s hang-limit watchdog (`device.cpp:kHangLimit`) must map to Vulkan fence timeouts; a
  stuck fence should still `exit()`.

### 9.4 Subgroup and work-group geometry (VERIFICATO for current code; re-derive for Vulkan)

- Current kernels assume **32-wide subgroups** (warps/wavefronts); `device_caps.hpp` and `narrow_sub_group` pick 16 or
  32. RDNA 3 supports wave32 and wave64; the code targets wave32.
- Work-group sizes seen: 256 (`native_moe.cpp`, `router.cpp:RT_MAX_THREADS=512`), `THREADS` multiples of 32. RDNA 3
  max work group = 256. Replicate; do not assume Vulkan lets you use 1024 (that's Intel).
- The int8 WMMA uses 32 lanes for a 16×16×16 tile. GLSL's `layout(nesting=32)` matches; the lane-to-element mapping in
  `amd_wmma.hpp` (row `2*e + hi` for gfx11) must be re-derived for GLSL's layout.

### 9.5 Integer and quantized ops (VERIFICATO)

All quant decode is integer/bit manipulation in the kernels (`iq_bits.hpp`, `native_mmvq.cpp`): `__byte_perm`-style
packing, sign-parity bit tricks, codebook table lookups. These compile to plain SPIR-V integer ops on any backend —
**no special hardware needed**, they are portable but must be re-checked bit-for-bit. The nonlinear IQ4_XS codebook
(`native_mmvq.cpp:114`) is a pinned 16-entry table — trivial in Vulkan, must stay exact.

### 9.6 Theory vs. driver reality (VERIFICATO framing)

Vulkan spec says X; RADV on gfx1103 may not implement X, or implement it with quirks. Every "Vulkan feature" in §9.2
is marked for verification against the **actual** Mesa/RADV version on the target — not assumed from the spec. The task
explicitly warns against this conflation.

## 10. Implementation roadmap

All future changes are **described, not executed** (per the task). Each stage has acceptance tests and a gate to the
next.

### Phase 0 — Inventory and baseline (VERIFICATO base)

- **Files**: this document; `CMakeLists.txt`; `cmake/StrataHip.cmake`; `src/core/{device,gpu,graph,vmm}.cpp`.
- **Work**: confirm the local SYCL baseline builds and its AMD tests pass on the 780M (requires a machine — **not run
  here**). Record the backend self-report strings (`Gemm::path()`, `iq_mmq_path()`, `matrix_report.cpp`) as the
  correctness baseline.
- **Dependencies**: intel/llvm + ROCm device libs for `gfx1103` (SYCL route), or glslang + RADV (Vulkan route).
- **Risk**: the SYCL baseline itself may not build for `gfx1103` on the available toolchain (device libs).
- **Acceptance**: `strata-device` and the SYCL AMD parity tests pass on the 780M.
- **Gate to Phase 1**: baseline green.

### Phase 1 — Vulkan resource layer

- **Files** (new): `src/core/vulkan.cpp` (+ `include/strata/core/vulkan.hpp`): device/queue/memory/event/copy/graph.
- **Work**: reproduce the `strata::gpu::*` contract on Vulkan. Memory planner from `src/plan/plan.hpp`. UMA +
  host-access for the 780M.
- **Dependencies**: glslang (SPIR-V), RADV with `VK_AMD_device_host_access`.
- **Risk**: host/device pointer identity for mapped memory (§9.3). **Highest risk in the whole project.**
- **Acceptance**: allocate device + host USM-equivalent, copy/memset, fence wait, free — all on the 780M, matching the
  SYCL memory tests.
- **Gate**: a "hello kernel" (`parallel_for`) writes correct results via Vulkan.

### Phase 2 — Priority primitives

- **Files** (new): SPIR-V shaders for `elementwise`, `rope`, `native_qsa` (rms_norm), `quantize_act`, `sampler`.
- **Work**: these are the simplest `parallel_for` kernels; no MMA. Translate the GLSL directly.
- **Dependencies**: phase 1.
- **Risk**: low; bit-exactness of RoPE and norm must match SYCL.
- **Acceptance**: parity tests against the SYCL kernels, bit-for-bit (as `rope_parity`, `quantize_act_parity` do).
- **Gate**: decode of a tiny model runs elementwise+rope+norm+sampler through Vulkan, matches SYCL output exactly.

### Phase 3 — Integration with the inference engine

- **Files**: `src/core/plain_model.cpp` (or a new `vulkan_model.cpp`), `src/core/graph.cpp` adapter.
- **Work**: route the engine's per-layer calls through the Vulkan `strata::gpu` layer; implement graph capture/replay.
- **Dependencies**: phase 2 + graph-capture working.
- **Risk**: graph capture is the least-documented Vulkan surface; SYCL's `command_graph` has no 1:1 API.
- **Acceptance**: a full decode pass on the 780M through Vulkan produces the same tokens as the SYCL build on the same
  input.
- **Gate**: greedy run parity on a small model.

### Phase 4 — Matrices and quantizations

- **Files** (new): GLSL MMA shaders (translate `iq_mmq.cpp`, `mma_gemm.cpp`, `xmx_gemm.cpp` AMD paths), per-type
  `mmvq` shaders (`native_mmvq.cpp`).
- **Work**: the 16×16×16 subgroup MMA in GLSL; re-derive gfx11 lane-planning from `amd_wmma.hpp`. All quant types from
  §5.2.
- **Dependencies**: phase 3.
- **Risk**: **MMA correctness** — the single most error-prone part. The gfx11 vs gfx12 row ordering must be re-derived.
- **Acceptance**: `iq_mmq` and MMVQ parity against the SYCL kernels bit-for-bit across all quant types (as
  `native_expert_parity`, `iq_mmq_parity` do).
- **Gate**: prefill (batched expert MMQ) through Vulkan matches SYCL on real GGUF rows.

### Phase 5 — Numerical stability and benchmarks

- **Files**: tests in `tests/`; records in `<data>/records/` (per `AGENTS.md`, not git).
- **Work**: numerical sweep across all quant types and model architectures; throughput benchmarks vs. the SYCL build.
- **Dependencies**: phases 2-4.
- **Risk**: numerical divergence in the MMA re-derivation; precision of `exp`/`sqrt` (note
  `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt` in `CMakeLists.txt:334-335`, which Vulkan shaders must match).
- **Acceptance**: KL divergence < the SYCL-vs-CPU tolerance; no NaN/Inf; throughput reported.
- **Gate**: all parity tests green, no regression vs. SYCL on the same text.

### Phase 6 — Optimization for the 780M

- **Files**: tuning in the shader launch configs (tile shapes, work-group sizes, `STRATA_*` env overrides).
- **Work**: derive tile/chunk sizes from the device report (the first rule), not hard-coded; match the B70/RTX/RX
  measurement discipline in `AGENTS.md`.
- **Dependencies**: phase 5.
- **Risk**: RADV quirks; the 12 CUs under-fill large tiles.
- **Acceptance**: throughput within a target band of the SYCL build on the 780M; checked with `STRATA_VRAM_LIMIT_MIB` /
  `STRATA_MAX_ALLOC_MIB` / `STRATA_NO_XMX`-style caps simulating a smaller card.
- **Gate**: meets the performance target.

## 11. File-by-file intervention matrix

"Unchanged" = no edit needed. "Adapter" = a thin wrapper. "Rewrite" = new SPIR-V shader. "Verify" = decide after a
prototype.

| File | Role | Intervention | Reason |
| --- | --- | --- | --- |
| `src/core/gpu.cpp` | GPU control layer | **Rewrite** as `vulkan.cpp` | Reproduce `strata::gpu::*` contract |
| `src/core/device.cpp` | Device discovery/arena | **Rewrite** (Vulkan device/queue/memory) | SYCL `device`/`queue`/USM |
| `src/core/runtime.hpp/.cpp` | Per-device runtime | **Adapter** / rewrite | Queues, events, preload, watchdog |
| `src/core/graph.cpp` | Graph capture/replay | **Rewrite** | SYCL `command_graph` → Vulkan |
| `src/core/vmm.cpp` | Virtual memory | **Rewrite** / verify | `sycl_ext_oneapi_virtual_mem` |
| `src/core/weights.cpp`, `layout.cpp`, `graph.cpp` | Host/device state | **Adapter** | Use new `strata::gpu` |
| `src/core/plain_model.cpp` | Model driver | **Adapter** (or new `vulkan_model.cpp`) | Route calls to new backend |
| `src/prefill/gemm.cpp` | Dense GEMM | **Rewrite** or route to ggml-vulkan | oneMath calls SYCL queues |
| `src/kernels/xe/iq_mmq.cpp` | Expert int8 MMQ (128 KB) | **Rewrite** as GLSL | `joint_matrix`/WMMA → GLSL MMA |
| `src/kernels/xe/mma_gemm.cpp` | Prompt GEMM (tensor cores) | **Rewrite** (AMD path only) | `joint_matrix` → GLSL |
| `src/kernels/xe/xmx_gemm.cpp` | Prompt GEMM (XMX) | **Verify** (Intel-only path) | XMX has no Vulkan equivalent; use DP4a/WMMA |
| `src/kernels/xe/dp4a_gemm.cpp` | GEMM without matrix engines | **Rewrite** | Portable integer path |
| `src/kernels/xe/native_mmvq.cpp` | Per-type MMVQ (52 KB) | **Rewrite** per type | Integer decode is portable |
| `src/kernels/xe/iq_bits.hpp` | i-quant word handling | **Translate** | Pure integer logic |
| `src/kernels/xe/amd_wmma.hpp` | AMD WMMA lane planning | **Re-derive** for GLSL | gfx11 row ordering |
| `src/kernels/xe/qsa_prompt_attn.cpp` | Prompt attention | **Rewrite** | `joint_matrix`/WMMA |
| `src/kernels/xe/qsa.cpp`, `qsa_decode_attn.cpp`, `qsa_select.cpp` | Attention | **Rewrite** | `submit`/`parallel_for` |
| `src/kernels/xe/native_qsa*.cpp` | Norm/gate | **Rewrite** | subgroup reduction |
| `src/kernels/xe/rope.cpp`, `native_rope.cpp` | RoPE | **Rewrite** | `parallel_for` |
| `src/kernels/xe/sampler.cpp` | Sampling | **Rewrite** | subgroup ops |
| `src/kernels/xe/router.cpp`, `native_router.cpp` | Top-k routing | **Rewrite** | subgroup shuffle |
| `src/kernels/xe/native_moe.cpp` | MoE combine | **Rewrite** | `parallel_for<2>` |
| `src/kernels/xe/bf16_gemv.cpp` | BF16 gemv | **Rewrite** | `local_accessor` reductions |
| `src/kernels/xe/gdn.cpp`, `native_gdn.cpp`, `fused_gdn.cpp` | Conv | **Rewrite** | `submit` |
| `src/kernels/xe/gr.cpp`, `fused_gr.cpp` | SSM GR | **Rewrite** | `submit` |
| `src/kernels/xe/elementwise.cpp` | Elementwise | **Rewrite** | `parallel_for` |
| `src/kernels/xe/quantize_act.cpp` | Act quantize | **Rewrite** | `parallel_for` |
| `src/kernels/xe/kv_q4.cpp`, `kv_q8.cpp`, `kv_stream.cpp` | KV cache | **Rewrite** | `parallel_for` |
| `src/kernels/xe/s2_expert_grouped.cpp`, `moe_mmq.cpp` | Grouped MMQ | **Rewrite** | grouped `nd_range` |
| `src/kernels/xe/cvec.cpp` | Conversation vectors | **Rewrite** | `parallel_for` |
| `src/kernels/xe/verify_kernels.cpp` | Kernel self-tests | **Adapter** | Drive new backend |
| `src/kernels/xe/device_caps.hpp`, `device_target.hpp` | Device queries | **Adapter** | Map to Vulkan queries |
| `src/kernels/cpu/*` | CPU experts | **Unchanged** | Independent of GPU backend |
| `cmake/StrataHip.cmake` | AMD target detection | **Unchanged** (SYCL route) / new Vulkan CMake | gfx1103 already accepted |
| `CMakeLists.txt` | Build | **Add** Vulkan mode / glslang | New `STRATA_BACKEND=vulkan` option |
| `third_party/main/oneMath` | Dense GEMM lib | **Adapter/replace** | SYCL-queue API |
| Fetched llama.cpp/ggml | CPU experts (+ optional ggml-vulkan) | **Unchanged** / optional | Build-time fetch |

## 12. Test and benchmark plan (future)

Per `AGENTS.md`, records go in `<data>/records/`, never git. Tests already present that a Vulkan build must match:

- **Parity tests** (bit-for-bit vs. SYCL/CPU): `native_expert_parity`, `iq_mmq_parity`, `rope_parity`,
  `quantize_act_parity`, `s2_gemv_parity`, `kv_q4_parity`, `kv_q8_parity`, `router_top10_parity`, `qsa_*_parity`,
  `elementwise_parity`, `bf16_gemv_parity` (all in `src/kernels/*_parity.cpp`).
- **CPU-only** (no GPU): `tests/` via `strata-device`, `iq_avx2_parity`, `q8k_quant_parity`, `pool_test`.
- **Numerical**: KL divergence between Vulkan and SYCL builds on the four Qwen3.8 files (Q4_K/Q5_K/Q6_K/Q8_0/BF16)
  and the GSQ-RCO IQ files, as `native_expert_parity` already exercises.
- **Memory**: run with `STRATA_VRAM_LIMIT_MIB`/`STRATA_MAX_ALLOC_MIB`/`STRATA_NO_XMX` caps simulating a smaller card
  (per `AGENTS.md`), confirm the Vulkan allocator behaves.
- **Benchmarks**: tokens/s prefill and decode on the 780M, vs. the SYCL/HIP build and vs. the user's current
  `llama.cpp`-Vulkan usage. **Not run here.**
- **Cross-check**: build both free and contrib modes (`AGENTS.md`), confirm the Vulkan path in each; verify on a
  smaller config via caps.

## 13. Risks, limits and alternatives

| Risk | Severity | Notes |
| --- | --- | --- |
| Host/device pointer identity for mapped UMA memory | High | SYCL's same-address USM has no Vulkan guarantee; `VK_EXT_external_memory_host` workaround |
| Graph capture/replay | High | SYCL `command_graph` has no 1:1 Vulkan API; most-uncertain surface |
| MMA re-derivation (gfx11 lane planning) | High | Correctness-critical; `amd_wmma.hpp` ordering is gfx-specific |
| oneMath is SYCL-only | Medium | Dense GEMMs need a Vulkan oneMath, XeStrata kernels, or ggml-vulkan |
| Device-library availability for gfx1103 (SYCL route) | Medium | `strata_hip_device_libs_check` probes at configure time |
| Precision (`prec-div`/`prec-sqrt`) | Medium | Shaders must match `-foffload-fp32-prec-div -foffload-fp32-prec-sqrt` |
| RADV feature gaps | Medium | Assume nothing from the Vulkan spec; verify each extension on the actual driver |
| 12 CU under-utilization | Medium | Tile shapes must be derived from the device report |
| Build size / toolchain | Low | glslang + Vulkan SDK vs. intel/llvm + ROCm |

**Alternatives to a full port:**

- **Stay on SYCL/HIP** — the 780M already runs; no new code, no risk. This is the default recommendation if the goal is
  simply "run on the 780M."
- **Option B hybrid** — keep SYCL, use ggml-vulkan only for dense GEMMs. Low risk, low value.
- **ggml-based reimplementation** — rebuild XeStrata's workload as a ggml graph and run it through ggml-vulkan. Large
  re-derivation of the custom kernels, but reuses ggml's mature Vulkan backend.

## 14. Evidence matrix and open questions

| Claim | Evidence | Tag |
| --- | --- | --- |
| XeStrata is SYCL-based, no Vulkan backend | 72 files include `<sycl>`; `gpu.cpp` maps CUDA→SYCL; no `vulkan` in `src/` | VERIFICATO |
| Kernels already run on RDNA 3/4 via WMMA | `amd_wmma.hpp`, `iq_mmq.cpp:67,838,897`, `qsa_prompt_attn.cpp:489` | VERIFICATO |
| gfx1103 is a supported SYCL/HIP target | `StrataHip.cmake:42` regex accepts `gfx1103` | VERIFICATO |
| gfx1103 uses the non-gfx12 WMMA path | `__GFX11__` covers gfx1103; `*_w32` (no `_gfx12`) builtins | VERIFICATO |
| `amd_wmma()` accepts gfx1103 | `amd_wmma.hpp:33` range `gfx1100..gfx1300` | VERIFICATO |
| Dense GEMMs go through rocBLAS on AMD | `gemm.cpp:Gemm::path()` | VERIFICATO |
| rocBLAS has gfx11 kernels on the target | depends on distro packages | NON VERIFICATO |
| Full ggml/llama.cpp fetched at build | `CMakeLists.txt:858-868` | VERIFICATO |
| Fetched ggml has `ggml-vulkan` | not inspected locally | NON VERIFICATO |
| ggml-vulkan can run XeStrata's kernels | XeStrata is not a ggml graph | PLAUSIBILE |
| 780M runs XeStrata via SYCL/HIP today | synthesis of the above | PLAUSIBILE (needs a machine to prove) |
| RADV has `VK_AMD_device_host_access` | driver knowledge, not local code | PLAUSIBILE (verify version) |
| GLSL supports 16×16×16 subgroup MMA | Vulkan spec / RDNA 3 | PLAUSIBILE (verify RADV) |
| Graph capture is feasible in Vulkan | general Vulkan | PLAUSIBILE (least-documented here) |

**Open questions (need a machine or external API check):**

1. Does the available intel/llvm build produce device libraries for `gfx1103`? (`strata_hip_device_libs_check`)
2. Does the target's rocBLAS have `gfx11` kernels?
3. What Mesa/RADV version is on the Beelink, and which of the §9.2 extensions are actually enabled?
4. Can `VK_AMD_device_host_access` give a true host/device pointer identity for the mapped-expert shares?
5. What is the fetched llama.cpp's `GGML_VULKAN` default and API surface?
6. GLSL 16×16×16 MMA correctness and RADV performance on the 780M — prototype required.

## 15. Conclusion

**Feasibility: feasible, but the need is smaller than a full port suggests.**

The decisive, verified fact is that **the Radeon 780M already runs XeStrata through the existing SYCL/HIP path** — the
int8 and prompt-attention kernels take AMD's WMMA instructions (`__GFX11__`, non-gfx12 builtins), `gfx1103` is an
accepted build target, and dense GEMMs go through rocBLAS. No Vulkan is required to use the 780M.

A **native Vulkan Compute backend** (Option A) is feasible and is the only approach that truly removes the SYCL/LLVM
dependency. It is large: rewrite the `strata::gpu` control layer, all 41 kernel files as SPIR-V, with the two hardest
parts being (a) host/device pointer identity for the 780M's UMA expert shares and (b) re-deriving the gfx11 WMMA
lane-planning in GLSL. Graph capture is the least-documented surface. The roadmap in §10 stages the work so each stage
is bit-exact-parity-verified before the next begins.

**ggml-vulkan** (Option B) cannot execute XeStrata's existing kernels — the engine is not a ggml graph — so its reuse
is limited to a dense-GEMM-only hybrid or a full ggml-based reimplementation.

Recommendation: if the goal is "run on the 780M," **stay on SYCL/HIP** (zero new code, verified). If the goal is
"remove the SYCL/LLVM dependency" or "share a Vulkan codebase with ggml tooling," proceed with Option A via the staged
roadmap, and budget most of the effort for the memory-identity and MMA-re-derivation risks. Every claim marked
**NON VERIFICATO** or **PLAUSIBILE** here must be closed with a prototype on the actual Beelink SER8 before the work is
trusted.

---

*Analysis performed on the local copy at commit `180d0776` (`main`), branch up to date with `origin/main`. No source
files were modified; no repository was cloned, downloaded, built, or benchmarked. External API/driver facts are marked
and must be verified on the target machine.*
