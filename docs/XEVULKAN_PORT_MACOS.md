# XeStrata Vulkan port — macOS artifacts

Static artifacts for the Vulkan port of XeStrata, generated/compiled/validated on macOS without the AMD Radeon
780M (`gfx1103`) or RADV.  Everything needing that hardware is explicitly deferred and marked **DA VERIFICARE SU
HARDWARE (Beelink SER8 + RADV)**.  See `docs/PROMPT_FOR_LLM_macos.md` for the role and scope.

Layout: `vulkan_port/shaders/`, `vulkan_port/harness/`, `vulkan_port/reference/`, `vulkan_port/api/`,
`vulkan_port/scripts/`.  The hardware checklist is `docs/PROMPT_FOR_LLM_macos_HARDWARE_CHECKLIST.md`.

## 1. GLSL shaders — subgroup-matrix variant and portable FMA variant

Two shaders for the 16×16×16 (f16 and i8) multiply, ports of `src/kernels/xe/amd_wmma.hpp`.

### `vulkan_port/shaders/wmma_subgroup.comp` — subgroup matrix (`VK_EXT_subgroup_matrix`)
- Uses `layout(num_rows=16, num_columns=16, nesting=32)` semantics via `subgroupMatrixMultiplyAccumulate`, the
  Vulkan equivalent of `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`.
- The gfx11 lane mapping `row_of(hi, e) = 2*e + hi` is translated explicitly: the gfx11 instruction scatters the
  32 lanes onto all 32 accumulator rows (half 0 → even rows, half 1 → odd rows, interleaved by `e`); GLSL has no
  per-lane scatter, so the SSBO is laid out with `row_of()` and the shader scatters lane-0's dense product back out
  (`wmma_subgroup_reference` in the CPU reference reproduces the same index math).
- Bindings: 0 = A (16×16 fp16, row-major), 1 = B (16×16 fp16, column-major), 2 = C (16×16 fp32), 3 = scale (1 fp32).
- **Expected to fail to compile on `glslc`/lavapipe/MoltenVK**: `VK_EXT_subgroup_matrix` is not advertised there.
  The `#error` guard makes this failure explicit rather than silent.  Run it only on RADV.

Compilation (RADV only):
```sh
glslc -fshader-stage=compute --target-env=vulkan1.3 shaders/wmma_subgroup.comp -o wmma_subgroup.spv
spirv-val wmma_subgroup.spv        # expect: success
```

### `vulkan_port/shaders/wmma_fma.comp` — portable FMA fallback (compiles and runs on lavapipe/MoltenVK)
- Same arithmetic and the same `row_of = 2*e + hi` lane mapping as the AMD instruction, but with vector FMA over
  the 32 lanes instead of the subgroup matrix intrinsic.  Each lane `(hi,e)` computes its K-dot product and writes
  `C[row_of(hi,e)][e]`.
- Bindings: 0 = A (16×16 fp16 row-major), 1 = B (16×16 fp16 column-major), 2 = C (32×16 fp32 out, row_of scatter),
  3 = scale.  std430; fp16 element stride 32 bytes, fp32 stride 64 bytes, both satisfy min-alignment.

Compilation + validation (macOS):
```sh
glslc -fshader-stage=compute --target-env=vulkan1.3 shaders/wmma_fma.comp -o wmma_fma.spv
spirv-val wmma_fma.spv             # expect: success
spirv-dis wmma_fma.spv             # inspect: 32 lanes, one FMA-per-K loop
```
`spirv-val` output on macOS: `success` (no errors, no warnings for the std430 bindings).

## 2. Minimal Vulkan C++ harness (compilable on macOS)

`vulkan_port/harness/main.cpp` + `vulkan_port/harness/CMakeLists.txt`:
- Creates instance/device; selects a GPU, forced via `STRATA_VULKAN_ICD=lavapipe|moltenvk` (falls back to the first
  device).
- Allocates host-visible buffers, loads deterministic inputs (16×16 fp16 tiles), compiles `wmma_fma.comp` with
  `glslc`, dispatches one compute group (32 lanes), reads C back, and compares element-by-element with the CPU
  reference (`vulkan_port/reference/cpu/wmma_reference.cpp`), printing a bit-exact verdict.
- RADV stub: on macOS the device is never RADV, so `radv_stub = true` skips the hardware-specific tests and prints
  "DA VERIFICARE SU HARDWARE".

Run on macOS:
```sh
# lavapipe (Mesa software Vulkan)
cmake -S harness -B build/harness -DCMAKE_PREFIX_PATH="$VULKAN_SDK" && cmake --build build/harness
export VK_ICD_FILENAMES="$LVP_ICD"            # lvp_icd.*.json from mesa-vulkan-drivers
STRATA_VULKAN_ICD=lavapipe build/harness/vulkan_harness

# MoltenVK
export VK_ICD_FILENAMES="$MOLTENVK_ICD"       # MoltenVK ICD json
STRATA_VULKAN_ICD=moltenvk build/harness/vulkan_harness
```
`scripts/run_smoke.sh` automates compile → validate → build → smoke.

**Significant on macOS:** API generica (instance/device/buffer/dispatch/fence/readback) and the FMA shader's
algorithm + lane mapping (bit-exact vs reference).
**Not significant on macOS:** the subgroup-matrix shader (does not compile there) and every RADV-only extension.

## 3. Bit-exact CPU reference of the WMMA algorithm

`vulkan_port/reference/cpu/wmma_reference.cpp` — independent of SYCL and Vulkan, reproduces only the arithmetic and
the gfx11 lane mapping (`KN=16`, `row_of(hi,e)=2*e+hi`, `lane16=id&15`, `half_of=id>>4`).  It is the reference truth
on macOS for `wmma_fma.comp`.

On the Beelink SER8 the same file is compiled against the SYCL `amd_wmma` kernel output and compared bit-exact to
close the loop **SYCL ↔ reference ↔ GLSL** (a parity test, mirroring the existing `*_parity.cpp` files).  Until then
it is host-only.

## 4. `vulkan.hpp` API design

`vulkan_port/api/vulkan_gpu.hpp` defines the public API replicating `strata::gpu::*` (stream, event,
`alloc_device`, `alloc_host`, copy, memset, wait, graph capture) with the same contract — `false`/null on error,
`last_error()`, `vk_status`.  `vulkan_port/api/test_vulkan_gpu.cpp` unit-tests the pure parts (structures, handle
mapping, state, error strings) on macOS; the driver-backed methods are mocked.

Decision points requiring a physical-machine choice (**DA VERIFICARE SU HARDWARE**):
1. **Host/device pointer identity** — SYCL maps a host pointer to its device counterpart by pointer identity;
   Vulkan does not guarantee this.  Options: (A) assume identity and use it directly; (B) a VMM handle table mapping
   each allocation to both its device and host addresses (chosen here, robust); (C) force `VK_DEVICE_HOST_ACCESS`
   so the two are the same address.  Criterion: (C) if the extension is present and simplest, else (B).
2. **Semaphore/fence format** — per-queue fences (SYCL events) vs cross-queue semaphores (`VK_KHR_semaphore` / core
   1.3).  Criterion: cross-queue deps present → semaphore; else fence.
3. **Graph capture** — decode's `plain_model.cpp:980` uses SYCL graph capture with no 1:1 Vulkan equivalent.
   Options: (A) secondary command buffers + `VK_EXT_command_buffer_rendering`; (B) immediate submit fallback.
   Criterion: capture reproduces immediate output exactly, else fall back.

## 5. Kernel → shader mapping and incremental porting plan

From the 41 files with launch/dispatch/stream code in `src/kernels/xe/` (verified: 41 .cpp).  Complexity is
bassa/media/alta, motivated; existing parity test named where present.  Groups ordered by rising risk.

| File | Operation | SYCL primitive | GLSL complexity | MMA dep | Parity test | macOS-validable? |
| --- | --- | --- | --- | --- | --- | --- |
| bf16_gemv.cpp | BF16 GEMV / mmvf | parallel_for, warp xor-sum | bassa | no | bf16_gemv_parity.cpp | yes — warp-sum port, no MMA |
| s_gemv.cpp, s2_gemv.cpp, s2_expert_grouped.cpp | FP32/GEMV grouped | sub_group reduce | bassa | no | s_gemv_parity.cpp, s2_gemv_parity.cpp | yes |
| rope.cpp | rotary embedding | parallel_for | bassa | no | rope_parity.cpp | yes |
| elementwise.cpp, quantize_act.cpp, dequant_bf16.cpp | elementwise/quant | parallel_for | bassa | no | elementwise_parity.cpp, quantize_act_parity.cpp, dequant_s2_parity.cpp | yes |
| cvec.cpp, sampler.cpp, router.cpp | reduction/softmax | sub_group shuffle | media | no | cvec_parity.cpp, sampler_parity.cpp, router_top10_parity.cpp | mostly (reductions, no MMA) |
| kv_q4.cpp, kv_q8.cpp, kv_stream.cpp | KV cache ops | parallel_for + atomics | media | no | kv_q4_parity.cpp, kv_q8_parity.cpp, kv_stream_parity.cpp | yes (no MMA) |
| qsa*.cpp, prompt_attn.cpp, native_flash_attn.cpp | attention/attn scores | warp sums, shuffles | alta | yes (mat-vec) | qsa_parity.cpp, qsa_prompt_attn_parity.cpp | no — attention needs MMA lane mapping |
| iq_mmq.cpp, iq_kernels.cpp | int8 IQ matmul | dp4a / i8 WMMA | alta | yes (i8) | iq_mmq_parity.cpp, iq_parity.cpp | no — i8 MMA + lane map |
| native_moe.cpp, shared_expert.cpp, native_ple_postops.cpp | grouped MoE | grouped parallel_for | alta | partial | shared_expert_parity.cpp, ple_parity.cpp | no — grouped MMA layout |
| gdn.cpp, gr.cpp, fused_gdn.cpp, fused_gr.cpp | GDN/GR normalisation | parallel_for + invert | media | no | gdn_parity.cpp, gr_parity.cpp | mostly (non-MMA core) |
| dp4a_gemm.cpp | FP16→FP32 GEMV via DP4a | subgroup dp4a | alta | yes (dp4a) | s_gemv_q8k_parity.cpp | no — dp4a MMA path |
| mma_gemm.cpp, xmx_gemm.cpp | dense GEMM | joint_matrix / WMMA | alta | yes (core) | — (matrix_report) | no — dense MMA, oneMath in contrib |

Grouping by rising risk: (1) primitives without MMA → fully validatable on macOS; (2) reductions/softmax/KV →
reductions port directly, no MMA; (3) attention/QSA → need the MMA lane mapping (Beelink); (4) IQ/GEMM/MoE/dense →
the MMA kernels themselves (Beelink, bit-exact vs SYCL).  The two dense GEMM files go through oneMath (contrib) or
XeStrata's own kernels; they are the highest risk and are not part of the f16/i8 16×16 tile focus here.

## 6. Comparison script and hardware checklist

`scripts/compare.py <ref.bin> <got.bin> [--dtype float|double|int32|uint32]` — reads both as raw little-endian
records and reports per-element bit-exact mismatches, max ULP (float types), and NaN/Inf divergence.  Exit 0 = exact,
1 = differ.  Used on macOS for FMA-vs-reference and on the Beelink for SYCL-vs-GLSL parity (item 6 of the checklist).

`docs/PROMPT_FOR_LLM_macos_HARDWARE_CHECKLIST.md` — the Beelink SER8 verification checklist, one item per
decision/verification: exact command, expected output, go/no-go.  Items: (1) device enumeration, (2)
`VK_EXT_subgroup_matrix` + subgroup layout query, (3) `VK_AMD_device_host_access`, (4) host/device pointer identity
via a mapped-buffer round-trip, (5) graph capture vs immediate submit, (6) bit-exact SYCL↔GLSL parity, (7)
performance (deferred, recorded in `records/`).  Each RADV/gfx1103-dependent claim is marked **DA VERIFICARE SU
HARDWARE** with the command that closes it; nothing here assumes what RADV supports.
