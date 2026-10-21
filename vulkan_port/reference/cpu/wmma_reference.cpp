// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// reference/cpu/wmma_reference.cpp - bit-exact CPU reference of the gfx11 16x16x16 WMMA in src/kernels/xe/amd_wmma.hpp.
//
// Independent of SYCL and of Vulkan: it reproduces ONLY the arithmetic and the lane mapping (row_of = 2*e + hi) so the
// macOS Vulkan harness (harness) can check the FMA shader's output element-by-element.  It is the reference truth on
// macOS for wmma_fma.comp.
//
// Mapping reproduced exactly (GFX11 branch of amd_wmma.hpp):
//   KN = 16, KOFF = 0, row_of(hi, e) = 2*e + hi.
//   lane16(sg) = id & 15, half_of(sg) = id >> 4.
//   load_rows: lane L reads A[lane16*ld + half_of*KOFF + e] for e in 0..15  -> A row-major: A[e*ld + lane16] with ld=16.
//   store: lane L writes C[row_of(hi,e)*ld + lane16] for e in 0..7  (8 accumulator values per lane, the f16 Wmma
//   produces 8 per lane; here we compute all 16 K and scatter the full 16x16 so the host comparison is complete).
//
// On the Beelink SER8 the SAME file will be compiled against the SYCL kernel's output and compared bit-exact to close
// the loop SYCL <-> reference <-> GLSL (parity test).  Until then this is host-only.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::vulkan_port::reference {

using h16 = _Float16;  // matches amd_wmma.hpp's h16 vector element type

// A (row-major, 16x16 fp16) and B (column-major, 16x16 fp16): element (k, idx) at buf[k*16 + idx].
// Output C (32x16 fp32): element (r, c) at out[r*16 + c], r in 0..31, c in 0..15.
inline void wmma_fma_reference(const h16* a, const h16* b, float* c, int64_t n_rows, float scale) {
    // n_rows is the output row count (32 for the full row_of() scatter); lanes = n_rows / 2.
    const int lanes = int(n_rows / 2);
    for (int lane = 0; lane < lanes; ++lane) {
        const int e = lane & 15;      // lane16
        const int hi = lane >> 4;     // half_of
        const int r = 2 * e + hi;     // row_of(hi, e)  -- gfx11
        float acc = 0.0f;
        for (int k = 0; k < 16; ++k) {
            const float av = float(a[size_t(k) * 16 + size_t(e)]);
            const float bv = float(b[size_t(k) * 16 + size_t(e)]);
            acc = acc + av * bv;      // fma, same order as the shader's loop
        }
        c[size_t(r) * 16 + size_t(e)] = scale * acc;
    }
}

// Reference for the subgroup-matrix variant's scatter (row_of lane assignment) so the two can be cross-checked even
// though the matrix-intrinsic shader does not run on macOS.  dense[r][c] is the dense [16][16] product; scatter it to
// the row_of() slot the gfx11 instruction would write.
inline void wmma_subgroup_reference(const h16* a, const h16* b, float* c, float scale) {
    float dense[16][16] = {{0.0f}};
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < 16; ++k) acc += float(a[size_t(k) * 16 + size_t(c)]) * float(b[size_t(k) * 16 + size_t(c)]);
            dense[r][c] = acc;
        }
    // scatter: dense[r][c] -> lane with row_of==r, e==c -> slot (row_of)*16 + e, matching the shader's store loop.
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c)
            c[size_t(r) * 16 + size_t(c)] = scale * dense[r][c];
}

}  // namespace strata::vulkan_port::reference
