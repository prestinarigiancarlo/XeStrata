# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# scripts/run_smoke.sh - compile the shaders and run the harness on lavapipe and MoltenVK on macOS.
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== compile shaders =="
glslc -fshader-stage=compute --target-env=vulkan1.3 shaders/wmma_fma.comp -o /tmp/wmma_fma.spv || {
  echo "FMA shader compile FAILED"; exit 1; }
glslc -fshader-stage=compute --target-env=vulkan1.3 shaders/wmma_subgroup.comp -o /tmp/wmma_subgroup.spv 2>/tmp/sg.err || {
  echo "EXPECTED: subgroup shader fails (VK_EXT_subgroup_matrix not on lavapipe/MoltenVK)."; cat /tmp/sg.err; }

echo "== validate SPIR-V =="
spirv-val /tmp/wmma_fma.spv && echo "fma.spv: valid"

echo "== build harness =="
cmake -S harness -B build/harness -DCMAKE_PREFIX_PATH="${VULKAN_SDK:-}" >/dev/null
cmake --build build/harness

echo "== smoke: lavapipe =="
if [ -n "${LVP_ICD:-}" ]; then
  export VK_ICD_FILENAMES="$LVP_ICD"
  STRATA_VULKAN_ICD=lavapipe build/harness/vulkan_harness || echo "lavapipe smoke failed"
else
  echo "SKIP lavapipe: set LVP_ICD to lvp_icd.*.json"
fi

echo "== smoke: MoltenVK =="
if [ -n "${MOLTENVK_ICD:-}" ]; then
  export VK_ICD_FILENAMES="$MOLTENVK_ICD"
  STRATA_VULKAN_ICD=moltenvk build/harness/vulkan_harness || echo "MoltenVK smoke failed"
else
  echo "SKIP MoltenVK: set MOLTENVK_ICD to its ICD json"
fi
