#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# scripts/compare.py - compare two binary float/int outputs bit-exact: report element mismatches, max ULP, NaN/Inf.
#
# Usage: compare.py <ref.bin> <got.bin> [--dtype float|double|int32] [--ulp]
# Reads both files as raw little-endian records of the given dtype and reports, per element, whether they match
# bit-for-bit, the max ULP distance for the float types, and any NaN/Inf divergence.  Exit code 0 = exact, 1 = differ.
import argparse
import struct
import sys
import math

DTYPES = {
    "float": ("<f", 4, "float"),
    "double": ("<d", 8, "double"),
    "int32": ("<i", 4, "int"),
    "uint32": ("<I", 4, "uint"),
}

def read(path, fmt, size):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // size
    return list(struct.unpack("<" + fmt * n, data[:n * size]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("got")
    ap.add_argument("--dtype", default="float", choices=sorted(DTYPES))
    args = ap.parse_args()

    fmt, size, kind = DTYPES[args.dtype]
    ref = read(args.ref, fmt, size)
    got = read(args.got, fmt, size)
    n = min(len(ref), len(got))

    if len(ref) != len(got):
        print(f"LENGTH MISMATCH: ref={len(ref)} got={len(got)}")
        return 1

    mism = 0
    max_ulp = 0.0
    nan_inf = 0
    first_few = []
    for i in range(n):
        r, g = ref[i], got[i]
        rnan = isinstance(r, float) and math.isnan(r)
        gnan = isinstance(g, float) and math.isnan(g)
        rinf = isinstance(r, float) and math.isinf(r)
        gin = isinstance(g, float) and math.isinf(g)
        if rnan or gnan or rinf or gin:
            if (rnan != gnan) or (rinf != gin) or (rnan and gnan and math.copysign(1, r) != math.copysign(1, g)):
                nan_inf += 1
                if len(first_few) < 10:
                    first_few.append((i, r, g, "nan/inf"))
            # NaN/Inf counts as a mismatch unless both are identical NaN/Inf sign
            if not (rnan == gnan and rinf == gin and (not (rnan or rinf))):
                mism += 1
            continue
        if kind in ("float", "double"):
            if r != g:
                mism += 1
                # ULP distance
                ra = struct.unpack("<Q", struct.pack("<" + fmt, r))[0]
                ga = struct.unpack("<Q", struct.pack("<" + fmt, g))[0]
                # normalize sign to make ULP contiguous
                if ra & 0x8000000000000000: ra = 0x8000000000000000 - ra
                if ga & 0x8000000000000000: ga = 0x8000000000000000 - ga
                ulp = abs(ra - ga)
                max_ulp = max(max_ulp, float(ulp))
        else:
            if r != g:
                mism += 1
        if len(first_few) < 10 and r != g and not (isinstance(r, float) and math.isnan(r)):
            first_few.append((i, r, g, "value"))

    print(f"elements={n} mismatches={mism} max_ulp={max_ulp:.0f} nan_inf_divergence={nan_inf}")
    for i, r, g, why in first_few[:10]:
        print(f"  [{i}] ref={r!r} got={g!r} ({why})")
    return 1 if mism else 0

if __name__ == "__main__":
    sys.exit(main())
