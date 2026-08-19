#!/usr/bin/env python3
"""compare_results.py — generic COMSOL-style result comparison.

Parses two COMSOL "Data" export text files (x y z <expr> ...), aligns
rows by coordinates, and reports per-field metrics (L2, L2 relative,
max relative, Linf) with a pass/fail verdict against configurable
tolerances. Exits non-zero when any field exceeds tolerance.

Usage:
    compare_results.py <reference.txt> <current.txt> \
        [--fields V,T,solid.disp] [--l2-rel 1e-2] [--max-rel 1e-1]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


def parse_result(path: Path):
    """Return (expressions, rows) where rows are (x, y, z, values...)."""
    expressions = []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line:
            continue
        if line.startswith("%"):
            m = re.match(r"% Expressions:\s*(\d+)", line)
            if m:
                expressions = [f"f{i}" for i in range(int(m.group(1)))]
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                continue
            rows.append((nums[0], nums[1], nums[2], nums[3:]))
    return expressions, rows


def align(ref_rows, cur_rows, coord_tol=1e-7):
    """Align by rounded coordinates; return parallel lists of value rows."""
    scale = 1.0 / coord_tol
    def key(r):
        return (round(r[0] * scale), round(r[1] * scale), round(r[2] * scale))
    ref_map = {}
    for r in ref_rows:
        k = key(r)
        if k in ref_map:
            raise ValueError(f"duplicate reference coordinate {r[:3]}")
        ref_map[k] = r
    cur_map = {}
    for r in cur_rows:
        k = key(r)
        if k in cur_map:
            raise ValueError(f"duplicate current coordinate {r[:3]}")
        cur_map[k] = r
    missing = [k for k in ref_map if k not in cur_map]
    if missing:
        raise ValueError(f"current file missing {len(missing)} reference points")
    keys = sorted(ref_map)
    return [ref_map[k] for k in keys], [cur_map[k] for k in keys]


def metrics(ref, cur):
    """Per-field (L2, L2_rel, max_rel, Linf). `max_rel` is the worst
    pointwise relative error (denominator = |ref|), which spikes near zeros;
    the verdict uses `linf_scale` (max abs error / max |ref|) instead."""
    n = len(ref)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    sq = sum((c - r) ** 2 for r, c in zip(ref, cur))
    ref_sq = sum(r * r for r in ref)
    max_abs_ref = max((abs(r) for r in ref), default=0.0)
    l2 = math.sqrt(sq / n)
    l2_rel = math.sqrt(sq / ref_sq) if ref_sq > 0 else 0.0
    linf = max((abs(c - r) for r, c in zip(ref, cur)), default=0.0)
    max_rel = max((abs(c - r) / max(abs(r), 1e-12)
                   for r, c in zip(ref, cur)), default=0.0)
    linf_scale = linf / max_abs_ref if max_abs_ref > 0 else 0.0
    return l2, l2_rel, max_rel, linf, linf_scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path)
    ap.add_argument("current", type=Path)
    ap.add_argument("--fields", default=None,
                    help="comma-separated fields to compare (default: all)")
    ap.add_argument("--l2-rel", type=float, default=1e-2,
                    help="max allowed L2 relative error per field")
    ap.add_argument("--linf-rel", type=float, default=5e-2,
                    help="max allowed Linf / max|ref| error per field")
    args = ap.parse_args()

    _, ref_rows = parse_result(args.reference)
    _, cur_rows = parse_result(args.current)
    if len(ref_rows) != len(cur_rows):
        raise SystemExit(f"point count mismatch: ref={len(ref_rows)} cur={len(cur_rows)}")
    ncol = len(ref_rows[0][3])
    if not args.fields:
        fields = [f"f{i}" for i in range(ncol)]
    else:
        fields = args.fields.split(",")

    ref_aligned, cur_aligned = align(ref_rows, cur_rows)

    ok = True
    print("field\tL2\tL2_rel\tmax_rel\tLinf\tLinf/max|ref|\tverdict")
    for i, name in enumerate(fields):
        if i >= ncol:
            print(f"{name}\t(no column)")
            ok = False
            continue
        ref = [r[3][i] for r in ref_aligned]
        cur = [r[3][i] for r in cur_aligned]
        l2, l2_rel, max_rel, linf, linf_scale = metrics(ref, cur)
        pass_ = l2_rel <= args.l2_rel and linf_scale <= args.linf_rel
        ok = ok and pass_
        print(f"{name}\t{l2:.6e}\t{l2_rel:.6e}\t{max_rel:.6e}\t{linf:.6e}\t"
              f"{linf_scale:.6e}\t{'PASS' if pass_ else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
