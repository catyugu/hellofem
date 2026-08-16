#!/usr/bin/env python3
"""run_case.py — single-script wrapper for the hellofem/COMSOL case pipeline.

Stage 1 (COMSOL):  Model.java --comsolcompile--> .class --comsolbatch--> solve + exports
Stage 2 (hellofem): mpfem app parses clean Java + mesh.mphtxt -> solves -> result_hellofem.txt
Stage 3 (compare):  compare_results.py result.txt vs result_hellofem.txt

Usage:
    python run_case.py probe <case-dir>            # Stage 1 only, validate COMSOL round-trip
    python run_case.py run <case-dir> [--no-clean] # full pipeline
    python run_case.py clean <case-dir>            # strip generated Save-As-Java into clean form
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # repo root
APP = ROOT / "app" / "mpfem"
COMCOL = shutil.which("comsolcompile")
COMSOLBATCH = shutil.which("comsolbatch")


def _run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r


def _comsol_ok(out: str) -> bool:
    """A batch may exit 0 while the model failed; grep the combined log for red flags."""
    bad = re.compile(r"\b(error|exception|failed|diverg|license|invalid)\b", re.I)
    return not bad.search(out)


def stage1_probe(case_dir: Path):
    """Compile Probe.java, batch-run it, print which exports succeeded."""
    src = case_dir / "Probe.java"
    rc = _run([COMCOL, str(src)], cwd=case_dir)
    if not _comsol_ok(rc.stdout or ""):
        raise SystemExit("comsolcompile reported errors — see compile log")
    out_mph = case_dir / "probe.mph"
    out_log = case_dir / "batch.log"
    r = _run([COMSOLBATCH, "-inputfile", str(src.with_suffix(".class")),
              "-outputfile", str(out_mph), "-batchlog", str(out_log),
              "-study", "std1", "-np", "1"], cwd=case_dir)
    log_text = ""
    if out_log.exists():
        log_text = out_log.read_text(encoding="utf-8", errors="replace")
    combined = (r.stdout or "") + log_text
    for line in combined.splitlines():
        if any(k in line for k in ("PROBE", "_OK", "_EXC")):
            print(line.strip())
    if not _comsol_ok(combined):
        raise SystemExit("COMSOL batch reported errors — see batch.log")
    print("PROBE batch done; files in", case_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["probe", "run", "clean"])
    ap.add_argument("case", type=Path, help="case directory")
    args = ap.parse_args()

    if COMCOL is None or COMSOLBATCH is None:
        raise SystemExit("comsolcompile/comsolbatch not found on PATH (COMSOL bin/win64)")

    case_dir = args.case.resolve()
    if not case_dir.is_absolute():
        case_dir = (APP / "cases" / case_dir).resolve()
    if not case_dir.exists():
        raise SystemExit(f"case dir not found: {case_dir}")

    if args.action == "probe":
        stage1_probe(case_dir)
    elif args.action == "run":
        raise SystemExit("'run' not implemented yet (Phase F)")
    elif args.action == "clean":
        raise SystemExit("'clean' not implemented yet (Phase A)")


if __name__ == "__main__":
    main()
