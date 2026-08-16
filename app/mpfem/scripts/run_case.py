#!/usr/bin/env python3
"""run_case.py — single-script wrapper for the hellofem/COMSOL case pipeline.

Stage 1 (COMSOL):  Model.java --comsolcompile--> .class --comsolbatch--> solve + exports
Stage 2 (clean):   generated_model.java (Save-As-Java) -> clean_model.java
Stage 3 (hellofem): mpfem_app clean_model.java mesh.mphtxt -> result_hellofem.txt
Stage 4 (compare):  compare_results.py result.txt vs result_hellofem.txt

Usage:
    python run_case.py probe <case-dir>            # Stage 1 only, validate COMSOL round-trip
    python run_case.py run <case-dir> [--no-comsol] [--no-clean]
    python run_case.py clean <case-dir>            # Stage 2 only
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]            # repo root
APP = ROOT / "app" / "mpfem"
COMCOL = shutil.which("comsolcompile")
COMSOLBATCH = shutil.which("comsolbatch")


def _run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        if r.stdout:
            print(r.stdout[-4000:])
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r


def _comsol_ok(out: str) -> bool:
    """A batch may exit 0 while the model failed. Match the real failure
    markers, not benign solver chatter like "Solution error estimates"."""
    bad = re.compile(
        r"Undefined variable|Failed to|Exception|\*\*\*\*\*Error\*\*\*\*\*|"
        r"diverg|license|invalid|Compilation error", re.I)
    return not bad.search(out)


def _exe(name: str) -> Path:
    exe = ROOT / "build" / "app" / (name + (".exe" if os.name == "nt" else ""))
    if not exe.exists():
        raise SystemExit(f"build output not found: {exe} (run cmake --build build)")
    return exe


def _split_statements(text: str) -> list[str]:
    """Split a Java file into statements on ';' outside string literals."""
    stmts = []
    cur = []
    i = 0
    in_str = False
    while i < len(text):
        c = text[i]
        if in_str:
            cur.append(c)
            if c == '\\':
                i += 1
                if i < len(text):
                    cur.append(text[i])
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
                cur.append(c)
            elif c == ';':
                stmts.append("".join(cur))
                cur = []
            else:
                cur.append(c)
        i += 1
    if cur:
        stmts.append("".join(cur))
    return stmts


def _keep_clean(stmt: str) -> bool:
    """True if a Save-As-Java statement is part of the model definition that
    the mpfem driver parses (params, materials, physics, multiphysics,
    study, the Data export). Geometry/mesh detail and non-Data exports are
    dropped — the mesh comes from mesh.mphtxt."""
    s = stmt.strip()
    if not s:
        return False
    if s.startswith("import") or s.startswith("public") or s.startswith("//") \
       or s.startswith("/*") or s.startswith("*/"):
        return False
    if not s.startswith("model."):
        return False
    if s.startswith("model.param("):
        return True
    if s.startswith("model.modelPath("):
        return False
    m = re.match(r"model\.component\([^)]*\)\.(\w+)\(", s)
    if m:
        sub = m.group(1)
        if sub == "geom":
            return False
        if sub == "material":
            return True
        if sub == "physics":
            return True
        if sub == "multiphysics":
            return True
        if sub == "mesh":
            # Keep only mesh().create / autoMeshSize / run (drop FreeTet/size detail).
            return re.search(r"mesh\(\)\.create|autoMeshSize|mesh\(\"[^\"]*\"\)\.run", s) is not None
        if sub == "study":
            return "study().create" in s or "study(\"" in s and "create" in s
        if sub == "result":
            # Keep only the Data result export (drop the mesh export).
            return ("export().create(\"data1\", \"Data\")" in s
                    or 'export("data1")' in s)
    if s.startswith("model.study("):
        return "create" in s
    if s.startswith("model.result()"):
        # Keep the Data export (tag "data1") statements only; drop the mesh
        # text export ("mesh1").
        return "data1" in s and "mesh1" not in s
    return False


def clean_case(case_dir: Path):
    """Stage 2: generated_model.java -> clean_model.java."""
    src = case_dir / "generated_model.java"
    dst = case_dir / "clean_model.java"
    if not src.exists():
        raise SystemExit(f"{src} not found — run the COMSOL stage first")
    kept = [s + ";" for s in _split_statements(src.read_text(encoding="utf-8"))
            if _keep_clean(s)]
    dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"clean_model.java written ({len(kept)} statements)")


def stage1_comsol(case_dir: Path, case_name: str):
    """Compile + batch-run the case, exporting result/mesh/mph/generated-java."""
    src = case_dir / f"{case_name}.java"
    rc = _run([COMCOL, str(src)], cwd=case_dir)
    if not _comsol_ok(rc.stdout or ""):
        raise SystemExit("comsolcompile reported errors — see compile log")
    out_mph = case_dir / f"{case_name}.mph"
    out_log = case_dir / "batch.log"
    r = _run([COMSOLBATCH, "-inputfile", str(src.with_suffix(".class")),
              "-outputfile", str(out_mph), "-batchlog", str(out_log),
              "-study", "std1", "-np", "1"], cwd=case_dir)
    log_text = ""
    if out_log.exists():
        log_text = out_log.read_text(encoding="utf-8", errors="replace")
    combined = (r.stdout or "") + log_text
    for line in combined.splitlines():
        if any(k in line for k in ("_OK", "_EXC", "GROUND=", "TERMINAL=", "CONV=")):
            print(line.strip())
    if not _comsol_ok(combined):
        raise SystemExit("COMSOL batch reported errors — see batch.log")


def stage_mpfem(case_dir: Path):
    """Stage 3: run mpfem_app on the clean model + mesh."""
    r = _run([str(_exe("mpfem_app")),
              str(case_dir / "clean_model.java"),
              str(case_dir / "mesh.mphtxt"),
              str(case_dir / "result_hellofem.txt")])
    return r


def stage_compare(case_dir: Path):
    """Stage 4: compare COMSOL reference vs hellofem result."""
    r = subprocess.run([sys.executable,
                        str(APP / "scripts" / "compare_results.py"),
                        str(case_dir / "result.txt"),
                        str(case_dir / "result_hellofem.txt"),
                        "--l2-rel", "1e-2", "--linf-rel", "5e-2"],
                       capture_output=True, text=True)
    print(r.stdout or "")
    if r.stderr:
        print(r.stderr, file=sys.stderr)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["probe", "run", "clean"])
    ap.add_argument("case", type=Path, help="case directory (name or path)")
    ap.add_argument("--no-comsol", action="store_true",
                    help="skip the COMSOL stage (reuse existing artifacts)")
    ap.add_argument("--no-clean", action="store_true",
                    help="skip regenerating clean_model.java")
    args = ap.parse_args()

    if args.action != "clean" and (COMCOL is None or COMSOLBATCH is None):
        raise SystemExit("comsolcompile/comsolbatch not found on PATH (COMSOL bin/win64)")

    case_dir = args.case
    if not case_dir.is_absolute():
        case_dir = APP / "cases" / case_dir
    case_dir = case_dir.resolve()
    if not case_dir.exists():
        raise SystemExit(f"case dir not found: {case_dir}")
    case_name = case_dir.name
    if not (case_dir / f"{case_name}.java").exists():
        raise SystemExit(f"{case_dir / (case_name + '.java')} not found")

    if args.action == "probe":
        stage1_comsol(case_dir, case_name)
    elif args.action == "clean":
        clean_case(case_dir)
    elif args.action == "run":
        if not args.no_comsol:
            stage1_comsol(case_dir, case_name)
        if not args.no_clean:
            clean_case(case_dir)
        stage_mpfem(case_dir)
        sys.exit(stage_compare(case_dir))


if __name__ == "__main__":
    main()
