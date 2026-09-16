"""Application-layer result verification.

For every case under app/mpfem/cases, runs mpfem_app on the case artifacts and
requires the comparison against the COMSOL reference result.txt to pass.

The COMSOL stage (comsolcompile + comsolbatch, ~40 s per case) runs only for a
case whose artifacts are missing or incomplete — deleting a case's artifacts
forces a fresh reference. Such a case is skipped when COMSOL is not on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]          # repo root
APP = ROOT / "app" / "mpfem"
RUN_CASE = APP / "scripts" / "run_case.py"
MPFEM_APP = ROOT / "build" / "app" / ("mpfem_app.exe" if sys.platform == "win32" else "mpfem_app")

# Inputs consumed by the mpfem stage plus the COMSOL reference it is checked
# against; all three are produced by the COMSOL stage.
REFERENCE_ARTIFACTS = ("mesh.mphtxt", "clean_model.java", "result.txt")

# A case directory is runnable when it holds its COMSOL Java model.
CASES = sorted(p.name for p in (APP / "cases").iterdir() if (p / f"{p.name}.java").exists())

pytestmark = pytest.mark.skipif(not MPFEM_APP.exists(), reason="needs a built mpfem_app")


def _has_reference(case_dir: Path) -> bool:
    return all((case_dir / name).exists() for name in REFERENCE_ARTIFACTS)


@pytest.mark.parametrize("case", CASES)
def test_case_matches_comsol(case):
    case_dir = APP / "cases" / case
    args = [sys.executable, str(RUN_CASE), "run", case]
    if _has_reference(case_dir):
        args += ["--no-comsol", "--no-clean"]
    elif shutil.which("comsolbatch") is None:
        pytest.skip("needs COMSOL on PATH to generate the case reference artifacts")

    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0, (
        f"{case}: pipeline failed (rc={r.returncode})\nstdout:\n{r.stdout[-4000:]}\n"
        f"stderr:\n{r.stderr[-2000:]}")
    assert "PASS" in r.stdout and "FAIL" not in r.stdout, (
        f"{case}: comparison did not pass\nstdout:\n{r.stdout[-4000:]}")
