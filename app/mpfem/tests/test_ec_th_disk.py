"""Case-pipeline pytest: EcThDiskCoupled (copper disk, electric-thermal coupled).

Runs the full hellofem pipeline for a second-order (P2) tetrahedral mesh:
COMSOL Java model -> mesh.mphtxt -> mpfem_app -> result_hellofem.txt, then
compares against the COMSOL reference result.txt. P2 geometry support is
provided by the arbitrary-order refactor (mesh -> element order path), where
the reader adapts the COMSOL edge-midpoint node ordering to the basix
reference ordering in the IO layer.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "app" / "mpfem" / "scripts"
CASE = "EcThDiskCoupled"

_HAS_COMSOL = shutil.which("comsolbatch") is not None
_HAS_BUILD = (ROOT / "build" / "app" / ("mpfem_app.exe" if sys.platform == "win32" else "mpfem_app")).exists()

pytestmark = pytest.mark.skipif(
    not (_HAS_COMSOL and _HAS_BUILD),
    reason="needs COMSOL on PATH and a built mpfem_app")


def test_case_pipeline():
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_case.py"), "run", CASE],
        capture_output=True, text=True)
    assert r.returncode == 0, (
        f"pipeline failed (rc={r.returncode})\nstdout:\n{r.stdout[-4000:]}\n"
        f"stderr:\n{r.stderr[-2000:]}")
    assert "PASS" in r.stdout and "FAIL" not in r.stdout
