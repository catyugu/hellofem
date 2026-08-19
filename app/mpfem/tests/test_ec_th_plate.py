"""Case-pipeline pytest: EcThPlateCoupled (aluminum plate, electric-thermal coupled)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "app" / "mpfem" / "scripts"
CASE = "EcThPlateCoupled"

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
