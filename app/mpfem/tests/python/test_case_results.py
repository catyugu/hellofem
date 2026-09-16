"""Application-layer result verification.

For every case under app/mpfem/cases, runs mpfem_app on the case artifacts and
requires the comparison against the COMSOL reference result.txt to pass.

The pipeline reuses a COMSOL reference only when its SHA-256 manifest matches
the Java source and all reference artifacts. Stale references are rebuilt when
COMSOL is available and otherwise skipped.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]  # repo root
sys.path.insert(0, str(ROOT))

from app.mpfem.scripts.run_case import reference_is_current

APP = ROOT / "app" / "mpfem"
RUN_CASE = APP / "scripts" / "run_case.py"
MPFEM_APP = (
    ROOT
    / "build"
    / "app"
    / ("mpfem_app.exe" if sys.platform == "win32" else "mpfem_app")
)

# A case directory is runnable when it holds its COMSOL Java model.
CASES = sorted(
    p.name for p in (APP / "cases").iterdir() if (p / f"{p.name}.java").exists()
)

pytestmark = pytest.mark.skipif(
    not MPFEM_APP.exists(), reason="needs a built mpfem_app"
)


@pytest.mark.parametrize("case", CASES)
def test_case_matches_comsol(case):
    case_dir = APP / "cases" / case
    if not reference_is_current(case_dir, case) and shutil.which("comsolbatch") is None:
        pytest.skip("needs COMSOL on PATH to rebuild a stale case reference")

    r = subprocess.run(
        [sys.executable, str(RUN_CASE), "run", case], capture_output=True, text=True
    )
    assert r.returncode == 0, (
        f"{case}: pipeline failed (rc={r.returncode})\nstdout:\n{r.stdout[-4000:]}\n"
        f"stderr:\n{r.stderr[-2000:]}"
    )
    assert (
        "PASS" in r.stdout and "FAIL" not in r.stdout
    ), f"{case}: comparison did not pass\nstdout:\n{r.stdout[-4000:]}"
    assert reference_is_current(case_dir, case), f"{case}: reference manifest is stale"
