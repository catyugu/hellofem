"""Case-pipeline pytest: EcThDiskCoupled (copper disk, electric-thermal coupled).

NOTE: This case generates a second-order (P2) tetrahedral mesh by COMSOL
(even though the Java code looks identical to the P1 variant EcThPlateCoupled —
the default model-level order differs between saved .mph files). hellofem
currently runs with `order=1` hardcoded in app/mpfem/main.cpp:212 and lacks
P2 Lagrange basis support, so this case is SKIPPED until we refactor the mesh
→ element order path (see issue tracker).
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


def test_case_skipped():
    """Placeholder: EcThDiskCoupled requires P2 basis support."""
    pytest.skip("requires P2 basis support (second-order mesh)")
