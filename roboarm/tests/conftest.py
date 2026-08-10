"""Test configuration that keeps generated state below roboarm/artifacts."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"


def pytest_configure() -> None:
    ARTIFACT_ROOT.mkdir(exist_ok=True)
    os.environ.setdefault(
        "PYTHONPYCACHEPREFIX",
        str(ARTIFACT_ROOT / "pycache"),
    )
    os.environ.setdefault(
        "XDG_CACHE_HOME",
        str(ARTIFACT_ROOT / "xdg-cache"),
    )
