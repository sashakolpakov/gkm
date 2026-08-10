"""Run the Phase-0 verification with every generated path inside roboarm."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"


def main() -> int:
    ARTIFACT_ROOT.mkdir(exist_ok=True)
    controlled_temp = ARTIFACT_ROOT / "tmp"
    controlled_temp.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONPYCACHEPREFIX": str(ARTIFACT_ROOT / "pycache"),
            "TMPDIR": str(controlled_temp),
            "XDG_CACHE_HOME": str(ARTIFACT_ROOT / "xdg-cache"),
        }
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "audited_pytest.py"),
        "-q",
        "--basetemp",
        str(ARTIFACT_ROOT / "pytest-tmp"),
        str(PROJECT_ROOT / "tests"),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
