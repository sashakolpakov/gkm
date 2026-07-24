#!/usr/bin/env python3
"""Adversarial smoke test for the container-backed Retrodict PythonTool."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrodict", type=Path, required=True)
    parser.add_argument("--thinharness", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--image", default="retrodict-analysis:locked")
    args = parser.parse_args()

    sys.path[:0] = [str(args.retrodict / "src"), str(args.thinharness)]
    os.environ["RETRODICT_ANALYSIS_IMAGE"] = args.image
    os.environ["RETRODICT_HOST_SENTINEL"] = "must-not-enter-analysis-container"

    from arc3.tools import PythonArgs, PythonTool

    tool = PythonTool(args.workspace, args.retrodict / "missing-analysis-venv-python")
    cases = {
        "workspace": (
            "from pathlib import Path; Path('scratch.txt').write_text('ok'); "
            "print(Path('scratch.txt').read_text())"
        ),
        "engine_import": "import arcengine",
        "host_source": (
            "from pathlib import Path; "
            "p=Path('/private/tmp/Retrodict/environment_files'); "
            "print(p.exists(), list(Path('/').rglob('ft09.py')))"
        ),
        "host_environment": (
            "import os; print(os.getenv('RETRODICT_HOST_SENTINEL')); "
            "print(sorted(k for k in os.environ if 'API_KEY' in k or 'TOKEN' in k or 'SECRET' in k))"
        ),
        "network": (
            "import socket; s=socket.socket(); s.settimeout(2); "
            "s.connect(('1.1.1.1', 53))"
        ),
    }
    results = {name: tool.run(PythonArgs(code=code, timeout=10)) for name, code in cases.items()}

    assert results["workspace"].ok, results["workspace"].content
    assert not results["engine_import"].ok and "ModuleNotFoundError" in results["engine_import"].content
    assert results["host_source"].ok and "False []" in results["host_source"].content
    assert results["host_environment"].ok
    assert "None" in results["host_environment"].content and "[]" in results["host_environment"].content
    assert not results["network"].ok

    for name, result in results.items():
        summary = " ".join(result.content.strip().split())[:240]
        print(f"{name}: ok={result.ok} {summary}")


if __name__ == "__main__":
    main()
