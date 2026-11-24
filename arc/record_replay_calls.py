#!/usr/bin/env python3
"""Observe an existing replay driver; do not rewrite or admit candidate code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

from executable_reuse import ReplayCalls


def read_sources(root: Path):
    files = {}
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink() or any(part in {".git", ".venv", "__pycache__"} for part in path.relative_to(root).parts):
            continue
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    if not files:
        raise ValueError("source snapshot contains no Python files")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entrypoint", action="append", default=[])
    parser.add_argument("--runtime-map", type=Path, help="JSON snapshot-filename to actual compiler-filename map")
    parser.add_argument("driver", type=Path, help="existing trusted replay driver executed in this process")
    parser.add_argument("driver_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    previous, current = read_sources(args.previous), read_sources(args.current)
    runtime_paths = (json.loads(args.runtime_map.read_text()) if args.runtime_map else
                     {name: str((args.current / name).resolve()) for name in current})
    observer = ReplayCalls(previous, current, runtime_paths, args.entrypoint)
    sys.argv = [str(args.driver), *args.driver_args]
    # Preserve the normal script import entry. The trusted driver retains its
    # existing candidate-loading/sandbox logic; a candidate directory must not
    # shadow verifier dependencies merely because it is being observed.
    sys.path[0] = str(args.driver.resolve().parent)
    try:
        with observer:
            runpy.run_path(str(args.driver), run_name="__main__")
    finally:
        report = observer.report()
        report["driver"] = str(args.driver)
        report["verification"] = "call accounting only; game success remains the replay driver's responsibility"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
