#!/usr/bin/env python3
"""Build and optionally compile-check the minimal arXiv source upload.

arXiv compiles from the root of the uploaded archive.  This script preserves
the relative paths used by ``arc_agi3.tex`` while excluding repository
evidence, companion documents, caches, and local LaTeX build products.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


SOURCE_FILES = (
    Path("arc_agi3.tex"),
    Path("references.bib"),
    Path("generated/marginal_complexity_by_level.tex"),
    Path("generated/comparator_stats.tex"),
    Path("figures/marginal_complexity_profiles.png"),
    Path("figures/bounded_campaign_profiles.png"),
)

_INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
_GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
_BIB_RE = re.compile(r"\\bibliography\{([^}]+)\}")


def _declared_dependencies(tex: str) -> set[Path]:
    dependencies = {Path(name) for name in _INPUT_RE.findall(tex)}
    dependencies.update(Path(name) for name in _GRAPHICS_RE.findall(tex))
    for group in _BIB_RE.findall(tex):
        dependencies.update(Path(f"{name.strip()}.bib") for name in group.split(","))
    return dependencies


def validate_source_tree(root: Path) -> None:
    missing = [str(path) for path in SOURCE_FILES if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError("missing arXiv source dependencies: " + ", ".join(missing))

    main = (root / "arc_agi3.tex").read_text(encoding="utf-8")
    declared = _declared_dependencies(main)
    expected = set(SOURCE_FILES) - {Path("arc_agi3.tex")}
    if declared != expected:
        missing_from_bundle = sorted(str(path) for path in declared - expected)
        unused_in_bundle = sorted(str(path) for path in expected - declared)
        details = []
        if missing_from_bundle:
            details.append("unbundled dependencies: " + ", ".join(missing_from_bundle))
        if unused_in_bundle:
            details.append("unused bundle files: " + ", ".join(unused_in_bundle))
        raise ValueError("; ".join(details))


def build_bundle(root: Path, output: Path) -> str:
    validate_source_tree(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in SOURCE_FILES:
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (root / relative).read_bytes())
    return hashlib.sha256(output.read_bytes()).hexdigest()


def compile_check(bundle: Path) -> None:
    required_tools = ("pdflatex", "bibtex")
    unavailable = [tool for tool in required_tools if shutil.which(tool) is None]
    if unavailable:
        raise RuntimeError("compile check requires: " + ", ".join(unavailable))

    with tempfile.TemporaryDirectory(prefix="gkm-arxiv-check-") as directory:
        work = Path(directory)
        with zipfile.ZipFile(bundle) as archive:
            archive.extractall(work)
        commands = (
            ("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "arc_agi3.tex"),
            ("bibtex", "arc_agi3"),
            ("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "arc_agi3.tex"),
            ("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "arc_agi3.tex"),
        )
        for command in commands:
            result = subprocess.run(
                command,
                cwd=work,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode:
                tail = "\n".join(result.stdout.splitlines()[-80:])
                raise RuntimeError(f"{' '.join(command)} failed:\n{tail}")
        if not (work / "arc_agi3.pdf").is_file():
            raise RuntimeError("compile check did not produce arc_agi3.pdf")


def parse_args() -> argparse.Namespace:
    manuscript_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=manuscript_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=manuscript_root / "build" / "arxiv" / "arc_agi3_arxiv.zip",
    )
    parser.add_argument("--compile-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    digest = build_bundle(args.root.resolve(), args.output.resolve())
    if args.compile_check:
        compile_check(args.output.resolve())
    print(f"{args.output}: sha256={digest}; files={len(SOURCE_FILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
