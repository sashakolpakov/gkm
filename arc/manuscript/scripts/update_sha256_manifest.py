#!/usr/bin/env python3
"""Write or verify the manuscript's fixed-scope SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable


MANIFEST_PATHS = (
    "../../.github/workflows/pages.yml",
    "../../README.md",
    "../../REPRODUCE_ARC.md",
    "../../docs/generated/arc_artifacts.rst",
    "../../docs/generated/marginal_complexity_by_level.rst",
    "../../docs/self_improving_agent.rst",
    "../ARC.md",
    "../README.md",
    "../audit_results/README.md",
    "../audit_results/marginal-literal-reuse.json",
    "../build_artifact_docs.py",
    "../test_build_artifact_docs.py",
    "../crack_lab/arc_agi3_leaderboard_v3_gate.py",
    "../crack_lab/replay_scorecard.py",
    "../crack_lab/releases/arc_agi3_gkm_v2_181/README.md",
    "../crack_lab/test_arc_agi3_leaderboard_v3_gate.py",
    "../crack_lab/test_replay_scorecard.py",
    "../crack_lab/test_verify_frozen_release.py",
    "../crack_lab/verify_frozen_release.py",
    "Makefile",
    "README.md",
    "BUILD_VERIFICATION.md",
    "SOCRATIC_PASSES.md",
    "arc_agi3.tex",
    "artifact_history/README.md",
    "build_artifact_history.py",
    "figure_sources/inverse_colimit_attachment_standalone.tex",
    "figures/bounded_campaign_profiles.pdf",
    "figures/bounded_campaign_profiles.png",
    "figures/ls20_sawtooth.pdf",
    "figures/ls20_sawtooth.png",
    "figures/marginal_complexity_profiles.pdf",
    "figures/marginal_complexity_profiles.png",
    "generated/arc_artifacts.tex",
    "generated/canonical-action-boundaries.json",
    "generated/canonical-action-protocol-audit.json",
    "generated/canonical-taint-audit.json",
    "generated/comparator_stats.md",
    "generated/comparator_stats.tex",
    "generated/marginal_complexity_by_level.json",
    "generated/marginal_complexity_by_level.md",
    "generated/marginal_complexity_by_level.tex",
    "gkm_one_page_summary.md",
    "gkm_one_page_summary.tex",
    "history_manifest.py",
    "opine_world_comparison.md",
    "references.bib",
    "repo_ground_truth_matrix.md",
    "reproduction_report.json",
    "requirements-figures.txt",
    "scripts/build_arxiv_bundle.py",
    "scripts/generate_empirical_tables.py",
    "scripts/generate_figures.py",
    "scripts/reproduce_manuscript.py",
    "scripts/test_build_arxiv_bundle.py",
    "scripts/test_generate_empirical_tables.py",
    "scripts/test_generate_figures.py",
    "scripts/test_reproduce_manuscript.py",
    "scripts/test_update_sha256_manifest.py",
    "scripts/update_sha256_manifest.py",
)


class ManifestError(RuntimeError):
    pass


def _safe_path(base: Path, value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value or not pure.parts:
        raise ManifestError(f"unsafe manifest path: {value!r}")
    candidate = base.joinpath(*pure.parts)
    repo = base.parents[1].resolve()
    try:
        resolved = candidate.resolve()
        relative = resolved.relative_to(repo)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"manifest path escapes repository: {value!r}") from exc
    cursor = repo
    for part in relative.parts[:-1]:
        cursor /= part
        metadata = cursor.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ManifestError(f"manifest path has a symlinked parent: {value}")
    return candidate


def _digest_file(path: Path, label: str) -> str:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ManifestError(f"manifest input is not a single-link file: {label}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_manifest(base: Path, paths: Iterable[str]) -> bytes:
    labels = tuple(paths)
    if len(labels) != len(set(labels)):
        raise ManifestError("manifest allowlist contains duplicate paths")
    return "".join(
        f"{_digest_file(_safe_path(base, label), label)}  {label}\n"
        for label in labels
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--manuscript-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    base = args.manuscript_root.resolve()
    manifest = base / "SHA256SUMS.txt"
    try:
        expected = render_manifest(base, MANIFEST_PATHS)
        if args.write:
            with tempfile.NamedTemporaryFile(
                dir=base, prefix=".SHA256SUMS.", delete=False
            ) as handle:
                handle.write(expected)
                temporary = Path(handle.name)
            os.replace(temporary, manifest)
            return 0
        if manifest.read_bytes() != expected:
            raise ManifestError(
                "SHA256SUMS.txt is stale; run update_sha256_manifest.py --write"
            )
    except (ManifestError, OSError) as exc:
        print(f"manifest verification failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
