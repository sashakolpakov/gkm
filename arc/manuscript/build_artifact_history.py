#!/usr/bin/env python3
"""Materialize the manuscript audit sidecar from preserved provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from history_manifest import HISTORIES, ROOT, SOLUTIONS, PromotionSource


OUTPUT = Path(__file__).resolve().parent / "artifact_history"


CORE_FILES = (
    "legs.py",
    "players.py",
    "solve.py",
    "legs_log.md",
    "checkpoint.json",
    "auto_solve_attempts.json",
)


def _git_bytes(ref: str, path: str) -> bytes | None:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True
    )
    return proc.stdout if proc.returncode == 0 else None


def _source_files(game: str, source: PromotionSource) -> dict[str, bytes]:
    artifact = SOLUTIONS / f"{game}_legs"
    if source.kind == "git":
        result = {}
        for name in CORE_FILES:
            data = _git_bytes(source.source, f"{source.prefix}/{name}")
            if data is not None:
                result[name] = data
        return result
    if source.kind == "wip":
        root = artifact / "wip_context" / source.source / "files"
    elif source.kind == "current":
        root = artifact
    else:
        raise ValueError(f"unknown promotion source kind: {source.kind}")
    return {name: (root / name).read_bytes() for name in CORE_FILES if (root / name).is_file()}


def _digest(files: dict[str, bytes]) -> str:
    hasher = hashlib.sha256()
    for name, data in sorted(files.items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(data)
    return hasher.hexdigest()[:12]


def _promotion_payload(game: str, source: PromotionSource) -> tuple[dict[str, bytes], dict[str, object]]:
    files = _source_files(game, source)
    required = {"legs.py", "players.py", "solve.py", "legs_log.md"}
    if not required.issubset(files):
        missing = ", ".join(sorted(required - files.keys()))
        raise ValueError(f"{game} through L{source.through_level}: missing {missing}")
    digest = _digest(files)
    checkpoint = json.loads(files["checkpoint.json"]) if "checkpoint.json" in files else None
    source_observed_reached = None
    if source.kind == "wip":
        source_metadata = (
            SOLUTIONS / f"{game}_legs" / "wip_context" / source.source / "metadata.json"
        )
        source_observed_reached = int(json.loads(source_metadata.read_text())["reached"])
        if source_observed_reached < source.through_level:
            raise ValueError(
                f"{game} through L{source.through_level}: WIP metadata reached only "
                f"L{source_observed_reached}"
            )
    elif source.kind == "current" and checkpoint is not None:
        source_observed_reached = int(checkpoint["reached"])
    metadata = {
        "schema_version": 1,
        "game": game,
        "through_level": source.through_level,
        "replay_validated": True,
        "source_observed_reached": source_observed_reached,
        "embedded_checkpoint_actions": len(checkpoint["final_path"]) if checkpoint else None,
        "embedded_checkpoint_records": checkpoint["records"] if checkpoint else None,
        "clean_core_digest": digest,
        "source": {"kind": source.kind, "value": source.source, "prefix": source.prefix},
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())
        },
    }
    return files, metadata


def _write_promotion(game: str, source: PromotionSource) -> dict[str, object]:
    files, metadata = _promotion_payload(game, source)
    digest = str(metadata["clean_core_digest"])
    destination = OUTPUT / game / f"level_{source.through_level:02d}" / digest
    files_dir = destination / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = files_dir / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"refusing to replace immutable promotion file {path}")
        path.write_bytes(data)
    (destination / "promotion.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _manifest(game: str, promotions: list[dict[str, object]]) -> dict[str, object]:
    history = HISTORIES[game]
    return {
        "schema_version": 1,
        "game": game,
        "verified_through_level": history.max_level,
        "final_replay_actions": history.replay_actions,
        "final_public_commit": history.final_public_commit,
        "metric": "positive net retained-description growth per source file",
        "ledger": [
            {
                "level": entry.level,
                "marginal_C": entry.marginal_C,
                "evidence": entry.evidence,
            }
            for entry in history.ledger
        ],
        "total_marginal_C": history.total_marginal_C,
        "promotions": promotions,
        "final_artifact": f"arc/crack_lab/agent_solutions/{game}_legs",
        "wip_context": f"arc/crack_lab/agent_solutions/{game}_legs/wip_context",
    }


def check() -> None:
    for game, history in HISTORIES.items():
        artifact = SOLUTIONS / f"{game}_legs"
        expected_promotions = []
        for source in history.promotions:
            files, metadata = _promotion_payload(game, source)
            digest = str(metadata["clean_core_digest"])
            destination = OUTPUT / game / f"level_{source.through_level:02d}" / digest
            if json.loads((destination / "promotion.json").read_text()) != metadata:
                raise ValueError(f"promotion metadata mismatch: {destination}")
            for name, data in files.items():
                if (destination / "files" / name).read_bytes() != data:
                    raise ValueError(f"promotion file mismatch: {destination / 'files' / name}")
            expected_promotions.append(metadata)
        manifest_path = OUTPUT / game / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest != _manifest(game, expected_promotions):
            raise ValueError(f"history manifest mismatch: {manifest_path}")
        levels = [row["level"] for row in manifest["ledger"]]
        if levels != list(range(1, history.max_level + 1)):
            raise ValueError(f"{game}: ledger is not complete and ordered")
        if sum(row["marginal_C"] for row in manifest["ledger"]) != manifest["total_marginal_C"]:
            raise ValueError(f"{game}: ledger does not sum to total_marginal_C")
        checkpoint = json.loads((artifact / "checkpoint.json").read_text())
        if not checkpoint["validated"] or checkpoint["reached"] != history.max_level:
            raise ValueError(f"{game}: final root artifact is not validated through the endpoint")
        if len(checkpoint["final_path"]) != history.replay_actions:
            raise ValueError(f"{game}: final replay length mismatch")


def build() -> None:
    for game, history in HISTORIES.items():
        history_root = OUTPUT / game
        promotions = [_write_promotion(game, source) for source in history.promotions]
        (history_root / "manifest.json").write_text(
            json.dumps(_manifest(game, promotions), indent=2) + "\n"
        )


def main() -> None:
    if sys.argv[1:] == ["--check"]:
        check()
    elif not sys.argv[1:]:
        build()
        check()
    else:
        raise SystemExit("usage: build_promoted_history.py [--check]")


if __name__ == "__main__":
    main()
