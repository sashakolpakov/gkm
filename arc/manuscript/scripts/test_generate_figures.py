from __future__ import annotations

import json
from pathlib import Path

import pytest

import generate_figures


def _write_checkpoint(root: Path, game: str, records: list[dict]) -> Path:
    path = root / f"{game}_legs" / "checkpoint.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "game": game,
                "reached": len(records),
                "total_marginal_C": sum(r["marginal_C"] for r in records),
                "records": records,
                "validated": True,
            }
        )
    )
    return path


def test_load_profile_reads_replay_validated_records(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        "demo",
        [
            {"level": 1, "marginal_C": 11, "reached": True},
            {"level": 2, "marginal_C": 3, "reached": True},
        ],
    )
    assert generate_figures._load_profile(tmp_path, "demo") == (11, 3)


def test_load_profile_rejects_nonconsecutive_levels(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        "demo",
        [
            {"level": 1, "marginal_C": 11, "reached": True},
            {"level": 3, "marginal_C": 3, "reached": True},
        ],
    )
    with pytest.raises(ValueError, match="not consecutive"):
        generate_figures._load_profile(tmp_path, "demo")


def test_load_profile_rejects_unvalidated_checkpoint(tmp_path: Path) -> None:
    path = _write_checkpoint(
        tmp_path,
        "demo",
        [{"level": 1, "marginal_C": 11, "reached": True}],
    )
    payload = json.loads(path.read_text())
    payload["validated"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not replay validated"):
        generate_figures._load_profile(tmp_path, "demo")
