from __future__ import annotations

import json
from pathlib import Path

import reproduce_manuscript


def _payload() -> dict:
    systems = {}
    for index, system in enumerate(reproduce_manuscript.SYSTEMS, start=1):
        systems[system] = {
            field: index for field in reproduce_manuscript.STAT_FIELDS
        }
        systems[system]["sharp_drops_with_literal_reuse"] = [{}] * index
    return {"summary": {"systems": systems}}


def test_summary_counts_coupled_witnesses() -> None:
    summary = reproduce_manuscript._summary(_payload())
    assert summary["GKM"]["sharp_drops_with_literal_reuse"] == 1
    assert summary["Retrodict"]["sharp_drops_with_literal_reuse"] == 4


def test_generated_stats_are_machine_readable(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    payload = json.loads(
        (repo / "arc/audit_results/marginal-literal-reuse.json").read_text()
    )
    summary = reproduce_manuscript._summary(payload)
    tex_path, md_path = reproduce_manuscript._write_generated_stats(
        summary, payload, tmp_path,
    )
    assert (
        rf"\newcommand{{\GKMExactWins}}{{"
        f"{summary['GKM']['exact_winning_checkpoints']}"
        "}"
    ) in tex_path.read_text()
    assert "| Retrodict | 170 memory (145 transitions) | 0 |" in md_path.read_text()
