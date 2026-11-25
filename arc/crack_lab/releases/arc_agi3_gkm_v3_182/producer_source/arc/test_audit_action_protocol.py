import json
from pathlib import Path

import audit_action_protocol as A


def _checkpoint(path: Path, game: str, actions: list, reached: int = 1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "game": game,
                "reached": reached,
                "final_path": actions,
                "validated": True,
            }
        ),
        encoding="utf-8",
    )


def _transcript(path: Path, *, output: str = "", message: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "python probe.py",
                "aggregated_output": output,
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": message},
        },
    ]
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def test_action_contract_rejects_aliases_and_out_of_range():
    for valid in (1, 7, [6, 0, 63], (6, 63, 0)):
        assert A.action_error(valid) is None
    for invalid in (
        True,
        0,
        6,
        8,
        "1",
        [6, 1],
        [5, 1, 2],
        [6, True, 2],
        [6, -1, 2],
        [6, 64, 2],
        [6, 2, 64],
    ):
        assert A.action_error(invalid) is not None


def test_report_finds_checkpoint_and_executed_probe_incidents(tmp_path):
    root = tmp_path / "agent_solutions"
    current = root / "bad_legs" / "checkpoint.json"
    promoted = (
        root
        / "bad_legs"
        / "promotion_evidence"
        / "level_01"
        / "files"
        / "checkpoint.json"
    )
    _checkpoint(current, "bad", [[6, 1, 64]], reached=1)
    _checkpoint(promoted, "bad", [[6, 1, 2]], reached=1)
    transcript = promoted.parent.parent / "codex_turns" / "turn.jsonl"
    _transcript(
        transcript,
        output=(
            "run (320, 320) 10 reward 0\n"
            f"{A.PROTOCOL_VIOLATION_MARKER}: coordinate outside 0..63\n"
        ),
        message="The experiment clicked (18,112) while invisible.",
    )

    report = A.run(root)

    assert report["evidence_verdict"] == "FAIL"
    assert report["strict_verdict"] == "FAIL"
    assert len(report["checkpoint_violations"]) == 1
    coordinates = {
        (row["x"], row["y"])
        for row in report["out_of_range_transcript_findings"]
    }
    assert (320, 320) in coordinates
    assert (18, 112) in coordinates
    assert any(
        row["pattern"] == "latched_protocol_violation"
        for row in report["out_of_range_transcript_findings"]
    )
    assert report["affected_games"] == ["bad"]
    assert report["affected_promoted_levels"] == 1


def test_clean_legacy_scan_exposes_evidence_limit(tmp_path):
    root = tmp_path / "agent_solutions"
    _checkpoint(
        root / "good_legs" / "checkpoint.json",
        "good",
        [[6, 0, 63]],
    )
    report = A.run(root)
    assert report["evidence_verdict"] == "PASS"
    assert report["strict_verdict"] == "FAIL"
    assert report["legacy_coordinate_games_without_complete_call_log"] == [
        "good"
    ]

    complete = A.run(root, complete_call_log=True)
    assert complete["strict_verdict"] == "PASS"
    assert complete["legacy_coordinate_games_without_complete_call_log"] == []
