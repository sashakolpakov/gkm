import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import audit_action_boundaries as A


def write_checkpoint(path: Path, *, game="fake", reached=2, actions=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "game": game,
                "reached": reached,
                "final_path": actions or [1, 2, 3],
                "validated": True,
            }
        )
    )


def test_audit_checkpoint_distinguishes_exact_overlong_and_unreproduced(tmp_path):
    checkpoint = tmp_path / "g_legs" / "checkpoint.json"
    write_checkpoint(checkpoint)

    exact = A.audit_checkpoint(
        tmp_path, "current", checkpoint, lambda game, path, level: list(path)
    )
    assert exact.verdict == "PASS"
    assert exact.exact_actions == 3

    overlong = A.audit_checkpoint(
        tmp_path, "current", checkpoint, lambda game, path, level: list(path[:2])
    )
    assert overlong.verdict == "OVERLONG"
    assert overlong.exact_actions == 2

    missing = A.audit_checkpoint(
        tmp_path, "current", checkpoint, lambda game, path, level: None
    )
    assert missing.verdict == "UNREPRODUCED"
    assert missing.exact_actions is None


def test_checkpoint_files_includes_current_and_promotion_evidence(tmp_path):
    current = tmp_path / "g_legs" / "checkpoint.json"
    historical = (
        tmp_path
        / "g_legs"
        / "promotion_evidence"
        / "level_01"
        / "files"
        / "checkpoint.json"
    )
    write_checkpoint(current)
    write_checkpoint(historical, reached=1)
    assert A.checkpoint_files(tmp_path) == [
        ("current", current),
        ("promotion_evidence", historical),
    ]


def test_complete_chain_gate_requires_every_promoted_level(
    tmp_path, monkeypatch
):
    current = tmp_path / "g_legs" / "checkpoint.json"
    level1 = (
        tmp_path
        / "g_legs"
        / "promotion_evidence"
        / "level_01"
        / "files"
        / "checkpoint.json"
    )
    level2 = (
        tmp_path
        / "g_legs"
        / "promotion_evidence"
        / "level_02"
        / "files"
        / "checkpoint.json"
    )
    write_checkpoint(current, game="g", reached=2)
    write_checkpoint(level1, game="g", reached=1)

    def exact_without_runtime(root, kind, checkpoint):
        data = json.loads(checkpoint.read_text())
        actions = data["final_path"]
        return A.BoundaryResult(
            path=str(checkpoint.relative_to(root)),
            game=data["game"],
            level=data["reached"],
            recorded_actions=len(actions),
            exact_actions=len(actions),
            kind=kind,
            verdict="PASS",
        )

    monkeypatch.setattr(A, "audit_checkpoint", exact_without_runtime)
    incomplete = A.run(
        tmp_path,
        isolate_games=False,
        require_complete_chain=True,
    )
    assert incomplete["verdict"] == "FAIL"
    assert incomplete["promotion_chain"] == {
        "expected": 2,
        "present": 1,
        "missing": [{
            "game": "g",
            "level": 2,
            "expected_path": (
                "g_legs/promotion_evidence/level_02/files/checkpoint.json"
            ),
        }],
        "issues": [],
        "complete": False,
    }

    write_checkpoint(level2, game="g", reached=2)
    complete = A.run(
        tmp_path,
        isolate_games=False,
        require_complete_chain=True,
    )
    assert complete["verdict"] == "PASS"
    assert complete["promotion_chain"]["complete"] is True
    assert complete["promotion_chain"]["expected"] == 2
    assert complete["promotion_chain"]["present"] == 2


def test_run_aggregates_isolated_game_workers(tmp_path, monkeypatch):
    write_checkpoint(tmp_path / "a_legs" / "checkpoint.json", game="a", reached=1)
    write_checkpoint(tmp_path / "b_legs" / "checkpoint.json", game="b", reached=1)
    calls = []

    def fake_run(argv, **kwargs):
        game = argv[argv.index("--game") + 1]
        calls.append(game)
        row = A.BoundaryResult(
            path=f"{game}_legs/checkpoint.json",
            game=game,
            level=1,
            recorded_actions=3,
            exact_actions=3,
            kind="current",
            verdict="PASS",
        )
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "results": [A.asdict(row)],
                    "verdict": "PASS",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    report = A.run(tmp_path)

    assert sorted(calls) == ["a", "b"]
    assert report["checkpoints"] == 2
    assert report["exact"] == 2
    assert report["verdict"] == "PASS"


def test_run_honors_isolated_game_filter_and_rejects_unknown(
    tmp_path, monkeypatch
):
    write_checkpoint(tmp_path / "a_legs" / "checkpoint.json", game="a", reached=1)
    write_checkpoint(tmp_path / "b_legs" / "checkpoint.json", game="b", reached=1)
    calls = []

    def fake_isolated(_root, game):
        calls.append(game)
        return [
            A.BoundaryResult(
                path=f"{game}_legs/checkpoint.json",
                game=game,
                level=1,
                recorded_actions=3,
                exact_actions=3,
                kind="current",
                verdict="PASS",
            )
        ]

    monkeypatch.setattr(A, "_isolated_game_results", fake_isolated)
    report = A.run(tmp_path, games={"b"})
    assert calls == ["b"]
    assert report["checkpoints"] == 1
    assert report["results"][0]["game"] == "b"

    with pytest.raises(ValueError, match="no checkpoint evidence"):
        A.run(tmp_path, games={"missing"})
