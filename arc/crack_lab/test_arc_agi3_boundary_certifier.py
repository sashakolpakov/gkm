from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import arc_agi3_boundary_certifier as C


CORE = {
    "legs.py": "def leg(env):\n    env.step(1)\n",
    "players.py": "from legs import leg\n",
    "solve.py": "from players import leg\n\ndef solve(env):\n    leg(env)\n",
}


def test_replay_verifier_isolates_generated_auxiliary_modules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspaces = []
    for index in (1, 2):
        workspace = tmp_path / f"game_{index}"
        workspace.mkdir()
        (workspace / "perception.py").write_text(f"VALUE = {index}\n")
        (workspace / "legs.py").write_text(
            "from perception import VALUE\n"
        )
        (workspace / "players.py").write_text(
            "from legs import VALUE\n"
        )
        (workspace / "solve.py").write_text(
            "import players\n\n"
            "def solve(env):\n"
            "    env.value = players.VALUE\n"
        )
        workspaces.append(workspace)

    def fake_run_program(_game, solve, *, time_cap):
        assert time_cap == 7
        env = types.SimpleNamespace(value=None)
        solve(env)
        return env.value, [], None

    monkeypatch.setattr(
        C.gkm_legs.A, "run_program", fake_run_program
    )
    sentinel = types.ModuleType("perception")
    sentinel.VALUE = 99
    monkeypatch.setitem(sys.modules, "perception", sentinel)

    observed = [
        C.gkm_legs.run_solve_file(
            "test",
            str(workspace / "solve.py"),
            time_cap=7,
            resume_checkpoint=False,
        )[0]
        for workspace in (*workspaces, workspaces[0])
    ]

    assert observed == [1, 2, 1]
    assert sys.modules["perception"] is sentinel


def _valid_transcript() -> str:
    return "".join(
        json.dumps(event, separators=(",", ":")) + "\n"
        for event in (
            {"type": "thread.started", "thread_id": "thread-test"},
            {"type": "turn.started"},
            {
                "type": "item.started",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                },
            },
            {"type": "turn.completed", "usage": {}},
        )
    )


def _write_core(directory: Path, *, legs: str | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in CORE.items():
        (directory / name).write_text(
            legs if name == "legs.py" and legs is not None else source
        )


def _write_legacy(
    root: Path,
    *,
    game: str = "zz99",
    level: int = 1,
    legs: str | None = None,
    transcript: str | None = None,
) -> Path:
    evidence = (
        root / f"{game}_legs" / "promotion_evidence"
        / f"level_{level:02d}"
    )
    _write_core(evidence / "files", legs=legs)
    (evidence / "proposer_last.log").write_text(
        _valid_transcript() if transcript is None else transcript
    )
    return evidence


def _write_wip(
    root: Path,
    *,
    game: str = "zz99",
    level: int = 1,
    phase: str = "reached_before_debrief",
    transcript: bool = True,
    legs: str | None = None,
) -> Path:
    attempt = (
        root / f"{game}_legs" / "wip_context" / f"level_{level:02d}"
        / f"{phase}_attempt"
    )
    files = attempt / "files"
    _write_core(files, legs=legs)
    if transcript:
        (files / "proposer_last.log").write_text(_valid_transcript())
    (attempt / "metadata.json").write_text(json.dumps({
        "game": game,
        "level": level,
        "phase": phase,
        "reached": level,
    }))
    return attempt


def test_reached_before_debrief_outranks_and_binds_schema1_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "agent_solutions"
    _write_legacy(
        root,
        legs="def leg(env):\n    env.step(2)\n",
    )
    retained = _write_wip(
        root,
        phase="reached_before_debrief",
        legs="def leg(env):\n    env.step(1)\n",
    )
    monkeypatch.setattr(
        C, "_source_replay", lambda *_args, **_kwargs: [1]
    )
    monkeypatch.setattr(C, "_path_replay", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        C.gkm_legs, "_file_taint_reason", lambda *_args, **_kwargs: None
    )

    candidates = C.boundary_candidates(root, game="zz99", level=1)
    selected, exact_path = C._select_and_replay(
        root, game="zz99", level=1, time_cap=1
    )

    assert [candidate.kind for candidate in candidates[:2]] == [
        "wip_snapshot",
        "legacy_promotion",
    ]
    assert selected.origin == retained
    assert selected.phase == "reached_before_debrief"
    assert selected.historical_source_boundary is True
    assert selected.retained_historical_phase is True
    assert selected.historical_transcript_complete is True
    assert exact_path == [1]

    game_root = tmp_path / "stage" / "zz99_legs"
    (game_root / "promotion_evidence").mkdir(parents=True)
    C._certify_boundary(
        source_root=root,
        game_root=game_root,
        game="zz99",
        level=1,
        records=[{"level": 1, "marginal_C": 7, "reached": True}],
        parent_checkpoint_sha256=None,
        parent_manifest_sha256=None,
        time_cap=1,
        scanner_sha256="a" * 64,
        engine_sha256="b" * 64,
        hasher_sha256="c" * 64,
    )
    boundary = game_root / "promotion_evidence" / "level_01"
    provenance = json.loads(
        (boundary / "files" / "provenance.json").read_text()
    )
    manifest = json.loads((boundary / "manifest.json").read_text())

    assert (boundary / "files" / "legs.py").read_text() == (
        "def leg(env):\n    env.step(1)\n"
    )
    assert provenance["source_origin"] == retained.relative_to(root).as_posix()
    assert provenance["historical_source_boundary"] is True
    assert provenance["historical_transcript_complete"] is True
    assert provenance["historical_transcript_failure"] is None
    assert provenance["posthoc_acquisition_marginal_admissible"] is True
    assert manifest["promoted_files_sha256"]["legs.py"] == C._sha256_file(
        boundary / "files" / "legs.py"
    )


def test_schema1_promoted_source_is_never_historical(
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent_solutions"
    _write_legacy(root)

    candidate = C.boundary_candidates(
        root, game="zz99", level=1
    )[0]

    assert candidate.kind == "legacy_promotion"
    assert candidate.phase == "legacy_schema1_promoted_source"
    assert candidate.historical_transcript_complete is True
    assert candidate.retained_historical_phase is False
    assert candidate.historical_source_boundary is False


def test_schema1_raw_transcript_failure_is_not_laundered(
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent_solutions"
    lines = _valid_transcript().splitlines()
    lines.insert(-1, "raw model-manager stderr")
    _write_legacy(root, transcript="\n".join(lines) + "\n")

    candidate = C.boundary_candidates(
        root, game="zz99", level=1
    )[0]

    assert candidate.kind == "legacy_promotion"
    assert candidate.historical_transcript_complete is False
    assert candidate.historical_transcript_failure == (
        "malformed_transcript_json_line_5"
    )
    assert candidate.historical_source_boundary is False


def test_raw_legacy_line_downgrades_retained_historical_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "agent_solutions"
    retained = _write_wip(root, phase="reached_before_debrief")
    transcript = retained / "files" / "proposer_last.log"
    lines = _valid_transcript().splitlines()
    lines.insert(
        -1,
        "2026-07-29T00:33:15Z ERROR raw CLI diagnostic",
    )
    transcript.write_text("\n".join(lines) + "\n")

    candidate = C.boundary_candidates(
        root, game="zz99", level=1
    )[0]

    assert candidate.phase == "reached_before_debrief"
    assert candidate.retained_historical_phase is True
    assert candidate.historical_transcript_complete is False
    assert candidate.historical_transcript_failure == (
        "malformed_transcript_json_line_5"
    )
    assert candidate.historical_source_boundary is False

    monkeypatch.setattr(
        C, "_select_and_replay", lambda *_args, **_kwargs: (candidate, [1])
    )
    monkeypatch.setattr(
        C.gkm_legs, "_file_taint_reason", lambda *_args, **_kwargs: None
    )
    game_root = tmp_path / "stage" / "zz99_legs"
    (game_root / "promotion_evidence").mkdir(parents=True)
    _checkpoint_hash, _manifest_hash, provenance = C._certify_boundary(
        source_root=root,
        game_root=game_root,
        game="zz99",
        level=1,
        records=[{"level": 1, "marginal_C": 7, "reached": True}],
        parent_checkpoint_sha256=None,
        parent_manifest_sha256=None,
        time_cap=1,
        scanner_sha256="a" * 64,
        engine_sha256="b" * 64,
        hasher_sha256="c" * 64,
    )

    assert provenance["retained_historical_phase"] is True
    assert provenance["historical_source_boundary"] is False
    assert provenance["historical_transcript_complete"] is False
    assert provenance["historical_transcript_failure"] == (
        "malformed_transcript_json_line_5"
    )
    assert provenance["posthoc_acquisition_marginal_admissible"] is False


def test_explicit_post_debrief_source_is_honestly_nonhistorical(
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent_solutions"
    _write_legacy(root)
    retained = _write_wip(root, phase="after_debrief")

    candidate = C.boundary_candidates(
        root, game="zz99", level=1
    )[0]

    assert candidate.origin == retained
    assert candidate.phase == "after_debrief"
    assert candidate.historical_transcript_complete is True
    assert candidate.retained_historical_phase is False
    assert candidate.historical_source_boundary is False


def test_legacy_source_uses_matching_wip_perception_dependency(
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent_solutions"
    legs = "from perception import arr\n\ndef leg(env):\n    arr(env.frame())\n"
    legacy = (
        root / "zz99_legs" / "promotion_evidence" / "level_01"
    )
    _write_core(legacy / "files", legs=legs)
    (legacy / "proposer_last.log").write_text("clean legacy transcript\n")
    wip = _write_wip(root, legs=legs)
    (wip / "files" / "perception.py").write_text(
        "def arr(frame):\n    return frame\n"
    )

    candidate = next(
        candidate
        for candidate in C.boundary_candidates(
            root, game="zz99", level=1
        )
        if candidate.kind == "legacy_promotion"
    )
    payloads, origins = C._source_payloads(candidate)

    assert candidate.kind == "legacy_promotion"
    assert candidate.phase == "legacy_schema1_promoted_source"
    assert candidate.historical_source_boundary is False
    assert candidate.dependency_dir == wip / "files"
    assert "perception.py" in payloads
    assert origins["perception.py"].endswith("/perception.py")


def test_missing_dependency_is_explicit_harness_seed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent_solutions"
    legacy = (
        root / "zz99_legs" / "promotion_evidence" / "level_01"
    )
    _write_core(
        legacy / "files",
        legs="from perception import arr\n\ndef leg(env):\n    arr(env.frame())\n",
    )
    candidate = C.boundary_candidates(
        root, game="zz99", level=1
    )[0]
    payloads, origins = C._source_payloads(candidate)

    assert payloads["perception.py"] == C.gkm_legs.PERCEPTION_SEED.encode()
    assert origins["perception.py"] == "harness:PERCEPTION_SEED"


def test_auto_solve_candidate_never_claims_historical_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "agent_solutions"
    _write_wip(
        root,
        phase="after_auto_solve_debrief",
    )

    candidate = C.boundary_candidates(
        root, game="zz99", level=1
    )[0]

    assert candidate.deterministic_reconstruction is True
    assert candidate.historical_source_boundary is False


def test_boundary_always_has_host_certification_transcript(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "agent_solutions"
    attempt = _write_wip(
        source_root,
        phase="after_auto_solve_debrief",
        transcript=False,
    )
    candidate = C.boundary_candidates(
        source_root, game="zz99", level=1
    )[0]
    assert candidate.transcript is None

    monkeypatch.setattr(
        C,
        "_select_and_replay",
        lambda *_args, **_kwargs: (candidate, [1]),
    )
    monkeypatch.setattr(
        C.gkm_legs, "_file_taint_reason", lambda *_args, **_kwargs: None
    )
    game_root = tmp_path / "stage" / "zz99_legs"
    (game_root / "promotion_evidence").mkdir(parents=True)

    _checkpoint_hash, _manifest_hash, provenance = C._certify_boundary(
        source_root=source_root,
        game_root=game_root,
        game="zz99",
        level=1,
        records=[{"level": 1, "marginal_C": 7, "reached": True}],
        parent_checkpoint_sha256=None,
        parent_manifest_sha256=None,
        time_cap=1,
        scanner_sha256="a" * 64,
        engine_sha256="b" * 64,
        hasher_sha256="c" * 64,
    )
    boundary = game_root / "promotion_evidence" / "level_01"
    transcript = json.loads(
        (boundary / "transcripts" / "certification.json").read_text()
    )
    action_protocol = json.loads(
        (boundary / "audits" / "action_protocol.json").read_text()
    )
    manifest = json.loads((boundary / "manifest.json").read_text())

    assert transcript["source_from_zero_replay"] == "PASS"
    assert transcript["path_from_zero_replay"] == "PASS"
    assert transcript["source_action_protocol"] == "PASS"
    assert transcript["path_action_protocol"] == "PASS"
    assert action_protocol["runtime_enforcement"] == (
        "shared_violation_latch_across_root_and_clones"
    )
    assert action_protocol["source_protocol_latch"] == "PASS"
    assert action_protocol["path_protocol_latch"] == "PASS"
    assert transcript["original_source_transcript_available"] is False
    assert manifest["transcripts"][0]["path"] == (
        "transcripts/certification.json"
    )
    assert provenance["deterministic_reconstruction"] is True
    assert provenance["posthoc_acquisition_marginal_admissible"] is False
    assert provenance["source_origin"] == attempt.relative_to(
        source_root
    ).as_posix()


def test_certifier_preserves_native_transcript_scan_semantics_after_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A public-clone traceback in command output is observation, not taint.

    The immutable command still has to be scanned.  Renaming
    ``proposer_last.log`` to a generic ``*.log`` used to make the scanner parse
    the entire JSONL blob as prose, incorrectly treating a private field name
    in allowed command output as an agent-authored introspection request.
    """
    source_root = tmp_path / "agent_solutions"
    transcript = "".join(
        json.dumps(event, separators=(",", ":")) + "\n"
        for event in (
            {"type": "thread.started", "thread_id": "thread-test"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "python public_clone_probe.py",
                    "aggregated_output": (
                        "Traceback from public clone implementation: "
                        "copying env._game\n"
                    ),
                    "exit_code": 1,
                    "status": "failed",
                },
            },
            {"type": "turn.completed", "usage": {}},
        )
    )
    evidence = _write_legacy(source_root, transcript=transcript)
    candidate = C.boundary_candidates(
        source_root, game="zz99", level=1
    )[0]
    assert C.gkm_legs._file_taint_reason(
        str(candidate.transcript), candidate.transcript.name
    ) is None

    monkeypatch.setattr(
        C,
        "_select_and_replay",
        lambda *_args, **_kwargs: (candidate, [1]),
    )
    game_root = tmp_path / "stage" / "zz99_legs"
    (game_root / "promotion_evidence").mkdir(parents=True)
    C._certify_boundary(
        source_root=source_root,
        game_root=game_root,
        game="zz99",
        level=1,
        records=[{"level": 1, "marginal_C": 7, "reached": True}],
        parent_checkpoint_sha256=None,
        parent_manifest_sha256=None,
        time_cap=1,
        scanner_sha256="a" * 64,
        engine_sha256="b" * 64,
        hasher_sha256="c" * 64,
    )

    boundary = game_root / "promotion_evidence" / "level_01"
    retained = boundary / "transcripts" / "proposer_last.log"
    assert retained.read_bytes() == (
        evidence / "proposer_last.log"
    ).read_bytes()
    assert C.gkm_legs._file_taint_reason(
        str(retained), "transcripts/proposer_last.log"
    ) is None
    taint = json.loads((boundary / "audits" / "taint.json").read_text())
    assert taint["verdict"] == "PASS"
    assert taint["findings"] == []
    assert "transcripts/proposer_last.log" in taint["checked_files_sha256"]


def test_certifier_rejects_authored_private_command_before_copy(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "agent_solutions"
    transcript = "".join(
        json.dumps(event, separators=(",", ":")) + "\n"
        for event in (
            {"type": "thread.started", "thread_id": "thread-test"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "python -c 'print(env._game)'",
                    "aggregated_output": "",
                    "exit_code": 0,
                    "status": "completed",
                },
            },
            {"type": "turn.completed", "usage": {}},
        )
    )
    _write_legacy(source_root, transcript=transcript)
    candidate = C.boundary_candidates(
        source_root, game="zz99", level=1
    )[0]

    assert C._source_taint_reason(candidate) == (
        "private game/runtime introspection in proposer_last.log"
    )
    with pytest.raises(
        C.CertificationError,
        match="private game/runtime introspection",
    ):
        C._select_and_replay(
            source_root, game="zz99", level=1, time_cap=1
        )


@pytest.mark.parametrize(
    "hidden_record",
    (
        {
            "type": "future.command",
            "command": "python -c 'print(env._game)'",
        },
        {
            "type": "item.completed",
            "item": {
                "type": "future_command",
                "command": "python -c 'print(env._game)'",
            },
        },
        ["python -c 'print(env._game)'"],
    ),
)
def test_certifier_rejects_private_command_in_unrecognized_json_record(
    tmp_path: Path,
    hidden_record: object,
) -> None:
    source_root = tmp_path / "agent_solutions"
    transcript = json.dumps(
        hidden_record, separators=(",", ":")
    ) + "\n"
    _write_legacy(source_root, transcript=transcript)
    candidate = C.boundary_candidates(
        source_root, game="zz99", level=1
    )[0]

    assert C._source_taint_reason(candidate) == (
        "private game/runtime introspection in proposer_last.log"
    )


def test_plan_reports_only_true_missing_boundaries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "agent_solutions"
    _write_wip(source_root, game="aa11", level=1)
    monkeypatch.setattr(
        C.release_gate,
        "_discover_inventory",
        lambda _root: ({"aa11": 1, "bb22": 1}, {}),
    )

    result = C.plan_migration(
        source_root=source_root,
        environments_root=tmp_path / "environment_files",
    )

    assert result["status"] == "INCOMPLETE"
    assert result["summary"]["with_candidate"] == 1
    assert result["summary"]["missing_candidate"] == 1
    missing = [
        (row["game"], row["level"])
        for row in result["boundaries"]
        if row["candidate_count"] == 0
    ]
    assert missing == [("bb22", 1)]


def test_exact_path_reconstruction_is_last_resort(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "agent_solutions"
    game_root = root / "zz99_legs"
    game_root.mkdir(parents=True)
    (game_root / "checkpoint.json").write_text(json.dumps({
        "game": "zz99",
        "reached": 1,
        "validated": True,
        "final_path": [1, [6, 2, 3]],
    }))
    monkeypatch.setattr(
        C.gkm_legs,
        "exact_level_boundary",
        lambda *_args, **_kwargs: [1, [6, 2, 3]],
    )
    monkeypatch.setattr(
        C.gkm_arena,
        "validate",
        lambda *_args, **_kwargs: True,
    )

    candidates = C.boundary_candidates(root, game="zz99", level=1)
    candidate = candidates[-1]
    payloads, origins = C._source_payloads(candidate)

    assert candidate.kind == "exact_path_reconstruction"
    assert candidate.historical_source_boundary is False
    assert candidate.deterministic_reconstruction is True
    assert candidate.priority[0] == C.PHASE_PRIORITY[
        "deterministic_exact_path_reconstruction"
    ]
    assert b"EXACT_PATH = [1, [6, 2, 3]]" in payloads["legs.py"]
    assert set(payloads) == set(C.REQUIRED_SOURCES)
    assert set(origins.values()) == {
        (game_root / "checkpoint.json").as_posix()
    }


def test_partial_plan_binds_claimed_prefix_and_explicit_suffix_gaps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "agent_solutions"
    for game in ("aa11", "bb22"):
        game_root = source_root / f"{game}_legs"
        game_root.mkdir(parents=True)
        (game_root / "checkpoint.json").write_text(json.dumps({
            "game": game,
            "reached": 1,
            "validated": True,
            "records": [
                {"level": 1, "marginal_C": 3, "reached": True}
            ],
            "total_marginal_C": 3,
            "final_path": [1],
        }))
        _write_wip(source_root, game=game, level=1)
    monkeypatch.setattr(
        C.release_gate,
        "_discover_inventory",
        lambda _root: ({"aa11": 2, "bb22": 2}, {}),
    )
    monkeypatch.setattr(
        C.gkm_legs,
        "exact_level_boundary",
        lambda *_args, **_kwargs: None,
    )

    result = C.plan_partial_migration(
        source_root=source_root,
        environments_root=tmp_path / "environment_files",
        expected_claimed_levels=2,
    )

    assert result["status"] == "PASS"
    assert result["claimed_inventory"] == {"aa11": 1, "bb22": 1}
    assert result["claimed_levels"] == 2
    assert result["authoritative_levels"] == 4
    assert result["unclaimed_boundaries"] == [
        {"game": "aa11", "level": 2},
        {"game": "bb22", "level": 2},
    ]
    assert result["summary"]["historical_source_boundary"] == 2
    assert result["summary"]["retrospective_certification"] == 0

    with pytest.raises(
        C.CertificationError,
        match="frontier count mismatch",
    ):
        C.plan_partial_migration(
            source_root=source_root,
            environments_root=tmp_path / "environment_files",
            expected_claimed_levels=1,
        )
