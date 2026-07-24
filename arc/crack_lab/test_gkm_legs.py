"""Offline unit tests for the enforced leg-library orchestration (gkm_legs.py).

No LLM, no game: the proposer and verifier are mocked so the control loop and the
MARGINAL free-energy accounting can be checked without credits. The point being
tested is the load-bearing property: reusing a leg is free, so later levels that add
no new legs have lower marginal novelty than early rule-learning levels.
"""
import os
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
import gkm_legs as L


def test_loc_ignores_blanks_and_comments():
    assert L._loc("def f():\n    return 1\n") == 2
    assert L._loc("\n# comment\n   \n") == 0


def test_marginal_complexity_reuse_is_free():
    legs = "def a(env):\n    pass\n"
    # reused leg (legs unchanged) + a 1-line player -> marginal C counts only the player
    assert L.marginal_complexity(legs, legs, "", "play(env)\n") == 1
    # adding a new leg is paid for
    legs2 = legs + "def b(env):\n    pass\n"
    assert L.marginal_complexity(legs, legs2, "", "") == 2


def test_marginal_complexity_nets_replacement_within_each_file():
    before = "old_call(env)\n"
    after = "new_call(env)\n"
    assert L.description_complexity(before) == L.description_complexity(after)
    assert L.marginal_complexity(before, after, "", "") == 0


def test_free_energy_rewards_levels_and_penalises_novelty():
    assert L.free_energy(3, 0) == -3.0
    assert L.free_energy(3, 100, lam=0.02) == -3.0 + 2.0
    # more levels for the same novelty is always lower F
    assert L.free_energy(4, 50) < L.free_energy(3, 50)


def test_level_record_upsert_and_legacy_checkpoint_deduplication(tmp_path):
    rep = L.Report(
        game="duptest",
        reached=3,
        records=[
            L.LevelRecord(level=1, marginal_C=10, reached=True),
            L.LevelRecord(level=3, marginal_C=14, reached=True),
            L.LevelRecord(level=3, marginal_C=184, reached=True),
        ],
        total_marginal_C=208,
    )
    L._save_checkpoint(str(tmp_path), rep)
    assert [(r.level, r.marginal_C) for r in rep.records] == [(1, 10), (3, 184)]
    assert rep.total_marginal_C == 194

    data = json.loads((tmp_path / L.CHECKPOINT_FILE).read_text())
    assert [(r["level"], r["marginal_C"]) for r in data["records"]] == [
        (1, 10), (3, 184)
    ]

    L._record_level(rep, 3, 190)
    assert [(r.level, r.marginal_C) for r in rep.records] == [(1, 10), (3, 190)]
    assert rep.total_marginal_C == 200


def test_checkpoint_normalizes_stale_total_from_unique_records(tmp_path):
    checkpoint = {
        "game": "staletotal",
        "reached": 2,
        "records": [
            {"level": 1, "marginal_C": 40, "reached": True},
            {"level": 2, "marginal_C": 7, "reached": True},
        ],
        "total_marginal_C": 12,
        "final_path": [1, 2],
        "validated": True,
    }
    (tmp_path / L.CHECKPOINT_FILE).write_text(json.dumps(checkpoint))

    rep = L._load_checkpoint(str(tmp_path))
    assert rep.total_marginal_C == 47

    L._save_checkpoint(str(tmp_path), rep)
    saved = json.loads((tmp_path / L.CHECKPOINT_FILE).read_text())
    assert saved["total_marginal_C"] == 47


def test_repository_promoted_artifacts_are_clean_and_consistent():
    artifacts = Path(__file__).with_name("agent_solutions")
    checked = 0
    for artifact in sorted(artifacts.glob("*_legs")):
        checkpoint_path = artifact / L.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            continue
        checkpoint = json.loads(checkpoint_path.read_text())
        records = checkpoint["records"]
        levels = [record["level"] for record in records]
        assert L.promoted_artifact_taint_reason(str(artifact)) is None
        assert checkpoint["validated"] is True
        assert checkpoint["final_path"]
        assert len(levels) == len(set(levels))
        assert checkpoint["total_marginal_C"] == sum(
            record["marginal_C"] for record in records
        )
        checked += 1
    assert checked >= 8


def test_workspace_lock_rejects_overlapping_orchestrator(tmp_path):
    first = L._acquire_workspace_lock(str(tmp_path))
    try:
        try:
            L._acquire_workspace_lock(str(tmp_path))
        except RuntimeError as ex:
            assert "another orchestrator" in str(ex)
        else:
            raise AssertionError("overlapping workspace lock was accepted")

        code = (
            "import gkm_legs as L\n"
            "try:\n"
            f"    L._acquire_workspace_lock({str(tmp_path)!r})\n"
            "except RuntimeError:\n"
            "    print('BLOCKED')\n"
            "else:\n"
            "    raise SystemExit('overlap accepted')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(filter(None, (
                    os.path.dirname(L.__file__), os.environ.get("PYTHONPATH", "")
                ))),
            },
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "BLOCKED"
    finally:
        L._release_workspace_lock(first)


def test_codex_command_is_explicitly_metered_and_sandboxed(tmp_path):
    cmd = L._codex_command(str(tmp_path), "do the bounded task", None, "medium")
    joined = " ".join(cmd)
    assert cmd[:2] == ["codex", "exec"]
    assert "--json" in cmd
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--strict-config" in cmd
    assert "--model gpt-5.6-sol" in joined
    assert 'model_reasoning_effort="medium"' in cmd
    assert 'web_search="disabled"' in cmd
    assert "sandbox_workspace_write.network_access=false" in cmd
    assert 'approval_policy="never"' in cmd
    assert "--sandbox workspace-write" in joined
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd
    assert "--add-dir" not in cmd

    try:
        L._codex_command(str(tmp_path), "task", None, "xhigh")
    except ValueError as ex:
        assert "medium" in str(ex) and "high" in str(ex)
    else:
        raise AssertionError("unsupported Codex effort was accepted")


def test_codex_environment_does_not_forward_api_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak-either")
    monkeypatch.setenv("PATH", "/test/bin")
    env = L._codex_environment()
    assert env["PATH"] == "/test/bin"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_codex_json_usage_parser_uses_turn_completed_only(tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_text(
        json.dumps({"type": "thread.started", "thread_id": "thread-1"}) + "\n" +
        "diagnostic that is not JSON\n" +
        json.dumps({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 70,
                "output_tokens": 30,
                "reasoning_output_tokens": 20,
            },
        }) + "\n"
    )
    usage = L._codex_usage_from_jsonl(str(log))
    assert usage == {
        "thread_id": "thread-1",
        "input_tokens": 100,
        "cached_input_tokens": 70,
        "output_tokens": 30,
        "reasoning_output_tokens": 20,
        "usage_reported": True,
        "observed_tokens": 130,
    }


def test_codex_agent_records_offline_fake_turn(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    ledger = tmp_path / "usage.jsonl"
    reset = 1_800_000_000
    snapshot = {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "plus",
                "secondary": {
                    "usedPercent": 6,
                    "resetsAt": reset,
                    "windowDurationMins": 10_080,
                },
            }
        }
    }

    monkeypatch.setattr(L.CUG, "query_rate_limits", lambda: snapshot)
    monkeypatch.setattr(
        L,
        "_codex_command",
        lambda ws, task, model, effort: [
            sys.executable,
            "-c",
            (
                "import json; "
                "print(json.dumps({'type':'thread.started','thread_id':'fake'})); "
                "print(json.dumps({'type':'turn.completed','usage':"
                "{'input_tokens':80,'cached_input_tokens':50,'output_tokens':20,"
                "'reasoning_output_tokens':12}}))"
            ),
        ],
    )

    record = L._codex_agent(
        str(ws),
        "offline fake task",
        None,
        1,
        reasoning_effort="medium",
        weekly_reserve=90,
        max_campaign_tokens=1_000,
        max_campaign_runs=1,
        ledger_path=str(ledger),
        run_label="fake:L1:propose",
    )
    assert record["returncode"] == 0
    assert record["thread_id"] == "fake"
    assert record["observed_tokens"] == 100
    assert record["weekly_remaining_before"] == 94
    assert record["weekly_remaining_after"] == 94
    assert record["reasoning_effort"] == "medium"
    assert json.loads(ledger.read_text())["run_label"] == "fake:L1:propose"
    immutable = ws / record["transcript"]
    assert immutable.is_file()
    assert (ws / "proposer_last.log").read_bytes() == immutable.read_bytes()

    L._record_codex_level_outcome(
        record,
        ledger_path=str(ledger),
        game="fake",
        level=1,
        reached_before=0,
        reached_after=1,
        path=[1, 2],
        marginal_C=17,
    )
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert rows[1]["event"] == "codex_level_outcome"
    assert rows[1]["thread_id"] == "fake"
    assert rows[1]["solved_target"] is True
    assert rows[1]["winning_marginal_C"] == 17


def test_orchestration_loop_with_mocks_shows_reuse_trend(tmp_path, monkeypatch):
    """Drive the loop with a mock proposer that INVENTS legs on L1-L2 (learning the
    rules) and REUSES them on L3-L4 (no new legs). Marginal novelty must drop."""
    def mock_propose(ws, K):
        with open(os.path.join(ws, "players.py"), "a") as f:
            f.write(f"\n\ndef play_level_{K}(env):\n    leg_1(env)\n    leg_2(env) if {K} > 1 else None\n")
        if K <= 2:  # early levels: invent a new leg
            with open(os.path.join(ws, "legs.py"), "a") as f:
                f.write(f"\n\ndef leg_{K}(env):\n    for _ in range(3):\n        pass\n")

    def mock_verify(game, solve_path):
        players = open(os.path.join(os.path.dirname(solve_path), "players.py")).read()
        n = len(re.findall(r"def play_level_\d+", players))
        return (n, [], None)   # empty path => real A.validate is skipped

    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    # isolated, clean workspace (fake game name; mocks ignore the real game)
    shutil.rmtree(os.path.join(L.SCRATCH, "gkm_legs_ws_legstest"), ignore_errors=True)
    rep = L.orchestrate("legstest", max_level=4, propose_fn=mock_propose,
                        verify_fn=mock_verify, debrief_fn=lambda w, k: None,
                        verbose=False)
    assert rep.reached == 4
    by = {r.level: r.marginal_C for r in rep.records}
    assert set(by) == {1, 2, 3, 4}
    # L1 invents a leg; L2-L4 reuse with auto-solve (player-stub-only marginal cost)
    assert by[3] < by[1]                        # reuse cheaper than invention
    assert by[4] <= by[2]                       # not strictly increasing
    assert by[3] == by[4]                       # pure reuse: identical marginal cost
    assert rep.total_marginal_C == sum(by.values())
    level3_wip = artifact_root / "legstest_legs" / "wip_context" / "level_03"
    assert any(p.name.startswith("after_auto_solve_debrief_") for p in level3_wip.iterdir())


def test_setup_workspace_builds_valid_dispatch():
    ws = L.setup_workspace("wa30")
    for f in ("legs.py", "players.py", "solve.py", "gkm_try.py", "legs_log.md", "perception.py"):
        assert os.path.exists(os.path.join(ws, f))
    import ast
    ast.parse(open(os.path.join(ws, "solve.py")).read())   # solve.py is valid Python
    ast.parse(open(os.path.join(ws, "perception.py")).read())


def test_codex_workspace_has_local_git_boundary(tmp_path):
    ws = tmp_path / "clean-room"
    ws.mkdir()
    for name, body in {
        "gkm_try.py": "print('ok')\n",
        "legs.py": "def leg(env):\n    pass\n",
        "players.py": "from legs import *\n",
        "solve.py": "def solve(env):\n    pass\n",
        "legs_log.md": "# local\n",
    }.items():
        (ws / name).write_text(body)

    L._initialize_codex_workspace_git(str(ws))
    top = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    assert Path(top).resolve() == ws.resolve()

    (ws / "legs.py").write_text("def leg(env):\n    return 1\n")
    diff = subprocess.run(
        ["git", "-C", str(ws), "diff", "--", "legs.py"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    assert "return 1" in diff
    assert L._workspace_taint_reason(str(ws)) is None


def test_solver_source_index_is_compact_and_navigable(tmp_path):
    (tmp_path / "legs.py").write_text(
        "def reusable_leg(env, steps=3):\n"
        "    \"\"\"Move through a bounded number of observed states.\"\"\"\n"
        "    for _ in range(steps):\n"
        "        observe(env)\n"
        "\n"
        "def other(env):\n"
        "    return reusable_leg(env)\n"
    )
    (tmp_path / "players.py").write_text(
        "def play_level_1(env):\n"
        "    reusable_leg(env)\n"
    )
    index = L._solver_source_index(str(tmp_path))
    assert "## legs.py" in index
    assert "L1--4" in index
    assert "`def reusable_leg(env, steps=3):`" in index
    assert "Move through a bounded number" in index
    assert "calls: observe, range" in index
    assert "for _ in range(steps)" not in index

    path = L._write_solver_source_index(str(tmp_path))
    assert Path(path).read_text() == index


def test_replay_harness_refuses_tainted_workspace():
    ws = L.setup_workspace("wa30", tag="taintrepro")
    with open(os.path.join(ws, "proposer_last.log"), "w") as f:
        f.write("cat environment_files/wa30/wa30.py\n")

    proc = subprocess.run(
        [sys.executable, "gkm_try.py"],
        cwd=ws,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode != 0
    assert "TAINTED WORKSPACE" in proc.stderr


def test_perception_seed_extracts_components_and_deltas(tmp_path):
    ws = L.setup_workspace("perceptiontest")
    import importlib.util
    spec = importlib.util.spec_from_file_location("perception", os.path.join(ws, "perception.py"))
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)

    import numpy as np
    frame = np.zeros((6, 6), dtype=int)
    frame[1:3, 1:3] = 4
    frame[4, 4] = 9
    blobs = P.connected_components(frame, colors=[4, 9])
    assert [(b.color, b.bbox, b.area) for b in blobs] == [
        (4, (1, 1, 2, 2), 4),
        (9, (4, 4, 4, 4), 1),
    ]
    after = frame.copy()
    after[2, 2] = 7
    delta = P.frame_delta(frame, after)
    assert delta["count"] == 1
    assert delta["bbox"] == (2, 2, 2, 2)

    class TinyEnv:
        def __init__(self, state=0):
            self.state = state

        def clone(self):
            return TinyEnv(self.state)

        def step(self, action, *coords):
            self.state += int(action)

        def frame(self):
            return np.asarray([[self.state]], dtype=int)

        def terminal(self):
            return False

    path = P.bounded_replay_bfs(
        TinyEnv(),
        goal_fn=lambda env, path: env.state >= 3,
        action_fn=lambda env: [1],
        max_states=10,
        max_depth=5,
    )
    assert path == [1, 1, 1]


def test_promote_and_seed_verified_artifact(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    ws = tmp_path / "ws"
    ws.mkdir()
    for name, body in {
        "legs.py": "def leg(env):\n    pass\n",
        "players.py": "from legs import *\n\ndef play_level_1(env):\n    leg(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
        "legs_log.md": "# log\n",
    }.items():
        (ws / name).write_text(body)
    codex_turn = ws / "codex_turn_20260722T000000000000Z_test.jsonl"
    codex_turn.write_text(json.dumps({"type": "thread.started", "thread_id": "clean"}) + "\n")

    rep = L.Report(
        game="artifacttest",
        reached=1,
        records=[L.LevelRecord(level=1, marginal_C=3, reached=True)],
        total_marginal_C=3,
        final_path=[1, 2, 3],
        validated=True,
    )
    assert L.promote_verified_artifact("artifacttest", str(ws), rep, verbose=False)
    assert (artifact_root / "artifacttest_legs" / "README.md").exists()
    manifest = json.loads(
        (artifact_root / "artifacttest_legs" / "promotion_evidence" /
         "level_01" / "manifest.json").read_text()
    )
    assert len(manifest["codex_transcripts"]) == 1
    evidence_turn = (
        artifact_root / "artifacttest_legs" / "promotion_evidence" /
        "level_01" / manifest["codex_transcripts"][0]["path"]
    )
    assert evidence_turn.read_bytes() == codex_turn.read_bytes()

    (ws / "players.py").write_text("contaminated unfinished edit\n")
    seeded = L.seed_workspace_from_artifact("artifacttest", str(ws), verbose=False)
    assert seeded is not None and seeded.reached == 1
    assert "play_level_1" in (ws / "players.py").read_text()


def test_tainted_workspace_cannot_promote_artifact(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    ws = tmp_path / "ws"
    ws.mkdir()
    for name, body in {
        "legs.py": "def leg(env):\n    pass\n",
        "players.py": "from legs import *\n\ndef play_level_1(env):\n    leg(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
        "legs_log.md": "# log\n",
        "proposer_last.log": "sed -n '1,80p' environment_files/wa30/x/wa30.py\n",
    }.items():
        (ws / name).write_text(body)

    rep = L.Report(
        game="tainttest",
        reached=1,
        records=[L.LevelRecord(level=1, marginal_C=1, reached=True)],
        total_marginal_C=1,
        final_path=[1],
        validated=True,
    )
    try:
        L.promote_verified_artifact("tainttest", str(ws), rep, verbose=False)
    except L.WorkspaceTainted as ex:
        assert "forbidden source/history access" in str(ex)
    else:
        raise AssertionError("tainted workspace promoted")
    assert not (artifact_root / "tainttest_legs" / "checkpoint.json").exists()


def test_private_runtime_introspection_taints_workspace(tmp_path):
    (tmp_path / "probe.py").write_text("print(env._game)\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "private game/runtime introspection" in reason


def test_oversized_taint_evidence_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "MAX_TAINT_SCAN_BYTES", 10)
    path = tmp_path / "proposer_last.log"
    path.write_text("x" * 11)
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "oversized unscanned evidence" in reason


def test_runtime_enumeration_and_frame_data_taint_workspace(tmp_path):
    for index, body in enumerate(("print(vars(env))\n", "print(env._fd)\n")):
        path = tmp_path / f"probe_{index}.py"
        path.write_text(body)
        reason = L._workspace_taint_reason(str(tmp_path))
        assert reason is not None
        assert "private game/runtime introspection" in reason
        path.unlink()


def test_other_catalog_game_source_taints_workspace(tmp_path):
    (tmp_path / "proposer_last.log").write_text("find .. -name 'g50t.py'\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "g50t.py" in reason


def test_external_web_or_network_attempt_taints_workspace(tmp_path):
    attempts = (
        "curl https://example.com/public-scorecard.json\n",
        "python3 -c \"import requests; requests.get('https://example.com')\"\n",
        "use web_search to find the game\n",
        "import socket\nsocket.create_connection(('1.1.1.1', 443))\n",
    )
    for index, body in enumerate(attempts):
        path = tmp_path / f"network_attempt_{index}.txt"
        path.write_text(body)
        reason = L._workspace_taint_reason(str(tmp_path))
        assert reason is not None
        assert "external web/network access" in reason
        path.unlink()


def test_loopback_reference_is_not_network_taint(tmp_path):
    (tmp_path / "client_note.txt").write_text(
        "The local service is http://127.0.0.1:8879/game/current\n"
    )
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_blocked_attempt_ledger_is_audit_evidence_not_execution_taint(tmp_path):
    (tmp_path / L.BLOCKED_ATTEMPTS_LOG).write_text(
        "bash: 'python3 -c \\\"print(env._game)\\\"'\n"
    )
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_adaptive_debrief_skips_literal_reuse_and_small_acquisitions():
    assert not L.should_run_debrief(
        "adaptive", auto_solved=True, pre_debrief_marginal_C=999
    )
    assert not L.should_run_debrief(
        "adaptive", auto_solved=False, pre_debrief_marginal_C=149
    )
    assert L.should_run_debrief(
        "adaptive", auto_solved=False, pre_debrief_marginal_C=150
    )
    assert L.should_run_debrief(
        "always", auto_solved=True, pre_debrief_marginal_C=0
    )
    assert not L.should_run_debrief(
        "never", auto_solved=False, pre_debrief_marginal_C=999
    )


def test_debrief_inline_code_mention_is_not_execution_taint(tmp_path):
    (tmp_path / "proposer_last.log").write_text(
        "The blocked `dir(legs)` command was recorded in the ledger.\n"
    )
    assert L._workspace_taint_reason(str(tmp_path)) is None

    (tmp_path / "probe.py").write_text("print(vars(env))\n")
    assert "private game/runtime introspection" in L._workspace_taint_reason(str(tmp_path))


def test_public_clone_traceback_private_field_is_not_agent_taint(tmp_path):
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "python probe.py",
                "aggregated_output": (
                    "Traceback: clone() -> self._game = "
                    "copy.deepcopy(_clone._game)\n"
                ),
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "clone failed"},
        },
    ]
    (tmp_path / "proposer_last.log").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    (tmp_path / "probe.py").write_text("clone = env.clone()\n")
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_codex_jsonl_private_command_still_taints_workspace(tmp_path):
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "python -c 'print(env._game)'",
            "aggregated_output": "anything\n",
        },
    }
    (tmp_path / "proposer_last.log").write_text(json.dumps(event) + "\n")
    reason = L._workspace_taint_reason(str(tmp_path))
    assert reason is not None
    assert "private game/runtime introspection" in reason


def test_public_harness_api_introspection_is_allowed(tmp_path):
    (tmp_path / "probe.py").write_text(
        "import inspect, gkm_arena as A\n"
        "print(dir(A), dir(A.Arena), dir(env))\n"
        "print(inspect.getsource(A.Arena.clone))\n"
    )
    assert L._workspace_taint_reason(str(tmp_path)) is None


def test_promoted_artifact_scan_excludes_forensic_wip(tmp_path):
    for name in L.PROMOTED_FILES:
        (tmp_path / name).write_text("clean promoted evidence\n")
    dirty = tmp_path / "wip_context" / "level_01" / "attempt" / "files"
    dirty.mkdir(parents=True)
    (dirty / "probe.py").write_text("print(env._game)\n")

    assert L._workspace_taint_reason(str(tmp_path)) is not None
    assert L.promoted_artifact_taint_reason(str(tmp_path)) is None

    (tmp_path / "legs.py").write_text("print(env._game)\n")
    assert "legs.py" in L.promoted_artifact_taint_reason(str(tmp_path))


def test_action_path_accepts_coordinate_clicks_without_changing_key_paths():
    assert L._load_action_path([1, 5, 2]) == [1, 5, 2]
    assert L._load_action_path([[6, 12, 34], [6, 0, 63]]) == [
        [6, 12, 34], [6, 0, 63]
    ]
    assert L._load_action_path([[5, 12, 34]]) is None
    assert L._action_path_key([1, [6, 12, 34], 5]) == (1, (6, 12, 34), 5)


def test_wip_context_snapshot_is_artifact_visible(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    ws = tmp_path / "ws"
    ws.mkdir()
    for name, body in {
        "legs.py": "def old_leg(env):\n    pass\n",
        "players.py": "from legs import *\n\ndef play_level_1(env):\n    old_leg(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
        "legs_log.md": "old leg context\n",
        "proposer_last.log": "fresh probe observation\n",
    }.items():
        (ws / name).write_text(body)
    rep = L.Report(
        game="snaptest",
        reached=1,
        records=[L.LevelRecord(level=1, marginal_C=1, reached=True)],
        total_marginal_C=1,
        final_path=[1],
        validated=True,
    )
    L._save_checkpoint(str(ws), rep)
    assert L.promote_verified_artifact("snaptest", str(ws), rep, verbose=False)
    snap = L.snapshot_wip_context("snaptest", str(ws), 2, "not_reached", 1, "probe failed", verbose=False)
    assert (artifact_root / "snaptest_legs" / "wip_context" / "level_02" / "latest.json").exists()
    assert "fresh probe observation" in (os.path.join(snap, "files", "proposer_last.log") and
                                         open(os.path.join(snap, "files", "proposer_last.log")).read())

    (ws / "players.py").write_text("contaminated unfinished edit\n")
    seeded = L.seed_workspace_from_artifact("snaptest", str(ws), verbose=False)
    assert seeded is not None and seeded.reached == 1
    assert "play_level_1" in (ws / "players.py").read_text()
    # WIP snapshots are forensic only: seeding must NOT inject probe context back
    # into the workspace (that stitching caused analysis paralysis; see FINDINGS).
    assert not (ws / "wip_context.md").exists()


def test_propose_task_is_minimal(tmp_path):
    """The proposer prompt is the known-good 7-sentence task; no artifact/probe
    context is stitched in (prompt bloat degraded the proposer; see FINDINGS)."""
    task = L._propose_task("ls20", 5, "raw substrate context", ["bfs_to_level_up"])

    assert "GOAL: make solve.py reach LEVEL 5" in task
    assert "bfs_to_level_up" in task
    assert "REUSE existing legs" in task
    assert "play_level_5" in task
    assert "python gkm_try.py" in task
    assert "VERIFIED ARTIFACT CONTEXT" not in task
    assert "wip" not in task.lower()


def test_tagged_workspace_uses_canonical_artifact_dir():
    assert L.artifact_dir("ls20") == L.artifact_dir("ls20", tag="continue")


def test_transient_proposer_failure_is_retried(tmp_path, monkeypatch):
    """A dropped-connection proposal (short log with an API error banner) must be
    retried instead of read as a capability failure; a genuine full-transcript
    failure must NOT be retried."""
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    calls = []

    def flaky_propose(ws, K):
        calls.append(K)
        if len(calls) == 1:  # first attempt: infrastructure failure, no work done
            with open(os.path.join(ws, "proposer_last.log"), "w") as f:
                f.write("API Error: Connection closed mid-response.\n")
            return
        with open(os.path.join(ws, "proposer_last.log"), "w") as f:
            f.write("wrote play_level_1 composing a new leg\n")
        with open(os.path.join(ws, "legs.py"), "a") as f:
            f.write("\n\ndef leg_1(env):\n    pass\n")
        with open(os.path.join(ws, "players.py"), "a") as f:
            f.write("\n\ndef play_level_1(env):\n    leg_1(env)\n")

    def mock_verify(game, solve_path):
        players = open(os.path.join(os.path.dirname(solve_path), "players.py")).read()
        n = len(re.findall(r"def play_level_\d+", players))
        return (n, [], None)

    shutil.rmtree(os.path.join(L.SCRATCH, "gkm_legs_ws_retrytest"), ignore_errors=True)
    rep = L.orchestrate("retrytest", max_level=1, propose_fn=flaky_propose,
                        verify_fn=mock_verify, debrief_fn=lambda w, k: None,
                        verbose=False)
    assert calls == [1, 1]          # retried once after the transient failure
    assert rep.reached == 1


def test_transient_retry_can_be_disabled_for_cost_bounded_campaign(
        tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        L, "artifact_dir",
        lambda game, tag="": str(artifact_root / f"{game}_legs"),
    )
    calls = []

    def dropped_propose(ws, K):
        calls.append(K)
        with open(os.path.join(ws, "proposer_last.log"), "w") as f:
            f.write("API Error: Connection closed mid-response.\n")

    def no_clear(game, solve_path):
        return 0, [], None

    shutil.rmtree(
        os.path.join(L.SCRATCH, "gkm_legs_ws_retryoff"), ignore_errors=True
    )
    rep = L.orchestrate(
        "retryoff", max_level=1, propose_fn=dropped_propose,
        verify_fn=no_clear, debrief_fn=lambda w, k: None,
        transient_retries=0, verbose=False,
    )
    assert calls == [1]
    assert rep.reached == 0


def test_transient_detector_requires_short_log(tmp_path):
    ws = tmp_path
    (ws / "proposer_last.log").write_text("API Error: Connection closed mid-response.\n")
    assert L._transient_proposer_failure(str(ws))
    (ws / "proposer_last.log").write_text("probing level...\n" * 200 + "api error once, recovered\n")
    assert not L._transient_proposer_failure(str(ws))


def test_noop_proposal_is_retried(tmp_path):
    """A proposer that signs off without touching any code (e.g. backgrounded its
    probe and exited) is a no-attempt: retry. A short log WITH code changes is a
    real (cheap) attempt: no retry."""
    (tmp_path / "proposer_last.log").write_text(
        "I'll stop here and wait for the background search to notify me.\n")
    assert L._transient_proposer_failure(str(tmp_path), code_changed=False)
    assert not L._transient_proposer_failure(str(tmp_path), code_changed=True)


def test_auto_solve_failure_recorded_and_skipped(tmp_path):
    ws = str(tmp_path)
    legs = "def solve_all(env):\n    pass\n"
    assert not L._auto_solve_failed_before(ws, 5, legs)
    L._record_auto_solve_failure(ws, 5, legs)
    assert L._auto_solve_failed_before(ws, 5, legs)
    # a changed library invalidates the negative record; other levels unaffected
    assert not L._auto_solve_failed_before(ws, 5, legs + "def new_leg(env):\n    pass\n")
    assert not L._auto_solve_failed_before(ws, 6, legs)


def test_seed_restores_wip_probes_without_clobbering(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    ws = tmp_path / "ws"
    ws.mkdir()
    for name, body in {
        "legs.py": "def leg(env):\n    pass\n",
        "players.py": "from legs import *\n\ndef play_level_1(env):\n    leg(env)\n",
        "solve.py": "def solve(env):\n    return None\n",
        "legs_log.md": "# log\n",
    }.items():
        (ws / name).write_text(body)
    rep = L.Report(game="probetest", reached=1,
                   records=[L.LevelRecord(level=1, marginal_C=1, reached=True)],
                   total_marginal_C=1, final_path=[1], validated=True)
    L._save_checkpoint(str(ws), rep)
    assert L.promote_verified_artifact("probetest", str(ws), rep, verbose=False)

    # an interrupted L2 attempt leaves probes + a candidate players.py in scratch
    (ws / "probe_l2.py").write_text("print('probe knowledge')\n")
    (ws / "players.py").write_text("UNVERIFIED candidate\n")
    L.snapshot_wip_context("probetest", str(ws), 2, "interrupted", 1, "killed", verbose=False)

    # scratch dies; a fresh seed must restore the probe but keep players.py verified
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    seeded = L.seed_workspace_from_artifact("probetest", str(ws2), verbose=False)
    assert seeded is not None and seeded.reached == 1
    assert (ws2 / "probe_l2.py").read_text() == "print('probe knowledge')\n"
    assert "play_level_1" in (ws2 / "players.py").read_text()  # verified, not candidate
    # a probe already present in scratch is NOT overwritten by the older snapshot
    (ws2 / "probe_l2.py").write_text("newer scratch state\n")
    L._restore_wip_probes("probetest", str(ws2), 2, verbose=False)
    assert (ws2 / "probe_l2.py").read_text() == "newer scratch state\n"


def test_seed_restores_level_one_wip_without_promoted_artifact(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    ws = tmp_path / "attempt"
    ws.mkdir()
    (ws / "probe_l1.py").write_text("print('mapped mechanic')\n")
    (ws / "players.py").write_text("UNVERIFIED candidate\n")
    L.snapshot_wip_context("unpromoted", str(ws), 1, "not_reached", 0,
                           "retry later", verbose=False)
    level_dir = artifact_root / "unpromoted_legs" / "wip_context" / "level_01"
    latest = json.loads((level_dir / "latest.json").read_text())["attempt"]
    cache = level_dir / latest / "files" / "__pycache__"
    cache.mkdir()
    (cache / "probe.pyc").write_bytes(b"cache")
    (level_dir / "frontier_scaffold.json").write_text(
        '{"version":"v2","created_at":"2026-07-24T00:00:00Z"}\n'
    )

    ws2 = tmp_path / "retry"
    ws2.mkdir()
    seeded = L.seed_workspace_from_artifact("unpromoted", str(ws2), verbose=False)
    assert seeded is None
    assert (ws2 / "probe_l1.py").read_text() == "print('mapped mechanic')\n"
    assert '"version":"v2"' in (ws2 / "frontier_scaffold.json").read_text()
    assert not (ws2 / "players.py").exists()


def test_frontier_brief_distills_agent_messages_and_probe_index(tmp_path):
    ws = tmp_path / "brief"
    ws.mkdir()
    events = [
        {"type": "item.completed", "item": {
            "type": "agent_message",
            "text": "Observed   a compact board transition.",
        }},
        {"type": "item.completed", "item": {
            "type": "command_execution",
            "aggregated_output": "RAW PIXELS MUST NOT ENTER THE BRIEF",
        }},
    ]
    (ws / "proposer_last.log").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    (ws / "focused_probe.py").write_text("print('probe')\n")
    (ws / "codex_turn_huge.jsonl").write_text("ignored\n")

    path = L._write_frontier_brief(str(ws), "briefgame", 2)
    assert path is not None
    text = (ws / "frontier_brief.md").read_text()
    assert "briefgame level 2" in text
    assert "Observed a compact board transition." in text
    assert "focused_probe.py" in text
    assert "RAW PIXELS" not in text
    assert "codex_turn_huge" not in text
    assert "unverified" in text.lower()


def test_frontier_brief_is_removed_when_no_prior_context(tmp_path):
    ws = tmp_path / "empty"
    ws.mkdir()
    (ws / "frontier_brief.md").write_text("stale")
    assert L._write_frontier_brief(str(ws), "empty", 1) is None
    assert not (ws / "frontier_brief.md").exists()


def test_generated_frontier_brief_is_covered_by_workspace_taint_gate(tmp_path):
    ws = tmp_path / "tainted_brief"
    ws.mkdir()
    event = {
        "type": "item.completed",
        "item": {
            "type": "agent_message",
            "text": "A traceback suggested reading env._game next.",
        },
    }
    (ws / "proposer_last.log").write_text(json.dumps(event) + "\n")
    L._write_frontier_brief(str(ws), "x", 1)
    assert "private game/runtime introspection" in L._workspace_taint_reason(str(ws))


def test_interrupt_snapshots_and_promotes(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(L, "artifact_dir", lambda game, tag="": str(artifact_root / f"{game}_legs"))

    def propose(ws, K):
        if K == 1:
            with open(os.path.join(ws, "players.py"), "a") as f:
                f.write(f"\n\ndef play_level_1(env):\n    pass\n")
        else:
            with open(os.path.join(ws, "probe_l2.py"), "w") as f:
                f.write("half-done probe\n")
            raise KeyboardInterrupt  # user hits Ctrl-C mid-L2

    def mock_verify(game, solve_path):
        players = open(os.path.join(os.path.dirname(solve_path), "players.py")).read()
        n = len(re.findall(r"def play_level_\d+", players))
        return (n, [1] * n, None)

    monkeypatch.setattr(L.A, "validate", lambda g, p, l: True)
    shutil.rmtree(os.path.join(L.SCRATCH, "gkm_legs_ws_inttest"), ignore_errors=True)
    import pytest
    with pytest.raises(KeyboardInterrupt):
        L.orchestrate("inttest", max_level=3, propose_fn=propose,
                      verify_fn=mock_verify, debrief_fn=lambda w, k: None,
                      verbose=False)
    # L1 was promoted despite the interrupt; the L2 probe context was snapshotted
    art = artifact_root / "inttest_legs"
    assert (art / "players.py").exists()
    level2 = art / "wip_context" / "level_02"
    assert level2.exists()
    import json as _json
    latest = _json.loads((level2 / "latest.json").read_text())
    assert latest["metadata"]["phase"] == "interrupted"
