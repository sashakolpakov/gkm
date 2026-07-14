"""Offline unit tests for the enforced leg-library orchestration (gkm_legs.py).

No LLM, no game: the proposer and verifier are mocked so the control loop and the
MARGINAL free-energy accounting can be checked without credits. The point being
tested is the load-bearing property: reusing a leg is free, so later levels that add
no new legs have lower marginal novelty than early rule-learning levels.
"""
import os
import re
import shutil
import subprocess
import sys
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


def test_setup_workspace_builds_valid_dispatch():
    ws = L.setup_workspace("wa30")
    for f in ("legs.py", "players.py", "solve.py", "gkm_try.py", "legs_log.md", "perception.py"):
        assert os.path.exists(os.path.join(ws, f))
    import ast
    ast.parse(open(os.path.join(ws, "solve.py")).read())   # solve.py is valid Python
    ast.parse(open(os.path.join(ws, "perception.py")).read())


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


def test_action_path_accepts_coordinate_clicks_without_changing_key_paths():
    assert L._load_action_path([1, 5, 2]) == [1, 5, 2]
    assert L._load_action_path([[6, 12, 34], [6, 0, 63]]) == [
        [6, 12, 34], [6, 0, 63]
    ]
    assert L._load_action_path([[5, 12, 34]]) is None


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
