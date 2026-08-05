"""Offline tests for the enforced predicate-library orchestration.

The proposer is injected; no LLM, no dataset, no network. Witness predicate
code lives only in these tests (representability floor, never shipped)."""
import copy
import json
import os
import stat
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bongard_arena as B
import bongard_legs as L
import phase_d_protocol as P
from test_bongard_arena import two_vs_one_problem, SQUARE, CIRCLE, _make_problem

WITNESS = "def p_ink(panel):\n    return float(panel.sum())\n"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect the artifact root into the sandbox so tests never touch the
    repo's agent_solutions."""
    monkeypatch.setattr(L, "artifact_dir",
                        lambda tag: str(tmp_path / f"art_{tag}"))
    return tmp_path


def _two_problems(seed=5):
    return [two_vs_one_problem(),
            _make_problem([[SQUARE, CIRCLE]] * 6,
                          [[CIRCLE]] * 3 + [[SQUARE]] * 3, seed=seed)]


def _basic_problems(seed=5):
    return [
        B.Problem(problem.problem_id, "basic", problem.concept,
                  problem.pos, problem.neg)
        for problem in _two_problems(seed)
    ]


def writing_proposer(code=WITNESS):
    def propose(task, ws, model, minutes):
        path = os.path.join(ws, L.LIBRARY_FILE)
        if code not in open(path).read():
            with open(path, "a") as f:
                f.write(code)
    return propose


def _codex_receipt(
        model=L.codex_headless.DEFAULT_CODEX_MODEL, *, task="fixture task",
        current_source=L.INITIAL_LIBRARY_SOURCE, current_log="",
        proposed_source=WITNESS, proposed_log="", rationale="fixture",
        panel_paths=None, panel_set_digest=None):
    policy = P.canonical_execution_policy(require_unrestricted_cli=True)
    unrestricted = policy["unrestricted"]
    runtime = policy["runtime"]["codex_cli"]
    thread_id = str(uuid.uuid4())
    if panel_paths is None:
        panel_view_digest = "1" * 64
        semantic_panel_digest = panel_set_digest or "sha256:" + "2" * 64
        input_digest = "3" * 64
    else:
        panel_view_digest = L.codex_headless.ordered_panel_view_digest(
            panel_paths)
        semantic_panel_digest = L.codex_headless.semantic_panel_set_digest(
            panel_paths)
        input_digest = L.codex_headless.predicate_proposer_input_digest(
            task, current_source, current_log, panel_paths)
    prompt = L.codex_headless._predicate_prompt(
        task, current_source, current_log)
    body = {
        "schema": P.PROPOSER_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": model,
        "reported_model": "",
        "model_identity_evidence": (
            "explicit-cli-model-flag;jsonl-omits-model"),
        "requested_reasoning_effort": unrestricted[
            "requested_reasoning_effort"],
        "input_tokens": 10,
        "cached_input_tokens": 0,
        "output_tokens": 2,
        "reasoning_output_tokens": 0,
        "thread_id": thread_id,
        "codex_cli_version": runtime["version"],
        "codex_launcher_digest": runtime["launcher_digest"],
        "task_digest": L._source_digest(task),
        "current_source_digest": L._source_digest(current_source),
        "current_log_digest": L._source_digest(current_log),
        "prompt_digest": L._source_digest(prompt),
        "input_digest_schema": L.codex_headless.PREDICATE_INPUT_DIGEST_SCHEMA,
        "input_digest": input_digest,
        "output_schema_digest": unrestricted[
            "proposer_output_schema_digest"],
        "panel_view_digest": panel_view_digest,
        "panel_set_digest": semantic_panel_digest,
        "structured_output_digest": (
            L.codex_headless.predicate_proposer_output_digest(
                proposed_source, proposed_log, rationale)),
        "proposed_source_digest": L._source_digest(proposed_source),
        "proposed_log_digest": L._source_digest(proposed_log),
        "event_stream_digest": L._source_digest("event:" + thread_id),
        "event_types": [
            "thread.started", "turn.started", "item.completed",
            "turn.completed"],
        "item_types": ["agent_message"],
        "isolation_policy": unrestricted["proposer_tool_surface"],
        "outcome": "success",
    }
    body["receipt_digest"] = L._canonical_digest(body)
    L._validate_proposer_receipt(body)
    return body


def writing_codex_proposer(code=WITNESS):
    def propose(task, ws, model, minutes):
        source_path = os.path.join(ws, L.LIBRARY_FILE)
        log_path = os.path.join(ws, L.LOG_FILE)
        current_source = open(source_path).read()
        current_log = open(log_path).read()
        if code not in current_source:
            with open(source_path, "a") as handle:
                handle.write(code)
        proposed_source = open(source_path).read()
        proposed_log = open(log_path).read()
        opaque_id = open(os.path.join(ws, "current_problem.txt")).read().strip()
        panel_paths = [
            os.path.join(ws, opaque_id, f"{side}_{index}.png")
            for side in ("pos", "neg") for index in range(6)
        ]
        rationale = "offline Codex fixture"
        return L.ProposerOutcome(
            rationale,
            _codex_receipt(
                model, task=task,
                current_source=current_source, current_log=current_log,
                proposed_source=proposed_source,
                proposed_log=proposed_log, rationale=rationale,
                panel_paths=panel_paths,
            ),
        )

    return propose


def test_solved_problems_and_reuse_is_free(sandbox):
    """First problem pays for the witness predicate; the second reuses it
    for marginal_C == 0 (the sawtooth's reuse floor)."""
    calls = []

    def propose(task, ws, model, minutes):
        calls.append(model)
        lib = os.path.join(ws, L.LIBRARY_FILE)
        if "p_ink" not in open(lib).read():
            with open(lib, "a") as f:
                f.write(WITNESS)

    rep = L.run(_two_problems(), tag="t1", ws=str(sandbox / "ws1"),
                propose_fn=propose, verbose=False)
    assert rep.solved == 2
    assert rep.records[0].marginal_C > 0
    assert rep.records[1].marginal_C == 0
    assert rep.records[0].definition_charge > 0
    assert rep.records[1].definition_charge == 0
    assert rep.records[0].structure_charge == B.CALL_COST + B.BINDING_COST
    assert rep.records[1].structure_charge == B.CALL_COST + B.BINDING_COST
    assert rep.paid_node_identities == sorted(
        node["identity"] for node in rep.records[0].used_definition_nodes)
    assert rep.source_trace_digest
    assert calls == [L.codex_headless.DEFAULT_CODEX_MODEL] * 2
    assert not any(r.escalated for r in rep.records)


def test_failed_attempt_reverts_library_and_saves_wip(sandbox):
    """Structural admission: junk written during an unsolved problem must not
    enter the shared library, but survives as WIP context."""
    def propose(task, ws, model, minutes):
        with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
            f.write("def p_useless(panel):\n    return 0.0\n")

    ws = str(sandbox / "ws2")
    rep = L.run([two_vs_one_problem()], tag="t2", ws=ws,
                propose_fn=propose, verbose=False)
    assert rep.solved == 0
    assert rep.records[0].attempts == len(L.DEFAULT_LADDER)
    assert rep.records[0].escalated
    assert "p_useless" not in open(os.path.join(ws, L.LIBRARY_FILE)).read()
    wip = os.path.join(L.artifact_dir("t2"), "wip_context", "problem_00")
    assert os.path.isdir(wip) and os.listdir(wip)


def test_escalation_ladder_logged(sandbox):
    """Sonnet fails, Opus succeeds -> escalated=True, model=opus."""
    def propose(task, ws, model, minutes):
        if model == "opus":
            with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
                f.write(WITNESS)

    rep = L.run([two_vs_one_problem()], tag="t3", ws=str(sandbox / "ws3"),
                propose_fn=propose, ladder=("sonnet", "opus"), verbose=False)
    assert rep.solved == 1
    assert rep.records[0].model == "opus"
    assert rep.records[0].escalated


def test_taint_refuses_promotion(sandbox):
    def propose(task, ws, model, minutes):
        with open(os.path.join(ws, "notes.md"), "w") as f:
            f.write("peeked at human_designed_shapes.tsv for the answer")
        with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
            f.write(WITNESS)

    with pytest.raises(L.WorkspaceTainted):
        L.run([two_vs_one_problem()], tag="t4", ws=str(sandbox / "ws4"),
              propose_fn=propose, verbose=False)


def test_resume_skips_solved_and_ground_truth_stays_out_of_ws(sandbox):
    ws = str(sandbox / "ws5")
    rep1 = L.run(_two_problems(), tag="t5", ws=ws,
                 propose_fn=writing_proposer(), verbose=False)
    assert rep1.solved == 2
    # workspace must not contain concept names or results.json
    for root, _dirs, files in os.walk(ws):
        assert "results.json" not in files
        for name in files:
            if name.endswith((".py", ".md", ".json", ".txt")):
                text = open(os.path.join(root, name)).read()
                assert "two_shapes_vs_one" not in text
    art = L.artifact_dir("t5")
    results = json.load(open(os.path.join(art, "results.json")))
    assert results["problem_00"]["concept"] == "two_shapes_vs_one"

    def must_not_be_called(task, ws_, model, minutes):
        raise AssertionError("solved problems must not re-run the proposer")

    rep2 = L.run(_two_problems(), tag="t5", ws=str(sandbox / "ws5b"),
                 propose_fn=must_not_be_called, verbose=False)
    assert rep2.solved == 2


def test_literal_cost_charges_lookup_tables():
    honest = "def p_a(panel):\n    return float(panel.sum())\n"
    table = "def p_a(panel):\n    return T[hash(panel.tobytes()) % 12]\nT = [" \
            + ", ".join(["1.0"] * 12) + "]\n"
    assert L.description_complexity(table) > L.description_complexity(honest)
    dense = f"TOKEN = {'x' * 512!r}\n" + honest
    assert L.description_complexity(dense) > \
        L.description_complexity(honest) + 20
    call_table = (
        "def p_a(panel):\n    return dict("
        + ", ".join(f"k{index}=0" for index in range(1000))
        + ")[\"k0\"]\n")
    assert L.description_complexity(call_table) > \
        L.description_complexity(honest) + 900


def test_interrupted_workspace_is_snapshotted_before_seed(sandbox):
    """Power-out fallback: an in-flight library edit that differs from the
    promoted artifact must be preserved as WIP, not silently overwritten."""
    ws = str(sandbox / "ws6")
    rep1 = L.run([two_vs_one_problem()], tag="t6", ws=ws,
                 propose_fn=writing_proposer(), verbose=False)
    assert rep1.solved == 1
    # simulate an interrupted next attempt: live edits + current problem marker
    with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
        f.write("def p_inflight(panel):\n    return 1.0\n")
    with open(os.path.join(ws, "current_problem.txt"), "w") as f:
        f.write("problem_01")
    L.seed_workspace_from_artifact("t6", ws, verbose=False)
    # workspace was restored to the verified artifact...
    assert "p_inflight" not in open(os.path.join(ws, L.LIBRARY_FILE)).read()
    # ...and the in-flight state survives as a WIP snapshot
    wip = os.path.join(L.artifact_dir("t6"), "wip_context",
                       "interrupted_problem_01")
    snaps = [os.path.join(wip, d) for d in os.listdir(wip)]
    assert any("p_inflight" in open(os.path.join(s, L.LIBRARY_FILE)).read()
               for s in snaps)


def test_interleave_corpus_stable_prefix():
    basic = [f"b{i}" for i in range(12)]
    abstract = [f"a{i}" for i in range(3)]
    full = L.interleave_corpus(basic, abstract)
    assert len(full) == 15
    assert full[4] == "a0" and full[9] == "a1" and full[14] == "a2"
    # stable prefix: truncating the corpus never reorders earlier slots
    assert L.interleave_corpus(basic, abstract)[:8] == full[:8]


def test_infra_failure_waits_then_stops_resumably(sandbox):
    """Session-limit/credit-out guardrail: an infra failure must not consume
    ladder attempts; after max waits the run stops with no verdict recorded,
    library unchanged, so a relaunch resumes at the same problem."""
    calls = []

    def propose(task, ws, model, minutes):
        calls.append(model)
        with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
            f.write("def p_junk(panel):\n    return 0.0\n")
        with open(os.path.join(ws, L.LOG_FILE), "a") as f:
            f.write("partial log edit\n")
        raise L.ProposerInfrastructureFailure("session limit")

    rep = L.run(_two_problems(), tag="t7", ws=str(sandbox / "ws7"),
                propose_fn=propose, verbose=False,
                infra_wait_seconds=0, max_infra_waits=2)
    # 1 first try + 2 retries after waits, all on rung 0, then stop
    assert calls == [L.codex_headless.DEFAULT_CODEX_MODEL] * 3
    assert rep.records == []  # no verdict recorded: not a solving failure
    lib = open(os.path.join(str(sandbox / "ws7"), L.LIBRARY_FILE)).read()
    assert "p_junk" not in lib
    assert open(os.path.join(str(sandbox / "ws7"), L.LOG_FILE)).read() == ""


def test_infra_recovery_consumes_no_attempt(sandbox):
    """One infra failure, then a working proposer: the ladder still has all
    its rungs and the problem solves on attempt 1."""
    state = {"n": 0}

    def propose(task, ws, model, minutes):
        state["n"] += 1
        if state["n"] == 1:
            raise L.ProposerInfrastructureFailure("rate limit exceeded")
        with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
            f.write(WITNESS)
        return "done"

    rep = L.run([two_vs_one_problem()], tag="t8", ws=str(sandbox / "ws8"),
                propose_fn=propose, verbose=False, infra_wait_seconds=0)
    assert rep.solved == 1
    assert rep.records[0].attempts == 1
    assert not rep.records[0].escalated


def test_model_generated_infrastructure_words_consume_a_scientific_attempt(
        sandbox):
    def proposer(_task, ws, _model, _minutes):
        with open(os.path.join(ws, L.LIBRARY_FILE), "a") as handle:
            handle.write(WITNESS)
        return "API error timeout rate limit"

    report = L.run(
        [two_vs_one_problem()], tag="transcript_words",
        ws=str(sandbox / "ws_transcript_words"),
        propose_fn=proposer, verbose=False)
    assert report.solved == 1
    assert report.records[0].attempts == 1


def test_stale_problem_dirs_pruned(sandbox):
    ws = str(sandbox / "ws9")
    rep = L.run(_two_problems(), tag="t9", ws=ws,
                propose_fn=writing_proposer(), verbose=False)
    assert rep.solved == 2
    dirs = [n for n in os.listdir(ws) if n.startswith("problem_")]
    assert dirs == ["problem_01"]  # only the last attempted problem remains


def test_stale_panel_pruning_refuses_unowned_content_without_deleting_it(
        sandbox):
    workspace = sandbox / "workspace_stale_unowned"
    workspace.mkdir()
    B.write_panels(
        str(workspace), two_vs_one_problem(), "problem_00")
    panel_dir = workspace / "problem_00"
    extra = panel_dir / "unowned.txt"
    extra.write_text("preserve me")
    before = {
        path.name: path.read_bytes()
        for path in panel_dir.iterdir()
    }
    with pytest.raises(RuntimeError, match="unexpected files"):
        L._prune_stale_problem_dirs(str(workspace), "problem_01")
    assert {
        path.name: path.read_bytes()
        for path in panel_dir.iterdir()
    } == before


def test_completed_failure_is_terminal_and_wip_is_preserved(sandbox):
    """A scientific failure stays in the denominator and cannot be retried
    after later library growth; its diagnostic workspace still survives."""
    def failing(task, ws, model, minutes):
        with open(os.path.join(ws, "probe_notes.md"), "w") as f:
            f.write("tried arc span, dead end")

    L.run([two_vs_one_problem()], tag="t10", ws=str(sandbox / "ws10a"),
          propose_fn=failing, verbose=False)

    def must_not_retry(*_args):
        raise AssertionError("terminal failure must not re-run the proposer")

    resumed = L.run(
        [two_vs_one_problem()], tag="t10", ws=str(sandbox / "ws10b"),
        propose_fn=must_not_retry, verbose=False)
    assert len(resumed.records) == 1 and not resumed.records[0].solved
    wip = os.path.join(
        L.artifact_dir("t10"), "wip_context", "problem_00")
    assert any(
        os.path.exists(os.path.join(wip, snapshot, "probe_notes.md"))
        for snapshot in os.listdir(wip))


def test_frozen_corpus_identity_binds_records_artifact_and_resume(sandbox):
    problems = _basic_problems()
    manifest = P.build_corpus_manifest(
        problems,
        source="basic",
        seed=19,
        limit_per_source=2,
        dataset_revision="unavailable",
    )
    bundle = P.build_corpus_bundle(problems, manifest)
    rep = L.run(
        problems,
        tag="bound",
        ws=str(sandbox / "ws_bound"),
        propose_fn=writing_proposer(),
        verbose=False,
        corpus_manifest=manifest,
        corpus_bundle=bundle,
    )
    assert rep.corpus_digest == manifest["corpus_digest"]
    assert all(record.track == "UNRESTRICTED" for record in rep.records)
    assert [record.panel_set_digest for record in rep.records] == [
        entry["panel_set_digest"] for entry in manifest["problems"]]
    art_manifest = json.load(open(os.path.join(
        L.artifact_dir("bound"), "corpus_manifest.json")))
    assert art_manifest["corpus_digest"] == manifest["corpus_digest"]

    with pytest.raises(RuntimeError, match="shorter"):
        L.run(
            problems[:1],
            tag="bound",
            ws=str(sandbox / "ws_bound_short"),
            propose_fn=lambda *args: (_ for _ in ()).throw(
                AssertionError("mismatch must fail before proposer")),
            verbose=False,
            corpus_manifest=manifest,
            corpus_bundle=bundle,
        )


def test_frozen_corpus_mismatch_fails_before_unrestricted_proposer(sandbox):
    problems = _basic_problems()
    manifest = P.build_corpus_manifest(
        problems,
        source="basic",
        seed=19,
        limit_per_source=2,
        dataset_revision="unavailable",
    )
    bundle = P.build_corpus_bundle(problems, manifest)
    changed = list(problems)
    panel = changed[0].pos[0].copy()
    panel[0, 0] ^= 1
    changed[0] = B.Problem(
        changed[0].problem_id,
        changed[0].category,
        changed[0].concept,
        [panel] + list(changed[0].pos[1:]),
        changed[0].neg,
    )
    with pytest.raises(P.PhaseDProtocolError, match="differs"):
        L.run(
            changed,
            tag="mismatch",
            ws=str(sandbox / "ws_mismatch"),
            propose_fn=lambda *args: (_ for _ in ()).throw(
                AssertionError("mismatch must fail before proposer")),
            verbose=False,
            corpus_manifest=manifest,
            corpus_bundle=bundle,
        )


def test_shuffled_side_control_runs_full_unrestricted_pipeline_in_isolated_arm(
        sandbox):
    problems = _basic_problems()
    manifest = P.build_corpus_manifest(
        problems,
        source="basic",
        seed=23,
        limit_per_source=2,
        dataset_revision="unavailable",
    )
    bundle = P.build_corpus_bundle(problems, manifest)
    control = P.build_shuffled_sides_control(
        problems, manifest, seed=71, replicate=1)
    rep = L.run(
        control.problems,
        tag="shuffled",
        ws=str(sandbox / "ws_shuffled"),
        propose_fn=writing_proposer(),
        verbose=False,
        corpus_manifest=manifest,
        corpus_bundle=bundle,
        condition=P.SHUFFLED_SIDES,
        control_manifest=control.manifest,
        base_problems=problems,
    )
    assert rep.solved == 0
    assert rep.condition == P.SHUFFLED_SIDES
    assert all(record.control_digest == control.manifest["control_digest"]
               for record in rep.records)
    art = L.artifact_dir("shuffled")
    assert os.path.exists(os.path.join(art, "control_manifest.json"))
    results = json.load(open(os.path.join(art, "results.json")))
    assert results["problem_00"]["concept"] == problems[0].concept
    assert results["problem_00"]["condition"] == P.SHUFFLED_SIDES

    with pytest.raises(RuntimeError, match="different experiment arm"):
        L.run(
            problems,
            tag="shuffled",
            ws=str(sandbox / "ws_observed_collision"),
            propose_fn=lambda *args: (_ for _ in ()).throw(
                AssertionError("arm collision must fail before proposer")),
            verbose=False,
            corpus_manifest=manifest,
            corpus_bundle=bundle,
            condition=P.OBSERVED,
        )


def test_tester_and_authoritative_verifier_share_exact_pricing_context(sandbox):
    ws = str(sandbox / "ws_pricing_parity")
    os.makedirs(ws)
    problem = two_vs_one_problem()
    B.write_panels(ws, problem, "problem_00")
    open(os.path.join(ws, "current_problem.txt"), "w").write("problem_00")
    open(os.path.join(ws, L.LIBRARY_FILE), "w").write(WITNESS)
    contract = L._pricing_context(
        P.SHARED, (), L._source_digest(L.INITIAL_LIBRARY_SOURCE))
    L._write_pricing_contract(ws, contract)
    tester = L._write_tester(ws)

    authoritative = L._verify_workspace(ws, problem, contract)
    completed = subprocess.run(
        [sys.executable, tester], cwd=ws, check=True,
        capture_output=True, text=True)
    assert completed.stdout.strip() == authoritative.result_line()
    assert authoritative.solved


def test_changed_helper_is_recharged_but_unchanged_predicate_is_reused(sandbox):
    first = """\
def helper(panel):
    return panel.sum()

def p_ink(panel):
    return float(helper(panel))
"""
    second = first.replace(
        "return panel.sum()", "return panel.astype(float).sum()")

    def proposer(_task, ws, _model, _minutes):
        path = os.path.join(ws, L.LIBRARY_FILE)
        source = open(path).read()
        if "def helper" not in source:
            open(path, "a").write(first)
        elif source.endswith(first):
            open(path, "w").write(source[:-len(first)] + second)

    report = L.run(
        _two_problems(), tag="changed_helper",
        ws=str(sandbox / "ws_changed_helper"),
        propose_fn=proposer, verbose=False)
    assert report.solved == 2
    charged = [
        node["key"] for node in report.records[1].used_definition_nodes
        if node["charged"]
    ]
    reused = [
        node["key"] for node in report.records[1].used_definition_nodes
        if not node["charged"]
    ]
    assert charged == ["function:helper"]
    assert reused == ["function:p_ink"]
    assert report.records[1].definition_charge > 0


def test_unused_inserted_definition_does_not_enter_paid_ledger(sandbox):
    source = WITNESS + "\ndef p_unused(panel):\n    return 0.0\n"
    report = L.run(
        [two_vs_one_problem()], tag="unused",
        ws=str(sandbox / "ws_unused"),
        propose_fn=writing_proposer(source), verbose=False)
    assert report.solved == 1
    assert [node["key"] for node in report.records[0].used_definition_nodes] \
        == ["function:p_ink"]
    assert len(report.paid_node_identities) == 1


def test_no_share_is_offline_held_fixed_repricing(sandbox):
    problems = _basic_problems()
    manifest = P.build_corpus_manifest(
        problems, source="basic", seed=41, limit_per_source=2,
        dataset_revision="unavailable")
    bundle = P.build_corpus_bundle(problems, manifest)
    preregistration = P.build_preregistration(
        manifest, tracks=["UNRESTRICTED"], scales=[2],
        shuffled_seed=73, shuffled_replicates=1)
    arm_tags = {
        arm["condition"]: arm["execution_tag"]
        for arm in preregistration["arms"]
        if arm["condition"] in {"primary", P.NO_SHARE}
    }
    shared_tag = arm_tags["primary"]
    no_share_tag = arm_tags[P.NO_SHARE]
    primary_binding = P.execution_binding(
        preregistration, "UNRESTRICTED:primary:n2")
    no_share_binding = P.execution_binding(
        preregistration, "UNRESTRICTED:no-share:n2")
    shared = L.run(
        problems, tag=shared_tag, ws=str(sandbox / "ws_shared_source"),
        propose_fn=writing_codex_proposer(), verbose=False,
        corpus_manifest=manifest, corpus_bundle=bundle,
        phase_execution_binding=primary_binding)
    first_cost = shared.records[0].full_definition_cost
    assert first_cost > 0
    assert [record.definition_charge for record in shared.records] == \
        [first_cost, 0]

    # Derivation reconciles duplicated scientific fields against the source
    # checkpoint before using results.json for its ground-truth-only columns.
    source_results_path = os.path.join(
        L.artifact_dir(shared_tag), "results.json")
    source_results = json.load(open(source_results_path))
    source_results["problem_00"]["solved"] = False
    source_results["problem_00"]["verification_digest"] = "tampered"
    open(source_results_path, "w").write(json.dumps(source_results))

    no_share = L.derive_no_share_artifact(
        shared_tag, no_share_tag, verbose=False,
        phase_execution_binding=no_share_binding,
        required_source_phase_execution_binding=primary_binding)
    repaired_source_results = json.load(open(source_results_path))
    assert repaired_source_results["problem_00"]["solved"] is True
    assert repaired_source_results["problem_00"]["verification_digest"] == \
        shared.records[0].verification_digest
    assert no_share.condition == P.NO_SHARE
    assert no_share.label_policy == P.OBSERVED
    assert no_share.sharing_policy == P.NO_SHARE
    assert no_share.parent_source_trace_digest == shared.source_trace_digest
    assert [record.solved for record in no_share.records] == [True, True]
    assert [record.rule_atoms for record in no_share.records] == [
        record.rule_atoms for record in shared.records]
    assert [record.definition_charge for record in no_share.records] == \
        [first_cost, first_cost]
    assert [record.source_verification_digest for record in no_share.records] == [
        record.verification_digest for record in shared.records]
    assert os.path.exists(os.path.join(
        L.artifact_dir(no_share_tag), "corpus_panels.json"))

    injected = copy.deepcopy(shared)
    injected.records[0].proposer_receipts = [
        L._injected_proposer_receipt(injected.records[0].model)]
    with pytest.raises(RuntimeError, match="real Codex CLI model receipts"):
        L.publish_phase_d_track_report(
            injected, preregistration, "UNRESTRICTED:primary:n2")
    primary_path = L.publish_phase_d_track_report(
        shared, preregistration, "UNRESTRICTED:primary:n2")
    no_share_path = L.publish_phase_d_track_report(
        no_share, preregistration, "UNRESTRICTED:no-share:n2",
        allow_test_injected_receipts=True)
    primary_report = json.load(open(primary_path))
    no_share_report = json.load(open(no_share_path))
    P.validate_track_report(primary_report, preregistration)
    P.validate_track_report(no_share_report, preregistration)
    assert primary_report["sharing_policy"] == P.SHARED
    assert no_share_report["sharing_policy"] == P.NO_SHARE

    with pytest.raises(ValueError, match="cannot launch a fresh proposer"):
        L.run(
            problems, tag="invalid_no_share",
            ws=str(sandbox / "ws_invalid_no_share"),
            propose_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("no-share must not call a proposer")),
            condition=P.NO_SHARE,
            corpus_manifest=manifest, corpus_bundle=bundle,
            verbose=False)


def test_resume_rejects_tampered_definition_receipt_before_proposer(sandbox):
    problems = _basic_problems()[:1]
    report = L.run(
        problems, tag="tampered_receipt",
        ws=str(sandbox / "ws_tampered_receipt"),
        propose_fn=writing_proposer(), verbose=False)
    assert report.solved == 1
    checkpoint = os.path.join(
        L.artifact_dir("tampered_receipt"), L.CHECKPOINT_FILE)
    payload = json.load(open(checkpoint))
    payload["records"][0]["used_definition_nodes"][0]["cost"] += 1
    open(checkpoint, "w").write(json.dumps(payload))

    with pytest.raises(RuntimeError, match="receipt does not reproduce"):
        L.run(
            problems, tag="tampered_receipt",
            ws=str(sandbox / "ws_tampered_receipt_resume"),
            propose_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("tamper must fail before proposer")),
            verbose=False)


def _run_one_bound(sandbox, tag, proposer=None):
    problems = _basic_problems()[:1]
    manifest = P.build_corpus_manifest(
        problems, source="basic", seed=91, limit_per_source=1,
        dataset_revision="unavailable")
    bundle = P.build_corpus_bundle(problems, manifest)
    report = L.run(
        problems, tag=tag, ws=str(sandbox / f"ws_{tag}"),
        propose_fn=proposer or writing_proposer(), verbose=False,
        corpus_manifest=manifest, corpus_bundle=bundle)
    return problems, manifest, bundle, report


def _rewrite_checkpoint_trace(path, mutate):
    payload = json.load(open(path))
    mutate(payload)
    records = [L.ProblemRecord(**record) for record in payload["records"]]
    payload["source_trace_digest"] = L._source_trace_digest(records)
    open(path, "w").write(json.dumps(payload))


def test_cold_replay_rejects_tampered_accuracy_even_with_new_trace(sandbox):
    problems, manifest, bundle, _report = _run_one_bound(
        sandbox, "tampered_accuracy")
    checkpoint = os.path.join(
        L.artifact_dir("tampered_accuracy"), L.CHECKPOINT_FILE)
    _rewrite_checkpoint_trace(
        checkpoint,
        lambda payload: payload["records"][0].update(
            {"heldout_accuracy": 0.75}),
    )
    with pytest.raises(RuntimeError, match="cold-replay.*differs"):
        L.run(
            problems, tag="tampered_accuracy",
            ws=str(sandbox / "ws_tampered_accuracy_resume"),
            propose_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("tamper must fail before proposer")),
            verbose=False, corpus_manifest=manifest, corpus_bundle=bundle)


def test_failed_candidate_source_is_stored_and_cold_replayed(sandbox):
    attempted = "def p_useless(panel):\n    return 0.0\n"
    problems, manifest, bundle, report = _run_one_bound(
        sandbox, "failed_source", writing_proposer(attempted))
    record = report.records[0]
    assert not record.solved
    assert attempted in record.attempted_source
    assert L._source_digest(record.attempted_source) == \
        record.attempted_source_digest

    checkpoint = os.path.join(
        L.artifact_dir("failed_source"), L.CHECKPOINT_FILE)

    def mutate(payload):
        changed = payload["records"][0]["attempted_source"].replace(
            "return 0.0", "return 1.0")
        payload["records"][0]["attempted_source"] = changed
        payload["records"][0]["attempted_source_digest"] = \
            L._source_digest(changed)

    _rewrite_checkpoint_trace(checkpoint, mutate)
    with pytest.raises(RuntimeError, match="cold-replay verification digest"):
        L.run(
            problems, tag="failed_source",
            ws=str(sandbox / "ws_failed_source_resume"),
            propose_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("tamper must fail before proposer")),
            verbose=False, corpus_manifest=manifest, corpus_bundle=bundle)


def test_resume_rejects_final_library_or_checkpoint_tag_drift(sandbox):
    problems, manifest, bundle, _report = _run_one_bound(
        sandbox, "artifact_source")
    art = L.artifact_dir("artifact_source")
    library = os.path.join(art, L.LIBRARY_FILE)
    original = open(library).read()
    open(library, "w").write(L.INITIAL_LIBRARY_SOURCE)
    with pytest.raises(RuntimeError, match="final accepted source"):
        L.run(
            problems, tag="artifact_source",
            ws=str(sandbox / "ws_artifact_source_resume"),
            propose_fn=lambda *_args: None, verbose=False,
            corpus_manifest=manifest, corpus_bundle=bundle)

    open(library, "w").write(original)
    checkpoint = os.path.join(art, L.CHECKPOINT_FILE)
    payload = json.load(open(checkpoint))
    payload["tag"] = "different_tag"
    open(checkpoint, "w").write(json.dumps(payload))
    with pytest.raises(RuntimeError, match="tag differs"):
        L.run(
            problems, tag="artifact_source",
            ws=str(sandbox / "ws_artifact_tag_resume"),
            propose_fn=lambda *_args: None, verbose=False,
            corpus_manifest=manifest, corpus_bundle=bundle)


def test_completed_resume_repairs_results_without_proposer(sandbox):
    problems, manifest, bundle, report = _run_one_bound(
        sandbox, "repair_results")
    results_path = os.path.join(L.artifact_dir("repair_results"), "results.json")
    open(results_path, "w").write("{not-json")

    def must_not_run(*_args):
        raise AssertionError("completed resume must not call proposer")

    resumed = L.run(
        problems, tag="repair_results",
        ws=str(sandbox / "ws_repair_results_resume"),
        propose_fn=must_not_run, verbose=False,
        corpus_manifest=manifest, corpus_bundle=bundle)
    repaired = json.load(open(results_path))
    assert resumed.source_trace_digest == report.source_trace_digest
    assert repaired["problem_00"]["solved"] is True
    assert repaired["problem_00"]["heldout_accuracy"] == 1.0
    assert repaired["problem_00"]["verification_digest"] == \
        report.records[0].verification_digest


def test_crash_between_artifact_source_and_checkpoint_recovers_without_proposer(
        sandbox, monkeypatch):
    tag = "staged_promotion_recovery"
    ws = str(sandbox / "ws_staged_promotion_recovery")
    original_save = L._save_checkpoint
    injected = False

    def crash_on_artifact_checkpoint(directory, report):
        nonlocal injected
        if directory == L.artifact_dir(tag) and not injected:
            injected = True
            raise RuntimeError("injected artifact checkpoint crash")
        return original_save(directory, report)

    monkeypatch.setattr(L, "_save_checkpoint", crash_on_artifact_checkpoint)
    with pytest.raises(RuntimeError, match="injected artifact checkpoint crash"):
        L.run(
            [two_vs_one_problem()], tag=tag, ws=ws,
            propose_fn=writing_proposer(), verbose=False)
    art = L.artifact_dir(tag)
    assert os.path.exists(os.path.join(art, L.PENDING_CHECKPOINT_FILE))

    monkeypatch.setattr(L, "_save_checkpoint", original_save)

    def must_not_propose(*_args):
        raise AssertionError("staged promotion recovery must not re-propose")

    recovered = L.run(
        [two_vs_one_problem()], tag=tag, ws=ws,
        propose_fn=must_not_propose, verbose=False)
    assert recovered.solved == 1
    assert not os.path.exists(os.path.join(art, L.PENDING_CHECKPOINT_FILE))
    assert L._load_checkpoint(art).source_trace_digest == \
        recovered.source_trace_digest


def test_crash_after_pending_marker_recovers_exact_nonempty_log_without_proposer(
        sandbox, monkeypatch):
    tag = "staged_promotion_log_recovery"
    interrupted_ws = str(sandbox / "ws_staged_promotion_log_interrupted")
    recovery_ws = str(sandbox / "ws_staged_promotion_log_recovery")
    expected_log = "non-empty verified log: café ☕\n".encode("utf-8")

    def source_and_log_proposer(task, ws, model, minutes):
        writing_proposer()(task, ws, model, minutes)
        with open(os.path.join(ws, L.LOG_FILE), "wb") as handle:
            handle.write(expected_log)

    original_atomic_text = L._atomic_text
    injected = False

    def crash_before_artifact_source(path, value):
        nonlocal injected
        if path == os.path.join(L.artifact_dir(tag), L.LIBRARY_FILE) \
                and not injected:
            injected = True
            assert os.path.exists(os.path.join(
                L.artifact_dir(tag), L.PENDING_CHECKPOINT_FILE))
            raise RuntimeError("injected pre-source promotion crash")
        return original_atomic_text(path, value)

    monkeypatch.setattr(L, "_atomic_text", crash_before_artifact_source)
    with pytest.raises(RuntimeError, match="injected pre-source promotion crash"):
        L.run(
            [two_vs_one_problem()], tag=tag, ws=interrupted_ws,
            propose_fn=source_and_log_proposer, verbose=False)
    art = L.artifact_dir(tag)
    pending_path = os.path.join(art, L.PENDING_CHECKPOINT_FILE)
    assert os.path.exists(pending_path)
    assert not os.path.exists(os.path.join(art, L.LOG_FILE))

    monkeypatch.setattr(L, "_atomic_text", original_atomic_text)

    def must_not_propose(*_args):
        raise AssertionError("staged promotion recovery must not re-propose")

    original_pending = open(pending_path, "rb").read()
    tampered = json.loads(original_pending)
    tampered["predicates_log"] = "resealed but wrong log\n"
    tampered["predicates_log_digest"] = L._source_digest(
        tampered["predicates_log"])
    tampered["pending_digest"] = L._canonical_digest({
        key: value for key, value in tampered.items()
        if key != "pending_digest"})
    open(pending_path, "w").write(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="differs from its report"):
        L.run(
            [two_vs_one_problem()], tag=tag,
            ws=str(sandbox / "ws_staged_promotion_log_tampered"),
            propose_fn=must_not_propose, verbose=False)
    assert os.path.exists(pending_path)
    open(pending_path, "wb").write(original_pending)

    recovered = L.run(
        [two_vs_one_problem()], tag=tag, ws=recovery_ws,
        propose_fn=must_not_propose, verbose=False)
    assert recovered.solved == 1
    assert open(os.path.join(art, L.LOG_FILE), "rb").read() == expected_log
    assert not os.path.exists(pending_path)
    assert L._load_checkpoint(art).source_trace_digest == \
        recovered.source_trace_digest


def test_corrupt_checkpoint_is_not_treated_as_a_fresh_run(sandbox):
    problems, manifest, bundle, _report = _run_one_bound(
        sandbox, "corrupt_checkpoint")
    checkpoint = os.path.join(
        L.artifact_dir("corrupt_checkpoint"), L.CHECKPOINT_FILE)
    open(checkpoint, "w").write("{not-json")
    with pytest.raises(RuntimeError, match="exists but is unreadable"):
        L.run(
            problems, tag="corrupt_checkpoint",
            ws=str(sandbox / "ws_corrupt_checkpoint_resume"),
            propose_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("corrupt checkpoint must fail before proposer")),
            verbose=False, corpus_manifest=manifest, corpus_bundle=bundle)


def test_checkpoint_schema_is_closed_and_solved_is_strict_boolean(sandbox):
    _run_one_bound(sandbox, "extra_checkpoint_key")
    extra_path = os.path.join(
        L.artifact_dir("extra_checkpoint_key"), L.CHECKPOINT_FILE)
    extra = json.load(open(extra_path))
    extra["unbound_claim"] = "not part of v3"
    open(extra_path, "w").write(json.dumps(extra))
    with pytest.raises(RuntimeError, match="top-level schema keys"):
        L._load_checkpoint(L.artifact_dir("extra_checkpoint_key"))

    _run_one_bound(sandbox, "missing_record_key")
    missing_path = os.path.join(
        L.artifact_dir("missing_record_key"), L.CHECKPOINT_FILE)
    missing = json.load(open(missing_path))
    missing["records"][0].pop("train_accuracy")
    open(missing_path, "w").write(json.dumps(missing))
    with pytest.raises(RuntimeError, match="record schema keys"):
        L._load_checkpoint(L.artifact_dir("missing_record_key"))

    _run_one_bound(sandbox, "nonboolean_solved")
    boolean_path = os.path.join(
        L.artifact_dir("nonboolean_solved"), L.CHECKPOINT_FILE)
    _rewrite_checkpoint_trace(
        boolean_path,
        lambda body: body["records"][0].update({"solved": 1}),
    )
    with pytest.raises(RuntimeError, match="must be boolean"):
        L._load_checkpoint(L.artifact_dir("nonboolean_solved"))


def test_source_change_after_child_verification_is_rejected(
        sandbox, monkeypatch):
    original = L._verify_source_snapshot

    def verify_then_mutate(source, problem, contract, *, filename):
        result = original(source, problem, contract, filename=filename)
        with open(filename, "a") as handle:
            handle.write("\n# concurrent mutation\n")
        return result

    monkeypatch.setattr(L, "_verify_source_snapshot", verify_then_mutate)
    with pytest.raises(RuntimeError, match="changed after authoritative"):
        L.run(
            [two_vs_one_problem()], tag="source_race",
            ws=str(sandbox / "ws_source_race"),
            propose_fn=writing_proposer(), verbose=False,
        )


def test_verifier_fingerprint_and_rule_cross_fields_are_mandatory(sandbox):
    _problems, _manifest, _bundle, _report = _run_one_bound(
        sandbox, "fingerprint_rule")
    checkpoint = os.path.join(
        L.artifact_dir("fingerprint_rule"), L.CHECKPOINT_FILE)
    payload = json.load(open(checkpoint))
    payload["verifier_fingerprint"]["runtime"]["numpy"] = "different"
    open(checkpoint, "w").write(json.dumps(payload))
    with pytest.raises(RuntimeError, match="verifier fingerprint"):
        L._load_checkpoint(L.artifact_dir("fingerprint_rule"))

    # Restore a valid run, then make the structured rule contradict its text.
    _run_one_bound(sandbox, "rule_cross_fields")
    rule_checkpoint = os.path.join(
        L.artifact_dir("rule_cross_fields"), L.CHECKPOINT_FILE)
    _rewrite_checkpoint_trace(
        rule_checkpoint,
        lambda body: body["records"][0]["rule_atoms"][0].update(
            {"op": "<="}),
    )
    with pytest.raises(RuntimeError, match="formatted rule disagrees"):
        L._load_checkpoint(L.artifact_dir("rule_cross_fields"))


def test_pricing_and_verifier_contract_versions_reject_legacy(tmp_path):
    assert L.REPORT_SCHEMA == "bongard.unrestricted-report/v8"
    assert L.PRICING_CONTRACT_SCHEMA == \
        "bongard.predicate-pricing-context/v3"
    assert L.VERIFIER_FINGERPRINT_SCHEMA == \
        "bongard.unrestricted-verifier/v3"
    contract = L._pricing_context(
        P.SHARED, (), L._source_digest(L.INITIAL_LIBRARY_SOURCE))
    legacy_contract = dict(contract)
    legacy_contract["schema"] = "bongard.predicate-pricing-context/v2"
    legacy_contract["context_digest"] = L._canonical_digest({
        key: value for key, value in legacy_contract.items()
        if key != "context_digest"
    })
    with pytest.raises(RuntimeError, match="pricing contract schema differs"):
        L._write_pricing_contract(str(tmp_path), legacy_contract)
    assert not (tmp_path / L.PRICING_CONTRACT_FILE).exists()

    fingerprint = L._verifier_fingerprint()
    assert fingerprint["runtime"]["python_hash_probes"] == [
        hash(f"bongard-unrestricted-replay/v3/{index}")
        for index in range(4)]
    legacy_fingerprint = dict(fingerprint)
    legacy_fingerprint["schema"] = "bongard.unrestricted-verifier/v2"
    legacy_fingerprint["fingerprint_digest"] = L._canonical_digest({
        key: value for key, value in legacy_fingerprint.items()
        if key != "fingerprint_digest"
    })
    with pytest.raises(RuntimeError, match="verifier fingerprint differs"):
        L._validate_verifier_fingerprint(legacy_fingerprint)


def test_verifier_child_resource_caps_are_derived_and_fingerprinted(
        monkeypatch):
    policy = B.verifier_resource_limit_policy()
    assert policy["policy_id"] == \
        "predicate-line-budget-plus-child-rlimit-cpu-as-data/v3"
    assert policy["child_cpu_limit_seconds"] < \
        policy["parent_wall_timeout_seconds"]
    assert policy["predicate_python_line_event_limit"] == \
        B.VERIFIER_PREDICATE_LINE_EVENT_LIMIT
    assert L._verifier_fingerprint()["selector_contract"][
        "resource_limits"] == policy

    virtual_size = 3 << 30
    memory_cap = virtual_size + B.VERIFIER_MEMORY_HEADROOM_BYTES
    lower_data_cap = memory_cap - 4096
    initial = {
        L.resource.RLIMIT_CPU: (
            L.resource.RLIM_INFINITY, L.resource.RLIM_INFINITY),
        L.resource.RLIMIT_AS: (
            L.resource.RLIM_INFINITY, L.resource.RLIM_INFINITY),
        L.resource.RLIMIT_DATA: (
            L.resource.RLIM_INFINITY, L.resource.RLIM_INFINITY),
    }
    applied = []
    monkeypatch.setattr(L, "_current_virtual_memory_bytes", lambda: virtual_size)
    monkeypatch.setattr(
        L.resource, "getrlimit", lambda kind: initial[kind])
    monkeypatch.setattr(
        L.resource, "setrlimit",
        lambda kind, limits: applied.append((kind, limits)))

    receipt = L._apply_verifier_resource_limits()
    assert receipt["virtual_memory_bytes"] == virtual_size
    assert receipt["derived_memory_cap_bytes"] == memory_cap
    assert receipt["applied"] == {
        "RLIMIT_CPU": B.VERIFIER_CHILD_CPU_LIMIT_SECONDS,
        "RLIMIT_AS": memory_cap,
        "RLIMIT_DATA": memory_cap,
    }
    assert applied == [
        (L.resource.RLIMIT_CPU,
         (B.VERIFIER_CHILD_CPU_LIMIT_SECONDS,
          B.VERIFIER_CHILD_CPU_LIMIT_SECONDS)),
        (L.resource.RLIMIT_AS, (memory_cap, memory_cap)),
        (L.resource.RLIMIT_DATA, (memory_cap, memory_cap)),
    ]

    initial[L.resource.RLIMIT_DATA] = (lower_data_cap, lower_data_cap)
    with pytest.raises(RuntimeError, match="inherited.*below.*fingerprinted"):
        L._apply_verifier_resource_limits()


def test_deterministic_predicate_line_budget_replays_expensive_source(sandbox):
    expensive = (
        "import numpy as np\n"
        "def p_metered(panel):\n"
        "    ys, xs = np.nonzero(np.asarray(panel) >= 0)\n"
        "    n = int(len(ys))\n"
        "    total = 0.0\n"
        "    for first in range(n):\n"
        "        for second in range(n):\n"
        "            total += float(first <= second)\n"
        "    return total\n"
    )
    report = L.run(
        [two_vs_one_problem()], tag="deterministic_line_budget",
        ws=str(sandbox / "ws_deterministic_line_budget"),
        propose_fn=writing_proposer(expensive), ladder=("metered-model",),
        verbose=False)
    record = report.records[0]
    assert record.solved is False
    assert record.rule == "PRICING_OR_LOAD_ERROR"
    assert record.status == L.VERIFIER_FAILURE_STATUS
    assert record.predicate_errors == 12
    assert L._load_checkpoint(
        L.artifact_dir("deterministic_line_budget")).source_trace_digest == \
        report.source_trace_digest


def test_verifier_failure_replay_rejects_fresh_unsolved_result(
        sandbox, monkeypatch):
    calls = 0

    def fail_once_then_finish_unsolved(source, problem, contract, *, filename):
        nonlocal calls
        calls += 1
        if calls == 1:
            return L._verification_failure(contract["sharing_policy"])
        return B.VerifyResult(
            False, 0.5, 0.5, "CONST_True", 0.0, 0, 36,
            sharing_policy=contract["sharing_policy"],
            selection_policy=B.PRICED_SELECTION_POLICY,
        )

    monkeypatch.setattr(
        L, "_verify_source_snapshot", fail_once_then_finish_unsolved)
    with pytest.raises(RuntimeError, match="cold-replay verification digest"):
        L.run(
            [two_vs_one_problem()], tag="failure_then_unsolved",
            ws=str(sandbox / "ws_failure_then_unsolved"),
            propose_fn=writing_proposer(
                "def p_useless(panel):\n    return 0.0\n"),
            ladder=("one-model",), verbose=False)
    assert calls >= 2


def test_verifier_failure_replay_rejects_downward_resealed_solution(sandbox):
    report = L.run(
        [two_vs_one_problem()], tag="resealed_solution",
        ws=str(sandbox / "ws_resealed_solution"),
        propose_fn=writing_proposer(), ladder=("one-model",), verbose=False)
    forged = copy.deepcopy(report)
    record = forged.records[0]
    record.solved = False
    record.heldout_accuracy = 0.0
    record.train_accuracy = 0.0
    record.rule = "PRICING_OR_LOAD_ERROR"
    record.rule_cost = 0.0
    record.marginal_C = 0
    record.status = L.VERIFIER_FAILURE_STATUS
    record.accepted_source_digest = ""
    record.accepted_source = ""
    record.predicate_names = []
    record.rule_atoms = []
    record.used_definition_nodes = []
    record.charged_definition_node_identities = []
    record.reused_definition_node_identities = []
    record.full_definition_cost = 0
    record.definition_charge = 0
    record.structure_charge = 0.0
    record.total_charge = 0.0
    record.predicate_errors = 12
    record.n_rotations = 36
    record.fold_rule_atoms = []
    sentinel_digest = L._verification_digest(
        L._verification_failure(P.SHARED),
        source_digest=record.attempted_source_digest,
        pricing_context_digest=record.pricing_context_digest,
        proposer_receipts_digest=L._proposer_receipts_digest(
            record.proposer_receipts),
    )
    record.verification_digest = sentinel_digest
    record.source_verification_digest = sentinel_digest
    forged.source_trace_digest = L._source_trace_digest(forged.records)
    L._validate_priced_report(forged)
    with pytest.raises(RuntimeError, match="cold-replay verification digest"):
        L._cold_replay_report(forged, [two_vs_one_problem()])


def test_resume_cross_checks_record_corpus_and_panel_identities(sandbox):
    _run_one_bound(sandbox, "identity_cross_fields")
    checkpoint = os.path.join(
        L.artifact_dir("identity_cross_fields"), L.CHECKPOINT_FILE)
    _rewrite_checkpoint_trace(
        checkpoint,
        lambda body: body["records"][0].update(
            {"panel_set_digest": "sha256:" + "0" * 64}),
    )
    with pytest.raises(RuntimeError, match="proposer panels differ|panel/corpus"):
        L._load_checkpoint(L.artifact_dir("identity_cross_fields"))


def test_source_trace_uses_numeric_problem_order():
    low = L.ProblemRecord(
        "problem_99", False, 0.0, "CONST_True", 0.0, 0,
        "sonnet", 1, False)
    high = L.ProblemRecord(
        "problem_100", False, 0.0, "CONST_True", 0.0, 0,
        "sonnet", 1, False)
    assert L._source_trace_digest([high, low]) == \
        L._source_trace_digest([low, high])


def test_authoritative_verifier_times_out_nonterminating_predicate(monkeypatch):
    source = """\
def p_loop(panel):
    while True:
        pass
"""
    monkeypatch.setattr(L, "AUTHORITATIVE_VERIFY_TIMEOUT_SECONDS", 0.1)
    contract = L._pricing_context(
        P.SHARED, (), L._source_digest(L.INITIAL_LIBRARY_SOURCE))
    started = time.monotonic()
    result = L._verify_source_snapshot(
        source, two_vs_one_problem(), contract)
    assert time.monotonic() - started < 3.0
    assert not result.solved
    assert result.rule == "PRICING_OR_LOAD_ERROR"
    assert result.predicate_errors == 12


def test_authoritative_verifier_drains_large_receipt_before_join(monkeypatch):
    constants = "\n".join(f"C_{index} = 0" for index in range(500))
    references = "\n".join(
        f"    value += C_{index}" for index in range(500))
    source = (
        constants
        + "\n\ndef p_ink(panel):\n"
        + "    value = panel.sum()\n"
        + references
        + "\n    return value\n")
    contract = L._pricing_context(
        P.SHARED, (), L._source_digest(L.INITIAL_LIBRARY_SOURCE))
    monkeypatch.setattr(L, "AUTHORITATIVE_VERIFY_TIMEOUT_SECONDS", 5.0)
    started = time.monotonic()
    result = L._verify_source_snapshot(
        source, two_vs_one_problem(), contract)
    assert time.monotonic() - started < 5.0
    assert result.solved
    assert result.definition_receipt is not None
    assert len(result.definition_receipt.used_nodes) >= 500


def test_preregistered_unrestricted_arm_pins_proposer_policy():
    problems = _basic_problems()[:1]
    manifest = P.build_corpus_manifest(
        problems, source="basic", seed=37, limit_per_source=1,
        dataset_revision="unavailable")
    preregistration = P.build_preregistration(
        manifest, tracks=["UNRESTRICTED"], scales=[1],
        shuffled_seed=73, shuffled_replicates=1)
    kwargs = {
        "corpus_digest": manifest["corpus_digest"],
        "condition": P.OBSERVED,
        "sharing_policy": P.SHARED,
        "scale": 1,
    }
    with pytest.raises(RuntimeError, match="ladder differs"):
        L._validate_preregistered_arm(
            preregistration, "UNRESTRICTED:primary:n1",
            ladder=("opus",), minutes=15, **kwargs)
    with pytest.raises(RuntimeError, match="minutes per attempt differ"):
        L._validate_preregistered_arm(
            preregistration, "UNRESTRICTED:primary:n1",
            ladder=L.DEFAULT_LADDER, minutes=14, **kwargs)
    with pytest.raises(RuntimeError, match="WIP restoration policy differs"):
        L._validate_preregistered_arm(
            preregistration, "UNRESTRICTED:primary:n1",
            ladder=L.DEFAULT_LADDER, minutes=15,
            restore_wip_context=False, **kwargs)


def test_preregistered_unrestricted_scale_gate_rejects_fresh_skip_and_shrink(
        tmp_path):
    base = _basic_problems()[0]
    problems = [
        B.Problem(f"problem-{index}", "basic", f"concept-{index}",
                  base.pos, base.neg)
        for index in range(25)
    ]
    manifest = P.build_corpus_manifest(
        problems, source="basic", seed=37, limit_per_source=25,
        dataset_revision="unavailable")
    preregistration = P.build_preregistration(
        manifest, tracks=["UNRESTRICTED"], scales=[1, 5, 25],
        shuffled_seed=73, shuffled_replicates=1)
    arms = {
        arm["scale"]: arm for arm in preregistration["arms"]
        if arm["track"] == "UNRESTRICTED" and arm["condition"] == "primary"
    }
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"byte-identical")
    L._validate_preregistered_scale_transition(
        preregistration, arms[1], None)
    with pytest.raises(RuntimeError, match="only the first"):
        L._validate_preregistered_scale_transition(
            preregistration, arms[5], None)
    with pytest.raises(RuntimeError, match="immediate predecessor"):
        L._validate_preregistered_scale_transition(
            preregistration, arms[25],
            SimpleNamespace(records=[object()]))
    with pytest.raises(RuntimeError, match="shrink"):
        L._validate_preregistered_scale_transition(
            preregistration, arms[5],
            SimpleNamespace(records=[object()] * 6))
    for scale, completed in ((5, 1), (5, 3), (25, 5), (25, 12)):
        L._validate_preregistered_scale_transition(
            preregistration, arms[scale],
            SimpleNamespace(records=[object()] * completed))
    assert sentinel.read_bytes() == b"byte-identical"


def test_phase_resume_rejects_rehashed_history_and_model_before_writes(
        sandbox):
    base = two_vs_one_problem()
    problems = [
        B.Problem(
            f"phase-{index}", "basic", f"concept-{index}",
            base.pos, base.neg)
        for index in range(5)]
    manifest = P.build_corpus_manifest(
        problems, source="basic", seed=91, limit_per_source=5,
        dataset_revision="unavailable")
    bundle = P.build_corpus_bundle(problems, manifest)
    preregistration = P.build_preregistration(
        manifest, tracks=["UNRESTRICTED"], scales=[1, 5],
        shuffled_seed=73, shuffled_replicates=1)
    arms = {
        arm["scale"]: arm for arm in preregistration["arms"]
        if arm["condition"] == "primary"}
    histories = {
        scale: P.execution_binding_family(preregistration, arm)
        for scale, arm in arms.items()}
    tag = arms[1]["execution_tag"]
    report = L.run(
        problems[:1], tag=tag, ws=str(sandbox / "ws_phase_seed"),
        propose_fn=writing_codex_proposer(), verbose=False,
        corpus_manifest=manifest, corpus_bundle=bundle,
        phase_execution_binding=histories[1][-1],
        phase_execution_binding_history=histories[1])

    art = L.artifact_dir(tag)
    forged = copy.deepcopy(report)
    forged.phase_execution_binding = histories[5][-1]
    forged.phase_execution_binding_history = [histories[5][-1]]
    forged.records[0].phase_execution_binding_digest = \
        histories[5][-1]["binding_digest"]
    L._save_checkpoint(art, forged)
    before = {
        str(path.relative_to(sandbox)): (
            "DIR" if path.is_dir() else path.read_bytes())
        for path in sorted(sandbox.rglob("*"))}
    with pytest.raises(RuntimeError, match="binding history differs"):
        L.run(
            problems, tag=tag, ws=str(sandbox / "ws_history_attack"),
            propose_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("history attack must fail before proposer")),
            verbose=False, corpus_manifest=manifest, corpus_bundle=bundle,
            phase_execution_binding=histories[5][-1],
            phase_predecessor_execution_binding=histories[5][-2],
            phase_execution_binding_history=histories[5])
    after = {
        str(path.relative_to(sandbox)): (
            "DIR" if path.is_dir() else path.read_bytes())
        for path in sorted(sandbox.rglob("*"))}
    assert before == after

    forged = copy.deepcopy(report)
    forged.records[0].model = "unregistered-model"
    forged_receipt = dict(forged.records[0].proposer_receipts[0])
    forged_receipt["requested_model"] = "unregistered-model"
    forged_receipt["receipt_digest"] = L._canonical_digest({
        key: value for key, value in forged_receipt.items()
        if key != "receipt_digest"
    })
    forged.records[0].proposer_receipts = [forged_receipt]
    forged_record = forged.records[0]
    pricing_context = L._pricing_context(
        P.SHARED, (), forged_record.baseline_source_digest)
    replayed = L._verify_source_snapshot(
        forged_record.attempted_source, problems[0], pricing_context)
    rebound_digest = L._verification_digest(
        replayed,
        source_digest=forged_record.attempted_source_digest,
        pricing_context_digest=forged_record.pricing_context_digest,
        proposer_receipts_digest=L._proposer_receipts_digest(
            forged_record.proposer_receipts),
    )
    forged_record.source_verification_digest = rebound_digest
    forged_record.verification_digest = rebound_digest
    L._save_checkpoint(art, forged)
    with pytest.raises(P.PhaseDProtocolError, match="preregistered ladder"):
        L.run(
            problems[:1], tag=tag, ws=str(sandbox / "ws_model_attack"),
            propose_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("model attack must fail before proposer")),
            verbose=False, corpus_manifest=manifest, corpus_bundle=bundle,
            phase_execution_binding=histories[1][-1],
            phase_execution_binding_history=histories[1])


@pytest.mark.parametrize("tag", (
    "../escape", "/absolute", "has/slash", "", "x" * 65,
))
def test_unrestricted_artifact_tags_are_contained(tag):
    with pytest.raises(ValueError, match="simple 1-64"):
        L.artifact_dir(tag)


def test_default_claude_proposer_has_no_shell_network_or_parent_access(
        tmp_path, monkeypatch):
    observed = {}

    class Completed:
        stdout = json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "done", "permission_denials": [],
            "modelUsage": {"claude-sonnet-5": {
                "inputTokens": 10, "outputTokens": 2}},
        })
        stderr = ""
        returncode = 0

    def fake_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["kwargs"] = kwargs
        return Completed()

    B.write_panels(str(tmp_path), two_vs_one_problem(), "problem_00")
    (tmp_path / "current_problem.txt").write_text("problem_00")
    (tmp_path / L.LIBRARY_FILE).write_text(L.INITIAL_LIBRARY_SOURCE)
    (tmp_path / L.LOG_FILE).write_text("")
    monkeypatch.setattr(L.subprocess, "run", fake_run)
    outcome = L.claude_propose(
        "task", str(tmp_path), "claude-sonnet-5", minutes=2,
        verbose=False)
    command = observed["cmd"]
    assert outcome.transcript == "done"
    assert outcome.receipt["actual_model"] == "claude-sonnet-5"
    assert outcome.receipt["input_tokens"] == 10
    assert outcome.receipt["output_tokens"] == 2
    assert observed["kwargs"]["cwd"] == str(tmp_path)
    assert "--dangerously-skip-permissions" not in command
    assert "--setting-sources" not in command
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert command[command.index("--tools") + 1] == "Read,Edit"
    assert command[command.index("--output-format") + 1] == "json"
    expected_allowed = L._proposer_allowed_tools("problem_00")
    allowed_start = command.index("--allowedTools") + 1
    allowed_end = command.index("--disallowedTools")
    assert command[allowed_start:allowed_end] == expected_allowed
    assert "Read(./**)" not in command
    denied_start = command.index("--disallowedTools") + 1
    denied_end = command.index("--permission-mode")
    assert set(command[denied_start:denied_end]) == {
        "Write", "Bash", "WebFetch", "WebSearch", "Agent"}
    settings = json.loads(command[command.index("--settings") + 1])
    assert settings["permissions"]["defaultMode"] == "dontAsk"
    assert settings["permissions"]["allow"] == expected_allowed
    assert settings["permissions"]["deny"] == [
        "Write", "Bash", "WebFetch", "WebSearch", "Agent"]


def test_proposer_prompt_is_manifest_derived_and_honest_about_outer_evaluation():
    leaked_command = "/private/parent/python bongard_try.py"
    task = L.build_task("problem_00", leaked_command)
    manifest = L.predicate_price.predicate_capability_manifest()
    canonical_manifest = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert canonical_manifest in task
    assert leaked_command not in task
    assert "functools, operator, statistics" not in task
    assert "WHOLE persistent source snapshot" in task
    assert "Only the outer harness evaluates the structured proposal" in task
    assert "there is no same-turn test rerun" in task.lower()
    assert "rotated leave-one-out" in task
    assert "does NOT detect such memorization" in task


@pytest.mark.parametrize("stdout", (
    json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "result": "substituted", "permission_denials": [],
        "modelUsage": {"claude-opus-4-8": {
            "inputTokens": 1, "outputTokens": 1}},
    }),
    json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "result": "missing", "permission_denials": [], "modelUsage": {},
    }),
    json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "result": "zero", "permission_denials": [],
        "modelUsage": {"claude-sonnet-5": {
            "inputTokens": 0, "outputTokens": 0}},
    }),
    "{malformed-json",
))
def test_claude_result_rejects_substitution_missing_usage_and_malformed_json(
        tmp_path, monkeypatch, stdout):
    class Completed:
        stderr = ""
        returncode = 0

    Completed.stdout = stdout
    B.write_panels(str(tmp_path), two_vs_one_problem(), "problem_00")
    (tmp_path / "current_problem.txt").write_text("problem_00")
    (tmp_path / L.LIBRARY_FILE).write_text(L.INITIAL_LIBRARY_SOURCE)
    (tmp_path / L.LOG_FILE).write_text("")
    monkeypatch.setattr(L.subprocess, "run", lambda *_args, **_kwargs: Completed())
    with pytest.raises(L.ProposerInfrastructureFailure):
        L.claude_propose(
            "task", str(tmp_path), "claude-sonnet-5", verbose=False)


def _codex_receipt_object(value):
    copied = dict(value)
    copied["event_types"] = tuple(copied["event_types"])
    copied["item_types"] = tuple(copied["item_types"])
    return L.codex_headless.CodexReceipt(**copied)


def _codex_workspace(tmp_path):
    B.write_panels(str(tmp_path), two_vs_one_problem(), "problem_00")
    (tmp_path / "current_problem.txt").write_text("problem_00")
    (tmp_path / L.LIBRARY_FILE).write_text(L.INITIAL_LIBRARY_SOURCE)
    (tmp_path / L.LOG_FILE).write_text("")
    return [
        str(tmp_path / "problem_00" / f"{side}_{index}.png")
        for side in ("pos", "neg") for index in range(6)
    ]


def test_codex_propose_applies_only_exact_causally_bound_output(
        tmp_path, monkeypatch):
    panel_paths = _codex_workspace(tmp_path)
    proposed_source = L.INITIAL_LIBRARY_SOURCE + WITNESS
    proposed_log = "added reusable ink measurement\n"
    rationale = "ink separates the panels"
    observed = {}

    def fake_run(task, paths, current_source, current_log, **kwargs):
        observed.update({
            "task": task, "paths": list(paths),
            "source": current_source, "log": current_log,
            "kwargs": kwargs,
        })
        receipt = _codex_receipt(
            kwargs["model"], task=task, current_source=current_source,
            current_log=current_log, proposed_source=proposed_source,
            proposed_log=proposed_log, rationale=rationale,
            panel_paths=paths,
        )
        return L.codex_headless.CodexProposal(
            proposed_source, proposed_log, rationale,
            _codex_receipt_object(receipt),
        )

    monkeypatch.setattr(L.codex_headless, "run_codex_proposer", fake_run)
    outcome = L.codex_propose(
        "bound task", str(tmp_path), L.codex_headless.DEFAULT_CODEX_MODEL,
        minutes=2, verbose=False)
    assert (tmp_path / L.LIBRARY_FILE).read_text() == proposed_source
    assert (tmp_path / L.LOG_FILE).read_text() == proposed_log
    assert outcome.transcript == rationale
    assert outcome.receipt["structured_output_digest"] == \
        L.codex_headless.predicate_proposer_output_digest(
            proposed_source, proposed_log, rationale)
    assert observed["paths"] == panel_paths
    assert observed["source"] == L.INITIAL_LIBRARY_SOURCE
    assert observed["log"] == ""


@pytest.mark.parametrize("forged_field", (
    "task_digest", "panel_view_digest", "panel_set_digest",
    "input_digest", "structured_output_digest", "proposed_source_digest",
))
def test_codex_propose_rejects_resealed_wrong_input_or_output_binding(
        tmp_path, monkeypatch, forged_field):
    _codex_workspace(tmp_path)
    proposed_source = L.INITIAL_LIBRARY_SOURCE + WITNESS
    proposed_log = "fixture log\n"
    rationale = "fixture rationale"

    def fake_run(task, paths, current_source, current_log, **kwargs):
        receipt = _codex_receipt(
            kwargs["model"], task=task, current_source=current_source,
            current_log=current_log, proposed_source=proposed_source,
            proposed_log=proposed_log, rationale=rationale,
            panel_paths=paths,
        )
        receipt[forged_field] = (
            "sha256:" + "f" * 64
            if forged_field == "panel_set_digest" else "f" * 64)
        receipt["receipt_digest"] = L._canonical_digest({
            key: value for key, value in receipt.items()
            if key != "receipt_digest"
        })
        return L.codex_headless.CodexProposal(
            proposed_source, proposed_log, rationale,
            _codex_receipt_object(receipt),
        )

    monkeypatch.setattr(L.codex_headless, "run_codex_proposer", fake_run)
    before = {
        L.LIBRARY_FILE: (tmp_path / L.LIBRARY_FILE).read_bytes(),
        L.LOG_FILE: (tmp_path / L.LOG_FILE).read_bytes(),
    }
    with pytest.raises(
            L.ProposerInfrastructureFailure,
            match="exact proposer input/output"):
        L.codex_propose(
            "bound task", str(tmp_path),
            L.codex_headless.DEFAULT_CODEX_MODEL,
            minutes=2, verbose=False)
    assert (tmp_path / L.LIBRARY_FILE).read_bytes() == before[L.LIBRARY_FILE]
    assert (tmp_path / L.LOG_FILE).read_bytes() == before[L.LOG_FILE]


def test_checkpoint_rejects_transplanted_or_duplicate_codex_turns(sandbox):
    report = L.run(
        _two_problems(), tag="codex_turn_identity",
        ws=str(sandbox / "workspace_codex_turn_identity"),
        propose_fn=writing_codex_proposer(), verbose=False)
    first = report.records[0].proposer_receipts[0]

    transplanted = copy.deepcopy(report)
    transplanted.records[1].proposer_receipts = [dict(first)]
    with pytest.raises(RuntimeError, match="task digest|input chain|panel binding"):
        L._validate_priced_report(transplanted)

    duplicated = copy.deepcopy(report)
    second = dict(duplicated.records[1].proposer_receipts[0])
    second["thread_id"] = first["thread_id"]
    second["event_stream_digest"] = first["event_stream_digest"]
    second["receipt_digest"] = L._canonical_digest({
        key: value for key, value in second.items()
        if key != "receipt_digest"
    })
    duplicated.records[1].proposer_receipts = [second]
    with pytest.raises(RuntimeError, match="reuses Codex turn identity"):
        L._validate_priced_report(duplicated)


def test_checkpoint_rejects_resealed_cross_problem_log_splice(sandbox):
    report = L.run(
        _two_problems(), tag="codex_log_chain",
        ws=str(sandbox / "workspace_codex_log_chain"),
        propose_fn=writing_codex_proposer(), verbose=False)
    forged = copy.deepcopy(report)
    foreign_log_digest = L._source_digest("FOREIGN LOG CONTEXT\n")
    second = forged.records[1]
    receipt = dict(second.proposer_receipts[0])
    second.baseline_log_digest = foreign_log_digest
    second.attempted_log_digest = foreign_log_digest
    receipt["current_log_digest"] = foreign_log_digest
    receipt["proposed_log_digest"] = foreign_log_digest
    receipt["receipt_digest"] = L._canonical_digest({
        key: value for key, value in receipt.items()
        if key != "receipt_digest"
    })
    second.proposer_receipts = [receipt]
    forged.source_trace_digest = L._source_trace_digest(forged.records)
    with pytest.raises(RuntimeError, match="log is not sequential"):
        L._validate_priced_report(forged)


@pytest.mark.parametrize("attack", ("replace", "remove"))
def test_promoted_artifact_requires_final_causally_bound_log(
        sandbox, attack):
    tag = f"promoted_log_{attack}"
    L.run(
        [two_vs_one_problem()], tag=tag,
        ws=str(sandbox / f"workspace_{tag}"),
        propose_fn=writing_codex_proposer(), verbose=False)
    artifact = L.artifact_dir(tag)
    log_path = os.path.join(artifact, L.LOG_FILE)
    if attack == "replace":
        with open(log_path, "w") as handle:
            handle.write("different unbound log\n")
    else:
        os.unlink(log_path)
    with pytest.raises(RuntimeError, match="predicate.*log|workspace file"):
        L._load_checkpoint(artifact)


@pytest.mark.parametrize("failure", ("timeout", "oserror", "nonzero"))
def test_codex_cli_failures_consume_no_attempt_and_resume_cleanly(
        sandbox, monkeypatch, failure):
    def fail_cli(*_args, **_kwargs):
        raise L.codex_headless.CodexProposerFailure(
            f"simulated Codex {failure}")

    monkeypatch.setattr(
        L.codex_headless, "run_codex_proposer", fail_cli)
    tag = f"cli_failure_{failure}"
    workspace = str(sandbox / f"workspace_{failure}")
    interrupted = L.run(
        [two_vs_one_problem()], tag=tag, ws=workspace, verbose=False,
        infra_wait_seconds=0, max_infra_waits=0)
    assert interrupted.records == []
    assert open(os.path.join(workspace, L.LIBRARY_FILE)).read() == \
        L.INITIAL_LIBRARY_SOURCE
    assert open(os.path.join(workspace, L.LOG_FILE)).read() == ""

    resumed = L.run(
        [two_vs_one_problem()], tag=tag, ws=workspace,
        propose_fn=writing_proposer(), verbose=False)
    assert resumed.solved == 1
    assert resumed.records[0].attempts == 1


def test_permission_denial_consumes_rung_rolls_back_and_persists_receipt(
        sandbox):
    calls = []

    def receipt(model, outcome, denials=()):
        return L._build_proposer_receipt(
            source="claude-cli", requested_model=model, actual_model=model,
            input_tokens=7, output_tokens=3,
            model_usage={model: {"inputTokens": 7, "outputTokens": 3}},
            outcome=outcome, permission_denials=denials)

    def propose(_task, ws, model, _minutes):
        calls.append(model)
        library = os.path.join(ws, L.LIBRARY_FILE)
        log = os.path.join(ws, L.LOG_FILE)
        if len(calls) == 1:
            with open(library, "a") as handle:
                handle.write("def p_denied(panel):\n    return 0.0\n")
            with open(log, "a") as handle:
                handle.write("denied partial log\n")
            return L.ProposerOutcome(
                "permission denied",
                receipt(model, "permission-denied", (
                    {"tool_name": "Edit", "tool_use_id": "denied-1"},)))
        assert "p_denied" not in open(library).read()
        assert open(log).read() == ""
        with open(library, "a") as handle:
            handle.write(WITNESS)
        return L.ProposerOutcome("saved", receipt(model, "success"))

    report = L.run(
        [two_vs_one_problem()], tag="permission_receipt",
        ws=str(sandbox / "workspace_permission_receipt"),
        propose_fn=propose, ladder=("first-model", "second-model"),
        verbose=False)
    record = report.records[0]
    assert calls == ["first-model", "second-model"]
    assert record.solved is True
    assert record.attempts == 2
    assert record.escalated is True
    assert [item["outcome"] for item in record.proposer_receipts] == [
        "permission-denied", "success"]
    checkpoint = json.load(open(os.path.join(
        L.artifact_dir("permission_receipt"), L.CHECKPOINT_FILE)))
    assert checkpoint["records"][0]["proposer_receipts"] == \
        record.proposer_receipts


def test_all_rungs_permission_denied_verifies_baseline_without_sentinel(sandbox):
    def propose(_task, _ws, model, _minutes):
        return L.ProposerOutcome(
            "permission denied",
            L._build_proposer_receipt(
                source="claude-cli", requested_model=model,
                actual_model=model, input_tokens=7, output_tokens=3,
                model_usage={
                    model: {"inputTokens": 7, "outputTokens": 3}},
                outcome="permission-denied",
                permission_denials=(
                    {"tool_name": "Edit", "tool_use_id": model},),
            ),
        )

    report = L.run(
        [two_vs_one_problem()], tag="all_permission_denied",
        ws=str(sandbox / "ws_all_permission_denied"),
        propose_fn=propose, ladder=("first-model", "second-model"),
        verbose=False)
    record = report.records[0]
    assert record.attempts == 2
    assert record.status == "UNSOLVED_UNRESTRICTED"
    assert record.rule == "CONST_True"
    assert not record.solved and record.total_charge == 0.0
    assert {receipt["outcome"] for receipt in record.proposer_receipts} == {
        "permission-denied"}


def test_permission_denial_still_admits_a_solving_shared_baseline(sandbox):
    def propose(_task, ws, model, _minutes):
        opaque_id = open(os.path.join(ws, "current_problem.txt")).read().strip()
        if opaque_id == "problem_00":
            writing_proposer()(None, ws, model, None)
            return None
        return L.ProposerOutcome(
            "permission denied",
            L._build_proposer_receipt(
                source="claude-cli", requested_model=model,
                actual_model=model, input_tokens=7, output_tokens=3,
                model_usage={
                    model: {"inputTokens": 7, "outputTokens": 3}},
                outcome="permission-denied",
                permission_denials=(
                    {"tool_name": "Edit", "tool_use_id": model},),
            ),
        )

    report = L.run(
        _two_problems(), tag="permission_reuse",
        ws=str(sandbox / "ws_permission_reuse"), propose_fn=propose,
        ladder=("first-model", "second-model"), verbose=False)
    reused = report.records[1]
    assert reused.solved
    assert reused.status == "SOLVED_UNRESTRICTED"
    assert reused.attempts == 1
    assert reused.definition_charge == reused.marginal_C == 0
    assert reused.proposer_receipts[0]["outcome"] == "permission-denied"


def test_run_rejects_symlink_workspace_before_proposer_or_outside_mutation(
        sandbox):
    outside = sandbox / "outside_workspace"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("untouched")
    workspace = sandbox / "linked_workspace"
    workspace.symlink_to(outside, target_is_directory=True)
    calls = []

    with pytest.raises(RuntimeError, match="non-symlink directory"):
        L.run(
            [two_vs_one_problem()], tag="linked_ws", ws=str(workspace),
            propose_fn=lambda *_args: calls.append(True), verbose=False)
    assert calls == []
    assert sentinel.read_text() == "untouched"
    assert sorted(path.name for path in outside.iterdir()) == ["sentinel.txt"]


def test_default_workspaces_are_private_unpredictable_and_retained(sandbox):
    first = L._private_default_workspace("private", parent=str(sandbox))
    second = L._private_default_workspace("private", parent=str(sandbox))
    assert first != second
    for workspace in (first, second):
        info = os.lstat(workspace)
        assert stat.S_IMODE(info.st_mode) == 0o700
        assert info.st_uid == os.geteuid()
        assert os.path.basename(workspace).startswith("bongard_ws_private_")
        assert os.path.isdir(workspace)


@pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
def test_run_rejects_linked_editable_file_without_mutating_target(
        sandbox, link_kind):
    workspace = sandbox / f"workspace_{link_kind}"
    workspace.mkdir()
    sentinel = sandbox / f"outside_{link_kind}.py"
    sentinel.write_text("outside-sentinel")
    library = workspace / L.LIBRARY_FILE
    if link_kind == "symlink":
        library.symlink_to(sentinel)
    else:
        os.link(sentinel, library)
    calls = []

    with pytest.raises(RuntimeError, match="workspace predicates.py must"):
        L.run(
            [two_vs_one_problem()], tag=f"linked_{link_kind}",
            ws=str(workspace),
            propose_fn=lambda *_args: calls.append(True), verbose=False)
    assert calls == []
    assert sentinel.read_text() == "outside-sentinel"


def test_post_proposer_preflight_precedes_infra_retry_and_workspace_write(
        sandbox):
    workspace = sandbox / "workspace_post_proposer"
    sentinel = sandbox / "outside_post_proposer.py"
    sentinel.write_text("outside-sentinel")
    calls = []

    def malicious_proposer(_task, ws, _model, _minutes):
        calls.append(True)
        library = os.path.join(ws, L.LIBRARY_FILE)
        os.unlink(library)
        os.symlink(sentinel, library)
        return "rate limit exceeded"

    with pytest.raises(RuntimeError, match="workspace predicates.py must"):
        L.run(
            [two_vs_one_problem()], tag="post_proposer",
            ws=str(workspace), propose_fn=malicious_proposer,
            verbose=False, infra_wait_seconds=0)
    assert calls == [True]
    assert sentinel.read_text() == "outside-sentinel"


def test_oversized_predicate_edit_is_rejected_before_read_ast_or_child(
        sandbox, monkeypatch):
    workspace = sandbox / "workspace_oversized_source"
    calls = []

    def oversized(_task, ws, _model, _minutes):
        calls.append(True)
        with open(os.path.join(ws, L.LIBRARY_FILE), "wb") as handle:
            handle.write(
                b"x" * (L.predicate_price.MAX_SOURCE_UTF8_BYTES + 1))
        return "must never become a scientific attempt"

    monkeypatch.setattr(
        L.predicate_price, "build_pricing_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized source reached AST pricing")))
    monkeypatch.setattr(
        L, "_verify_source_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized source launched verifier child")))

    with pytest.raises(RuntimeError, match="predicates.py exceeds its byte limit"):
        L.run(
            [two_vs_one_problem()], tag="oversized_source",
            ws=str(workspace), propose_fn=oversized, verbose=False)
    assert calls == [True]
    assert (workspace / L.LIBRARY_FILE).read_text() == \
        L.INITIAL_LIBRARY_SOURCE
    assert (workspace / L.LOG_FILE).read_text() == ""


def test_snapshot_and_seed_reject_link_sources_without_copying_external_bytes(
        sandbox):
    secret = sandbox / "outside_secret.txt"
    secret.write_text("do-not-copy")
    workspace = sandbox / "workspace_snapshot_link"
    workspace.mkdir()
    (workspace / "notes.md").symlink_to(secret)

    with pytest.raises(RuntimeError, match="cannot snapshot"):
        L.snapshot_wip(
            "snapshot_link", str(workspace), "problem_00", verbose=False)
    assert not (sandbox / "art_snapshot_link" / "wip_context").exists()

    artifact = sandbox / "art_seed_link"
    artifact.mkdir()
    (artifact / L.LIBRARY_FILE).symlink_to(secret)
    seed_workspace = sandbox / "workspace_seed_link"
    seed_workspace.mkdir()
    with pytest.raises(RuntimeError, match="artifact seed predicates.py"):
        L.seed_workspace_from_artifact(
            "seed_link", str(seed_workspace), verbose=False)
    assert secret.read_text() == "do-not-copy"


def test_wip_restore_rejects_source_and_destination_links_without_mutation(
        sandbox):
    secret = sandbox / "restore_secret.txt"
    secret.write_text("outside-sentinel")
    artifact = sandbox / "art_restore_links"
    snapshot = artifact / "wip_context" / "problem_00" / "snapshot"
    snapshot.mkdir(parents=True)
    source = snapshot / "notes.md"
    source.symlink_to(secret)
    workspace = sandbox / "workspace_restore_links"
    workspace.mkdir()

    with pytest.raises(RuntimeError, match="WIP source"):
        L._restore_wip_context(
            "restore_links", str(workspace), "problem_00", verbose=False)
    assert not (workspace / "notes.md").exists()
    assert secret.read_text() == "outside-sentinel"

    source.unlink()
    source.write_text("safe source")
    destination = workspace / "notes.md"
    destination.symlink_to(secret)
    with pytest.raises(RuntimeError, match="WIP destination"):
        L._restore_wip_context(
            "restore_links", str(workspace), "problem_00", verbose=False)
    assert destination.is_symlink()
    assert secret.read_text() == "outside-sentinel"


def test_outer_verifier_feedback_reaches_the_next_restricted_attempt(sandbox):
    prompts = []

    def propose(task, ws, model, minutes):
        prompts.append(task)
        with open(os.path.join(ws, L.LIBRARY_FILE), "a") as handle:
            handle.write(
                "def p_useless(panel):\n    return 0.0\n"
                if len(prompts) == 1 else WITNESS)

    report = L.run(
        [two_vs_one_problem()], tag="feedback",
        ws=str(sandbox / "ws_feedback"), propose_fn=propose,
        ladder=("first", "second"), verbose=False)
    assert report.solved == 1
    assert len(prompts) == 2
    assert "AUTHORITATIVE FEEDBACK FROM THE PREVIOUS ATTEMPT" not in prompts[0]
    assert "AUTHORITATIVE FEEDBACK FROM THE PREVIOUS ATTEMPT" in prompts[1]
    assert "solved=False" in prompts[1]


def test_arm_collision_is_rejected_before_control_binding_mutation(sandbox):
    problems = _basic_problems()
    manifest = P.build_corpus_manifest(
        problems, source="basic", seed=83, limit_per_source=2,
        dataset_revision="unavailable")
    bundle = P.build_corpus_bundle(problems, manifest)
    tag = "preflight_collision"
    L.run(
        problems, tag=tag, ws=str(sandbox / "ws_preflight_observed"),
        propose_fn=writing_proposer(), verbose=False,
        corpus_manifest=manifest, corpus_bundle=bundle)
    artifact = L.artifact_dir(tag)
    checkpoint_before = open(
        os.path.join(artifact, L.CHECKPOINT_FILE), "rb").read()
    control = P.build_shuffled_sides_control(
        problems, manifest, seed=91, replicate=0)
    with pytest.raises(RuntimeError, match="different experiment arm"):
        L.run(
            control.problems, tag=tag,
            ws=str(sandbox / "ws_preflight_shuffled"),
            propose_fn=lambda *args: (_ for _ in ()).throw(
                AssertionError("collision must fail before proposer")),
            verbose=False, corpus_manifest=manifest, corpus_bundle=bundle,
            condition=P.SHUFFLED_SIDES,
            control_manifest=control.manifest, base_problems=problems)
    assert not os.path.exists(os.path.join(artifact, "control_manifest.json"))
    assert open(os.path.join(artifact, L.CHECKPOINT_FILE), "rb").read() == \
        checkpoint_before


def test_promotion_uses_replayed_source_after_late_workspace_mutation(
        sandbox, monkeypatch):
    tag = "immutable_promotion"
    ws = str(sandbox / "ws_immutable_promotion")
    report = L.run(
        [two_vs_one_problem()], tag=tag, ws=ws,
        propose_fn=writing_proposer(), verbose=False)
    expected = L._expected_final_source(report)
    results = json.load(open(os.path.join(L.artifact_dir(tag), "results.json")))
    original_save = L._save_checkpoint
    mutated = False

    def save_then_mutate(directory, value):
        nonlocal mutated
        original_save(directory, value)
        if directory == ws and not mutated:
            with open(os.path.join(ws, L.LIBRARY_FILE), "a") as handle:
                handle.write("\n# delayed proposer write\n")
            mutated = True

    monkeypatch.setattr(L, "_save_checkpoint", save_then_mutate)
    L.promote_verified_artifact(
        tag, ws, report, results, verbose=False)
    assert mutated
    assert open(os.path.join(
        L.artifact_dir(tag), L.LIBRARY_FILE)).read() == expected
