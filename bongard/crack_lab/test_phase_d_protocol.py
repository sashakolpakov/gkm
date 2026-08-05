"""Offline tests for the frozen-corpus Phase D protocol."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bongard_arena
import codex_proposer
import dataset
import phase_d_protocol as P
import predicate_pricing
import prepare_phase_d
import run_semantic_cone
import semantic_artifacts
import semantic_replay

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "downloads", "Bongard-LOGO")


def _panel(signal: int, identity: int) -> np.ndarray:
    panel = np.zeros((8, 8), dtype=np.uint8)
    panel[0, 0] = signal
    slot = identity % 63 + 1
    panel[slot // 8, slot % 8] = 1
    return panel


def _problem(index: int = 0, category: str = "basic") -> dataset.Problem:
    return dataset.Problem(
        problem_id=f"secret-source-id-{index}",
        category=category,
        concept=f"secret-concept-{index}",
        pos=tuple(_panel(1, index * 12 + offset) for offset in range(6)),
        neg=tuple(_panel(0, index * 12 + 6 + offset) for offset in range(6)),
    )


def _problem_128(index: int = 0) -> dataset.Problem:
    tiny = _problem(index)
    return dataset.Problem(
        tiny.problem_id,
        tiny.category,
        tiny.concept,
        tuple(np.pad(panel, ((0, 120), (0, 120))) for panel in tiny.pos),
        tuple(np.pad(panel, ((0, 120), (0, 120))) for panel in tiny.neg),
    )


def _manifest(problems, source="basic", limit=None):
    return P.build_corpus_manifest(
        problems,
        source=source,
        seed=17,
        limit_per_source=limit or len(problems),
        panel_size=8,
        dataset_revision="a" * 40,
    )


def _observed_report_fixture(problem=None):
    problem = problem or _problem()
    manifest = _manifest([problem])
    bundle = P.build_corpus_bundle([problem], manifest)
    panel_digest = manifest["problems"][0]["panel_set_digest"]
    rounds = [{
        "round": 0,
        "proposer_kind": "offline-test",
        "parse_error": "",
        "candidate_count": 0,
        "candidate_ids": [],
        "hypothesis_digests": [],
        "model_receipts": [],
    }]
    evidence = run_semantic_cone._terminal_evidence(
        rounds, [], [], None, 0.02, [])
    record = run_semantic_cone.ProblemResult(
        opaque_id="problem_00", category="basic", solved=False,
        selected_hypothesis="", selected_description="", selected_rule="",
        support_errors=12, loo_errors=12, rotated_loo_errors=72,
        rotated_loo_checks=72, n_examples=12, complexity=0, rounds_used=1,
        proposer_kind="offline-test", track="SEMANTIC-PURE",
        condition=P.OBSERVED, sharing_policy=P.SHARED,
        corpus_digest=manifest["corpus_digest"],
        panel_set_digest=panel_digest, control_digest="",
        status="NO_PROPOSALS", proposer_error="", candidates=[],
        candidate_manifest=[], selection={}, terminal_evidence=evidence,
        terminal_evidence_digest=semantic_replay.canonical_json_digest(evidence),
    )
    args = SimpleNamespace(
        condition=P.OBSERVED, proposer="offline-test", model="offline",
        max_tokens=1, rounds=1, tag="fixture", source="basic", seed=17,
        limit=1, max_support_errors=0, max_loo_errors=0,
        max_rotated_loo_errors=0, lambda_value=0.02,
    )
    payload = run_semantic_cone._checkpoint_payload(
        args, [record], manifest, 1, None, bundle)
    results = {"problem_00": run_semantic_cone._result_payload(problem, record)}
    return problem, manifest, bundle, payload, results


def test_corpus_manifest_is_deterministic_ground_truth_free_and_byte_bound():
    problems = [_problem(0), _problem(1)]
    first = _manifest(problems)
    second = _manifest(problems)
    assert first == second
    encoded = json.dumps(first, sort_keys=True)
    assert "secret-source-id" not in encoded
    assert "secret-concept" not in encoded
    assert first["sampling"]["count_policy"] == P.COUNT_POLICY
    assert first["problem_count"] == 2
    P.assert_corpus_matches(first, problems)

    changed = list(problems)
    changed_panel = problems[0].pos[0].copy()
    changed_panel[7, 7] = 1
    changed[0] = dataset.Problem(
        problems[0].problem_id,
        problems[0].category,
        problems[0].concept,
        (changed_panel,) + problems[0].pos[1:],
        problems[0].neg,
    )
    with pytest.raises(P.PhaseDProtocolError, match="differs"):
        P.assert_corpus_matches(first, changed)
    with pytest.raises(P.PhaseDProtocolError, match="differs"):
        P.assert_corpus_matches(first, list(reversed(problems)))


def test_manifest_rejects_forged_nested_panel_set_even_with_new_outer_digest():
    manifest = _manifest([_problem()])
    forged = copy.deepcopy(manifest)
    forged["problems"][0]["panels"][0]["content_digest"] = "sha256:" + "0" * 64
    forged["corpus_digest"] = semantic_replay.canonical_json_digest(
        {key: value for key, value in forged.items() if key != "corpus_digest"})
    with pytest.raises(P.PhaseDProtocolError, match="panel_set_digest"):
        P.validate_corpus_manifest(forged)


def test_corpus_bundle_embeds_exact_bytes_without_ground_truth_and_detects_tamper():
    problems = [_problem(0), _problem(1)]
    manifest = _manifest(problems)
    bundle = P.build_corpus_bundle(problems, manifest)
    P.validate_corpus_bundle(bundle, manifest)
    encoded = json.dumps(bundle)
    assert "secret-source-id" not in encoded and "secret-concept" not in encoded
    reconstructed = P.problems_from_corpus_bundle(bundle, manifest)
    assert all(
        np.array_equal(expected, observed)
        for original, replayed in zip(problems, reconstructed)
        for expected, observed in zip(
            original.pos + original.neg, replayed.pos + replayed.neg)
    )

    forged = copy.deepcopy(bundle)
    forged["problems"][0]["panels"][0]["data"] = "AA=="
    forged["bundle_digest"] = semantic_replay.canonical_json_digest(
        {key: value for key, value in forged.items() if key != "bundle_digest"})
    with pytest.raises(P.PhaseDProtocolError, match="panel is invalid"):
        P.validate_corpus_bundle(forged, manifest)


def test_sample_corpus_names_per_source_limit_and_uses_shared_interleave(monkeypatch):
    seen = []
    basic = [_problem(index, "basic") for index in range(8)]
    abstract = [_problem(20 + index, "abstract") for index in range(2)]

    def fake_sample(dataset_dir, limit, seed, source, panel_size):
        seen.append({
            "dataset_dir": dataset_dir,
            "limit": limit,
            "seed": seed,
            "source": source,
            "panel_size": panel_size,
        })
        return basic if source == "basic" else abstract

    monkeypatch.setattr(P.dataset, "sample_problems", fake_sample)
    sampled = P.sample_corpus(
        "/opaque/dataset",
        limit_per_source=8,
        seed=9,
        source="both",
        panel_size=8,
    )
    assert seen == [
        {
            "dataset_dir": "/opaque/dataset", "limit": 8, "seed": 9,
            "source": "basic", "panel_size": 8,
        },
        {
            "dataset_dir": "/opaque/dataset", "limit": 8, "seed": 9,
            "source": "abstract", "panel_size": 8,
        },
    ]
    assert [problem.problem_id for problem in sampled] == [
        *(problem.problem_id for problem in basic[:4]),
        abstract[0].problem_id,
        *(problem.problem_id for problem in basic[4:8]),
        abstract[1].problem_id,
    ]


@pytest.mark.skipif(not os.path.isdir(DATASET_DIR),
                    reason="downloads/Bongard-LOGO not present")
def test_phase_d_source_streams_are_independent_and_prefix_stable():
    abstract = P.sample_corpus(
        DATASET_DIR, limit_per_source=2, seed=11, source="abstract")
    combined = P.sample_corpus(
        DATASET_DIR, limit_per_source=2, seed=11, source="both")
    combined_abstract = [problem for problem in combined
                         if problem.category == "abstract"]
    assert [problem.problem_id for problem in combined_abstract] == [
        problem.problem_id for problem in abstract]
    assert all(
        np.array_equal(left, right)
        for expected, observed in zip(abstract, combined_abstract)
        for left, right in zip(expected.pos + expected.neg,
                               observed.pos + observed.neg)
    )

    larger = P.sample_corpus(
        DATASET_DIR, limit_per_source=4, seed=11, source="abstract")
    assert [problem.problem_id for problem in larger[:2]] == [
        problem.problem_id for problem in abstract]
    assert all(
        np.array_equal(left, right)
        for expected, observed in zip(abstract, larger[:2])
        for left, right in zip(expected.pos + expected.neg,
                               observed.pos + observed.neg)
    )


def test_shuffled_sides_is_deterministic_balanced_and_full_pipeline_negative():
    problem = _problem()
    manifest = _manifest([problem])
    first = P.build_shuffled_sides_control([problem], manifest, seed=41, replicate=0)
    second = P.build_shuffled_sides_control([problem], manifest, seed=41, replicate=0)
    different = P.build_shuffled_sides_control([problem], manifest, seed=41, replicate=1)
    assert first.manifest == second.manifest
    assert first.manifest["control_digest"] != different.manifest["control_digest"]

    assignment = first.manifest["problems"][0]["assignment"]
    assert len({(item["source_side"], item["source_index"])
                for item in assignment}) == 12
    for target_side in ("pos", "neg"):
        source_sides = [
            item["source_side"] for item in assignment
            if item["target_side"] == target_side
        ]
        assert source_sides.count("pos") == 3
        assert source_sides.count("neg") == 3
    P.assert_shuffled_control_matches(
        first.manifest, manifest, [problem], first.problems)

    # This is the same full verifier used by the unrestricted runner, now fed
    # pseudo-sides rather than merely relabelling an already selected rule.
    result = bongard_arena.verify(
        {"p_original_side_signal": lambda panel: float(panel[0, 0])},
        first.problems[0],
    )
    assert not result.solved


def test_shuffle_builder_does_not_read_concept_and_tampering_is_detected():
    base = _problem()

    class ConceptTrap:
        category = base.category
        pos = base.pos
        neg = base.neg

        @property
        def problem_id(self):
            raise AssertionError("control construction must not read source problem ID")

        @property
        def concept(self):
            raise AssertionError("control construction must not read concept")

    trapped = ConceptTrap()
    manifest = _manifest([trapped])
    control = P.build_shuffled_sides_control([trapped], manifest, seed=5)
    damaged = list(control.problems)
    changed = damaged[0].pos[0].copy()
    changed[7, 7] ^= 1
    damaged[0] = dataset.Problem(
        damaged[0].problem_id,
        damaged[0].category,
        damaged[0].concept,
        (changed,) + damaged[0].pos[1:],
        damaged[0].neg,
    )
    with pytest.raises(P.PhaseDProtocolError, match="controlled panels differ"):
        P.assert_shuffled_control_matches(
            control.manifest, manifest, [trapped], damaged)


def test_no_share_reprices_definitions_once_per_problem_not_per_reference():
    uses = [
        ("problem_00", ("segment", "segment")),
        ("problem_01", ("segment", "curve")),
    ]
    costs = {"segment": 4, "curve": 7}
    structure = {"problem_00": 2, "problem_01": 3}
    shared = P.complexity_trace(
        uses, costs, sharing_policy=P.SHARED, structure_costs=structure)
    no_share = P.complexity_trace(
        uses, costs, sharing_policy=P.NO_SHARE, structure_costs=structure)
    assert [record["definition_charge"] for record in shared["records"]] == [4, 7]
    assert [record["definition_charge"] for record in no_share["records"]] == [4, 11]
    assert shared["total_structure_charge"] == no_share["total_structure_charge"] == 5
    assert shared["total_charge"] == 16
    assert no_share["total_charge"] == 20
    assert no_share["records"][1]["charged_definitions"] == ["curve", "segment"]


def _test_unrestricted_receipt(
        execution_policy, *, requested_model=None, reported_model=None,
        model_identity_evidence="jsonl-reported-model",
        input_tokens=10, event_stream_digest="d" * 64,
        panel_set_digest="sha256:" + "c" * 64,
        current_source_digest="a" * 64,
        current_log_digest="b" * 64,
        proposed_source_digest="a" * 64,
        proposed_log_digest="b" * 64,
        nonce=0):
    unrestricted = execution_policy["unrestricted"]
    runtime = execution_policy["runtime"]["codex_cli"]
    requested = requested_model or unrestricted["proposer_ladder"][0]
    reported = requested if reported_model is None else reported_model
    receipt = {
        "schema": P.PROPOSER_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": requested,
        "reported_model": reported,
        "model_identity_evidence": model_identity_evidence,
        "requested_reasoning_effort": unrestricted[
            "requested_reasoning_effort"],
        "input_tokens": input_tokens,
        "cached_input_tokens": 3,
        "output_tokens": 2,
        "reasoning_output_tokens": 1,
        "thread_id": f"12345678-1234-4234-8234-{nonce + 1:012x}",
        "codex_cli_version": runtime["version"],
        "codex_launcher_digest": runtime["launcher_digest"],
        "task_digest": "4" * 64,
        "current_source_digest": current_source_digest,
        "current_log_digest": current_log_digest,
        "prompt_digest": "5" * 64,
        "input_digest_schema": codex_proposer.PREDICATE_INPUT_DIGEST_SCHEMA,
        "input_digest": f"{nonce + 6:064x}"[-64:],
        "output_schema_digest": unrestricted[
            "proposer_output_schema_digest"],
        "panel_view_digest": f"{nonce + 7:064x}"[-64:],
        "panel_set_digest": panel_set_digest,
        "structured_output_digest": "8" * 64,
        "proposed_source_digest": proposed_source_digest,
        "proposed_log_digest": proposed_log_digest,
        "event_stream_digest": event_stream_digest,
        "event_types": [
            "thread.started", "turn.started", "item.completed",
            "turn.completed",
        ],
        "item_types": ["agent_message"],
        "isolation_policy": unrestricted["proposer_tool_surface"],
        "outcome": "success",
    }
    receipt["receipt_digest"] = semantic_replay.canonical_json_digest(
        receipt)[7:]
    codex_proposer.validate_codex_receipt(receipt)
    return receipt


def _test_unrestricted_record_evidence(
        execution_policy, panel_set_digest, *, nonce=0):
    source_digest = "a" * 64
    log_digest = hashlib.sha256(b"").hexdigest()
    receipt = _test_unrestricted_receipt(
        execution_policy,
        panel_set_digest=panel_set_digest,
        current_source_digest=source_digest,
        current_log_digest=log_digest,
        proposed_source_digest=source_digest,
        proposed_log_digest=log_digest,
        event_stream_digest=f"{nonce + 32:064x}"[-64:],
        nonce=nonce,
    )
    return {
        "model": execution_policy["unrestricted"]["proposer_ladder"][0],
        "attempts": 1,
        "proposer_receipts": [receipt],
        "proposer_feedback": [""],
        "proposer_panel_set_digest": panel_set_digest,
        "baseline_source_digest": source_digest,
        "attempted_source_digest": source_digest,
        "baseline_log_digest": log_digest,
        "attempted_log_digest": log_digest,
    }


def _test_semantic_receipt():
    receipt = {
        "schema": P.SEMANTIC_PROPOSER_RECEIPT_SCHEMA,
        "source": "anthropic-messages-api",
        "requested_model": "claude-sonnet-5",
        "actual_model": "claude-sonnet-5",
        "input_tokens": 10,
        "output_tokens": 2,
        "stop_reason": "tool_use",
    }
    receipt["receipt_digest"] = semantic_replay.canonical_json_digest(receipt)
    return receipt


def _valid_report(preregistration, arm):
    trace_digest = preregistration["preregistration_digest"]
    parent_trace_digest = trace_digest if arm["condition"] == P.NO_SHARE else ""
    if arm["condition"] == P.SHUFFLED_SIDES:
        control = next(
            item for item in preregistration["shuffled_sides"]["controls"]
            if item["replicate"] == arm["replicate"])
        panel_digests = control["panel_set_digests"]
    else:
        panel_digests = preregistration["corpus_panel_set_digests"]
    execution_bindings = P.execution_binding_family(preregistration, arm)
    records = [
        {
            "opaque_id": f"problem_{index:02d}",
            "status": (
                "SOLVED_UNRESTRICTED" if index == 0 else
                "UNSOLVED_UNRESTRICTED")
            if arm["track"] == "UNRESTRICTED" else (
                "SOLVED_SEMANTIC_PURE" if index == 0 else "NO_PROPOSALS"),
            "solved": index == 0,
            "track": arm["track"],
            "condition": arm["condition"],
            "label_policy": arm["label_policy"],
            "sharing_policy": arm["sharing_policy"],
            "corpus_digest": preregistration["corpus_digest"],
            "panel_set_digest": panel_digests[index],
            "control_digest": arm["control_digest"],
            "report_source_trace_digest": trace_digest,
            "parent_source_trace_digest": parent_trace_digest,
            "phase_execution_binding_digest": next(
                binding["binding_digest"]
                for binding in execution_bindings
                if index < binding["scale"]),
        }
        for index in range(arm["scale"])
    ]
    if arm["track"] == "UNRESTRICTED":
        for index, record in enumerate(records):
            record.update(_test_unrestricted_record_evidence(
                preregistration["execution_policy"],
                record["panel_set_digest"], nonce=index))
    else:
        for record in records:
            record["terminal_evidence"] = {
                "schema": run_semantic_cone.TERMINAL_EVIDENCE_SCHEMA,
                "proposal_outcome": "NO_PROPOSALS",
                "rounds": [{
                    "round": 0,
                    "proposer_kind": "anthropic",
                    "parse_error": "",
                    "candidate_count": 0,
                    "candidate_ids": [],
                    "hypothesis_digests": [],
                    "model_receipts": [_test_semantic_receipt()],
                }],
                "selection": {},
            }
    if arm["condition"] == P.NO_SHARE:
        primary_records = copy.deepcopy(records)
        primary_arm = next(
            candidate for candidate in preregistration["arms"]
            if candidate["arm_id"] ==
            f"{arm['track']}:primary:n{arm['scale']}")
        primary_bindings = P.execution_binding_family(
            preregistration, primary_arm)
        for index, record in enumerate(primary_records):
            record["condition"] = "primary"
            record["label_policy"] = P.OBSERVED
            record["sharing_policy"] = P.SHARED
            record["parent_source_trace_digest"] = ""
            record["phase_execution_binding_digest"] = next(
                binding["binding_digest"] for binding in primary_bindings
                if index < binding["scale"])
        parent_trace_digest = P._report_source_trace_digest(
            arm["track"], primary_records)
        for record in records:
            record["parent_source_trace_digest"] = parent_trace_digest
    report = {
        "schema": P.TRACK_REPORT_SCHEMA,
        "preregistration_digest": preregistration["preregistration_digest"],
        "corpus_digest": preregistration["corpus_digest"],
        "arm_id": arm["arm_id"],
        "execution_tag": arm["execution_tag"],
        "track": arm["track"],
        "condition": arm["condition"],
        "label_policy": arm["label_policy"],
        "sharing_policy": arm["sharing_policy"],
        "scale": arm["scale"],
        "replicate": arm["replicate"],
        "control_digest": arm["control_digest"],
        "report_source_trace_digest": trace_digest,
        "parent_source_trace_digest": parent_trace_digest,
        "records": records,
        "solved": 1,
        "attempted": arm["scale"],
    }
    _restamp_report(report)
    return report


def _restamp_report(report):
    trace = P._report_source_trace_digest(report["track"], report["records"])
    report["report_source_trace_digest"] = trace
    for record in report["records"]:
        record["report_source_trace_digest"] = trace
    return report


def _rebind_no_share_parent(no_share, primary):
    parent = primary["report_source_trace_digest"]
    no_share["parent_source_trace_digest"] = parent
    for record in no_share["records"]:
        record["parent_source_trace_digest"] = parent
    return no_share


def test_preregistration_uses_frozen_prefixes_and_keeps_tracks_separate():
    problems = [_problem(index) for index in range(5)]
    manifest = _manifest(problems)
    preregistration = P.build_preregistration(
        manifest,
        tracks=("UNRESTRICTED", "SEMANTIC-PURE"),
        scales=(1, 5),
        shuffled_seed=73,
        shuffled_replicates=2,
    )
    assert P.corpus_prefix_ids(manifest, 5) == tuple(
        f"problem_{index:02d}" for index in range(5))
    # Per scale: both tracks have primary + two shuffles; only the learned
    # unrestricted library gets the scientifically meaningful no-share arm.
    assert len(preregistration["arms"]) == 14
    assert preregistration["no_share"]["tracks"] == ["UNRESTRICTED"]

    unrestricted_arm = next(
        arm for arm in preregistration["arms"]
        if arm["arm_id"] == "UNRESTRICTED:primary:n1")
    semantic_arm = next(
        arm for arm in preregistration["arms"]
        if arm["arm_id"] == "SEMANTIC-PURE:primary:n1")
    unrestricted = _valid_report(preregistration, unrestricted_arm)
    semantic = _valid_report(preregistration, semantic_arm)
    separated = P.validate_report_collection(
        [unrestricted, semantic], preregistration)
    assert tuple(separated) == ("UNRESTRICTED", "SEMANTIC-PURE")
    assert len(separated["UNRESTRICTED"]) == 1
    assert len(separated["SEMANTIC-PURE"]) == 1

    mixed = copy.deepcopy(unrestricted)
    mixed["records"][0]["track"] = "SEMANTIC-PURE"
    with pytest.raises(P.PhaseDProtocolError, match="mixes"):
        P.validate_track_report(mixed, preregistration)
    with pytest.raises(P.PhaseDProtocolError, match="duplicate"):
        P.validate_report_collection(
            [unrestricted, unrestricted], preregistration)


def test_preregistration_and_report_aggregates_are_digest_checked():
    manifest = _manifest([_problem()])
    preregistration = P.build_preregistration(
        manifest,
        tracks=("UNRESTRICTED",),
        scales=(1,),
        shuffled_seed=1,
    )
    arm = next(arm for arm in preregistration["arms"]
               if arm["condition"] == "primary")
    report = _valid_report(preregistration, arm)
    report["solved"] = 0
    with pytest.raises(P.PhaseDProtocolError, match="aggregate"):
        P.validate_track_report(report, preregistration)

    forged = copy.deepcopy(preregistration)
    forged["tracks"] = ["SEMANTIC-PURE"]
    with pytest.raises(P.PhaseDProtocolError, match="digest"):
        P.validate_preregistration(forged)


def _redigest_preregistration(value):
    value["preregistration_digest"] = semantic_replay.canonical_json_digest({
        key: item for key, item in value.items()
        if key != "preregistration_digest"
    })


def test_preregistration_is_exact_cartesian_plan_with_bound_controls():
    problems = [_problem(index) for index in range(25)]
    manifest = _manifest(problems)
    preregistration = P.build_preregistration(
        manifest,
        tracks=["UNRESTRICTED", "SEMANTIC-PURE"],
        scales=[1, 5, 25],
        shuffled_seed=73,
        shuffled_replicates=3,
    )
    P.validate_preregistration(
        preregistration, corpus_manifest=manifest)
    assert len(preregistration["arms"]) == 27
    assert preregistration["corpus_problem_count"] == 25
    assert preregistration["shuffled_sides"]["replicates"] == [0, 1, 2]
    for binding in preregistration["shuffled_sides"]["controls"]:
        control = P.build_shuffled_control_manifest(
            manifest, seed=73, replicate=binding["replicate"])
        assert binding["control_digest"] == control["control_digest"]
        assert binding["panel_set_digests"] == [
            entry["controlled_panel_set_digest"]
            for entry in control["problems"]
        ]
        shuffled_arms = [
            arm for arm in preregistration["arms"]
            if arm["replicate"] == binding["replicate"]
        ]
        assert shuffled_arms
        assert {arm["control_digest"] for arm in shuffled_arms} == {
            binding["control_digest"]}
    for track in preregistration["tracks"]:
        primary_tags = {
            arm["execution_tag"] for arm in preregistration["arms"]
            if arm["track"] == track and arm["condition"] == "primary"}
        assert len(primary_tags) == 1
        for replicate in preregistration["shuffled_sides"]["replicates"]:
            shuffled_tags = {
                arm["execution_tag"] for arm in preregistration["arms"]
                if arm["track"] == track
                and arm["condition"] == P.SHUFFLED_SIDES
                and arm["replicate"] == replicate}
            assert len(shuffled_tags) == 1
    no_share_tags = [
        arm["execution_tag"] for arm in preregistration["arms"]
        if arm["condition"] == P.NO_SHARE]
    assert len(no_share_tags) == len(set(no_share_tags)) == 3
    assert all(1 <= len(tag) <= 64 for tag in no_share_tags)
    runtime = preregistration["execution_policy"]["runtime"]
    assert runtime["python_hash_seed_env"] == \
        os.environ.get("PYTHONHASHSEED", "random")
    assert runtime["python_hash_probes"] == [
        hash(f"bongard-phase-d/v6/{index}") for index in range(4)]
    assert runtime["codex_cli"]["version"]
    assert len(runtime["codex_cli"]["launcher_digest"]) == 64
    assert preregistration["execution_policy"]["unrestricted"][
        "proposer_tool_surface"] == codex_proposer.CODEX_ISOLATION_POLICY
    unrestricted = preregistration["execution_policy"]["unrestricted"]
    assert preregistration["execution_policy"]["schema"] == \
        "bongard.phase-d-execution-policy/v5"
    assert preregistration["schema"] == \
        "bongard.phase-d-preregistration/v6"
    assert P.PREREGISTRATION_SCHEMA == \
        "bongard.phase-d-preregistration/v6"
    assert P.TRACK_REPORT_SCHEMA == "bongard.phase-d-track-report/v7"
    assert P.PROPOSER_RECEIPT_SCHEMA == \
        codex_proposer.CODEX_RECEIPT_SCHEMA
    assert unrestricted["proposer"] == "codex-cli"
    assert unrestricted["checkpoint_schema"] == \
        "bongard.unrestricted-report/v8"
    assert unrestricted["verifier_failure_policy"] == \
        "canonical-zero-admission-exact-cold-replay/v1"
    assert unrestricted["proposer_ladder"] == [
        codex_proposer.DEFAULT_CODEX_MODEL] * 3
    assert unrestricted["requested_reasoning_effort"] == \
        codex_proposer.DEFAULT_REASONING_EFFORT
    assert unrestricted["definition_pricing"] == \
        "ast-transitive-closure-loc-literals-call-cardinality/v3"
    assert unrestricted["default_workspace_policy"] == \
        "private-harness-workspace-plus-separate-auth-and-image-only-" \
        "mode0700-codex-views/v1"
    assert unrestricted["proposer_result_policy"] == \
        "codex-jsonl-one-turn-schema-positive-usage-no-tool-events-" \
        "causal-input-output-chain/v2"
    assert unrestricted["proposer_receipt_schema"] == P.PROPOSER_RECEIPT_SCHEMA
    assert unrestricted["proposer_input_digest_schema"] == \
        codex_proposer.PREDICATE_INPUT_DIGEST_SCHEMA
    assert unrestricted["proposer_output_schema_digest"] == \
        codex_proposer.PREDICATE_PROPOSAL_SCHEMA_DIGEST
    assert unrestricted["proposer_turn_identity_policy"] == \
        "unique-thread-and-event-stream-per-adaptive-turn/v1"
    assert preregistration["execution_policy"]["semantic_pure"][
        "max_model_attempts_per_round"] == \
        P.SEMANTIC_MAX_MODEL_ATTEMPTS_PER_ROUND == 3
    assert unrestricted["predicate_pricing_policy_id"] == \
        predicate_pricing.PREDICATE_PRICING_POLICY_ID
    assert unrestricted["predicate_purity_policy_id"] == \
        predicate_pricing.PREDICATE_PURITY_POLICY_ID
    assert unrestricted["predicate_capability_manifest"] == \
        json.loads(json.dumps(
            predicate_pricing.predicate_capability_manifest(),
            sort_keys=True, separators=(",", ":")))
    resource_policy = bongard_arena.verifier_resource_limit_policy()
    assert unrestricted["authoritative_verifier_resource_limits"] == \
        resource_policy
    assert resource_policy["child_cpu_limit_seconds"] < \
        resource_policy["parent_wall_timeout_seconds"]


def test_semantic_only_policy_does_not_require_codex_cli(monkeypatch):
    manifest = _manifest([_problem()])
    monkeypatch.setattr(
        P, "_codex_cli_fingerprint",
        lambda: (_ for _ in ()).throw(P.PhaseDProtocolError("no CLI")),
    )
    semantic = P.build_preregistration(
        manifest, tracks=["SEMANTIC-PURE"], scales=[1],
        shuffled_seed=7, shuffled_replicates=1)
    assert semantic["execution_policy"]["runtime"]["codex_cli"] == {
        "version": "not-required:no-unrestricted-track",
        "launcher_digest": "",
    }
    with pytest.raises(P.PhaseDProtocolError, match="no CLI"):
        P.build_preregistration(
            manifest, tracks=["UNRESTRICTED"], scales=[1],
            shuffled_seed=7, shuffled_replicates=1)


def test_preregistration_rejects_rehashed_noncanonical_plan_fields():
    manifest = _manifest([_problem(), _problem(1)])
    original = P.build_preregistration(
        manifest,
        tracks=["UNRESTRICTED", "SEMANTIC-PURE"],
        scales=[1, 2],
        shuffled_seed=73,
        shuffled_replicates=2,
    )

    mutations = []
    duplicate_scale = copy.deepcopy(original)
    duplicate_scale["scales"] = [1, 1]
    mutations.append((duplicate_scale, "scales"))
    skipped_replicate = copy.deepcopy(original)
    skipped_replicate["shuffled_sides"]["replicates"] = [0, 2]
    mutations.append((skipped_replicate, "replicates"))
    missing_arm = copy.deepcopy(original)
    missing_arm["arms"].pop()
    mutations.append((missing_arm, "Cartesian"))
    renamed_arm = copy.deepcopy(original)
    renamed_arm["arms"][0]["arm_id"] = "invented"
    mutations.append((renamed_arm, "Cartesian"))
    wrong_scale = copy.deepcopy(original)
    wrong_scale["arms"][0]["scale"] = 2
    mutations.append((wrong_scale, "Cartesian"))
    changed_policy = copy.deepcopy(original)
    changed_policy["scale_policy"] = "choose-best-prefix-after-results"
    mutations.append((changed_policy, "scale policy"))
    changed_runner_policy = copy.deepcopy(original)
    changed_runner_policy["execution_policy"]["semantic_pure"]["lambda"] = 0.5
    mutations.append((changed_runner_policy, "execution policy"))
    legacy_execution_policy = copy.deepcopy(original)
    legacy_execution_policy["execution_policy"]["schema"] = \
        "bongard.phase-d-execution-policy/v1"
    legacy_execution_policy["execution_policy"]["policy_digest"] = \
        semantic_replay.canonical_json_digest({
            key: item
            for key, item in legacy_execution_policy[
                "execution_policy"].items()
            if key != "policy_digest"
        })
    mutations.append((legacy_execution_policy, "execution policy"))
    changed_execution_tag = copy.deepcopy(original)
    changed_execution_tag["arms"][0]["execution_tag"] = "invented-tag"
    mutations.append((changed_execution_tag, "Cartesian"))
    changed_hash_secret = copy.deepcopy(original)
    changed_hash_secret["execution_policy"]["runtime"][
        "python_hash_probes"][0] += 1
    changed_hash_secret["execution_policy"]["policy_digest"] = \
        semantic_replay.canonical_json_digest({
            key: item
            for key, item in changed_hash_secret["execution_policy"].items()
            if key != "policy_digest"
        })
    mutations.append((changed_hash_secret, "execution policy"))
    omitted_no_share = copy.deepcopy(original)
    omitted_no_share["no_share"]["tracks"] = []
    mutations.append((omitted_no_share, "must apply exactly"))

    for candidate, message in mutations:
        _redigest_preregistration(candidate)
        with pytest.raises(P.PhaseDProtocolError, match=message):
            P.validate_preregistration(candidate)

    # A self-consistent but substituted control table is caught when the
    # preregistration is checked against its frozen corpus.
    changed_control = copy.deepcopy(original)
    replacement = "sha256:" + "f" * 64
    changed_control["shuffled_sides"]["controls"][0][
        "control_digest"] = replacement
    for arm in changed_control["arms"]:
        if arm["condition"] == P.SHUFFLED_SIDES and arm["replicate"] == 0:
            arm["control_digest"] = replacement
            arm["execution_tag"] = P._execution_tag(
                track=arm["track"], condition=arm["condition"],
                scale=arm["scale"], replicate=arm["replicate"],
                control_digest=replacement,
                corpus_digest=changed_control["corpus_digest"],
                execution_policy_digest=changed_control[
                    "execution_policy"]["policy_digest"],
            )
    _redigest_preregistration(changed_control)
    P.validate_preregistration(changed_control)
    with pytest.raises(P.PhaseDProtocolError, match="does not reproduce"):
        P.validate_preregistration(
            changed_control, corpus_manifest=manifest)


def test_preregistration_rejects_hybrid_until_a_phase_d_runner_exists():
    manifest = _manifest([_problem()])
    with pytest.raises(P.PhaseDProtocolError, match="no implemented.*runner"):
        P.build_preregistration(
            manifest,
            tracks=["HYBRID"],
            scales=[1],
            shuffled_seed=7,
        )


def test_semantic_preregistered_arm_preflight_binds_policy_and_control(
        tmp_path):
    problems = [_problem()]
    manifest = _manifest(problems)
    preregistration = P.build_preregistration(
        manifest,
        tracks=["SEMANTIC-PURE"],
        scales=[1],
        shuffled_seed=73,
        shuffled_replicates=1,
    )
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps(preregistration))
    args = SimpleNamespace(
        proposer="anthropic", model="sonnet", max_tokens=8000, rounds=4,
        max_support_errors=0, max_loo_errors=0,
        max_rotated_loo_errors=0, lambda_value=0.02,
    )
    args.tag = next(
        arm["execution_tag"] for arm in preregistration["arms"]
        if arm["arm_id"] == "SEMANTIC-PURE:shuffled-sides:n1:r0")
    args.out_dir = os.path.join(
        run_semantic_cone.SEMANTIC_RUNS_DIR, args.tag)
    control = P.build_shuffled_control_manifest(
        manifest, seed=73, replicate=0)
    loaded, arm = run_semantic_cone._load_preregistered_semantic_arm(
        str(path), "SEMANTIC-PURE:shuffled-sides:n1:r0",
        corpus_manifest=manifest, args=args,
        condition=P.SHUFFLED_SIDES, scale=1,
        control_manifest=control,
    )
    assert loaded == preregistration
    assert arm["control_digest"] == control["control_digest"]

    changed_model = SimpleNamespace(**vars(args))
    changed_model.model = "different-model"
    with pytest.raises(SystemExit, match="execution policy differs"):
        run_semantic_cone._load_preregistered_semantic_arm(
            str(path), "SEMANTIC-PURE:shuffled-sides:n1:r0",
            corpus_manifest=manifest, args=changed_model,
            condition=P.SHUFFLED_SIDES, scale=1,
            control_manifest=control,
        )

    changed_control = copy.deepcopy(control)
    changed_control["control_digest"] = "sha256:" + "0" * 64
    with pytest.raises(SystemExit, match="control differs"):
        run_semantic_cone._load_preregistered_semantic_arm(
            str(path), "SEMANTIC-PURE:shuffled-sides:n1:r0",
            corpus_manifest=manifest, args=args,
            condition=P.SHUFFLED_SIDES, scale=1,
            control_manifest=changed_control,
        )

    # A byte-identical second hard-link is not acceptable provenance: the
    # preregistration path must name one immutable inode with nlink == 1.
    backing = tmp_path / "preregistration-backing.json"
    path.replace(backing)
    os.link(backing, path)
    with pytest.raises(SystemExit, match="invalid Phase D preregistration"):
        run_semantic_cone._load_preregistered_semantic_arm(
            str(path), "SEMANTIC-PURE:shuffled-sides:n1:r0",
            corpus_manifest=manifest, args=args,
            condition=P.SHUFFLED_SIDES, scale=1,
            control_manifest=control,
        )


def test_build_track_report_uses_exact_preregistered_arm():
    manifest = _manifest([_problem(), _problem(1)])
    preregistration = P.build_preregistration(
        manifest,
        tracks=["UNRESTRICTED"],
        scales=[2],
        shuffled_seed=7,
        shuffled_replicates=1,
    )
    arm_id = "UNRESTRICTED:primary:n2"
    binding_digest = P.execution_binding(
        preregistration, arm_id)["binding_digest"]
    records = [
        {"opaque_id": f"problem_{index:02d}", "solved": index == 0,
         "status": (
             "SOLVED_UNRESTRICTED" if index == 0
             else "UNSOLVED_UNRESTRICTED"),
         "track": "UNRESTRICTED",
         "phase_execution_binding_digest": binding_digest,
         **_test_unrestricted_record_evidence(
             preregistration["execution_policy"],
             preregistration["corpus_panel_set_digests"][index],
             nonce=index)}
        for index in range(2)
    ]
    report = P.build_track_report(
        preregistration,
        arm_id=arm_id,
        records=records,
    )
    assert report["attempted"] == 2 and report["solved"] == 1
    assert report["arm_id"] == arm_id
    assert report["control_digest"] == ""
    assert report["records"][0]["panel_set_digest"] == \
        preregistration["corpus_panel_set_digests"][0]
    P.validate_track_report(report, preregistration)


def test_track_report_requires_canonical_verifier_failure_evidence():
    manifest = _manifest([_problem()])
    preregistration = P.build_preregistration(
        manifest, tracks=["UNRESTRICTED"], scales=[1],
        shuffled_seed=7, shuffled_replicates=1)
    arm = next(item for item in preregistration["arms"]
               if item["arm_id"] == "UNRESTRICTED:primary:n1")
    report = _valid_report(preregistration, arm)
    record = report["records"][0]
    record.update({
        "solved": False,
        "status": "VERIFIER_FAILURE_UNRESTRICTED",
        "heldout_accuracy": 0.0,
        "train_accuracy": 0.0,
        "rule": "PRICING_OR_LOAD_ERROR",
        "rule_cost": 0.0,
        "marginal_C": 0,
        "accepted_source_digest": "",
        "accepted_source": "",
        "predicate_names": [],
        "rule_atoms": [],
        "fold_rule_atoms": [],
        "used_definition_nodes": [],
        "charged_definition_node_identities": [],
        "reused_definition_node_identities": [],
        "full_definition_cost": 0,
        "definition_charge": 0,
        "structure_charge": 0.0,
        "total_charge": 0.0,
        "predicate_errors": 12,
        "n_rotations": 36,
    })
    report["solved"] = 0
    trace = P._report_source_trace_digest("UNRESTRICTED", report["records"])
    report["report_source_trace_digest"] = trace
    record["report_source_trace_digest"] = trace
    P.validate_track_report(report, preregistration)

    tampered = copy.deepcopy(report)
    tampered["records"][0]["heldout_accuracy"] = 0.5
    with pytest.raises(P.PhaseDProtocolError, match="canonical verifier-failure"):
        P.validate_track_report(tampered, preregistration)

    mislabeled = copy.deepcopy(report)
    mislabeled["records"][0]["status"] = "UNSOLVED_UNRESTRICTED"
    with pytest.raises(P.PhaseDProtocolError, match="does not identify"):
        P.validate_track_report(mislabeled, preregistration)


def test_track_report_rejects_panel_control_and_trace_stamp_drift():
    manifest = _manifest([_problem()])
    preregistration = P.build_preregistration(
        manifest,
        tracks=["UNRESTRICTED"],
        scales=[1],
        shuffled_seed=7,
        shuffled_replicates=1,
    )
    arm = next(
        arm for arm in preregistration["arms"]
        if arm["condition"] == P.SHUFFLED_SIDES)
    report = _valid_report(preregistration, arm)

    wrong_panel = copy.deepcopy(report)
    wrong_panel["records"][0]["panel_set_digest"] = "sha256:" + "0" * 64
    with pytest.raises(P.PhaseDProtocolError, match="different panel set"):
        P.validate_track_report(wrong_panel, preregistration)


def test_track_report_rejects_resealed_cross_problem_log_splice():
    manifest = _manifest([_problem(), _problem(1)])
    preregistration = P.build_preregistration(
        manifest, tracks=["UNRESTRICTED"], scales=[2],
        shuffled_seed=7, shuffled_replicates=1)
    arm = next(
        item for item in preregistration["arms"]
        if item["arm_id"] == "UNRESTRICTED:primary:n2")
    report = _valid_report(preregistration, arm)
    second = report["records"][1]
    foreign = hashlib.sha256(b"FOREIGN LOG CONTEXT\n").hexdigest()
    second["baseline_log_digest"] = foreign
    second["attempted_log_digest"] = foreign
    receipt = second["proposer_receipts"][0]
    receipt["current_log_digest"] = foreign
    receipt["proposed_log_digest"] = foreign
    receipt["receipt_digest"] = semantic_replay.canonical_json_digest({
        key: value for key, value in receipt.items()
        if key != "receipt_digest"
    })[7:]
    _restamp_report(report)
    with pytest.raises(P.PhaseDProtocolError, match="log trace is not sequential"):
        P.validate_track_report(report, preregistration)


def test_track_report_rejects_rehashed_proposer_model_substitution():
    manifest = _manifest([_problem()])
    preregistration = P.build_preregistration(
        manifest, tracks=["UNRESTRICTED"], scales=[1],
        shuffled_seed=7, shuffled_replicates=1)
    arm = next(
        item for item in preregistration["arms"]
        if item["arm_id"] == "UNRESTRICTED:primary:n1")
    report = _valid_report(preregistration, arm)
    receipt = report["records"][0]["proposer_receipts"][0]
    receipt["reported_model"] = "gpt-5.5-codex"
    receipt["receipt_digest"] = semantic_replay.canonical_json_digest({
        key: value for key, value in receipt.items()
        if key != "receipt_digest"
    })[7:]
    _restamp_report(report)
    with pytest.raises(P.PhaseDProtocolError, match="reported model differs"):
        P.validate_track_report(report, preregistration)

    relabelled = _valid_report(preregistration, arm)
    relabelled_record = relabelled["records"][0]
    relabelled_receipt = relabelled_record["proposer_receipts"][0]
    relabelled_record["model"] = "gpt-5.5-codex"
    relabelled_receipt["requested_model"] = "gpt-5.5-codex"
    relabelled_receipt["reported_model"] = "gpt-5.5-codex"
    relabelled_receipt["receipt_digest"] = \
        semantic_replay.canonical_json_digest({
            key: value for key, value in relabelled_receipt.items()
            if key != "receipt_digest"
        })[7:]
    _restamp_report(relabelled)
    with pytest.raises(P.PhaseDProtocolError, match="preregistered ladder"):
        P.validate_track_report(relabelled, preregistration)

    # An omitted provider-side model is honest only when the receipt says so;
    # an empty reported model may not masquerade as JSONL identity evidence.
    false_reported_evidence = _valid_report(preregistration, arm)
    false_receipt = false_reported_evidence["records"][0][
        "proposer_receipts"][0]
    false_receipt["reported_model"] = ""
    false_receipt["receipt_digest"] = \
        semantic_replay.canonical_json_digest({
            key: value for key, value in false_receipt.items()
            if key != "receipt_digest"
        })[7:]
    _restamp_report(false_reported_evidence)
    with pytest.raises(P.PhaseDProtocolError, match="reported model differs"):
        P.validate_track_report(false_reported_evidence, preregistration)

    honest_omission = _valid_report(preregistration, arm)
    omitted_receipt = honest_omission["records"][0][
        "proposer_receipts"][0]
    omitted_receipt["reported_model"] = ""
    omitted_receipt["model_identity_evidence"] = \
        "explicit-cli-model-flag;jsonl-omits-model"
    omitted_receipt["receipt_digest"] = \
        semantic_replay.canonical_json_digest({
            key: value for key, value in omitted_receipt.items()
            if key != "receipt_digest"
        })[7:]
    _restamp_report(honest_omission)
    P.validate_track_report(honest_omission, preregistration)

    clean_report = _valid_report(preregistration, arm)
    wrong_control = copy.deepcopy(clean_report)
    wrong_control["control_digest"] = "sha256:" + "1" * 64
    with pytest.raises(P.PhaseDProtocolError, match="control_digest differs"):
        P.validate_track_report(wrong_control, preregistration)

    with pytest.raises(P.PhaseDProtocolError, match="source trace"):
        P.build_track_report(
            preregistration,
            arm_id=arm["arm_id"],
            records=clean_report["records"],
            report_source_trace_digest="sha256:" + "2" * 64,
        )

    invented_status = copy.deepcopy(clean_report)
    invented_status["records"][0]["status"] = "SOLVED_FAKE"
    with pytest.raises(P.PhaseDProtocolError, match="not a terminal"):
        P.validate_track_report(invented_status, preregistration)


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("codex_cli_version", "codex-cli forged"),
        ("codex_launcher_digest", "1" * 64),
        ("output_schema_digest", "2" * 64),
        ("requested_reasoning_effort", "high"),
        ("isolation_policy", "read-only-but-unbound/v999"),
    ],
)
def test_track_report_binds_codex_launcher_schema_reasoning_and_isolation(
        field, forged):
    manifest = _manifest([_problem()])
    preregistration = P.build_preregistration(
        manifest, tracks=["UNRESTRICTED"], scales=[1],
        shuffled_seed=7, shuffled_replicates=1)
    arm = next(
        item for item in preregistration["arms"]
        if item["arm_id"] == "UNRESTRICTED:primary:n1")
    report = _valid_report(preregistration, arm)
    receipt = report["records"][0]["proposer_receipts"][0]
    receipt[field] = forged
    receipt["receipt_digest"] = semantic_replay.canonical_json_digest({
        key: value for key, value in receipt.items()
        if key != "receipt_digest"
    })[7:]
    _restamp_report(report)
    with pytest.raises(P.PhaseDProtocolError, match="Codex"):
        P.validate_track_report(report, preregistration)


def test_semantic_track_report_rejects_impossible_four_receipt_round():
    manifest = _manifest([_problem()])
    preregistration = P.build_preregistration(
        manifest, tracks=["SEMANTIC-PURE"], scales=[1],
        shuffled_seed=7, shuffled_replicates=1)
    arm = next(
        item for item in preregistration["arms"]
        if item["arm_id"] == "SEMANTIC-PURE:primary:n1")
    report = _valid_report(preregistration, arm)
    report["records"][0]["terminal_evidence"]["rounds"][0][
        "model_receipts"] = [
            _test_semantic_receipt() for _ in range(4)]
    _restamp_report(report)
    with pytest.raises(P.PhaseDProtocolError, match="receipt"):
        P.validate_track_report(report, preregistration)


def test_track_report_rejects_rehashed_wrong_execution_tranche():
    manifest = _manifest([_problem(index) for index in range(5)])
    preregistration = P.build_preregistration(
        manifest, tracks=["UNRESTRICTED"], scales=[1, 5],
        shuffled_seed=7, shuffled_replicates=1)
    arm = next(
        item for item in preregistration["arms"]
        if item["arm_id"] == "UNRESTRICTED:primary:n5")
    report = _valid_report(preregistration, arm)
    report["records"][0]["phase_execution_binding_digest"] = \
        P.execution_binding(preregistration, arm["arm_id"])["binding_digest"]
    _restamp_report(report)
    with pytest.raises(P.PhaseDProtocolError, match="execution tranche"):
        P.validate_track_report(report, preregistration)


def test_semantic_no_share_cannot_be_preregistered():
    manifest = _manifest([_problem()])
    with pytest.raises(P.PhaseDProtocolError, match="only.*unrestricted library"):
        P.build_preregistration(
            manifest,
            tracks=["SEMANTIC-PURE"],
            scales=[1],
            shuffled_seed=7,
            no_share_tracks=["SEMANTIC-PURE"],
        )


def test_complete_report_collection_requires_all_arms_and_nested_prefixes():
    manifest = _manifest([_problem(), _problem(1)])
    preregistration = P.build_preregistration(
        manifest,
        tracks=["UNRESTRICTED", "SEMANTIC-PURE"],
        scales=[1, 2],
        shuffled_seed=7,
        shuffled_replicates=1,
    )
    reports = [
        _valid_report(preregistration, arm)
        for arm in preregistration["arms"]
    ]
    grouped = P.validate_complete_report_collection(
        reports, preregistration)
    assert len(grouped["UNRESTRICTED"]) == 6
    assert len(grouped["SEMANTIC-PURE"]) == 4

    with pytest.raises(P.PhaseDProtocolError, match="incomplete"):
        P.validate_complete_report_collection(reports[:-1], preregistration)

    divergent = json.loads(json.dumps(reports))
    larger = next(
        report for report in divergent
        if report["arm_id"] == "UNRESTRICTED:primary:n2")
    larger["records"][0]["diagnostic"] = "DIVERGED"
    _restamp_report(larger)
    with pytest.raises(P.PhaseDProtocolError, match="not nested prefixes"):
        P.validate_complete_report_collection(divergent, preregistration)


def test_no_share_report_cannot_change_primary_outcome_or_rule():
    manifest = _manifest([_problem()])
    preregistration = P.build_preregistration(
        manifest,
        tracks=["UNRESTRICTED"],
        scales=[1],
        shuffled_seed=7,
        shuffled_replicates=1,
    )
    reports = [
        _valid_report(preregistration, arm)
        for arm in preregistration["arms"]
    ]
    primary = next(report for report in reports
                   if report["condition"] == "primary")
    no_share = next(report for report in reports
                    if report["condition"] == P.NO_SHARE)
    primary["records"][0]["rule"] = "p_a>=0.5"
    no_share["records"][0]["rule"] = "p_b>=0.5"
    _restamp_report(primary)
    _restamp_report(no_share)
    _rebind_no_share_parent(no_share, primary)
    with pytest.raises(P.PhaseDProtocolError, match="changes primary rule"):
        P.validate_complete_report_collection(reports, preregistration)


def test_no_share_allows_only_explicit_accounting_differences():
    manifest = _manifest([_problem()])
    preregistration = P.build_preregistration(
        manifest,
        tracks=["UNRESTRICTED"],
        scales=[1],
        shuffled_seed=7,
        shuffled_replicates=1,
    )
    reports = [
        _valid_report(preregistration, arm)
        for arm in preregistration["arms"]
    ]
    primary = next(report for report in reports
                   if report["condition"] == "primary")
    no_share = next(report for report in reports
                    if report["condition"] == P.NO_SHARE)
    primary_record = primary["records"][0]
    no_share_record = no_share["records"][0]
    shared_evidence = {
        "heldout_accuracy": 1.0,
        "source_verification_digest": "sha256:" + "3" * 64,
        "accepted_source_digest": "sha256:" + "4" * 64,
        "used_definition_nodes": [{
            "key": "function:p_signal",
            "identity": "sha256:" + "5" * 64,
            "cost": 7,
            "charged": True,
        }],
    }
    primary_record.update(copy.deepcopy(shared_evidence))
    no_share_record.update(copy.deepcopy(shared_evidence))
    primary_record.update({
        "definition_charge": 0,
        "total_charge": 2.0,
        "pricing_context_digest": "sha256:" + "6" * 64,
        "verification_digest": "sha256:" + "7" * 64,
    })
    no_share_record.update({
        "definition_charge": 7,
        "total_charge": 9.0,
        "pricing_context_digest": "sha256:" + "8" * 64,
        "verification_digest": "sha256:" + "9" * 64,
    })
    no_share_record["used_definition_nodes"][0]["charged"] = False
    _restamp_report(primary)
    _restamp_report(no_share)
    _rebind_no_share_parent(no_share, primary)
    P.validate_complete_report_collection(reports, preregistration)

    changed_score = copy.deepcopy(reports)
    next(report for report in changed_score
         if report["condition"] == P.NO_SHARE)["records"][0][
             "heldout_accuracy"] = 0.5
    _restamp_report(next(report for report in changed_score
                         if report["condition"] == P.NO_SHARE))
    with pytest.raises(P.PhaseDProtocolError, match="heldout_accuracy"):
        P.validate_complete_report_collection(changed_score, preregistration)

    changed_identity = copy.deepcopy(reports)
    next(report for report in changed_identity
         if report["condition"] == P.NO_SHARE)["records"][0][
             "used_definition_nodes"][0]["identity"] = "sha256:" + "a" * 64
    _restamp_report(next(report for report in changed_identity
                         if report["condition"] == P.NO_SHARE))
    with pytest.raises(P.PhaseDProtocolError, match="used_definition_nodes"):
        P.validate_complete_report_collection(changed_identity, preregistration)


def test_semantic_prepare_only_freezes_manifest_before_proposer(tmp_path, monkeypatch):
    problems = [_problem_128()]
    monkeypatch.setattr(
        run_semantic_cone.phase_d_protocol,
        "sample_corpus",
        lambda *args, **kwargs: problems,
    )
    monkeypatch.setattr(
        run_semantic_cone,
        "AnthropicCofiberedProposer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepare-only must not construct a paid proposer")),
    )
    monkeypatch.setattr(
        run_semantic_cone.semantic_replay, "BONGARD_ROOT", tmp_path)
    out_dir = tmp_path / "prepared"
    args = SimpleNamespace(
        proposer="anthropic",
        out_dir=str(out_dir),
        max_support_errors=0,
        max_loo_errors=0,
        max_rotated_loo_errors=0,
        limit=1,
        rounds=1,
        max_tokens=1,
        corpus_size=1,
        lambda_value=0.02,
        dataset_dir=str(tmp_path / "dataset"),
        seed=17,
        source="basic",
        condition=P.OBSERVED,
        control_seed=1,
        control_replicate=0,
        prepare_only=True,
        model="unused",
        tag="unused",
    )
    run_semantic_cone.run(args)
    manifest = json.loads((out_dir / "corpus_manifest.json").read_text())
    P.validate_corpus_manifest(manifest)
    assert manifest["problem_count"] == 1
    assert not (out_dir / "workspace").exists()


def test_semantic_prepare_only_freezes_balanced_control_before_proposer(
        tmp_path, monkeypatch):
    problems = [_problem_128()]
    monkeypatch.setattr(
        run_semantic_cone.phase_d_protocol,
        "sample_corpus",
        lambda *args, **kwargs: problems,
    )
    monkeypatch.setattr(
        run_semantic_cone,
        "AnthropicCofiberedProposer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepare-only must not construct a paid proposer")),
    )
    monkeypatch.setattr(
        run_semantic_cone.semantic_replay, "BONGARD_ROOT", tmp_path)
    out_dir = tmp_path / "prepared_control"
    args = SimpleNamespace(
        proposer="anthropic",
        out_dir=str(out_dir),
        max_support_errors=0,
        max_loo_errors=0,
        max_rotated_loo_errors=0,
        limit=1,
        rounds=1,
        max_tokens=1,
        corpus_size=1,
        lambda_value=0.02,
        dataset_dir=str(tmp_path / "dataset"),
        seed=17,
        source="basic",
        condition=P.SHUFFLED_SIDES,
        control_seed=91,
        control_replicate=2,
        prepare_only=True,
        model="unused",
        tag="unused",
    )
    run_semantic_cone.run(args)
    corpus_manifest = json.loads(
        (out_dir / "corpus_manifest.json").read_text())
    control_manifest = json.loads(
        (out_dir / "control_manifest.json").read_text())
    P.validate_shuffled_control_manifest(control_manifest, corpus_manifest)
    assert control_manifest["replicate"] == 2
    assert not (out_dir / "workspace").exists()


def test_shuffled_control_runs_full_semantic_pipeline_and_reports_zero_solve(
        tmp_path, monkeypatch):
    problems = [_problem_128()]
    monkeypatch.setattr(
        run_semantic_cone.phase_d_protocol,
        "sample_corpus",
        lambda *args, **kwargs: problems,
    )
    monkeypatch.setattr(
        run_semantic_cone.semantic_replay, "BONGARD_ROOT", tmp_path)
    monkeypatch.setattr(
        semantic_artifacts, "artifact_dir",
        lambda tag: str(tmp_path / f"artifact_{tag}"),
    )

    class EmptyProposer:
        def __init__(self, model, max_tokens):
            self.model = model

        def propose(self, problem_id, panel_paths):
            assert len(panel_paths) == 12
            return run_semantic_cone.ProposalBundle(
                problem_id=problem_id,
                hypotheses=(),
                raw_text="no admissible proposal",
                proposer_kind="offline-test",
            )

        def refine(self, problem_id, feedback):
            raise AssertionError("empty one-round proposer must not refine")

    monkeypatch.setattr(
        run_semantic_cone, "AnthropicCofiberedProposer", EmptyProposer)
    out_dir = tmp_path / "semantic_control_run"
    args = SimpleNamespace(
        proposer="anthropic",
        out_dir=str(out_dir),
        max_support_errors=0,
        max_loo_errors=0,
        max_rotated_loo_errors=0,
        limit=1,
        rounds=1,
        max_tokens=1,
        corpus_size=1,
        lambda_value=0.02,
        dataset_dir=str(tmp_path / "dataset"),
        seed=17,
        source="basic",
        condition=P.SHUFFLED_SIDES,
        control_seed=91,
        control_replicate=0,
        prepare_only=False,
        model="offline",
        tag="control",
    )
    run_semantic_cone.run(args)
    checkpoint = json.loads((out_dir / "checkpoint.json").read_text())
    record = checkpoint["records"][0]
    control_manifest = json.loads(
        (out_dir / "control_manifest.json").read_text())
    assert checkpoint["condition"] == P.SHUFFLED_SIDES
    assert checkpoint["attempted"] == 1 and checkpoint["solved"] == 0
    assert record["condition"] == P.SHUFFLED_SIDES
    assert record["control_digest"] == control_manifest["control_digest"]
    assert record["panel_set_digest"] == control_manifest["problems"][0][
        "controlled_panel_set_digest"]
    artifact = tmp_path / "artifact_control"
    assert json.loads((artifact / "checkpoint.json").read_text())[
        "artifact_state"] == "RUN_COMPLETE"
    assert (artifact / "control_manifest.json").exists()

    monkeypatch.setattr(
        run_semantic_cone,
        "AnthropicCofiberedProposer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("terminal prefix resume must not construct proposer")),
    )
    run_semantic_cone.run(args)
    resumed = json.loads((out_dir / "checkpoint.json").read_text())
    assert resumed["records"] == checkpoint["records"]


def test_zero_solve_run_report_preserves_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(
        semantic_artifacts, "artifact_dir", lambda tag: str(tmp_path / tag))
    _, manifest, bundle, payload, results = _observed_report_fixture()
    art = semantic_artifacts.publish_run_report(
        "zero", payload, results, manifest, corpus_bundle=bundle)
    checkpoint = json.loads(open(os.path.join(art, "checkpoint.json")).read())
    persisted_results = json.loads(open(os.path.join(art, "results.json")).read())
    assert checkpoint["artifact_state"] == "RUN_COMPLETE"
    assert len(checkpoint["records"]) == 1
    assert persisted_results["problem_00"]["solved"] is False
    assert os.path.exists(os.path.join(art, "corpus_manifest.json"))
    assert os.path.exists(os.path.join(art, "corpus_panels.json"))


def test_run_report_strictly_validates_bundle_prefix_and_result_identity(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        semantic_artifacts, "artifact_dir", lambda tag: str(tmp_path / tag))
    _, manifest, bundle, payload, results = _observed_report_fixture()

    with pytest.raises(semantic_artifacts.ReplayCertificationError,
                       match="embedded corpus bundle"):
        semantic_artifacts.publish_run_report(
            "missing_bundle", payload, results, manifest)

    noncontiguous = copy.deepcopy(payload)
    noncontiguous["records"][0]["opaque_id"] = "problem_01"
    with pytest.raises(semantic_artifacts.ReplayCertificationError,
                       match="contiguous"):
        semantic_artifacts.publish_run_report(
            "noncontiguous", noncontiguous, results, manifest,
            corpus_bundle=bundle)

    mismatched_results = copy.deepcopy(results)
    mismatched_results["problem_00"]["panel_set_digest"] = "sha256:" + "0" * 64
    with pytest.raises(semantic_artifacts.ReplayCertificationError,
                       match="differs from its checkpoint"):
        semantic_artifacts.publish_run_report(
            "bad_results", payload, mismatched_results, manifest,
            corpus_bundle=bundle)
    assert not (tmp_path / "bad_results").exists()

    two_problems = [_problem(), _problem(1)]
    two_manifest = _manifest(two_problems)
    two_bundle = P.build_corpus_bundle(two_problems, two_manifest)
    incomplete = copy.deepcopy(payload)
    incomplete["dataset"].update({
        "corpus_digest": two_manifest["corpus_digest"],
        "corpus_bundle_digest": two_bundle["bundle_digest"],
        "active_prefix_size": 2,
        "frozen_problem_count": 2,
        "source": two_manifest["sampling"]["source"],
        "seed": two_manifest["sampling"]["seed"],
        "count_policy": two_manifest["sampling"]["count_policy"],
        "limit_per_source": two_manifest["sampling"]["limit_per_source"],
        "order_policy": two_manifest["sampling"]["order_policy"],
        "repository_commit": two_manifest["sampling"]["dataset_revision"],
    })
    incomplete["records"][0]["corpus_digest"] = two_manifest["corpus_digest"]
    incomplete_results = copy.deepcopy(results)
    incomplete_results["problem_00"]["corpus_digest"] = \
        two_manifest["corpus_digest"]
    with pytest.raises(semantic_artifacts.ReplayCertificationError,
                       match="complete active record prefix"):
        semantic_artifacts.publish_run_report(
            "incomplete", incomplete, incomplete_results, two_manifest,
            corpus_bundle=two_bundle)


def test_semantic_tag_binding_refuses_reuse_and_clears_stale_certification(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        semantic_artifacts, "artifact_dir", lambda tag: str(tmp_path / tag))
    problem, manifest, bundle, payload, results = _observed_report_fixture()
    art = semantic_artifacts.publish_run_report(
        "bound", payload, results, manifest, corpus_bundle=bundle)

    for dirname in ("replay_specs", "replay_receipts"):
        directory = tmp_path / "bound" / dirname
        directory.mkdir()
        (directory / "problem_99.json").write_text("{}", encoding="utf-8")
    semantic_artifacts.atomic_json(
        os.path.join(art, "promoted_cones.json"),
        [{"opaque_id": "problem_99"}])
    track_reports = tmp_path / "bound" / "track_reports"
    track_reports.mkdir()
    immutable_report = track_reports / "semantic-pure_observed_shared_n1_r0.json"
    immutable_report.write_text('{"immutable":true}\n', encoding="utf-8")
    semantic_artifacts.publish_run_report(
        "bound", payload, results, manifest, corpus_bundle=bundle)
    assert json.loads((tmp_path / "bound" / "promoted_cones.json").read_text()) == []
    assert not list((tmp_path / "bound" / "replay_specs").glob("*.json"))
    assert not list((tmp_path / "bound" / "replay_receipts").glob("*.json"))
    assert immutable_report.read_text(encoding="utf-8") == \
        '{"immutable":true}\n'

    _, other_manifest, other_bundle, other_payload, other_results = \
        _observed_report_fixture(_problem(1))
    with pytest.raises(semantic_artifacts.ReplayCertificationError,
                       match="different arm"):
        semantic_artifacts.publish_run_report(
            "bound", other_payload, other_results, other_manifest,
            corpus_bundle=other_bundle)

    stale_control = P.build_shuffled_sides_control(
        [problem], manifest, seed=19).manifest
    semantic_artifacts.atomic_json(
        os.path.join(art, "control_manifest.json"), stale_control)
    with pytest.raises(semantic_artifacts.ReplayCertificationError,
                       match="control manifest contradicts"):
        semantic_artifacts.publish_run_report(
            "bound", payload, results, manifest, corpus_bundle=bundle)


def test_semantic_infrastructure_error_stays_pending_and_resumes(
        tmp_path, monkeypatch):
    problems = [_problem_128()]
    monkeypatch.setattr(
        run_semantic_cone.phase_d_protocol,
        "sample_corpus",
        lambda *args, **kwargs: problems,
    )
    monkeypatch.setattr(
        run_semantic_cone.semantic_replay, "BONGARD_ROOT", tmp_path)
    monkeypatch.setattr(
        semantic_artifacts, "artifact_dir",
        lambda tag: str(tmp_path / f"pending_{tag}"),
    )

    class FailingProposer:
        def __init__(self, model, max_tokens):
            pass

        def propose(self, problem_id, panel_paths):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        run_semantic_cone, "AnthropicCofiberedProposer", FailingProposer)
    out_dir = tmp_path / "pending_run"
    args = SimpleNamespace(
        proposer="anthropic",
        out_dir=str(out_dir),
        max_support_errors=0,
        max_loo_errors=0,
        max_rotated_loo_errors=0,
        limit=1,
        rounds=1,
        max_tokens=1,
        corpus_size=1,
        lambda_value=0.02,
        dataset_dir=str(tmp_path / "dataset"),
        seed=17,
        source="basic",
        condition=P.OBSERVED,
        control_seed=1,
        control_replicate=0,
        prepare_only=False,
        model="offline",
        tag="infra",
    )
    run_semantic_cone.run(args)
    pending = json.loads((out_dir / "checkpoint.json").read_text())
    assert pending["attempted"] == 0 and pending["records"] == []
    assert not (tmp_path / "pending_infra" / "checkpoint.json").exists()

    class EmptyProposer:
        def __init__(self, model, max_tokens):
            pass

        def propose(self, problem_id, panel_paths):
            return run_semantic_cone.ProposalBundle(
                problem_id, (), "no proposal", "offline-test")

    monkeypatch.setattr(
        run_semantic_cone, "AnthropicCofiberedProposer", EmptyProposer)
    run_semantic_cone.run(args)
    completed = json.loads((out_dir / "checkpoint.json").read_text())
    assert completed["attempted"] == 1
    assert completed["records"][0]["status"] == "NO_PROPOSALS"


def test_offline_phase_d_preparation_freezes_all_arms_and_controls(
        tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    problems = [_problem_128(index) for index in range(5)]
    monkeypatch.setattr(
        prepare_phase_d.protocol,
        "sample_corpus",
        lambda *args, **kwargs: problems,
    )
    monkeypatch.setattr(prepare_phase_d.semantic_replay, "BONGARD_ROOT", tmp_path)
    out_dir = tmp_path / "preregistered"
    args = SimpleNamespace(
        out_dir=str(out_dir),
        dataset_dir=str(tmp_path / "dataset"),
        limit_per_source=5,
        seed=17,
        source="basic",
        tracks=("UNRESTRICTED", "SEMANTIC-PURE"),
        scales=(1, 5),
        shuffled_seed=101,
        shuffled_replicates=2,
        no_share_tracks=("UNRESTRICTED",),
    )
    first = prepare_phase_d.prepare(args)
    second = prepare_phase_d.prepare(args)
    assert first == second
    assert first["problem_count"] == 5
    assert first["arm_count"] == 14
    assert first["bundle_digest"].startswith("sha256:")
    assert len(first["control_digests"]) == 2
    preregistration = json.loads(
        (out_dir / "phase_d_preregistration.json").read_text())
    corpus_manifest = json.loads(
        (out_dir / "corpus_manifest.json").read_text())
    P.validate_preregistration(
        preregistration, corpus_manifest=corpus_manifest)
    corpus_bundle = json.loads((out_dir / "corpus_panels.json").read_text())
    P.validate_corpus_bundle(corpus_bundle, corpus_manifest)
    for replicate in range(2):
        control_manifest = json.loads(
            (out_dir / f"shuffled_sides_r{replicate:02d}.json").read_text())
        P.validate_shuffled_control_manifest(control_manifest, corpus_manifest)
        binding = preregistration["shuffled_sides"]["controls"][replicate]
        assert binding["control_digest"] == control_manifest["control_digest"]

    # Keeping the advertised digest while modifying actual content used to be
    # silently accepted by _write_once.  The second preparation must now
    # validate and compare the complete canonical document.
    preregistration["arms"][0]["scale"] = 999
    (out_dir / "phase_d_preregistration.json").write_text(
        json.dumps(preregistration))
    with pytest.raises(SystemExit, match="existing.*artifact is invalid"):
        prepare_phase_d.prepare(args)


def test_phase_d_preparation_rejects_random_hash_seed_before_mutation(
        tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setattr(
        prepare_phase_d.protocol, "sample_corpus",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("hash-seed preflight must precede sampling")))
    out_dir = tmp_path / "must-not-exist"
    args = SimpleNamespace(
        out_dir=str(out_dir), dataset_dir=str(tmp_path / "dataset"),
        limit_per_source=1, seed=17, source="basic",
        tracks=("SEMANTIC-PURE",), scales=(1,), shuffled_seed=101,
        shuffled_replicates=1, no_share_tracks=(),
    )
    with pytest.raises(SystemExit, match="PYTHONHASHSEED=0"):
        prepare_phase_d.prepare(args)
    assert not out_dir.exists()


def test_phase_d_preparation_rejects_symlink_destination_without_writes(
        tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    problems = [_problem_128()]
    monkeypatch.setattr(
        prepare_phase_d.protocol, "sample_corpus",
        lambda *args, **kwargs: problems)
    monkeypatch.setattr(prepare_phase_d.semantic_replay, "BONGARD_ROOT", tmp_path)
    out_dir = tmp_path / "preregistered"
    out_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("sentinel")
    destination = out_dir / "phase_d_preregistration.json"
    destination.symlink_to(outside)
    args = SimpleNamespace(
        out_dir=str(out_dir), dataset_dir=str(tmp_path / "dataset"),
        limit_per_source=1, seed=17, source="basic",
        tracks=("SEMANTIC-PURE",), scales=(1,), shuffled_seed=101,
        shuffled_replicates=1, no_share_tracks=(),
    )
    with pytest.raises(SystemExit, match="existing.*artifact is invalid"):
        prepare_phase_d.prepare(args)
    assert outside.read_text() == "sentinel"
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "phase_d_preregistration.json"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
@pytest.mark.parametrize("kind", ("hardlink", "fifo", "oversize"))
def test_phase_d_write_once_preflight_rejects_unsafe_existing_files(
        tmp_path, kind):
    destination = tmp_path / "phase_d_preregistration.json"
    if kind == "hardlink":
        source = tmp_path / "second-name.json"
        source.write_text("{}\n", encoding="utf-8")
        os.link(source, destination)
    elif kind == "fifo":
        os.mkfifo(destination)
    else:
        with open(destination, "wb") as handle:
            handle.truncate(prepare_phase_d.artifact_io.MAX_JSON_BYTES + 1)

    with pytest.raises(SystemExit, match="existing.*artifact is invalid"):
        prepare_phase_d._preflight_write_once(
            str(destination), {}, lambda value: None)


def test_phase_d_write_once_preflight_rejects_path_replacement_during_read(
        tmp_path, monkeypatch):
    destination = tmp_path / "phase_d_preregistration.json"
    payload = {"stable": True}
    encoded = semantic_replay.canonical_json_bytes(payload) + b"\n"
    destination.write_bytes(encoded)
    replacement = tmp_path / "replacement.json"
    original_lstat = os.lstat
    replaced = False

    def replace_before_identity_check(path, *args, **kwargs):
        nonlocal replaced
        if os.path.abspath(os.fspath(path)) == os.path.abspath(destination) \
                and not replaced:
            replaced = True
            replacement.write_bytes(encoded)
            os.replace(replacement, destination)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(
        prepare_phase_d.artifact_io.os, "lstat",
        replace_before_identity_check)
    with pytest.raises(SystemExit, match="changed while being read"):
        prepare_phase_d._preflight_write_once(
            str(destination), payload, lambda value: None)
    assert replaced
