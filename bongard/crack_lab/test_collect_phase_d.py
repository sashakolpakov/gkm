"""Adversarial tests for artifact-certified Phase D campaign collection."""
from __future__ import annotations

import copy
import json
import os
import shutil
import stat
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bongard_legs as BL
import codex_proposer
import collect_phase_d as C
import dataset
import phase_d_protocol as P
import run_semantic_cone as R
import semantic_artifacts as SA
import semantic_replay


def _panel(side_signal: int, identity: int) -> np.ndarray:
    panel = np.zeros((8, 8), dtype=np.uint8)
    panel[0, 0] = side_signal
    panel.flat[8 + identity % 56] = 1
    return panel


def _problem(index: int) -> dataset.Problem:
    return dataset.Problem(
        problem_id=f"harness-secret-{index}",
        category="basic",
        concept=f"harness-concept-{index}",
        pos=tuple(_panel(1, index * 12 + offset) for offset in range(6)),
        neg=tuple(_panel(0, index * 12 + 6 + offset) for offset in range(6)),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(semantic_replay.canonical_json_bytes(value) + b"\n")


def _codex_receipt(
        preregistration: dict, *, opaque_id: str,
        panel_set_digest: str,
        current_source: str = BL.INITIAL_LIBRARY_SOURCE,
        current_log: str = "",
        proposed_source: str = BL.INITIAL_LIBRARY_SOURCE,
        proposed_log: str = "",
        input_tokens: int = 10, identity_scope: str = "fixture") -> dict:
    execution_policy = preregistration["execution_policy"]
    unrestricted = execution_policy["unrestricted"]
    runtime = execution_policy["runtime"]["codex_cli"]
    model = unrestricted["proposer_ladder"][0]
    task = BL.build_task(opaque_id, "")
    prompt = codex_proposer._predicate_prompt(
        task, current_source, current_log)
    identity = BL._source_digest(f"{identity_scope}:{opaque_id}")
    thread_id = (
        f"{identity[:8]}-{identity[8:12]}-4{identity[13:16]}-"
        f"8{identity[17:20]}-{identity[20:32]}")
    receipt = {
        "schema": P.PROPOSER_RECEIPT_SCHEMA,
        "source": "codex-cli",
        "requested_model": model,
        "reported_model": model,
        "model_identity_evidence": "jsonl-reported-model",
        "requested_reasoning_effort": unrestricted[
            "requested_reasoning_effort"],
        "input_tokens": input_tokens,
        "cached_input_tokens": 3,
        "output_tokens": 2,
        "reasoning_output_tokens": 1,
        "thread_id": thread_id,
        "codex_cli_version": runtime["version"],
        "codex_launcher_digest": runtime["launcher_digest"],
        "task_digest": BL._source_digest(task),
        "current_source_digest": BL._source_digest(current_source),
        "current_log_digest": BL._source_digest(current_log),
        "prompt_digest": BL._source_digest(prompt),
        "input_digest_schema": codex_proposer.PREDICATE_INPUT_DIGEST_SCHEMA,
        "input_digest": BL._canonical_digest({
            "fixture": "predicate-input", "opaque_id": opaque_id,
            "panel_set_digest": panel_set_digest,
            "current_source_digest": BL._source_digest(current_source),
            "current_log_digest": BL._source_digest(current_log),
        }),
        "output_schema_digest": unrestricted[
            "proposer_output_schema_digest"],
        "panel_view_digest": BL._canonical_digest({
            "fixture": "panel-view", "panel_set_digest": panel_set_digest}),
        "panel_set_digest": panel_set_digest,
        "structured_output_digest": (
            codex_proposer.predicate_proposer_output_digest(
                proposed_source, proposed_log, "fixture")),
        "proposed_source_digest": BL._source_digest(proposed_source),
        "proposed_log_digest": BL._source_digest(proposed_log),
        "event_stream_digest": BL._source_digest(
            f"event:{thread_id}:{panel_set_digest}"),
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


def _control_for(
        problems: list[dataset.Problem], manifest: dict,
        preregistration: dict, arm: dict) -> dict | None:
    if arm["condition"] != P.SHUFFLED_SIDES:
        return None
    return P.build_shuffled_sides_control(
        problems, manifest,
        seed=preregistration["shuffled_sides"]["seed"],
        replicate=arm["replicate"],
    ).manifest


def _unrestricted_report(
        *, tag: str, condition: str, count: int, manifest: dict,
        bundle: dict, control: dict | None, preregistration: dict,
        arm: dict) -> BL.Report:
    fingerprint = BL._verifier_fingerprint()
    fingerprint_digest = fingerprint["fingerprint_digest"]
    baseline_source = BL.INITIAL_LIBRARY_SOURCE
    baseline_digest = BL._source_digest(baseline_source)
    context = BL._pricing_context(P.SHARED, (), baseline_digest)
    panel_digests = (
        [entry["controlled_panel_set_digest"] for entry in control["problems"]]
        if control is not None else
        [entry["panel_set_digest"] for entry in manifest["problems"]]
    )
    binding_history = P.execution_binding_family(preregistration, arm)
    model = preregistration["execution_policy"]["unrestricted"][
        "proposer_ladder"][0]
    records = [
        BL.ProblemRecord(
            opaque_id=f"problem_{index:02d}",
            solved=False,
            heldout_accuracy=0.0,
            rule="PRICING_OR_LOAD_ERROR",
            rule_cost=0.0,
            marginal_C=0,
            model=model,
            attempts=1,
            escalated=False,
            phase_execution_binding_digest=next(
                binding["binding_digest"] for binding in binding_history
                if index < binding["scale"]),
            proposer_receipts=[_codex_receipt(
                preregistration,
                opaque_id=f"problem_{index:02d}",
                panel_set_digest=panel_digests[index],
                current_source=baseline_source,
                proposed_source=baseline_source,
                identity_scope=tag)],
            proposer_feedback=[""],
            proposer_panel_set_digest=panel_digests[index],
            baseline_log_digest=BL._source_digest(""),
            attempted_log_digest=BL._source_digest(""),
            status=BL.VERIFIER_FAILURE_STATUS,
            track="UNRESTRICTED",
            condition=condition,
            sharing_policy=P.SHARED,
            corpus_digest=manifest["corpus_digest"],
            panel_set_digest=panel_digests[index],
            control_digest=(control["control_digest"] if control else ""),
            label_policy=condition,
            baseline_source_digest=baseline_digest,
            attempted_source_digest=baseline_digest,
            attempted_source=baseline_source,
            pricing_context_digest=context["context_digest"],
            verification_digest="a" * 64,
            source_verification_digest="a" * 64,
            train_accuracy=0.0,
            predicate_errors=12,
            n_rotations=36,
            verifier_fingerprint_digest=fingerprint_digest,
        )
        for index in range(count)
    ]
    for record in records:
        verification_digest = BL._verification_digest(
            BL._verification_failure(P.SHARED),
            source_digest=record.attempted_source_digest,
            pricing_context_digest=record.pricing_context_digest,
            proposer_receipts_digest=BL._proposer_receipts_digest(
                record.proposer_receipts),
        )
        record.verification_digest = verification_digest
        record.source_verification_digest = verification_digest
    report = BL.Report(
        tag=tag,
        records=records,
        condition=condition,
        label_policy=condition,
        corpus_digest=manifest["corpus_digest"],
        corpus_bundle_digest=bundle["bundle_digest"],
        control_digest=(control["control_digest"] if control else ""),
        verifier_fingerprint=fingerprint,
        phase_execution_binding=binding_history[-1],
        phase_execution_binding_history=binding_history,
    )
    report.source_trace_digest = BL._source_trace_digest(report.records)
    BL._validate_priced_report(report)
    return report


def _unrestricted_results(
        report: BL.Report, problems: list[dataset.Problem]) -> dict:
    return {
        record.opaque_id: {
            "problem_id": problems[index].problem_id,
            "category": problems[index].category,
            "concept": problems[index].concept,
            **BL._record_result_evidence(record),
        }
        for index, record in enumerate(report.records)
    }


def _write_unrestricted_artifact(
        root: Path, report: BL.Report, manifest: dict, bundle: dict,
        problems: list[dataset.Problem], control: dict | None = None) -> Path:
    artifact = root / f"{report.tag}_predicates"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "predicates.py").write_text(BL.INITIAL_LIBRARY_SOURCE)
    (artifact / "predicates_log.md").write_text("")
    _write_json(artifact / "corpus_manifest.json", manifest)
    _write_json(artifact / "corpus_panels.json", bundle)
    if control is not None:
        _write_json(artifact / "control_manifest.json", control)
    _write_json(artifact / "results.json", _unrestricted_results(report, problems))
    _write_json(artifact / "checkpoint.json", report.to_json())
    return artifact


def _semantic_records(
        *, condition: str, count: int, manifest: dict,
        control: dict | None, preregistration: dict,
        arm: dict) -> list[R.ProblemResult]:
    panel_digests = (
        [entry["controlled_panel_set_digest"] for entry in control["problems"]]
        if control is not None else
        [entry["panel_set_digest"] for entry in manifest["problems"]]
    )
    binding_history = P.execution_binding_family(preregistration, arm)
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
    terminal_evidence = {
        "schema": R.TERMINAL_EVIDENCE_SCHEMA,
        "proposal_outcome": "NO_PROPOSALS",
        "rounds": [{
            "round": 0,
            "proposer_kind": "anthropic",
            "parse_error": "",
            "candidate_count": 0,
            "candidate_ids": [],
            "hypothesis_digests": [],
            "model_receipts": [receipt],
        }],
        "selection": {},
    }
    return [
        R.ProblemResult(
            opaque_id=f"problem_{index:02d}",
            category=manifest["problems"][index]["category"],
            solved=False,
            selected_hypothesis="",
            selected_description="",
            selected_rule="",
            support_errors=0,
            loo_errors=0,
            rotated_loo_errors=0,
            rotated_loo_checks=0,
            n_examples=12,
            complexity=0,
            rounds_used=1,
            proposer_kind="anthropic",
            track="SEMANTIC-PURE",
            condition=condition,
            sharing_policy=P.SHARED,
            corpus_digest=manifest["corpus_digest"],
            panel_set_digest=panel_digests[index],
            control_digest=(control["control_digest"] if control else ""),
            status="NO_PROPOSALS",
            proposer_error="",
            candidates=[],
            candidate_manifest=[],
            terminal_evidence=copy.deepcopy(terminal_evidence),
            terminal_evidence_digest=(
                semantic_replay.canonical_json_digest(terminal_evidence)),
            phase_execution_binding_digest=next(
                binding["binding_digest"] for binding in binding_history
                if index < binding["scale"]),
        )
        for index in range(count)
    ]


def _semantic_checkpoint(
        *, tag: str, condition: str, records: list[R.ProblemResult],
        manifest: dict, bundle: dict, control: dict | None,
        preregistration: dict, arm: dict) -> dict:
    args = SimpleNamespace(
        condition=condition,
        proposer="anthropic",
        model="sonnet",
        max_tokens=8000,
        rounds=4,
        tag=tag,
        source=manifest["sampling"]["source"],
        seed=manifest["sampling"]["seed"],
        limit=manifest["sampling"]["limit_per_source"],
        max_support_errors=0,
        max_loo_errors=0,
        max_rotated_loo_errors=0,
        lambda_value=0.02,
        phase_execution_binding=P.execution_binding(
            preregistration, arm["arm_id"]),
        phase_predecessor_execution_binding={},
        phase_execution_binding_history=P.execution_binding_family(
            preregistration, arm),
        phase_python_hash_runtime=R._phase_python_hash_runtime(
            preregistration["execution_policy"]),
    )
    payload = R._checkpoint_payload(
        args, records, manifest, len(records), control, bundle)
    payload["artifact_state"] = "RUN_COMPLETE"
    return payload


def _write_semantic_artifact(
        root: Path, tag: str, checkpoint: dict, manifest: dict,
        bundle: dict, problems: list[dataset.Problem],
        control: dict | None = None) -> Path:
    artifact = root / f"{tag}_semantic"
    artifact.mkdir(parents=True, exist_ok=True)
    results = {
        record["opaque_id"]: R._result_payload(
            problems[index], R.ProblemResult(**record))
        for index, record in enumerate(checkpoint["records"])
    }
    binding = SA._artifact_binding(checkpoint, manifest, bundle, control)
    _write_json(artifact / "artifact_binding.json", binding)
    _write_json(artifact / "corpus_manifest.json", manifest)
    _write_json(artifact / "corpus_panels.json", bundle)
    if control is not None:
        _write_json(artifact / "control_manifest.json", control)
    _write_json(artifact / "results.json", results)
    _write_json(artifact / "promoted_cones.json", [])
    _write_json(artifact / "checkpoint.json", checkpoint)
    return artifact


def _track_report_from_unrestricted(
        preregistration: dict, arm: dict, report: BL.Report) -> dict:
    records = []
    for record in report.records[:arm["scale"]]:
        value = asdict(record)
        value["runner_condition"] = value["condition"]
        value["condition"] = arm["condition"]
        value["label_policy"] = arm["label_policy"]
        value["sharing_policy"] = arm["sharing_policy"]
        records.append(value)
    source_trace = "sha256:" + BL._source_trace_digest(
        report.records[:arm["scale"]])
    parent = (
        "sha256:" + report.parent_source_trace_digest
        if report.parent_source_trace_digest else "")
    return P.build_track_report(
        preregistration, arm_id=arm["arm_id"], records=records,
        report_source_trace_digest=source_trace,
        parent_source_trace_digest=parent,
    )


def _track_report_from_semantic(
        preregistration: dict, arm: dict, checkpoint: dict) -> dict:
    runner_records = checkpoint["records"][:arm["scale"]]
    records = []
    for record in runner_records:
        value = copy.deepcopy(record)
        value["runner_condition"] = value["condition"]
        value["condition"] = arm["condition"]
        value["label_policy"] = arm["label_policy"]
        value["sharing_policy"] = arm["sharing_policy"]
        records.append(value)
    return P.build_track_report(
        preregistration, arm_id=arm["arm_id"], records=records,
        report_source_trace_digest=
        semantic_replay.canonical_json_digest(runner_records),
    )


@pytest.fixture
def certified_campaign(tmp_path, monkeypatch):
    # Only the expensive verifier execution is stubbed.  Artifact paths,
    # schemas, corpus/control reconstruction, accounting, and record equality
    # all run through the production validators.
    monkeypatch.setattr(BL, "_cold_replay_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        R, "_replay_terminal_record", lambda *args, **kwargs: {})

    problems = [_problem(index) for index in range(25)]
    manifest = P.build_corpus_manifest(
        problems,
        source="basic",
        seed=20260709,
        limit_per_source=25,
        panel_size=8,
        dataset_revision="a" * 40,
    )
    bundle = P.build_corpus_bundle(problems, manifest)
    preregistration = P.build_preregistration(
        manifest,
        tracks=["UNRESTRICTED", "SEMANTIC-PURE"],
        scales=[1, 5, 25],
        shuffled_seed=73,
        shuffled_replicates=3,
    )
    assert len(preregistration["arms"]) == 27
    root = tmp_path / "agent_solutions"
    root.mkdir()
    checkpoints: dict[str, object] = {}
    artifacts: dict[str, Path] = {}

    # One growing artifact backs every scale in a primary/shuffled family.
    families: dict[str, dict] = {}
    for arm in preregistration["arms"]:
        if arm["condition"] != P.NO_SHARE:
            families[arm["execution_tag"]] = arm
    for tag, representative in families.items():
        control = _control_for(
            problems, manifest, preregistration, representative)
        condition = (
            P.OBSERVED if representative["condition"] == "primary"
            else representative["condition"])
        if representative["track"] == "UNRESTRICTED":
            checkpoint = _unrestricted_report(
                tag=tag, condition=condition, count=25,
                manifest=manifest, bundle=bundle, control=control,
                preregistration=preregistration, arm=representative)
            artifacts[tag] = _write_unrestricted_artifact(
                root, checkpoint, manifest, bundle, problems, control)
        else:
            records = _semantic_records(
                condition=condition, count=25,
                manifest=manifest, control=control,
                preregistration=preregistration, arm=representative)
            checkpoint = _semantic_checkpoint(
                tag=tag, condition=condition, records=records,
                manifest=manifest, bundle=bundle, control=control,
                preregistration=preregistration, arm=representative)
            artifacts[tag] = _write_semantic_artifact(
                root, tag, checkpoint, manifest, bundle, problems, control)
        checkpoints[tag] = checkpoint

    primary_arm = next(
        arm for arm in preregistration["arms"]
        if arm["track"] == "UNRESTRICTED"
        and arm["condition"] == "primary")
    primary_checkpoint = checkpoints[primary_arm["execution_tag"]]
    assert isinstance(primary_checkpoint, BL.Report)
    for arm in preregistration["arms"]:
        if arm["condition"] != P.NO_SHARE:
            continue
        repriced = BL.reprice_no_share(
            primary_checkpoint,
            tag=arm["execution_tag"],
            max_problems=arm["scale"],
            phase_execution_binding=P.execution_binding(
                preregistration, arm["arm_id"]),
        )
        artifacts[arm["execution_tag"]] = _write_unrestricted_artifact(
            root, repriced, manifest, bundle, problems)
        checkpoints[arm["execution_tag"]] = repriced

    reports: dict[str, dict] = {}
    report_paths: dict[str, Path] = {}
    for arm in preregistration["arms"]:
        checkpoint = checkpoints[arm["execution_tag"]]
        if arm["track"] == "UNRESTRICTED":
            assert isinstance(checkpoint, BL.Report)
            report = _track_report_from_unrestricted(
                preregistration, arm, checkpoint)
        else:
            assert isinstance(checkpoint, dict)
            report = _track_report_from_semantic(
                preregistration, arm, checkpoint)
        path = artifacts[arm["execution_tag"]] / "track_reports" / (
            arm["arm_id"].replace(":", "__") + ".json")
        _write_json(path, report)
        reports[arm["arm_id"]] = report
        report_paths[arm["arm_id"]] = path

    prereg_path = tmp_path / "phase_d_preregistration.json"
    _write_json(prereg_path, preregistration)
    return {
        "preregistration": preregistration,
        "prereg_path": prereg_path,
        "root": root,
        "artifacts": artifacts,
        "checkpoints": checkpoints,
        "reports": reports,
        "report_paths": report_paths,
        "report_dirs": sorted({
            path.parent for path in report_paths.values()}, key=str),
        "manifest": manifest,
        "bundle": bundle,
        "problems": problems,
    }


def _collect(fixture: dict, output: Path) -> dict:
    return C.collect_campaign(
        str(fixture["prereg_path"]),
        [str(path) for path in fixture["report_dirs"]],
        str(output),
    )


def test_exact_certified_default_27_arm_collection(certified_campaign, tmp_path):
    campaign = _collect(certified_campaign, tmp_path / "campaign.json")
    preregistration = certified_campaign["preregistration"]
    assert campaign["schema"] == C.CAMPAIGN_SCHEMA
    assert campaign["schema"] == "bongard.phase-d-campaign/v6"
    assert campaign["arm_count"] == 27
    assert [report["arm_id"] for report in campaign["reports"]] == [
        arm["arm_id"] for arm in preregistration["arms"]]
    assert [report["execution_tag"] for report in campaign["reports"]] == [
        arm["execution_tag"] for arm in preregistration["arms"]]
    assert [item["arm_id"] for item in campaign[
            "artifact_certifications"]] == [
        arm["arm_id"] for arm in preregistration["arms"]]
    assert all(
        item["certification_digest"] ==
        semantic_replay.canonical_json_digest({
            key: value for key, value in item.items()
            if key != "certification_digest"})
        for item in campaign["artifact_certifications"])
    assert len(campaign["aggregates"]) == 15
    assert all(
        cell["unsolved"] ==
        cell["ordinary_unsolved"] + cell["verifier_failures"]
        for cell in campaign["aggregates"])
    assert campaign["campaign_digest"] == \
        semantic_replay.canonical_json_digest({
            key: value for key, value in campaign.items()
            if key != "campaign_digest"})
    C.validate_campaign_artifact(campaign, preregistration)

    tampered = copy.deepcopy(campaign)
    tampered["artifact_certifications"][0]["checkpoint_digest"] = \
        "sha256:" + "0" * 64
    tampered["campaign_digest"] = semantic_replay.canonical_json_digest({
        key: value for key, value in tampered.items()
        if key != "campaign_digest"})
    with pytest.raises(C.CampaignCollectionError,
                       match="certification digest"):
        C.validate_campaign_artifact(tampered, preregistration)


def test_collector_rejects_codex_turn_identity_reused_by_independent_arm(
        certified_campaign, tmp_path):
    preregistration = certified_campaign["preregistration"]
    primary_arm = next(
        arm for arm in preregistration["arms"]
        if arm["track"] == "UNRESTRICTED"
        and arm["condition"] == "primary")
    target_arm = next(
        arm for arm in preregistration["arms"]
        if arm["track"] == "UNRESTRICTED"
        and arm["condition"] == P.SHUFFLED_SIDES
        and arm["replicate"] == 0)
    source = certified_campaign["checkpoints"][primary_arm["execution_tag"]]
    target = copy.deepcopy(
        certified_campaign["checkpoints"][target_arm["execution_tag"]])
    assert isinstance(source, BL.Report) and isinstance(target, BL.Report)

    source_receipt = source.records[0].proposer_receipts[0]
    target_receipt = target.records[0].proposer_receipts[0]
    target_receipt["thread_id"] = source_receipt["thread_id"]
    target_receipt["event_stream_digest"] = source_receipt[
        "event_stream_digest"]
    target_receipt["receipt_digest"] = BL._canonical_digest({
        key: value for key, value in target_receipt.items()
        if key != "receipt_digest"
    })
    target.source_trace_digest = BL._source_trace_digest(target.records)
    BL._validate_priced_report(target)

    artifact = certified_campaign["artifacts"][target_arm["execution_tag"]]
    _write_json(artifact / "checkpoint.json", target.to_json())
    _write_json(
        artifact / "results.json",
        _unrestricted_results(target, certified_campaign["problems"]),
    )
    for arm in preregistration["arms"]:
        if arm["execution_tag"] != target_arm["execution_tag"]:
            continue
        report = _track_report_from_unrestricted(
            preregistration, arm, target)
        _write_json(certified_campaign["report_paths"][arm["arm_id"]], report)

    with pytest.raises(
            C.CampaignCollectionError,
            match="reuse Codex turn identity evidence"):
        _collect(certified_campaign, tmp_path / "duplicate-turn.json")


def test_free_floating_self_hashed_report_is_rejected(
        certified_campaign, tmp_path):
    arm = certified_campaign["preregistration"]["arms"][0]
    report = certified_campaign["reports"][arm["arm_id"]]
    floating = tmp_path / "reports" / (
        arm["arm_id"].replace(":", "__") + ".json")
    _write_json(floating, report)
    with pytest.raises(C.CampaignCollectionError, match="origin.*execution-tag"):
        C.discover_track_reports(
            [str(floating.parent)], certified_campaign["preregistration"])


def test_artifact_snapshot_open_is_nonblocking_against_fifo_swap(
        tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    destination = tmp_path / "snapshot" / "source.json"
    source.write_text("{}")
    observed = []
    real_open = C.os.open

    def recording_open(path, flags, *args, **kwargs):
        if os.fspath(path) == os.fspath(source):
            observed.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(C.os, "open", recording_open)
    C._copy_stable_regular_file(str(source), str(destination))
    assert destination.read_text() == "{}"
    assert observed
    if hasattr(os, "O_NONBLOCK"):
        assert all(flags & os.O_NONBLOCK for flags in observed)
    if hasattr(os, "O_CLOEXEC"):
        assert all(flags & os.O_CLOEXEC for flags in observed)


def test_wrong_execution_tag_artifact_origin_is_rejected(
        certified_campaign, tmp_path):
    arm = certified_campaign["preregistration"]["arms"][0]
    wrong = tmp_path / "wrong-tag_predicates" / "track_reports" / (
        arm["arm_id"].replace(":", "__") + ".json")
    _write_json(wrong, certified_campaign["reports"][arm["arm_id"]])
    with pytest.raises(C.CampaignCollectionError, match="origin.*execution-tag"):
        C.discover_track_reports(
            [str(wrong.parent)], certified_campaign["preregistration"])


def test_tampered_artifact_evidence_is_rejected(certified_campaign):
    arm = next(
        arm for arm in certified_campaign["preregistration"]["arms"]
        if arm["track"] == "SEMANTIC-PURE"
        and arm["condition"] == "primary"
        and arm["scale"] == 1)
    artifact = certified_campaign["artifacts"][arm["execution_tag"]]
    bundle = json.loads((artifact / "corpus_panels.json").read_text())
    bundle["problems"][0]["panels"][0]["data_base64"] = "AAAA"
    _write_json(artifact / "corpus_panels.json", bundle)
    with pytest.raises(C.CampaignCollectionError, match="corpus evidence"):
        C.discover_track_reports(
            [str(artifact / "track_reports")],
            certified_campaign["preregistration"],
        )


def test_unrestricted_pending_promotion_survives_snapshot_and_is_rejected(
        certified_campaign):
    arm = next(
        arm for arm in certified_campaign["preregistration"]["arms"]
        if arm["track"] == "UNRESTRICTED"
        and arm["condition"] == "primary" and arm["scale"] == 1)
    artifact = certified_campaign["artifacts"][arm["execution_tag"]]
    checkpoint = certified_campaign["checkpoints"][arm["execution_tag"]]
    assert isinstance(checkpoint, BL.Report)
    pending = BL._pending_promotion_payload(checkpoint, b"")
    _write_json(artifact / BL.PENDING_CHECKPOINT_FILE, pending)

    with pytest.raises(
            C.CampaignCollectionError, match="incomplete staged promotion"):
        C.discover_track_reports(
            [str(artifact / "track_reports")],
            certified_campaign["preregistration"],
        )


def test_semantic_checkpoint_sampling_provenance_is_preregistered(
        certified_campaign):
    arm = next(
        arm for arm in certified_campaign["preregistration"]["arms"]
        if arm["track"] == "SEMANTIC-PURE"
        and arm["condition"] == "primary" and arm["scale"] == 1)
    artifact = certified_campaign["artifacts"][arm["execution_tag"]]
    checkpoint = json.loads((artifact / "checkpoint.json").read_text())
    checkpoint["dataset"]["source"] = "forged-source"
    _write_json(artifact / "checkpoint.json", checkpoint)
    with pytest.raises(C.CampaignCollectionError, match="sampling provenance"):
        C.discover_track_reports(
            [str(artifact / "track_reports")],
            certified_campaign["preregistration"],
        )


def test_collector_rejects_test_injected_unrestricted_receipt(
        certified_campaign):
    arm = next(
        arm for arm in certified_campaign["preregistration"]["arms"]
        if arm["track"] == "UNRESTRICTED"
        and arm["condition"] == "primary" and arm["scale"] == 1)
    artifact = certified_campaign["artifacts"][arm["execution_tag"]]
    checkpoint = copy.deepcopy(
        certified_campaign["checkpoints"][arm["execution_tag"]])
    assert isinstance(checkpoint, BL.Report)
    model = certified_campaign["preregistration"]["execution_policy"][
        "unrestricted"]["proposer_ladder"][0]
    checkpoint.records[0].proposer_receipts = [BL._build_proposer_receipt(
        source="test-injected", requested_model=model,
        actual_model="test-injected", input_tokens=0, output_tokens=0,
        model_usage={}, outcome="test-injected", permission_denials=())]
    checkpoint.source_trace_digest = BL._source_trace_digest(checkpoint.records)
    BL._validate_priced_report(checkpoint)
    _write_json(artifact / "checkpoint.json", checkpoint.to_json())
    with pytest.raises(C.CampaignCollectionError, match="test-injected"):
        C._load_unrestricted_checkpoint(
            str(artifact), arm, certified_campaign["preregistration"])


def test_snapshot_rechecks_every_source_after_copy(tmp_path, monkeypatch):
    source = tmp_path / "run_predicates"
    source.mkdir()
    checkpoint = source / "checkpoint.json"
    checkpoint.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "snapshots"
    destination.mkdir()
    original = C._copy_stable_regular_file
    mutated = False

    def copy_then_mutate(
            source_path, destination_path, *, maximum_bytes=C.MAX_JSON_BYTES):
        nonlocal mutated
        original(
            source_path, destination_path, maximum_bytes=maximum_bytes)
        if not mutated:
            mutated = True
            Path(source_path).write_text('{"changed":true}\n', encoding="utf-8")

    monkeypatch.setattr(C, "_copy_stable_regular_file", copy_then_mutate)
    with pytest.raises(C.CampaignCollectionError, match="changed after copying"):
        C._snapshot_artifact(str(source), str(destination), "UNRESTRICTED")


def test_symlinked_predicate_source_is_rejected(
        certified_campaign, tmp_path):
    arm = next(
        arm for arm in certified_campaign["preregistration"]["arms"]
        if arm["track"] == "UNRESTRICTED"
        and arm["condition"] == "primary"
        and arm["scale"] == 1)
    artifact = certified_campaign["artifacts"][arm["execution_tag"]]
    source = artifact / "predicates.py"
    source.unlink()
    replacement = tmp_path / "outside-predicates.py"
    replacement.write_text(BL.INITIAL_LIBRARY_SOURCE)
    source.symlink_to(replacement)
    with pytest.raises(C.CampaignCollectionError, match="regular file"):
        C.discover_track_reports(
            [str(artifact / "track_reports")],
            certified_campaign["preregistration"],
        )


def test_rehashed_report_that_differs_from_checkpoint_is_rejected(
        certified_campaign):
    arm = next(
        arm for arm in certified_campaign["preregistration"]["arms"]
        if arm["track"] == "SEMANTIC-PURE"
        and arm["condition"] == "primary"
        and arm["scale"] == 1)
    path = certified_campaign["report_paths"][arm["arm_id"]]
    report = copy.deepcopy(certified_campaign["reports"][arm["arm_id"]])
    report["records"][0]["fabricated_evidence"] = "self-hashed"
    trace = P._report_source_trace_digest(report["track"], report["records"])
    report["report_source_trace_digest"] = trace
    report["records"][0]["report_source_trace_digest"] = trace
    P.validate_track_report(report, certified_campaign["preregistration"])
    _write_json(path, report)
    with pytest.raises(C.CampaignCollectionError, match="records differ"):
        C.discover_track_reports(
            [str(path.parent)], certified_campaign["preregistration"])


def test_internally_valid_no_share_must_equal_exact_primary_reprice(
        certified_campaign):
    arm = next(
        arm for arm in certified_campaign["preregistration"]["arms"]
        if arm["condition"] == P.NO_SHARE and arm["scale"] == 1)
    artifact = certified_campaign["artifacts"][arm["execution_tag"]]
    checkpoint = copy.deepcopy(
        certified_campaign["checkpoints"][arm["execution_tag"]])
    assert isinstance(checkpoint, BL.Report)
    record = checkpoint.records[0]
    checkpoint.records[0].proposer_receipts = [_codex_receipt(
        certified_campaign["preregistration"],
        opaque_id=record.opaque_id,
        panel_set_digest=record.panel_set_digest,
        current_source=record.attempted_source,
        proposed_source=record.attempted_source,
        input_tokens=11)]
    checkpoint.source_trace_digest = BL._source_trace_digest(checkpoint.records)
    BL._validate_priced_report(checkpoint)
    _write_json(artifact / "checkpoint.json", checkpoint.to_json())
    _write_json(
        artifact / "results.json",
        _unrestricted_results(checkpoint, certified_campaign["problems"]),
    )
    report = _track_report_from_unrestricted(
        certified_campaign["preregistration"], arm, checkpoint)
    path = certified_campaign["report_paths"][arm["arm_id"]]
    _write_json(path, report)
    with pytest.raises(C.CampaignCollectionError, match="exact primary-prefix reprice"):
        C.discover_track_reports(
            [str(path.parent)], certified_campaign["preregistration"])


def test_solved_semantic_artifact_is_fresh_replayed_without_mutation(
        tmp_path, monkeypatch):
    from test_semantic_cone import (
        _object_count_hypothesis, _terminal_candidate_fixture)

    problem, record = _terminal_candidate_fixture()
    hypothesis = _object_count_hypothesis()
    registry = R.default_registry()
    verification = R.verify_hypothesis(hypothesis, registry, problem)
    manifest = P.build_corpus_manifest(
        [problem], source="basic", seed=17, limit_per_source=1,
        dataset_revision="unavailable")
    bundle = P.build_corpus_bundle([problem], manifest)
    preregistration = P.build_preregistration(
        manifest, tracks=["SEMANTIC-PURE"], scales=[1],
        shuffled_seed=73, shuffled_replicates=1)
    arm = next(
        item for item in preregistration["arms"]
        if item["condition"] == "primary")
    binding_history = P.execution_binding_family(preregistration, arm)
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
    terminal_evidence = copy.deepcopy(record.terminal_evidence)
    for round_record in terminal_evidence["rounds"]:
        round_record["proposer_kind"] = "anthropic"
        round_record["model_receipts"] = [receipt]
    record = replace(
        record,
        proposer_kind="anthropic",
        terminal_evidence=terminal_evidence,
        terminal_evidence_digest=semantic_replay.canonical_json_digest(
            terminal_evidence),
        phase_execution_binding_digest=binding_history[0]["binding_digest"],
    )
    tag = arm["execution_tag"]
    artifact = tmp_path / "agent_solutions" / f"{tag}_semantic"
    artifact.mkdir(parents=True)
    monkeypatch.setattr(semantic_replay, "BONGARD_ROOT", tmp_path)
    args = SimpleNamespace(
        condition=P.OBSERVED, proposer="anthropic", model="sonnet",
        max_tokens=8000, rounds=4, max_support_errors=0,
        max_loo_errors=0, max_rotated_loo_errors=0, lambda_value=0.02,
        source="basic", seed=17, limit=1, tag=tag,
        phase_execution_binding=binding_history[-1],
        phase_predecessor_execution_binding={},
        phase_execution_binding_history=binding_history,
        phase_python_hash_runtime=R._phase_python_hash_runtime(
            preregistration["execution_policy"]),
    )
    origins = [{
        "round": 0,
        "round_candidate_index": 0,
        "round_candidate_count": 1,
    }]
    spec = R._write_replay_spec(
        args, str(artifact), "problem_00", problem,
        hypothesis.to_dict(), verification, registry,
        [verification], [hypothesis.to_dict()], origins,
        manifest, manifest["problems"][0], None, None,
        bundle["bundle_digest"], record.terminal_evidence,
    )
    record = replace(
        record,
        corpus_digest=manifest["corpus_digest"],
        panel_set_digest=manifest["problems"][0]["panel_set_digest"],
        replay_spec_digest=spec.spec_digest,
    )
    checkpoint = R._checkpoint_payload(
        args, [record], manifest, 1, None, bundle)
    checkpoint["artifact_state"] = "PROMOTED"
    promoted = [{
        "opaque_id": "problem_00",
        "hypothesis": hypothesis.to_dict(),
        "verification": verification.to_dict(),
        "selection": record.selection,
        "runspec_digest": spec.spec_digest,
        "rounds_used": record.rounds_used,
    }]
    _write_json(artifact / "corpus_manifest.json", manifest)
    _write_json(artifact / "corpus_panels.json", bundle)
    receipts = SA._cold_replay_specs(
        str(artifact), promoted, checkpoint,
        corpus_manifest=manifest, corpus_bundle=bundle)
    assert receipts and receipts[0]["status"] == "PASS"
    results = {"problem_00": R._result_payload(problem, record)}
    _write_json(artifact / "artifact_binding.json",
                SA._artifact_binding(checkpoint, manifest, bundle, None))
    _write_json(artifact / "results.json", results)
    _write_json(artifact / "promoted_cones.json", promoted)
    _write_json(artifact / "checkpoint.json", checkpoint)
    report = _track_report_from_semantic(
        preregistration, arm, checkpoint)
    report_path = artifact / "track_reports" / (
        arm["arm_id"].replace(":", "__") + ".json")
    _write_json(report_path, report)

    spec_path = artifact / "replay_specs" / "problem_00.json"
    forged_spec = json.loads(spec_path.read_text())
    forged_spec["provenance"]["experiment"][
        "phase_execution_binding"] = {}
    forged_spec["spec_digest"] = semantic_replay.canonical_json_digest({
        key: value for key, value in forged_spec.items()
        if key != "spec_digest"})
    _write_json(spec_path, forged_spec)
    forged_checkpoint = copy.deepcopy(checkpoint)
    forged_checkpoint["records"][0]["replay_spec_digest"] = \
        forged_spec["spec_digest"]
    forged_promoted = copy.deepcopy(promoted)
    forged_promoted[0]["runspec_digest"] = forged_spec["spec_digest"]
    with pytest.raises(SA.ReplayCertificationError, match="Phase execution tranche"):
        SA._cold_replay_specs(
            str(artifact), forged_promoted, forged_checkpoint,
            corpus_manifest=manifest, corpus_bundle=bundle)
    semantic_replay.save_runspec(str(spec_path), spec, allowed_root=tmp_path)

    before = {
        str(path.relative_to(artifact)): path.read_bytes()
        for path in artifact.rglob("*") if path.is_file()}
    discovered = C.discover_track_reports(
        [str(report_path.parent)], preregistration)
    after = {
        str(path.relative_to(artifact)): path.read_bytes()
        for path in artifact.rglob("*") if path.is_file()}
    assert before == after
    assert discovered[0].certification["replay_receipts_digest"] == \
        semantic_replay.canonical_json_digest(receipts)

    stored_path = artifact / "replay_receipts" / "problem_00.json"
    tampered = json.loads(stored_path.read_text())
    tampered["status"] = "FAIL"
    _write_json(stored_path, tampered)
    with pytest.raises(C.CampaignCollectionError, match="stored artifact receipts"):
        C.discover_track_reports(
            [str(report_path.parent)], preregistration)


def test_collection_is_input_order_independent_and_write_once(
        certified_campaign, tmp_path, monkeypatch):
    prereg_path = certified_campaign["prereg_path"]
    directories = certified_campaign["report_dirs"]
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = C.collect_campaign(
        str(prereg_path), [str(path) for path in directories], str(first_path))
    real_fsync = C.os.fsync
    fsynced_directory = False

    def recording_fsync(descriptor):
        nonlocal fsynced_directory
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            fsynced_directory = True
        return real_fsync(descriptor)

    monkeypatch.setattr(C.os, "fsync", recording_fsync)
    second = C.collect_campaign(
        str(prereg_path), [str(path) for path in reversed(directories)],
        str(second_path))
    assert fsynced_directory
    assert first == second
    original = first_path.read_bytes()
    original_mtime = first_path.stat().st_mtime_ns
    again = C.collect_campaign(
        str(prereg_path), [str(path) for path in directories], str(first_path))
    assert again == first
    assert first_path.read_bytes() == original
    assert first_path.stat().st_mtime_ns == original_mtime


def test_missing_and_duplicate_certified_arm_sets_fail_closed(
        certified_campaign, tmp_path):
    missing_path = next(iter(certified_campaign["report_paths"].values()))
    missing_path.unlink()
    with pytest.raises(C.CampaignCollectionError, match="incomplete.*missing"):
        _collect(certified_campaign, tmp_path / "missing.json")

    # Restore, then place a second complete copy under another correctly named
    # artifact root. Duplicate detection retains and reports both origins.
    arm = certified_campaign["preregistration"]["arms"][0]
    original_artifact = certified_campaign["artifacts"][arm["execution_tag"]]
    original_report = certified_campaign["reports"][arm["arm_id"]]
    original_path = certified_campaign["report_paths"][arm["arm_id"]]
    _write_json(original_path, original_report)
    duplicate_root = tmp_path / "duplicate-root"
    duplicate_artifact = duplicate_root / original_artifact.name
    shutil.copytree(original_artifact, duplicate_artifact)
    dirs = [str(path) for path in certified_campaign["report_dirs"]]
    dirs.append(str(duplicate_artifact / "track_reports"))
    with pytest.raises(C.CampaignCollectionError, match="duplicate track report"):
        C.collect_campaign(
            str(certified_campaign["prereg_path"]), dirs,
            str(tmp_path / "duplicate.json"))
