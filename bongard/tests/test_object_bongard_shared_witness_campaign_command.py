"""Focused offline tests for the sealed shared-witness TRAIN campaign."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from threading import Lock

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardBatchPlan
from bongard.object_bongard_shared_witness import (
    ObjectBongardSharedWitnessContrast,
    ObjectBongardSharedWitnessRubricSpec,
)
from bongard.object_bongard_shared_witness_calibration_command import (
    RESULT_SCHEMA as CALIBRATION_RESULT_SCHEMA,
    VerifiedObjectBongardSharedWitnessCalibration,
)
import bongard.object_bongard_shared_witness_campaign_command as campaign_command
from bongard.object_bongard_shared_witness_campaign_command import (
    MAX_PHYSICAL_CALLS,
    QUERY_DENOMINATOR,
    run_object_bongard_shared_witness_campaign_command,
    verify_object_bongard_shared_witness_campaign_command_directory,
)
from bongard.object_bongard_shared_witness_observer import (
    _endpoint_mapping,
    _neutral_endpoint_cues,
    object_bongard_shared_witness_panel_prompt,
)
from bongard.tests.test_prototype_scene_observer import (
    LAUNCHER_DIGEST,
    MODEL_CATALOG,
    NO_TOOLS_ATTESTATION,
    _png,
    _receipt,
)
from bongard.transport import (
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexStructuredResult,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "bongard/data/object_bongard_rubric_train_20260808.plan.json"


def _seal(body: dict[str, object], field: str) -> dict[str, object]:
    result = json.loads(canonical_json(body))
    result[field] = "sha256:" + canonical_digest(result)
    return result


def _install_calibration_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    accepted: bool,
) -> Path:
    calibration_root = tmp_path / (
        "accepted_calibration" if accepted else "rejected_calibration"
    )
    calibration_root.mkdir()
    selected_rank = 0 if accepted else None
    selected_spec = "9" * 64 if accepted else None
    selected_candidate = "a" * 64 if accepted else None
    result = _seal(
        {
            "schema": CALIBRATION_RESULT_SCHEMA,
            "historical_source_digest": "4" * 64,
            "authorization_digest": "sha256:" + "1" * 64,
            "execution_precommit_digest": "sha256:" + "2" * 64,
            "batch_digest": "sha256:" + "3" * 64,
            "freeze_digest": "sha256:" + "5" * 64,
            "assessment_digest": "sha256:" + "6" * 64,
            "cold_replay_digest": "sha256:" + "7" * 64,
            "nomination_artifact_digest": "8" * 64,
            "accepted": accepted,
            "selected_candidate_rank": selected_rank,
            "selected_spec_digest": selected_spec,
            "selected_candidate_digest": selected_candidate,
            "fresh_call_count": 48,
            "reused_call_count": 0,
            "physical_call_denominator": 48,
            "campaign_gate_lineage_complete": True,
            "all_48_artifacts_frozen_and_reloaded_before_assessment": True,
            "model_calls_during_assessment_or_replay": 0,
            "query_pixels_used": False,
            "fresh_broad_cohort_pixels_used": False,
            "official_test_pixels_used": False,
        },
        "result_digest",
    )
    (calibration_root / "result.json").write_bytes(canonical_json(result) + b"\n")
    verified = VerifiedObjectBongardSharedWitnessCalibration(
        output_root=calibration_root.resolve(),
        nomination_authorization_digest="sha256:" + "b" * 64,
        nomination_execution_precommit_digest="sha256:" + "c" * 64,
        nomination_result_digest="sha256:" + "d" * 64,
        nomination_replay_digest="sha256:" + "e" * 64,
        nomination_artifact_digest="8" * 64,
        source_digest="4" * 64,
        authorization_digest="sha256:" + "1" * 64,
        execution_precommit_digest="sha256:" + "2" * 64,
        batch_digest="sha256:" + "3" * 64,
        freeze_digest="sha256:" + "5" * 64,
        assessment_digest="sha256:" + "6" * 64,
        replay_digest="sha256:" + "7" * 64,
        result_digest=result["result_digest"],
        accepted=accepted,
        selected_candidate_rank=selected_rank,
        selected_spec_digest=selected_spec,
        fresh_call_count=48,
        reused_call_count=0,
    )

    def fake_verify(output_root, **_kwargs):
        assert Path(output_root).resolve() == calibration_root.resolve()
        return verified

    monkeypatch.setattr(
        campaign_command,
        "verify_object_bongard_shared_witness_calibration",
        fake_verify,
    )
    return calibration_root


def _semantic_payload() -> dict[str, object]:
    return {
        "proposal_0": {
            "shared_anchor": "patterned loop network",
            "visual_axis": "junction organization",
            "group_0_endpoint": "shared hub",
            "group_1_endpoint": "distributed junction",
        },
        "proposal_1": {
            "shared_anchor": "decorated contour network",
            "visual_axis": "contour termination",
            "group_0_endpoint": "closed circuit",
            "group_1_endpoint": "free ended",
        },
    }


def _test_specs() -> tuple[
    ObjectBongardSharedWitnessRubricSpec,
    ObjectBongardSharedWitnessRubricSpec,
]:
    payload = _semantic_payload()
    return tuple(
        ObjectBongardSharedWitnessRubricSpec.from_contrast(
            "f" * 64,
            ObjectBongardSharedWitnessContrast.create(
                rank,
                shared_anchor=proposal["shared_anchor"],
                visual_axis=proposal["visual_axis"],
                group_0_endpoint=proposal["group_0_endpoint"],
                group_1_endpoint=proposal["group_1_endpoint"],
            ),
        )
        for rank, proposal in enumerate(
            (payload["proposal_0"], payload["proposal_1"])
        )
    )  # type: ignore[return-value]


def _entity_payload(
    spec: ObjectBongardSharedWitnessRubricSpec,
    schema: dict[str, object],
    *,
    target: str,
    foil: str,
) -> dict[str, object]:
    cues = _neutral_endpoint_cues(spec)
    target_id, foil_id = _endpoint_mapping(spec, cues)
    judgments = {target_id: target, foil_id: foil}
    scope = schema["properties"]["entities"]["items"]["properties"]["scope"][  # type: ignore[index]
        "enum"
    ][0]
    return {
        "entity_id": "e00",
        "scope": scope,
        "bbox_q16": {"x0": 1000, "y0": 2000, "x1": 12000, "y1": 15000},
        "locator": "leftmost visible individual figure",
        "anchor_support": "clear",
        "anchor_evidence": "one complete patterned figure is visible",
        "cue_support": [
            {
                "cue_id": cue.cue_id,
                "judgment": judgments[cue.cue_id],
                "evidence": "the visible endpoint organization is inspectable",
            }
            for cue in cues
        ],
    }


def test_exact_campaign_freezes_before_query_scores_24_and_cold_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_root = _install_calibration_fixture(
        tmp_path, monkeypatch, accepted=True
    )
    plan = ObjectBongardBatchPlan.from_data(json.loads(PLAN.read_text()))
    output = tmp_path / "campaign"
    lock = Lock()
    calls: list[tuple[str, str | None]] = []
    snapshots = {"cache": 0, "catalog": 0, "fingerprint": 0, "attestation": 0}
    task_by_panel = {
        panel_id: (index, task)
        for index, task in enumerate(plan.tasks)
        for panel_id in (
            *task.side_0_support_panel_ids,
            *task.side_1_support_panel_ids,
            task.side_0_query_panel_id,
            task.side_1_query_panel_id,
        )
    }
    assert len(task_by_panel) == 14 * len(plan.tasks)
    panel_bytes = {
        panel_id: _png(index + 1)
        for index, panel_id in enumerate(sorted(task_by_panel))
    }
    panel_by_digest = {
        hashlib.sha256(payload).hexdigest(): panel_id
        for panel_id, payload in panel_bytes.items()
    }
    assert len(panel_by_digest) == len(panel_bytes)
    side_0 = {
        panel_id
        for task in plan.tasks
        for panel_id in (*task.side_0_support_panel_ids, task.side_0_query_panel_id)
    }
    prompt_to_spec = {
        object_bongard_shared_witness_panel_prompt(spec): spec
        for spec in _test_specs()
    }

    def panel_reader(panel_id: str):
        index, task = task_by_panel[panel_id]
        if panel_id in (task.side_0_query_panel_id, task.side_1_query_panel_id):
            task_root = output / "tasks" / f"{index:02d}_{task.task_id}"
            assert (task_root / "freeze.json").is_file()
            assert (task_root / "freeze_commit.json").is_file()
        payload = panel_bytes[panel_id]
        return payload, {
            "schema": "synthetic-offline-panel-receipt/v1",
            "panel_id": panel_id,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def transport(prompt, paths, names, schema, **_kwargs):
        assert (output / "plan.json").is_file()
        assert (output / "authorization.json").is_file()
        assert (output / "execution_precommit.json").is_file()
        if len(names) == 12:
            payload = _semantic_payload()
            panel_id = None
            kind = "semantic"
        else:
            assert names == ("panel.png",)
            digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
            panel_id = panel_by_digest[digest]
            spec = prompt_to_spec[prompt]
            target, foil = (
                ("clear", "none") if panel_id in side_0 else ("none", "clear")
            )
            payload = {
                "inventory_status": "complete",
                "entities": [
                    _entity_payload(spec, schema, target=target, foil=foil)
                ],
            }
            kind = "panel"
        with lock:
            calls.append((kind, panel_id))
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    def cache_snapshotter():
        snapshots["cache"] += 1
        return CloudPolicyCacheSnapshot(None)

    def catalog_snapshotter():
        snapshots["catalog"] += 1
        return MODEL_CATALOG

    def fingerprinter(_executable, *, expected_launcher_digest):
        snapshots["fingerprint"] += 1
        assert expected_launcher_digest == LAUNCHER_DIGEST
        return {
            "version": PINNED_CODEX_CLI_VERSION,
            "launcher_digest": LAUNCHER_DIGEST,
        }

    def attester(**kwargs):
        snapshots["attestation"] += 1
        assert kwargs["expected_launcher_digest"] == LAUNCHER_DIGEST
        return NO_TOOLS_ATTESTATION

    result = run_object_bongard_shared_witness_campaign_command(
        output,
        calibration_root=calibration_root,
        parallel_workers=4,
        expected_launcher_sha256=LAUNCHER_DIGEST,
        cloud_policy_cache_snapshotter=cache_snapshotter,
        model_catalog_snapshotter=catalog_snapshotter,
        launcher_fingerprinter=fingerprinter,
        runtime_attester=attester,
        underlying_transport=transport,
        panel_reader=panel_reader,
        archive_identity={
            "schema": "synthetic-offline-archive/v1",
            "digest": "sha256:" + "a" * 64,
        },
    )

    assert snapshots == {"cache": 1, "catalog": 1, "fingerprint": 1, "attestation": 1}
    assert len(calls) == result.physical_model_calls == MAX_PHYSICAL_CALLS == 324
    assert sum(kind == "semantic" for kind, _ in calls) == 12
    assert sum(kind == "panel" for kind, _ in calls) == 312
    assert result.correct_count == result.score_denominator == QUERY_DENOMINATOR == 24
    assert result.query_observer_calls == QUERY_DENOMINATOR
    assert result.campaign["status_counts"] == {
        "complete": 12,
        "language_gap": 0,
        "witness_gap": 0,
        "error_gap": 0,
        "task_exception": 0,
    }
    authorization = json.loads((output / "authorization.json").read_text())
    precommit = json.loads((output / "execution_precommit.json").read_text())
    replay = json.loads((output / "cold_replay.json").read_text())
    parent = authorization["accepted_calibration_parent"]
    for record in (authorization, precommit, result.campaign, replay):
        assert record["calibration_result_digest"] == parent[
            "calibration_result_digest"
        ]
        assert record["calibration_cold_replay_digest"] == parent[
            "calibration_cold_replay_digest"
        ]
        assert record["calibration_historical_source_digest"] == parent[
            "calibration_historical_source_digest"
        ]
    assert precommit["accepted_calibration_parent"] == parent
    assert result.campaign["accepted_calibration_parent_digest"] == parent[
        "parent_digest"
    ]
    assert replay["accepted_calibration_parent_digest"] == parent["parent_digest"]
    before = tuple(calls)
    replayed = verify_object_bongard_shared_witness_campaign_command_directory(
        output, calibration_root=calibration_root
    )
    assert replayed == result
    assert tuple(calls) == before


def test_missing_or_rejected_calibration_touches_no_campaign_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected_root = _install_calibration_fixture(
        tmp_path, monkeypatch, accepted=False
    )
    touches: list[str] = []

    def forbidden(*_args, **_kwargs):
        touches.append("called")
        raise AssertionError("calibration gate allowed a campaign resource")

    monkeypatch.setattr(campaign_command, "_load_exact_cohort", forbidden)
    monkeypatch.setattr(campaign_command, "_load_default_archive", forbidden)

    for label, calibration_root in (
        ("missing", tmp_path / "missing_calibration"),
        ("rejected", rejected_root),
    ):
        output = tmp_path / f"campaign_{label}"
        with pytest.raises(
            campaign_command.ObjectBongardSharedWitnessCampaignCommandError
        ):
            run_object_bongard_shared_witness_campaign_command(
                output,
                calibration_root=calibration_root,
                underlying_transport=forbidden,
                panel_reader=forbidden,
                archive_identity={"schema": "must-not-be-reached/v1"},
            )
        assert not output.exists()

    assert touches == []


def test_campaign_command_imports_no_old_pipeline_atlas_ranker_or_lean() -> None:
    source = ROOT / "bongard/object_bongard_shared_witness_campaign_command.py"
    tree = ast.parse(source.read_text("utf-8"))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    forbidden = ("panel_rubric_campaign", "object_bongard_rubric_campaign")
    assert not any(any(item in module for item in forbidden) for module in modules)
    assert not any(
        "atlas" in module or "ranker" in module or "lean" in module.casefold()
        for module in modules
    )
