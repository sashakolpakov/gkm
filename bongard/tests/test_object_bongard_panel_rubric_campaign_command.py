"""Focused offline test for the sealed twelve-task panel-rubric campaign."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from threading import Lock

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_batch import ObjectBongardBatchPlan
import bongard.object_bongard_panel_rubric_campaign_command as campaign_command
from bongard.object_bongard_panel_rubric_calibration_command import (
    CALIBRATION_RESULT_SCHEMA,
    VerifiedObjectBongardPanelRubricCalibration,
)
from bongard.object_bongard_panel_rubric_campaign_command import (
    MAX_PHYSICAL_CALLS,
    QUERY_DENOMINATOR,
    run_object_bongard_panel_rubric_campaign_command,
    verify_object_bongard_panel_rubric_campaign_command_directory,
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
    selected_digest = "9" * 64 if accepted else None
    result_body = {
        "schema": CALIBRATION_RESULT_SCHEMA,
        "command_schema": "gkm.bongard-panel-rubric-calibration-command.v1",
        "authorization_digest": "sha256:" + "1" * 64,
        "execution_precommit_digest": "sha256:" + "2" * 64,
        "plan_digest": "3" * 64,
        "source_digest": "4" * 64,
        "batch_digest": "5" * 64,
        "freeze_digest": "sha256:" + "6" * 64,
        "assessment_digest": "7" * 64,
        "cold_replay_digest": "sha256:" + "8" * 64,
        "accepted": accepted,
        "selected_candidate_rank": selected_rank,
        "selected_candidate_digest": selected_digest,
        "fresh_call_count": 24,
        "reused_call_count": 0,
        "physical_call_denominator": 24,
        "all_24_artifacts_frozen_before_support_labels": True,
        "model_calls_during_assessment_or_replay": 0,
        "query_pixels_opened": False,
        "broad_cohort_pixels_opened": False,
        "official_test_pixels_opened": False,
        "predicate_authority_id": (
            "bongard.grounded-multimodal-predicate-authority/python-v1"
        ),
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }
    result = {
        **result_body,
        "record_digest": "sha256:" + canonical_digest(result_body),
    }
    (calibration_root / "result.json").write_bytes(
        canonical_json(result) + b"\n"
    )
    verified = VerifiedObjectBongardPanelRubricCalibration(
        output_root=calibration_root.resolve(),
        authorization_digest=result["authorization_digest"],
        execution_precommit_digest=result["execution_precommit_digest"],
        plan_digest=result["plan_digest"],
        batch_digest=result["batch_digest"],
        freeze_digest=result["freeze_digest"],
        assessment_digest=result["assessment_digest"],
        replay_digest=result["cold_replay_digest"],
        result_digest=result["record_digest"],
        accepted=accepted,
        selected_candidate_rank=selected_rank,
        fresh_call_count=24,
        reused_call_count=0,
    )

    def fake_verify(output_root, **_kwargs):
        assert Path(output_root).resolve() == calibration_root.resolve()
        return verified

    monkeypatch.setattr(
        campaign_command,
        "verify_object_bongard_panel_rubric_calibration",
        fake_verify,
    )
    return calibration_root


def test_exact_campaign_freezes_before_query_scores_24_and_cold_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_root = _install_calibration_fixture(
        tmp_path, monkeypatch, accepted=True
    )
    plan = ObjectBongardBatchPlan.from_data(__import__("json").loads(PLAN.read_text()))
    output = tmp_path / "campaign"
    lock = Lock()
    panel_bytes: dict[str, bytes] = {}
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
    panel_seed = {
        panel_id: index + 1 for index, panel_id in enumerate(sorted(task_by_panel))
    }

    def panel_reader(panel_id: str):
        index, task = task_by_panel[panel_id]
        if panel_id in (task.side_0_query_panel_id, task.side_1_query_panel_id):
            task_root = output / "tasks" / f"{index:02d}_{task.task_id}"
            assert (task_root / "freeze.json").is_file()
            assert (task_root / "freeze_commit.json").is_file()
        with lock:
            payload = panel_bytes.setdefault(
                panel_id, _png(panel_seed[panel_id])
            )
        return payload, {
            "schema": "synthetic-offline-panel-receipt/v1",
            "panel_id": panel_id,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    side_0 = {
        panel_id
        for task in plan.tasks
        for panel_id in (*task.side_0_support_panel_ids, task.side_0_query_panel_id)
    }

    def transport(prompt, paths, names, schema, **_kwargs):
        assert (output / "plan.json").is_file()
        assert (output / "authorization.json").is_file()
        assert (output / "execution_precommit.json").is_file()
        if len(names) == 12:
            payload = {
                "proposal_0": {
                    "group_0_cue_text": "A bird-like object with oblique angular wings.",
                    "group_1_cue_text": "A rounded object with mostly smooth contours.",
                },
                "proposal_1": {
                    "group_0_cue_text": "One object has several acute oblique angles.",
                    "group_1_cue_text": "One object is dominated by curved boundaries.",
                },
            }
            panel_id = None
            kind = "semantic"
        else:
            assert names == ("panel.png",)
            digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
            panel_id = next(
                key for key, value in panel_bytes.items()
                if hashlib.sha256(value).hexdigest() == digest
            )
            level = 4 if panel_id in side_0 else 0
            payload = {"lower": level, "upper": level}
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

    result = run_object_bongard_panel_rubric_campaign_command(
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
        archive_identity={"schema": "synthetic-offline-archive/v1", "digest": "sha256:" + "a" * 64},
    )

    assert snapshots == {"cache": 1, "catalog": 1, "fingerprint": 1, "attestation": 1}
    assert len(calls) == result.physical_model_calls == MAX_PHYSICAL_CALLS == 324
    assert sum(kind == "semantic" for kind, _ in calls) == 12
    assert sum(kind == "panel" for kind, _ in calls) == 312
    assert result.correct_count == result.score_denominator == QUERY_DENOMINATOR == 24
    assert result.campaign["status_counts"] == {
        "complete": 12,
        "language_gap": 0,
        "task_exception": 0,
        "witness_gap": 0,
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
        assert record["calibration_source_digest"] == parent[
            "calibration_source_digest"
        ]
    assert precommit["accepted_calibration_parent"] == parent
    assert result.campaign["accepted_calibration_parent_digest"] == parent[
        "parent_digest"
    ]
    assert replay["accepted_calibration_parent_digest"] == parent[
        "parent_digest"
    ]
    before = tuple(calls)
    replayed = verify_object_bongard_panel_rubric_campaign_command_directory(
        output, calibration_root=calibration_root
    )
    assert replayed == result
    assert tuple(calls) == before


def test_missing_or_rejected_calibration_opens_no_campaign_pixels_or_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected_root = _install_calibration_fixture(
        tmp_path, monkeypatch, accepted=False
    )
    panel_calls: list[str] = []
    transport_calls: list[str] = []
    archive_constructions: list[str] = []

    def panel_reader(panel_id: str):
        panel_calls.append(panel_id)
        raise AssertionError("calibration gate allowed a panel read")

    def transport(*_args, **_kwargs):
        transport_calls.append("called")
        raise AssertionError("calibration gate allowed model transport")

    def archive_loader(*_args, **_kwargs):
        archive_constructions.append("called")
        raise AssertionError("calibration gate allowed archive construction")

    monkeypatch.setattr(campaign_command, "_load_default_archive", archive_loader)

    for label, calibration_root, injected_reader in (
        ("missing", tmp_path / "missing_calibration", False),
        ("rejected", rejected_root, True),
    ):
        output = tmp_path / f"campaign_{label}"
        with pytest.raises(
            campaign_command.ObjectBongardPanelRubricCampaignCommandError
        ):
            kwargs = {
                "calibration_root": calibration_root,
                "underlying_transport": transport,
            }
            if injected_reader:
                kwargs.update(
                    {
                        "panel_reader": panel_reader,
                        "archive_identity": {
                            "schema": "synthetic-offline-archive/v1",
                            "digest": "sha256:" + "a" * 64,
                        },
                    }
                )
            run_object_bongard_panel_rubric_campaign_command(output, **kwargs)
        assert not output.exists()

    assert panel_calls == []
    assert transport_calls == []
    assert archive_constructions == []


def test_campaign_command_has_no_atlas_ranker_or_lean_import() -> None:
    source = ROOT / "bongard/object_bongard_panel_rubric_campaign_command.py"
    tree = ast.parse(source.read_text("utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("atlas" in name or "ranker" in name or "lean" in name.lower() for name in imported)
