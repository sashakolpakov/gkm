"""Focused offline test for the sealed twelve-task panel-rubric campaign."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from threading import Lock

from bongard.object_bongard_batch import ObjectBongardBatchPlan
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


def test_exact_campaign_freezes_before_query_scores_24_and_cold_replays(
    tmp_path: Path,
) -> None:
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
    before = tuple(calls)
    replayed = verify_object_bongard_panel_rubric_campaign_command_directory(output)
    assert replayed == result
    assert tuple(calls) == before


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
