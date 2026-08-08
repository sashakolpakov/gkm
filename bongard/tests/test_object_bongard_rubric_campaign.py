"""Focused production-path tests for the broad object-rubric campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger
from bongard.object_bongard_batch import (
    FAMILIES,
    object_bongard_task_inventory_digest,
    plan_object_bongard_batch,
)
from bongard.object_bongard_release_gate import (
    ObjectBongardReleaseStore,
    create_object_bongard_execution_precommit,
    prepare_object_bongard_release,
)
from bongard.object_bongard_rubric_campaign import (
    ObjectBongardPhysicalCallBudget,
    ObjectBongardRubricCampaignRuntime,
    cold_replay_object_bongard_rubric_campaign_task,
    object_bongard_rubric_campaign_source_bindings,
    run_object_bongard_rubric_campaign_task,
    verify_object_bongard_rubric_campaign_metadata,
)
from bongard.object_bongard_rubric_ranker import (
    object_bongard_rubric_ranker_transport_source_digest,
)
from bongard.object_bongard_rubric_task_runner import (
    ObjectBongardRubricTaskRunStatus,
)
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.official_panel_archive import OfficialPanelArchive
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.release import OfficialReleaseDescriptor
from bongard.tests.no_tools_fixture import canonical_no_tools_runtime
from bongard.tests.test_object_bongard_rubric_ranker import _text_receipt
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    _png,
    _receipt,
)
from bongard.transport import CloudPolicyCacheSnapshot, CodexStructuredResult


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json(dict(value)) + b"\n")


def _fixture(tmp_path: Path):
    inventory = tuple(
        sorted(
            f"{family}_task{index:02d}"
            for family in FAMILIES
            for index in range(3)
        )
    )
    train = inventory
    used = tuple(sorted(f"{family}_task00" for family in FAMILIES))
    inventory_digest = object_bongard_task_inventory_digest(inventory)
    corpus_digest = _address({"synthetic": "rubric-campaign-corpus"})
    predecessor = ExposureLedger.create(corpus_digest)
    archive_path = tmp_path / "ShapeBongard_V2.zip"
    visual_level_by_digest: dict[str, int] = {}
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        seed = 0
        for task_id in inventory:
            family = task_id.split("_", 1)[0]
            for side in ("0", "1"):
                for index in range(7):
                    png = _png(seed)
                    seed += 1
                    visual_level_by_digest[hashlib.sha256(png).hexdigest()] = (
                        3 if side == "1" else 0
                    )
                    bundle.writestr(
                        f"ShapeBongard_V2/{family}/images/"
                        f"{task_id}/{side}/{index}.png",
                        png,
                    )
    archive_bytes = archive_path.read_bytes()
    split_path = tmp_path / "ShapeBongard_V2_split.json"
    split_raw = {"train": list(inventory), "val": [], "test": []}
    _write_json(split_path, split_raw)
    split_bytes = split_path.read_bytes()
    descriptor = OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-object-rubric-campaign-test",
        archive_filename=archive_path.name,
        archive_sha256="sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        archive_size_bytes=len(archive_bytes),
        split_filename=split_path.name,
        split_sha256="sha256:" + hashlib.sha256(split_bytes).hexdigest(),
        split_size_bytes=len(split_bytes),
        upstream_repository="https://github.com/NVlabs/Bongard-LOGO",
        upstream_commit="2" * 40,
        family_counts=tuple((family, 3) for family in FAMILIES),
        primary_split_counts=(
            ("test", 0),
            ("train", len(inventory)),
            ("val", 0),
        ),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256=inventory_digest,
        corpus_manifest_sha256=corpus_digest,
    )
    plan = plan_object_bongard_batch(
        task_ids=inventory,
        train_task_ids=train,
        exact_used_task_ids=used,
        selection_seed="rubric-campaign-vertical-slice-test",
        requested_per_family=1,
        release_descriptor_digest=descriptor.digest,
        split_source_digest=descriptor.split_sha256,
        task_inventory_digest=inventory_digest,
        exposure_predecessor_digest=predecessor.digest,
        historical_exposure_digest=_address({"historical": []}),
    )
    archive = OfficialPanelArchive.load(
        descriptor,
        archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )
    catalog, no_tools = canonical_no_tools_runtime(LAUNCHER_DIGEST)
    policy = CloudPolicyCacheSnapshot(None)
    visual_runtime = ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=3,
        verbose=False,
        executable="/private/synthetic-codex",
        cloud_policy_cache_snapshot=policy,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=no_tools,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    rank_runtime = ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=EFFORT,
        minutes=3,
        verbose=False,
        executable="/private/synthetic-codex",
        cloud_policy_cache_snapshot=policy,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=LAUNCHER_DIGEST,
        no_tools_attestation=no_tools,
        transport_source_digest=(
            object_bongard_rubric_ranker_transport_source_digest()
        ),
    )
    runtime = ObjectBongardRubricCampaignRuntime(
        visual=visual_runtime,
        rank=rank_runtime,
        max_workers=2,
        max_physical_model_calls=64,
    )
    precommit = create_object_bongard_execution_precommit(
        plan=plan,
        predecessor=predecessor,
        descriptor=descriptor,
        archive=archive,
        task_ids=inventory,
        train_task_ids=train,
        exact_used_task_ids=used,
        runtime_source_bindings=object_bongard_rubric_campaign_source_bindings(),
        configuration={
            "runtime_binding_digest": runtime.binding_digest,
            "max_workers": runtime.max_workers,
            "max_physical_model_calls": runtime.max_physical_model_calls,
            "headless": True,
        },
        exposure_observed_at="2026-08-08T12:00:00Z",
    )
    store = ObjectBongardReleaseStore((tmp_path / "store").absolute())
    prepared = prepare_object_bongard_release(
        store=store, plan=plan, precommit=precommit, predecessor=predecessor
    )
    return prepared, archive, runtime, visual_level_by_digest


class _VisualTransport:
    def __init__(self, levels: Mapping[str, int]) -> None:
        self.levels = levels
        self.calls = 0

    def __call__(
        self,
        prompt: str,
        paths: Sequence[str],
        names: Sequence[str],
        schema: Mapping[str, Any],
        **_kwargs: object,
    ) -> CodexStructuredResult:
        self.calls += 1
        if len(names) == 12:
            payload: dict[str, object] = {
                "profiles": [
                    {
                        "group_id": "group_0",
                        "rubric": (
                            "A winged angular form with several slanted spans."
                        ),
                        "feature_ids": ["bird_like_support_ppm"],
                    },
                    {
                        "group_id": "group_1",
                        "rubric": (
                            "A rounded compact form with a curved boundary."
                        ),
                        "feature_ids": ["rounded_leaf_support_ppm"],
                    },
                ]
            }
        else:
            assert names[0] == "scene.png" and len(names) == 2
            digest = hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest()
            level = self.levels[digest]
            slots = re.findall(
                r"^- ([A-Za-z0-9_.:-]+): row ", prompt, flags=re.MULTILINE
            )
            payload = {
                "scene": {"lower": level, "upper": level},
                "slots": [
                    {"slot_id": slot, "lower": level, "upper": level}
                    for slot in slots
                ],
            }
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )


class _RankTransport:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, prompt: str, schema: Mapping[str, Any], **_kwargs: object
    ) -> CodexStructuredResult:
        self.calls += 1
        aliases = re.findall(r"^- (r\d{3});", prompt, flags=re.MULTILINE)
        payload = {"ordered_aliases": aliases}
        return CodexStructuredResult(
            payload, _text_receipt(prompt, schema, payload)
        )


def test_one_task_crosses_real_release_gate_only_after_durable_freeze(
    tmp_path: Path,
) -> None:
    prepared, archive, runtime, levels = _fixture(tmp_path)
    visual = _VisualTransport(levels)
    rank = _RankTransport()
    budget = ObjectBongardPhysicalCallBudget(runtime.max_physical_model_calls)

    persisted = run_object_bongard_rubric_campaign_task(
        task=prepared.plan.tasks[0],
        prepared=prepared,
        archive=archive,
        runtime=runtime,
        journals_root=tmp_path / "journals",
        budget=budget,
        visual_transport=visual,
        rank_transport=rank,
    )

    execution = persisted.execution
    assert execution.task_run.status is ObjectBongardRubricTaskRunStatus.COMPLETE
    assert execution.task_run.freeze_commit_calls_made == 1
    assert execution.task_run.freeze_reload_calls_made == 1
    assert execution.task_run.query_source_calls_made == 1
    assert execution.fixed_score_denominator == 2
    assert execution.correct_count == 2
    assert execution.abstention_count == 0
    assert len(execution.support_observations) == 12
    assert len(execution.query_observations) == 2
    assert execution.task_freeze_store_receipt is not None
    assert execution.task_commit_store_receipt is not None
    assert execution.task_freeze_store_receipt.object_kind == "task-freeze"
    assert execution.task_commit_store_receipt.object_kind == (
        "task-decision-commit"
    )
    assert persisted.store_receipt.object_kind == "rubric-task-execution"
    assert budget.count == visual.calls + rank.calls
    assert rank.calls == 1
    calls_before_replay = (visual.calls, rank.calls, budget.count)
    replayed = cold_replay_object_bongard_rubric_campaign_task(
        execution,
        expected_execution_digest=execution.record_digest,
        execution_store_receipt=persisted.store_receipt,
        prepared=prepared,
        archive=archive,
        runtime=runtime,
        journals_root=tmp_path / "journals",
    )
    assert replayed == execution
    assert (visual.calls, rank.calls, budget.count) == calls_before_replay


def test_checked_in_preregistration_replays_exact_broad_unused_train_cohort() -> None:
    root = Path(__file__).parents[2]
    split = (
        root
        / "downloads/ShapeBongard_V2_full/ShapeBongard_V2/"
        "ShapeBongard_V2_split.json"
    )
    predecessor = (
        root
        / "downloads/ShapeBongard_V2_full/"
        "prototype_pair_python_campaign_20260807_object_v1/objects/"
        "exposure_successor/"
        "1bcde18e387539f13c4006b4a147e61c75feacb86bb031f10a6e8ba3412fe48d.json"
    )
    if not split.is_file() or not predecessor.is_file():
        return
    metadata = verify_object_bongard_rubric_campaign_metadata(
        preregistration_path=(
            root / "bongard/data/object_bongard_rubric_train_20260808.prereg.json"
        ),
        expected_preregistration_digest=(
            "sha256:b4e29960a9524f5785139a3ddf462d5ddec784d52eb0f2678cb1674820dd8107"
        ),
        plan_path=(
            root / "bongard/data/object_bongard_rubric_train_20260808.plan.json"
        ),
        descriptor_path=root / "bongard/data/shape_bongard_v2_release_v1.json",
        split_path=split,
        predecessor_path=predecessor,
    )
    assert len(metadata.task_ids) == 12_000
    assert len(metadata.train_task_ids) == 9_300
    assert len(metadata.exact_used_task_ids) == 278
    assert len(metadata.plan.tasks) == 12
    assert metadata.plan.record_digest == (
        "sha256:760edd40d91c67fd3c5e3b6f94119754f5368441b479f0940c2c7bd77c17b941"
    )
