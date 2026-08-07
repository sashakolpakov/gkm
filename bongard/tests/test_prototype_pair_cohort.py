from __future__ import annotations

import ast
import builtins
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest
from bongard.exposure import ExposureLedger, semantic_resolver_policy_digest
from bongard.grounded_multimodal_predicates import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.historical_exposure import load_historical_exposure
from bongard.prototype_pair_cohort import (
    BIRD_FAMILIES,
    CALIBRATION_CLUSTERS_PER_TAG,
    HYPOTHESIS_COUNT,
    OFFICIAL_BASIC_GENERATOR_SHA256,
    OFFICIAL_BASIC_SAMPLER_SHA256,
    OFFICIAL_UPSTREAM_COMMIT,
    OFFICIAL_UPSTREAM_REPOSITORY,
    OPAQUE_TAG_IDS,
    PROTOTYPE_POSITIVE_INDICES,
    PrototypePairCohortError,
    PrototypePairCohortPlan,
    TARGETED_ENGINEERING_TOLERANCE_PPM,
    ZERO_ERROR_FAMILY_UPPER_PPM,
    plan_prototype_pair_cohort,
    prototype_pair_seed_commitment,
    task_id_inventory_digest,
    verify_prototype_pair_cohort_plan,
)
from bongard.release import OfficialReleaseDescriptor


SEED = "externally committed prototype pair cohort seed"

_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
_CHECKED_IN_PREREG = (
    _DATA_ROOT / "prototype_pair_targeted_engineering_20260807.prereg.json"
)
_CHECKED_IN_PLAN = (
    _DATA_ROOT / "prototype_pair_targeted_engineering_20260807.plan.json"
)


def _split_bytes(task_ids: tuple[str, ...]) -> bytes:
    return json.dumps(
        {"test": [], "train": list(task_ids), "val": []},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _descriptor(
    task_ids: tuple[str, ...], split_bytes: bytes
) -> OfficialReleaseDescriptor:
    split_digest = "sha256:" + hashlib.sha256(split_bytes).hexdigest()
    return OfficialReleaseDescriptor(
        release_id="ShapeBongard_V2-metadata-test",
        archive_filename="ShapeBongard_V2.zip",
        archive_sha256="sha256:" + "a" * 64,
        archive_size_bytes=1,
        split_filename="ShapeBongard_V2_split.json",
        split_sha256=split_digest,
        split_size_bytes=len(split_bytes),
        upstream_repository=OFFICIAL_UPSTREAM_REPOSITORY,
        upstream_commit=OFFICIAL_UPSTREAM_COMMIT,
        family_counts=(("bd", len(task_ids)),),
        primary_split_counts=(
            ("test", 0),
            ("train", len(task_ids)),
            ("val", 0),
        ),
        regime_counts=(("BA", 0), ("CM", 0), ("FF", 0), ("NV", 0)),
        task_ids_sha256=task_id_inventory_digest(task_ids),
        corpus_manifest_sha256="sha256:" + "b" * 64,
    )


def _fixture(
    *,
    spoke_count: int = 14,
    candidate_count: int = 3,
    expose_task_ids: tuple[str, ...] = (),
):
    historical = load_historical_exposure()
    unused = [
        shape
        for shape in historical.unused_basic_shape_families
        if shape not in BIRD_FAMILIES
    ]
    targets = ["bird1", *unused[: 2 * candidate_count - 1]]
    partner_cursor = 2 * candidate_count - 1
    task_ids: set[str] = set()
    candidate_ids: list[str] = []
    for candidate_index in range(candidate_count):
        shape_a = targets[2 * candidate_index]
        shape_b = targets[2 * candidate_index + 1]
        candidate_id = f"bd_{shape_a}-{shape_b}_0000"
        candidate_ids.append(candidate_id)
        task_ids.add(candidate_id)
        task_ids.add(f"bd_{shape_a}_0000")
        task_ids.add(f"bd_{shape_b}_0000")
        for _ in range(spoke_count):
            partner = unused[partner_cursor]
            partner_cursor += 1
            task_ids.add(f"bd_{shape_a}-{partner}_0000")
        for _ in range(spoke_count):
            partner = unused[partner_cursor]
            partner_cursor += 1
            task_ids.add(f"bd_{shape_b}-{partner}_0000")
    inventory = tuple(sorted(task_ids))
    split = _split_bytes(inventory)
    release = _descriptor(inventory, split)
    exposure = ExposureLedger.create(release.corpus_manifest_sha256)
    for task_id in expose_task_ids:
        exposure = exposure.record(
            phase="metadata-test",
            actor="test",
            purpose="exact-used exclusion",
            task_ids=(task_id,),
            observed_at="2026-08-07T00:00:00Z",
        )
    return historical, release, split, inventory, exposure, tuple(candidate_ids)


def _kwargs(
    historical,
    release,
    split,
    inventory,
    exposure,
    *,
    seed: str = SEED,
):
    return {
        "release_descriptor": release,
        "split_bytes": split,
        "task_ids": inventory,
        "exposure_predecessor": exposure,
        "historical_seed": historical,
        "selection_seed": seed,
        "expected_seed_commitment": prototype_pair_seed_commitment(seed),
        "expected_release_descriptor_digest": release.digest,
        "expected_corpus_manifest_digest": release.corpus_manifest_sha256,
        "expected_split_source_digest": release.split_sha256,
        "expected_task_inventory_digest": release.task_ids_sha256,
        "expected_exposure_predecessor_digest": exposure.digest,
        "expected_historical_seed_digest": historical.seed_digest,
        "expected_resolver_policy_digest": semantic_resolver_policy_digest(
            historical
        ),
        "expected_basic_sampler_sha256": OFFICIAL_BASIC_SAMPLER_SHA256,
        "expected_basic_generator_sha256": OFFICIAL_BASIC_GENERATOR_SHA256,
    }


def test_plans_disjoint_cross_calibration_and_honest_statistical_claim() -> None:
    historical, release, split, inventory, exposure, candidate_ids = _fixture()
    plan = plan_prototype_pair_cohort(
        **_kwargs(historical, release, split, inventory, exposure)
    )

    assert tuple(item.task_id for item in plan.candidates) == tuple(
        sorted(candidate_ids)
    )
    assert plan.drill.task_id in candidate_ids
    assert len(plan.prototypes) == 2
    assert all(
        item.side == "positive"
        and item.panel_indices == PROTOTYPE_POSITIVE_INDICES
        and len(item.panel_ids) == 3
        for item in plan.prototypes
    )
    groups = {
        tag_id: [
            item
            for item in plan.calibration_clusters
            if item.group_tag_id == tag_id
        ]
        for tag_id in OPAQUE_TAG_IDS
    }
    assert {tag_id: len(rows) for tag_id, rows in groups.items()} == {
        tag_id: CALIBRATION_CLUSTERS_PER_TAG for tag_id in OPAQUE_TAG_IDS
    }
    target_a, target_b = plan.drill.ordered_shapes
    assert all(
        target_a in row.ordered_shapes
        and target_b not in row.ordered_shapes
        and dict(row.expected_tag_states)
        == {OPAQUE_TAG_IDS[0]: "present", OPAQUE_TAG_IDS[1]: "absent"}
        and row.score_tag_ids == OPAQUE_TAG_IDS
        and row.side == "positive"
        for row in groups[OPAQUE_TAG_IDS[0]]
    )
    assert all(
        target_b in row.ordered_shapes
        and target_a not in row.ordered_shapes
        and dict(row.expected_tag_states)
        == {OPAQUE_TAG_IDS[0]: "absent", OPAQUE_TAG_IDS[1]: "present"}
        and row.score_tag_ids == OPAQUE_TAG_IDS
        for row in groups[OPAQUE_TAG_IDS[1]]
    )
    assert len(plan.selected_task_ids) == 31
    assert plan.hypothesis_count == HYPOTHESIS_COUNT == 4
    assert plan.clusters_per_hypothesis == 14
    assert plan.zero_error_family_upper_ppm == ZERO_ERROR_FAMILY_UPPER_PPM == 268_752
    assert (
        plan.targeted_engineering_tolerance_ppm
        == TARGETED_ENGINEERING_TOLERANCE_PPM
        == 300_000
    )
    assert plan.stronger_250k_claim_authorized is False
    assert plan.zero_errors_required_for_tolerance is True
    assert len(plan.planner_source_sha256) == 64
    assert plan.planner_algorithm_digest.startswith("sha256:")
    assert all(
        row["cluster_count"] == 14
        for row in plan.to_data()["calibration"]["hypotheses"]
    )


def test_bird_candidate_is_reported_but_not_forced() -> None:
    historical, release, split, inventory, exposure, candidate_ids = _fixture()
    plan = plan_prototype_pair_cohort(
        **_kwargs(historical, release, split, inventory, exposure)
    )
    bird_candidate = next(task_id for task_id in candidate_ids if "bird1" in task_id)
    assert plan.bird_candidate_task_ids == (bird_candidate,)
    assert next(
        row for row in plan.candidates if row.task_id == bird_candidate
    ).bird_family_matches == ("bird1",)
    assert plan.to_data()["selection"][
        "bird_candidates_reported_without_selection_preference"
    ] is True


def test_seed_selection_is_deterministic_and_cold_replays_serialized_plan() -> None:
    historical, release, split, inventory, exposure, _candidate_ids = _fixture()
    kwargs = _kwargs(historical, release, split, inventory, exposure)
    first = plan_prototype_pair_cohort(**kwargs)
    second = plan_prototype_pair_cohort(**kwargs)
    assert first == second
    archived = first.to_data()
    assert PrototypePairCohortPlan.from_data(archived) == first
    assert verify_prototype_pair_cohort_plan(
        archived,
        **kwargs,
        expected_plan_digest=first.record_digest,
    ) == first

    changed = json.loads(json.dumps(archived))
    changed["runtime_authority"]["lean_affects_selection_or_decision"] = True
    with pytest.raises(PrototypePairCohortError):
        PrototypePairCohortPlan.from_data(changed)
    changed = json.loads(json.dumps(archived))
    changed["calibration"]["clusters"][0]["panel_index"] = (
        changed["calibration"]["clusters"][0]["panel_index"] + 1
    ) % 7
    with pytest.raises(PrototypePairCohortError, match="digest"):
        PrototypePairCohortPlan.from_data(changed)

    with pytest.raises(PrototypePairCohortError, match="authority"):
        replace(first, planner_algorithm_digest="sha256:" + "0" * 64)

    tampered_object = replace(first, namespace="changed-but-valid-namespace")
    with pytest.raises(PrototypePairCohortError, match="external commitment"):
        verify_prototype_pair_cohort_plan(
            tampered_object,
            **kwargs,
            expected_plan_digest=first.record_digest,
        )


def test_exact_used_tasks_are_excluded_before_candidate_selection() -> None:
    baseline = _fixture()
    exposed_candidate = baseline[-1][0]
    historical, release, split, inventory, exposure, candidate_ids = _fixture(
        expose_task_ids=(exposed_candidate,)
    )
    plan = plan_prototype_pair_cohort(
        **_kwargs(historical, release, split, inventory, exposure)
    )
    assert exposed_candidate in plan.excluded_exact_used_train_basic_task_ids
    assert exposed_candidate not in {item.task_id for item in plan.candidates}
    assert exposed_candidate not in plan.selected_task_ids
    assert len(plan.candidates) == 2
    assert set(item.task_id for item in plan.candidates) == set(candidate_ids) - {
        exposed_candidate
    }


def test_insufficient_other_task_occurrence_has_no_candidate() -> None:
    historical, release, split, inventory, exposure, _candidate_ids = _fixture(
        spoke_count=13,
        candidate_count=1,
    )
    with pytest.raises(PrototypePairCohortError, match="at least 14"):
        plan_prototype_pair_cohort(
            **_kwargs(historical, release, split, inventory, exposure)
        )


def test_authentication_tampering_and_exact_used_prototype_fail_closed() -> None:
    historical, release, split, inventory, exposure, candidate_ids = _fixture()
    kwargs = _kwargs(historical, release, split, inventory, exposure)
    with pytest.raises(PrototypePairCohortError, match="inventory"):
        plan_prototype_pair_cohort(**{**kwargs, "task_ids": tuple(reversed(inventory))})
    with pytest.raises(PrototypePairCohortError, match="split bytes"):
        plan_prototype_pair_cohort(**{**kwargs, "split_bytes": split + b" "})
    with pytest.raises(PrototypePairCohortError, match="source pin"):
        plan_prototype_pair_cohort(
            **{**kwargs, "expected_basic_sampler_sha256": "f" * 64}
        )

    first_shapes = next(
        row.ordered_shapes
        for row in plan_prototype_pair_cohort(**kwargs).candidates
        if row.task_id == candidate_ids[0]
    )
    prototype_id = f"bd_{first_shapes[0]}_0000"
    historical, release, split, inventory, exposure, _ = _fixture(
        expose_task_ids=(prototype_id,)
    )
    plan = plan_prototype_pair_cohort(
        **_kwargs(historical, release, split, inventory, exposure)
    )
    assert candidate_ids[0] not in {item.task_id for item in plan.candidates}
    assert prototype_id in plan.excluded_exact_used_train_basic_task_ids


def test_planner_has_metadata_only_boundary_and_grounded_python_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical, release, split, inventory, exposure, _candidate_ids = _fixture()
    kwargs = _kwargs(historical, release, split, inventory, exposure)

    def forbidden_open(*_args, **_kwargs):
        raise AssertionError("planner attempted filesystem I/O")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    plan = plan_prototype_pair_cohort(**kwargs)
    assert plan.panel_bytes_read is False
    assert plan.panel_paths_resolved is False
    assert plan.action_program_json_authorized is False
    assert plan.action_program_json_read is False
    assert plan.predicate_authority_id == PYTHON_PREDICATE_AUTHORITY_ID
    assert plan.python_is_canonical_authority is True
    assert plan.lean_required is False
    assert plan.lean_defines_artifact_identity is False
    assert plan.lean_affects_selection_or_decision is False
    assert plan.lean_required_for_replay is False
    assert plan.optional_secondary_checker_detachable is True
    assert plan.benchmark_claim_authorized is False
    assert plan.unseen_claim_authorized is False
    assert plan.validation_split_authorized is False
    assert plan.official_test_authorized is False
    assert "arbitrary prose" in plan.weak_label_authority

    parameters = set(inspect.signature(plan_prototype_pair_cohort).parameters)
    assert not any(
        forbidden in name
        for name in parameters
        if not name.startswith("expected_") or not name.endswith("_digest")
        for forbidden in ("panel", "path", "action_program", "corpus_manifest")
    )
    source = inspect.getsource(inspect.getmodule(plan_prototype_pair_cohort))
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "pathlib" not in imported
    assert not any("lean" in name.lower() for name in imported)
    assert "CorpusManifest" not in imported
    assert "TaskManifest" not in imported


def test_checked_in_targeted_engineering_prereg_and_plan_are_bound() -> None:
    prereg = json.loads(_CHECKED_IN_PREREG.read_bytes())
    archived_digest = prereg.pop("record_digest")
    assert canonical_digest(prereg) == archived_digest

    plan = PrototypePairCohortPlan.from_data(
        json.loads(_CHECKED_IN_PLAN.read_bytes())
    )
    assert prereg["selection"]["plan_digest"] == plan.record_digest
    assert prereg["selection"]["drill_task_id"] == plan.drill.task_id
    assert prereg["planner"]["source_sha256"] == plan.planner_source_sha256
    assert (
        prereg["planner"]["algorithm_digest"]
        == plan.planner_algorithm_digest
    )
    assert prereg["seed"]["commitment"] == plan.selection_seed_commitment
    assert (
        prereg["execution"]["panel_bytes_opened_before_preregistration"]
        is False
    )
    assert prereg["execution"]["official_test_authorized"] is False
    assert prereg["authority"]["python_is_canonical_authority"] is True
    assert prereg["authority"]["lean_required"] is False
    assert prereg["authority"]["lean_affects_selection_or_decision"] is False
