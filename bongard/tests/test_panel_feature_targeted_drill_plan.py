from __future__ import annotations

import copy

import pytest

from bongard.exposure import ExposureLedger
from bongard.object_bongard_batch import object_bongard_task_inventory_digest
from bongard.panel_feature_targeted_drill_plan import (
    PanelFeatureTargetedDrillPlan,
    PanelFeatureTargetedDrillPlanError,
    plan_panel_feature_targeted_drill,
    verify_panel_feature_targeted_drill_plan,
)


CORPUS = "sha256:" + "a" * 64
RELEASE = "sha256:" + "b" * 64
SPLIT = "sha256:" + "c" * 64
SEMANTIC = "hd_convex-has_four_straight_lines"


def _inventory() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                *(f"{SEMANTIC}_{index:04d}" for index in range(7)),
                "hd_has_curve-necked_0000",
                "bd_example-shape_0000",
                "ff_example_0000",
            )
        )
    )


def _predecessor() -> ExposureLedger:
    inventory = _inventory()
    return ExposureLedger.create(CORPUS).record(
        phase="prior-engineering-drill",
        actor="test",
        purpose="disclose-one-generator-semantic",
        task_ids=(f"{SEMANTIC}_0001",),
        source="test-fixture",
        observed_at="2026-08-09T00:00:00Z",
        known_task_ids=inventory,
    )


def _plan(*, seed: str = "targeted-seed", requested: int = 2):
    inventory = _inventory()
    train = tuple(item for item in inventory if item != f"{SEMANTIC}_0006")
    predecessor = _predecessor()
    return plan_panel_feature_targeted_drill(
        task_ids=inventory,
        train_task_ids=train,
        predecessor=predecessor,
        target_semantic_key=SEMANTIC,
        selection_seed=seed,
        requested_task_count=requested,
        release_descriptor_digest=RELEASE,
        split_source_digest=SPLIT,
        task_inventory_digest=object_bongard_task_inventory_digest(inventory),
    )


def test_targeted_plan_reuses_semantics_but_selects_exact_unused_images() -> None:
    plan = _plan()

    assert plan.semantic_reuse_witness_task_ids == (f"{SEMANTIC}_0001",)
    assert plan.exact_unused_candidate_count == 5
    assert len(plan.tasks) == 2
    assert all(item.task_id.startswith(SEMANTIC + "_") for item in plan.tasks)
    assert all(item.task_id != f"{SEMANTIC}_0001" for item in plan.tasks)
    assert all(item.task_id != f"{SEMANTIC}_0006" for item in plan.tasks)
    assert PanelFeatureTargetedDrillPlan.from_data(plan.to_data()) == plan

    inventory = _inventory()
    train = tuple(item for item in inventory if item != f"{SEMANTIC}_0006")
    assert (
        verify_panel_feature_targeted_drill_plan(
            plan,
            task_ids=inventory,
            train_task_ids=train,
            predecessor=_predecessor(),
            selection_seed="targeted-seed",
        )
        == plan
    )
    for task in plan.tasks:
        for support, query in (
            (task.side_0_support_panel_ids, task.side_0_query_panel_id),
            (task.side_1_support_panel_ids, task.side_1_query_panel_id),
        ):
            assert len(support) == 6
            assert query not in support
            assert len(set(support) | {query}) == 7


def test_plan_serialization_states_the_exact_nonbenchmark_claim() -> None:
    data = _plan(requested=1).to_data()

    assert data["selection_inputs_include_pixels"] is False
    assert data["selection_inputs_include_pixel_paths"] is False
    assert data["selection_inputs_include_action_programs"] is False
    assert data["panel_bytes_opened_during_selection"] is False
    assert data["selected_tasks_exact_image_unused"] is True
    assert data["selected_semantics_previously_disclosed"] is True
    assert data["semantics_fresh_claim_authorized"] is False
    assert data["official_test_authorized"] is False
    assert data["query_identities_sealed_before_support_pixels"] is True
    assert data["python_is_canonical_authority"] is True
    assert data["lean_required"] is False
    assert data["lean_removable"] is True


def test_seed_and_tamper_change_or_invalidate_the_frozen_plan() -> None:
    first = _plan(seed="one", requested=1)
    second = _plan(seed="two", requested=1)
    assert first.selection_seed_digest != second.selection_seed_digest
    assert first.record_digest != second.record_digest

    tampered = copy.deepcopy(first.to_data())
    tampered["selected_semantics_previously_disclosed"] = False
    with pytest.raises(PanelFeatureTargetedDrillPlanError):
        PanelFeatureTargetedDrillPlan.from_data(tampered)

    tampered = copy.deepcopy(first.to_data())
    tampered["tasks"][0]["side_0_query_panel_id"] = tampered["tasks"][0][
        "side_0_support_panel_ids"
    ][0]
    with pytest.raises((PanelFeatureTargetedDrillPlanError, ValueError)):
        PanelFeatureTargetedDrillPlan.from_data(tampered)


def test_missing_disclosure_and_bad_metadata_fail_closed() -> None:
    inventory = _inventory()
    train = tuple(item for item in inventory if item != f"{SEMANTIC}_0006")
    common = dict(
        task_ids=inventory,
        train_task_ids=train,
        target_semantic_key=SEMANTIC,
        selection_seed="seed",
        requested_task_count=1,
        release_descriptor_digest=RELEASE,
        split_source_digest=SPLIT,
        task_inventory_digest=object_bongard_task_inventory_digest(inventory),
    )
    with pytest.raises(PanelFeatureTargetedDrillPlanError, match="not previously"):
        plan_panel_feature_targeted_drill(
            predecessor=ExposureLedger.create(CORPUS),
            **common,
        )
    with pytest.raises(PanelFeatureTargetedDrillPlanError, match="omit"):
        plan_panel_feature_targeted_drill(
            predecessor=_predecessor(),
            **{**common, "target_semantic_key": f"{SEMANTIC}_0000"},
        )
    with pytest.raises(PanelFeatureTargetedDrillPlanError, match="sorted"):
        plan_panel_feature_targeted_drill(
            predecessor=_predecessor(),
            **{**common, "task_ids": tuple(reversed(inventory))},
        )
    with pytest.raises(PanelFeatureTargetedDrillPlanError, match="too few"):
        _plan(requested=6)
