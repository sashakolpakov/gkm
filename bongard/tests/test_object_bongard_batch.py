from __future__ import annotations

import copy

import pytest

from bongard.object_bongard_batch import (
    FAMILIES,
    ObjectBongardBatchError,
    ObjectBongardBatchPlan,
    object_bongard_task_inventory_digest,
    plan_object_bongard_batch,
    verify_object_bongard_batch_plan,
)


def _address(character: str) -> str:
    return "sha256:" + character * 64


def _task_ids() -> tuple[str, ...]:
    return tuple(
        sorted(f"{family}_task{index:02d}" for family in FAMILIES for index in range(6))
    )


def _plan(*, seed: str = "sealed-batch-seed", requested: int = 2):
    inventory = _task_ids()
    train = tuple(item for item in inventory if not item.endswith("05"))
    used = tuple(sorted(f"{family}_task00" for family in FAMILIES))
    return plan_object_bongard_batch(
        task_ids=inventory,
        train_task_ids=train,
        exact_used_task_ids=used,
        selection_seed=seed,
        requested_per_family=requested,
        release_descriptor_digest=_address("1"),
        split_source_digest=_address("2"),
        task_inventory_digest=object_bongard_task_inventory_digest(inventory),
        exposure_predecessor_digest=_address("4"),
        historical_exposure_digest=_address("5"),
    )


def test_plan_is_cross_family_exact_unused_and_query_sealed() -> None:
    plan = _plan()

    assert len(plan.tasks) == 6
    assert plan.candidate_counts == (("bd", 4), ("ff", 4), ("hd", 4))
    assert all(not item.task_id.endswith("00") for item in plan.tasks)
    assert {item.family for item in plan.tasks} == set(FAMILIES)
    assert ObjectBongardBatchPlan.from_data(plan.to_data()) == plan
    inventory = _task_ids()
    train = tuple(item for item in inventory if not item.endswith("05"))
    used = tuple(sorted(f"{family}_task00" for family in FAMILIES))
    assert verify_object_bongard_batch_plan(
        plan,
        task_ids=inventory,
        train_task_ids=train,
        exact_used_task_ids=used,
        selection_seed="sealed-batch-seed",
    ) == plan
    for task in plan.tasks:
        for support, query in (
            (task.side_0_support_panel_ids, task.side_0_query_panel_id),
            (task.side_1_support_panel_ids, task.side_1_query_panel_id),
        ):
            assert len(support) == 6
            assert query not in support
            assert len(set(support) | {query}) == 7


def test_seed_changes_the_frozen_plan_without_opening_more_inputs() -> None:
    first = _plan(seed="seed-one")
    second = _plan(seed="seed-two")

    assert first.selection_seed_digest != second.selection_seed_digest
    assert first.record_digest != second.record_digest
    assert first.release_descriptor_digest == second.release_descriptor_digest


def test_tampered_query_identity_and_policy_are_rejected() -> None:
    plan = _plan()
    tampered_query = copy.deepcopy(plan.to_data())
    tampered_query["tasks"][0]["side_0_query_panel_id"] = (  # type: ignore[index]
        tampered_query["tasks"][0]["side_0_support_panel_ids"][0]  # type: ignore[index]
    )
    with pytest.raises(ObjectBongardBatchError):
        ObjectBongardBatchPlan.from_data(tampered_query)

    tampered_policy = copy.deepcopy(plan.to_data())
    tampered_policy["official_test_authorized"] = True
    with pytest.raises(ObjectBongardBatchError):
        ObjectBongardBatchPlan.from_data(tampered_policy)


def test_insufficient_unused_family_and_unsorted_inventory_fail_closed() -> None:
    with pytest.raises(ObjectBongardBatchError, match="too few"):
        _plan(requested=5)

    inventory = _task_ids()
    with pytest.raises(ObjectBongardBatchError, match="unique and sorted"):
        plan_object_bongard_batch(
            task_ids=tuple(reversed(inventory)),
            train_task_ids=inventory,
            exact_used_task_ids=(),
            selection_seed="seed",
            requested_per_family=1,
            release_descriptor_digest=_address("1"),
            split_source_digest=_address("2"),
            task_inventory_digest=object_bongard_task_inventory_digest(inventory),
            exposure_predecessor_digest=_address("4"),
            historical_exposure_digest=_address("5"),
        )
