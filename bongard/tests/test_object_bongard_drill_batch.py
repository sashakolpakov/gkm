from __future__ import annotations

from pathlib import Path

import pytest

from bongard.cohorts import classify_task
from bongard.exposure import (
    ExposureLedger,
    basic_morphology_cluster_id,
    semantic_policy_blocked_keys,
)
from bongard.historical_exposure import load_historical_exposure
from bongard.object_bongard_batch import object_bongard_task_inventory_digest
from bongard.object_bongard_drill_batch import (
    FREEFORM_POLICY,
    ObjectBongardDrillBatchError,
    ObjectBongardDrillBatchPlan,
    plan_object_bongard_drill_batch,
    verify_object_bongard_drill_batch_plan,
)


BD_SAFE = (
    "bd_asymm_trap_bridge_0000",
    "bd_asymm_unbala_goldfish-regular_x_0000",
    "bd_inverse_trapez_parallel_0000",
    "bd_symmetric_clamp-irregular_arc_cup_0000",
    "bd_thin_rec_down_right_triangle_0000",
    "bd_three_mismatch_sectors2-mismatch_triangle_rec3_0000",
    "bd_trapez_parallelogram_0000",
)
HD_SAFE = (
    "hd_exist_quadrangle-symmetric_transposed_0016",
    "hd_exist_regular-exist_triangle_0014",
    "hd_exist_regular-exist_triangle_0015",
    "hd_has_five_straight_lines-thin_shape_0013",
    "hd_has_obtuse_angle-has_line_crossing_0011",
    "hd_has_six_straight_lines-has_acute_angle_0002",
    "hd_unbalanced_two-exist_sector_0012",
)


def _address(character: str) -> str:
    return "sha256:" + character * 64


def _inputs() -> tuple[tuple[str, ...], object, ExposureLedger, tuple[str, ...]]:
    historical = load_historical_exposure()
    dev_pair = historical.abstract_partition.dev[0]
    sealed_pair = historical.abstract_partition.sealed[0]
    excluded = (
        "ff_nact3_3_0041",
        "bd_hat_sector7-jar_triangle3_0000",
        "bd_jar_square3_0000",
        "bd_open_triangle6_0000",
        f"hd_{dev_pair[0]}-{dev_pair[1]}_0000",
        f"hd_{sealed_pair[0]}-{sealed_pair[1]}_0000",
    )
    inventory = tuple(sorted(set((*BD_SAFE, *HD_SAFE, *excluded))))
    predecessor = ExposureLedger.create(_address("a"))
    return inventory, historical, predecessor, excluded


def _plan(
    *,
    predecessor: ExposureLedger | None = None,
):
    inventory, historical, empty, _excluded = _inputs()
    ledger = predecessor or empty
    return plan_object_bongard_drill_batch(
        task_ids=inventory,
        train_task_ids=inventory,
        predecessor=ledger,
        selection_seed="already-frozen-public-drill-seed",
        requested_per_family=6,
        release_descriptor_digest=_address("1"),
        split_source_digest=_address("2"),
        task_inventory_digest=object_bongard_task_inventory_digest(inventory),
        historical=historical,
    )


def _tokens(task_id: str) -> set[str]:
    record = classify_task(task_id)
    if record.family == "bd":
        return {
            token
            for concept in record.parsed.concepts
            for token in (
                "basic_family:" + concept,
                "basic_morphology:" + basic_morphology_cluster_id(concept),
            )
        }
    return {
        *("abstract_attribute:" + item for item in record.parsed.concepts),
        "abstract_pair:" + "\0".join(record.parsed.concepts),
    }


def test_plan_is_exact_unused_strict_drill_disjoint_and_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # These are the only corpus operations that can return panel bytes.  The
    # planner has neither a corpus/archive nor a path argument and must not
    # reach either operation.
    import bongard.corpus as corpus_module
    from bongard.official_panel_archive import OfficialPanelArchive

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("metadata-only planning attempted to open a panel")

    monkeypatch.setattr(corpus_module, "_file_address", forbidden)
    monkeypatch.setattr(OfficialPanelArchive, "read_panel", forbidden)

    inventory, historical, predecessor, excluded = _inputs()
    plan = _plan(predecessor=predecessor)
    assert ObjectBongardDrillBatchPlan.from_data(plan.to_data()) == plan
    assert verify_object_bongard_drill_batch_plan(
        plan,
        task_ids=inventory,
        train_task_ids=inventory,
        predecessor=predecessor,
        selection_seed="already-frozen-public-drill-seed",
        historical=historical,
    ) == plan

    assert len(plan.tasks) == 12
    assert {family: sum(task.family == family for task in plan.tasks) for family in ("bd", "hd")} == {
        "bd": 6,
        "hd": 6,
    }
    assert not any(task.family == "ff" for task in plan.tasks)
    assert not set(excluded) & {task.task_id for task in plan.tasks}
    assert plan.to_data()["freeform_policy"] == FREEFORM_POLICY

    blocked = {
        key.concepts[0]
        for key in semantic_policy_blocked_keys(historical)
        if key.kind == "basic_morphology_cluster"
    }
    used_tokens: set[str] = set()
    for task in plan.tasks:
        record = classify_task(task.task_id, historical, split="train")
        assert record.historically_clean
        assert record.semantic_cohort == "drill"
        assert task.task_id not in predecessor.exposed_task_ids
        if task.family == "bd":
            assert not any(
                basic_morphology_cluster_id(concept) in blocked
                for concept in record.parsed.concepts
            )
        tokens = _tokens(task.task_id)
        assert not tokens & used_tokens
        used_tokens.update(tokens)
        for support, query in (
            (task.side_0_support_panel_ids, task.side_0_query_panel_id),
            (task.side_1_support_panel_ids, task.side_1_query_panel_id),
        ):
            assert len(support) == 6
            assert query not in support
            assert len(set(support) | {query}) == 7


def test_exact_name_drill_is_not_enough_for_blocked_basic_morphology() -> None:
    historical = load_historical_exposure()
    blocked = classify_task("bd_open_triangle6_0000", historical, split="train")
    assert blocked.historically_clean
    assert blocked.semantic_cohort == "drill"
    assert basic_morphology_cluster_id(blocked.parsed.concepts[0]) == "open_triangle"
    plan = _plan()
    assert "bd_open_triangle6_0000" not in {
        task.task_id for task in plan.tasks
    }
    assert dict(plan.morphology_excluded_counts)["bd"] >= 1


def test_stale_predecessor_invalidates_a_frozen_plan() -> None:
    inventory, historical, predecessor, _excluded = _inputs()
    plan = _plan(predecessor=predecessor)
    stale_task = plan.tasks[0].task_id
    successor = predecessor.record(
        phase="test-exact-task-release",
        actor="test",
        purpose="make frozen drill plan stale",
        task_ids=(stale_task,),
        observed_at="2026-08-08T00:00:00Z",
        known_task_ids=inventory,
        require_unseen=True,
    )
    with pytest.raises(ObjectBongardDrillBatchError):
        verify_object_bongard_drill_batch_plan(
            plan,
            task_ids=inventory,
            train_task_ids=inventory,
            predecessor=successor,
            selection_seed="already-frozen-public-drill-seed",
            historical=historical,
        )


def test_inventory_order_or_policy_tampering_fails_closed() -> None:
    inventory, historical, predecessor, _excluded = _inputs()
    with pytest.raises(ObjectBongardDrillBatchError, match="unique and sorted"):
        plan_object_bongard_drill_batch(
            task_ids=tuple(reversed(inventory)),
            train_task_ids=inventory,
            predecessor=predecessor,
            selection_seed="already-frozen-public-drill-seed",
            requested_per_family=6,
            release_descriptor_digest=_address("1"),
            split_source_digest=_address("2"),
            task_inventory_digest=object_bongard_task_inventory_digest(inventory),
            historical=historical,
        )

    raw = _plan().to_data()
    raw["freeform_policy"] = "allow-unpartitioned-freeform"
    with pytest.raises(ObjectBongardDrillBatchError):
        ObjectBongardDrillBatchPlan.from_data(raw)
