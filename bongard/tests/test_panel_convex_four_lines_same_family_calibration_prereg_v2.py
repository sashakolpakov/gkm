"""Metadata-only checks for the same-family whole-task calibration prereg."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bongard.canonical import canonical_digest
from bongard.corpus import SplitIndex
from bongard.panel_typed_axis_slate_v2 import (
    AXIS_DOMAINS,
    ALGORITHM_ID,
    CLOSED_ATOM_COUNT,
    CROSS_AXIS_PAIR_COUNT,
    MAX_FORMULA_COUNT,
    Axis,
    typed_axis_slate_algorithm_digest,
    typed_axis_slate_source_digest,
)
from bongard.release import load_official_release


SEMANTIC = "hd_convex-has_four_straight_lines"
BONGARD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BONGARD_ROOT.parent
PLAN_PATH = (
    BONGARD_ROOT
    / "data/panel_convex_four_lines_same_family_calibration_20260810_v2.json"
)


def _task_ids(first: int, last: int) -> list[str]:
    return [f"{SEMANTIC}_{index:04d}" for index in range(first, last + 1)]


def _panels(task_ids: list[str]) -> list[str]:
    return [
        f"hd/{task_id}/{side}/{ordinal}.png"
        for task_id in task_ids
        for side in (1, 0)
        for ordinal in range(7)
    ]


def _load_plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_bytes())


def test_v2_partition_is_exact_all_panel_calibration_with_no_query_role() -> None:
    plan = _load_plan()
    body = dict(plan)
    record_digest = body.pop("record_digest")
    assert record_digest == "sha256:" + canonical_digest(body)
    assert record_digest == (
        "sha256:77a8aba2868ab3369a40befca470ee686eb998543dcae27d4f4b1f68a7df0b5a"
    )

    partition = plan["family_partition"]
    calibration_tasks = _task_ids(2, 17)
    assert partition["calibration_task_ids"] == calibration_tasks
    assert partition["calibration_task_count"] == 16
    assert partition["calibration_panel_count"] == 224
    assert partition["calibration_sides"] == [1, 0]
    assert partition["panel_ordinals"] == list(range(7))
    assert partition["calibration_panel_roles"] == (
        "all_14_are_calibration;_there_is_no_query_role"
    )
    assert partition["diagnostic_tainted_task_ids"] == _task_ids(1, 1)
    assert partition["target_sealed_task_ids"] == _task_ids(0, 0)
    assert partition["official_validation_sealed_task_ids"] == _task_ids(18, 19)

    panels = _panels(calibration_tasks)
    assert len(panels) == len(set(panels)) == 224
    assert all(f"/{SEMANTIC}_0000/" not in panel for panel in panels)
    assert all(f"/{SEMANTIC}_0001/" not in panel for panel in panels)
    assert all(f"/{SEMANTIC}_0018/" not in panel for panel in panels)
    assert all(f"/{SEMANTIC}_0019/" not in panel for panel in panels)

    release = load_official_release()
    bindings = plan["dataset_bindings"]
    assert bindings["split_source_digest"] == release.split_sha256
    assert bindings["task_inventory_digest"] == release.task_ids_sha256
    assert bindings["corpus_manifest_digest"] == release.corpus_manifest_sha256
    split_path = (
        REPOSITORY_ROOT
        / "downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json"
    )
    if split_path.is_file():
        split = SplitIndex.load(split_path)
        assert "sha256:" + canonical_digest(split.to_manifest_dict()) == bindings[
            "split_manifest_digest"
        ]
        assert all(split.assignment(task).split == "train" for task in _task_ids(0, 17))
        assert all(split.assignment(task).split == "val" for task in _task_ids(18, 19))


def test_whole_task_conformal_score_and_scope_are_frozen_exactly() -> None:
    plan = _load_plan()
    calibration = plan["calibration_protocol"]
    assert calibration["alpha"] == 0.1
    assert calibration["calibration_task_count"] == 16
    assert calibration["head_order"] == [
        "straight_action_count",
        "catalog_convexity",
    ]
    assert calibration["order_statistic_one_indexed"] == 16
    assert calibration["q_value_rule"] == "maximum_of_the_16_whole-task_scores"
    assert calibration["task_score"] == (
        "maximum_over_all_14_task_panels_and_both_relevant_heads_of_"
        "1_minus_probability_of_the_true_class"
    )
    assert (
        calibration["coverage_fraction_numerator"],
        calibration["coverage_fraction_denominator"],
        calibration["coverage_fraction_value"],
    ) == (16, 17, 16 / 17)
    assert calibration["repetition_is_unit_of_exchangeability"] is True
    assert calibration["within_task_panels_claimed_exchangeable"] is False
    claim = calibration["coverage_claim"]
    assert "whole same-family task repetitions only" in claim
    assert "official-validation" in claim
    assert "official-TEST" in claim

    loto = plan["leave_one_task_out_diagnostic"]
    assert loto["q_minus_i_rule"] == "maximum_of_the_other_15_whole-task_scores"
    assert loto["heldout_recovered_rule"] == (
        "heldout_task_score_is_at_most_q_minus_i"
    )
    assert loto["report_recovered_count_and_fraction_over_16"] is True
    assert loto["diagnostic_only_no_gate_tuning_or_selection_authority"] is True


def test_catalog_adapter_cannot_claim_generic_geometric_convexity() -> None:
    plan = _load_plan()
    adapter = plan["adapter_contract"]
    assert adapter["catalog_model_class_order"] == [
        "catalog_unresolved",
        "nonconvex",
        "convex",
    ]
    assert adapter["catalog_model_to_typed_value"] == {
        "convex": "catalog_convex",
        "nonconvex": "catalog_nonconvex",
    }
    assert adapter["catalog_set_containing_unresolved_disposition"] == "gap"
    assert adapter["catalog_typed_axis"] == Axis.CATALOG_CONVEXITY.value
    assert adapter["catalog_typed_domain"] == list(
        AXIS_DOMAINS[Axis.CATALOG_CONVEXITY]
    )
    assert adapter["generic_geometric_turning_axis_present"] is False
    assert adapter["three_class_catalog_head_can_populate_geometric_turning"] is False
    assert "turning_convexity" not in json.dumps(adapter, sort_keys=True)

    core = plan["typed_axis_core"]
    assert core["algorithm_id"] == ALGORITHM_ID
    assert core["source_sha256"] == typed_axis_slate_source_digest()
    assert core["algorithm_digest"] == typed_axis_slate_algorithm_digest()
    assert (
        core["closed_atom_count"],
        core["cross_axis_pair_count"],
        core["maximum_formula_count"],
    ) == (CLOSED_ATOM_COUNT, CROSS_AXIS_PAIR_COUNT, MAX_FORMULA_COUNT) == (
        57,
        1309,
        1366,
    )
    source = (REPOSITORY_ROOT / core["source_path"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == core["source_sha256"]


def test_target_aligned_efficiency_gate_is_fixed_and_fail_closed() -> None:
    plan = _load_plan()
    gate = plan["efficiency_gate"]
    assert gate["global_q_only"] is True
    assert gate["evaluated_after_global_q_freeze"] is True
    assert gate["support_primary_side"] == 1
    assert gate["support_contrast_side"] == 0
    assert gate["support_ordinals"] == [0, 1, 2, 3, 5, 6]
    assert gate["support_excluded_ordinal"] == 4
    assert gate["support_excluded_ordinal_has_query_role"] is False
    assert gate["support_rows_per_task"] == 12
    assert gate["formula_inventory_count"] == 1366
    assert gate["fixed_formula"] == [
        ["straight_action_count", 4],
        ["catalog_convexity", "catalog_convex"],
    ]
    assert gate["formula_admitted_task_count_at_least"] == 14
    assert gate["formula_admitted_task_denominator"] == 16
    assert gate["formula_or_cell_error_count_must_equal"] == 0
    assert gate["straight_mean_class_set_size_at_most"] == 4.0
    assert gate["straight_count_4_singleton_panel_fraction_at_least"] == 0.25
    assert gate["catalog_typed_decisive_panel_fraction_at_least"] == 0.3
    assert gate["target_aligned_gate_is_additional_efficiency_screen_not_coverage_theorem"] is True
    assert gate["failure_action"] == "global_target_gap_with_target_pixels_sealed"
    assert gate[
        "no_tuning_reroll_checkpoint_replacement_adapter_replacement_or_threshold_change"
    ] is True
    assert gate["typed_inventory_nomination_candidate_selection_authority"] is False


def test_chronology_seals_components_q_target_and_official_validation() -> None:
    plan = _load_plan()
    assert plan["metadata_only_preregistration"] is True
    assert plan["new_family_panel_pixels_read_before_commit"] is False
    assert plan["new_family_action_labels_read_before_commit"] is False
    assert plan["new_family_action_programs_read_before_commit"] is False
    assert plan["predicate_authority"] == "python"
    assert plan["target_release_requires_separate_authorization"] is True
    assert plan["supersedes"] == {
        "commit": "fe9e92eb35f6e41af393e9a664092368129c57ab",
        "record_digest": "sha256:a8a94af6b430018ce9f80550a7bce910c8095c781d4fa51113abdb505a1e3cd7",
        "reason": (
            "replace_development-plus-heldout-query_drill_with_all-14-panel_"
            "whole-task_same-family_conformal_calibration"
        ),
    }
    authorization = plan["authorization"]
    assert authorization["action_programs_or_labels_before_prediction_fsync"] is False
    assert authorization["target_0000_pixels"] is False
    assert authorization["diagnostic_0001_pixels"] is False
    assert authorization["official_validation_0018_0019_pixels"] is False
    assert authorization["official_TEST_pixels"] is False

    chronology = plan["chronology"]
    freeze_index = next(
        index for index, step in enumerate(chronology) if step.startswith("freeze_the_final")
    )
    infer_index = next(
        index for index, step in enumerate(chronology) if step.startswith("infer_all_224")
    )
    q_index = next(
        index for index, step in enumerate(chronology) if step.startswith("freeze_q")
    )
    target_index = next(
        index
        for index, step in enumerate(chronology)
        if step.startswith("if_the_gate_passes")
    )
    assert freeze_index < infer_index < q_index < target_index

    precommit = plan["execution_precommit"]
    assert precommit["must_exist_before_first_calibration_family_pixel"] is True
    assert set(precommit["required_frozen_addresses"]) >= {
        "checkpoint_state_dict_sha256",
        "typed_adapter_source_sha256",
        "typed_adapter_algorithm_digest",
        "whole-task_score_source_sha256",
        "whole-task_score_algorithm_digest",
        "efficiency_gate_digest",
        "calibration_panel_identity_manifest_address",
    }

    before = plan["exposure_accounting"]["before"]
    after = plan["exposure_accounting"]["maximum_after_authorized_calibration"]
    assert (
        before["exposed_train"],
        before["exposed_validation"],
        before["exposed_official_test"],
    ) == (290, 24, 0)
    assert (
        after["exposed_train"],
        after["exposed_validation"],
        after["exposed_official_test"],
    ) == (306, 24, 0)
