"""Exact metadata-only tests for the fixed-32 skeleton calibration plan."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest, canonical_json
import bongard.panel_action_count_skeleton_graph_calibration_prereg as prereg


BONGARD = Path(__file__).resolve().parents[1]
ROOT = BONGARD.parent
SOURCE = BONGARD / "panel_action_count_skeleton_graph_calibration_prereg.py"
PLAN = (
    BONGARD
    / "data/panel_action_count_skeleton_graph_calibration_preregistration_20260810_v1.json"
)
OBSERVER_SOURCE = BONGARD / "panel_action_count_skeleton_graph_dev_command.py"
DEVELOPMENT_ROOT = (
    ROOT
    / "downloads/ShapeBongard_V2_full/panel_action_count_skeleton_graph_dev_20260810_v2"
)
DEVELOPMENT_PRECOMMIT = DEVELOPMENT_ROOT / "precommit.json"
DEVELOPMENT_RESULT = DEVELOPMENT_ROOT / "result.json"
V3_PLAN = BONGARD / "data/panel_action_count_cnn_preregistration_20260810_v3.json"
V3_MANIFEST = (
    BONGARD / "data/panel_action_count_cnn_calibration_panels_20260810_v3.json"
)
FAMILY_PLAN = (
    BONGARD / "data/panel_convex_four_lines_same_family_calibration_20260810_v2.json"
)
EXPOSURE_LEDGER = (
    ROOT
    / "downloads/ShapeBongard_V2_full/panel_soft_exact_unused_train_20260809_ranked_v1"
    / "research-exposure-successors"
    / "6995ea9cfda2f384cb0ba1b1cdc3611c965227c60fdb281d1e2e56fffa357b56.exposure.json"
)
EXPECTED_RECORD_DIGEST = (
    "sha256:7ebecfaf1a745a1d07d5c0805ba0a36f48ebd8871662be3432f82fcf55a09724"
)
EXPECTED_SOURCE_SHA256 = (
    "sha256:9413f2f00a32fa38adcbab0d745a398881a20437f930f9d202ffff74e35b67a6"
)
EXPECTED_PLAN_FILE_SHA256 = (
    "sha256:0431ef93c44b2186a8f30d5f080d719ed88e48bd80aad997f8e9fd19929b0038"
)


def _build() -> dict[str, object]:
    return prereg.build_preregistration(
        repository_root=ROOT,
        authority_source_path=SOURCE,
        observer_source_path=OBSERVER_SOURCE,
        development_precommit_path=DEVELOPMENT_PRECOMMIT,
        development_result_path=DEVELOPMENT_RESULT,
        v3_plan_path=V3_PLAN,
        v3_calibration_manifest_path=V3_MANIFEST,
        same_family_plan_path=FAMILY_PLAN,
        exposure_ledger_path=EXPOSURE_LEDGER,
    )


def _archived() -> dict[str, object]:
    raw = PLAN.read_bytes()
    value = json.loads(raw)
    assert isinstance(value, dict)
    assert raw == canonical_json(value) + b"\n"
    body = dict(value)
    found = body.pop("record_digest")
    assert found == "sha256:" + canonical_digest(body)
    assert found == EXPECTED_RECORD_DIGEST
    assert prereg._address(raw) == EXPECTED_PLAN_FILE_SHA256
    assert value["preregistration_authority"]["source_sha256"] == (
        EXPECTED_SOURCE_SHA256
    )
    return value


def test_exact_plan_rebuild_reads_only_disclosed_metadata(monkeypatch) -> None:
    archived = _archived()
    allowed = {
        SOURCE.resolve(),
        OBSERVER_SOURCE.resolve(),
        DEVELOPMENT_PRECOMMIT.resolve(),
        DEVELOPMENT_RESULT.resolve(),
        V3_PLAN.resolve(),
        V3_MANIFEST.resolve(),
        FAMILY_PLAN.resolve(),
        EXPOSURE_LEDGER.resolve(),
    }
    original = Path.read_bytes
    opened: list[Path] = []

    def guarded_read(path: Path) -> bytes:
        resolved = path.resolve()
        opened.append(resolved)
        if resolved not in allowed:
            raise AssertionError(f"preregistration opened an undeclared input: {resolved}")
        lowered = resolved.name.lower()
        if resolved.suffix.lower() == ".png":
            raise AssertionError("preregistration opened a panel pixel")
        if any(
            token in lowered
            for token in (
                "action_program",
                "catalog_label",
                "delayed_label",
                "labels_sealed",
                "model.pkl",
                "predictions.json",
                "features.json",
            )
        ):
            raise AssertionError("preregistration opened labels, programs, or live artifacts")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    assert _build() == archived
    assert set(opened) == allowed
    assert len(opened) == len(allowed)


def test_source_ast_has_no_pixel_label_model_or_concurrent_passed_fit_import() -> None:
    tree = ast.parse(SOURCE.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {
        "__future__",
        "bongard.canonical",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "re",
        "typing",
    }
    rendered = ast.dump(tree)
    for forbidden in ("PIL", "pickle", "sklearn", "torch", "action_programs"):
        assert forbidden not in imported
    assert "Image" not in imported
    assert "fit_authoritative_estimators" not in rendered


def test_fixed32_release_classes_and_full_54_pair_projection_are_exact() -> None:
    plan = _archived()
    observer = plan["observer_release_binding"]
    assert observer["commit"] == prereg.OBSERVER_COMMIT
    assert observer["source_sha256"] == prereg.OBSERVER_SOURCE_SHA256
    assert observer["config_digest"] == prereg.OBSERVER_CONFIG_DIGEST
    assert observer["fixed_n_estimators"] == 32
    assert observer["fixed_n_jobs"] == 1
    assert observer["required_heads"] == ["direct_pair", "catalog_three_class"]

    contract = plan["score_and_projection_contract"]
    observed = [tuple(pair) for pair in contract["observed_direct_pair_class_order"]]
    valid = [tuple(pair) for pair in contract["valid_pair_class_order_for_conformal_projection"]]
    assert [10 * straight + arc for straight, arc in observed] == list(
        prereg.OBSERVED_PAIR_CODES
    )
    assert [10 * straight + arc for straight, arc in valid] == list(
        prereg.VALID_PAIR_CODES
    )
    assert len(observed) == len(set(observed)) == 33
    assert len(valid) == len(set(valid)) == 54
    assert set(observed) < set(valid)
    assert (0, 0) not in valid
    assert all(1 <= straight + arc <= 9 for straight, arc in valid)
    assert contract["missing_valid_pair_probability"] == 0.0
    assert contract["catalog_class_order"] == [-1, 0, 1]
    assert contract["task_score"].startswith("maximum_over_all_14_task_panels")
    projection = contract["typed_projection"]
    assert projection["empty_pair_class_set_disposition"] == "error"
    assert projection["pair_0_0_disposition"] == "error"
    assert projection["catalog_empty_class_set_disposition"] == "error"
    assert projection["catalog_set_containing_minus_one_disposition"] == (
        "whole-axis_gap"
    )
    assert projection["other_five_typed_axes_disposition"] == "gap"


def test_generic_and_same_family_task_max_protocols_are_frozen_separately() -> None:
    plan = _archived()
    generic = plan["generic_v3_calibration"]
    assert generic["alpha"] == 0.05
    assert generic["calibration_task_count"] == 100
    assert generic["calibration_panel_count"] == 1_400
    assert generic["order_statistic_one_indexed"] == math.ceil(101 * 0.95) == 96
    assert generic["q_rule"] == "sorted_whole_task_scores[95]"
    assert generic["identity_binding"]["rank_slice"] == [1100, 1200]
    assert generic["target_authority"] == "cannot_authorize_target_under_any_outcome"

    family = plan["same_family_calibration"]
    assert family["alpha"] == 0.10
    assert family["calibration_task_count"] == 16
    assert family["calibration_panel_count"] == 224
    assert family["order_statistic_one_indexed"] == math.ceil(17 * 0.9) == 16
    assert family["q_rule"] == "sorted_whole_task_scores[15]"
    gate = family["efficiency_gate"]
    assert gate["global_q_only"] is True
    assert gate["evaluated_after_global_q_freeze"] is True
    assert gate["formula_inventory_count"] == 1_366
    assert gate["formula_admitted_task_count_at_least"] == 14
    assert gate["raw_direct_pair_head_used"] is True
    assert gate[
        "straight_candidates_are_marginal_projection_of_full_54_pair_set"
    ] is True
    assert gate["failure_action"] == "global_target_gap_with_target_pixels_sealed"
    identity = family["identity_binding"]
    assert identity["task_ids"] == list(prereg.SAME_FAMILY_TASK_IDS)
    assert identity["target_sealed_task_ids"] == [
        "hd_convex-has_four_straight_lines_0000"
    ]
    assert identity["diagnostic_tainted_task_ids"] == [
        "hd_convex-has_four_straight_lines_0001"
    ]
    assert identity["official_validation_sealed_task_ids"] == [
        "hd_convex-has_four_straight_lines_0018",
        "hd_convex-has_four_straight_lines_0019",
    ]
    for campaign in (generic, family):
        barrier = campaign["prediction_before_label_barrier"]
        assert barrier == {
            "action_label_or_program_loader_constructed_before_reload": False,
            "directory_fsync_required": True,
            "file_fsync_required": True,
            "prediction_artifact_must_reload_byte_identically": True,
            "prediction_record_digest_must_reverify_after_reload": True,
            "prediction_rows_complete_before_label_open": True,
        }
        assert campaign["coverage_unit"] == "whole_14-panel_task_repetition"
        assert campaign["within_task_panels_claimed_exchangeable"] is False


def test_passed_fit_slot_is_unresolved_fail_closed_and_pure() -> None:
    plan = _archived()
    slot = plan["passed_fit_authority_slot"]
    assert slot == prereg.passed_fit_slot()
    assert slot["status"] == "unresolved_at_metadata_preregistration"
    assert slot["placeholder_values"] == {
        field: None for field in prereg.PASSED_FIT_ADDRESS_FIELDS
    }
    assert slot["expected_module"] == (
        "bongard.panel_action_count_skeleton_graph_passed_fit_protocol"
    )
    addresses = {
        field: "sha256:" + str(index) * 64
        for index, field in enumerate(prereg.PASSED_FIT_ADDRESS_FIELDS, start=1)
    }
    resolved = prereg.resolve_passed_fit_slot(
        slot,
        outcome_schema=prereg.PASSED_FIT_PROTOCOL_SCHEMA,
        addresses=addresses,
    )
    assert resolved["status"] == "resolved_passed_fit_execution_precommit"
    assert all(resolved[field] == addresses[field] for field in addresses)

    with pytest.raises(
        prereg.SkeletonGraphCalibrationPreregistrationError,
        match="only an exact passed-fit protocol",
    ):
        prereg.resolve_passed_fit_slot(
            slot,
            outcome_schema=prereg.PASSED_FIT_GAP_SCHEMA,
            addresses=addresses,
        )
    with pytest.raises(
        prereg.SkeletonGraphCalibrationPreregistrationError,
        match="address inventory differs",
    ):
        prereg.resolve_passed_fit_slot(
            slot,
            outcome_schema=prereg.PASSED_FIT_PROTOCOL_SCHEMA,
            addresses=dict(list(addresses.items())[:-1]),
        )
    forged = dict(slot)
    forged["status"] = "resolved_without_execution_precommit"
    with pytest.raises(
        prereg.SkeletonGraphCalibrationPreregistrationError,
        match="placeholder differs",
    ):
        prereg.resolve_passed_fit_slot(
            forged,
            outcome_schema=prereg.PASSED_FIT_PROTOCOL_SCHEMA,
            addresses=addresses,
        )


def test_population_grants_and_role_free_raw_predictions_cannot_self_expand_scope() -> None:
    plan = _archived()
    population = plan["population_scope_contract"]
    assert population["generic_grant_target_authority"] is False
    assert population["population_scope_self_detectable_from_pixels"] is False
    assert population["confidence_can_establish_population_membership"] is False
    assert population["novel_carrier_disposition_without_external_grant"] == "gap"
    assert "separate_exact_external_population-membership_grant" in population[
        "same_family_grant_target_authority"
    ]

    raw = plan["raw_prediction_contract"]
    assert raw["catalog_probability_order"] == [-1, 0, 1]
    assert len(raw["direct_pair_probability_order"]) == 33
    assert set(raw["forbidden_fields"]) == {
        "action_label",
        "formula",
        "ordinal",
        "panel_path",
        "role",
        "side",
        "task_id",
    }
    assert raw["role_binding_occurs_only_after_durable_prediction_reload"] is True
    assert plan["authorization"] == {
        "action_labels_or_programs_before_durable_prediction_reload": False,
        "calibration_pixels_authorized_by_this_metadata_record": False,
        "diagnostic_0001_pixels": False,
        "official_TEST_pixels": False,
        "official_validation_0018_0019_pixels": False,
        "target_0000_pixels": False,
    }
    assert set(plan["current_state"].values()) == {0, False}
