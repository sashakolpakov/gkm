"""Metadata-only calibration preregistration for the fixed-32 skeleton observer.

This module has no calibration runner.  It reads only already-frozen metadata
authorities while building the preregistration record.  In particular, it has
no panel, action-program, label, model-loading, fitting, or inference entry
point.  A future execution precommit must resolve the passed-fit address slot
before it may authorize even the first calibration pixel.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest, canonical_json


SCHEMA = "gkm.bongard-skeleton-graph-calibration-preregistration.v1"

OBSERVER_COMMIT = "dffd14a232bd213653c1d3b5eaffb08bb716cdd9"
OBSERVER_SOURCE_SHA256 = (
    "sha256:7399d4e0a3b05f14adff890b11a4674ece8904cc3e60cb9ea0b857fcd107a523"
)
OBSERVER_CONFIG_DIGEST = (
    "sha256:7dff25c405ddd05419a6c20b7c53559b9f6524735c09d271ca3dbd74c477b665"
)
DEVELOPMENT_PRECOMMIT_SCHEMA = "gkm.bongard-skeleton-graph-development-precommit.v2"
DEVELOPMENT_PRECOMMIT_RECORD_DIGEST = (
    "sha256:73e754c62c8f876af655dea0d6a30c4140f821f6173c275e4ed6e0b4b46a4bcc"
)
DEVELOPMENT_PRECOMMIT_FILE_SHA256 = (
    "sha256:d8d29c6bf4e216a91af9b26e0a4fd10e348c5536c51ff37f29402ab89fb4cf03"
)
DEVELOPMENT_RESULT_SCHEMA = "gkm.bongard-skeleton-graph-development-result.v2"
DEVELOPMENT_RESULT_RECORD_DIGEST = (
    "sha256:167a7f6b44affb0975145ffc3d7c2da1d038652e757408bc93aba7c1126991c9"
)
DEVELOPMENT_RESULT_FILE_SHA256 = (
    "sha256:cdc7fbe3b4c8221bc43c66cfae0d9cb94e5ea26e41069b91727db8e9753476df"
)
MODEL_FILE_SHA256 = (
    "sha256:25d1c21a117fe2bb2c68f9328351ef86f8b403019afafa182ae6b7d73aed2c52"
)
FEATURE_ARTIFACT_FILE_SHA256 = (
    "sha256:5f2ac055d6641aa9e3dcf532bcfd8cd37f6689363fb16ea56a7834faddb5af46"
)
PREDICTION_ARTIFACT_FILE_SHA256 = (
    "sha256:1551da8b9adb8f9cd8c0d94841901fdeabfc41e93684f9c97f1792f53547a76a"
)

V3_PLAN_SCHEMA = "gkm.bongard-action-count-catalog-cnn-preregistration.v3"
V3_PLAN_RECORD_DIGEST = (
    "sha256:bb4524a0958cd21f2d4d49bc6a9caa964ccb96c67fbf7c6192185f7b2f363dcb"
)
V3_PLAN_FILE_SHA256 = (
    "sha256:71c68771b356658843c3d848cdeea0ba7f2d96fffacd1816ef72934214b055d0"
)
V3_CALIBRATION_MANIFEST_SCHEMA = "gkm.bongard-action-count-cnn-calibration-panel-ids.v3"
V3_CALIBRATION_MANIFEST_RECORD_DIGEST = (
    "sha256:17088e6b72544a12829b255b4ada9f3b50e03423595c295185dbcfb02f9f515f"
)
V3_CALIBRATION_MANIFEST_FILE_SHA256 = (
    "sha256:d2f891e7fb5236dea5a2609d95c862bae103b3fe0f85724dea7b9b07a1caab9d"
)

SAME_FAMILY_PLAN_SCHEMA = (
    "gkm.bongard-convex-four-lines-same-family-calibration-preregistration.v2"
)
SAME_FAMILY_PLAN_RECORD_DIGEST = (
    "sha256:77a8aba2868ab3369a40befca470ee686eb998543dcae27d4f4b1f68a7df0b5a"
)
SAME_FAMILY_PLAN_FILE_SHA256 = (
    "sha256:5806422f2186a412ad4eba68de0deb4ab42133713ab7f3e3c88ef0cf5ea44c9c"
)

EXPOSURE_LEDGER_DIGEST = (
    "sha256:6995ea9cfda2f384cb0ba1b1cdc3611c965227c60fdb281d1e2e56fffa357b56"
)
EXPOSURE_LEDGER_FILE_SHA256 = (
    "sha256:8c5034e77f769a67b1bc16b41881e14887592e070e730d062049ea33e1467ff8"
)
CORPUS_MANIFEST_DIGEST = (
    "sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138"
)

PASSED_FIT_MODULE = "bongard.panel_action_count_skeleton_graph_passed_fit_protocol"
PASSED_FIT_SOURCE_PATH = "bongard/panel_action_count_skeleton_graph_passed_fit_protocol.py"
PASSED_FIT_PROTOCOL_SCHEMA = "gkm.bongard-skeleton-graph-passed-fit-protocol.v1"
PASSED_FIT_GAP_SCHEMA = "gkm.bongard-skeleton-graph-passed-fit-gap.v1"
PASSED_FIT_ADDRESS_FIELDS = (
    "passed_fit_authority_source_sha256",
    "passed_fit_algorithm_digest",
    "passed_fit_record_digest",
)

OBSERVED_PAIR_CODES = (
    1, 2, 4, 6, 8, 11, 12, 20, 21, 22, 23, 30, 31, 32, 33, 34,
    40, 41, 42, 43, 44, 50, 51, 52, 60, 61, 62, 63, 70, 71, 80, 81, 90,
)
VALID_PAIR_CODES = tuple(
    10 * straight + arc
    for straight in range(10)
    for arc in range(10)
    if 1 <= straight + arc <= 9
)
CATALOG_CLASS_ORDER = (-1, 0, 1)
SAME_FAMILY_TASK_IDS = tuple(
    f"hd_convex-has_four_straight_lines_{index:04d}" for index in range(2, 18)
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class SkeletonGraphCalibrationPreregistrationError(RuntimeError):
    """A metadata authority or fail-closed preregistration edge differs."""


def _address(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise SkeletonGraphCalibrationPreregistrationError(
            f"{label} is not a SHA-256 address"
        )
    return value


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SkeletonGraphCalibrationPreregistrationError(
            f"cannot read {label}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SkeletonGraphCalibrationPreregistrationError(f"{label} is not an object")
    return value, raw


def _canonical_record(
    path: Path,
    label: str,
    *,
    schema: str,
    record_digest: str,
    file_sha256: str,
    canonical_bytes_required: bool = True,
) -> dict[str, Any]:
    value, raw = _object(path, label)
    body = dict(value)
    found = body.pop("record_digest", None)
    if (
        (canonical_bytes_required and raw != canonical_json(value) + b"\n")
        or value.get("schema") != schema
        or found != "sha256:" + canonical_digest(body)
        or found != record_digest
        or _address(raw) != file_sha256
    ):
        raise SkeletonGraphCalibrationPreregistrationError(
            f"{label} is not the exact frozen authority"
        )
    return value


def _relative(repository_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError as exc:
        raise SkeletonGraphCalibrationPreregistrationError(
            "metadata authority leaves repository root"
        ) from exc


def passed_fit_slot() -> dict[str, Any]:
    """Return the exact unresolved slot committed by this metadata plan."""

    return {
        "accepted_outcome_schemas": {
            "gap": PASSED_FIT_GAP_SCHEMA,
            "passed": PASSED_FIT_PROTOCOL_SCHEMA,
        },
        "expected_module": PASSED_FIT_MODULE,
        "expected_source_path": PASSED_FIT_SOURCE_PATH,
        "placeholder_values": {field: None for field in PASSED_FIT_ADDRESS_FIELDS},
        "required_passed_address_fields": list(PASSED_FIT_ADDRESS_FIELDS),
        "resolution_authority": "separate_write-once_execution_precommit",
        "resolution_deadline": "before_first_calibration_pixel_authorization",
        "status": "unresolved_at_metadata_preregistration",
        "unresolved_or_gap_disposition": "no_pixel_authorization_and_global_gap",
    }


def resolve_passed_fit_slot(
    slot: Mapping[str, Any], *, outcome_schema: str, addresses: Mapping[str, str]
) -> dict[str, Any]:
    """Resolve the pure address slot; this imports and reads nothing.

    The caller remains responsible for independently loading and verifying the
    addressed passed-fit protocol in a later execution-precommit module.  This
    helper only makes missing, GAP, malformed, or renamed edges fail closed.
    """

    if dict(slot) != passed_fit_slot():
        raise SkeletonGraphCalibrationPreregistrationError(
            "passed-fit placeholder differs from the committed slot"
        )
    if outcome_schema != PASSED_FIT_PROTOCOL_SCHEMA:
        raise SkeletonGraphCalibrationPreregistrationError(
            "only an exact passed-fit protocol can authorize calibration pixels"
        )
    if set(addresses) != set(PASSED_FIT_ADDRESS_FIELDS):
        raise SkeletonGraphCalibrationPreregistrationError(
            "passed-fit address inventory differs"
        )
    resolved = {
        field: _require_address(addresses[field], field)
        for field in PASSED_FIT_ADDRESS_FIELDS
    }
    return {
        **resolved,
        "expected_module": PASSED_FIT_MODULE,
        "outcome_schema": outcome_schema,
        "status": "resolved_passed_fit_execution_precommit",
    }


def _validate_v3_identifiers(
    plan: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    state = plan.get("current_state")
    binding = plan.get("identifier_manifest_bindings", {}).get(
        "calibration_panel_ids"
    )
    plan_cohort = plan.get("cohorts", {}).get("calibration")
    manifest_cohort = manifest.get("cohorts", {}).get("calibration")
    if not all(isinstance(value, dict) for value in (state, binding, plan_cohort, manifest_cohort)):
        raise SkeletonGraphCalibrationPreregistrationError(
            "generic V3 metadata boundary is malformed"
        )
    assert isinstance(state, dict)
    assert isinstance(binding, dict)
    assert isinstance(plan_cohort, dict)
    assert isinstance(manifest_cohort, dict)
    if state != {
        "fresh_action_program_or_target_rows_read": 0,
        "fresh_calibration_panel_png_bytes_read": 0,
        "fresh_evaluation_panel_png_bytes_read": 0,
        "fresh_plaintext_targets_materialized": False,
        "model_training_started": False,
        "selected_png_bytes_read_by_v3_authority": 0,
    }:
        raise SkeletonGraphCalibrationPreregistrationError(
            "generic V3 identifiers are no longer fresh"
        )
    if binding != {
        "path": "bongard/data/panel_action_count_cnn_calibration_panels_20260810_v3.json",
        "record_digest": V3_CALIBRATION_MANIFEST_RECORD_DIGEST,
        "source_sha256": V3_CALIBRATION_MANIFEST_FILE_SHA256,
    }:
        raise SkeletonGraphCalibrationPreregistrationError(
            "generic V3 calibration-manifest binding differs"
        )
    task_ids = manifest_cohort.get("task_ids")
    panel_ids = manifest_cohort.get("panel_ids")
    if (
        not isinstance(task_ids, list)
        or not isinstance(panel_ids, list)
        or len(task_ids) != 100
        or len(set(task_ids)) != 100
        or len(panel_ids) != 1_400
        or len(set(panel_ids)) != 1_400
        or task_ids != plan_cohort.get("task_ids")
        or plan_cohort.get("rank_slice") != [1100, 1200]
        or plan_cohort.get("task_count") != 100
        or plan_cohort.get("panel_count") != 1_400
    ):
        raise SkeletonGraphCalibrationPreregistrationError(
            "generic V3 cohort identity or cardinality differs"
        )
    expected_panels = [
        f"hd/{task_id}/{side}/{ordinal}.png"
        for task_id in task_ids
        for side in (1, 0)
        for ordinal in range(7)
    ]
    if panel_ids != expected_panels:
        raise SkeletonGraphCalibrationPreregistrationError(
            "generic V3 panel order differs"
        )
    return task_ids, panel_ids


def _validate_family_partition(plan: Mapping[str, Any]) -> None:
    partition = plan.get("family_partition")
    authorization = plan.get("authorization")
    gate = plan.get("efficiency_gate")
    if (
        plan.get("metadata_only_preregistration") is not True
        or plan.get("new_family_panel_pixels_read_before_commit") is not False
        or plan.get("new_family_action_labels_read_before_commit") is not False
        or plan.get("new_family_action_programs_read_before_commit") is not False
        or not isinstance(partition, dict)
        or not isinstance(authorization, dict)
        or not isinstance(gate, dict)
    ):
        raise SkeletonGraphCalibrationPreregistrationError(
            "same-family predecessor is not metadata-only"
        )
    if (
        partition.get("calibration_task_ids") != list(SAME_FAMILY_TASK_IDS)
        or partition.get("calibration_task_count") != 16
        or partition.get("calibration_panel_count") != 224
        or partition.get("target_sealed_task_ids")
        != ["hd_convex-has_four_straight_lines_0000"]
        or partition.get("diagnostic_tainted_task_ids")
        != ["hd_convex-has_four_straight_lines_0001"]
        or partition.get("official_validation_sealed_task_ids")
        != [
            "hd_convex-has_four_straight_lines_0018",
            "hd_convex-has_four_straight_lines_0019",
        ]
        or authorization.get("target_0000_pixels") is not False
        or authorization.get("diagnostic_0001_pixels") is not False
        or authorization.get("official_validation_0018_0019_pixels") is not False
        or authorization.get("action_programs_or_labels_before_prediction_fsync")
        is not False
        or gate.get("global_q_only") is not True
        or gate.get("evaluated_after_global_q_freeze") is not True
        or gate.get("formula_inventory_count") != 1_366
        or gate.get("formula_admitted_task_count_at_least") != 14
        or gate.get("formula_admitted_task_denominator") != 16
        or gate.get("failure_action")
        != "global_target_gap_with_target_pixels_sealed"
        or gate.get(
            "no_tuning_reroll_checkpoint_replacement_adapter_replacement_or_threshold_change"
        )
        is not True
    ):
        raise SkeletonGraphCalibrationPreregistrationError(
            "same-family partition or seal differs"
        )


def _score_contract() -> dict[str, Any]:
    observed_pairs = [[code // 10, code % 10] for code in OBSERVED_PAIR_CODES]
    valid_pairs = [[code // 10, code % 10] for code in VALID_PAIR_CODES]
    return {
        "catalog_class_order": list(CATALOG_CLASS_ORDER),
        "catalog_true_class_score": "1_minus_raw_probability_of_true_catalog_class",
        "conformal_class_set_rule": "include_class_c_iff_1_minus_p_c_is_at_most_q",
        "direct_pair_encoding": "10_times_straight_action_count_plus_arc_action_count",
        "head_order": ["direct_pair", "catalog_three_class"],
        "invalid_true_pair_disposition": "error_not_a_calibration_score",
        "missing_valid_pair_probability": 0.0,
        "observed_direct_pair_class_count": len(observed_pairs),
        "observed_direct_pair_class_order": observed_pairs,
        "pair_true_class_score": (
            "1_minus_raw_probability_if_true_pair_is_one_of_33_observed_classes;_"
            "otherwise_1_minus_zero_for_a_valid_missing_class"
        ),
        "raw_direct_pair_probability_count": len(observed_pairs),
        "raw_catalog_probability_count": 3,
        "typed_projection": {
            "arc_candidate_set": "unique_arc_coordinates_of_pair_class_set",
            "catalog_empty_class_set_disposition": "error",
            "catalog_set_containing_minus_one_disposition": "whole-axis_gap",
            "catalog_zero_value": "catalog_nonconvex",
            "catalog_one_value": "catalog_convex",
            "empty_pair_class_set_disposition": "error",
            "other_five_typed_axes_disposition": "gap",
            "pair_0_0_disposition": "error",
            "primitive_value_for_arc_positive_straight_zero": "arc_only_a",
            "primitive_value_for_both_positive": "mixed_a_arcs",
            "primitive_value_for_straight_positive_arc_zero": "straight_only",
            "straight_candidate_set": "unique_straight_coordinates_of_pair_class_set",
        },
        "task_score": (
            "maximum_over_all_14_task_panels_and_both_heads_of_"
            "one_minus_probability_of_the_true_class"
        ),
        "valid_pair_class_count": len(valid_pairs),
        "valid_pair_class_order_for_conformal_projection": valid_pairs,
        "valid_pair_domain": "straight_and_arc_in_0_to_9_and_1_le_sum_le_9",
        "valid_unobserved_pair_classes_are_zero_filled_only_for_conformal_projection": True,
    }


def _campaign(
    *,
    alpha: float,
    task_count: int,
    panel_count: int,
    identity_binding: Mapping[str, Any],
    population_scope: str,
    target_authority: str,
) -> dict[str, Any]:
    order = math.ceil((task_count + 1) * (1.0 - alpha))
    return {
        "alpha": alpha,
        "calibration_panel_count": panel_count,
        "calibration_task_count": task_count,
        "coverage_unit": "whole_14-panel_task_repetition",
        "identity_binding": dict(identity_binding),
        "order_statistic_one_indexed": order,
        "population_scope": population_scope,
        "prediction_before_label_barrier": {
            "action_label_or_program_loader_constructed_before_reload": False,
            "directory_fsync_required": True,
            "file_fsync_required": True,
            "prediction_artifact_must_reload_byte_identically": True,
            "prediction_record_digest_must_reverify_after_reload": True,
            "prediction_rows_complete_before_label_open": True,
        },
        "q_rule": f"sorted_whole_task_scores[{order - 1}]",
        "target_authority": target_authority,
        "task_score_contract": "shared_exact_two-head_task-max-contract",
        "within_task_panels_claimed_exchangeable": False,
    }


def build_preregistration(
    *,
    repository_root: Path,
    authority_source_path: Path,
    observer_source_path: Path,
    development_precommit_path: Path,
    development_result_path: Path,
    v3_plan_path: Path,
    v3_calibration_manifest_path: Path,
    same_family_plan_path: Path,
    exposure_ledger_path: Path,
) -> dict[str, Any]:
    """Build the exact record while opening metadata authorities only."""

    authority_raw = authority_source_path.read_bytes()
    observer_raw = observer_source_path.read_bytes()
    if _address(observer_raw) != OBSERVER_SOURCE_SHA256:
        raise SkeletonGraphCalibrationPreregistrationError(
            "fixed-32 observer source differs"
        )

    precommit = _canonical_record(
        development_precommit_path,
        "skeleton development precommit",
        schema=DEVELOPMENT_PRECOMMIT_SCHEMA,
        record_digest=DEVELOPMENT_PRECOMMIT_RECORD_DIGEST,
        file_sha256=DEVELOPMENT_PRECOMMIT_FILE_SHA256,
    )
    result = _canonical_record(
        development_result_path,
        "skeleton development result",
        schema=DEVELOPMENT_RESULT_SCHEMA,
        record_digest=DEVELOPMENT_RESULT_RECORD_DIGEST,
        file_sha256=DEVELOPMENT_RESULT_FILE_SHA256,
    )
    if (
        "sha256:" + str(precommit.get("source_sha256")) != OBSERVER_SOURCE_SHA256
        or precommit.get("config_digest") != OBSERVER_CONFIG_DIGEST
        or precommit.get("protocol", {}).get("n_estimators") != 32
        or precommit.get("protocol", {}).get("n_jobs") != 1
        or result.get("config_digest") != OBSERVER_CONFIG_DIGEST
        or result.get("precommit_record_digest") != DEVELOPMENT_PRECOMMIT_RECORD_DIGEST
        or result.get("model_file_sha256") != MODEL_FILE_SHA256
        or result.get("feature_artifact_file_sha256") != FEATURE_ARTIFACT_FILE_SHA256
        or result.get("prediction_artifact_file_sha256")
        != PREDICTION_ARTIFACT_FILE_SHA256
        or result.get("promoted_heads") != ["direct_pair", "catalog_three_class"]
        or result.get("development_gate", {}).get("passed") is not True
        or result.get("claim_scope")
        != "finite_catalog_known_carrier_style_pose_transfer_engineering"
        or result.get("external_population_grant_required") is not True
        or result.get("population_scope_self_detectable_from_pixels") is not False
    ):
        raise SkeletonGraphCalibrationPreregistrationError(
            "fixed-32 development release binding differs"
        )

    v3_plan = _canonical_record(
        v3_plan_path,
        "fresh V3 plan",
        schema=V3_PLAN_SCHEMA,
        record_digest=V3_PLAN_RECORD_DIGEST,
        file_sha256=V3_PLAN_FILE_SHA256,
    )
    v3_manifest = _canonical_record(
        v3_calibration_manifest_path,
        "fresh V3 calibration identity manifest",
        schema=V3_CALIBRATION_MANIFEST_SCHEMA,
        record_digest=V3_CALIBRATION_MANIFEST_RECORD_DIGEST,
        file_sha256=V3_CALIBRATION_MANIFEST_FILE_SHA256,
    )
    generic_task_ids, generic_panel_ids = _validate_v3_identifiers(v3_plan, v3_manifest)

    family_plan = _canonical_record(
        same_family_plan_path,
        "same-family V2 plan",
        schema=SAME_FAMILY_PLAN_SCHEMA,
        record_digest=SAME_FAMILY_PLAN_RECORD_DIGEST,
        file_sha256=SAME_FAMILY_PLAN_FILE_SHA256,
        canonical_bytes_required=False,
    )
    _validate_family_partition(family_plan)

    ledger, ledger_raw = _object(exposure_ledger_path, "exposure ledger")
    if (
        _address(ledger_raw) != EXPOSURE_LEDGER_FILE_SHA256
        or ledger.get("schema") != "gkm.bongard-exposure-ledger.v1"
        or ledger.get("ledger_digest") != EXPOSURE_LEDGER_DIGEST
        or ledger.get("corpus_digest") != CORPUS_MANIFEST_DIGEST
    ):
        raise SkeletonGraphCalibrationPreregistrationError(
            "exposure-ledger predecessor differs"
        )

    generic_identity = {
        "manifest_path": _relative(repository_root, v3_calibration_manifest_path),
        "manifest_record_digest": V3_CALIBRATION_MANIFEST_RECORD_DIGEST,
        "manifest_source_sha256": V3_CALIBRATION_MANIFEST_FILE_SHA256,
        "panel_count": len(generic_panel_ids),
        "panel_ids_digest": "sha256:" + canonical_digest(generic_panel_ids),
        "rank_slice": [1100, 1200],
        "task_count": len(generic_task_ids),
        "task_ids_digest": "sha256:" + canonical_digest(generic_task_ids),
    }
    family_identity = {
        "diagnostic_tainted_task_ids": ["hd_convex-has_four_straight_lines_0001"],
        "official_validation_sealed_task_ids": [
            "hd_convex-has_four_straight_lines_0018",
            "hd_convex-has_four_straight_lines_0019",
        ],
        "panel_count": 224,
        "panel_order": "task_then_side_1_then_side_0_then_ordinal_0_through_6",
        "plan_path": _relative(repository_root, same_family_plan_path),
        "plan_record_digest": SAME_FAMILY_PLAN_RECORD_DIGEST,
        "plan_source_sha256": SAME_FAMILY_PLAN_FILE_SHA256,
        "target_sealed_task_ids": ["hd_convex-has_four_straight_lines_0000"],
        "task_count": 16,
        "task_ids": list(SAME_FAMILY_TASK_IDS),
        "task_ids_digest": "sha256:" + canonical_digest(SAME_FAMILY_TASK_IDS),
    }
    predecessor_gate = family_plan["efficiency_gate"]
    assert isinstance(predecessor_gate, dict)
    family_campaign = _campaign(
        alpha=0.10,
        task_count=16,
        panel_count=224,
        identity_binding=family_identity,
        population_scope="convex-four-lines_TRAIN_repetitions_0002_through_0017",
        target_authority=(
            "necessary_same-family_scope_evidence_only;_does_not_open_0000"
        ),
    )
    family_campaign["efficiency_gate"] = {
        **predecessor_gate,
        "raw_direct_pair_head_used": True,
        "source_v2_efficiency_gate_digest": (
            "sha256:" + canonical_digest(predecessor_gate)
        ),
        "straight_candidates_are_marginal_projection_of_full_54_pair_set": True,
    }

    body: dict[str, Any] = {
        "authorization": {
            "action_labels_or_programs_before_durable_prediction_reload": False,
            "calibration_pixels_authorized_by_this_metadata_record": False,
            "diagnostic_0001_pixels": False,
            "official_TEST_pixels": False,
            "official_validation_0018_0019_pixels": False,
            "target_0000_pixels": False,
        },
        "chronology": [
            "commit_and_cold-rebuild_this_metadata-only_record_before_any_new_calibration_pixel_label_or_action-program_byte",
            "implement_commit_and_independently_verify_the_skeleton_passed-fit_protocol_then_resolve_all_three_required_addresses_in_a_write-once_execution_precommit",
            "if_the_passed-fit_outcome_is_missing_malformed_or_GAP_emit_global_GAP_before_any_calibration_pixel",
            "append_the_exact_selected_task_ids_to_an_exposure-ledger_successor_and_freeze_exact_panel_identity_sha256_and_sizes_before_first_pixel_decode",
            "run_role-free_fixed-model_inference_for_every_campaign_panel_and_write_the_complete_prelabel_prediction_artifact",
            "fsync_prediction_file_and_directory_reload_exact_bytes_and_reverify_record_digest_before_constructing_or_invoking_any_action-label_or-program_loader",
            "open_only_the_frozen_campaign_labels_compute_one_two-head_max_score_per_whole_task_and_freeze_the_preregistered_order_statistic",
            "a_generic_grant_never_authorizes_the_target;_a_same-family_grant_is_necessary_but_still_requires_a_separate_external_population-membership_grant_and_target-release_authorization",
            "leave_target_0000_tainted_0001_validation_0018_0019_and_every_official_TEST_pixel_sealed",
        ],
        "claim": "metadata-only-fixed32-skeleton-calibration-preregistration-not-a-live-runner-or-benchmark",
        "current_state": {
            "action_label_rows_read_by_this_preregistration": 0,
            "action_program_bytes_read_by_this_preregistration": 0,
            "calibration_model_calls": 0,
            "calibration_panel_png_bytes_read": 0,
            "passed_fit_slot_resolved": False,
            "target_panel_png_bytes_read": 0,
        },
        "exposure_predecessor": {
            "ledger_digest": EXPOSURE_LEDGER_DIGEST,
            "ledger_path": _relative(repository_root, exposure_ledger_path),
            "ledger_source_sha256": EXPOSURE_LEDGER_FILE_SHA256,
            "mutation_by_preregistration": False,
            "successor_required_before_first_campaign_pixel": True,
        },
        "generic_v3_calibration": _campaign(
            alpha=0.05,
            task_count=100,
            panel_count=1_400,
            identity_binding=generic_identity,
            population_scope="fresh_generic_HD_TRAIN_known-carrier_style-pose_population",
            target_authority="cannot_authorize_target_under_any_outcome",
        ),
        "metadata_only_preregistration": True,
        "observer_release_binding": {
            "claim_scope": "finite_catalog_known_carrier_style_pose_transfer_engineering",
            "commit": OBSERVER_COMMIT,
            "config_digest": OBSERVER_CONFIG_DIGEST,
            "development_precommit_file_sha256": DEVELOPMENT_PRECOMMIT_FILE_SHA256,
            "development_precommit_path": _relative(
                repository_root, development_precommit_path
            ),
            "development_precommit_record_digest": DEVELOPMENT_PRECOMMIT_RECORD_DIGEST,
            "development_result_file_sha256": DEVELOPMENT_RESULT_FILE_SHA256,
            "development_result_path": _relative(repository_root, development_result_path),
            "development_result_record_digest": DEVELOPMENT_RESULT_RECORD_DIGEST,
            "feature_artifact_file_sha256": FEATURE_ARTIFACT_FILE_SHA256,
            "fixed_n_estimators": 32,
            "fixed_n_jobs": 1,
            "model_file_sha256": MODEL_FILE_SHA256,
            "prediction_artifact_file_sha256": PREDICTION_ARTIFACT_FILE_SHA256,
            "required_heads": ["direct_pair", "catalog_three_class"],
            "source_path": _relative(repository_root, observer_source_path),
            "source_sha256": OBSERVER_SOURCE_SHA256,
        },
        "passed_fit_authority_slot": passed_fit_slot(),
        "population_scope_contract": {
            "confidence_can_establish_population_membership": False,
            "generic_grant_schema": "gkm.bongard-skeleton-graph-generic-population-grant.v1",
            "generic_grant_target_authority": False,
            "novel_carrier_disposition_without_external_grant": "gap",
            "population_scope_self_detectable_from_pixels": False,
            "same_family_grant_schema": "gkm.bongard-skeleton-graph-same-family-population-grant.v1",
            "same_family_grant_target_authority": (
                "necessary_only;_target_requires_separate_exact_external_"
                "population-membership_grant_and_release_authorization"
            ),
            "task_id_or_role_can_enter_model_inference": False,
        },
        "preregistration_authority": {
            "source_path": _relative(repository_root, authority_source_path),
            "source_sha256": _address(authority_raw),
        },
        "raw_prediction_contract": {
            "catalog_probability_order": list(CATALOG_CLASS_ORDER),
            "direct_pair_probability_order": [
                [code // 10, code % 10] for code in OBSERVED_PAIR_CODES
            ],
            "forbidden_fields": [
                "action_label",
                "formula",
                "ordinal",
                "panel_path",
                "role",
                "side",
                "task_id",
            ],
            "required_fields": [
                "anonymous_panel_token",
                "png_sha256",
                "png_size",
                "feature_vector_sha256",
                "direct_pair_probabilities_33",
                "catalog_probabilities_3",
            ],
            "role_binding_occurs_only_after_durable_prediction_reload": True,
        },
        "same_family_calibration": family_campaign,
        "schema": SCHEMA,
        "score_and_projection_contract": _score_contract(),
        "supersession": {
            "fresh_v3_identity_manifest_retained_exactly": True,
            "old_cnn_calibration_head_score_and_runner_authority": "superseded",
            "old_cnn_plan_record_digest": V3_PLAN_RECORD_DIGEST,
            "old_cnn_plan_source_sha256": V3_PLAN_FILE_SHA256,
            "same_family_partition_retained_exactly": True,
            "same_family_v2_observer_head_adapter_and_score_authority": "superseded",
            "same_family_v2_plan_record_digest": SAME_FAMILY_PLAN_RECORD_DIGEST,
            "same_family_v2_plan_source_sha256": SAME_FAMILY_PLAN_FILE_SHA256,
        },
    }
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


__all__ = [
    "CATALOG_CLASS_ORDER",
    "OBSERVED_PAIR_CODES",
    "PASSED_FIT_ADDRESS_FIELDS",
    "PASSED_FIT_GAP_SCHEMA",
    "PASSED_FIT_PROTOCOL_SCHEMA",
    "SAME_FAMILY_TASK_IDS",
    "SCHEMA",
    "SkeletonGraphCalibrationPreregistrationError",
    "VALID_PAIR_CODES",
    "build_preregistration",
    "passed_fit_slot",
    "resolve_passed_fit_slot",
]
