"""Pre-pixel v2 preregistration for typed action-count/catalog learning.

V2 supersedes the metadata-only v1 plan before any selected PNG was opened.
It retains v1's first 1,000 hash-ranked exact-unused TRAIN tasks, changes the
last v1 cohort into conformal calibration, and adds the next 100 eligible
tasks as a sealed evaluation cohort.  Catalog targets are exact metadata
labels, not geometric truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.corpus import SplitIndex
from bongard.exposure import ExposureLedger
from bongard.panel_action_count_cnn_preregister import (
    ACTION_STYLES,
    SELECTION_SEED,
    _line_profile,
    _record_digest,
    _task_ids_from_action_plan,
    _task_ids_from_family_plan,
    _task_ids_from_historical,
)
from bongard.panel_convexity_catalog_audit import (
    RAW_LABEL_TO_CLASS,
    build_catalog_binding,
    catalog_label_for_actions,
    convexity_catalog_algorithm_digest,
    convexity_catalog_source_digest,
)
from bongard.release import load_official_release


SCHEMA = "gkm.bongard-action-count-catalog-cnn-preregistration.v2"
TRAIN_TASK_COUNT = 800
VALIDATION_TASK_COUNT = 100
CALIBRATION_TASK_COUNT = 100
EVALUATION_TASK_COUNT = 100
PANELS_PER_TASK = 14
ALPHA = 0.05
CALIBRATION_ORDER_STATISTIC = math.ceil((CALIBRATION_TASK_COUNT + 1) * (1 - ALPHA))


class ActionCountCNNV2PreregistrationError(RuntimeError):
    """V2 metadata, custody, or exact source bindings differ."""


def _address(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActionCountCNNV2PreregistrationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ActionCountCNNV2PreregistrationError(f"{label} is not an object")
    return value, raw


def _canonical_record(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    value, raw = _read_object(path, label)
    if raw != canonical_json(value) + b"\n":
        raise ActionCountCNNV2PreregistrationError(f"{label} is not canonical")
    _record_digest(value, label=label)
    return value, raw


def _load_tsv(path: Path) -> tuple[list[dict[str, str]], bytes]:
    raw = path.read_bytes()
    try:
        rows = list(csv.DictReader(raw.decode("utf-8").splitlines(), delimiter="\t"))
    except (UnicodeError, csv.Error) as exc:
        raise ActionCountCNNV2PreregistrationError(f"cannot parse TSV: {exc}") from exc
    if not rows:
        raise ActionCountCNNV2PreregistrationError("TSV is empty")
    return rows, raw


def _combined_rows(
    programs: Mapping[str, Any], task_ids: Sequence[str], binding: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        sides = programs.get(task_id)
        if not isinstance(sides, list) or len(sides) != 2:
            raise ActionCountCNNV2PreregistrationError(f"{task_id}: invalid task")
        for side_index, side in enumerate(sides):
            if not isinstance(side, list) or len(side) != 7:
                raise ActionCountCNNV2PreregistrationError(f"{task_id}: invalid side")
            folder = 1 - side_index
            for panel_index, panel in enumerate(side):
                if not isinstance(panel, list) or len(panel) != 1:
                    raise ActionCountCNNV2PreregistrationError(
                        f"{task_id}: panel is not one object"
                    )
                actions = panel[0]
                if not isinstance(actions, list) or any(
                    not isinstance(action, str) for action in actions
                ):
                    raise ActionCountCNNV2PreregistrationError(
                        f"{task_id}: invalid action list"
                    )
                parsed = [action.split("_", 2)[:2] for action in actions]
                if any(
                    len(item) != 2
                    or item[0] not in {"line", "arc"}
                    or item[1] not in ACTION_STYLES
                    for item in parsed
                ):
                    raise ActionCountCNNV2PreregistrationError(
                        f"{task_id}: invalid styled action"
                    )
                lines = [style for kind, style in parsed if kind == "line"]
                arcs = [style for kind, style in parsed if kind == "arc"]
                if len(lines) > 9 or len(arcs) > 9:
                    raise ActionCountCNNV2PreregistrationError(
                        f"{task_id}: action count leaves 0..9"
                    )
                catalog = catalog_label_for_actions(actions, binding)
                rows.append(
                    {
                        "arc_action_count": len(arcs),
                        "catalog_convexity_class": catalog.supervised_class,
                        "catalog_convexity_target": int(catalog.raw_label),
                        "catalog_match_kind": catalog.match_kind,
                        "crossing_task_stratum": "has_line_crossing" in task_id,
                        "line_decoration_stratum": _line_profile(lines),
                        "panel_id": f"hd/{task_id}/{folder}/{panel_index}.png",
                        "straight_action_count": len(lines),
                        "straight_count_4_stratum": len(lines) == 4,
                        "thin_task_stratum": "thin_shape" in task_id,
                    }
                )
    return rows


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "straight_action_count",
        "arc_action_count",
        "catalog_convexity_class",
        "catalog_match_kind",
        "line_decoration_stratum",
    )
    counters = {
        field: {
            str(value): sum(str(row[field]) == str(value) for row in rows)
            for value in sorted({str(row[field]) for row in rows})
        }
        for field in fields
    }
    return {
        **counters,
        "catalog_known_binary_panel_count": sum(
            row["catalog_convexity_target"] in {0, 1} for row in rows
        ),
        "crossing_task_panel_count": sum(row["crossing_task_stratum"] for row in rows),
        "panel_count": len(rows),
        "straight_count_4_panel_count": sum(
            row["straight_count_4_stratum"] for row in rows
        ),
        "thin_task_panel_count": sum(row["thin_task_stratum"] for row in rows),
    }


def _manifest(schema: str, claim: str, data: Mapping[str, Any]) -> dict[str, Any]:
    body = {"claim": claim, **data, "schema": schema}
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def build_v2_preregistration(
    *,
    repository_root: Path,
    dataset_root: Path,
    authority_source_path: Path,
    v1_plan_path: Path,
    v1_development_path: Path,
    v1_evaluation_panels_path: Path,
    v1_evaluation_labels_path: Path,
    action_count_plan_path: Path,
    family_plan_path: Path,
    historical_exposure_path: Path,
    cumulative_exposure_ledger_path: Path,
    catalog_audit_path: Path,
    shape_rows_path: Path,
    attribute_rows_path: Path,
    hd_programs_path: Path,
    bd_programs_path: Path,
    split_path: Path,
    release_descriptor_path: Path,
    development_output_path: Path,
    calibration_panels_output_path: Path,
    calibration_labels_output_path: Path,
    evaluation_panels_output_path: Path,
    evaluation_labels_output_path: Path,
) -> tuple[dict[str, Any], ...]:
    v1, v1_raw = _canonical_record(v1_plan_path, "v1 plan")
    v1_dev, v1_dev_raw = _canonical_record(v1_development_path, "v1 development")
    v1_eval_panels, v1_eval_panels_raw = _canonical_record(
        v1_evaluation_panels_path, "v1 evaluation panels"
    )
    v1_eval_labels, v1_eval_labels_raw = _canonical_record(
        v1_evaluation_labels_path, "v1 evaluation labels"
    )
    if (
        v1["current_state"]["selected_panel_png_bytes_read"] != 0
        or v1["current_state"]["model_training_started"] is not False
    ):
        raise ActionCountCNNV2PreregistrationError("v1 is not pre-pixel")

    programs, hd_raw = _read_object(hd_programs_path, "HD programs")
    bd_programs, bd_raw = _read_object(bd_programs_path, "BD programs")
    split = SplitIndex.load(split_path)
    release = load_official_release(release_descriptor_path)
    prior, prior_raw = _canonical_record(action_count_plan_path, "action-count plan")
    family, family_raw = _canonical_record(family_plan_path, "family plan")
    historical, historical_raw = _read_object(
        historical_exposure_path, "historical exposure"
    )
    cumulative_value, cumulative_raw = _read_object(
        cumulative_exposure_ledger_path, "cumulative exposure ledger"
    )
    cumulative = ExposureLedger.from_dict(cumulative_value)
    catalog_audit, catalog_audit_raw = _read_object(catalog_audit_path, "catalog audit")
    _record_digest(catalog_audit, label="catalog audit")
    shape_rows, shape_raw = _load_tsv(shape_rows_path)
    attribute_rows, attribute_raw = _load_tsv(attribute_rows_path)

    if cumulative.digest != v1["exclusions"]["cumulative_research_exposure_ledger"][
        "ledger_digest"
    ]:
        raise ActionCountCNNV2PreregistrationError("cumulative ledger differs from v1")
    if cumulative.corpus_digest != release.corpus_manifest_sha256:
        raise ActionCountCNNV2PreregistrationError("cumulative ledger corpus differs")
    source_bindings = catalog_audit["source_bindings"]
    for name, path, raw in (
        ("shape_rows", shape_rows_path, shape_raw),
        ("attribute_rows", attribute_rows_path, attribute_raw),
        ("hd_action_programs", hd_programs_path, hd_raw),
        ("bd_action_programs", bd_programs_path, bd_raw),
    ):
        expected = source_bindings[name]
        if expected["sha256"] != _address(raw):
            raise ActionCountCNNV2PreregistrationError(f"catalog {name} differs")
    if catalog_audit["algorithm"]["source_sha256"] != convexity_catalog_source_digest():
        raise ActionCountCNNV2PreregistrationError("catalog authority source differs")
    if catalog_audit["algorithm"]["algorithm_digest"] != (
        convexity_catalog_algorithm_digest()
    ):
        raise ActionCountCNNV2PreregistrationError("catalog algorithm differs")
    binding = build_catalog_binding(
        shape_rows=shape_rows,
        attribute_rows=attribute_rows,
        hd_programs=programs,
        bd_programs=bd_programs,
    )
    if len(binding.direct_by_signature) != 627 or len(binding.alias_by_signature) != 4:
        raise ActionCountCNNV2PreregistrationError("catalog inventory differs")

    hd_train = sorted(set(split.canonical_groups["train"]).intersection(programs))
    semantic_excluded = {
        task_id
        for task_id in hd_train
        if "convex" in task_id or "has_four_straight_lines" in task_id
    }
    cumulative_ids = {
        task_id for event in cumulative.events for task_id in event.task_ids
    }
    exclusions = semantic_excluded.union(
        _task_ids_from_action_plan(prior),
        _task_ids_from_family_plan(family),
        _task_ids_from_historical(historical),
        cumulative_ids,
    )
    eligible = sorted(set(hd_train).difference(exclusions))
    ranked = sorted(
        eligible,
        key=lambda task_id: (
            hashlib.sha256((SELECTION_SEED + "\0" + task_id).encode()).hexdigest(),
            task_id,
        ),
    )
    selected = ranked[
        : TRAIN_TASK_COUNT
        + VALIDATION_TASK_COUNT
        + CALIBRATION_TASK_COUNT
        + EVALUATION_TASK_COUNT
    ]
    if len(selected) != 1100:
        raise ActionCountCNNV2PreregistrationError("not enough eligible tasks")
    v1_selected = v1["oracle_taint_record"]["selected_task_ids"]
    if selected[:1000] != v1_selected:
        raise ActionCountCNNV2PreregistrationError("v2 is not exact v1 extension")
    train_tasks = selected[:800]
    validation_tasks = selected[800:900]
    calibration_tasks = selected[900:1000]
    evaluation_tasks = selected[1000:1100]
    if train_tasks != v1["cohorts"]["train"]["task_ids"]:
        raise ActionCountCNNV2PreregistrationError("training cohort changed")
    if validation_tasks != v1["cohorts"]["validation"]["task_ids"]:
        raise ActionCountCNNV2PreregistrationError("validation cohort changed")
    if calibration_tasks != v1["cohorts"]["evaluation"]["task_ids"]:
        raise ActionCountCNNV2PreregistrationError("v1 evaluation was not rekeyed exactly")

    rows = {
        "train": _combined_rows(programs, train_tasks, binding),
        "validation": _combined_rows(programs, validation_tasks, binding),
        "calibration": _combined_rows(programs, calibration_tasks, binding),
        "evaluation": _combined_rows(programs, evaluation_tasks, binding),
    }
    if any(len(value) != len(tasks) * 14 for value, tasks in zip(
        rows.values(), (train_tasks, validation_tasks, calibration_tasks, evaluation_tasks)
    )):
        raise ActionCountCNNV2PreregistrationError("cohort panel count differs")

    development = _manifest(
        "gkm.bongard-action-count-catalog-cnn-development-labels.v2",
        "oracle-tainted-train-and-checkpoint-selection-labels",
        {
            "cohorts": {
                "train": {"rows": rows["train"], "summary": _summary(rows["train"])},
                "validation": {
                    "rows": rows["validation"],
                    "summary": _summary(rows["validation"]),
                },
            }
        },
    )
    calibration_panels = _manifest(
        "gkm.bongard-action-count-catalog-cnn-calibration-panels.v2",
        "label-free-until-calibration-predictions-fsynced",
        {"panel_ids": [row["panel_id"] for row in rows["calibration"]]},
    )
    calibration_labels = _manifest(
        "gkm.bongard-action-count-catalog-cnn-calibration-labels.v2",
        "sealed-until-calibration-logits-and-probabilities-fsynced-and-reloaded",
        {"rows": rows["calibration"], "summary": _summary(rows["calibration"])},
    )
    evaluation_panels = _manifest(
        "gkm.bongard-action-count-catalog-cnn-evaluation-panels.v2",
        "label-free-frozen-evaluation-panels",
        {"panel_ids": [row["panel_id"] for row in rows["evaluation"]]},
    )
    evaluation_labels = _manifest(
        "gkm.bongard-action-count-catalog-cnn-evaluation-labels.v2",
        "sealed-until-calibrated-evaluation-predictions-fsynced-and-reloaded",
        {"rows": rows["evaluation"], "summary": _summary(rows["evaluation"])},
    )
    manifests = {
        "development_labels": (development_output_path, development),
        "calibration_panels_label_free": (
            calibration_panels_output_path,
            calibration_panels,
        ),
        "calibration_labels_sealed": (
            calibration_labels_output_path,
            calibration_labels,
        ),
        "evaluation_panels_label_free": (
            evaluation_panels_output_path,
            evaluation_panels,
        ),
        "evaluation_labels_sealed": (
            evaluation_labels_output_path,
            evaluation_labels,
        ),
    }
    manifest_bindings = {
        name: {
            "path": str(path.relative_to(repository_root)),
            "record_digest": value["record_digest"],
            "source_sha256": _address(canonical_json(value) + b"\n"),
        }
        for name, (path, value) in manifests.items()
    }
    manifest_bindings["calibration_labels_sealed"][
        "execution_open_condition"
    ] = "calibration_predictions_fsynced_and_reloaded"
    manifest_bindings["evaluation_labels_sealed"][
        "execution_open_condition"
    ] = "evaluation_predictions_fsynced_and_reloaded"

    cohort_tasks = {
        "train": train_tasks,
        "validation": validation_tasks,
        "calibration": calibration_tasks,
        "evaluation": evaluation_tasks,
    }
    cohorts = {
        name: {
            "action_and_catalog_label_rows_digest": "sha256:"
            + canonical_digest(rows[name]),
            "panel_count": len(rows[name]),
            "panel_ids_digest": "sha256:"
            + canonical_digest([row["panel_id"] for row in rows[name]]),
            "task_count": len(tasks),
            "task_ids": tasks,
            "task_ids_digest": "sha256:" + canonical_digest(tasks),
        }
        for name, tasks in cohort_tasks.items()
    }
    all_panel_ids = [
        row["panel_id"]
        for name in ("train", "validation", "calibration", "evaluation")
        for row in rows[name]
    ]
    source_raw = authority_source_path.read_bytes()
    plan_body: dict[str, Any] = {
        "calibration_protocol": {
            "alpha": ALPHA,
            "calibration_task_count": CALIBRATION_TASK_COUNT,
            "class_set_rule": "include_class_c_iff_1_minus_p_c_is_at_most_q",
            "individual_head_task_score": (
                "maximum_over_all_14_task_panels_of_1_minus_probability_of_true_class"
            ),
            "joint_task_score": (
                "maximum_over_all_14_panels_and_all_three_heads_of_1_minus_p_true"
            ),
            "canonical_deployment_q": "joint_q_only",
            "individual_head_q_values_are_diagnostics_only": True,
            "logits_and_probabilities_fsynced_and_reloaded_before_labels": True,
            "order_statistic_one_indexed": CALIBRATION_ORDER_STATISTIC,
            "q_rule": "ceil((n+1)*(1-alpha))_one-indexed_order_statistic",
            "zero_miss_max_used": False,
        },
        "chronology": [
            "commit_and_cold-replay_v2_and_all_five_manifests_before_any_selected_png_byte",
            "commit_trainer_then_precommit_only_11200_train_plus_1400_validation_PNG_sha256_sizes_runtime_and_source",
            "train_on_11200_and_select_checkpoint_only_with_the_fixed_1400-panel_validation_rule",
            "apply_fixed_validation_gate_and_if_it_fails_emit_gap_without_opening_any_calibration_or_evaluation_PNG_byte",
            "only_after_validation_pass_precommit_and_infer_the_1400_calibration_PNGs",
            "freeze_checkpoint_and_state-dict_digest_before_calibration_panels",
            "fsync_reload_calibration_logits_probabilities_before_opening_calibration_labels",
            "freeze_individual_and_joint_q_before_evaluation_panels",
            "only_after_q_freeze_precommit_and_infer_the_1400_evaluation_PNGs",
            "fsync_reload_evaluation_predictions_and_sets_before_opening_evaluation_labels",
            "leave_target_family_and_official_validation_TEST_pixels_sealed",
        ],
        "claim": "oracle-supervised-exact-unused-official-TRAIN-representation-engineering-not-bongard-benchmark",
        "cohorts": cohorts,
        "current_state": {
            "calibration_panel_png_bytes_read": 0,
            "calibration_labels_opened_by_execution": False,
            "evaluation_panel_png_bytes_read": 0,
            "evaluation_labels_opened_by_execution": False,
            "model_training_started": False,
            "official_validation_or_test_pixels_read": False,
            "staged_pixel_precommits_created": 0,
            "selected_panel_png_bytes_read": 0,
            "target_family_panel_pixels_read": False,
        },
        "dataset_and_authority_bindings": {
            "catalog_algorithm_digest": catalog_audit["algorithm"]["algorithm_digest"],
            "catalog_audit_record_digest": catalog_audit["record_digest"],
            "catalog_audit_source_sha256": _address(catalog_audit_raw),
            "catalog_authority_source_sha256": convexity_catalog_source_digest(),
            "catalog_compatibility_alias_count": len(binding.alias_by_signature),
            "catalog_direct_signature_count": len(binding.direct_by_signature),
            "cumulative_exposure_ledger_digest": cumulative.digest,
            "cumulative_exposure_source_sha256": _address(cumulative_raw),
            "hd_action_program_raw_sha256": _address(hd_raw),
            "official_release_descriptor_digest": release.digest,
            "split_manifest_digest": "sha256:" + canonical_digest(split.to_manifest_dict()),
            "split_source_sha256": _address(split_path.read_bytes()),
        },
        "exclusion_and_selection": {
            "action_count_plan_record_digest": prior["record_digest"],
            "eligible_hd_train_task_count": len(eligible),
            "eligible_task_ids_digest": "sha256:" + canonical_digest(eligible),
            "family_plan_record_digest": family["record_digest"],
            "hash_order": "sha256_utf8_seed_NUL_task_id_then_task_id",
            "selected_task_count": len(selected),
            "selected_task_ids_digest": "sha256:" + canonical_digest(selected),
            "selection_seed": SELECTION_SEED,
            "semantic_exclusion": "task_id_contains_convex_or_has_four_straight_lines",
            "v1_first_1000_reused_exactly": True,
        },
        "formal_claim_limits": {
            "catalog_labels_are_geometric_pixel_truth": False,
            "conformal_exchangeability_population": (
                "future_task_exchangeable_with_hash-sampled_eligible_official_TRAIN_pool_"
                "after_all_bound_exclusions"
            ),
            "conformal_grant_formally_transfers_to_target_family": False,
            "finite_sample_claim": (
                "marginal_whole-task_coverage_under_exchangeability_only_not_conditional_"
                "per-class_or_target-shift_coverage"
            ),
            "same_family_target_use": (
                "engineering_only_and_requires_preregistered_same-family_development_"
                "and_heldout-engineering_drill"
            ),
            "target_exclusions_may_be_called_certified": False,
        },
        "manifest_bindings": manifest_bindings,
        "metrics_and_checkpoint_selection": {
            "checkpoint_lexicographic_maximize": [
                "validation_straight_and_known-catalog-joint_exact",
                "validation_straight_top1_all_panels",
                "validation_known-catalog_binary_balanced_accuracy",
                "validation_arc_top1_all_panels",
                "negative_validation_total_cross_entropy",
                "negative_epoch_index_for_earliest_tie",
            ],
            "evaluation_go_no_go_all_must_hold": {
                "arc_top1_at_least": 0.85,
                "empirical_joint_whole-task_set_coverage_at_least": 0.90,
                "known_catalog_binary_balanced_accuracy_at_least": 0.70,
                "known_catalog_typed_decisive_rate_at_least": 0.30,
                "mean_straight_joint-q_set_size_at_most": 4.0,
                "straight_and_known_catalog_joint_exact_at_least": 0.55,
                "straight_joint-q_singleton_rate_at_least": 0.25,
                "straight_top1_at_least": 0.70,
                "true-straight-count-4_joint-q_singleton_rate_at_least": 0.25,
            },
            "metric_denominators": {
                "known_catalog_binary_balanced_accuracy": (
                    "mean_of_recall_on_true_nonconvex_rows_and_recall_on_true_convex_rows;_"
                    "catalog_unresolved_rows_excluded"
                ),
                "known_catalog_typed_decisive_rate": (
                    "denominator_all_evaluation_rows_with_true_catalog_target_0_or_1;_"
                    "numerator_joint-q_catalog_set_is_exactly_singleton_0_or_singleton_1"
                ),
                "straight_and_known_catalog_joint_exact": (
                    "denominator_all_rows_with_true_catalog_target_0_or_1;_numerator_"
                    "straight_top1_and_catalog_top1_both_correct"
                ),
                "straight_joint-q_singleton_rate": "all_evaluation_panels",
                "true-straight-count-4_joint-q_singleton_rate": (
                    "all_evaluation_panels_whose_true_straight_count_is_4"
                ),
            },
            "required_confusions": [
                "straight_10x10",
                "arc_10x10",
                "catalog_3x3_in_order_catalog_unresolved_nonconvex_convex",
            ],
            "required_strata": [
                "straight_true_count_4",
                "thin_shape_task_name",
                "has_line_crossing_task_name",
                "each_line_decoration_stratum",
                "known_catalog_binary_rows",
                "catalog_unresolved_rows_empirical_coverage_mean-set-width_and_typed-GAP-rate",
                "overall_catalog_typed-GAP-rate",
            ],
            "validation_gate_before_any_calibration_pixel": {
                "arc_top1_at_least": 0.80,
                "known_catalog_binary_balanced_accuracy_at_least": 0.65,
                "on_failure": "emit_typed_FIT-validation_GAP_and_leave_calibration_and_evaluation_pixels_and_labels_sealed",
                "straight_top1_at_least": 0.65,
            },
        },
        "oracle_taint_record": {
            "future_selectors_must_exclude_all_selected_task_ids": True,
            "permanent": True,
            "selected_panel_count": len(all_panel_ids),
            "selected_panel_ids_digest": "sha256:" + canonical_digest(all_panel_ids),
            "selected_task_count": len(selected),
            "selected_task_ids": selected,
            "selected_task_ids_digest": "sha256:" + canonical_digest(selected),
        },
        "preregistration_authority": {
            "source_path": str(authority_source_path.relative_to(repository_root)),
            "source_sha256": _address(source_raw),
        },
        "supersession": {
            "v1_plan_record_digest": v1["record_digest"],
            "v1_plan_source_sha256": _address(v1_raw),
            "v1_selected_png_bytes_read": 0,
            "v1_training_started": False,
            "v1_was_metadata_only_and_remains_oracle_taint_evidence": True,
            "v1_development_source_sha256": _address(v1_dev_raw),
            "v1_evaluation_panel_source_sha256": _address(v1_eval_panels_raw),
            "v1_evaluation_label_source_sha256": _address(v1_eval_labels_raw),
        },
        "training_protocol": {
            "augmentation_and_shuffle_key": (
                "sha256(seed_NUL_epoch_NUL_source_png_sha256);_path_task_side_and_panel_id_"
                "are_forbidden"
            ),
            "augmentation_index": (
                "unsigned_big-endian_integer_of_SHA256(seed_NUL_epoch_NUL_source_png_sha256)_modulo_8"
            ),
            "augmentation_transform_order": [
                "identity",
                "rotate_90_degrees_counterclockwise",
                "rotate_180_degrees",
                "rotate_270_degrees_counterclockwise",
                "horizontal_flip",
                "horizontal_flip_then_rotate_90_degrees_counterclockwise",
                "horizontal_flip_then_rotate_180_degrees",
                "horizontal_flip_then_rotate_270_degrees_counterclockwise",
            ],
            "batch_size": 64,
            "catalog_head_class_order": [
                "catalog_unresolved",
                "nonconvex",
                "convex",
            ],
            "catalog_unresolved_downstream_rule": (
                "any_calibrated_set_containing_catalog_unresolved_emits_axis_GAP_"
                "never_absence_and_never_typed_not_applicable"
            ),
            "class_weights": "per-head_inverse_sqrt_frequency_nonzero-mean-normalized",
            "cpu_threads": 1,
            "epochs": 16,
            "heads": {"arc": 10, "catalog_convexity": 3, "straight": 10},
            "image_size": 96,
            "initialization": (
                "after_torch_manual_seed_260810_initialize_convolution_weights_kaiming-normal_"
                "fan-out_relu_batchnorm_weight_1_bias_0_linear_weights_xavier-uniform_linear_bias_0"
            ),
            "learning_rate": 0.001,
            "loss": (
                "sum_of_three_per-head_inverse-sqrt-class-weighted_mean_cross-entropies"
            ),
            "model": {
                "blocks": (
                    "four_sequential_blocks_channels_16_32_64_96_each_Conv2d_kernel3_"
                    "stride2_padding1_biasFalse_then_BatchNorm2d_eps1e-5_momentum0.1_"
                    "affineTrue_track-running-statsTrue_then_ReLU_inplaceFalse"
                ),
                "dropout": False,
                "pool": "AdaptiveAvgPool2d_output_1x1_then_flatten_96",
                "three_heads": "independent_Linear_biasTrue_96-to-10_96-to-10_96-to-3",
            },
            "optimizer": "AdamW",
            "optimizer_parameters": {
                "betas": [0.9, 0.999],
                "eps": 1e-08,
                "weight_decay": 0.0001,
            },
            "preprocessing": {
                "decode": "Pillow_open_exact_precommitted_bytes_convert_L_single_frame_PNG",
                "empty_ink": "ERROR_and_remains_in_denominator",
                "ink_mask": "grayscale_value_strictly_less_than_250",
                "normalization": "float32_(255-resized_grayscale)/255_no_other_normalization",
                "square_pad": (
                    "tight_ink_bbox_then_white_square_side_max(height,width)+2*ceil(0.08*max(height,width));_"
                    "center_with_any_odd_extra_pixel_on_bottom_or_right"
                ),
                "resize": "Pillow_96x96_Resampling.BILINEAR",
            },
            "pretrained_or_network_weights": False,
            "random_seed": 260810,
            "torch_deterministic_algorithms": True,
            "weight_decay": 0.0001,
            "checkpoint": (
                "model_state_dict_selected_epoch_architecture_id_and_class_orders_only;_"
                "no_optimizer_scheduler_or_RNG_state;_tensor-stable_state-dict_SHA256_is_authority"
            ),
            "staged_pixel_custody": {
                "calibration_and_evaluation_bytes_may_not_be_hashed_stat-ed_or_decoded_in_train-validation_stage": True,
                "calibration_tensors_not_decoded_before_validation_gate_pass": True,
                "evaluation_tensors_not_decoded_before_joint_q_freeze": True,
                "train-validation_precommit_panel_count": 12600,
                "calibration_precommit_panel_count_after_validation_pass": 1400,
                "evaluation_precommit_panel_count_after_q_freeze": 1400,
            },
        },
        "transport_and_language_limits": {
            "formula_synthesis_present": False,
            "lean_present": False,
            "lean_required": False,
            "model_inputs": ["preprocessed_grayscale_panel_pixels_only"],
            "task_id_semantics_or_side_enter_model_tensor_order_or_augmentation": False,
        },
    }
    plan = {**plan_body, "record_digest": "sha256:" + canonical_digest(plan_body)}
    return (
        plan,
        development,
        calibration_panels,
        calibration_labels,
        evaluation_panels,
        evaluation_labels,
    )


def write_outputs(paths: Sequence[Path], values: Sequence[Mapping[str, Any]]) -> None:
    if len(paths) != len(values):
        raise ActionCountCNNV2PreregistrationError("output cardinality differs")
    for path, value in zip(paths, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(value) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository-root", "dataset-root", "v1-plan", "v1-development",
        "v1-evaluation-panels", "v1-evaluation-labels", "action-count-plan",
        "family-plan", "historical-exposure", "cumulative-exposure-ledger",
        "catalog-audit", "shape-rows", "attribute-rows", "hd-programs",
        "bd-programs", "split", "release-descriptor", "plan-output",
        "development-output", "calibration-panels-output",
        "calibration-labels-output", "evaluation-panels-output",
        "evaluation-labels-output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    values = build_v2_preregistration(
        repository_root=args.repository_root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        authority_source_path=Path(__file__).resolve(),
        v1_plan_path=args.v1_plan.resolve(),
        v1_development_path=args.v1_development.resolve(),
        v1_evaluation_panels_path=args.v1_evaluation_panels.resolve(),
        v1_evaluation_labels_path=args.v1_evaluation_labels.resolve(),
        action_count_plan_path=args.action_count_plan.resolve(),
        family_plan_path=args.family_plan.resolve(),
        historical_exposure_path=args.historical_exposure.resolve(),
        cumulative_exposure_ledger_path=args.cumulative_exposure_ledger.resolve(),
        catalog_audit_path=args.catalog_audit.resolve(),
        shape_rows_path=args.shape_rows.resolve(),
        attribute_rows_path=args.attribute_rows.resolve(),
        hd_programs_path=args.hd_programs.resolve(),
        bd_programs_path=args.bd_programs.resolve(),
        split_path=args.split.resolve(),
        release_descriptor_path=args.release_descriptor.resolve(),
        development_output_path=args.development_output.resolve(),
        calibration_panels_output_path=args.calibration_panels_output.resolve(),
        calibration_labels_output_path=args.calibration_labels_output.resolve(),
        evaluation_panels_output_path=args.evaluation_panels_output.resolve(),
        evaluation_labels_output_path=args.evaluation_labels_output.resolve(),
    )
    paths = (
        args.plan_output.resolve(),
        args.development_output.resolve(),
        args.calibration_panels_output.resolve(),
        args.calibration_labels_output.resolve(),
        args.evaluation_panels_output.resolve(),
        args.evaluation_labels_output.resolve(),
    )
    write_outputs(paths, values)
    print(json.dumps([value["record_digest"] for value in values]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
