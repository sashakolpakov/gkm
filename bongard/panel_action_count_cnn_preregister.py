"""Metadata-only preregistration for supervised HD action-count learning.

This authority deliberately stops before PNG bytes.  It selects exact TRAIN
tasks from split/task metadata, then binds the action-program-derived labels
and panel paths that will be permanently oracle-tainted.  A later execution
precommit must bind every PNG byte and the trainer source before decoding or
training.
"""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.corpus import SplitIndex
from bongard.exposure import ExposureLedger
from bongard.release import load_official_release


SCHEMA = "gkm.bongard-action-count-cnn-preregistration.v1"
MANIFEST_SCHEMA = "gkm.bongard-action-count-cnn-label-manifest.v1"
SELECTION_SEED = "gkm-panel-action-count-cnn-supervised-train-dev-20260810-v1"
TRAIN_TASK_COUNT = 800
VALIDATION_TASK_COUNT = 100
EVALUATION_TASK_COUNT = 100
PANELS_PER_TASK = 14
ACTION_STYLES = ("circle", "normal", "square", "triangle", "zigzag")
TARGET_FAMILY = "hd_convex-has_four_straight_lines"


class ActionCountCNNPreregistrationError(RuntimeError):
    """The metadata-only preregistration cannot be constructed exactly."""


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActionCountCNNPreregistrationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ActionCountCNNPreregistrationError(f"{label} must be a JSON object")
    return value, raw


def _record_digest(value: Mapping[str, Any], *, label: str) -> str:
    found = value.get("record_digest")
    if not isinstance(found, str):
        raise ActionCountCNNPreregistrationError(f"{label} lacks record_digest")
    body = dict(value)
    del body["record_digest"]
    expected = "sha256:" + canonical_digest(body)
    if found != expected:
        raise ActionCountCNNPreregistrationError(f"{label} record digest differs")
    return found


def _task_ids_from_family_plan(plan: Mapping[str, Any]) -> list[str]:
    partition = plan.get("frozen_partition")
    if not isinstance(partition, dict):
        raise ActionCountCNNPreregistrationError("family plan partition is invalid")
    task_ids: list[str] = []
    for value in partition.values():
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ActionCountCNNPreregistrationError("family plan task IDs are invalid")
        task_ids.extend(value)
    if len(task_ids) != len(set(task_ids)):
        raise ActionCountCNNPreregistrationError("family plan task IDs overlap")
    return sorted(task_ids)


def _task_ids_from_action_plan(plan: Mapping[str, Any]) -> list[str]:
    cohorts = plan.get("cohorts")
    if not isinstance(cohorts, dict):
        raise ActionCountCNNPreregistrationError("action-count cohorts are invalid")
    task_ids: list[str] = []
    for cohort in cohorts.values():
        if not isinstance(cohort, dict):
            raise ActionCountCNNPreregistrationError("action-count cohort is invalid")
        values = cohort.get("task_ids")
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ActionCountCNNPreregistrationError(
                "action-count cohort task IDs are invalid"
            )
        task_ids.extend(values)
    if len(task_ids) != len(set(task_ids)):
        raise ActionCountCNNPreregistrationError("action-count cohort task IDs overlap")
    return sorted(task_ids)


def _task_ids_from_historical(seed_envelope: Mapping[str, Any]) -> list[str]:
    seed = seed_envelope.get("seed")
    if not isinstance(seed, dict):
        raise ActionCountCNNPreregistrationError("historical exposure seed is invalid")
    exact = seed.get("exact_official_exposure")
    if not isinstance(exact, dict) or not isinstance(exact.get("task_ids"), list):
        raise ActionCountCNNPreregistrationError(
            "historical exact official exposure is invalid"
        )
    task_ids: list[str] = []
    for row in exact["task_ids"]:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise ActionCountCNNPreregistrationError(
                "historical exact task record is invalid"
            )
        task_ids.append(row["task_id"])
    if len(task_ids) != len(set(task_ids)):
        raise ActionCountCNNPreregistrationError("historical exact task IDs overlap")
    return sorted(task_ids)


def _line_profile(styles: Sequence[str]) -> str:
    if not styles:
        return "no_straight_actions"
    normal = sum(style == "normal" for style in styles)
    if normal == len(styles):
        return "normal_only"
    if normal == 0:
        return "decorated_only"
    return "mixed_normal_and_decorated"


def _rows_for_tasks(
    programs: Mapping[str, Any], task_ids: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        sides = programs.get(task_id)
        if not isinstance(sides, list) or len(sides) != 2:
            raise ActionCountCNNPreregistrationError(
                f"{task_id}: action program must have two sides"
            )
        for side_index, side in enumerate(sides):
            if not isinstance(side, list) or len(side) != 7:
                raise ActionCountCNNPreregistrationError(
                    f"{task_id}: action-program side must have seven panels"
                )
            folder = 1 if side_index == 0 else 0
            for panel_index, panel in enumerate(side):
                if not isinstance(panel, list) or len(panel) != 1:
                    raise ActionCountCNNPreregistrationError(
                        f"{task_id}: panel action program is invalid"
                    )
                actions = panel[0]
                if not isinstance(actions, list) or any(
                    not isinstance(action, str) for action in actions
                ):
                    raise ActionCountCNNPreregistrationError(
                        f"{task_id}: action list is invalid"
                    )
                parsed = [action.split("_", 2)[:2] for action in actions]
                if any(
                    len(action) != 2
                    or action[0] not in {"line", "arc"}
                    or action[1] not in ACTION_STYLES
                    for action in parsed
                ):
                    raise ActionCountCNNPreregistrationError(
                        f"{task_id}: unsupported action token"
                    )
                lines = [style for kind, style in parsed if kind == "line"]
                arcs = [style for kind, style in parsed if kind == "arc"]
                if len(lines) > 9 or len(arcs) > 9:
                    raise ActionCountCNNPreregistrationError(
                        f"{task_id}: action count exceeds closed domain 0..9"
                    )
                rows.append(
                    {
                        "arc_action_count": len(arcs),
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
    straight = Counter(str(row["straight_action_count"]) for row in rows)
    arc = Counter(str(row["arc_action_count"]) for row in rows)
    decoration = Counter(str(row["line_decoration_stratum"]) for row in rows)
    return {
        "arc_action_count": dict(sorted(arc.items(), key=lambda item: int(item[0]))),
        "crossing_task_panel_count": sum(bool(row["crossing_task_stratum"]) for row in rows),
        "line_decoration_stratum": dict(sorted(decoration.items())),
        "panel_count": len(rows),
        "straight_action_count": dict(
            sorted(straight.items(), key=lambda item: int(item[0]))
        ),
        "straight_count_4_panel_count": sum(
            bool(row["straight_count_4_stratum"]) for row in rows
        ),
        "thin_task_panel_count": sum(bool(row["thin_task_stratum"]) for row in rows),
    }


def build_preregistration(
    *,
    repository_root: Path,
    dataset_root: Path,
    authority_source_path: Path,
    development_label_manifest_path: Path,
    evaluation_panel_manifest_path: Path,
    evaluation_label_manifest_path: Path,
    action_count_plan_path: Path,
    family_plan_path: Path,
    historical_exposure_path: Path,
    cumulative_exposure_ledger_path: Path,
    action_program_audit_path: Path,
    release_descriptor_path: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Build the exact plan and label manifest without opening a PNG."""

    split_path = dataset_root / "ShapeBongard_V2_split.json"
    action_path = dataset_root / "hd/hd_action_programs.json"
    split = SplitIndex.load(split_path)
    programs, action_raw = _read_json_object(action_path, label="HD action programs")
    prior, prior_raw = _read_json_object(
        action_count_plan_path, label="existing action-count plan"
    )
    family, family_raw = _read_json_object(family_plan_path, label="family plan")
    historical, historical_raw = _read_json_object(
        historical_exposure_path, label="historical exposure"
    )
    cumulative_raw_value, cumulative_raw = _read_json_object(
        cumulative_exposure_ledger_path, label="cumulative exposure ledger"
    )
    cumulative_ledger = ExposureLedger.from_dict(cumulative_raw_value)
    audit, audit_raw = _read_json_object(
        action_program_audit_path, label="action-program audit"
    )
    release_raw = release_descriptor_path.read_bytes()
    release = load_official_release(release_descriptor_path)
    if cumulative_ledger.corpus_digest != release.corpus_manifest_sha256:
        raise ActionCountCNNPreregistrationError(
            "cumulative exposure ledger belongs to another corpus"
        )
    prior_digest = _record_digest(prior, label="existing action-count plan")
    family_digest = _record_digest(family, label="family plan")

    train_ids = set(split.canonical_groups["train"])
    hd_train = sorted(train_ids.intersection(programs))
    semantic_excluded = sorted(
        task_id
        for task_id in hd_train
        if "convex" in task_id or "has_four_straight_lines" in task_id
    )
    prior_ids = _task_ids_from_action_plan(prior)
    family_ids = _task_ids_from_family_plan(family)
    historical_ids = _task_ids_from_historical(historical)
    cumulative_ids = sorted(
        {task_id for event in cumulative_ledger.events for task_id in event.task_ids}
    )
    if len(cumulative_ledger.events) != 158 or len(cumulative_ids) != 314:
        raise ActionCountCNNPreregistrationError(
            "cumulative exposure ledger does not have the preregistered 158 events / 314 tasks"
        )
    explicit_excluded = set(prior_ids).union(
        family_ids, historical_ids, cumulative_ids
    )
    exclusion_union = set(semantic_excluded).union(explicit_excluded)
    eligible = sorted(set(hd_train).difference(exclusion_union))
    ranked = sorted(
        eligible,
        key=lambda task_id: (
            hashlib.sha256((SELECTION_SEED + "\0" + task_id).encode("utf-8")).hexdigest(),
            task_id,
        ),
    )
    selected = ranked[
        : TRAIN_TASK_COUNT + VALIDATION_TASK_COUNT + EVALUATION_TASK_COUNT
    ]
    if len(selected) != TRAIN_TASK_COUNT + VALIDATION_TASK_COUNT + EVALUATION_TASK_COUNT:
        raise ActionCountCNNPreregistrationError("not enough eligible TRAIN tasks")
    train_tasks = selected[:TRAIN_TASK_COUNT]
    validation_tasks = selected[
        TRAIN_TASK_COUNT : TRAIN_TASK_COUNT + VALIDATION_TASK_COUNT
    ]
    evaluation_tasks = selected[TRAIN_TASK_COUNT + VALIDATION_TASK_COUNT :]

    # Selection is complete before any selected action-program value is used.
    train_rows = _rows_for_tasks(programs, train_tasks)
    validation_rows = _rows_for_tasks(programs, validation_tasks)
    evaluation_rows = _rows_for_tasks(programs, evaluation_tasks)
    if len(train_rows) != TRAIN_TASK_COUNT * PANELS_PER_TASK:
        raise ActionCountCNNPreregistrationError("training panel count differs")
    if len(validation_rows) != VALIDATION_TASK_COUNT * PANELS_PER_TASK:
        raise ActionCountCNNPreregistrationError("validation panel count differs")
    if len(evaluation_rows) != EVALUATION_TASK_COUNT * PANELS_PER_TASK:
        raise ActionCountCNNPreregistrationError("evaluation panel count differs")
    panel_ids = [
        row["panel_id"]
        for row in train_rows + validation_rows + evaluation_rows
    ]
    if len(panel_ids) != len(set(panel_ids)):
        raise ActionCountCNNPreregistrationError("selected panel IDs overlap")

    development_label_manifest_body: dict[str, Any] = {
        "claim": "oracle-tainted-supervised-train-metadata-not-a-bongard-benchmark",
        "cohorts": {
            "train": {"rows": train_rows, "summary": _summary(train_rows)},
            "validation": {
                "rows": validation_rows,
                "summary": _summary(validation_rows),
            },
        },
        "panel_order": (
            "selected_task_hash_rank_then_action_program_side_0_as_folder_1_panels_0_to_6_"
            "then_side_1_as_folder_0_panels_0_to_6"
        ),
        "schema": MANIFEST_SCHEMA,
    }
    development_label_manifest = dict(development_label_manifest_body)
    development_label_manifest["record_digest"] = (
        "sha256:" + canonical_digest(development_label_manifest_body)
    )
    development_manifest_bytes = canonical_json(development_label_manifest) + b"\n"

    evaluation_panel_manifest_body: dict[str, Any] = {
        "claim": "label-free-frozen-evaluation-panel-custody",
        "panel_ids": [row["panel_id"] for row in evaluation_rows],
        "schema": "gkm.bongard-action-count-cnn-evaluation-panel-manifest.v1",
    }
    evaluation_panel_manifest = dict(evaluation_panel_manifest_body)
    evaluation_panel_manifest["record_digest"] = (
        "sha256:" + canonical_digest(evaluation_panel_manifest_body)
    )
    evaluation_panel_manifest_bytes = canonical_json(evaluation_panel_manifest) + b"\n"

    evaluation_label_manifest_body: dict[str, Any] = {
        "claim": "sealed-until-evaluation-predictions-are-fsynced",
        "rows": evaluation_rows,
        "schema": "gkm.bongard-action-count-cnn-evaluation-label-manifest.v1",
        "summary": _summary(evaluation_rows),
    }
    evaluation_label_manifest = dict(evaluation_label_manifest_body)
    evaluation_label_manifest["record_digest"] = (
        "sha256:" + canonical_digest(evaluation_label_manifest_body)
    )
    evaluation_label_manifest_bytes = canonical_json(evaluation_label_manifest) + b"\n"

    authority_raw = authority_source_path.read_bytes()
    split_raw = split_path.read_bytes()
    historical_seed_digest = historical.get("seed_digest")
    audit_digest = audit.get("digest")
    if not isinstance(historical_seed_digest, str) or not isinstance(audit_digest, str):
        raise ActionCountCNNPreregistrationError("input metadata digest is missing")

    plan_body: dict[str, Any] = {
        "chronology": [
            "freeze_and_commit_this_plan_and_all_three_exact_manifests_before_any_selected_png_byte_is_read",
            "implement_and_commit_the_trainer_against_the_frozen_protocol",
            "run_pixel_precommit_to_bind_every_selected_png_sha256_and_size_plus_trainer_source_and_runtime_before_decode",
            "cold_verify_the_pixel_precommit_then_decode_only_the_14000_selected_train-split_pngs",
            "train_on_11200_training_panels_and_select_checkpoint_only_by_the_preregistered_1400-panel-validation_rule",
            "freeze_the_selected_checkpoint_and_state-dict_digest_then_infer_the_1400_label-free-evaluation_panels",
            "fsync_and_reload_all_evaluation_predictions_before_opening_the_separate_sealed-evaluation-label_manifest",
            "score_the_frozen_evaluation_as_the_only_go-no-go_result_and_emit_confusions_and_strata",
            "never_open_target_family_official_validation_or_official_test_pixels_in_this_lane",
        ],
        "claim": "oracle-supervised-official-train-only-action-count-representation-engineering-not-bongard-benchmark",
        "cohorts": {
            "evaluation": {
                "action_label_rows_digest": "sha256:"
                + canonical_digest(evaluation_rows),
                "panel_count": len(evaluation_rows),
                "panel_ids_digest": "sha256:"
                + canonical_digest([row["panel_id"] for row in evaluation_rows]),
                "task_count": len(evaluation_tasks),
                "task_ids": evaluation_tasks,
                "task_ids_digest": "sha256:" + canonical_digest(evaluation_tasks),
            },
            "train": {
                "action_label_rows_digest": "sha256:" + canonical_digest(train_rows),
                "panel_count": len(train_rows),
                "panel_ids_digest": "sha256:"
                + canonical_digest([row["panel_id"] for row in train_rows]),
                "task_count": len(train_tasks),
                "task_ids": train_tasks,
                "task_ids_digest": "sha256:" + canonical_digest(train_tasks),
            },
            "validation": {
                "action_label_rows_digest": "sha256:"
                + canonical_digest(validation_rows),
                "panel_count": len(validation_rows),
                "panel_ids_digest": "sha256:"
                + canonical_digest([row["panel_id"] for row in validation_rows]),
                "task_count": len(validation_tasks),
                "task_ids": validation_tasks,
                "task_ids_digest": "sha256:" + canonical_digest(validation_tasks),
            },
        },
        "current_state": {
            "model_training_started": False,
            "official_test_pixels_read": False,
            "official_validation_pixels_read": False,
            "pixel_precommit_created": False,
            "evaluation_labels_opened_by_training_execution": False,
            "evaluation_predictions_created": False,
            "selected_action_labels_read_for_preregistration": True,
            "selected_panel_png_bytes_read": 0,
            "target_family_panel_pixels_read": False,
        },
        "dataset_bindings": {
            "action_program_audit_record_digest": audit_digest,
            "action_program_audit_source_sha256": _sha256_bytes(audit_raw),
            "corpus_manifest_digest": release.corpus_manifest_sha256,
            "hd_action_program_parsed_canonical_sha256": _sha256_bytes(
                canonical_json(programs)
            ),
            "hd_action_program_raw_sha256": _sha256_bytes(action_raw),
            "hd_action_program_size_bytes": len(action_raw),
            "official_release_descriptor_digest": release.digest,
            "official_release_descriptor_source_sha256": _sha256_bytes(release_raw),
            "split_manifest_digest": "sha256:" + canonical_digest(split.to_manifest_dict()),
            "split_source_sha256": _sha256_bytes(split_raw),
            "split_source_size_bytes": len(split_raw),
            "task_inventory_digest": release.task_ids_sha256,
        },
        "execution_boundary": {
            "decoder": "Pillow_grayscale_L_then_deterministic_ink_bbox_square_pad_and_96x96_bilinear",
            "pixel_precommit_must_bind": [
                "all_14000_ordered_panel_ids",
                "each_png_sha256",
                "each_png_size_bytes",
                "ordered_png_manifest_digest",
                "trainer_source_sha256",
                "preregistration_plan_source_sha256",
                "all_three_manifest_source_sha256_values",
                "python_torch_numpy_pillow_versions",
                "platform_and_cpu_thread_contract",
            ],
            "png_sha256_or_size_known_at_preregistration": False,
            "selected_png_existence_stat_or_bytes_required_to_select": False,
            "training_requires_cold_verified_pixel_precommit": True,
        },
        "exclusions": {
            "action_count_plan": {
                "record_digest": prior_digest,
                "source_sha256": _sha256_bytes(prior_raw),
                "task_count": len(prior_ids),
                "task_ids_digest": "sha256:" + canonical_digest(prior_ids),
            },
            "eligible_hd_train_task_count": len(eligible),
            "eligible_hd_train_task_ids_digest": "sha256:" + canonical_digest(eligible),
            "family_drill_plan": {
                "record_digest": family_digest,
                "source_sha256": _sha256_bytes(family_raw),
                "task_count": len(family_ids),
                "task_ids_digest": "sha256:" + canonical_digest(family_ids),
            },
            "cumulative_research_exposure_ledger": {
                "event_count": len(cumulative_ledger.events),
                "exposed_task_count": len(cumulative_ids),
                "exposed_task_ids_digest": "sha256:"
                + canonical_digest(cumulative_ids),
                "ledger_digest": cumulative_ledger.digest,
                "source_path": str(
                    cumulative_exposure_ledger_path.relative_to(repository_root)
                ),
                "source_sha256": _sha256_bytes(cumulative_raw),
            },
            "historical_exposure": {
                "exact_task_count": len(historical_ids),
                "exact_task_ids_digest": "sha256:" + canonical_digest(historical_ids),
                "seed_digest": historical_seed_digest,
                "source_sha256": _sha256_bytes(historical_raw),
            },
            "official_validation_or_test_task_selected": False,
            "semantic_closure_rule": "exclude_hd_train_task_if_task_id_contains_convex_or_has_four_straight_lines",
            "semantic_closure_task_count": len(semantic_excluded),
            "semantic_closure_task_ids_digest": "sha256:"
            + canonical_digest(semantic_excluded),
            "target_family_0000_through_0019_selected": False,
            "union_task_count_with_cross-source_overlap": len(exclusion_union),
            "union_task_ids_digest": "sha256:" + canonical_digest(sorted(exclusion_union)),
        },
        "label_contract": {
            "arc_action_count": "number_of_arc_actions_in_the_panel_render_program",
            "class_domain": list(range(10)),
            "decoration_does_not_multiply_carrier_actions": True,
            "folder_mapping": "action_program_side_0_is_folder_1_and_side_1_is_folder_0",
            "label_manifest_contains_task_names_only_inside_panel_paths_and_evaluation_strata": True,
            "line_decoration_stratum": (
                "derived_from_line_action_styles_as_no_straight_actions_normal_only_"
                "decorated_only_or_mixed_normal_and_decorated"
            ),
            "straight_action_count": "number_of_line_actions_in_the_panel_render_program",
            "task_id_semantic_name_or_side_never_enters_model_tensor": True,
        },
        "manifest_bindings": {
            "development_labels": {
                "path": str(development_label_manifest_path.relative_to(repository_root)),
                "record_digest": development_label_manifest["record_digest"],
                "source_sha256": _sha256_bytes(development_manifest_bytes),
            },
            "evaluation_labels_sealed": {
                "may_be_opened_by_execution_only_after_fsynced_predictions": True,
                "path": str(evaluation_label_manifest_path.relative_to(repository_root)),
                "record_digest": evaluation_label_manifest["record_digest"],
                "source_sha256": _sha256_bytes(evaluation_label_manifest_bytes),
            },
            "evaluation_panels_label_free": {
                "path": str(evaluation_panel_manifest_path.relative_to(repository_root)),
                "record_digest": evaluation_panel_manifest["record_digest"],
                "source_sha256": _sha256_bytes(evaluation_panel_manifest_bytes),
            },
        },
        "metrics": {
            "checkpoint_selection": (
                "lexicographic_maximize_validation_joint_exact_then_mean_of_straight_and_arc_"
                "top1_then_minimize_total_cross_entropy_then_choose_earliest_epoch"
            ),
            "evaluation_go_no_go_all_must_hold": {
                "arc_top1_accuracy_at_least": 0.85,
                "joint_exact_accuracy_at_least": 0.65,
                "straight_top1_accuracy_at_least": 0.70,
                "straight_true_count_4_accuracy_at_least": 0.70,
            },
            "evaluation_is_only_go_no_go_cohort": True,
            "primary": [
                "straight_top1_accuracy",
                "arc_top1_accuracy",
                "joint_exact_accuracy",
            ],
            "required_confusions": [
                "straight_10_by_10_true_rows_predicted_columns",
                "arc_10_by_10_true_rows_predicted_columns",
            ],
            "required_strata": [
                "straight_true_count_4",
                "thin_shape_task_name",
                "has_line_crossing_task_name",
                "each_line_decoration_stratum",
            ],
            "errors_or_missing_panels_remain_in_denominator": True,
            "evaluation_predictions_must_be_fsynced_and_reloaded_before_labels_open": True,
            "training_metrics_are_diagnostic_not_checkpoint_selection_inputs": True,
        },
        "oracle_taint_record": {
            "accounting_unit": "exact_task_id_and_all_fourteen_panels",
            "future_cohort_selectors_must_exclude_selected_task_ids_digest": (
                "sha256:" + canonical_digest(selected)
            ),
            "permanent": True,
            "selected_panel_count": len(panel_ids),
            "selected_panel_ids_digest": "sha256:" + canonical_digest(panel_ids),
            "selected_task_count": len(selected),
            "selected_task_ids": selected,
            "selected_task_ids_digest": "sha256:" + canonical_digest(selected),
        },
        "preregistration_authority": {
            "source_path": str(authority_source_path.relative_to(repository_root)),
            "source_sha256": _sha256_bytes(authority_raw),
        },
        "selection": {
            "algorithm": (
                "intersect_hd_action_program_task_keys_with_official_train_then_apply_"
                "all_exclusions_then_sort_by_sha256_utf8_seed_NUL_task_id_then_task_id_"
                "take_first_1000_first_800_train_next_100_validation_last_100_evaluation"
            ),
            "algorithm_digest": "sha256:"
            + canonical_digest(
                {
                    "evaluation_task_count": EVALUATION_TASK_COUNT,
                    "hash": "sha256",
                    "separator_hex": "00",
                    "seed": SELECTION_SEED,
                    "train_task_count": TRAIN_TASK_COUNT,
                    "validation_task_count": VALIDATION_TASK_COUNT,
                }
            ),
            "official_hd_train_task_count": len(hd_train),
            "pixel_independent": True,
            "selected_action_program_values_do_not_affect_task_ranking": True,
            "selected_task_count": len(selected),
            "selected_task_ids_digest": "sha256:" + canonical_digest(selected),
            "selection_seed": SELECTION_SEED,
            "selection_seed_digest": _sha256_bytes(SELECTION_SEED.encode("utf-8")),
        },
        "supervision_and_claim_limits": {
            "all_selected_tasks_and_panels_permanently_oracle_tainted": True,
            "bongard_side_or_class_is_not_a_training_target": True,
            "formula_synthesis_present": False,
            "lean_present": False,
            "lean_removable": True,
            "lean_required": False,
            "official_validation_or_test_authorized": False,
            "pixels_are_the_only_model_inputs": True,
            "scientific_bongard_benchmark_claim_authorized": False,
            "target_task_or_family_authorized": False,
        },
        "training_protocol": {
            "augmentation": (
                "train_only_hash_derived_D4_transform_from_seed_epoch_and_panel_id;_"
                "validation_and_evaluation_have_no_augmentation"
            ),
            "batch_size": 64,
            "checkpoint_selection_uses_validation": True,
            "evaluation_labels_used_for_checkpoint_selection_or_tuning": False,
            "class_weight": "per_head_inverse_sqrt_nonzero_train_frequency_normalized_to_nonzero_mean_one",
            "cpu_threads": 1,
            "epochs": 16,
            "image_size": 96,
            "learning_rate": 0.001,
            "model": (
                "one_channel_shared_CNN_blocks_16_32_64_96_each_3x3_batchnorm_relu_"
                "stride2_then_adaptive_average_pool_then_independent_linear_10_class_heads"
            ),
            "optimizer": "AdamW",
            "pretrained_weights": False,
            "random_seed": 260810,
            "scheduler": "none",
            "torch_deterministic_algorithms": True,
            "weight_decay": 0.0001,
        },
    }
    plan = dict(plan_body)
    plan["record_digest"] = "sha256:" + canonical_digest(plan_body)
    return (
        plan,
        development_label_manifest,
        evaluation_panel_manifest,
        evaluation_label_manifest,
    )


def write_preregistration(
    *,
    plan: Mapping[str, Any],
    development_label_manifest: Mapping[str, Any],
    evaluation_panel_manifest: Mapping[str, Any],
    evaluation_label_manifest: Mapping[str, Any],
    plan_path: Path,
    development_label_manifest_path: Path,
    evaluation_panel_manifest_path: Path,
    evaluation_label_manifest_path: Path,
) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    for path, value in (
        (development_label_manifest_path, development_label_manifest),
        (evaluation_panel_manifest_path, evaluation_panel_manifest),
        (evaluation_label_manifest_path, evaluation_label_manifest),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(value) + b"\n")
    plan_path.write_bytes(canonical_json(plan) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--development-label-manifest-output", type=Path, required=True)
    parser.add_argument("--evaluation-panel-manifest-output", type=Path, required=True)
    parser.add_argument("--evaluation-label-manifest-output", type=Path, required=True)
    parser.add_argument("--action-count-plan", type=Path, required=True)
    parser.add_argument("--family-plan", type=Path, required=True)
    parser.add_argument("--historical-exposure", type=Path, required=True)
    parser.add_argument("--cumulative-exposure-ledger", type=Path, required=True)
    parser.add_argument("--action-program-audit", type=Path, required=True)
    parser.add_argument("--release-descriptor", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    authority_source = Path(__file__).resolve()
    plan, development, evaluation_panels, evaluation_labels = build_preregistration(
        repository_root=args.repository_root.resolve(),
        dataset_root=args.dataset_root.resolve(),
        authority_source_path=authority_source,
        development_label_manifest_path=args.development_label_manifest_output.resolve(),
        evaluation_panel_manifest_path=args.evaluation_panel_manifest_output.resolve(),
        evaluation_label_manifest_path=args.evaluation_label_manifest_output.resolve(),
        action_count_plan_path=args.action_count_plan.resolve(),
        family_plan_path=args.family_plan.resolve(),
        historical_exposure_path=args.historical_exposure.resolve(),
        cumulative_exposure_ledger_path=args.cumulative_exposure_ledger.resolve(),
        action_program_audit_path=args.action_program_audit.resolve(),
        release_descriptor_path=args.release_descriptor.resolve(),
    )
    write_preregistration(
        plan=plan,
        development_label_manifest=development,
        evaluation_panel_manifest=evaluation_panels,
        evaluation_label_manifest=evaluation_labels,
        plan_path=args.plan_output.resolve(),
        development_label_manifest_path=args.development_label_manifest_output.resolve(),
        evaluation_panel_manifest_path=args.evaluation_panel_manifest_output.resolve(),
        evaluation_label_manifest_path=args.evaluation_label_manifest_output.resolve(),
    )
    print(
        json.dumps(
            {
                "development_label_manifest_record_digest": development[
                    "record_digest"
                ],
                "evaluation_label_manifest_record_digest": evaluation_labels[
                    "record_digest"
                ],
                "evaluation_panel_manifest_record_digest": evaluation_panels[
                    "record_digest"
                ],
                "plan_record_digest": plan["record_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "ActionCountCNNPreregistrationError",
    "EVALUATION_TASK_COUNT",
    "MANIFEST_SCHEMA",
    "PANELS_PER_TASK",
    "SCHEMA",
    "SELECTION_SEED",
    "TRAIN_TASK_COUNT",
    "VALIDATION_TASK_COUNT",
    "build_preregistration",
    "main",
    "write_preregistration",
)
