"""Run the preregistered four-task FIT-only action-decomposition ablation.

The command has no phase selector.  It can open only four already-oracle-tainted
FIT task batches named by the frozen parent outcome.  Each panel is shown as
raw/crop/edge with a blank fourth quadrant.  The model returns up to four typed
decomposition tuples separating normal/decorated straight/arc actions.  Python
alone projects finite total-count sets.  Predictions are durably reloaded before
the action-program labels open, and cold replay rebuilds every view with zero
physical model calls.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnCallFailed,
    ObjectBongardTurnRuntime,
)
from bongard.panel_action_count_multiview_fit_command import _residual_curve
from bongard.panel_action_count_phase_command import (
    DEFAULT_ACTION_PROGRAM_FILE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_LAUNCHER_SHA256,
    DEFAULT_MODEL,
    DEFAULT_PLAN,
    DEFAULT_REASONING_EFFORT,
    PLAN_RECORD_DIGEST,
    STYLES,
    _load_plan,
    _panel_path,
    _presentation,
    _read_action_programs_after_predictions,
    _read_png,
    _truth_records,
)
from bongard.panel_action_decomposition_threeview_adapter import (
    PARENT_OUTCOME_DIGEST,
    build_action_decomposition_threeview,
    panel_action_decomposition_threeview_adapter_source_digest,
    threeview_algorithm_record,
)
from bongard.panel_probe_custody import (
    load_or_create_probe_runtime as _runtime,
    make_probe_record as _record,
    panel_probe_custody_source_digest,
    read_probe_record as _read_record,
    write_once_or_verify_probe_record as _write_once_or_verify,
)
from bongard.panel_probe_transport import (
    call_panel_probe as _call,
    panel_probe_transport_source_digest,
)
from bongard.transport import CodexStructuredResult, run_codex_named_images_structured


PARENT_OUTCOME_SCHEMA = "gkm.bongard-action-count-multiview-fit-outcome.v1"
PARENT_OUTCOME_FILE_SHA256 = (
    "942779a08ba4029a35af2572ed202a7e3de5e469d6f921360fb4a06348e61a46"
)
SELECTED_TASK_IDS = (
    "hd_thin_shape_0010",
    "hd_has_six_straight_lines-thin_shape_0007",
    "hd_has_line_crossing-exist_quadrangle_0002",
    "hd_has_acute_angle-necked_0013",
)
SELECTED_TRUTH_MANIFEST_DIGEST = (
    "sha256:82ac3ce3169e95090725a8a0e945af4d9a9cae581fe7437a440ccca853a48162"
)
SELECTED_PANEL_IDS_DIGEST = (
    "sha256:4f1aae15cbb0a91b8d678c1571061ca09014b4f1522b8eb701e593a66056f74a"
)
VIEW_NAMES = tuple(f"view_{index:02d}.png" for index in range(14))
COMPONENTS = (
    "normal_straight_count",
    "decorated_straight_count",
    "normal_arc_count",
    "decorated_arc_count",
)
MAX_TUPLES = 4
ERROR_VALUES = ("none", "unreadable")
AUTHORIZATION_SCHEMA = "gkm.bongard-action-decomposition-fit-authorization.v1"
PRECOMMIT_SCHEMA = "gkm.bongard-action-decomposition-fit-precommit.v1"
TASK_PREDICTION_SCHEMA = "gkm.bongard-action-decomposition-task-prediction.v1"
PREDICTION_BATCH_SCHEMA = "gkm.bongard-action-decomposition-prediction-batch.v1"
LABEL_RELEASE_SCHEMA = "gkm.bongard-action-decomposition-label-release.v1"
RESULT_SCHEMA = "gkm.bongard-action-decomposition-fit-result.v1"
REPLAY_SCHEMA = "gkm.bongard-action-decomposition-fit-cold-replay.v1"
DEFAULT_PARENT_OUTCOME = (
    Path(__file__).resolve().parent
    / "data/panel_action_count_multiview_fit_outcome_20260810_v1.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "downloads/ShapeBongard_V2_full/panel_action_decomposition_fit_ablation_20260810_v1"
)


class ActionDecompositionFitAblationError(RuntimeError):
    """The frozen ablation, view, prediction, label, or replay differs."""


def panel_action_decomposition_fit_ablation_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _load_parent_outcome(
    path: str | Path, *, expected_digest: str = PARENT_OUTCOME_DIGEST
) -> tuple[dict[str, Any], str]:
    source = Path(os.path.abspath(os.fspath(path)))
    value = _read_record(source)
    raw = source.read_bytes()
    expected_frontend = {
        "status": "recommended_not_executed",
        "opened_fit_task_ids": list(SELECTED_TASK_IDS),
        "physical_calls": 4,
        "panels": 56,
        "views": "raw_plus_square_crop_plus_binary_edge; ablate_coarse_carrier_density",
        "typed_output": (
            "up_to_four_finite_alternative_tuples_over_normal_connected_straight_"
            "decorated_carrier_straight_normal_arc_decorated_arc; Python projects total counts"
        ),
        "renderer_grammar": (
            "connected corner-to-corner solid runs are normal actions; disconnected "
            "repeated markers or dense repetitive zigzags are decorated carrier actions"
        ),
        "control_reason": (
            "acute-angle task preserves a cohort where multiview fixed large baseline undercounts"
        ),
        "calibration_or_heldout_calls": 0,
        "target_calls": 0,
        "batch_protocol_is_fit_tuning_only": True,
    }
    if (
        value.get("schema") != PARENT_OUTCOME_SCHEMA
        or value.get("record_digest") != expected_digest
        or hashlib.sha256(raw).hexdigest() != PARENT_OUTCOME_FILE_SHA256
        or value.get("smallest_next_fit_frontend") != expected_frontend
        or value.get("release_disposition")
        != "multiview_fit_observer_not_qualified_for_absence_calibration_or_query_release"
    ):
        raise ActionDecompositionFitAblationError("parent FIT outcome differs")
    return value, hashlib.sha256(raw).hexdigest()


def action_decomposition_output_schema() -> dict[str, object]:
    properties: dict[str, object] = {}
    for name in VIEW_NAMES:
        stem = name.removesuffix(".png")
        properties[f"{stem}_decomposition_counts"] = {
            "type": "array",
            "items": {"type": "integer", "enum": list(range(10))},
        }
        properties[f"{stem}_error_code"] = {
            "type": "string",
            "enum": list(ERROR_VALUES),
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def action_decomposition_prompt() -> str:
    names = ", ".join(VIEW_NAMES)
    return (
        "Inspect fourteen independent drawings named " + names + ". Their order is "
        "randomized. They are not groups, examples of a shared rule, classes, support, "
        "or query roles. Never infer a task concept. Each named image shows one drawing: "
        "top-left is the alpha-composited source at original scale, top-right is its "
        "deterministic square ink crop, bottom-left is its binary inner edge, and "
        "bottom-right is deliberately blank. For each drawing return up to four ranked "
        "decomposition tuples. Return them in decomposition_counts as one flat array. "
        "Every consecutive group of four integers is one tuple in this exact order: "
        "normal_straight_count, decorated_straight_count, normal_arc_count, "
        "decorated_arc_count. The first tuple is mandatory and best. Later tuples are "
        "only genuinely plausible alternatives. Thus a readable array has length exactly "
        "4, 8, 12, or 16 and every integer is in 0..9. "
        "normal_straight_count counts continuous solid corner-to-corner straight drawing "
        "actions. Every visible direction-changing corner separates normal actions, even "
        "when the whole path is thin, narrow, self-overlapping, or forms attached "
        "triangular lobes. Do not call a connected solid triangular lobe marker decoration. "
        "decorated_straight_count counts broader straight carrier actions rendered as "
        "disconnected repeated circles, squares, or triangles, or as dense repetitive "
        "zigzag texture. Marker perimeters and zigzag teeth do not add actions. "
        "normal_arc_count and decorated_arc_count apply the same distinction to curved "
        "actions. A crossing alone does not split a continuing action; a genuine corner, "
        "endpoint, or transition to another ordered action does. The straight-component "
        "sum and arc-component sum of each tuple must each lie in 0..9. Used tuples must "
        "be distinct. For a readable image error_code is none. If unreadable, set "
        "error_code unreadable and return the exact empty decomposition_counts array. "
        "Return no prose, confidence, total counts, intervals, "
        "formula, predicate, closure or convexity judgment, polarity, task label, or class "
        "decision. Dataset IDs, action programs, truth counts, phases, and side labels are "
        "unavailable."
    )


def _prepare_inputs(
    dataset_root: Path,
    task_ids: Sequence[str],
    plan_digest: str,
) -> tuple[dict[str, bytes], tuple[dict[str, Any], ...]]:
    montages: dict[str, bytes] = {}
    task_inputs: list[dict[str, Any]] = []
    for task_id in task_ids:
        rows: list[dict[str, Any]] = []
        for name, panel_id in _presentation(task_id, plan_digest):
            raw = _read_png(_panel_path(dataset_root, panel_id))
            montage, view = build_action_decomposition_threeview(raw)
            if (
                view["source_png_sha256"] != hashlib.sha256(raw).hexdigest()
                or view["montage_png_sha256"] != hashlib.sha256(montage).hexdigest()
            ):
                raise ActionDecompositionFitAblationError("three-view byte binding differs")
            montages[panel_id] = montage
            rows.append(
                {
                    "model_visible_name": name,
                    "panel_id": panel_id,
                    "source_png_sha256": view["source_png_sha256"],
                    "source_png_size_bytes": len(raw),
                    "threeview_record": view,
                }
            )
        task_inputs.append({"task_id": task_id, "presentation": rows})
    return montages, tuple(task_inputs)


def _authorization_precommit(
    *,
    plan: Mapping[str, Any],
    parent: Mapping[str, Any],
    parent_file_sha256: str,
    selected_truth_manifest_digest: str,
    task_inputs: Sequence[Mapping[str, Any]],
    workers: int,
    model: str,
    reasoning_effort: str,
    minutes: int,
    executable: str,
    launcher_sha256: str,
    runtime_injected: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = action_decomposition_prompt()
    schema = action_decomposition_output_schema()
    algorithm = threeview_algorithm_record()
    authorization = _record(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "command_source_digest": (
                panel_action_decomposition_fit_ablation_source_digest()
            ),
            "adapter_source_digest": (
                panel_action_decomposition_threeview_adapter_source_digest()
            ),
            "custody_source_digest": panel_probe_custody_source_digest(),
            "probe_transport_source_digest": panel_probe_transport_source_digest(),
            "threeview_algorithm": algorithm,
            "parent_fit_outcome_record_digest": parent["record_digest"],
            "parent_fit_outcome_file_sha256": parent_file_sha256,
            "plan_record_digest": plan["record_digest"],
            "phase": "fit_ablation",
            "fit_only": True,
            "selected_task_ids": list(SELECTED_TASK_IDS),
            "selected_panel_ids_digest": SELECTED_PANEL_IDS_DIGEST,
            "sealed_selected_truth_manifest_digest": selected_truth_manifest_digest,
            "task_inputs": list(task_inputs),
            "task_count": 4,
            "panel_count": 56,
            "model_visible_names": list(VIEW_NAMES),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "output_schema_digest": "sha256:" + canonical_digest(schema),
            "output_contract": {
                "ranked_decomposition_tuple_count_maximum": MAX_TUPLES,
                "flattened_tuple_component_order": list(COMPONENTS),
                "readable_flat_array_lengths": [4, 8, 12, 16],
                "unreadable_flat_array": [],
                "each_array_item_schema_enum": list(range(10)),
                "schema_valid_semantic_shape_error_is_panel_local": True,
                "Python_projects_total_candidate_sets": True,
                "model_total_counts_or_intervals_present": False,
            },
            "individual_action_labels_opened_by_process_before_prediction": False,
            "fit_labels_historically_open_for_protocol_tuning": True,
            "all_selected_fit_pixels_previously_opened": True,
            "new_cohort_pixels_opened": False,
            "calibration_heldout_family_or_target_pixels_opened": False,
            "action_program_values_used_in_presentation_order": False,
            "model_visible_task_side_split_phase_or_label": False,
            "workers": workers,
            "runtime_request": {
                "model": model,
                "reasoning_effort": reasoning_effort,
                "minutes": minutes,
                "executable": executable,
                "launcher_sha256": launcher_sha256,
                "injected_for_test": runtime_injected,
            },
            "fit_batch_protocol_calibration_transfer_to_query_style_calls": False,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
            "engineering_only": True,
            "scientific_benchmark": False,
        }
    )
    precommit = _record(
        {
            "schema": PRECOMMIT_SCHEMA,
            "authorization_digest": authorization["record_digest"],
            "parent_fit_outcome_record_digest": parent["record_digest"],
            "plan_record_digest": plan["record_digest"],
            "phase": "fit_ablation",
            "task_ids": list(SELECTED_TASK_IDS),
            "sealed_selected_truth_manifest_digest": selected_truth_manifest_digest,
            "physical_call_plan": {
                "named_threeview_batch_calls_per_task": 1,
                "task_count": 4,
                "maximum_physical_calls": 4,
            },
            "one_call_contains_exactly_fourteen_neutral_threeview_images": True,
            "typed_decomposition_components": list(COMPONENTS),
            "noncontiguous_alternative_tuples_preserved": True,
            "total_candidate_sets_projected_only_by_python": True,
            "formula_or_predicate_synthesis_present": False,
            "predictions_must_be_fsynced_and_reloaded_before_action_program_open": True,
            "labels_model_visible": False,
            "exactly_once_external_journals_required": True,
            "cold_replay_model_calls": 0,
            "new_cohort_calibration_heldout_family_and_target_calls": 0,
            "fit_batch_is_frontend_protocol_tuning_only": True,
            "workers": workers,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
        }
    )
    return authorization, precommit


def _project_candidates(
    tuples: Sequence[Mapping[str, int]],
) -> tuple[list[int], list[int], list[list[int]]]:
    straight: list[int] = []
    arc: list[int] = []
    joint: list[list[int]] = []
    for item in tuples:
        straight_total = item["normal_straight_count"] + item["decorated_straight_count"]
        arc_total = item["normal_arc_count"] + item["decorated_arc_count"]
        if straight_total not in straight:
            straight.append(straight_total)
        if arc_total not in arc:
            arc.append(arc_total)
        pair = [straight_total, arc_total]
        if pair not in joint:
            joint.append(pair)
    return straight, arc, joint


def _parsed_row(
    item: Mapping[str, Any],
    *,
    raw_counts: list[int],
    candidates: Sequence[Mapping[str, int]],
    error_code: str,
) -> dict[str, Any]:
    frozen_candidates = [dict(row) for row in candidates]
    straight, arc, joint = _project_candidates(frozen_candidates)
    return {
        "panel_id": item["panel_id"],
        "model_visible_name": item["model_visible_name"],
        "source_png_sha256": item["source_png_sha256"],
        "threeview_record_digest": item["threeview_record"]["record_digest"],
        "montage_png_sha256": item["threeview_record"]["montage_png_sha256"],
        "raw_decomposition_counts": list(raw_counts),
        "flattened_tuple_component_order": list(COMPONENTS),
        "decomposition_candidates": frozen_candidates,
        "best_decomposition": frozen_candidates[0] if frozen_candidates else None,
        "straight_candidate_counts": straight,
        "arc_candidate_counts": arc,
        "joint_total_candidates": joint,
        "totals_projected_only_by_python": True,
        "error_code": error_code,
    }


def _parse_payload(
    payload: Mapping[str, Any],
    presentation: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    schema = action_decomposition_output_schema()
    if set(payload) != set(schema["properties"]):
        raise ActionDecompositionFitAblationError("typed payload fields differ")
    rows: list[dict[str, Any]] = []
    for item in presentation:
        stem = item["model_visible_name"].removesuffix(".png")
        error = payload[f"{stem}_error_code"]
        raw_counts = payload[f"{stem}_decomposition_counts"]
        if error not in ERROR_VALUES:
            raise ActionDecompositionFitAblationError("typed error code differs")
        if (
            type(raw_counts) is not list
            or any(type(value) is not int or not 0 <= value <= 9 for value in raw_counts)
        ):
            raise ActionDecompositionFitAblationError(
                "schema-bounded decomposition array differs"
            )
        if error == "unreadable":
            rows.append(
                _parsed_row(
                    item,
                    raw_counts=raw_counts,
                    candidates=(),
                    error_code=(
                        "unreadable" if not raw_counts else "invalid_unreadable_payload"
                    ),
                )
            )
            continue
        if len(raw_counts) not in (4, 8, 12, 16):
            rows.append(
                _parsed_row(
                    item,
                    raw_counts=raw_counts,
                    candidates=(),
                    error_code="invalid_tuple_array_length",
                )
            )
            continue
        candidates = [
            dict(zip(COMPONENTS, raw_counts[offset : offset + 4], strict=True))
            for offset in range(0, len(raw_counts), 4)
        ]
        if any(
            row["normal_straight_count"] + row["decorated_straight_count"] > 9
            or row["normal_arc_count"] + row["decorated_arc_count"] > 9
            for row in candidates
        ):
            rows.append(
                _parsed_row(
                    item,
                    raw_counts=raw_counts,
                    candidates=(),
                    error_code="invalid_projected_total",
                )
            )
            continue
        canonical = [tuple(row[name] for name in COMPONENTS) for row in candidates]
        if len(canonical) != len(set(canonical)):
            rows.append(
                _parsed_row(
                    item,
                    raw_counts=raw_counts,
                    candidates=(),
                    error_code="duplicate_decomposition_tuple",
                )
            )
            continue
        rows.append(
            _parsed_row(
                item,
                raw_counts=raw_counts,
                candidates=candidates,
                error_code="none",
            )
        )
    return tuple(rows)


def _error_rows(
    presentation: Sequence[Mapping[str, Any]], error_code: str
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "panel_id": item["panel_id"],
            "model_visible_name": item["model_visible_name"],
            "source_png_sha256": item["source_png_sha256"],
            "threeview_record_digest": item["threeview_record"]["record_digest"],
            "montage_png_sha256": item["threeview_record"]["montage_png_sha256"],
            "raw_decomposition_counts": None,
            "flattened_tuple_component_order": list(COMPONENTS),
            "decomposition_candidates": [],
            "best_decomposition": None,
            "straight_candidate_counts": [],
            "arc_candidate_counts": [],
            "joint_total_candidates": [],
            "totals_projected_only_by_python": True,
            "error_code": error_code,
        }
        for item in presentation
    )


def _observe_task(
    *,
    root: Path,
    task_input: Mapping[str, Any],
    montages: Mapping[str, bytes],
    authorization_digest: str,
    precommit_digest: str,
    runtime: ObjectBongardTurnRuntime,
    underlying_transport: Callable[..., CodexStructuredResult],
    replay: bool,
) -> dict[str, Any]:
    task_id = task_input["task_id"]
    presentation = task_input["presentation"]
    images = tuple(
        (row["model_visible_name"], montages[row["panel_id"]])
        for row in presentation
    )
    journal_root = root / "journals" / task_id
    if replay and not (journal_root / "outcome.json").is_file():
        raise ActionDecompositionFitAblationError(
            "cold replay found no terminal journal"
        )
    journal = ObjectBongardNamedImageTurnJournalTransport(
        journal_root,
        authorization_digest=authorization_digest,
        execution_precommit_digest=precommit_digest,
        task_id=task_id,
        turn_kind="action_decomposition_threeview_batch",
        expected_prompt=action_decomposition_prompt(),
        expected_images=images,
        expected_output_schema=action_decomposition_output_schema(),
        runtime=runtime,
        underlying_transport=underlying_transport,
    )
    payload: Mapping[str, Any] | None = None
    receipt_digest: str | None = None
    transport_failed = False
    try:
        payload, receipt = _call(
            images,
            prompt=action_decomposition_prompt(),
            schema=action_decomposition_output_schema(),
            journal=journal,
            runtime=runtime,
        )
        receipt_digest = receipt.receipt_digest
    except ObjectBongardTurnCallFailed:
        transport_failed = True
    terminal = journal.verify()
    if transport_failed:
        if terminal.terminal_status != "failure":
            raise ActionDecompositionFitAblationError(
                "failed call has no failure terminal"
            )
        rows = _error_rows(presentation, "transport_error")
        status = "error"
        error_code = "transport_error"
    else:
        if terminal.terminal_status != "success" or payload is None:
            raise ActionDecompositionFitAblationError(
                "successful call has no success terminal"
            )
        try:
            rows = _parse_payload(payload, presentation)
        except ActionDecompositionFitAblationError:
            rows = _error_rows(presentation, "invalid_typed_payload")
            status = "error"
            error_code = "invalid_typed_payload"
        else:
            status = "success"
            error_code = None
    return _record(
        {
            "schema": TASK_PREDICTION_SCHEMA,
            "task_id": task_id,
            "presentation": list(presentation),
            "rows": list(rows),
            "status": status,
            "error_code": error_code,
            "receipt_digest": receipt_digest,
            "journal_terminal": terminal.to_data(),
            "external_journal_terminal_verified": True,
            "individual_action_labels_opened_by_process_before_prediction": False,
            "model_visible_task_or_side_labels": False,
            "finite_decomposition_tuples_preserved": True,
            "total_candidate_sets_projected_only_by_python": True,
        }
    )


def _truth_decomposition(truth: Mapping[str, Any]) -> dict[str, int]:
    normal_straight = truth["line_action_count_by_style"]["normal"]
    normal_arc = truth["arc_action_count_by_style"]["normal"]
    return {
        "normal_straight_count": normal_straight,
        "decorated_straight_count": truth["straight_action_count"] - normal_straight,
        "normal_arc_count": normal_arc,
        "decorated_arc_count": truth["arc_action_count"] - normal_arc,
    }


def _axis_metrics(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], axis: str
) -> dict[str, Any]:
    top1_exact = set_coverage = cardinality = valid = errors = 0
    panel_residuals: list[int] = []
    task_residuals: dict[str, list[int]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for prediction, truth in pairs:
        target = truth[f"{axis}_action_count"]
        candidates = prediction[f"{axis}_candidate_counts"]
        if prediction["error_code"] != "none" or not candidates:
            errors += 1
            confusion[str(target)]["error"] += 1
            continue
        valid += 1
        best = candidates[0]
        residual = min(abs(target - value) for value in candidates)
        top1_exact += int(best == target)
        set_coverage += int(target in candidates)
        cardinality += len(candidates)
        panel_residuals.append(residual)
        task_residuals[prediction["panel_id"].split("/")[1]].append(residual)
        confusion[str(target)][",".join(str(value) for value in candidates)] += 1
    task_max = [
        max(values)
        for _task_id, values in sorted(task_residuals.items())
        if len(values) == 14
    ]
    denominator = len(pairs)
    return {
        "denominator": denominator,
        "valid_prediction_count": valid,
        "error_count": errors,
        "top1_exact_count": top1_exact,
        "top1_exact_rate": [top1_exact, denominator],
        "finite_candidate_set_coverage_count": set_coverage,
        "finite_candidate_set_coverage_rate": [set_coverage, denominator],
        "candidate_cardinality_sum": cardinality,
        "mean_candidate_cardinality": [cardinality, valid],
        "finite_candidate_set_confusion": {
            target: dict(sorted(counter.items()))
            for target, counter in sorted(
                confusion.items(), key=lambda item: int(item[0])
            )
        },
        "panel_residual_curve": _residual_curve(panel_residuals),
        "task_max_residual_curve": _residual_curve(task_max),
        "task_max_curve_excludes_tasks_with_any_error": True,
    }


def _decomposition_metrics(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> dict[str, Any]:
    top1_exact = candidate_coverage = joint_total_coverage = valid = errors = 0
    component_top1 = Counter()
    component_coverage = Counter()
    for prediction, truth in pairs:
        candidates = prediction["decomposition_candidates"]
        if prediction["error_code"] != "none" or not candidates:
            errors += 1
            continue
        valid += 1
        target = _truth_decomposition(truth)
        top1_exact += int(candidates[0] == target)
        candidate_coverage += int(target in candidates)
        target_pair = [truth["straight_action_count"], truth["arc_action_count"]]
        joint_total_coverage += int(target_pair in prediction["joint_total_candidates"])
        for component in COMPONENTS:
            component_top1[component] += int(
                candidates[0][component] == target[component]
            )
            component_coverage[component] += int(
                any(row[component] == target[component] for row in candidates)
            )
    denominator = len(pairs)
    return {
        "denominator": denominator,
        "valid_prediction_count": valid,
        "error_count": errors,
        "top1_exact_decomposition_count": top1_exact,
        "top1_exact_decomposition_rate": [top1_exact, denominator],
        "finite_decomposition_set_coverage_count": candidate_coverage,
        "finite_decomposition_set_coverage_rate": [candidate_coverage, denominator],
        "joint_total_pair_coverage_count": joint_total_coverage,
        "joint_total_pair_coverage_rate": [joint_total_coverage, denominator],
        "component_top1_exact_counts": {
            component: component_top1[component] for component in COMPONENTS
        },
        "component_finite_set_marginal_coverage_counts": {
            component: component_coverage[component] for component in COMPONENTS
        },
    }


def _count_four_audit(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> dict[str, Any]:
    counts = Counter()
    for prediction, truth in pairs:
        candidates = prediction["straight_candidate_counts"]
        truth_is_four = truth["straight_action_count"] == 4
        if prediction["error_code"] != "none" or not candidates:
            disposition = "error"
        elif candidates == [4]:
            disposition = "present"
        elif 4 in candidates:
            disposition = "indeterminate"
        else:
            disposition = "absent"
        counts[("truth_four" if truth_is_four else "truth_not_four", disposition)] += 1
    return {
        "truth_four": {
            disposition: counts[("truth_four", disposition)]
            for disposition in ("present", "indeterminate", "absent", "error")
        },
        "truth_not_four": {
            disposition: counts[("truth_not_four", disposition)]
            for disposition in ("present", "indeterminate", "absent", "error")
        },
        "raw_dispositions_are_uncalibrated_and_cannot_authorize_absence": True,
    }


def _measurement(
    *,
    plan: Mapping[str, Any],
    parent: Mapping[str, Any],
    predictions: Mapping[str, Any],
    label_release: Mapping[str, Any],
    truth_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predicted = {
        row["panel_id"]: row
        for task in predictions["task_predictions"]
        for row in task["rows"]
    }
    truth = {row["panel_id"]: row for row in truth_records}
    if set(predicted) != set(truth):
        raise ActionDecompositionFitAblationError(
            "prediction/truth panel inventories differ"
        )
    pairs = tuple(
        (predicted[panel_id], truth[panel_id]) for panel_id in sorted(truth)
    )
    per_task = {
        task_id: {
            "panel_count": len(task_pairs),
            "straight": _axis_metrics(task_pairs, "straight"),
            "arc": _axis_metrics(task_pairs, "arc"),
            "decomposition": _decomposition_metrics(task_pairs),
        }
        for task_id in SELECTED_TASK_IDS
        for task_pairs in [
            tuple(pair for pair in pairs if pair[0]["panel_id"].split("/")[1] == task_id)
        ]
    }
    return _record(
        {
            "schema": RESULT_SCHEMA,
            "parent_fit_outcome_record_digest": parent["record_digest"],
            "plan_record_digest": plan["record_digest"],
            "phase": "fit_ablation",
            "prediction_batch_digest": predictions["record_digest"],
            "label_release_digest": label_release["record_digest"],
            "task_count": 4,
            "panel_count": 56,
            "successful_task_count": sum(
                task["status"] == "success" for task in predictions["task_predictions"]
            ),
            "failed_task_count": sum(
                task["status"] != "success" for task in predictions["task_predictions"]
            ),
            "straight": _axis_metrics(pairs, "straight"),
            "arc": _axis_metrics(pairs, "arc"),
            "decomposition": _decomposition_metrics(pairs),
            "count_four_audit": _count_four_audit(pairs),
            "per_task": per_task,
            "raw_typed_decomposition_tuples_persisted": True,
            "finite_total_sets_projected_only_by_python": True,
            "no_interval_hull_used_for_scoring": True,
            "all_terminal_panels_remain_in_denominator": True,
            "labels_opened_only_after_receipted_prediction_batch_fsync_and_reload": True,
            "model_calls_for_scoring": 0,
            "new_cohort_calibration_heldout_family_or_target_pixels_opened": False,
            "fit_batch_protocol_calibration_transfer_to_query_style_calls": False,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
            "scientific_benchmark": False,
        }
    )


def _forbidden_transport(*_args: Any, **_kwargs: Any) -> CodexStructuredResult:
    raise AssertionError("cold replay attempted a physical model transport")


def run_action_decomposition_fit_ablation(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    action_program_file: str | Path = DEFAULT_ACTION_PROGRAM_FILE,
    plan_file: str | Path = DEFAULT_PLAN,
    parent_outcome_file: str | Path = DEFAULT_PARENT_OUTCOME,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 20,
    executable: str = "codex",
    launcher_sha256: str = DEFAULT_LAUNCHER_SHA256,
    workers: int = 4,
    verbose: bool = False,
    expected_plan_digest: str = PLAN_RECORD_DIGEST,
    expected_parent_digest: str = PARENT_OUTCOME_DIGEST,
    expected_selected_truth_manifest_digest: str = SELECTED_TRUTH_MANIFEST_DIGEST,
    runtime_override: ObjectBongardTurnRuntime | None = None,
    underlying_transport: Callable[..., CodexStructuredResult] = (
        run_codex_named_images_structured
    ),
) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ActionDecompositionFitAblationError("workers must lie in 1..4")
    root = Path(os.path.abspath(os.fspath(output_root))) / "fit_ablation"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ActionDecompositionFitAblationError("output root is unsafe")
    if (root / "label_release.json").exists() and not (root / "predictions.json").exists():
        raise ActionDecompositionFitAblationError(
            "labels exist without a durable prediction batch"
        )

    parent, parent_file_sha256 = _load_parent_outcome(
        parent_outcome_file, expected_digest=expected_parent_digest
    )
    plan = _load_plan(plan_file, expected_digest=expected_plan_digest)
    fit_task_ids = plan.get("cohorts", {}).get("fit", {}).get("task_ids")
    if (
        type(fit_task_ids) is not list
        or any(task_id not in fit_task_ids for task_id in SELECTED_TASK_IDS)
        or len(SELECTED_TASK_IDS) != 4
        or len(set(SELECTED_TASK_IDS)) != 4
        or any(
            "convex" in task_id or "has_four_straight_lines" in task_id
            for task_id in SELECTED_TASK_IDS
        )
    ):
        raise ActionDecompositionFitAblationError(
            "selected tasks violate frozen FIT closure"
        )
    dataset = Path(os.path.abspath(os.fspath(dataset_root)))
    montages, task_inputs = _prepare_inputs(
        dataset, SELECTED_TASK_IDS, plan["record_digest"]
    )
    logical_panel_ids = tuple(
        f"hd/{task_id}/{folder}/{index}.png"
        for task_id in SELECTED_TASK_IDS
        for folder in (1, 0)
        for index in range(7)
    )
    if "sha256:" + canonical_digest(logical_panel_ids) != SELECTED_PANEL_IDS_DIGEST:
        raise ActionDecompositionFitAblationError(
            "selected panel inventory differs"
        )
    authorization, precommit = _authorization_precommit(
        plan=plan,
        parent=parent,
        parent_file_sha256=parent_file_sha256,
        selected_truth_manifest_digest=expected_selected_truth_manifest_digest,
        task_inputs=task_inputs,
        workers=workers,
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        executable=executable,
        launcher_sha256=launcher_sha256,
        runtime_injected=runtime_override is not None,
    )
    _write_once_or_verify(root / "authorization.json", authorization)
    _write_once_or_verify(root / "execution_precommit.json", precommit)
    if runtime_override is None:
        runtime, runtime_evidence = _runtime(
            output_root=root,
            authorization=authorization,
            precommit=precommit,
            model=model,
            reasoning_effort=reasoning_effort,
            minutes=minutes,
            executable=executable,
            launcher_sha256=launcher_sha256,
            verbose=verbose,
        )
    else:
        runtime = runtime_override
        if (
            runtime.model != model
            or runtime.reasoning_effort != reasoning_effort
            or runtime.minutes != minutes
            or runtime.executable != executable
            or runtime.expected_launcher_digest != launcher_sha256
        ):
            raise ActionDecompositionFitAblationError(
                "injected runtime differs from request"
            )
        runtime_evidence = _record(
            {
                "schema": "gkm.bongard-action-decomposition-synthetic-runtime.v1",
                "authorization_digest": authorization["record_digest"],
                "execution_precommit_digest": precommit["record_digest"],
                "runtime_binding": runtime.binding,
                "synthetic_test_only": True,
            }
        )
        _write_once_or_verify(root / "runtime.json", runtime_evidence)

    def execute(task_input: Mapping[str, Any]) -> dict[str, Any]:
        return _observe_task(
            root=root,
            task_input=task_input,
            montages=montages,
            authorization_digest=authorization["record_digest"],
            precommit_digest=precommit["record_digest"],
            runtime=runtime,
            underlying_transport=underlying_transport,
            replay=False,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        task_predictions = tuple(pool.map(execute, task_inputs))
    for prediction in task_predictions:
        _write_once_or_verify(
            root / "task_predictions" / f"{prediction['task_id']}.json", prediction
        )
    predictions = _record(
        {
            "schema": PREDICTION_BATCH_SCHEMA,
            "authorization_digest": authorization["record_digest"],
            "execution_precommit_digest": precommit["record_digest"],
            "runtime_evidence_digest": runtime_evidence["record_digest"],
            "parent_fit_outcome_record_digest": parent["record_digest"],
            "plan_record_digest": plan["record_digest"],
            "phase": "fit_ablation",
            "task_predictions": list(task_predictions),
            "task_prediction_digests": [row["record_digest"] for row in task_predictions],
            "external_journal_terminal_records": [
                row["journal_terminal"] for row in task_predictions
            ],
            "task_count": 4,
            "panel_count": 56,
            "all_external_journals_terminal_and_verified": True,
            "raw_flattened_decomposition_arrays_persisted": True,
            "finite_decomposition_and_total_sets_persisted": True,
            "individual_action_labels_opened_by_process": False,
            "prediction_batch_fsynced_before_label_source_open": True,
            "new_cohort_calibration_heldout_family_or_target_pixels_opened": False,
        }
    )
    _write_once_or_verify(root / "predictions.json", predictions)
    if _read_record(root / "predictions.json") != predictions:
        raise ActionDecompositionFitAblationError(
            "prediction batch did not reload before label source open"
        )

    programs, action_raw = _read_action_programs_after_predictions(
        Path(os.path.abspath(os.fspath(action_program_file))),
        expected_raw_digest=plan["dataset_bindings"]["hd_action_program_raw_sha256"],
    )
    truth = _truth_records(programs, SELECTED_TASK_IDS)
    if (
        "sha256:" + canonical_digest(truth)
        != expected_selected_truth_manifest_digest
    ):
        raise ActionDecompositionFitAblationError(
            "selected FIT action-label manifest differs"
        )
    label_release = _record(
        {
            "schema": LABEL_RELEASE_SCHEMA,
            "parent_fit_outcome_record_digest": parent["record_digest"],
            "plan_record_digest": plan["record_digest"],
            "phase": "fit_ablation",
            "prediction_batch_digest": predictions["record_digest"],
            "action_program_raw_sha256": "sha256:" + hashlib.sha256(action_raw).hexdigest(),
            "selected_truth_manifest_digest": (
                expected_selected_truth_manifest_digest
            ),
            "prediction_batch_reloaded_before_action_program_open": True,
            "labels_visible_to_model": False,
            "labels_opened_by_python_after_predictions": True,
            "new_cohort_calibration_heldout_family_or_target_labels_opened": False,
        }
    )
    _write_once_or_verify(root / "label_release.json", label_release)
    result = _measurement(
        plan=plan,
        parent=parent,
        predictions=predictions,
        label_release=label_release,
        truth_records=truth,
    )
    _write_once_or_verify(root / "result.json", result)

    replay_montages, replay_inputs = _prepare_inputs(
        dataset, SELECTED_TASK_IDS, plan["record_digest"]
    )
    if replay_inputs != task_inputs or replay_montages != montages:
        raise ActionDecompositionFitAblationError(
            "cold-rebuilt three-view pixels differ"
        )
    replayed_predictions = tuple(
        _observe_task(
            root=root,
            task_input=task_input,
            montages=replay_montages,
            authorization_digest=authorization["record_digest"],
            precommit_digest=precommit["record_digest"],
            runtime=runtime,
            underlying_transport=_forbidden_transport,
            replay=True,
        )
        for task_input in replay_inputs
    )
    if replayed_predictions != task_predictions:
        raise ActionDecompositionFitAblationError(
            "cold-replayed predictions differ"
        )
    replayed_result = _measurement(
        plan=plan,
        parent=parent,
        predictions=predictions,
        label_release=label_release,
        truth_records=truth,
    )
    if replayed_result != result:
        raise ActionDecompositionFitAblationError(
            "cold-replayed measurement differs"
        )
    replay = _record(
        {
            "schema": REPLAY_SCHEMA,
            "parent_fit_outcome_record_digest": parent["record_digest"],
            "plan_record_digest": plan["record_digest"],
            "phase": "fit_ablation",
            "authorization_digest": authorization["record_digest"],
            "execution_precommit_digest": precommit["record_digest"],
            "prediction_batch_digest": predictions["record_digest"],
            "label_release_digest": label_release["record_digest"],
            "result_digest": result["record_digest"],
            "journal_count": 4,
            "model_calls_during_replay": 0,
            "all_raw_and_threeview_bytes_rebuilt": True,
            "threeview_rebuild_exact": True,
            "pillow_version_bound_by_algorithm_record": True,
            "external_journal_terminals_exactly_replayed": True,
            "labels_opened_during_model_calls": False,
            "predictions_exactly_replayed": True,
            "measurement_exactly_replayed": True,
            "new_cohort_calibration_heldout_family_or_target_pixels_opened": False,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
        }
    )
    _write_once_or_verify(root / "cold_replay.json", replay)
    return {"result": result, "cold_replay": replay}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--action-program-file", default=str(DEFAULT_ACTION_PROGRAM_FILE))
    parser.add_argument("--plan-file", default=str(DEFAULT_PLAN))
    parser.add_argument("--parent-outcome-file", default=str(DEFAULT_PARENT_OUTCOME))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--minutes", type=int, default=20)
    parser.add_argument("--executable", default="codex")
    parser.add_argument("--launcher-sha256", default=DEFAULT_LAUNCHER_SHA256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    completed = run_action_decomposition_fit_ablation(
        dataset_root=args.dataset_root,
        action_program_file=args.action_program_file,
        plan_file=args.plan_file,
        parent_outcome_file=args.parent_outcome_file,
        output_root=args.output_root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        minutes=args.minutes,
        executable=args.executable,
        launcher_sha256=args.launcher_sha256,
        workers=args.workers,
        verbose=args.verbose,
    )
    print(
        canonical_json(
            {
                "result_digest": completed["result"]["record_digest"],
                "cold_replay_digest": completed["cold_replay"]["record_digest"],
            }
        ).decode()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
