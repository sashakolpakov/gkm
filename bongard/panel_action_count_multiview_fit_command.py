"""Run the FIT-only multiview action-boundary observer.

This command is deliberately incapable of opening the calibration, held-out,
same-family, or target cohorts.  Each already-opened FIT task is one exactly-once
turn over fourteen neutral four-quadrant images.  The model returns only typed
straight/arc count hypotheses: one best count, up to three non-contiguous
alternatives, and a conservative interval.  The receipted prediction batch is
durably written and reloaded before this process opens the action-program file.

Python scores top-1 accuracy, finite-set coverage, interval coverage, and panel
and task-max residual curves.  A cold replay is required to reproduce every
journal, prediction, and metric without a physical model call.
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
from bongard.panel_action_count_multiview_adapter import (
    build_action_count_multiview,
    multiview_algorithm_record,
    panel_action_count_multiview_adapter_source_digest,
)
from bongard.panel_action_count_phase_command import (
    DEFAULT_ACTION_PROGRAM_FILE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_LAUNCHER_SHA256,
    DEFAULT_MODEL,
    DEFAULT_PLAN,
    DEFAULT_REASONING_EFFORT,
    PLAN_RECORD_DIGEST,
    _load_plan,
    _panel_path,
    _presentation,
    _profile,
    _read_action_programs_after_predictions,
    _read_png,
    _truth_records,
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


FIT_PHASE = "fit"
AUTHORIZATION_SCHEMA = "gkm.bongard-action-count-multiview-fit-authorization.v1"
PRECOMMIT_SCHEMA = "gkm.bongard-action-count-multiview-fit-precommit.v1"
TASK_PREDICTION_SCHEMA = "gkm.bongard-action-count-multiview-task-prediction.v1"
PREDICTION_BATCH_SCHEMA = "gkm.bongard-action-count-multiview-prediction-batch.v1"
LABEL_RELEASE_SCHEMA = "gkm.bongard-action-count-multiview-label-release.v1"
RESULT_SCHEMA = "gkm.bongard-action-count-multiview-fit-result.v1"
REPLAY_SCHEMA = "gkm.bongard-action-count-multiview-fit-cold-replay.v1"
VIEW_NAMES = tuple(f"view_{index:02d}.png" for index in range(14))
AXES = ("straight", "arc")
ALTERNATIVE_SLOTS = 3
UNUSED_COUNT_SENTINEL = 10
ERROR_VALUES = ("none", "unreadable")
DEFAULT_OUTPUT_ROOT = Path(
    "downloads/ShapeBongard_V2_full/panel_action_count_multiview_fit_20260810_v1"
)


class ActionCountMultiviewFitError(RuntimeError):
    """The FIT authority, view, journal, prediction, label, or replay differs."""


def panel_action_count_multiview_fit_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def multiview_action_count_output_schema() -> dict[str, object]:
    properties: dict[str, object] = {}
    for name in VIEW_NAMES:
        stem = name.removesuffix(".png")
        for axis in AXES:
            properties[f"{stem}_{axis}_best_count"] = {
                "type": "integer",
            }
            for slot in range(1, ALTERNATIVE_SLOTS + 1):
                properties[f"{stem}_{axis}_alternative_{slot}"] = {
                    "type": "integer",
                }
            properties[f"{stem}_{axis}_count_lower"] = {
                "type": "integer",
            }
            properties[f"{stem}_{axis}_count_upper"] = {
                "type": "integer",
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


def multiview_action_count_prompt() -> str:
    names = ", ".join(VIEW_NAMES)
    return (
        "Inspect fourteen independent drawings named " + names + ". Their order "
        "is randomized. They are not groups, examples of a rule, classes, support "
        "or query roles. Never infer a shared task concept. Each named image is one "
        "drawing shown four deterministic ways: top-left is the alpha-composited source "
        "at its original canvas scale; "
        "top-right is a square crop around all ink; bottom-left is a binary ink-edge "
        "view; bottom-right is a blurred coarse carrier-density view. Edge and density "
        "views are advisory and can fragment or merge marker chains; inspect all four. "
        "For each drawing estimate exactly two axes. straight_count is the number of "
        "underlying straight carrier actions. arc_count is the number of underlying "
        "curved carrier actions. Count each carrier action once even when rendered as "
        "zigzags, dots, circles, squares, triangles, or texture changes. Decoration "
        "does not create an action. Adjacent nearly-collinear actions remain separate "
        "only when a visible construction boundary supports that split. Additional "
        "arcs do not change straight_count. Counts lie in 0..9. For each axis return "
        "best_count, then up to three genuinely plausible non-contiguous alternatives. "
        "Use integer 10 as the unused sentinel; unused alternative slots must trail "
        "used slots. Alternatives must be distinct from best_count and each other and "
        "must be ascending. Also return an inclusive count_lower/count_upper interval "
        "containing best_count and every alternative; widen it for additional unresolved "
        "counts. For a readable drawing error_code is none and best_count is 0..9. If "
        "unreadable, return error_code unreadable, best_count and all alternatives as "
        "10, and interval 0..9 for both axes. Return no prose, geometry trace, closure "
        "or convexity judgment, candidate predicate, formula, polarity, threshold, task "
        "label, or class decision. Dataset IDs, construction programs, truth counts, "
        "phases, and side labels are unavailable."
    )


def _prepare_multiview_inputs(
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
            montage, view_record = build_action_count_multiview(raw)
            if (
                view_record["source_png_sha256"] != hashlib.sha256(raw).hexdigest()
                or view_record["montage_png_sha256"]
                != hashlib.sha256(montage).hexdigest()
            ):
                raise ActionCountMultiviewFitError("multiview byte binding differs")
            montages[panel_id] = montage
            rows.append(
                {
                    "model_visible_name": name,
                    "panel_id": panel_id,
                    "source_png_sha256": view_record["source_png_sha256"],
                    "source_png_size_bytes": len(raw),
                    "multiview_record": view_record,
                }
            )
        task_inputs.append({"task_id": task_id, "presentation": rows})
    return montages, tuple(task_inputs)


def _authorization_precommit(
    *,
    plan: Mapping[str, Any],
    task_inputs: Sequence[Mapping[str, Any]],
    workers: int,
    model: str,
    reasoning_effort: str,
    minutes: int,
    executable: str,
    launcher_sha256: str,
    runtime_injected: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = multiview_action_count_prompt()
    schema = multiview_action_count_output_schema()
    algorithm = multiview_algorithm_record()
    authorization = _record(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "command_source_digest": panel_action_count_multiview_fit_source_digest(),
            "adapter_source_digest": panel_action_count_multiview_adapter_source_digest(),
            "custody_source_digest": panel_probe_custody_source_digest(),
            "probe_transport_source_digest": panel_probe_transport_source_digest(),
            "multiview_algorithm": algorithm,
            "plan_record_digest": plan["record_digest"],
            "phase": FIT_PHASE,
            "fit_only": True,
            "task_inputs": list(task_inputs),
            "task_count": len(task_inputs),
            "panel_count": 14 * len(task_inputs),
            "model_visible_names": list(VIEW_NAMES),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "output_schema_digest": "sha256:" + canonical_digest(schema),
            "output_contract": {
                "best_count": "integer_0_through_9",
                "alternative_slots": ALTERNATIVE_SLOTS,
                "unused_alternative_sentinel": UNUSED_COUNT_SENTINEL,
                "fallback_interval": "inclusive_integer_0_through_9",
            },
            "individual_action_labels_opened_by_process_before_prediction": False,
            "fit_labels_historically_open_for_protocol_tuning": True,
            "fit_batch_protocol_calibration_transfer_to_query_style_calls": False,
            "calibration_or_heldout_pixels_opened": False,
            "target_semantics_pixels_or_labels_opened": False,
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
            "plan_record_digest": plan["record_digest"],
            "phase": FIT_PHASE,
            "task_ids": [row["task_id"] for row in task_inputs],
            "physical_call_plan": {
                "named_multiview_batch_calls_per_task": 1,
                "task_count": len(task_inputs),
                "maximum_physical_calls": len(task_inputs),
            },
            "one_call_contains_exactly_fourteen_neutral_multiview_images": True,
            "typed_output_axes": list(AXES),
            "noncontiguous_alternatives_preserved_before_interval_scoring": True,
            "formula_or_predicate_synthesis_present": False,
            "predictions_must_be_fsynced_and_reloaded_before_action_program_open": True,
            "labels_model_visible": False,
            "exactly_once_external_journals_required": True,
            "cold_replay_model_calls": 0,
            "calibration_heldout_family_and_target_calls": 0,
            "fit_batch_is_frontend_protocol_tuning_only": True,
            "workers": workers,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
        }
    )
    return authorization, precommit


def _parse_axis(payload: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    best = payload[f"{prefix}_best_count"]
    alternatives = tuple(
        payload[f"{prefix}_alternative_{slot}"]
        for slot in range(1, ALTERNATIVE_SLOTS + 1)
    )
    lower = payload[f"{prefix}_count_lower"]
    upper = payload[f"{prefix}_count_upper"]
    if (
        type(best) is not int
        or type(lower) is not int
        or type(upper) is not int
        or any(type(value) is not int for value in alternatives)
        or not 0 <= lower <= upper <= 9
        or not 0 <= best <= 10
        or any(not 0 <= value <= 10 for value in alternatives)
    ):
        raise ActionCountMultiviewFitError("typed count fields differ")
    return {
        "best": best,
        "alternatives": alternatives,
        "lower": lower,
        "upper": upper,
    }


def _validate_readable_axis(axis: Mapping[str, Any]) -> tuple[int, ...]:
    best = axis["best"]
    alternatives = axis["alternatives"]
    if best == UNUSED_COUNT_SENTINEL:
        raise ActionCountMultiviewFitError("readable axis has no best count")
    used = tuple(value for value in alternatives if value != UNUSED_COUNT_SENTINEL)
    if (
        alternatives != used + (UNUSED_COUNT_SENTINEL,) * (ALTERNATIVE_SLOTS - len(used))
        or used != tuple(sorted(used))
        or len(set(used)) != len(used)
        or best in used
    ):
        raise ActionCountMultiviewFitError("alternative count set differs")
    candidates = (best,) + used
    if any(not axis["lower"] <= value <= axis["upper"] for value in candidates):
        raise ActionCountMultiviewFitError("fallback interval excludes a candidate")
    return candidates


def _parse_payload(
    payload: Mapping[str, Any],
    presentation: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    schema = multiview_action_count_output_schema()
    if set(payload) != set(schema["properties"]):
        raise ActionCountMultiviewFitError("typed payload fields differ")
    rows: list[dict[str, Any]] = []
    for item in presentation:
        stem = item["model_visible_name"].removesuffix(".png")
        error = payload[f"{stem}_error_code"]
        axes = {axis: _parse_axis(payload, f"{stem}_{axis}") for axis in AXES}
        if error not in ERROR_VALUES:
            raise ActionCountMultiviewFitError("typed error code differs")
        if error == "unreadable":
            if any(
                axis["best"] != UNUSED_COUNT_SENTINEL
                or axis["alternatives"] != (UNUSED_COUNT_SENTINEL,) * ALTERNATIVE_SLOTS
                or (axis["lower"], axis["upper"]) != (0, 9)
                for axis in axes.values()
            ):
                raise ActionCountMultiviewFitError("unreadable sentinel payload differs")
            candidates = {axis: tuple() for axis in AXES}
        else:
            candidates = {
                axis: _validate_readable_axis(values)
                for axis, values in axes.items()
            }
        row: dict[str, Any] = {
            "panel_id": item["panel_id"],
            "model_visible_name": item["model_visible_name"],
            "source_png_sha256": item["source_png_sha256"],
            "multiview_record_digest": item["multiview_record"]["record_digest"],
            "montage_png_sha256": item["multiview_record"]["montage_png_sha256"],
        }
        for axis in AXES:
            row[f"{axis}_best_count"] = (
                None if error != "none" else axes[axis]["best"]
            )
            row[f"{axis}_alternative_counts"] = (
                [] if error != "none" else list(candidates[axis][1:])
            )
            row[f"{axis}_candidate_counts"] = (
                [] if error != "none" else list(candidates[axis])
            )
            row[f"{axis}_count_lower"] = (
                None if error != "none" else axes[axis]["lower"]
            )
            row[f"{axis}_count_upper"] = (
                None if error != "none" else axes[axis]["upper"]
            )
        row["error_code"] = error
        rows.append(row)
    return tuple(rows)


def _error_rows(
    presentation: Sequence[Mapping[str, Any]], error_code: str
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for item in presentation:
        row: dict[str, Any] = {
            "panel_id": item["panel_id"],
            "model_visible_name": item["model_visible_name"],
            "source_png_sha256": item["source_png_sha256"],
            "multiview_record_digest": item["multiview_record"]["record_digest"],
            "montage_png_sha256": item["multiview_record"]["montage_png_sha256"],
        }
        for axis in AXES:
            row[f"{axis}_best_count"] = None
            row[f"{axis}_alternative_counts"] = []
            row[f"{axis}_candidate_counts"] = []
            row[f"{axis}_count_lower"] = None
            row[f"{axis}_count_upper"] = None
        row["error_code"] = error_code
        rows.append(row)
    return tuple(rows)


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
        raise ActionCountMultiviewFitError("cold replay found no terminal journal")
    journal = ObjectBongardNamedImageTurnJournalTransport(
        journal_root,
        authorization_digest=authorization_digest,
        execution_precommit_digest=precommit_digest,
        task_id=task_id,
        turn_kind="action_count_multiview_batch",
        expected_prompt=multiview_action_count_prompt(),
        expected_images=images,
        expected_output_schema=multiview_action_count_output_schema(),
        runtime=runtime,
        underlying_transport=underlying_transport,
    )
    payload: Mapping[str, Any] | None = None
    receipt_digest: str | None = None
    transport_failed = False
    try:
        payload, receipt = _call(
            images,
            prompt=multiview_action_count_prompt(),
            schema=multiview_action_count_output_schema(),
            journal=journal,
            runtime=runtime,
        )
        receipt_digest = receipt.receipt_digest
    except ObjectBongardTurnCallFailed:
        transport_failed = True
    terminal = journal.verify()
    if transport_failed:
        if terminal.terminal_status != "failure":
            raise ActionCountMultiviewFitError("failed call has no failure terminal")
        rows = _error_rows(presentation, "transport_error")
        status = "error"
        error_code = "transport_error"
    else:
        if terminal.terminal_status != "success" or payload is None:
            raise ActionCountMultiviewFitError("successful call has no success terminal")
        try:
            rows = _parse_payload(payload, presentation)
        except ActionCountMultiviewFitError:
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
            "noncontiguous_alternatives_preserved": True,
        }
    )


def _residual_curve(values: Sequence[int]) -> dict[str, Any]:
    if any(type(value) is not int or not 0 <= value <= 9 for value in values):
        raise ActionCountMultiviewFitError("residual domain differs")
    denominator = len(values)
    histogram = Counter(values)
    return {
        "denominator": denominator,
        "residual_histogram": {
            str(radius): histogram.get(radius, 0) for radius in range(10)
        },
        "coverage_by_expansion_radius": [
            {
                "radius": radius,
                "covered_count": sum(value <= radius for value in values),
                "coverage_rate": [sum(value <= radius for value in values), denominator],
            }
            for radius in range(10)
        ],
        "minimum_zero_omission_radius": max(values) if values else None,
    }


def _axis_residuals(
    prediction: Mapping[str, Any], truth: Mapping[str, Any], axis: str
) -> tuple[int, int, int] | None:
    if prediction["error_code"] != "none":
        return None
    target = truth[f"{axis}_action_count"]
    best = prediction[f"{axis}_best_count"]
    candidates = prediction[f"{axis}_candidate_counts"]
    lower = prediction[f"{axis}_count_lower"]
    upper = prediction[f"{axis}_count_upper"]
    if (
        type(best) is not int
        or type(candidates) is not list
        or not candidates
        or type(lower) is not int
        or type(upper) is not int
    ):
        return None
    top1 = abs(target - best)
    finite_set = min(abs(target - value) for value in candidates)
    interval = lower - target if target < lower else target - upper if target > upper else 0
    return top1, finite_set, interval


def _axis_metrics(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], axis: str
) -> dict[str, Any]:
    top1_exact = set_coverage = interval_coverage = 0
    interval_width = candidate_cardinality = valid = errors = 0
    top1_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_set_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    panel_residuals: dict[str, list[int]] = {
        "top1": [],
        "finite_candidate_set": [],
        "fallback_interval": [],
    }
    task_residuals: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {name: [] for name in panel_residuals}
    )
    for prediction, truth in rows:
        target = truth[f"{axis}_action_count"]
        residuals = _axis_residuals(prediction, truth, axis)
        if residuals is None:
            errors += 1
            top1_confusion[str(target)]["error"] += 1
            candidate_set_confusion[str(target)]["error"] += 1
            continue
        valid += 1
        best = prediction[f"{axis}_best_count"]
        candidates = prediction[f"{axis}_candidate_counts"]
        lower = prediction[f"{axis}_count_lower"]
        upper = prediction[f"{axis}_count_upper"]
        top1_exact += int(best == target)
        set_coverage += int(target in candidates)
        interval_coverage += int(lower <= target <= upper)
        interval_width += upper - lower
        candidate_cardinality += len(candidates)
        top1_confusion[str(target)][str(best)] += 1
        candidate_set_confusion[str(target)][",".join(str(v) for v in candidates)] += 1
        task_id = prediction["panel_id"].split("/")[1]
        for name, residual in zip(panel_residuals, residuals, strict=True):
            panel_residuals[name].append(residual)
            task_residuals[task_id][name].append(residual)
    denominator = len(rows)
    task_max: dict[str, list[int]] = {name: [] for name in panel_residuals}
    task_rows: list[dict[str, Any]] = []
    for task_id in sorted(task_residuals):
        values = task_residuals[task_id]
        if any(len(values[name]) != 14 for name in values):
            continue
        maxima = {name: max(values[name]) for name in values}
        for name, value in maxima.items():
            task_max[name].append(value)
        task_rows.append({"task_id": task_id, **maxima})
    return {
        "denominator": denominator,
        "valid_prediction_count": valid,
        "error_count": errors,
        "top1_exact_count": top1_exact,
        "top1_exact_rate": [top1_exact, denominator],
        "finite_candidate_set_coverage_count": set_coverage,
        "finite_candidate_set_coverage_rate": [set_coverage, denominator],
        "fallback_interval_coverage_count": interval_coverage,
        "fallback_interval_coverage_rate": [interval_coverage, denominator],
        "candidate_cardinality_sum": candidate_cardinality,
        "mean_candidate_cardinality": [candidate_cardinality, valid],
        "interval_width_sum": interval_width,
        "mean_interval_width": [interval_width, valid],
        "top1_confusion": {
            truth: dict(sorted(counter.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 99))
            for truth, counter in sorted(top1_confusion.items(), key=lambda item: int(item[0]))
        },
        "finite_candidate_set_confusion": {
            truth: dict(sorted(counter.items()))
            for truth, counter in sorted(candidate_set_confusion.items(), key=lambda item: int(item[0]))
        },
        "panel_residual_curves": {
            name: _residual_curve(values) for name, values in panel_residuals.items()
        },
        "task_max_residuals": task_rows,
        "task_max_residual_curves": {
            name: _residual_curve(values) for name, values in task_max.items()
        },
        "task_max_curve_excludes_tasks_with_any_error": True,
    }


def _measurement(
    *,
    plan: Mapping[str, Any],
    predictions: Mapping[str, Any],
    label_release: Mapping[str, Any],
    truth_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prediction_rows = {
        row["panel_id"]: row
        for task in predictions["task_predictions"]
        for row in task["rows"]
    }
    truth_by_id = {row["panel_id"]: row for row in truth_records}
    if set(prediction_rows) != set(truth_by_id):
        raise ActionCountMultiviewFitError("prediction/truth panel inventories differ")
    paired = tuple(
        (prediction_rows[panel_id], truth_by_id[panel_id])
        for panel_id in sorted(truth_by_id)
    )
    strata_rows: dict[str, dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]]] = {
        "straight_action_count": defaultdict(list),
        "line_decoration_profile": defaultdict(list),
        "arc_presence": defaultdict(list),
        "stroke_style_presence": defaultdict(list),
    }
    for pair in paired:
        _prediction, truth = pair
        line_profile, arc_presence, styles = _profile(truth)
        strata_rows["straight_action_count"][str(truth["straight_action_count"])].append(pair)
        strata_rows["line_decoration_profile"][line_profile].append(pair)
        strata_rows["arc_presence"][arc_presence].append(pair)
        for style in styles:
            strata_rows["stroke_style_presence"][style].append(pair)
    strata: dict[str, Any] = {}
    for dimension, categories in strata_rows.items():
        strata[dimension] = {
            category: {
                "straight": _axis_metrics(rows, "straight"),
                "arc": _axis_metrics(rows, "arc"),
            }
            for category, rows in sorted(categories.items())
        }
    return _record(
        {
            "schema": RESULT_SCHEMA,
            "plan_record_digest": plan["record_digest"],
            "phase": FIT_PHASE,
            "prediction_batch_digest": predictions["record_digest"],
            "label_release_digest": label_release["record_digest"],
            "task_count": len(predictions["task_predictions"]),
            "panel_count": len(paired),
            "successful_task_count": sum(
                task["status"] == "success" for task in predictions["task_predictions"]
            ),
            "failed_task_count": sum(
                task["status"] != "success" for task in predictions["task_predictions"]
            ),
            "straight": _axis_metrics(paired, "straight"),
            "arc": _axis_metrics(paired, "arc"),
            "strata": strata,
            "raw_top1_finite_set_and_fallback_interval_all_persisted": True,
            "noncontiguous_alternatives_scored_without_interval_collapse": True,
            "all_terminal_panels_remain_in_denominator": True,
            "labels_opened_only_after_receipted_prediction_batch_fsync_and_reload": True,
            "model_calls_for_scoring": 0,
            "calibration_or_heldout_pixels_opened": False,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "lean_removable": True,
            "scientific_benchmark": False,
            "fit_batch_protocol_calibration_transfer_to_target": False,
        }
    )


def _forbidden_transport(*_args: Any, **_kwargs: Any) -> CodexStructuredResult:
    raise AssertionError("cold replay attempted a physical model transport")


def run_multiview_action_count_fit(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    action_program_file: str | Path = DEFAULT_ACTION_PROGRAM_FILE,
    plan_file: str | Path = DEFAULT_PLAN,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = 20,
    executable: str = "codex",
    launcher_sha256: str = DEFAULT_LAUNCHER_SHA256,
    workers: int = 4,
    verbose: bool = False,
    expected_plan_digest: str = PLAN_RECORD_DIGEST,
    runtime_override: ObjectBongardTurnRuntime | None = None,
    underlying_transport: Callable[..., CodexStructuredResult] = (
        run_codex_named_images_structured
    ),
) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ActionCountMultiviewFitError("workers must lie in 1..4")
    root = Path(os.path.abspath(os.fspath(output_root))) / FIT_PHASE
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ActionCountMultiviewFitError("output root is unsafe")
    if (root / "label_release.json").exists() and not (root / "predictions.json").exists():
        raise ActionCountMultiviewFitError("labels exist without a durable prediction batch")

    plan = _load_plan(plan_file, expected_digest=expected_plan_digest)
    cohort = plan["cohorts"].get(FIT_PHASE)
    if type(cohort) is not dict or type(cohort.get("task_ids")) is not list:
        raise ActionCountMultiviewFitError("FIT cohort differs")
    task_ids = tuple(cohort["task_ids"])
    if (
        (runtime_override is None and len(task_ids) != 20)
        or len(task_ids) != len(set(task_ids))
        or any(
            "convex" in task_id or "has_four_straight_lines" in task_id
            for task_id in task_ids
        )
    ):
        raise ActionCountMultiviewFitError("FIT cohort violates frozen closure")

    dataset = Path(os.path.abspath(os.fspath(dataset_root)))
    montages, task_inputs = _prepare_multiview_inputs(
        dataset, task_ids, plan["record_digest"]
    )
    authorization, precommit = _authorization_precommit(
        plan=plan,
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
            raise ActionCountMultiviewFitError("injected runtime differs from request")
        runtime_evidence = _record(
            {
                "schema": "gkm.bongard-action-count-multiview-synthetic-runtime.v1",
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
            "plan_record_digest": plan["record_digest"],
            "phase": FIT_PHASE,
            "task_predictions": list(task_predictions),
            "task_prediction_digests": [row["record_digest"] for row in task_predictions],
            "external_journal_terminal_records": [
                row["journal_terminal"] for row in task_predictions
            ],
            "task_count": len(task_predictions),
            "panel_count": 14 * len(task_predictions),
            "all_external_journals_terminal_and_verified": True,
            "raw_top1_finite_set_and_fallback_interval_persisted": True,
            "individual_action_labels_opened_by_process": False,
            "prediction_batch_fsynced_before_label_source_open": True,
            "calibration_or_heldout_pixels_opened": False,
        }
    )
    _write_once_or_verify(root / "predictions.json", predictions)
    persisted_predictions = _read_record(root / "predictions.json")
    if persisted_predictions != predictions:
        raise ActionCountMultiviewFitError(
            "prediction batch did not reload before label source open"
        )

    programs, action_raw = _read_action_programs_after_predictions(
        Path(os.path.abspath(os.fspath(action_program_file))),
        expected_raw_digest=plan["dataset_bindings"]["hd_action_program_raw_sha256"],
    )
    truth = _truth_records(programs, task_ids)
    if "sha256:" + canonical_digest(truth) != cohort["action_label_manifest_digest"]:
        raise ActionCountMultiviewFitError("FIT action-label manifest differs")
    label_release = _record(
        {
            "schema": LABEL_RELEASE_SCHEMA,
            "plan_record_digest": plan["record_digest"],
            "phase": FIT_PHASE,
            "prediction_batch_digest": predictions["record_digest"],
            "action_program_raw_sha256": "sha256:" + hashlib.sha256(action_raw).hexdigest(),
            "action_label_manifest_digest": cohort["action_label_manifest_digest"],
            "prediction_batch_reloaded_before_action_program_open": True,
            "labels_visible_to_model": False,
            "labels_opened_by_python_after_predictions": True,
            "calibration_or_heldout_labels_opened": False,
        }
    )
    _write_once_or_verify(root / "label_release.json", label_release)
    result = _measurement(
        plan=plan,
        predictions=predictions,
        label_release=label_release,
        truth_records=truth,
    )
    _write_once_or_verify(root / "result.json", result)

    # Re-read every FIT source byte and recompute every derived view.  Equality
    # binds cold replay to both the exact raw panels and the exact Pillow-backed
    # transform implementation/environment, not merely to stored view digests.
    replay_montages, replay_task_inputs = _prepare_multiview_inputs(
        dataset, task_ids, plan["record_digest"]
    )
    if replay_task_inputs != task_inputs or replay_montages != montages:
        raise ActionCountMultiviewFitError("cold-rebuilt multiview pixels differ")
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
        for task_input in replay_task_inputs
    )
    if replayed_predictions != task_predictions:
        raise ActionCountMultiviewFitError("cold-replayed predictions differ")
    replayed_result = _measurement(
        plan=plan,
        predictions=predictions,
        label_release=label_release,
        truth_records=truth,
    )
    if replayed_result != result:
        raise ActionCountMultiviewFitError("cold-replayed measurement differs")
    replay = _record(
        {
            "schema": REPLAY_SCHEMA,
            "plan_record_digest": plan["record_digest"],
            "phase": FIT_PHASE,
            "authorization_digest": authorization["record_digest"],
            "execution_precommit_digest": precommit["record_digest"],
            "prediction_batch_digest": predictions["record_digest"],
            "label_release_digest": label_release["record_digest"],
            "result_digest": result["record_digest"],
            "journal_count": len(task_inputs),
            "model_calls_during_replay": 0,
            "all_raw_and_multiview_bytes_rebuilt": True,
            "multiview_rebuild_exact": True,
            "pillow_version_bound_by_algorithm_record": True,
            "external_journal_terminals_exactly_replayed": True,
            "labels_opened_during_model_calls": False,
            "predictions_exactly_replayed": True,
            "measurement_exactly_replayed": True,
            "calibration_or_heldout_pixels_opened": False,
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
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--minutes", type=int, default=20)
    parser.add_argument("--executable", default="codex")
    parser.add_argument("--launcher-sha256", default=DEFAULT_LAUNCHER_SHA256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    completed = run_multiview_action_count_fit(
        dataset_root=args.dataset_root,
        action_program_file=args.action_program_file,
        plan_file=args.plan_file,
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
