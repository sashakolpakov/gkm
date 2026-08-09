"""Run one preregistered decoration-aware action-count measurement phase.

Each selected HD TRAIN task is one exactly-once named-image turn containing its
fourteen drawings in a deterministic, side-obscuring order.  The model returns
only typed straight/arc count intervals and an error status for each neutral
view.  The complete receipted prediction batch is durably written before this
process opens the action-program file.  Python then computes exact-count,
interval-coverage, width, confusion, and decoration-stratified measurements.
The journals and metrics are cold-replayed with a transport that must never run.
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
import stat
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnCallFailed,
    ObjectBongardTurnRuntime,
)
from bongard.panel_probe_custody import (
    DEFAULT_PROBE_LAUNCHER_SHA256 as DEFAULT_LAUNCHER_SHA256,
    DEFAULT_PROBE_MODEL as DEFAULT_MODEL,
    DEFAULT_PROBE_REASONING_EFFORT as DEFAULT_REASONING_EFFORT,
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


PLAN_RECORD_DIGEST = (
    "sha256:476ff0d602d43ddc6e4c8c6a964939a01c8471936eece71d0caba8a35bda396a"
)
PLAN_SCHEMA = "gkm.bongard-decoration-aware-action-count-calibration-preregistration.v1"
AUTHORIZATION_SCHEMA = "gkm.bongard-action-count-phase-authorization.v1"
PRECOMMIT_SCHEMA = "gkm.bongard-action-count-phase-precommit.v1"
TASK_PREDICTION_SCHEMA = "gkm.bongard-action-count-task-prediction.v1"
PREDICTION_BATCH_SCHEMA = "gkm.bongard-action-count-prediction-batch.v1"
LABEL_RELEASE_SCHEMA = "gkm.bongard-action-count-label-release.v1"
RESULT_SCHEMA = "gkm.bongard-action-count-measurement-result.v1"
REPLAY_SCHEMA = "gkm.bongard-action-count-phase-cold-replay.v1"
PHASES = ("fit", "calibration", "heldout")
STYLES = ("circle", "normal", "square", "triangle", "zigzag")
VIEW_NAMES = tuple(f"view_{index:02d}.png" for index in range(14))
COUNT_FIELDS = (
    "straight_action_count_lower",
    "straight_action_count_upper",
    "arc_action_count_lower",
    "arc_action_count_upper",
)
ERROR_VALUES = ("none", "unreadable")
DEFAULT_PLAN = Path(__file__).resolve().parent / "data/panel_action_count_calibration_preregistration_20260809_v1.json"
DEFAULT_DATASET_ROOT = Path("downloads/ShapeBongard_V2_full/ShapeBongard_V2")
DEFAULT_ACTION_PROGRAM_FILE = DEFAULT_DATASET_ROOT / "hd/hd_action_programs.json"
DEFAULT_OUTPUT_ROOT = Path("downloads/ShapeBongard_V2_full/panel_action_count_measurement_20260809_v1")
MAX_PNG_BYTES = 8 * 1024 * 1024


class ActionCountPhaseError(RuntimeError):
    """The plan, pixels, journal, labels, measurement, or replay differs."""


def panel_action_count_phase_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _load_plan(path: str | Path, *, expected_digest: str) -> dict[str, Any]:
    source = Path(os.path.abspath(os.fspath(path)))
    if source.is_symlink() or not source.is_file():
        raise ActionCountPhaseError("preregistration path is unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ActionCountPhaseError("preregistration is malformed") from exc
    if type(value) is not dict or raw != canonical_json(value) + b"\n":
        raise ActionCountPhaseError("preregistration is not canonical")
    body = dict(value)
    digest = body.pop("record_digest", None)
    if (
        value.get("schema") != PLAN_SCHEMA
        or digest != "sha256:" + canonical_digest(body)
        or digest != expected_digest
        or value.get("current_state", {}).get("selected_panel_pixels_read") is not False
        or value.get("current_state", {}).get("model_calls_made") != 0
        or value.get("calibration_authority", {}).get("python_is_canonical_authority") is not True
        or value.get("calibration_authority", {}).get("lean_required") is not False
    ):
        raise ActionCountPhaseError("preregistration authority differs")
    return value


def action_count_batch_output_schema() -> dict[str, object]:
    properties: dict[str, object] = {}
    for name in VIEW_NAMES:
        stem = name.removesuffix(".png")
        for field in COUNT_FIELDS:
            properties[f"{stem}_{field}"] = {
                "type": "integer",
                "enum": list(range(10)),
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


def action_count_batch_prompt() -> str:
    names = ", ".join(VIEW_NAMES)
    return (
        "Inspect fourteen independent drawings named " + names + ". Their order is "
        "randomized. They are not groups, examples of a rule, positive/negative "
        "classes, or support/query roles. Never infer a shared task concept. For each "
        "drawing report an inclusive integer interval for exactly two typed axes. "
        "straight_action_count is the number of underlying straight carrier actions. "
        "arc_action_count is the number of underlying curved carrier actions. Count "
        "each carrier action once even when it is rendered as zigzags, dots, circles, "
        "squares, triangles, or texture changes. Decoration does not create extra "
        "carrier actions. Additional arcs do not change the straight count. All bounds "
        "lie in 0..9 and lower must not exceed upper. Widen intervals for unresolved "
        "junctions or boundaries. For each filename stem return "
        "straight_action_count_lower, straight_action_count_upper, "
        "arc_action_count_lower, arc_action_count_upper, and error_code. error_code is "
        "none unless the drawing is unreadable; when unreadable return both intervals "
        "as 0..9. Return no prose, closure or convexity judgment, candidate predicate, "
        "formula, polarity, threshold, task label, or class decision. Dataset IDs, "
        "construction programs, truth counts, phases, and side labels are unavailable."
    )


def _logical_panel_ids(task_id: str) -> tuple[str, ...]:
    return tuple(
        f"hd/{task_id}/{folder}/{index}.png"
        for folder in (1, 0)
        for index in range(7)
    )


def _presentation(task_id: str, plan_digest: str) -> tuple[tuple[str, str], ...]:
    panel_ids = _logical_panel_ids(task_id)
    ordered = sorted(
        panel_ids,
        key=lambda panel_id: (
            hashlib.sha256(
                (plan_digest + "\0" + task_id + "\0" + panel_id).encode()
            ).hexdigest(),
            panel_id,
        ),
    )
    return tuple(zip(VIEW_NAMES, ordered, strict=True))


def _panel_path(dataset_root: Path, panel_id: str) -> Path:
    parts = panel_id.split("/")
    if len(parts) != 4 or parts[0] != "hd" or parts[3] != f"{int(parts[3].removesuffix('.png'))}.png":
        raise ActionCountPhaseError("panel identity grammar differs")
    return dataset_root / parts[0] / "images" / parts[1] / parts[2] / parts[3]


def _read_png(path: Path) -> bytes:
    if path.is_symlink():
        raise ActionCountPhaseError("panel path is a symlink")
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_PNG_BYTES:
            raise ActionCountPhaseError("panel file is not bounded")
        data = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise ActionCountPhaseError("panel file is unavailable") from exc
    if (
        data[:8] != b"\x89PNG\r\n\x1a\n"
        or len(data) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ActionCountPhaseError("panel bytes changed or are not PNG")
    return data


def _prepare_pixels(
    dataset_root: Path,
    task_ids: Sequence[str],
    plan_digest: str,
) -> tuple[dict[str, bytes], tuple[dict[str, Any], ...]]:
    pixels: dict[str, bytes] = {}
    task_inputs: list[dict[str, Any]] = []
    for task_id in task_ids:
        presentation = _presentation(task_id, plan_digest)
        rows: list[dict[str, Any]] = []
        for name, panel_id in presentation:
            payload = _read_png(_panel_path(dataset_root, panel_id))
            pixels[panel_id] = payload
            rows.append(
                {
                    "model_visible_name": name,
                    "panel_id": panel_id,
                    "png_sha256": hashlib.sha256(payload).hexdigest(),
                    "png_size_bytes": len(payload),
                }
            )
        task_inputs.append({"task_id": task_id, "presentation": rows})
    return pixels, tuple(task_inputs)


def _authorization_precommit(
    *,
    plan: Mapping[str, Any],
    phase: str,
    task_inputs: Sequence[Mapping[str, Any]],
    workers: int,
    model: str,
    reasoning_effort: str,
    minutes: int,
    executable: str,
    launcher_sha256: str,
    runtime_injected: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = action_count_batch_prompt()
    schema = action_count_batch_output_schema()
    authorization = _record(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "command_source_digest": panel_action_count_phase_source_digest(),
            "probe_custody_source_digest": panel_probe_custody_source_digest(),
            "probe_transport_source_digest": panel_probe_transport_source_digest(),
            "plan_record_digest": plan["record_digest"],
            "phase": phase,
            "task_inputs": list(task_inputs),
            "task_count": len(task_inputs),
            "panel_count": 14 * len(task_inputs),
            "model_visible_names": list(VIEW_NAMES),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "output_schema_digest": "sha256:" + canonical_digest(schema),
            "individual_action_labels_opened": False,
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
            "lean_required": False,
            "engineering_only": True,
            "scientific_benchmark": False,
        }
    )
    precommit = _record(
        {
            "schema": PRECOMMIT_SCHEMA,
            "authorization_digest": authorization["record_digest"],
            "phase": phase,
            "task_ids": [row["task_id"] for row in task_inputs],
            "physical_call_plan": {
                "named_image_batch_calls_per_task": 1,
                "task_count": len(task_inputs),
                "maximum_physical_calls": len(task_inputs),
            },
            "one_call_contains_exactly_fourteen_neutral_images": True,
            "typed_output_axes": ["straight_action_count", "arc_action_count"],
            "formula_or_predicate_synthesis_present": False,
            "predictions_must_be_fsynced_before_action_program_open": True,
            "labels_model_visible": False,
            "exactly_once_journals_required": True,
            "cold_replay_model_calls": 0,
            "workers": workers,
            "python_is_canonical_authority": True,
            "lean_required": False,
        }
    )
    return authorization, precommit


def _parse_payload(
    payload: Mapping[str, Any],
    presentation: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    schema = action_count_batch_output_schema()
    if set(payload) != set(schema["properties"]):
        raise ActionCountPhaseError("typed payload fields differ")
    rows: list[dict[str, Any]] = []
    for item in presentation:
        name = item["model_visible_name"]
        stem = name.removesuffix(".png")
        values = {field: payload[f"{stem}_{field}"] for field in COUNT_FIELDS}
        error = payload[f"{stem}_error_code"]
        if (
            any(type(value) is not int or not 0 <= value <= 9 for value in values.values())
            or values["straight_action_count_lower"] > values["straight_action_count_upper"]
            or values["arc_action_count_lower"] > values["arc_action_count_upper"]
            or error not in ERROR_VALUES
            or (
                error == "unreadable"
                and tuple(values[field] for field in COUNT_FIELDS) != (0, 9, 0, 9)
            )
        ):
            raise ActionCountPhaseError("typed interval payload differs")
        rows.append(
            {
                "panel_id": item["panel_id"],
                "model_visible_name": name,
                "png_sha256": item["png_sha256"],
                **values,
                "error_code": error,
            }
        )
    return tuple(rows)


def _error_rows(
    presentation: Sequence[Mapping[str, Any]], error_code: str
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "panel_id": item["panel_id"],
            "model_visible_name": item["model_visible_name"],
            "png_sha256": item["png_sha256"],
            "straight_action_count_lower": None,
            "straight_action_count_upper": None,
            "arc_action_count_lower": None,
            "arc_action_count_upper": None,
            "error_code": error_code,
        }
        for item in presentation
    )


def _observe_task(
    *,
    root: Path,
    task_input: Mapping[str, Any],
    pixels: Mapping[str, bytes],
    authorization_digest: str,
    precommit_digest: str,
    runtime: ObjectBongardTurnRuntime,
    underlying_transport: Callable[..., CodexStructuredResult],
    replay: bool,
) -> dict[str, Any]:
    task_id = task_input["task_id"]
    presentation = task_input["presentation"]
    images = tuple(
        (row["model_visible_name"], pixels[row["panel_id"]]) for row in presentation
    )
    journal_root = root / "journals" / task_id
    if replay and not (journal_root / "outcome.json").is_file():
        raise ActionCountPhaseError("cold replay found no terminal journal")
    journal = ObjectBongardNamedImageTurnJournalTransport(
        journal_root,
        authorization_digest=authorization_digest,
        execution_precommit_digest=precommit_digest,
        task_id=task_id,
        turn_kind="action_count_batch",
        expected_prompt=action_count_batch_prompt(),
        expected_images=images,
        expected_output_schema=action_count_batch_output_schema(),
        runtime=runtime,
        underlying_transport=underlying_transport,
    )
    payload: Mapping[str, Any] | None = None
    receipt_digest: str | None = None
    transport_failed = False
    try:
        payload, receipt = _call(
            images,
            prompt=action_count_batch_prompt(),
            schema=action_count_batch_output_schema(),
            journal=journal,
            runtime=runtime,
        )
        receipt_digest = receipt.receipt_digest
    except ObjectBongardTurnCallFailed:
        transport_failed = True
    terminal = journal.verify()
    if transport_failed:
        if terminal.terminal_status != "failure":
            raise ActionCountPhaseError("failed call has no failure terminal")
        rows = _error_rows(presentation, "transport_error")
        status = "error"
        error_code = "transport_error"
    else:
        if terminal.terminal_status != "success" or payload is None:
            raise ActionCountPhaseError("successful call has no success terminal")
        try:
            rows = _parse_payload(payload, presentation)
        except ActionCountPhaseError:
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
            "individual_action_labels_opened_before_prediction": False,
            "model_visible_task_or_side_labels": False,
        }
    )


def _read_action_programs_after_predictions(
    path: Path,
    *,
    expected_raw_digest: str,
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ActionCountPhaseError("action-program path is unsafe")
    raw = path.read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != expected_raw_digest:
        raise ActionCountPhaseError("action-program source digest differs")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ActionCountPhaseError("action-program source is malformed") from exc
    if type(value) is not dict:
        raise ActionCountPhaseError("action-program source root differs")
    return value, raw


def _truth_records(
    programs: Mapping[str, Any], task_ids: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        sides = programs.get(task_id)
        if type(sides) is not list or len(sides) != 2:
            raise ActionCountPhaseError("action-program task shape differs")
        for side_index, side in enumerate(sides):
            if type(side) is not list or len(side) != 7:
                raise ActionCountPhaseError("action-program side shape differs")
            folder = 1 if side_index == 0 else 0
            for panel_index, panel in enumerate(side):
                if type(panel) is not list or len(panel) != 1 or type(panel[0]) is not list:
                    raise ActionCountPhaseError("HD panel must have one shape program")
                parsed = [action.split("_", 2)[:2] for action in panel[0]]
                if any(
                    type(action) is not str
                    or len(item) != 2
                    or item[0] not in {"line", "arc"}
                    or item[1] not in STYLES
                    for action, item in zip(panel[0], parsed, strict=True)
                ):
                    raise ActionCountPhaseError("action-program action differs")
                lines = [style for kind, style in parsed if kind == "line"]
                arcs = [style for kind, style in parsed if kind == "arc"]
                records.append(
                    {
                        "panel_id": f"hd/{task_id}/{folder}/{panel_index}.png",
                        "straight_action_count": len(lines),
                        "arc_action_count": len(arcs),
                        "line_action_count_by_style": {
                            style: lines.count(style) for style in STYLES
                        },
                        "arc_action_count_by_style": {
                            style: arcs.count(style) for style in STYLES
                        },
                    }
                )
    return tuple(records)


def _profile(truth: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    straight = truth["straight_action_count"]
    normal = truth["line_action_count_by_style"]["normal"]
    decorated = straight - normal
    line_profile = (
        "no_straight_actions"
        if straight == 0
        else "normal_only"
        if decorated == 0
        else "decorated_only"
        if normal == 0
        else "mixed_normal_and_decorated"
    )
    arc_presence = "with_arc" if truth["arc_action_count"] else "without_arc"
    styles = tuple(
        style
        for style in STYLES
        if truth["line_action_count_by_style"][style]
        or truth["arc_action_count_by_style"][style]
    )
    return line_profile, arc_presence, styles


def _axis_metrics(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], axis: str
) -> dict[str, Any]:
    exact = coverage = width = valid = errors = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for prediction, truth in rows:
        target = truth[f"{axis}_action_count"]
        lower = prediction[f"{axis}_action_count_lower"]
        upper = prediction[f"{axis}_action_count_upper"]
        if type(lower) is not int or type(upper) is not int or prediction["error_code"] != "none":
            errors += 1
            bucket = "error"
        else:
            valid += 1
            width += upper - lower
            coverage += int(lower <= target <= upper)
            exact += int(lower == upper == target)
            bucket = str(lower) if lower == upper else f"[{lower},{upper}]"
        confusion[str(target)][bucket] += 1
    denominator = len(rows)
    return {
        "denominator": denominator,
        "exact_count": exact,
        "exact_rate": [exact, denominator],
        "coverage_count": coverage,
        "coverage_rate": [coverage, denominator],
        "valid_interval_count": valid,
        "error_count": errors,
        "interval_width_sum": width,
        "mean_interval_width": [width, valid],
        "confusion": {
            truth: dict(sorted(counter.items()))
            for truth, counter in sorted(confusion.items(), key=lambda item: int(item[0]))
        },
    }


def _measurement(
    *,
    plan: Mapping[str, Any],
    phase: str,
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
        raise ActionCountPhaseError("prediction/truth panel inventories differ")
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
        prediction, truth = pair
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
            "phase": phase,
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
            "all_terminal_panels_remain_in_denominator": True,
            "labels_opened_only_after_receipted_prediction_batch_fsync": True,
            "model_calls_for_scoring": 0,
            "python_is_canonical_authority": True,
            "lean_present": False,
            "lean_required": False,
            "scientific_benchmark": False,
        }
    )


def _forbidden_transport(*_args: Any, **_kwargs: Any) -> CodexStructuredResult:
    raise AssertionError("cold replay attempted a physical model transport")


def run_action_count_phase(
    *,
    phase: str,
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
    underlying_transport: Callable[..., CodexStructuredResult] = run_codex_named_images_structured,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ActionCountPhaseError("phase must be fit, calibration, or heldout")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ActionCountPhaseError("workers must lie in 1..4")
    root = Path(os.path.abspath(os.fspath(output_root))) / phase
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ActionCountPhaseError("output root is unsafe")
    if (root / "label_release.json").exists() and not (root / "predictions.json").exists():
        raise ActionCountPhaseError("labels exist without a durable prediction batch")
    plan = _load_plan(plan_file, expected_digest=expected_plan_digest)
    cohort = plan["cohorts"].get(phase)
    if type(cohort) is not dict or type(cohort.get("task_ids")) is not list:
        raise ActionCountPhaseError("phase cohort differs")
    task_ids = tuple(cohort["task_ids"])
    if (
        len(task_ids) != len(set(task_ids))
        or any(
            "convex" in task_id or "has_four_straight_lines" in task_id
            for task_id in task_ids
        )
    ):
        raise ActionCountPhaseError("cohort violates target semantic closure")
    pixels, task_inputs = _prepare_pixels(
        Path(os.path.abspath(os.fspath(dataset_root))), task_ids, plan["record_digest"]
    )
    authorization, precommit = _authorization_precommit(
        plan=plan,
        phase=phase,
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
            raise ActionCountPhaseError("injected runtime differs from request")
        runtime_evidence = _record(
            {
                "schema": "gkm.bongard-action-count-synthetic-runtime.v1",
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
            pixels=pixels,
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
            "phase": phase,
            "task_predictions": list(task_predictions),
            "task_prediction_digests": [row["record_digest"] for row in task_predictions],
            "task_count": len(task_predictions),
            "panel_count": 14 * len(task_predictions),
            "all_journals_terminal": True,
            "individual_action_labels_opened": False,
            "prediction_batch_fsynced_before_label_source_open": True,
        }
    )
    _write_once_or_verify(root / "predictions.json", predictions)

    programs, action_raw = _read_action_programs_after_predictions(
        Path(os.path.abspath(os.fspath(action_program_file))),
        expected_raw_digest=plan["dataset_bindings"]["hd_action_program_raw_sha256"],
    )
    truth = _truth_records(programs, task_ids)
    if "sha256:" + canonical_digest(truth) != cohort["action_label_manifest_digest"]:
        raise ActionCountPhaseError("phase action-label manifest differs")
    label_release = _record(
        {
            "schema": LABEL_RELEASE_SCHEMA,
            "plan_record_digest": plan["record_digest"],
            "phase": phase,
            "prediction_batch_digest": predictions["record_digest"],
            "action_program_raw_sha256": "sha256:" + hashlib.sha256(action_raw).hexdigest(),
            "action_label_manifest_digest": cohort["action_label_manifest_digest"],
            "prediction_batch_reloaded_before_action_program_open": (
                _read_record(root / "predictions.json") == predictions
            ),
            "labels_visible_to_model": False,
            "labels_opened_by_python_after_predictions": True,
        }
    )
    _write_once_or_verify(root / "label_release.json", label_release)
    result = _measurement(
        plan=plan,
        phase=phase,
        predictions=predictions,
        label_release=label_release,
        truth_records=truth,
    )
    _write_once_or_verify(root / "result.json", result)

    replayed_predictions = tuple(
        _observe_task(
            root=root,
            task_input=task_input,
            pixels=pixels,
            authorization_digest=authorization["record_digest"],
            precommit_digest=precommit["record_digest"],
            runtime=runtime,
            underlying_transport=_forbidden_transport,
            replay=True,
        )
        for task_input in task_inputs
    )
    if replayed_predictions != task_predictions:
        raise ActionCountPhaseError("cold-replayed predictions differ")
    replayed_result = _measurement(
        plan=plan,
        phase=phase,
        predictions=predictions,
        label_release=label_release,
        truth_records=truth,
    )
    if replayed_result != result:
        raise ActionCountPhaseError("cold-replayed measurement differs")
    replay = _record(
        {
            "schema": REPLAY_SCHEMA,
            "plan_record_digest": plan["record_digest"],
            "phase": phase,
            "authorization_digest": authorization["record_digest"],
            "execution_precommit_digest": precommit["record_digest"],
            "prediction_batch_digest": predictions["record_digest"],
            "label_release_digest": label_release["record_digest"],
            "result_digest": result["record_digest"],
            "journal_count": len(task_inputs),
            "model_calls_during_replay": 0,
            "labels_opened_during_model_calls": False,
            "predictions_exactly_replayed": True,
            "measurement_exactly_replayed": True,
            "python_is_canonical_authority": True,
            "lean_required": False,
        }
    )
    _write_once_or_verify(root / "cold_replay.json", replay)
    return {"result": result, "cold_replay": replay}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
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
    completed = run_action_count_phase(
        phase=args.phase,
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
