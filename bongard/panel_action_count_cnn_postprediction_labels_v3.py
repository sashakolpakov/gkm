"""Frozen V3 gate for deriving labels only after durable predictions.

Importing this module reads no files.  The only public operation that obtains
label sources first validates, fsyncs, and byte-for-byte reloads a complete
prediction artifact.  Action programs and the audited catalog stay behind an
injected ``source_loader`` so the causal order is explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json


PREDICTION_SCHEMA = (
    "gkm.bongard-action-count-catalog-cnn-prelabel-predictions.v3"
)
CONTRACT_SCHEMA = (
    "gkm.bongard-action-count-catalog-cnn-postprediction-label-contract.v3"
)
ACTION_STYLES = frozenset({"circle", "normal", "square", "triangle", "zigzag"})
CATALOG_CLASSES = {
    -1: "catalog_unresolved",
    0: "nonconvex",
    1: "convex",
}


class PostPredictionLabelError(RuntimeError):
    """Prediction custody or delayed label derivation is invalid."""


@dataclass(frozen=True)
class CatalogTarget:
    """Exact audited-catalog result supplied after the prediction barrier."""

    raw_target: int
    supervised_class: str
    match_kind: str

    def __post_init__(self) -> None:
        if self.raw_target not in CATALOG_CLASSES:
            raise PostPredictionLabelError("catalog target leaves {-1,0,1}")
        if self.supervised_class != CATALOG_CLASSES[self.raw_target]:
            raise PostPredictionLabelError("catalog target/class disagree")
        if not self.match_kind:
            raise PostPredictionLabelError("catalog match kind is empty")


@dataclass(frozen=True)
class LabelAuthorityBindings:
    """Pre-pixel source addresses that delayed label inputs must match."""

    hd_action_program_raw_sha256: str
    catalog_algorithm_digest: str
    catalog_audit_record_digest: str
    catalog_authority_source_sha256: str

    def __post_init__(self) -> None:
        _require_digest(self.hd_action_program_raw_sha256, "HD action programs")
        _require_digest(self.catalog_algorithm_digest, "catalog algorithm")
        _require_digest(self.catalog_audit_record_digest, "catalog audit")
        _require_digest(self.catalog_authority_source_sha256, "catalog source")


@dataclass(frozen=True)
class LabelSources:
    """Sources that a runtime loader may return only after the barrier."""

    hd_action_program_raw: bytes
    catalog_lookup: Callable[[Sequence[str]], CatalogTarget]
    authority_bindings: LabelAuthorityBindings


@dataclass(frozen=True)
class PredictionBarrier:
    """Evidence passed to the delayed source loader after durable reload."""

    stage: str
    panel_ids: tuple[str, ...]
    prediction_record_digest: str
    prediction_source_sha256: str
    checkpoint_state_dict_sha256: str
    config_digest: str
    protocol: str = "fsync-file-and-parent-then-byte-identical-reload/v3"


POSTPREDICTION_LABEL_CONTRACT: Mapping[str, Any] = {
    "schema": CONTRACT_SCHEMA,
    "allowed_stages": ["calibration", "evaluation"],
    "causal_order": [
        "write_complete_canonical_prediction_record",
        "fsync_prediction_file_and_parent_directory",
        "reload_and_require_byte_identity",
        "validate_exact_stage_checkpoint_config_and_panel_order",
        "invoke_label_source_loader",
        "derive_straight_arc_and_audited_catalog_targets",
    ],
    "calibration_prediction_rows": (
        "probabilities_for_all_three_heads_in_exact_manifest_panel_order"
    ),
    "evaluation_prediction_rows": (
        "same_probabilities_plus_joint-q-derived-class-sets-frozen-before-labels"
    ),
    "catalog_typed_projection": {
        "axis": "catalog_convexity",
        "singleton_class_1": "catalog_nonconvex",
        "singleton_class_2": "catalog_convex",
        "any_set_containing_class_0_catalog_unresolved": "whole-axis-GAP",
        "geometric_turning_axis_used": False,
        "not_applicable_used_for_catalog_unresolved": False,
    },
    "pre_barrier_action_program_or_catalog_access_allowed": False,
    "import_time_file_access_allowed": False,
    "delayed_sources_must_match_pre-pixel_sha256_bindings": True,
}


def source_digest() -> str:
    """Return this frozen authority's byte digest without opening label data."""

    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def contract_digest() -> str:
    return "sha256:" + canonical_digest(POSTPREDICTION_LABEL_CONTRACT)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
    ):
        raise PostPredictionLabelError(f"{label} is not a SHA-256 address")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise PostPredictionLabelError(f"{label} is not hexadecimal") from exc
    return value


def _canonical_record(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PostPredictionLabelError(f"prediction record is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PostPredictionLabelError("prediction record is not an object")
    if raw != canonical_json(value) + b"\n":
        raise PostPredictionLabelError("prediction record is not canonical")
    body = dict(value)
    found = body.pop("record_digest", None)
    expected = "sha256:" + canonical_digest(body)
    if found != expected:
        raise PostPredictionLabelError("prediction record digest differs")
    return value


def _durable_read(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise PostPredictionLabelError("prediction artifact is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    directory = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if path.read_bytes() != raw:
        raise PostPredictionLabelError("prediction bytes changed across durable reload")
    return raw


def _probabilities(value: object, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise PostPredictionLabelError(f"{label} has wrong class cardinality")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PostPredictionLabelError(f"{label} contains a non-number")
        number = float(item)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise PostPredictionLabelError(f"{label} leaves [0,1]")
        result.append(number)
    if not math.isclose(sum(result), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise PostPredictionLabelError(f"{label} does not sum to one")
    return tuple(result)


def _class_set(value: object, size: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value != sorted(set(value))
        or any(item < 0 or item >= size for item in value)
    ):
        raise PostPredictionLabelError(f"{label} is not a canonical class set")
    return tuple(value)


def _validate_prediction_record(
    value: Mapping[str, Any],
    *,
    expected_stage: str,
    expected_panel_ids: Sequence[str],
    expected_plan_record_digest: str,
    expected_panel_manifest_record_digest: str,
    expected_checkpoint_state_dict_sha256: str,
    expected_config_digest: str,
) -> None:
    if value.get("schema") != PREDICTION_SCHEMA:
        raise PostPredictionLabelError("prediction schema differs")
    if expected_stage not in {"calibration", "evaluation"}:
        raise PostPredictionLabelError("expected stage is invalid")
    if value.get("stage") != expected_stage:
        raise PostPredictionLabelError("prediction stage differs")
    expected_bindings = {
        "plan_record_digest": expected_plan_record_digest,
        "panel_manifest_record_digest": expected_panel_manifest_record_digest,
        "checkpoint_state_dict_sha256": expected_checkpoint_state_dict_sha256,
        "config_digest": expected_config_digest,
    }
    for key, expected in expected_bindings.items():
        _require_digest(expected, f"expected {key}")
        if value.get(key) != expected:
            raise PostPredictionLabelError(f"prediction {key} differs")
    panel_ids = value.get("panel_ids")
    if panel_ids != list(expected_panel_ids):
        raise PostPredictionLabelError("prediction panel order differs")
    if any(not isinstance(panel_id, str) for panel_id in panel_ids):
        raise PostPredictionLabelError("prediction panel ID is invalid")
    if value.get("straight_class_order") != list(range(10)):
        raise PostPredictionLabelError("straight class order differs")
    if value.get("arc_class_order") != list(range(10)):
        raise PostPredictionLabelError("arc class order differs")
    if value.get("catalog_class_order") != [
        "catalog_unresolved",
        "nonconvex",
        "convex",
    ]:
        raise PostPredictionLabelError("catalog class order differs")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != len(panel_ids):
        raise PostPredictionLabelError("prediction row count differs")
    joint_q = value.get("joint_q")
    joint_q_record_digest = value.get("joint_q_record_digest")
    if expected_stage == "calibration":
        if joint_q is not None or joint_q_record_digest is not None:
            raise PostPredictionLabelError("calibration predictions contain joint q")
    else:
        if (
            isinstance(joint_q, bool)
            or not isinstance(joint_q, (int, float))
            or not math.isfinite(float(joint_q))
            or not 0.0 <= float(joint_q) <= 1.0
        ):
            raise PostPredictionLabelError("evaluation joint q is invalid")
        _require_digest(joint_q_record_digest, "joint q record digest")
    for index, (panel_id, row) in enumerate(zip(panel_ids, rows)):
        if not isinstance(row, dict) or row.get("panel_id") != panel_id:
            raise PostPredictionLabelError(f"prediction row {index} ID differs")
        straight = _probabilities(
            row.get("straight_probabilities"), 10, f"row {index} straight"
        )
        arc = _probabilities(row.get("arc_probabilities"), 10, f"row {index} arc")
        catalog = _probabilities(
            row.get("catalog_probabilities"), 3, f"row {index} catalog"
        )
        if expected_stage == "calibration":
            if any(
                key in row
                for key in ("straight_class_set", "arc_class_set", "catalog_class_set")
            ):
                raise PostPredictionLabelError(
                    "calibration row contains post-calibration class set"
                )
        else:
            threshold = float(joint_q)
            for name, probabilities, size in (
                ("straight", straight, 10),
                ("arc", arc, 10),
                ("catalog", catalog, 3),
            ):
                found = _class_set(
                    row.get(f"{name}_class_set"), size, f"row {index} {name} set"
                )
                expected = tuple(
                    class_index
                    for class_index, probability in enumerate(probabilities)
                    if 1.0 - probability <= threshold
                )
                if found != expected:
                    raise PostPredictionLabelError(
                        f"row {index} {name} set differs from joint q"
                    )


def _actions_for_panel(programs: Mapping[str, Any], panel_id: str) -> list[str]:
    parts = panel_id.split("/")
    if len(parts) != 4 or parts[0] != "hd" or not parts[3].endswith(".png"):
        raise PostPredictionLabelError(f"invalid HD panel ID: {panel_id}")
    task_id = parts[1]
    try:
        folder = int(parts[2])
        panel_index = int(parts[3][:-4])
    except ValueError as exc:
        raise PostPredictionLabelError(f"invalid panel coordinates: {panel_id}") from exc
    if folder not in {0, 1} or panel_index not in range(7):
        raise PostPredictionLabelError(f"panel coordinates leave closed domain: {panel_id}")
    task = programs.get(task_id)
    side_index = 1 - folder
    if not isinstance(task, list) or len(task) != 2:
        raise PostPredictionLabelError(f"missing action-program task: {task_id}")
    side = task[side_index]
    if not isinstance(side, list) or len(side) != 7:
        raise PostPredictionLabelError(f"invalid action-program side: {panel_id}")
    panel = side[panel_index]
    if not isinstance(panel, list) or len(panel) != 1:
        raise PostPredictionLabelError(f"invalid action-program panel: {panel_id}")
    actions = panel[0]
    if not isinstance(actions, list) or any(not isinstance(item, str) for item in actions):
        raise PostPredictionLabelError(f"invalid action list: {panel_id}")
    return actions


def _derive_rows(
    panel_ids: Sequence[str],
    programs: Mapping[str, Any],
    catalog_lookup: Callable[[Sequence[str]], CatalogTarget],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for panel_id in panel_ids:
        actions = _actions_for_panel(programs, panel_id)
        parsed = [action.split("_", 2)[:2] for action in actions]
        if any(
            len(item) != 2
            or item[0] not in {"line", "arc"}
            or item[1] not in ACTION_STYLES
            for item in parsed
        ):
            raise PostPredictionLabelError(f"unsupported action token: {panel_id}")
        straight_count = sum(kind == "line" for kind, _style in parsed)
        arc_count = sum(kind == "arc" for kind, _style in parsed)
        if straight_count > 9 or arc_count > 9:
            raise PostPredictionLabelError(f"action count leaves 0..9: {panel_id}")
        catalog = catalog_lookup(actions)
        if not isinstance(catalog, CatalogTarget):
            raise PostPredictionLabelError("catalog lookup returned the wrong type")
        rows.append(
            {
                "arc_action_count": arc_count,
                "catalog_convexity_class": catalog.supervised_class,
                "catalog_convexity_target": catalog.raw_target,
                "catalog_match_kind": catalog.match_kind,
                "panel_id": panel_id,
                "straight_action_count": straight_count,
            }
        )
    return tuple(rows)


def derive_labels_after_durable_predictions(
    *,
    prediction_path: Path,
    expected_stage: str,
    expected_panel_ids: Sequence[str],
    expected_plan_record_digest: str,
    expected_panel_manifest_record_digest: str,
    expected_checkpoint_state_dict_sha256: str,
    expected_config_digest: str,
    expected_label_authority_bindings: LabelAuthorityBindings,
    source_loader: Callable[[PredictionBarrier], LabelSources],
) -> tuple[dict[str, Any], ...]:
    """Open label sources only after the prediction durability barrier."""

    raw = _durable_read(prediction_path)
    prediction = _canonical_record(raw)
    _validate_prediction_record(
        prediction,
        expected_stage=expected_stage,
        expected_panel_ids=expected_panel_ids,
        expected_plan_record_digest=expected_plan_record_digest,
        expected_panel_manifest_record_digest=expected_panel_manifest_record_digest,
        expected_checkpoint_state_dict_sha256=expected_checkpoint_state_dict_sha256,
        expected_config_digest=expected_config_digest,
    )
    barrier = PredictionBarrier(
        stage=expected_stage,
        panel_ids=tuple(expected_panel_ids),
        prediction_record_digest=prediction["record_digest"],
        prediction_source_sha256=_sha256_bytes(raw),
        checkpoint_state_dict_sha256=expected_checkpoint_state_dict_sha256,
        config_digest=expected_config_digest,
    )
    sources = source_loader(barrier)
    if not isinstance(sources, LabelSources):
        raise PostPredictionLabelError("source loader returned the wrong type")
    if sources.authority_bindings != expected_label_authority_bindings:
        raise PostPredictionLabelError("delayed label authority bindings differ")
    if _sha256_bytes(sources.hd_action_program_raw) != (
        expected_label_authority_bindings.hd_action_program_raw_sha256
    ):
        raise PostPredictionLabelError("HD action-program bytes differ")
    try:
        programs = json.loads(sources.hd_action_program_raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PostPredictionLabelError(f"HD action programs are invalid: {exc}") from exc
    if not isinstance(programs, dict):
        raise PostPredictionLabelError("HD action programs are not an object")
    return _derive_rows(expected_panel_ids, programs, sources.catalog_lookup)
