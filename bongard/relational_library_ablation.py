"""Finite-library resubstitution ablation over an exposed coverage pilot.

This command is deliberately downstream of :mod:`relational_coverage_drill`.
It accepts only a completed, content-addressed coverage-v1 report and the
exact exposure successor named by that report.  The coverage event already
authorized the selected train/validation pixels; this command creates no new
exposure event and cannot select another task.

The query inventory is the complete frozen 2,520-member positive v3 library.
It is committed before any panel is reopened.  Results use the canonical v3
primitive clause evaluators plus a prepared reducer that preserves
``evaluate_relational_query``'s conjunction, existential, and scenario order.
Frozen runtime samples and an exhaustive fixture test check equivalence with
the public evaluator.  The output is therefore a post-hoc
library-coverage/resubstitution diagnostic, never a model benchmark or a
generalization estimate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from bongard.artifacts import canonical_json
from bongard.evidence import Disposition
from bongard.exposure import ExposureLedger
from bongard.loop_geometry import (
    LOOP_GEOMETRY_ALGORITHM_ID,
    loop_geometry_algorithm_digest,
    loop_geometry_source_digest,
)
from bongard.loop_scene_witnesses import (
    LOOP_SCENE_ALGORITHM_ID,
    LoopScenePacket,
    extract_loop_scene_witnesses,
    loop_scene_catalog_digest,
    loop_scene_extractor_digest,
)
from bongard.point_contact import (
    POINT_CONTACT_ALGORITHM_ID,
    point_contact_algorithm_digest,
    point_contact_source_digest,
)
from bongard.relational_coverage_drill import _read_png_no_follow
from bongard.relational_visual_query import (
    ALLOWED_AREA_RATIOS,
    ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES,
    ALLOWED_SIDE_COUNTS,
    AreaRatioClause,
    EdgeObliquenessClause,
    PointContactClause,
    RELATIONAL_QUERY_ALGORITHM_ID,
    Rational,
    RelationalQueryResult,
    RelationalVisualQuery,
    SideCountClause,
    _area_ratio,
    _edge_obliqueness,
    _point_contact,
    _side_count,
    enumerate_factorized_shape_ratio_queries,
    evaluate_relational_query,
    relational_query_algorithm_digest,
    relational_query_source_digest,
)
from bongard.visual_witness_bundle import (
    VISUAL_WITNESS_BUNDLE_ALGORITHM_ID,
    visual_witness_bundle_catalog_digest,
    visual_witness_bundle_extractor_digest,
)


SCHEMA = "gkm.bongard-relational-library-ablation.v1"
ALGORITHM_ID = "bongard.relational-library-ablation/complete-v3-library-v1"
COVERAGE_SCHEMA_V1 = "gkm.bongard-relational-coverage-drill.v1"
COVERAGE_SELECTION_SCHEMA_V1 = "gkm.bongard-relational-coverage-selection.v1"
COVERAGE_MANIFEST_SCHEMA_V1 = "gkm.bongard-selected-png-manifest.v1"
COVERAGE_ALGORITHM_V1 = "bongard.relational-coverage-drill/hash-stratified-v1"
QUERY_LIBRARY_SIZE = 2_520
DEFAULT_EXTRACTION_WORKERS = 4
CANONICAL_EQUIVALENCE_SAMPLE_INDICES = (
    0,
    1,
    8,
    2_519,
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TASK_ID = re.compile(r"^(ff|bd|hd)_.+_[0-9]{4}\Z")
_PANEL_ID = re.compile(
    r"^(ff|bd|hd)/((?:ff|bd|hd)_.+_[0-9]{4})/([01])/([0-6])\.png\Z"
)

_DISPOSITIONS = (
    Disposition.PRESENT,
    Disposition.CERTIFIED_ABSENT,
    Disposition.INDETERMINATE,
    Disposition.ERROR,
)

_RESTRICTIONS = {
    "allowed_splits": ["train", "val"],
    "official_test_pixels_authorized": False,
    "action_program_json_authorized": False,
    "proposer_or_model_authorized": False,
    "candidate_dependent_extraction_authorized": False,
}


class RelationalLibraryAblationError(RuntimeError):
    """The immutable input chain or the selected-only replay is invalid."""


def _address(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _bytes_address(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _exact_fields(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise RelationalLibraryAblationError(
            f"{label} fields differ from schema: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise RelationalLibraryAblationError(
            f"{label} must be a prefixed lowercase SHA-256"
        )
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise RelationalLibraryAblationError(
            f"{label} must be an unprefixed lowercase SHA-256"
        )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RelationalLibraryAblationError(
            f"{label} must be an integer at least {minimum}"
        )
    return value


def _json_object_no_duplicates(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RelationalLibraryAblationError(
                    f"{label} contains duplicate JSON field {key!r}"
                )
            result[key] = value
        return result

    try:
        result = json.loads(payload, object_pairs_hook=hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RelationalLibraryAblationError(
            f"cannot decode {label} as JSON: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise RelationalLibraryAblationError(f"{label} must be a JSON object")
    return result


def _load_coverage_report(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RelationalLibraryAblationError(
            f"cannot read coverage report {path}: {exc}"
        ) from exc
    report = _json_object_no_duplicates(payload, "coverage report")
    if payload != canonical_json(report) + b"\n":
        raise RelationalLibraryAblationError(
            "coverage report bytes are not the canonical write-once encoding"
        )
    return report


def _load_exposure_successor(path: Path) -> ExposureLedger:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RelationalLibraryAblationError(
            f"cannot read exposure successor {path}: {exc}"
        ) from exc
    raw = _json_object_no_duplicates(payload, "exposure successor")
    try:
        return ExposureLedger.from_dict(raw)
    except Exception as exc:
        raise RelationalLibraryAblationError(
            f"exposure successor failed integrity validation: {exc}"
        ) from exc


def _current_extractor_identities() -> dict[str, str]:
    """Return only identities that govern replayed packet semantics.

    The completed coverage report's coverage-script source digest is a
    historical receipt.  It is authenticated by the report digest but is not
    compared with today's coverage CLI source; otherwise a selection-policy
    maintenance edit would make an immutable, valid report unreadable.
    """

    return {
        "loop_scene_algorithm_id": LOOP_SCENE_ALGORITHM_ID,
        "loop_scene_catalog_digest": loop_scene_catalog_digest(),
        "loop_scene_extractor_digest": loop_scene_extractor_digest(),
        "loop_geometry_algorithm_id": LOOP_GEOMETRY_ALGORITHM_ID,
        "loop_geometry_algorithm_digest": loop_geometry_algorithm_digest(),
        "loop_geometry_python_source_digest": loop_geometry_source_digest(),
        "point_contact_algorithm_id": POINT_CONTACT_ALGORITHM_ID,
        "point_contact_algorithm_digest": point_contact_algorithm_digest(),
        "point_contact_python_source_digest": point_contact_source_digest(),
        "visual_witness_bundle_algorithm_id": VISUAL_WITNESS_BUNDLE_ALGORITHM_ID,
        "visual_witness_bundle_catalog_digest": (
            visual_witness_bundle_catalog_digest()
        ),
        "visual_witness_bundle_extractor_digest": (
            visual_witness_bundle_extractor_digest()
        ),
        "reference_execution": "python-canonical/v1",
    }


@dataclass(frozen=True, slots=True)
class _SelectedTask:
    task_id: str
    family: str
    split: str
    generator: str
    panels: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _ValidatedCoverage:
    report: Mapping[str, Any]
    successor: ExposureLedger
    source_corpus_manifest_digest: str
    selected_manifest_digest: str
    selection_digest: str
    tasks: tuple[_SelectedTask, ...]
    receipts: Mapping[str, Mapping[str, Any]]


def _validate_selection(
    selection_value: object,
    *,
    source_digest: str,
    split_digest: str,
    predecessor_digest: str,
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    selection = _exact_fields(
        selection_value,
        frozenset(
            {
                "schema",
                "algorithm_id",
                "namespace",
                "allowed_splits",
                "per_generator",
                "per_split_family",
                "source_corpus_manifest_digest",
                "split_source_digest",
                "exposure_predecessor_digest",
                "exact_unused_count",
                "strict_dev_protection",
                "generator_stratification_qualification",
                "generator_shortlist_count",
                "selected",
                "digest",
            }
        ),
        "coverage selection",
    )
    if (
        selection["schema"] != COVERAGE_SELECTION_SCHEMA_V1
        or selection["algorithm_id"] != COVERAGE_ALGORITHM_V1
    ):
        raise RelationalLibraryAblationError("unsupported coverage selection")
    if selection["allowed_splits"] != ["train", "val"]:
        raise RelationalLibraryAblationError(
            "coverage selection does not preserve the official-test restriction"
        )
    expected_bindings = {
        "source_corpus_manifest_digest": source_digest,
        "split_source_digest": split_digest,
        "exposure_predecessor_digest": predecessor_digest,
    }
    for field, expected in expected_bindings.items():
        if selection[field] != expected:
            raise RelationalLibraryAblationError(
                f"coverage selection {field} disagrees with its parent"
            )
    content = dict(selection)
    recorded_digest = _require_address(content.pop("digest"), "selection digest")
    if _address(content) != recorded_digest:
        raise RelationalLibraryAblationError("coverage selection digest mismatch")
    selected = selection["selected"]
    if not isinstance(selected, list) or not selected:
        raise RelationalLibraryAblationError(
            "coverage selection must contain selected tasks"
        )
    normalized: list[Mapping[str, Any]] = []
    for raw in selected:
        item = _exact_fields(
            raw,
            frozenset(
                {
                    "task_id",
                    "family",
                    "split",
                    "generator",
                    "generator_rank",
                    "family_rank",
                }
            ),
            "selected task",
        )
        if item["split"] not in {"train", "val"}:
            raise RelationalLibraryAblationError(
                "selected task is not authenticated train/validation data"
            )
        if not isinstance(item["task_id"], str) or _TASK_ID.fullmatch(item["task_id"]) is None:
            raise RelationalLibraryAblationError("selected task ID is malformed")
        family = item["task_id"].split("_", 1)[0]
        if item["family"] != family:
            raise RelationalLibraryAblationError(
                "selected task family disagrees with task ID"
            )
        if not isinstance(item["generator"], str) or not item["generator"]:
            raise RelationalLibraryAblationError("selected generator is malformed")
        _require_digest(item["generator_rank"], "generator rank")
        _require_digest(item["family_rank"], "family rank")
        normalized.append(item)
    if len({item["task_id"] for item in normalized}) != len(normalized):
        raise RelationalLibraryAblationError("coverage selection repeats a task ID")
    expected_order = sorted(
        normalized,
        key=lambda item: (
            item["split"],
            item["family"],
            item["generator"],
            item["task_id"],
        ),
    )
    if normalized != expected_order:
        raise RelationalLibraryAblationError("coverage selection order is not canonical")
    return recorded_digest, tuple(normalized)


def _validate_panel_manifest(
    task_value: object, selected: Mapping[str, Any]
) -> _SelectedTask:
    task = _exact_fields(
        task_value,
        frozenset({"task_id", "family", "split", "generator", "panels", "digest"}),
        "selected task manifest",
    )
    for field in ("task_id", "family", "split", "generator"):
        if task[field] != selected[field]:
            raise RelationalLibraryAblationError(
                f"selected task manifest {field} disagrees with selection"
            )
    content = dict(task)
    recorded = _require_address(content.pop("digest"), "task manifest digest")
    if _address(content) != recorded:
        raise RelationalLibraryAblationError("selected task manifest digest mismatch")
    panels = task["panels"]
    if not isinstance(panels, list) or len(panels) != 14:
        raise RelationalLibraryAblationError(
            "selected task manifest must name exactly fourteen panels"
        )
    normalized: list[Mapping[str, Any]] = []
    expected_pairs = {
        (polarity, index)
        for polarity in ("positive", "negative")
        for index in range(7)
    }
    observed_pairs: set[tuple[str, int]] = set()
    for raw in panels:
        panel = _exact_fields(
            raw,
            frozenset(
                {
                    "panel_id",
                    "polarity",
                    "index",
                    "filename",
                    "sha256",
                    "size_bytes",
                }
            ),
            "selected panel manifest entry",
        )
        match = (
            _PANEL_ID.fullmatch(panel["panel_id"])
            if isinstance(panel["panel_id"], str)
            else None
        )
        if match is None:
            raise RelationalLibraryAblationError("selected panel_id is malformed")
        family, task_id, label, index_text = match.groups()
        index = _integer(panel["index"], "panel index")
        polarity = panel["polarity"]
        expected_polarity = "positive" if label == "1" else "negative"
        if (
            family != task["family"]
            or task_id != task["task_id"]
            or polarity != expected_polarity
            or index != int(index_text)
            or panel["filename"] != f"{index}.png"
        ):
            raise RelationalLibraryAblationError(
                "selected panel metadata disagrees with its canonical panel_id"
            )
        if index > 6:
            raise RelationalLibraryAblationError("selected panel index exceeds six")
        _require_address(panel["sha256"], "selected PNG digest")
        _integer(panel["size_bytes"], "selected PNG size", minimum=1)
        observed_pairs.add((polarity, index))
        normalized.append(panel)
    if observed_pairs != expected_pairs or len(observed_pairs) != len(normalized):
        raise RelationalLibraryAblationError(
            "selected task manifest does not contain exactly indices 0..6 per side"
        )
    expected_order = sorted(
        normalized,
        key=lambda panel: (
            0 if panel["polarity"] == "positive" else 1,
            panel["index"],
        ),
    )
    if normalized != expected_order:
        raise RelationalLibraryAblationError(
            "selected panel manifest order is not canonical"
        )
    return _SelectedTask(
        task_id=task["task_id"],
        family=task["family"],
        split=task["split"],
        generator=task["generator"],
        panels=tuple(normalized),
    )


def _validate_coverage_inputs(
    report: Mapping[str, Any],
    successor: ExposureLedger,
    *,
    report_path: Path,
    successor_path: Path,
) -> _ValidatedCoverage:
    top = _exact_fields(
        report,
        frozenset(
            {
                "schema",
                "algorithm_id",
                "input_digest",
                "source",
                "exposure",
                "restrictions",
                "algorithm_identities",
                "selection",
                "selected_task_manifest",
                "panel_receipts",
                "aggregates",
                "output_digest",
            }
        ),
        "coverage report",
    )
    if (
        top["schema"] != COVERAGE_SCHEMA_V1
        or top["algorithm_id"] != COVERAGE_ALGORITHM_V1
    ):
        raise RelationalLibraryAblationError(
            "only immutable relational coverage-v1 reports are supported"
        )
    report_content = dict(top)
    output_digest = _require_address(
        report_content.pop("output_digest"), "coverage output digest"
    )
    if _address(report_content) != output_digest:
        raise RelationalLibraryAblationError("coverage output digest mismatch")
    if report_path.name != output_digest.removeprefix("sha256:") + ".coverage.json":
        raise RelationalLibraryAblationError(
            "coverage report filename disagrees with its output digest"
        )
    source = _exact_fields(
        top["source"],
        frozenset({"corpus_manifest_digest", "split_source_digest"}),
        "coverage source",
    )
    corpus_digest = _require_address(
        source["corpus_manifest_digest"], "source corpus manifest digest"
    )
    split_digest = _require_address(
        source["split_source_digest"], "source split digest"
    )
    exposure = _exact_fields(
        top["exposure"],
        frozenset(
            {
                "predecessor_digest",
                "successor_digest",
                "successor_event_count",
                "successor_filename",
                "precommit_before_selected_png_access",
            }
        ),
        "coverage exposure",
    )
    predecessor_digest = _require_address(
        exposure["predecessor_digest"], "exposure predecessor digest"
    )
    successor_digest = _require_address(
        exposure["successor_digest"], "exposure successor digest"
    )
    event_count = _integer(
        exposure["successor_event_count"], "successor event count", minimum=1
    )
    if exposure["precommit_before_selected_png_access"] is not True:
        raise RelationalLibraryAblationError(
            "coverage report lacks the exposure-before-pixels commitment"
        )
    if successor.digest != successor_digest or successor.corpus_digest != corpus_digest:
        raise RelationalLibraryAblationError(
            "exposure successor digest/corpus binding disagrees with coverage report"
        )
    if len(successor.events) != event_count:
        raise RelationalLibraryAblationError(
            "exposure successor event count disagrees with coverage report"
        )
    expected_successor_name = successor_digest.removeprefix("sha256:") + ".exposure.json"
    if (
        exposure["successor_filename"] != expected_successor_name
        or successor_path.name != expected_successor_name
    ):
        raise RelationalLibraryAblationError(
            "exposure successor filename disagrees with its digest"
        )
    predecessor = ExposureLedger(successor.corpus_digest, successor.events[:-1])
    if predecessor.digest != predecessor_digest:
        raise RelationalLibraryAblationError(
            "exposure successor is not the reported predecessor plus one event"
        )
    if top["restrictions"] != _RESTRICTIONS:
        raise RelationalLibraryAblationError(
            "coverage report does not enforce the exact official-test/action/model restrictions"
        )
    algorithms = _exact_fields(
        top["algorithm_identities"],
        frozenset(
            {
                "coverage_algorithm_id",
                "coverage_python_source_digest",
                "loop_scene_algorithm_id",
                "loop_scene_catalog_digest",
                "loop_scene_extractor_digest",
                "loop_geometry_algorithm_id",
                "loop_geometry_algorithm_digest",
                "loop_geometry_python_source_digest",
                "point_contact_algorithm_id",
                "point_contact_algorithm_digest",
                "point_contact_python_source_digest",
                "visual_witness_bundle_algorithm_id",
                "visual_witness_bundle_catalog_digest",
                "visual_witness_bundle_extractor_digest",
                "reference_execution",
            }
        ),
        "coverage algorithm identities",
    )
    if algorithms["coverage_algorithm_id"] != COVERAGE_ALGORITHM_V1:
        raise RelationalLibraryAblationError("coverage algorithm identity is unsupported")
    _require_digest(
        algorithms["coverage_python_source_digest"],
        "historical coverage Python source digest",
    )
    current_extractors = _current_extractor_identities()
    changed = {
        field: (algorithms[field], expected)
        for field, expected in current_extractors.items()
        if algorithms[field] != expected
    }
    if changed:
        raise RelationalLibraryAblationError(
            "current Python extractor identities differ from the completed coverage run: "
            + ", ".join(sorted(changed))
        )
    selection_digest, selected = _validate_selection(
        top["selection"],
        source_digest=corpus_digest,
        split_digest=split_digest,
        predecessor_digest=predecessor_digest,
    )
    manifest = _exact_fields(
        top["selected_task_manifest"],
        frozenset(
            {
                "schema",
                "source_corpus_manifest_digest",
                "split_source_digest",
                "selection_digest",
                "tasks",
                "digest",
            }
        ),
        "selected-only PNG manifest",
    )
    if manifest["schema"] != COVERAGE_MANIFEST_SCHEMA_V1:
        raise RelationalLibraryAblationError("unsupported selected-only PNG manifest")
    if (
        manifest["source_corpus_manifest_digest"] != corpus_digest
        or manifest["split_source_digest"] != split_digest
        or manifest["selection_digest"] != selection_digest
    ):
        raise RelationalLibraryAblationError(
            "selected-only PNG manifest disagrees with its source/selection"
        )
    manifest_content = dict(manifest)
    manifest_digest = _require_address(
        manifest_content.pop("digest"), "selected-only PNG manifest digest"
    )
    if _address(manifest_content) != manifest_digest:
        raise RelationalLibraryAblationError(
            "selected-only PNG manifest digest mismatch"
        )
    task_values = manifest["tasks"]
    if not isinstance(task_values, list) or len(task_values) != len(selected):
        raise RelationalLibraryAblationError(
            "selected-only PNG manifest task inventory differs from selection"
        )
    tasks = tuple(
        _validate_panel_manifest(task, selected_item)
        for task, selected_item in zip(task_values, selected, strict=True)
    )
    selected_ids = tuple(item.task_id for item in tasks)
    event = successor.events[-1]
    if (
        event.phase != "relational-coverage-drill"
        or event.panel_ids
        or event.task_ids != tuple(sorted(selected_ids))
        or event.source != f"relational-coverage-input:{top['input_digest']}"
    ):
        raise RelationalLibraryAblationError(
            "selected task IDs are not exactly the terminal coverage exposure event"
        )
    input_commitment = {
        "schema": "gkm.bongard-relational-coverage-input.v1",
        "source_corpus_manifest_digest": corpus_digest,
        "split_source_digest": split_digest,
        "exposure_predecessor_digest": predecessor_digest,
        "selection_digest": selection_digest,
        "algorithm_identities": dict(algorithms),
        "restrictions": _RESTRICTIONS,
    }
    if _address(input_commitment) != top["input_digest"]:
        raise RelationalLibraryAblationError("coverage input commitment digest mismatch")
    receipt_values = top["panel_receipts"]
    if not isinstance(receipt_values, list) or len(receipt_values) != 14 * len(tasks):
        raise RelationalLibraryAblationError(
            "coverage panel receipt inventory is incomplete"
        )
    manifest_panels = {
        panel["panel_id"]: panel for task in tasks for panel in task.panels
    }
    receipts: dict[str, Mapping[str, Any]] = {}
    for raw in receipt_values:
        receipt = _exact_fields(
            raw,
            frozenset(
                {
                    "panel_id",
                    "png_sha256",
                    "status",
                    "error_type",
                    "loop_scene_packet_digest",
                }
            ),
            "coverage panel receipt",
        )
        panel_id = receipt["panel_id"]
        if panel_id not in manifest_panels or panel_id in receipts:
            raise RelationalLibraryAblationError(
                "coverage panel receipt is unknown or duplicated"
            )
        if receipt["png_sha256"] != manifest_panels[panel_id]["sha256"]:
            raise RelationalLibraryAblationError(
                "coverage panel receipt PNG digest disagrees with manifest"
            )
        if receipt["status"] == "present":
            if receipt["error_type"] is not None:
                raise RelationalLibraryAblationError(
                    "present coverage receipt cannot name an error"
                )
            _require_digest(
                receipt["loop_scene_packet_digest"], "coverage packet digest"
            )
        elif receipt["status"] == "error":
            if (
                not isinstance(receipt["error_type"], str)
                or not receipt["error_type"]
                or receipt["loop_scene_packet_digest"] is not None
            ):
                raise RelationalLibraryAblationError(
                    "error coverage receipt is malformed"
                )
        else:
            raise RelationalLibraryAblationError(
                "coverage panel receipt status is not canonical"
            )
        receipts[panel_id] = receipt
    if list(receipt["panel_id"] for receipt in receipt_values) != sorted(receipts):
        raise RelationalLibraryAblationError(
            "coverage panel receipts are not panel-ID sorted"
        )
    return _ValidatedCoverage(
        report=top,
        successor=successor,
        source_corpus_manifest_digest=corpus_digest,
        selected_manifest_digest=manifest_digest,
        selection_digest=selection_digest,
        tasks=tasks,
        receipts=receipts,
    )


def _assert_packet_identity(packet: LoopScenePacket, panel_address: str) -> None:
    expected = _current_extractor_identities()
    if packet.panel_digest != panel_address.removeprefix("sha256:"):
        raise RelationalLibraryAblationError(
            "re-extracted loop packet is not bound to the selected PNG bytes"
        )
    packet_fields = {
        "extractor_artifact_digest": expected["loop_scene_extractor_digest"],
        "loop_geometry_algorithm_digest": expected[
            "loop_geometry_algorithm_digest"
        ],
        "point_contact_algorithm_digest": expected[
            "point_contact_algorithm_digest"
        ],
        "parent_bundle_extractor_digest": expected[
            "visual_witness_bundle_extractor_digest"
        ],
    }
    changed = {
        field: (getattr(packet, field), wanted)
        for field, wanted in packet_fields.items()
        if getattr(packet, field) != wanted
    }
    if changed:
        raise RelationalLibraryAblationError(
            "re-extracted packet carries stale algorithm identities: "
            + ", ".join(sorted(changed))
        )


def _selected_panel_path(root: Path, task: _SelectedTask, panel: Mapping[str, Any]) -> Path:
    # No value from the report is accepted as a filesystem path.  The strict
    # panel-id grammar above leaves only these two official image layouts.
    candidates = tuple(
        root / task.family / component / task.task_id
        for component in ("images", "png")
    )
    present = tuple(path for path in candidates if path.is_dir())
    if len(present) != 1:
        raise RelationalLibraryAblationError(
            f"selected task has {len(present)} official image layouts: {task.task_id}"
        )
    label = "1" if panel["polarity"] == "positive" else "0"
    return present[0] / label / panel["filename"]


@dataclass(frozen=True, slots=True)
class _PanelOutcome:
    disposition: Disposition
    reason_codes: tuple[str, ...]


def _failure_reason_codes(
    result: RelationalQueryResult, query: RelationalVisualQuery
) -> tuple[str, ...]:
    if result.disposition not in {Disposition.INDETERMINATE, Disposition.ERROR}:
        return ()
    predicate_by_clause = {
        clause.clause_id: clause.predicate.value for clause in query.clauses
    }
    reasons: set[str] = set()
    for scenario in result.scenarios:
        if scenario.disposition not in {Disposition.INDETERMINATE, Disposition.ERROR}:
            continue
        reasons.add(f"scenario:{scenario.reason_code}")
        for _, disposition in scenario.role_domain:
            if disposition in {Disposition.INDETERMINATE, Disposition.ERROR}:
                reasons.add(f"role_domain:{disposition.value}")
        for binding in scenario.bindings:
            for clause in binding.clauses:
                if clause.disposition in {Disposition.INDETERMINATE, Disposition.ERROR}:
                    reasons.add(
                        "clause:"
                        + predicate_by_clause[clause.clause_id]
                        + ":"
                        + clause.disposition.value
                    )
    return tuple(sorted(reasons))


@dataclass(frozen=True, slots=True)
class _QueryPlan:
    numerator_side_count: int
    denominator_side_count: int
    ratio: tuple[int, int]
    denominator_obliqueness_millidegrees: int | None
    require_point_contact: bool


@dataclass(frozen=True, slots=True)
class _PreparedBinding:
    first_side_counts: Mapping[int, Disposition]
    second_side_counts: Mapping[int, Disposition]
    area_ratios: Mapping[tuple[int, int], Disposition]
    second_obliqueness: Mapping[int, Disposition]
    point_contact: Disposition


@dataclass(frozen=True, slots=True)
class _PreparedScenario:
    role_domain: tuple[Disposition, ...]
    bindings: tuple[_PreparedBinding, ...]


@dataclass(frozen=True, slots=True)
class _PreparedPacket:
    scenarios: tuple[_PreparedScenario, ...]


def _query_plan(query: RelationalVisualQuery) -> _QueryPlan:
    side_by_role: dict[str, int] = {}
    ratio: tuple[int, int] | None = None
    obliqueness: int | None = None
    require_contact = False
    for clause in query.clauses:
        if isinstance(clause, SideCountClause):
            side_by_role[clause.role_id] = clause.count
        elif isinstance(clause, AreaRatioClause):
            ratio = (clause.ratio.numerator, clause.ratio.denominator)
        elif isinstance(clause, EdgeObliquenessClause):
            if clause.role_id != "role-01":
                raise RelationalLibraryAblationError(
                    "finite library obliqueness clause is not denominator-bound"
                )
            obliqueness = clause.threshold_millidegrees
        elif isinstance(clause, PointContactClause):
            require_contact = True
        else:  # pragma: no cover - closed v3 constructor makes this unreachable
            raise RelationalLibraryAblationError("unknown finite-library clause")
    if set(side_by_role) != {"role-00", "role-01"} or ratio is None:
        raise RelationalLibraryAblationError(
            "finite-library query lacks its two side clauses or area ratio"
        )
    return _QueryPlan(
        numerator_side_count=side_by_role["role-00"],
        denominator_side_count=side_by_role["role-01"],
        ratio=ratio,
        denominator_obliqueness_millidegrees=obliqueness,
        require_point_contact=require_contact,
    )


def _conjoin_dispositions(dispositions: Sequence[Disposition]) -> Disposition:
    if Disposition.CERTIFIED_ABSENT in dispositions:
        return Disposition.CERTIFIED_ABSENT
    if Disposition.ERROR in dispositions:
        return Disposition.ERROR
    if Disposition.INDETERMINATE in dispositions:
        return Disposition.INDETERMINATE
    return Disposition.PRESENT


def _existential_dispositions(
    bindings: Sequence[Disposition], role_domain: Sequence[Disposition]
) -> Disposition:
    if Disposition.PRESENT in bindings:
        return Disposition.PRESENT
    if Disposition.ERROR in bindings or Disposition.ERROR in role_domain:
        return Disposition.ERROR
    if (
        Disposition.INDETERMINATE in bindings
        or Disposition.INDETERMINATE in role_domain
    ):
        return Disposition.INDETERMINATE
    return Disposition.CERTIFIED_ABSENT


def _scenario_consensus(dispositions: Sequence[Disposition]) -> Disposition:
    if all(item is Disposition.PRESENT for item in dispositions):
        return Disposition.PRESENT
    if all(item is Disposition.CERTIFIED_ABSENT for item in dispositions):
        return Disposition.CERTIFIED_ABSENT
    if Disposition.ERROR in dispositions:
        return Disposition.ERROR
    return Disposition.INDETERMINATE


def _prepare_packet(packet: LoopScenePacket) -> _PreparedPacket:
    """Precompute canonical primitive dispositions, never new semantics."""

    prepared_scenarios: list[_PreparedScenario] = []
    for scenario in packet.scenarios:
        role_domain = tuple(
            loop.substantiveness.disposition for loop in scenario.loops
        )
        eligible = tuple(
            loop
            for loop in scenario.loops
            if loop.substantiveness.disposition is Disposition.PRESENT
        )
        side_by_loop = {
            loop.loop_id: {
                count: _side_count(
                    SideCountClause("clause-00", "role-00", count), loop
                ).disposition
                for count in ALLOWED_SIDE_COUNTS
            }
            for loop in eligible
        }
        obliqueness_by_loop = {
            loop.loop_id: {
                threshold: _edge_obliqueness(
                    EdgeObliquenessClause(
                        "clause-00", "role-01", threshold
                    ),
                    loop,
                ).disposition
                for threshold in ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES
            }
            for loop in eligible
        }
        contacts = {item.loop_ids: item for item in scenario.contacts}
        bindings: list[_PreparedBinding] = []
        for first in eligible:
            for second in eligible:
                if first.loop_id == second.loop_id:
                    continue
                area_ratios = {
                    ratio: _area_ratio(
                        AreaRatioClause(
                            "clause-00",
                            "role-00",
                            "role-01",
                            Rational(*ratio),
                        ),
                        first,
                        second,
                    ).disposition
                    for ratio in ALLOWED_AREA_RATIOS
                }
                contact = _point_contact(
                    PointContactClause(
                        "clause-00", "role-00", "role-01"
                    ),
                    first,
                    second,
                    contacts,
                ).disposition
                bindings.append(
                    _PreparedBinding(
                        first_side_counts=side_by_loop[first.loop_id],
                        second_side_counts=side_by_loop[second.loop_id],
                        area_ratios=area_ratios,
                        second_obliqueness=obliqueness_by_loop[second.loop_id],
                        point_contact=contact,
                    )
                )
        prepared_scenarios.append(
            _PreparedScenario(role_domain=role_domain, bindings=tuple(bindings))
        )
    return _PreparedPacket(tuple(prepared_scenarios))


def _prepared_binding_dispositions(
    binding: _PreparedBinding, plan: _QueryPlan
) -> tuple[tuple[str, Disposition], ...]:
    result: list[tuple[str, Disposition]] = [
        (
            "loop.side_count_equal",
            binding.first_side_counts[plan.numerator_side_count],
        ),
        (
            "loop.side_count_equal",
            binding.second_side_counts[plan.denominator_side_count],
        ),
        ("loop.area_ratio_at_most", binding.area_ratios[plan.ratio]),
    ]
    if plan.denominator_obliqueness_millidegrees is not None:
        result.append(
            (
                "loop.edge_obliqueness_at_least",
                binding.second_obliqueness[
                    plan.denominator_obliqueness_millidegrees
                ],
            )
        )
    if plan.require_point_contact:
        result.append(("pair.point_contact", binding.point_contact))
    return tuple(result)


def _evaluate_prepared_query(
    packet: _PreparedPacket, plan: _QueryPlan
) -> _PanelOutcome:
    scenario_dispositions: list[Disposition] = []
    scenario_clauses: list[tuple[tuple[str, Disposition], ...]] = []
    for scenario in packet.scenarios:
        clauses_by_binding = tuple(
            _prepared_binding_dispositions(binding, plan)
            for binding in scenario.bindings
        )
        binding_dispositions = tuple(
            _conjoin_dispositions(tuple(item[1] for item in clauses))
            for clauses in clauses_by_binding
        )
        scenario_dispositions.append(
            _existential_dispositions(binding_dispositions, scenario.role_domain)
        )
        scenario_clauses.append(clauses_by_binding)
    disposition = _scenario_consensus(scenario_dispositions)
    if disposition not in {Disposition.INDETERMINATE, Disposition.ERROR}:
        return _PanelOutcome(disposition, ())
    reasons: set[str] = set()
    for scenario, scenario_disposition, clauses_by_binding in zip(
        packet.scenarios,
        scenario_dispositions,
        scenario_clauses,
        strict=True,
    ):
        if scenario_disposition not in {
            Disposition.INDETERMINATE,
            Disposition.ERROR,
        }:
            continue
        reason = (
            "unresolved_binding"
            if scenario_disposition is Disposition.INDETERMINATE
            else "binding_error"
        )
        reasons.add(f"scenario:{reason}")
        for role_disposition in scenario.role_domain:
            if role_disposition in {Disposition.INDETERMINATE, Disposition.ERROR}:
                reasons.add(f"role_domain:{role_disposition.value}")
        for clauses in clauses_by_binding:
            for predicate, clause_disposition in clauses:
                if clause_disposition in {
                    Disposition.INDETERMINATE,
                    Disposition.ERROR,
                }:
                    reasons.add(
                        f"clause:{predicate}:{clause_disposition.value}"
                    )
    return _PanelOutcome(disposition, tuple(sorted(reasons)))


def _verify_prepared_semantics(
    packet: LoopScenePacket,
    prepared: _PreparedPacket,
    queries: Sequence[RelationalVisualQuery],
    plans: Sequence[_QueryPlan],
    vectorized_dispositions: Sequence[Disposition],
) -> None:
    """Model-free runtime audit against the public canonical evaluator."""

    for index in CANONICAL_EQUIVALENCE_SAMPLE_INDICES:
        canonical = evaluate_relational_query(queries[index], packet)
        expected = _PanelOutcome(
            canonical.disposition,
            _failure_reason_codes(canonical, queries[index]),
        )
        actual = _evaluate_prepared_query(prepared, plans[index])
        if (
            actual != expected
            or vectorized_dispositions[index] is not expected.disposition
        ):
            raise RelationalLibraryAblationError(
                "precomputed evaluator differs from evaluate_relational_query "
                f"at frozen library index {index}"
            )


_DISPOSITION_TO_RANK = {
    Disposition.PRESENT: 0,
    Disposition.INDETERMINATE: 1,
    Disposition.ERROR: 2,
    Disposition.CERTIFIED_ABSENT: 3,
}
_RANK_TO_DISPOSITION = {
    value: key for key, value in _DISPOSITION_TO_RANK.items()
}


def _evaluate_prepared_library(
    packet: _PreparedPacket,
    plans: Sequence[_QueryPlan],
    *,
    chunk_size: int = 256,
) -> tuple[Disposition, ...]:
    """Vectorize the exact reducer over bindings without changing semantics."""

    count = len(plans)
    numerator_index = np.asarray(
        [ALLOWED_SIDE_COUNTS.index(plan.numerator_side_count) for plan in plans],
        dtype=np.intp,
    )
    denominator_index = np.asarray(
        [ALLOWED_SIDE_COUNTS.index(plan.denominator_side_count) for plan in plans],
        dtype=np.intp,
    )
    ratio_index = np.asarray(
        [ALLOWED_AREA_RATIOS.index(plan.ratio) for plan in plans],
        dtype=np.intp,
    )
    obliqueness_index = np.asarray(
        [
            -1
            if plan.denominator_obliqueness_millidegrees is None
            else ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES.index(
                plan.denominator_obliqueness_millidegrees
            )
            for plan in plans
        ],
        dtype=np.intp,
    )
    contact_required = np.asarray(
        [plan.require_point_contact for plan in plans], dtype=bool
    )
    scenario_results: list[np.ndarray] = []
    for scenario in packet.scenarios:
        role_error = Disposition.ERROR in scenario.role_domain
        role_indeterminate = Disposition.INDETERMINATE in scenario.role_domain
        binding_count = len(scenario.bindings)
        result = np.empty(count, dtype=np.uint8)
        if binding_count == 0:
            if role_error:
                result.fill(_DISPOSITION_TO_RANK[Disposition.ERROR])
            elif role_indeterminate:
                result.fill(_DISPOSITION_TO_RANK[Disposition.INDETERMINATE])
            else:
                result.fill(_DISPOSITION_TO_RANK[Disposition.CERTIFIED_ABSENT])
            scenario_results.append(result)
            continue
        first_side = np.asarray(
            [
                [
                    _DISPOSITION_TO_RANK[binding.first_side_counts[value]]
                    for value in ALLOWED_SIDE_COUNTS
                ]
                for binding in scenario.bindings
            ],
            dtype=np.uint8,
        )
        second_side = np.asarray(
            [
                [
                    _DISPOSITION_TO_RANK[binding.second_side_counts[value]]
                    for value in ALLOWED_SIDE_COUNTS
                ]
                for binding in scenario.bindings
            ],
            dtype=np.uint8,
        )
        area = np.asarray(
            [
                [
                    _DISPOSITION_TO_RANK[binding.area_ratios[value]]
                    for value in ALLOWED_AREA_RATIOS
                ]
                for binding in scenario.bindings
            ],
            dtype=np.uint8,
        )
        obliqueness = np.asarray(
            [
                [
                    _DISPOSITION_TO_RANK[binding.second_obliqueness[value]]
                    for value in ALLOWED_OBLIQUENESS_THRESHOLDS_MILLIDEGREES
                ]
                for binding in scenario.bindings
            ],
            dtype=np.uint8,
        )
        contact = np.asarray(
            [
                _DISPOSITION_TO_RANK[binding.point_contact]
                for binding in scenario.bindings
            ],
            dtype=np.uint8,
        )
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            query_slice = slice(start, stop)
            conjunction = np.maximum(
                first_side[:, numerator_index[query_slice]],
                second_side[:, denominator_index[query_slice]],
            )
            np.maximum(
                conjunction,
                area[:, ratio_index[query_slice]],
                out=conjunction,
            )
            oblique_indices = obliqueness_index[query_slice]
            oblique_values = obliqueness[:, np.maximum(oblique_indices, 0)]
            oblique_values[:, oblique_indices < 0] = 0
            np.maximum(conjunction, oblique_values, out=conjunction)
            contact_values = np.where(
                contact_required[query_slice][None, :],
                contact[:, None],
                np.uint8(0),
            )
            np.maximum(conjunction, contact_values, out=conjunction)
            any_present = np.any(conjunction == 0, axis=0)
            any_error = np.any(conjunction == 2, axis=0)
            any_indeterminate = np.any(conjunction == 1, axis=0)
            chunk_result = np.full(
                stop - start,
                _DISPOSITION_TO_RANK[Disposition.CERTIFIED_ABSENT],
                dtype=np.uint8,
            )
            chunk_result[(~any_present) & (any_error | role_error)] = (
                _DISPOSITION_TO_RANK[Disposition.ERROR]
            )
            chunk_result[
                (~any_present)
                & (~(any_error | role_error))
                & (any_indeterminate | role_indeterminate)
            ] = _DISPOSITION_TO_RANK[Disposition.INDETERMINATE]
            chunk_result[any_present] = _DISPOSITION_TO_RANK[Disposition.PRESENT]
            result[query_slice] = chunk_result
        scenario_results.append(result)
    stacked = np.stack(scenario_results, axis=0)
    panel_result = np.full(
        count,
        _DISPOSITION_TO_RANK[Disposition.INDETERMINATE],
        dtype=np.uint8,
    )
    all_present = np.all(stacked == 0, axis=0)
    all_absent = np.all(stacked == 3, axis=0)
    any_error = np.any(stacked == 2, axis=0)
    panel_result[all_present] = _DISPOSITION_TO_RANK[Disposition.PRESENT]
    panel_result[all_absent] = _DISPOSITION_TO_RANK[Disposition.CERTIFIED_ABSENT]
    panel_result[(~all_present) & (~all_absent) & any_error] = (
        _DISPOSITION_TO_RANK[Disposition.ERROR]
    )
    return tuple(_RANK_TO_DISPOSITION[int(value)] for value in panel_result)


def _profile(
    panels: Sequence[Mapping[str, Any]], outcomes: Sequence[_PanelOutcome]
) -> dict[str, dict[str, int]]:
    result = {
        polarity: {disposition.value: 0 for disposition in _DISPOSITIONS}
        for polarity in ("positive", "negative")
    }
    for panel, outcome in zip(panels, outcomes, strict=True):
        result[panel["polarity"]][outcome.disposition.value] += 1
    return result


def _profile_key(counts: Mapping[str, int]) -> str:
    return ",".join(f"{item.value}={counts[item.value]}" for item in _DISPOSITIONS)


def _best_score(
    profile: Mapping[str, Mapping[str, int]]
) -> tuple[int, ...]:
    correct = (
        profile["positive"][Disposition.PRESENT.value]
        + profile["negative"][Disposition.CERTIFIED_ABSENT.value]
    )
    errors = sum(profile[side][Disposition.ERROR.value] for side in profile)
    indeterminate = sum(
        profile[side][Disposition.INDETERMINATE.value] for side in profile
    )
    wrong = (
        profile["positive"][Disposition.CERTIFIED_ABSENT.value]
        + profile["negative"][Disposition.PRESENT.value]
    )
    positive = profile["positive"]
    negative = profile["negative"]
    # The suffix makes score equality imply an identical confusion/disposition
    # profile, so one reported profile never stands in for unlike tied profiles.
    return (
        correct,
        -errors,
        -indeterminate,
        -wrong,
        positive[Disposition.PRESENT.value],
        negative[Disposition.CERTIFIED_ABSENT.value],
        -positive[Disposition.ERROR.value],
        -negative[Disposition.ERROR.value],
        -positive[Disposition.INDETERMINATE.value],
        -negative[Disposition.INDETERMINATE.value],
        -positive[Disposition.CERTIFIED_ABSENT.value],
        -negative[Disposition.PRESENT.value],
    )


def _task_ablation(
    task: _SelectedTask,
    packet_by_panel: Mapping[str, LoopScenePacket | None],
    prepared_by_packet: Mapping[str, _PreparedPacket],
    library_dispositions_by_packet: Mapping[str, tuple[Disposition, ...]],
    extraction_receipts: Sequence[Mapping[str, Any]],
    plans: Sequence[_QueryPlan],
    query_digests: Sequence[str],
) -> dict[str, Any]:
    panels = task.panels
    receipt_by_panel = {item["panel_id"]: item for item in extraction_receipts}
    packet_digest_by_panel = {
        panel["panel_id"]: (
            None
            if packet_by_panel[panel["panel_id"]] is None
            else packet_by_panel[panel["panel_id"]].digest()
        )
        for panel in panels
    }
    all_outcomes: list[tuple[_PanelOutcome, ...]] = []
    profiles: list[dict[str, dict[str, int]]] = []
    if len(plans) != len(query_digests):
        raise RelationalLibraryAblationError("query plan/digest inventory differs")
    for query_index in range(len(query_digests)):
        query_outcomes: list[_PanelOutcome] = []
        for panel in panels:
            packet = packet_by_panel[panel["panel_id"]]
            if packet is None:
                receipt = receipt_by_panel[panel["panel_id"]]
                query_outcomes.append(
                    _PanelOutcome(
                        Disposition.ERROR,
                        (f"extractor:{receipt['error_type']}",),
                    )
                )
                continue
            packet_digest = packet_digest_by_panel[panel["panel_id"]]
            if packet_digest is None:  # pragma: no cover - packet branch above
                raise RelationalLibraryAblationError("packet digest is unavailable")
            query_outcomes.append(
                _PanelOutcome(
                    library_dispositions_by_packet[packet_digest][query_index],
                    (),
                )
            )
        outcome_tuple = tuple(query_outcomes)
        all_outcomes.append(outcome_tuple)
        profiles.append(_profile(panels, outcome_tuple))

    positive_positions = tuple(
        index for index, panel in enumerate(panels) if panel["polarity"] == "positive"
    )
    negative_positions = tuple(
        index for index, panel in enumerate(panels) if panel["polarity"] == "negative"
    )
    if len(positive_positions) != 7 or len(negative_positions) != 7:
        raise RelationalLibraryAblationError("task panel inventory is not 7+7")

    def forward_exact(
        outcomes: Sequence[_PanelOutcome],
        positives: Sequence[int],
        negatives: Sequence[int],
    ) -> bool:
        return all(
            outcomes[index].disposition is Disposition.PRESENT for index in positives
        ) and all(
            outcomes[index].disposition is Disposition.CERTIFIED_ABSENT
            for index in negatives
        )

    full_exact_indices = tuple(
        query_index
        for query_index, outcomes in enumerate(all_outcomes)
        if forward_exact(outcomes, positive_positions, negative_positions)
    )
    all_positive_indices = tuple(
        query_index
        for query_index, outcomes in enumerate(all_outcomes)
        if all(
            outcomes[index].disposition is Disposition.PRESENT
            for index in positive_positions
        )
    )
    negative_profiles: Counter[str] = Counter()
    for query_index in all_positive_indices:
        counts = {item.value: 0 for item in _DISPOSITIONS}
        for index in negative_positions:
            counts[all_outcomes[query_index][index].disposition.value] += 1
        negative_profiles[_profile_key(counts)] += 1

    folds: list[dict[str, Any]] = []
    for omitted_index in range(7):
        fit_positives = tuple(
            position
            for position in positive_positions
            if panels[position]["index"] != omitted_index
        )
        fit_negatives = tuple(
            position
            for position in negative_positions
            if panels[position]["index"] != omitted_index
        )
        heldout_positive = next(
            position
            for position in positive_positions
            if panels[position]["index"] == omitted_index
        )
        heldout_negative = next(
            position
            for position in negative_positions
            if panels[position]["index"] == omitted_index
        )
        separators = tuple(
            query_index
            for query_index, outcomes in enumerate(all_outcomes)
            if forward_exact(outcomes, fit_positives, fit_negatives)
        )
        heldout_profiles: Counter[str] = Counter()
        generalizing: list[int] = []
        for query_index in separators:
            outcomes = all_outcomes[query_index]
            positive_disposition = outcomes[heldout_positive].disposition
            negative_disposition = outcomes[heldout_negative].disposition
            heldout_profiles[
                positive_disposition.value + "/" + negative_disposition.value
            ] += 1
            if (
                positive_disposition is Disposition.PRESENT
                and negative_disposition is Disposition.CERTIFIED_ABSENT
            ):
                generalizing.append(query_index)
        folds.append(
            {
                "omitted_index_per_side": omitted_index,
                "fit_panel_count": 12,
                "heldout_panel_ids": [
                    panels[heldout_positive]["panel_id"],
                    panels[heldout_negative]["panel_id"],
                ],
                "fit_exact_forward_separator_count": len(separators),
                "fit_exact_forward_separator_query_digests": [
                    query_digests[index] for index in separators
                ],
                "heldout_disposition_profile_counts": {
                    key: heldout_profiles[key] for key in sorted(heldout_profiles)
                },
                "heldout_forward_correct_query_count": len(generalizing),
                "heldout_forward_correct_query_digests": [
                    query_digests[index] for index in generalizing
                ],
                "any_fit_separator_is_heldout_forward_correct": bool(generalizing),
            }
        )

    scores = tuple(_best_score(profile) for profile in profiles)
    best_score = max(scores)
    best_indices = tuple(
        index for index, score in enumerate(scores) if score == best_score
    )
    representative_index = min(best_indices, key=lambda index: query_digests[index])
    representative_outcomes_list: list[_PanelOutcome] = []
    for panel_index, panel in enumerate(panels):
        packet = packet_by_panel[panel["panel_id"]]
        if packet is None:
            representative_outcomes_list.append(
                all_outcomes[representative_index][panel_index]
            )
            continue
        packet_digest = packet_digest_by_panel[panel["panel_id"]]
        if packet_digest is None:  # pragma: no cover
            raise RelationalLibraryAblationError("packet digest is unavailable")
        representative_outcomes_list.append(
            _evaluate_prepared_query(
                prepared_by_packet[packet_digest],
                plans[representative_index],
            )
        )
    representative_outcomes = tuple(representative_outcomes_list)
    unresolved_frequency: list[dict[str, Any]] = []
    for panel_index, panel in enumerate(panels):
        disposition_counts: Counter[str] = Counter()
        for query_index in best_indices:
            outcome = all_outcomes[query_index][panel_index]
            if outcome.disposition in {Disposition.INDETERMINATE, Disposition.ERROR}:
                disposition_counts[outcome.disposition.value] += 1
        if disposition_counts:
            unresolved_frequency.append(
                {
                    "panel_id": panel["panel_id"],
                    "disposition_counts_across_best_queries": {
                        key: disposition_counts[key]
                        for key in sorted(disposition_counts)
                    },
                    "reason_detail_policy": (
                        "exact reason codes are reported for the deterministic "
                        "representative best query; this frequency is disposition-only"
                    ),
                }
            )
    representative_panel_outcomes = [
        {
            "panel_id": panel["panel_id"],
            "polarity": panel["polarity"],
            "index": panel["index"],
            "disposition": outcome.disposition.value,
            "failure_reason_codes": list(outcome.reason_codes),
        }
        for panel, outcome in zip(panels, representative_outcomes, strict=True)
    ]
    extraction_failures = [
        dict(item) for item in extraction_receipts if item["status"] == "error"
    ]
    return {
        "task_id": task.task_id,
        "split": task.split,
        "family": task.family,
        "generator": task.generator,
        "panel_count": 14,
        "extraction_replay": {
            "receipts": list(extraction_receipts),
            "failure_count": len(extraction_failures),
            "failures": extraction_failures,
        },
        "full_7_plus_7_resubstitution": {
            "exact_forward_separator_count": len(full_exact_indices),
            "exact_forward_separator_query_digests": [
                query_digests[index] for index in full_exact_indices
            ],
            "all_positive_present_query_count": len(all_positive_indices),
            "all_positive_present_query_digests": [
                query_digests[index] for index in all_positive_indices
            ],
            "negative_disposition_profiles_among_all_positive_queries": {
                key: negative_profiles[key] for key in sorted(negative_profiles)
            },
        },
        "paired_leave_one_index_out": {
            "qualification": (
                "all seven deterministic folds are reported; each library member "
                "is fit-checked on the remaining 6+6 and evaluated on the two "
                "omitted panels; no fold is selected as a headline"
            ),
            "folds": folds,
        },
        "best_honest_forward_profile": {
            "ranking": [
                "maximize positive PRESENT plus negative CERTIFIED_ABSENT",
                "then minimize ERROR",
                "then minimize INDETERMINATE",
                "then minimize forward-wrong resolved outcomes",
                "then fixed positive/negative component ordering so tied queries share one exact profile",
                "never reverse polarity or negate a candidate",
            ],
            "score": list(best_score),
            "profile": profiles[best_indices[0]],
            "query_count": len(best_indices),
            "query_digests": [query_digests[index] for index in best_indices],
            "representative_query_digest": query_digests[representative_index],
            "representative_panel_outcomes": representative_panel_outcomes,
            "unresolved_panel_frequency_across_all_best_queries": (
                unresolved_frequency
            ),
        },
    }


class _Aggregate:
    def __init__(self) -> None:
        self.tasks = 0
        self.full_exact = 0
        self.tasks_full_exact = 0
        self.all_positive = 0
        self.tasks_all_positive = 0
        self.folds = 0
        self.folds_with_fit = 0
        self.folds_with_generalizer = 0
        self.fit_separators = 0
        self.generalizers = 0
        self.extractor_failures = 0
        self.best_correct_histogram: Counter[str] = Counter()
        self.best_indeterminate_histogram: Counter[str] = Counter()
        self.best_error_histogram: Counter[str] = Counter()

    def record(self, task: Mapping[str, Any]) -> None:
        self.tasks += 1
        full = task["full_7_plus_7_resubstitution"]
        exact = full["exact_forward_separator_count"]
        all_positive = full["all_positive_present_query_count"]
        self.full_exact += exact
        self.tasks_full_exact += int(exact > 0)
        self.all_positive += all_positive
        self.tasks_all_positive += int(all_positive > 0)
        self.extractor_failures += task["extraction_replay"]["failure_count"]
        for fold in task["paired_leave_one_index_out"]["folds"]:
            self.folds += 1
            fit = fold["fit_exact_forward_separator_count"]
            generalizing = fold["heldout_forward_correct_query_count"]
            self.fit_separators += fit
            self.generalizers += generalizing
            self.folds_with_fit += int(fit > 0)
            self.folds_with_generalizer += int(generalizing > 0)
        profile = task["best_honest_forward_profile"]["profile"]
        correct = (
            profile["positive"][Disposition.PRESENT.value]
            + profile["negative"][Disposition.CERTIFIED_ABSENT.value]
        )
        indeterminate = sum(
            profile[side][Disposition.INDETERMINATE.value] for side in profile
        )
        error = sum(profile[side][Disposition.ERROR.value] for side in profile)
        self.best_correct_histogram[str(correct)] += 1
        self.best_indeterminate_histogram[str(indeterminate)] += 1
        self.best_error_histogram[str(error)] += 1

    def to_data(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "full_7_plus_7_exact_forward_separators": self.full_exact,
            "tasks_with_any_full_7_plus_7_exact_forward_separator": (
                self.tasks_full_exact
            ),
            "all_positive_present_queries": self.all_positive,
            "tasks_with_any_all_positive_present_query": self.tasks_all_positive,
            "paired_leave_one_out_folds": self.folds,
            "folds_with_any_fit_separator": self.folds_with_fit,
            "folds_with_any_heldout_forward_correct_separator": (
                self.folds_with_generalizer
            ),
            "fit_separator_occurrences_across_folds": self.fit_separators,
            "heldout_forward_correct_separator_occurrences_across_folds": (
                self.generalizers
            ),
            "extractor_failure_panels": self.extractor_failures,
            "best_forward_correct_panel_histogram": dict(
                sorted(self.best_correct_histogram.items())
            ),
            "best_indeterminate_panel_histogram": dict(
                sorted(self.best_indeterminate_histogram.items())
            ),
            "best_error_panel_histogram": dict(
                sorted(self.best_error_histogram.items())
            ),
        }


def _aggregates(task_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    global_aggregate = _Aggregate()
    by_split: dict[str, _Aggregate] = defaultdict(_Aggregate)
    by_family: dict[str, _Aggregate] = defaultdict(_Aggregate)
    by_generator: dict[str, _Aggregate] = defaultdict(_Aggregate)
    for task in task_results:
        global_aggregate.record(task)
        by_split[task["split"]].record(task)
        by_family[task["family"]].record(task)
        by_generator[f"{task['family']}/{task['generator']}"].record(task)
    return {
        "global": global_aggregate.to_data(),
        "by_split": {key: by_split[key].to_data() for key in sorted(by_split)},
        "by_family": {key: by_family[key].to_data() for key in sorted(by_family)},
        "by_generator": {
            key: by_generator[key].to_data() for key in sorted(by_generator)
        },
    }


def _write_once_durable(path: Path, payload: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError:
        if destination.read_bytes() != payload:
            raise RelationalLibraryAblationError(
                f"refusing to overwrite different artifact at {destination}"
            )
    else:
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RelationalLibraryAblationError(
                        f"short write to {destination}"
                    )
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if destination.read_bytes() != payload:
        raise RelationalLibraryAblationError(
            f"durable artifact verification failed: {destination}"
        )
    return destination


@dataclass(frozen=True, slots=True)
class RelationalLibraryAblationResult:
    report: Mapping[str, Any]
    report_path: Path


def run_relational_library_ablation(
    *,
    coverage_report_path: str | Path,
    exposure_successor_path: str | Path,
    corpus_root: str | Path,
    output_store: str | Path,
    png_reader: Callable[[Path], bytes] = _read_png_no_follow,
    extractor: Callable[[bytes], LoopScenePacket] = extract_loop_scene_witnesses,
    extraction_workers: int = DEFAULT_EXTRACTION_WORKERS,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> RelationalLibraryAblationResult:
    """Replay selected pixels and evaluate the complete forward v3 library."""

    coverage_path = Path(coverage_report_path).expanduser().resolve()
    successor_path = Path(exposure_successor_path).expanduser().resolve()
    report = _load_coverage_report(coverage_path)
    successor = _load_exposure_successor(successor_path)
    validated = _validate_coverage_inputs(
        report,
        successor,
        report_path=coverage_path,
        successor_path=successor_path,
    )
    root = Path(corpus_root).expanduser().resolve()
    if not root.is_dir():
        raise RelationalLibraryAblationError(
            f"corpus root is not a directory: {root}"
        )
    if (
        isinstance(extraction_workers, bool)
        or not isinstance(extraction_workers, int)
        or not 1 <= extraction_workers <= 16
    ):
        raise RelationalLibraryAblationError(
            "extraction_workers must be an integer from 1 through 16"
        )
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")

    # Freeze and authenticate the whole finite library before resolving a
    # selected task directory or opening a selected PNG.
    queries = enumerate_factorized_shape_ratio_queries()
    if len(queries) != QUERY_LIBRARY_SIZE:
        raise RelationalLibraryAblationError(
            f"complete v3 query library has {len(queries)}, expected {QUERY_LIBRARY_SIZE}"
        )
    query_digests = tuple(query.digest() for query in queries)
    if len(set(query_digests)) != QUERY_LIBRARY_SIZE:
        raise RelationalLibraryAblationError(
            "complete v3 query library contains duplicate query digests"
        )
    query_algorithm_digest = relational_query_algorithm_digest()
    if any(query.algorithm_digest != query_algorithm_digest for query in queries):
        raise RelationalLibraryAblationError(
            "query library contains a stale evaluator identity"
        )
    library_inventory_digest = _address(list(query_digests))
    plans = tuple(_query_plan(query) for query in queries)
    current_extractors = _current_extractor_identities()
    algorithm_identities = {
        "ablation_algorithm_id": ALGORITHM_ID,
        "ablation_python_source_digest": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        **current_extractors,
        "relational_query_algorithm_id": RELATIONAL_QUERY_ALGORITHM_ID,
        "relational_query_algorithm_digest": query_algorithm_digest,
        "relational_query_python_source_digest": relational_query_source_digest(),
        "query_library_size": QUERY_LIBRARY_SIZE,
        "query_library_inventory_digest": library_inventory_digest,
        "selected_png_reextraction_workers": extraction_workers,
        "prepared_evaluator_semantics": (
            "primitive clause dispositions are produced by the canonical v3 "
            "clause evaluators; conjunction, existential, and scenario consensus "
            "preserve evaluate_relational_query ordering"
        ),
        "canonical_equivalence_sample_indices": list(
            CANONICAL_EQUIVALENCE_SAMPLE_INDICES
        ),
    }
    restrictions = {
        **_RESTRICTIONS,
        "new_exposure_event_created": False,
        "selected_manifest_only_png_replay": True,
        "polarity_flip_authorized": False,
        "negation_rescue_authorized": False,
        "official_benchmark_or_generalization_claim_authorized": False,
    }
    input_commitment = {
        "schema": "gkm.bongard-relational-library-ablation-input.v1",
        "coverage_output_digest": report["output_digest"],
        "coverage_selection_digest": validated.selection_digest,
        "selected_png_manifest_digest": validated.selected_manifest_digest,
        "exposure_successor_digest": successor.digest,
        "source_corpus_manifest_digest": validated.source_corpus_manifest_digest,
        "algorithm_identities": algorithm_identities,
        "restrictions": restrictions,
    }
    input_digest = _address(input_commitment)

    jobs = tuple(
        (task, panel) for task in validated.tasks for panel in task.panels
    )

    def replay_selected_panel(
        job: tuple[_SelectedTask, Mapping[str, Any]],
    ) -> tuple[str, str, LoopScenePacket | None, dict[str, Any]]:
        task, panel = job
        path = _selected_panel_path(root, task, panel)
        payload = png_reader(path)
        if not isinstance(payload, bytes):
            raise RelationalLibraryAblationError("PNG reader returned non-bytes")
        panel_address = _bytes_address(payload)
        if (
            panel_address != panel["sha256"]
            or len(payload) != panel["size_bytes"]
        ):
            raise RelationalLibraryAblationError(
                f"selected PNG bytes differ from manifest: {panel['panel_id']}"
            )
        old_receipt = validated.receipts[panel["panel_id"]]
        try:
            packet = extractor(payload)
            if not isinstance(packet, LoopScenePacket):
                raise TypeError("extractor did not return LoopScenePacket")
            _assert_packet_identity(packet, panel_address)
        except Exception as exc:
            if old_receipt["status"] == "present":
                raise RelationalLibraryAblationError(
                    "current extraction failed for a panel whose authenticated "
                    f"coverage receipt was present: {panel['panel_id']}; "
                    f"{type(exc).__module__}.{type(exc).__qualname__}"
                ) from exc
            return (
                task.task_id,
                panel["panel_id"],
                None,
                {
                    "panel_id": panel["panel_id"],
                    "png_sha256": panel_address,
                    "status": "error",
                    "error_type": old_receipt["error_type"],
                    "packet_digest": None,
                    "coverage_receipt_status": "error",
                    "coverage_packet_digest_match": None,
                    "current_reextract_status": "error",
                    "current_reextract_error_type": type(exc).__module__
                    + "."
                    + type(exc).__qualname__,
                },
            )
        packet_digest = packet.digest()
        if old_receipt["status"] == "error":
            return (
                task.task_id,
                panel["panel_id"],
                None,
                {
                    "panel_id": panel["panel_id"],
                    "png_sha256": panel_address,
                    "status": "error",
                    "error_type": old_receipt["error_type"],
                    "packet_digest": None,
                    "coverage_receipt_status": "error",
                    "coverage_packet_digest_match": None,
                    "current_reextract_status": "present_but_noncomparable",
                    "current_reextract_error_type": None,
                },
            )
        if old_receipt["loop_scene_packet_digest"] != packet_digest:
            raise RelationalLibraryAblationError(
                "current packet digest differs from the authenticated "
                f"coverage receipt: {panel['panel_id']}"
            )
        return (
            task.task_id,
            panel["panel_id"],
            packet,
            {
                "panel_id": panel["panel_id"],
                "png_sha256": panel_address,
                "status": "present",
                "error_type": None,
                "packet_digest": packet_digest,
                "coverage_receipt_status": "present",
                "coverage_packet_digest_match": True,
                "current_reextract_status": "present",
                "current_reextract_error_type": None,
            },
        )

    packet_by_panel: dict[str, LoopScenePacket | None] = {}
    extraction_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with ThreadPoolExecutor(
        max_workers=extraction_workers,
        thread_name_prefix="bongard-selected-reextract",
    ) as executor:
        for completed, result in enumerate(
            executor.map(replay_selected_panel, jobs), start=1
        ):
            task_id, panel_id, packet, receipt = result
            packet_by_panel[panel_id] = packet
            extraction_by_task[task_id].append(receipt)
            if progress_callback is not None:
                progress_callback("selected-png-reextraction", completed, len(jobs))

    prepared_by_packet: dict[str, _PreparedPacket] = {}
    library_dispositions_by_packet: dict[str, tuple[Disposition, ...]] = {}
    unique_packet_total = len(
        {
            receipt["packet_digest"]
            for receipts in extraction_by_task.values()
            for receipt in receipts
            if receipt["packet_digest"] is not None
        }
    )
    for packet in packet_by_panel.values():
        if packet is None:
            continue
        packet_digest = packet.digest()
        if packet_digest in prepared_by_packet:
            continue
        prepared = _prepare_packet(packet)
        library_dispositions = _evaluate_prepared_library(prepared, plans)
        _verify_prepared_semantics(
            packet,
            prepared,
            queries,
            plans,
            library_dispositions,
        )
        prepared_by_packet[packet_digest] = prepared
        library_dispositions_by_packet[packet_digest] = library_dispositions
        if progress_callback is not None:
            progress_callback(
                "finite-library-packet-evaluation",
                len(prepared_by_packet),
                unique_packet_total,
            )
    task_results = tuple(
        _task_ablation(
            task,
            packet_by_panel,
            prepared_by_packet,
            library_dispositions_by_packet,
            extraction_by_task[task.task_id],
            plans,
            query_digests,
        )
        for task in validated.tasks
    )
    if progress_callback is not None:
        progress_callback("report-construction", len(task_results), len(task_results))
    report_content: dict[str, Any] = {
        "schema": SCHEMA,
        "algorithm_id": ALGORITHM_ID,
        "input_digest": input_digest,
        "source": {
            "coverage_output_digest": report["output_digest"],
            "coverage_selection_digest": validated.selection_digest,
            "selected_png_manifest_digest": validated.selected_manifest_digest,
            "exposure_successor_digest": successor.digest,
            "source_corpus_manifest_digest": validated.source_corpus_manifest_digest,
        },
        "qualification": {
            "evaluation_kind": "resubstitution/library-coverage",
            "benchmark_or_generalization_result": False,
            "engineering_panel_protocol": "all already-exposed 7+7 panels",
            "candidate_inventory": "complete frozen finite Python v3 library",
            "candidate_pixel_access": False,
            "orientation": (
                "positive panels require PRESENT; negative panels require "
                "CERTIFIED_ABSENT; no polarity reversal"
            ),
            "coverage_selected_task_count": len(validated.tasks),
            "downstream_exposure_delta": 0,
            "inventory_caveat": (
                "a coverage sampler's protected-ID reserve is not proof that "
                "other selected tasks avoid semantic-key collisions with DEV; "
                "recompute exact-unused and strict-DEV capacity from the durable "
                "successor before any later evaluation"
            ),
        },
        "restrictions": restrictions,
        "algorithm_identities": algorithm_identities,
        "query_library": {
            "count": QUERY_LIBRARY_SIZE,
            "inventory_digest": library_inventory_digest,
            "query_algorithm_digest": query_algorithm_digest,
            "unique_reextracted_packet_count": len(prepared_by_packet),
            "canonical_equivalence_checks_per_unique_packet": len(
                CANONICAL_EQUIVALENCE_SAMPLE_INDICES
            ),
            "canonical_equivalence_check_count": (
                len(prepared_by_packet)
                * len(CANONICAL_EQUIVALENCE_SAMPLE_INDICES)
            ),
        },
        "tasks": list(task_results),
        "aggregates": _aggregates(task_results),
    }
    output: dict[str, Any] = {
        **report_content,
        "output_digest": _address(report_content),
    }
    output_path = _write_once_durable(
        Path(output_store)
        / (output["output_digest"].removeprefix("sha256:") + ".ablation.json"),
        canonical_json(output) + b"\n",
    )
    cold = _json_object_no_duplicates(output_path.read_bytes(), "ablation output")
    if cold != output or _address({k: v for k, v in cold.items() if k != "output_digest"}) != cold["output_digest"]:
        raise RelationalLibraryAblationError(
            "write-once ablation output failed cold digest replay"
        )
    return RelationalLibraryAblationResult(output, output_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the complete positive relational v3 library on an "
            "already-exposed selected-only coverage manifest."
        )
    )
    parser.add_argument("--coverage-report", required=True, type=Path)
    parser.add_argument("--exposure-successor", required=True, type=Path)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--output-store", required=True, type=Path)
    parser.add_argument(
        "--extraction-workers", type=int, default=DEFAULT_EXTRACTION_WORKERS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    def progress(phase: str, completed: int, total: int) -> None:
        if completed == total or completed == 1 or completed % 14 == 0:
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "completed": completed,
                        "total": total,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    result = run_relational_library_ablation(
        coverage_report_path=args.coverage_report,
        exposure_successor_path=args.exposure_successor,
        corpus_root=args.corpus_root,
        output_store=args.output_store,
        extraction_workers=args.extraction_workers,
        progress_callback=progress,
    )
    print(
        json.dumps(
            {
                "output_digest": result.report["output_digest"],
                "report_path": str(result.report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALGORITHM_ID",
    "QUERY_LIBRARY_SIZE",
    "RelationalLibraryAblationError",
    "RelationalLibraryAblationResult",
    "SCHEMA",
    "main",
    "run_relational_library_ablation",
]
