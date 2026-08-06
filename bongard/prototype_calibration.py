"""Development-only calibration for neutral support-prototype margins.

The public entry point accepts an already discovered corpus, an explicit list
of development task IDs, a seed, and a finite positive margin grid.  It does
not discover a split, call a model, synthesize code, write an artifact, or use
Lean.  The seed selects the same one-positive/one-negative holdout indices as
the official benchmark protocol.  All fourteen panels are first measured by
the candidate-independent neutral extractor; group projection and support
fitting happen only afterwards.

For each task and each closed feature group, six positive and six negative
support packets fit the two centroids.  A strict support pass requires all six
positive packets to be PRESENT and all six negative packets to be
CERTIFIED_ABSENT under the fixed affirmative orientation.  If any required
support extraction is non-PRESENT, that task/group is retained in every
denominator with an explicit ``unfittable_support`` record, null fit identities,
and ERROR predicate outcomes; no query result is fabricated.  The two held-out
development labels are used only for margin calibration.  Abstentions and
errors count as wrong, exactly as in the benchmark runner.

Selection is lexicographic by development image correctness, puzzle
correctness, strict support-pass count, then smallest margin.  Since all
candidate rows have the same denominators, integer counts implement the
stated rates without floating-point ranking ambiguity.  Under a symmetric
abstention band this generally makes the smallest candidate weakly dominate;
the complete grid is retained so that fact is visible rather than hidden.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import bongard.support_prototypes as _prototype_implementation
from bongard.benchmark import PROTOCOL_VERSION
from bongard.corpus import ShapeBongardCorpus
from bongard.evidence import Disposition, Evidence
from bongard.legs.neutral_features import (
    FEATURE_GROUP_IDS,
    NeutralFeatureExtraction,
    extract_neutral_features,
    feature_group_catalog,
    feature_group_catalog_digest,
    feature_space_for_group,
    project_neutral_feature_extraction,
    verify_neutral_feature_extraction,
)
from bongard.support_prototypes import (
    FrozenFeatureSpace,
    FrozenPanelFeatures,
    PositivePrototypeFormula,
    SupportPrototypePlan,
    evaluate_frozen_support_member,
    evaluate_support_prototype,
    fit_support_prototypes,
    panel_side_assignment_digest,
)


CALIBRATION_SCHEMA = "bongard.support-prototype-development-calibration/v2"
TASK_SELECTION_SCHEMA = "bongard.support-prototype-development-selection/v1"
TASK_PLAN_SCHEMA = "bongard.support-prototype-development-task-plan/v1"
LABEL_AUTHORIZATION = "explicit-caller-supplied-development-task-ids-only"
SELECTION_OBJECTIVE = (
    "max-development-image-correct",
    "max-development-puzzle-correct",
    "max-strict-support-pass-tasks",
    "min-positive-margin",
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_DISPOSITIONS = tuple(item.value for item in Disposition)
_UNFITTABLE_REASON = "not_evaluated_due_to_unfittable_support"


class PrototypeCalibrationError(ValueError):
    """Development calibration input or evidence violates the protocol."""


class PrototypeCalibrationIntegrityError(PrototypeCalibrationError):
    """A serialized calibration record differs from its commitment."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_fields(
    data: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if not isinstance(data, Mapping) or set(data) != expected:
        raise PrototypeCalibrationIntegrityError(
            f"{label} fields differ from calibration schema"
        )


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PrototypeCalibrationIntegrityError(
            f"{label} must be a non-empty exact string"
        )
    return value


def _require_digest(value: object, label: str) -> str:
    text = _require_text(value, label)
    if _DIGEST.fullmatch(text) is None:
        raise PrototypeCalibrationIntegrityError(
            f"{label} must be a lowercase sha256"
        )
    return text


def _bare_digest(value: str, label: str) -> str:
    text = value.removeprefix("sha256:")
    if _DIGEST.fullmatch(text) is None:
        raise PrototypeCalibrationError(f"{label} is not a SHA-256 digest")
    return text


def _positive_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrototypeCalibrationIntegrityError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PrototypeCalibrationIntegrityError(
            f"{label} must be finite and strictly positive"
        )
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrototypeCalibrationIntegrityError(
            f"{label} must be a non-negative integer"
        )
    return value


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise PrototypeCalibrationIntegrityError(f"{label} must be a JSON list")
    return value


def _source_digest(path: Path) -> str:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PrototypeCalibrationError(f"source changed while hashing: {path}")
    return hashlib.sha256(payload).hexdigest()


def _implementation_source_digests() -> dict[str, str]:
    prototype_path = Path(_prototype_implementation.__file__).resolve()
    calibration_path = Path(__file__).resolve()
    neutral_sources = {
        feature_space_for_group(group_id).extractor_artifact_digest
        for group_id in FEATURE_GROUP_IDS
    }
    if len(neutral_sources) != 1:
        raise PrototypeCalibrationError(
            "neutral feature groups do not share one extractor source"
        )
    return {
        "bongard.legs.neutral_features": next(iter(neutral_sources)),
        "bongard.prototype_calibration": _source_digest(calibration_path),
        "bongard.support_prototypes": _source_digest(prototype_path),
    }


def _derive_hex(domain: str, *parts: str) -> str:
    digest = hashlib.sha256()
    digest.update((PROTOCOL_VERSION + "\0" + domain).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def _query_index(task_id: str, seed: str, side: str) -> int:
    return int(
        _derive_hex("selection:" + side + "-query", task_id, seed), 16
    ) % 7


def _normalise_task_ids(task_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(task_ids, (str, bytes)) or not isinstance(task_ids, Sequence):
        raise TypeError("development task IDs must be an explicit sequence")
    result = tuple(task_ids)
    if not result:
        raise PrototypeCalibrationError("development task IDs cannot be empty")
    if any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in result
    ):
        raise PrototypeCalibrationError(
            "development task IDs must be non-empty exact strings"
        )
    if len(result) != len(set(result)):
        raise PrototypeCalibrationError("development task IDs contain duplicates")
    return tuple(sorted(result))


def _normalise_margins(candidate_margins: Sequence[float]) -> tuple[float, ...]:
    if isinstance(candidate_margins, (str, bytes)) or not isinstance(
        candidate_margins, Sequence
    ):
        raise TypeError("candidate margins must be an explicit sequence")
    result = tuple(
        _positive_real(value, "candidate margin") for value in candidate_margins
    )
    if not result:
        raise PrototypeCalibrationError("candidate margin grid cannot be empty")
    if len(result) != len(set(result)):
        raise PrototypeCalibrationError("candidate margin grid contains duplicates")
    return tuple(sorted(result))


def _disposition_counts() -> Counter[str]:
    return Counter({name: 0 for name in _DISPOSITIONS})


def _is_positive_alignment(evidence: Evidence[bool]) -> bool:
    return (
        evidence.disposition is Disposition.PRESENT
        and evidence.unwrap() is True
    )


def _is_negative_alignment(evidence: Evidence[bool]) -> bool:
    return evidence.disposition is Disposition.CERTIFIED_ABSENT


def _selection_key(row: Mapping[str, Any]) -> tuple[int, int, int, float]:
    query = row["development_query"]
    support = row["support"]
    return (
        query["correct_image_count"],
        query["correct_puzzle_count"],
        support["strict_pass_task_count"],
        -float(row["margin"]),
    )


def _validate_dispositions(
    value: object, *, expected_total: int, label: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_DISPOSITIONS):
        raise PrototypeCalibrationIntegrityError(
            f"{label} must contain all four dispositions"
        )
    counts = tuple(
        _nonnegative_int(value[name], f"{label}.{name}") for name in _DISPOSITIONS
    )
    if sum(counts) != expected_total:
        raise PrototypeCalibrationIntegrityError(
            f"{label} disposition total differs from panel count"
        )


def _validate_candidate_row(
    row: object, *, expected_margin: float, task_count: int
) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise PrototypeCalibrationIntegrityError("candidate row must be an object")
    _require_fields(
        row,
        frozenset({"margin", "task_count", "support", "development_query"}),
        "candidate row",
    )
    margin = _positive_real(row["margin"], "candidate row margin")
    if margin != expected_margin:
        raise PrototypeCalibrationIntegrityError(
            "candidate rows differ from the frozen margin grid"
        )
    if _nonnegative_int(row["task_count"], "candidate task_count") != task_count:
        raise PrototypeCalibrationIntegrityError("candidate task count drift")

    support = row["support"]
    if not isinstance(support, Mapping):
        raise PrototypeCalibrationIntegrityError("support counts must be an object")
    _require_fields(
        support,
        frozenset(
            {
                "panel_count",
                "positive_aligned_count",
                "negative_aligned_count",
                "strict_pass_task_count",
                "strict_pass_rate",
                "not_evaluated_due_to_unfittable_support_count",
                "dispositions",
            }
        ),
        "support counts",
    )
    support_panels = _nonnegative_int(support["panel_count"], "support panel_count")
    if support_panels != 12 * task_count:
        raise PrototypeCalibrationIntegrityError("support panel count drift")
    positive = _nonnegative_int(
        support["positive_aligned_count"], "positive support alignment count"
    )
    negative = _nonnegative_int(
        support["negative_aligned_count"], "negative support alignment count"
    )
    passed = _nonnegative_int(
        support["strict_pass_task_count"], "strict support pass task count"
    )
    support_unfittable = _nonnegative_int(
        support["not_evaluated_due_to_unfittable_support_count"],
        "unfittable support observation count",
    )
    if positive > 6 * task_count or negative > 6 * task_count or passed > task_count:
        raise PrototypeCalibrationIntegrityError("support alignment count overflow")
    if support_unfittable > support_panels:
        raise PrototypeCalibrationIntegrityError(
            "unfittable support observation count overflow"
        )
    if support["strict_pass_rate"] != [passed, task_count]:
        raise PrototypeCalibrationIntegrityError("strict support rate/count mismatch")
    _validate_dispositions(
        support["dispositions"], expected_total=support_panels, label="support"
    )
    if support["dispositions"][Disposition.ERROR.value] < support_unfittable:
        raise PrototypeCalibrationIntegrityError(
            "unfittable support observations are not counted as ERROR"
        )

    query = row["development_query"]
    if not isinstance(query, Mapping):
        raise PrototypeCalibrationIntegrityError("query counts must be an object")
    _require_fields(
        query,
        frozenset(
            {
                "image_count",
                "correct_image_count",
                "image_accuracy",
                "puzzle_count",
                "correct_puzzle_count",
                "puzzle_accuracy",
                "not_evaluated_due_to_unfittable_support_count",
                "dispositions",
            }
        ),
        "development query counts",
    )
    images = _nonnegative_int(query["image_count"], "query image_count")
    correct_images = _nonnegative_int(
        query["correct_image_count"], "query correct_image_count"
    )
    puzzles = _nonnegative_int(query["puzzle_count"], "query puzzle_count")
    correct_puzzles = _nonnegative_int(
        query["correct_puzzle_count"], "query correct_puzzle_count"
    )
    query_unfittable = _nonnegative_int(
        query["not_evaluated_due_to_unfittable_support_count"],
        "unfittable query observation count",
    )
    if images != 2 * task_count or puzzles != task_count:
        raise PrototypeCalibrationIntegrityError("development query count drift")
    if correct_images > images or correct_puzzles > puzzles:
        raise PrototypeCalibrationIntegrityError("development correctness overflow")
    if query_unfittable > images:
        raise PrototypeCalibrationIntegrityError(
            "unfittable query observation count overflow"
        )
    if query["image_accuracy"] != [correct_images, images]:
        raise PrototypeCalibrationIntegrityError("image accuracy/count mismatch")
    if query["puzzle_accuracy"] != [correct_puzzles, puzzles]:
        raise PrototypeCalibrationIntegrityError("puzzle accuracy/count mismatch")
    _validate_dispositions(
        query["dispositions"], expected_total=images, label="development query"
    )
    if query["dispositions"][Disposition.ERROR.value] < query_unfittable:
        raise PrototypeCalibrationIntegrityError(
            "unfittable query observations are not counted as ERROR"
        )
    return row


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, label)


def _validate_extraction_records(
    value: object,
    *,
    label: str,
    expected_positive_indices: tuple[int, ...],
    expected_negative_indices: tuple[int, ...],
) -> tuple[Mapping[str, Any], ...]:
    records = _sequence(value, label)
    expected_keys = tuple(
        [("positive", index) for index in expected_positive_indices]
        + [("negative", index) for index in expected_negative_indices]
    )
    if len(records) != len(expected_keys):
        raise PrototypeCalibrationIntegrityError(f"{label} panel count drift")
    parsed: list[Mapping[str, Any]] = []
    for raw, expected_key in zip(records, expected_keys, strict=True):
        if not isinstance(raw, Mapping):
            raise PrototypeCalibrationIntegrityError(
                f"{label} extraction must be an object"
            )
        _require_fields(
            raw,
            frozenset(
                {
                    "side",
                    "index",
                    "panel_digest",
                    "disposition",
                    "reason",
                    "certificate",
                    "error_type",
                    "receipt_digest",
                }
            ),
            f"{label} extraction",
        )
        side = raw["side"]
        index = _nonnegative_int(raw["index"], f"{label} panel index")
        if (side, index) != expected_key:
            raise PrototypeCalibrationIntegrityError(
                f"{label} extraction order or identity drift"
            )
        _require_digest(raw["panel_digest"], f"{label} panel digest")
        _require_digest(raw["receipt_digest"], f"{label} receipt digest")
        disposition = raw["disposition"]
        if disposition not in _DISPOSITIONS:
            raise PrototypeCalibrationIntegrityError(
                f"{label} has an unknown extraction disposition"
            )
        reason = _optional_text(raw["reason"], f"{label} reason")
        certificate = _optional_text(
            raw["certificate"], f"{label} certificate"
        )
        error_type = _optional_text(raw["error_type"], f"{label} error type")
        expected_optional = {
            Disposition.PRESENT.value: (False, False, False),
            Disposition.CERTIFIED_ABSENT.value: (False, True, False),
            Disposition.INDETERMINATE.value: (True, False, False),
            Disposition.ERROR.value: (True, False, True),
        }[disposition]
        if (
            reason is not None,
            certificate is not None,
            error_type is not None,
        ) != expected_optional:
            raise PrototypeCalibrationIntegrityError(
                f"{label} optional fields disagree with its disposition"
            )
        parsed.append(raw)
    return tuple(parsed)


def _validate_content(content: Mapping[str, Any]) -> None:
    _require_fields(
        content,
        frozenset(
            {
                "schema",
                "selection_protocol",
                "label_boundary",
                "seed",
                "seed_digest",
                "task_ids",
                "task_plan_digest",
                "candidate_margin_grid",
                "feature_catalog",
                "source_digests",
                "selection_objective",
                "tasks",
                "groups",
            }
        ),
        "calibration record",
    )
    if content["schema"] != CALIBRATION_SCHEMA:
        raise PrototypeCalibrationIntegrityError("unsupported calibration schema")
    if content["selection_protocol"] != PROTOCOL_VERSION:
        raise PrototypeCalibrationIntegrityError("unknown query-selection protocol")
    seed = _require_text(content["seed"], "calibration seed")
    if content["seed_digest"] != _digest(
        {"schema": TASK_SELECTION_SCHEMA, "seed": seed}
    ):
        raise PrototypeCalibrationIntegrityError("calibration seed digest drift")

    boundary = content["label_boundary"]
    if boundary != {
        "authorization": LABEL_AUTHORIZATION,
        "official_test_tasks_rejected": True,
        "query_labels_used_for": "development-margin-calibration-only",
    }:
        raise PrototypeCalibrationIntegrityError("development label boundary drift")
    if content["selection_objective"] != list(SELECTION_OBJECTIVE):
        raise PrototypeCalibrationIntegrityError("margin selection objective drift")

    raw_task_ids = _sequence(content["task_ids"], "task_ids")
    if any(not isinstance(item, str) for item in raw_task_ids):
        raise PrototypeCalibrationIntegrityError("task IDs must be strings")
    task_ids = tuple(raw_task_ids)
    if not task_ids or task_ids != tuple(sorted(set(task_ids))):
        raise PrototypeCalibrationIntegrityError(
            "task IDs must be non-empty, unique, and sorted"
        )
    raw_margins = _sequence(content["candidate_margin_grid"], "margin grid")
    margins = tuple(
        _positive_real(item, "candidate margin") for item in raw_margins
    )
    if not margins or margins != tuple(sorted(set(margins))):
        raise PrototypeCalibrationIntegrityError(
            "candidate margins must be non-empty, unique, and sorted"
        )

    catalog = content["feature_catalog"]
    if not isinstance(catalog, Mapping):
        raise PrototypeCalibrationIntegrityError("feature catalog must be an object")
    _require_fields(catalog, frozenset({"digest", "groups"}), "feature catalog")
    catalog_groups = _sequence(catalog["groups"], "feature catalog groups")
    expected_catalog_digest = _digest(
        {
            "schema": "bongard.neutral-feature-group-catalog/v1",
            "groups": catalog_groups,
        }
    )
    if _require_digest(catalog["digest"], "feature catalog digest") != (
        expected_catalog_digest
    ):
        raise PrototypeCalibrationIntegrityError("feature catalog digest drift")
    catalog_ids = tuple(
        item.get("group_id") if isinstance(item, Mapping) else None
        for item in catalog_groups
    )
    if catalog_ids != FEATURE_GROUP_IDS:
        raise PrototypeCalibrationIntegrityError(
            "calibration does not cover the complete frozen feature catalog"
        )

    sources = content["source_digests"]
    if not isinstance(sources, Mapping) or set(sources) != {
        "bongard.legs.neutral_features",
        "bongard.prototype_calibration",
        "bongard.support_prototypes",
    }:
        raise PrototypeCalibrationIntegrityError("implementation source set drift")
    for name, value in sources.items():
        _require_digest(value, f"source digest {name}")

    raw_groups = _sequence(content["groups"], "calibration groups")
    if len(raw_groups) != len(FEATURE_GROUP_IDS):
        raise PrototypeCalibrationIntegrityError("calibration group count drift")
    validated_groups: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for expected_group_id, raw_group in zip(
        FEATURE_GROUP_IDS, raw_groups, strict=True
    ):
        if not isinstance(raw_group, Mapping):
            raise PrototypeCalibrationIntegrityError(
                "calibration group must be an object"
            )
        _require_fields(
            raw_group,
            frozenset(
                {
                    "group_id",
                    "feature_space",
                    "feature_space_digest",
                    "candidate_counts",
                    "selected_margin",
                }
            ),
            "calibration group",
        )
        if raw_group["group_id"] != expected_group_id:
            raise PrototypeCalibrationIntegrityError("calibration group order drift")
        if not isinstance(raw_group["feature_space"], Mapping):
            raise PrototypeCalibrationIntegrityError("feature space must be an object")
        space = FrozenFeatureSpace.from_data(raw_group["feature_space"])
        if _require_digest(
            raw_group["feature_space_digest"], "feature space digest"
        ) != space.digest():
            raise PrototypeCalibrationIntegrityError("feature space digest drift")
        rows = _sequence(raw_group["candidate_counts"], "candidate counts")
        if len(rows) != len(margins):
            raise PrototypeCalibrationIntegrityError("candidate count grid drift")
        validated = tuple(
            _validate_candidate_row(
                row, expected_margin=margin, task_count=len(task_ids)
            )
            for row, margin in zip(rows, margins, strict=True)
        )
        validated_groups[expected_group_id] = validated
        selected = _positive_real(
            raw_group["selected_margin"], "selected margin"
        )
        if selected != max(validated, key=_selection_key)["margin"]:
            raise PrototypeCalibrationIntegrityError(
                "selected margin differs from the frozen objective"
            )

    raw_tasks = _sequence(content["tasks"], "task bindings")
    if tuple(
        item.get("task_id") if isinstance(item, Mapping) else None
        for item in raw_tasks
    ) != task_ids:
        raise PrototypeCalibrationIntegrityError("task binding order drift")
    unfittable_by_group = Counter({group_id: 0 for group_id in FEATURE_GROUP_IDS})
    for task in raw_tasks:
        assert isinstance(task, Mapping)
        _require_fields(
            task,
            frozenset(
                {
                    "task_id",
                    "family",
                    "declared_split",
                    "task_source_digest",
                    "positive_query_index",
                    "negative_query_index",
                    "selection_digest",
                    "groups",
                }
            ),
            "task binding",
        )
        task_id = _require_text(task["task_id"], "task binding ID")
        _require_text(task["family"], "task family")
        if task["declared_split"] not in (None, "train", "val"):
            raise PrototypeCalibrationIntegrityError(
                "calibration record contains a non-development split"
            )
        source_digest = _require_digest(
            task["task_source_digest"], "task source digest"
        )
        positive_query = _nonnegative_int(
            task["positive_query_index"], "positive query index"
        )
        negative_query = _nonnegative_int(
            task["negative_query_index"], "negative query index"
        )
        if positive_query >= 7 or negative_query >= 7:
            raise PrototypeCalibrationIntegrityError("query index is outside 7+7")
        expected_selection = _digest(
            {
                "schema": TASK_SELECTION_SCHEMA,
                "selection_protocol": PROTOCOL_VERSION,
                "seed_digest": content["seed_digest"],
                "task_id": task_id,
                "task_source_digest": source_digest,
                "positive_query_index": positive_query,
                "negative_query_index": negative_query,
            }
        )
        if _require_digest(task["selection_digest"], "selection digest") != (
            expected_selection
        ):
            raise PrototypeCalibrationIntegrityError("task selection digest drift")
        bindings = _sequence(task["groups"], "task group bindings")
        if tuple(
            item.get("group_id") if isinstance(item, Mapping) else None
            for item in bindings
        ) != FEATURE_GROUP_IDS:
            raise PrototypeCalibrationIntegrityError("task group bindings drift")
        for binding in bindings:
            assert isinstance(binding, Mapping)
            _require_fields(
                binding,
                frozenset(
                    {
                        "group_id",
                        "status",
                        "non_evaluation_reason",
                        "fit_plan_digest",
                        "prototype_digest",
                        "support_extractions",
                        "query_extractions",
                    }
                ),
                "task group binding",
            )
            group_id = binding["group_id"]
            support_indices_positive = tuple(
                index for index in range(7) if index != positive_query
            )
            support_indices_negative = tuple(
                index for index in range(7) if index != negative_query
            )
            support_extractions = _validate_extraction_records(
                binding["support_extractions"],
                label="support",
                expected_positive_indices=support_indices_positive,
                expected_negative_indices=support_indices_negative,
            )
            _validate_extraction_records(
                binding["query_extractions"],
                label="development query",
                expected_positive_indices=(positive_query,),
                expected_negative_indices=(negative_query,),
            )
            status = binding["status"]
            support_all_present = all(
                item["disposition"] == Disposition.PRESENT.value
                for item in support_extractions
            )
            if status == "fitted":
                if binding["non_evaluation_reason"] is not None:
                    raise PrototypeCalibrationIntegrityError(
                        "fitted task/group has a non-evaluation reason"
                    )
                _require_digest(binding["fit_plan_digest"], "fit plan digest")
                _require_digest(binding["prototype_digest"], "prototype digest")
                if not support_all_present:
                    raise PrototypeCalibrationIntegrityError(
                        "fitted task/group contains non-PRESENT support extraction"
                    )
            elif status == "unfittable_support":
                if binding["non_evaluation_reason"] != _UNFITTABLE_REASON:
                    raise PrototypeCalibrationIntegrityError(
                        "unfittable support reason drift"
                    )
                if binding["fit_plan_digest"] is not None or (
                    binding["prototype_digest"] is not None
                ):
                    raise PrototypeCalibrationIntegrityError(
                        "unfittable support must have null fit identities"
                    )
                if support_all_present:
                    raise PrototypeCalibrationIntegrityError(
                        "unfittable support has no non-PRESENT extraction"
                    )
                unfittable_by_group[group_id] += 1
            else:
                raise PrototypeCalibrationIntegrityError(
                    "unknown task/group calibration status"
                )

    for group_id, rows in validated_groups.items():
        expected_tasks = unfittable_by_group[group_id]
        for row in rows:
            support = row["support"]
            query = row["development_query"]
            if support[
                "not_evaluated_due_to_unfittable_support_count"
            ] != 12 * expected_tasks:
                raise PrototypeCalibrationIntegrityError(
                    "unfittable support aggregate differs from task bindings"
                )
            if query[
                "not_evaluated_due_to_unfittable_support_count"
            ] != 2 * expected_tasks:
                raise PrototypeCalibrationIntegrityError(
                    "unfittable query aggregate differs from task bindings"
                )

    expected_plan = _digest(
        {
            "schema": TASK_PLAN_SCHEMA,
            "seed_digest": content["seed_digest"],
            "tasks": raw_tasks,
        }
    )
    if _require_digest(content["task_plan_digest"], "task plan digest") != (
        expected_plan
    ):
        raise PrototypeCalibrationIntegrityError("task plan digest drift")


@dataclass(frozen=True)
class PrototypeCalibrationRecord:
    """Immutable canonical-JSON calibration record with a content digest."""

    _content_json: bytes

    def __post_init__(self) -> None:
        if not isinstance(self._content_json, bytes):
            raise TypeError("calibration content must be canonical JSON bytes")
        try:
            content = json.loads(self._content_json)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PrototypeCalibrationIntegrityError(
                "calibration content is not JSON"
            ) from exc
        if not isinstance(content, Mapping):
            raise PrototypeCalibrationIntegrityError(
                "calibration content must be a JSON object"
            )
        if _canonical_json(content) != self._content_json:
            raise PrototypeCalibrationIntegrityError(
                "calibration content is not canonical JSON"
            )
        _validate_content(content)

    @classmethod
    def create(cls, content: Mapping[str, Any]) -> "PrototypeCalibrationRecord":
        return cls(_canonical_json(content))

    @classmethod
    def from_data(
        cls, data: Mapping[str, Any]
    ) -> "PrototypeCalibrationRecord":
        if not isinstance(data, Mapping):
            raise TypeError("calibration record must be a JSON object")
        if "record_digest" not in data:
            raise PrototypeCalibrationIntegrityError(
                "calibration record has no digest"
            )
        content = dict(data)
        claimed = _require_digest(
            content.pop("record_digest"), "calibration record digest"
        )
        record = cls.create(content)
        if record.digest() != claimed:
            raise PrototypeCalibrationIntegrityError(
                "calibration record digest drift"
            )
        return record

    def content_data(self) -> dict[str, Any]:
        return json.loads(self._content_json)

    def digest(self) -> str:
        return hashlib.sha256(self._content_json).hexdigest()

    def to_data(self) -> dict[str, Any]:
        return {**self.content_data(), "record_digest": self.digest()}

    def canonical_json(self) -> bytes:
        """Return canonical JSON including the non-self-referential digest."""

        return _canonical_json(self.to_data())

    def selected_margin(self, group_id: str) -> float:
        for group in self.content_data()["groups"]:
            if group["group_id"] == group_id:
                return float(group["selected_margin"])
        raise KeyError(f"unknown calibrated feature group: {group_id}")

    def to_freeze_policy(self):
        """Derive the current runner policy from the calibrated commitments.

        Parsing a historical record verifies its internal content identity.
        Policy derivation additionally requires that the runtime neutral
        catalog and every runtime feature space still equal the calibrated
        records.  The lower-level policy intentionally does not duplicate the
        calibration digest; an outer run artifact binds both objects.
        """

        from bongard.prototype_artifacts import PrototypeFreezePolicy

        data = self.content_data()
        if data["feature_catalog"]["digest"] != feature_group_catalog_digest():
            raise PrototypeCalibrationIntegrityError(
                "current neutral feature catalog differs from calibration"
            )
        allowed_groups: dict[str, tuple[FrozenFeatureSpace, float]] = {}
        for calibrated in data["groups"]:
            group_id = calibrated["group_id"]
            archived_space = FrozenFeatureSpace.from_data(
                calibrated["feature_space"]
            )
            current_space = feature_space_for_group(group_id)
            if archived_space != current_space or (
                calibrated["feature_space_digest"] != current_space.digest()
            ):
                raise PrototypeCalibrationIntegrityError(
                    f"current feature space differs for {group_id}"
                )
            allowed_groups[group_id] = (
                current_space,
                float(calibrated["selected_margin"]),
            )
        policy = PrototypeFreezePolicy.create(
            feature_catalog_digest=data["feature_catalog"]["digest"],
            allowed_groups=allowed_groups,
        )
        for group_id, (space, margin) in allowed_groups.items():
            selected = policy.select(group_id, space)
            if selected.decision_margin != margin:
                raise PrototypeCalibrationIntegrityError(
                    f"derived policy margin differs for {group_id}"
                )
        return policy


def _new_accumulator(task_count: int) -> dict[str, Any]:
    return {
        "task_count": task_count,
        "support_positive": 0,
        "support_negative": 0,
        "support_pass": 0,
        "support_unfittable": 0,
        "support_dispositions": _disposition_counts(),
        "query_correct": 0,
        "query_puzzle_correct": 0,
        "query_unfittable": 0,
        "query_dispositions": _disposition_counts(),
    }


def _candidate_row(margin: float, accumulator: Mapping[str, Any]) -> dict[str, Any]:
    task_count = accumulator["task_count"]
    support_panels = 12 * task_count
    query_images = 2 * task_count
    return {
        "margin": margin,
        "task_count": task_count,
        "support": {
            "panel_count": support_panels,
            "positive_aligned_count": accumulator["support_positive"],
            "negative_aligned_count": accumulator["support_negative"],
            "strict_pass_task_count": accumulator["support_pass"],
            "strict_pass_rate": [accumulator["support_pass"], task_count],
            "not_evaluated_due_to_unfittable_support_count": accumulator[
                "support_unfittable"
            ],
            "dispositions": {
                name: accumulator["support_dispositions"][name]
                for name in _DISPOSITIONS
            },
        },
        "development_query": {
            "image_count": query_images,
            "correct_image_count": accumulator["query_correct"],
            "image_accuracy": [accumulator["query_correct"], query_images],
            "puzzle_count": task_count,
            "correct_puzzle_count": accumulator["query_puzzle_correct"],
            "puzzle_accuracy": [
                accumulator["query_puzzle_correct"],
                task_count,
            ],
            "not_evaluated_due_to_unfittable_support_count": accumulator[
                "query_unfittable"
            ],
            "dispositions": {
                name: accumulator["query_dispositions"][name]
                for name in _DISPOSITIONS
            },
        },
    }


def _extraction_record(
    side: str,
    index: int,
    extraction: NeutralFeatureExtraction,
) -> dict[str, object]:
    evidence = extraction.evidence
    return {
        "side": side,
        "index": index,
        "panel_digest": extraction.receipt.input_identity.sha256,
        "disposition": evidence.disposition.value,
        "reason": evidence.reason,
        "certificate": evidence.certificate,
        "error_type": evidence.error_type,
        "receipt_digest": extraction.receipt.digest(),
    }


def calibrate_prototype_margins(
    corpus: ShapeBongardCorpus,
    development_task_ids: Sequence[str],
    *,
    seed: str,
    candidate_margins: Sequence[float],
) -> PrototypeCalibrationRecord:
    """Calibrate one fixed positive decision margin per neutral group.

    The caller, not this function, defines the development cohort.  When the
    supplied corpus has official split metadata, any task declared ``test`` is
    rejected before a panel is read.  Unassigned tasks remain admissible only
    under the explicit caller-development authorization recorded in the
    result.
    """

    if not isinstance(corpus, ShapeBongardCorpus):
        raise TypeError("calibration requires a ShapeBongardCorpus")
    if not isinstance(seed, str) or not seed or seed != seed.strip():
        raise PrototypeCalibrationError(
            "calibration seed must be a non-empty exact string"
        )
    task_ids = _normalise_task_ids(development_task_ids)
    margins = _normalise_margins(candidate_margins)
    for task_id in task_ids:
        assignment = corpus.assignment(task_id)
        if assignment.split == "test":
            raise PrototypeCalibrationError(
                f"official test task is forbidden in development calibration: {task_id}"
            )

    catalog = feature_group_catalog()
    if tuple(item.group_id for item in catalog) != FEATURE_GROUP_IDS:
        raise PrototypeCalibrationError("neutral feature catalog order drift")
    catalog_data = [item.to_data() for item in catalog]
    if feature_group_catalog_digest() != _digest(
        {
            "schema": "bongard.neutral-feature-group-catalog/v1",
            "groups": catalog_data,
        }
    ):
        raise PrototypeCalibrationError("neutral feature catalog digest drift")

    spaces = {
        group_id: feature_space_for_group(group_id)
        for group_id in FEATURE_GROUP_IDS
    }
    seed_digest = _digest({"schema": TASK_SELECTION_SCHEMA, "seed": seed})
    accumulators = {
        group_id: {
            margin: _new_accumulator(len(task_ids)) for margin in margins
        }
        for group_id in FEATURE_GROUP_IDS
    }
    task_bindings: list[dict[str, Any]] = []

    for task_id in task_ids:
        task = corpus.task(task_id)
        if len(task.positive) != 7 or len(task.negative) != 7:
            raise PrototypeCalibrationError(
                f"development task is not exact 7+7: {task_id}"
            )
        task_manifest = task.build_manifest()
        task_source_digest = _bare_digest(
            task_manifest.digest, f"task manifest {task_id}"
        )
        manifests = {
            (panel.polarity, panel.index): panel for panel in task_manifest.panels
        }
        if len(manifests) != 14:
            raise PrototypeCalibrationError(
                f"task manifest is not exact 7+7: {task_id}"
            )

        panel_bytes: dict[tuple[str, int], bytes] = {}
        full_extractions: dict[tuple[str, int], NeutralFeatureExtraction] = {}
        for side, paths in (("positive", task.positive), ("negative", task.negative)):
            for index, path in enumerate(paths):
                payload = path.read_bytes()
                expected = _bare_digest(
                    manifests[(side, index)].sha256,
                    f"panel manifest {task_id}/{side}/{index}",
                )
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise PrototypeCalibrationError(
                        f"panel bytes changed after task manifest: {task_id}/{side}/{index}"
                    )
                panel_bytes[(side, index)] = payload
                extraction = extract_neutral_features(payload)
                verify_neutral_feature_extraction(extraction, payload)
                full_extractions[(side, index)] = extraction

        positive_query_index = _query_index(task_id, seed, "positive")
        negative_query_index = _query_index(task_id, seed, "negative")
        selection_data = {
            "schema": TASK_SELECTION_SCHEMA,
            "selection_protocol": PROTOCOL_VERSION,
            "seed_digest": seed_digest,
            "task_id": task_id,
            "task_source_digest": task_source_digest,
            "positive_query_index": positive_query_index,
            "negative_query_index": negative_query_index,
        }
        group_bindings: list[dict[str, Any]] = []

        for group_id in FEATURE_GROUP_IDS:
            space = spaces[group_id]
            projected_extractions: dict[
                tuple[str, int], NeutralFeatureExtraction
            ] = {}
            packets: dict[tuple[str, int], FrozenPanelFeatures] = {}
            for key, extraction in full_extractions.items():
                projected = project_neutral_feature_extraction(extraction, group_id)
                packet = verify_neutral_feature_extraction(
                    projected, panel_bytes[key]
                )
                projected_extractions[key] = projected
                if packet is not None:
                    packets[key] = packet

            positive_support_keys = tuple(
                ("positive", index)
                for index in range(7)
                if index != positive_query_index
            )
            negative_support_keys = tuple(
                ("negative", index)
                for index in range(7)
                if index != negative_query_index
            )
            support_keys = positive_support_keys + negative_support_keys
            query_keys = (
                ("positive", positive_query_index),
                ("negative", negative_query_index),
            )
            support_extractions = [
                _extraction_record(side, index, projected_extractions[(side, index)])
                for side, index in support_keys
            ]
            query_extractions = [
                _extraction_record(side, index, projected_extractions[(side, index)])
                for side, index in query_keys
            ]
            support_fittable = all(
                projected_extractions[key].evidence.disposition
                is Disposition.PRESENT
                for key in support_keys
            )
            if not support_fittable:
                group_bindings.append(
                    {
                        "group_id": group_id,
                        "status": "unfittable_support",
                        "non_evaluation_reason": _UNFITTABLE_REASON,
                        "fit_plan_digest": None,
                        "prototype_digest": None,
                        "support_extractions": support_extractions,
                        "query_extractions": query_extractions,
                    }
                )
                for margin in margins:
                    accumulator = accumulators[group_id][margin]
                    accumulator["support_unfittable"] += 12
                    accumulator["support_dispositions"][
                        Disposition.ERROR.value
                    ] += 12
                    accumulator["query_unfittable"] += 2
                    accumulator["query_dispositions"][
                        Disposition.ERROR.value
                    ] += 2
                continue

            positive_support = tuple(
                packets[key] for key in positive_support_keys
            )
            negative_support = tuple(
                packets[key] for key in negative_support_keys
            )
            plan = SupportPrototypePlan(
                feature_space_digest=space.digest(),
                support_assignment_digest=panel_side_assignment_digest(
                    tuple(item.panel_digest for item in positive_support),
                    tuple(item.panel_digest for item in negative_support),
                ),
                minimum_per_side=6,
            )
            prototype = fit_support_prototypes(
                plan,
                space,
                positive_support,
                negative_support,
                expected_plan_digest=plan.digest(),
            )
            group_bindings.append(
                {
                    "group_id": group_id,
                    "status": "fitted",
                    "non_evaluation_reason": None,
                    "fit_plan_digest": plan.digest(),
                    "prototype_digest": prototype.digest(),
                    "support_extractions": support_extractions,
                    "query_extractions": query_extractions,
                }
            )

            for margin in margins:
                formula = PositivePrototypeFormula(
                    claim="the panel matches the frozen positive support prototype",
                    feature_space_digest=space.digest(),
                    prototype_digest=prototype.digest(),
                    support_assignment_digest=prototype.support_assignment_digest,
                    decision_margin=margin,
                )
                accumulator = accumulators[group_id][margin]
                positive_evidence = tuple(
                    evaluate_frozen_support_member(
                        formula, prototype, space, packet
                    )
                    for packet in positive_support
                )
                negative_evidence = tuple(
                    evaluate_frozen_support_member(
                        formula, prototype, space, packet
                    )
                    for packet in negative_support
                )
                for evidence in positive_evidence + negative_evidence:
                    accumulator["support_dispositions"][
                        evidence.disposition.value
                    ] += 1
                positive_correct = sum(
                    _is_positive_alignment(item) for item in positive_evidence
                )
                negative_correct = sum(
                    _is_negative_alignment(item) for item in negative_evidence
                )
                accumulator["support_positive"] += positive_correct
                accumulator["support_negative"] += negative_correct
                if positive_correct == 6 and negative_correct == 6:
                    accumulator["support_pass"] += 1

                query_correctness: list[bool] = []
                for key, positive_label in zip(
                    query_keys, (True, False), strict=True
                ):
                    extraction = projected_extractions[key]
                    if extraction.evidence.disposition is Disposition.PRESENT:
                        query_evidence = evaluate_support_prototype(
                            formula,
                            prototype,
                            space,
                            packets[key],
                        )
                        disposition = query_evidence.disposition
                        correct = (
                            _is_positive_alignment(query_evidence)
                            if positive_label
                            else _is_negative_alignment(query_evidence)
                        )
                    else:
                        # The extraction state is preserved as-is.  It is an
                        # abstention, never reinterpreted from the dev label.
                        disposition = extraction.evidence.disposition
                        correct = False
                    accumulator["query_dispositions"][disposition.value] += 1
                    query_correctness.append(correct)
                positive_query_correct, negative_query_correct = query_correctness
                accumulator["query_correct"] += (
                    positive_query_correct + negative_query_correct
                )
                if positive_query_correct and negative_query_correct:
                    accumulator["query_puzzle_correct"] += 1

        assignment = corpus.assignment(task_id)
        task_bindings.append(
            {
                "task_id": task_id,
                "family": task.family,
                "declared_split": assignment.split,
                "task_source_digest": task_source_digest,
                "positive_query_index": positive_query_index,
                "negative_query_index": negative_query_index,
                "selection_digest": _digest(selection_data),
                "groups": group_bindings,
            }
        )

    group_records: list[dict[str, Any]] = []
    for group_id in FEATURE_GROUP_IDS:
        rows = [
            _candidate_row(margin, accumulators[group_id][margin])
            for margin in margins
        ]
        selected = max(rows, key=_selection_key)["margin"]
        space = spaces[group_id]
        group_records.append(
            {
                "group_id": group_id,
                "feature_space": space.to_data(),
                "feature_space_digest": space.digest(),
                "candidate_counts": rows,
                "selected_margin": selected,
            }
        )

    task_plan_digest = _digest(
        {
            "schema": TASK_PLAN_SCHEMA,
            "seed_digest": seed_digest,
            "tasks": task_bindings,
        }
    )
    content = {
        "schema": CALIBRATION_SCHEMA,
        "selection_protocol": PROTOCOL_VERSION,
        "label_boundary": {
            "authorization": LABEL_AUTHORIZATION,
            "official_test_tasks_rejected": True,
            "query_labels_used_for": "development-margin-calibration-only",
        },
        "seed": seed,
        "seed_digest": seed_digest,
        "task_ids": list(task_ids),
        "task_plan_digest": task_plan_digest,
        "candidate_margin_grid": list(margins),
        "feature_catalog": {
            "digest": feature_group_catalog_digest(),
            "groups": catalog_data,
        },
        "source_digests": _implementation_source_digests(),
        "selection_objective": list(SELECTION_OBJECTIVE),
        "tasks": task_bindings,
        "groups": group_records,
    }
    return PrototypeCalibrationRecord.create(content)


__all__ = (
    "CALIBRATION_SCHEMA",
    "LABEL_AUTHORIZATION",
    "SELECTION_OBJECTIVE",
    "PrototypeCalibrationError",
    "PrototypeCalibrationIntegrityError",
    "PrototypeCalibrationRecord",
    "calibrate_prototype_margins",
)
