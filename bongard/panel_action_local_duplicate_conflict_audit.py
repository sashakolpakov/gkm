"""Metadata-only duplicate-PNG versus pose-free-target conflict audit.

The fit precommit already contains exact PNG content addresses and supervised
label triples.  This module joins those stored addresses to the frozen
development action authority without opening any PNG.  A PNG digest is safe
for pose-free descriptor loss iff all of its effective occurrences have one
and only one pose-free target digest.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.panel_action_local_supervision_authority import (
    ALGORITHM_ID as LOCAL_SUPERVISION_ALGORITHM_ID,
    DEVELOPMENT_RECORD_DIGEST,
    DEVELOPMENT_SOURCE_SHA256,
    Disposition,
    compile_pose_free_panel,
    load_development_authority,
)


AUDIT_SCHEMA = "gkm.bongard-pose-free-target-duplicate-conflict-audit.v1"
TARGET_SCHEMA = "gkm.bongard-pose-free-carrier-target.v1"
ALGORITHM_ID = "exact-png-digest-by-pose-free-target-cardinality/v1"

FIT_PRECOMMIT_SCHEMA = "gkm.bongard-action-count-catalog-cnn-fit-pixel-precommit.v2"
FIT_PRECOMMIT_RECORD_DIGEST = (
    "sha256:e8c7c15fbfb723c5b2305094f035e2567c1fb9b7e80b9f13eeae32fe35d1b15a"
)
FIT_PRECOMMIT_SOURCE_SHA256 = (
    "sha256:bfc1267150bac4f823dc72fc483d64f2c32e98ace90ff63bad2556fe4d6aec97"
)
LOCAL_AUTHORITY_COMMIT = "8fd4de9a84505899cad42135b2f6305365871df6"
LOCAL_AUTHORITY_RECORD_DIGEST = (
    "sha256:ff7e5de2d4018f5c788a3d2a400ee2ed397497e9b74db29bb509847197babb95"
)
LOCAL_AUTHORITY_SOURCE_SHA256 = (
    "sha256:6b8af81b123d130e7c901ca9ea9fad92da84f38f52083568ef9f60b62b94acf4"
)

EXPECTED_ORIGINAL_COUNTS = {"train": 11_200, "validation": 1_400}
EXPECTED_EFFECTIVE_COUNTS = {"train": 11_200, "validation": 1_392}
MAX_PRECOMMIT_BYTES = 16 * 1024 * 1024
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class DuplicateTargetAuditError(RuntimeError):
    """The bound metadata or the replayed audit differs."""


def _address(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _module_source_sha256() -> str:
    return _address(Path(__file__).resolve().read_bytes())


def _stable_regular_bytes(path: Path, maximum: int) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise DuplicateTargetAuditError(f"metadata source is not regular: {path}")
    if before.st_size <= 0 or before.st_size > maximum:
        raise DuplicateTargetAuditError(f"metadata source size is outside bound: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            raw = handle.read(maximum + 1)
            after_read = os.fstat(handle.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    after = path.lstat()
    fingerprint = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )
    if not (
        fingerprint(before)
        == fingerprint(opened)
        == fingerprint(after_read)
        == fingerprint(after)
    ):
        raise DuplicateTargetAuditError(f"metadata source changed during read: {path}")
    if len(raw) != before.st_size or len(raw) > maximum:
        raise DuplicateTargetAuditError(f"metadata byte count differs: {path}")
    return raw


def _load_precommit(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _stable_regular_bytes(path, MAX_PRECOMMIT_BYTES)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DuplicateTargetAuditError(f"cannot decode fit precommit: {exc}") from exc
    if type(value) is not dict or raw != canonical_json(value) + b"\n":
        raise DuplicateTargetAuditError("fit precommit is not canonical JSON plus newline")
    body = dict(value)
    digest = body.pop("record_digest", None)
    if (
        value.get("schema") != FIT_PRECOMMIT_SCHEMA
        or digest != FIT_PRECOMMIT_RECORD_DIGEST
        or digest != "sha256:" + canonical_digest(body)
        or _address(raw) != FIT_PRECOMMIT_SOURCE_SHA256
        or value.get("v3_development_ids_record_digest") != DEVELOPMENT_RECORD_DIGEST
        or value.get("v3_development_ids_source_sha256") != DEVELOPMENT_SOURCE_SHA256
    ):
        raise DuplicateTargetAuditError("fit precommit binding differs")
    return value, raw


def _pose_free_target(supervision: Mapping[str, Any]) -> dict[str, object]:
    if supervision.get("disposition") != Disposition.CERTIFIED.value:
        raise DuplicateTargetAuditError("development pose-free supervision is a GAP")
    carrier = supervision.get("carrier_instance_count")
    shape = supervision.get("shape_instance_count")
    shape_multiset = supervision.get("shape_multiset")
    if (
        type(carrier) is not dict
        or carrier.get("disposition") != Disposition.CERTIFIED.value
        or type(carrier.get("value")) is not int
        or type(shape) is not dict
        or shape.get("disposition") != Disposition.CERTIFIED.value
        or type(shape.get("value")) is not int
        or type(shape_multiset) is not list
    ):
        raise DuplicateTargetAuditError("pose-free supervision fields differ")
    return {
        "algorithm_id": LOCAL_SUPERVISION_ALGORITHM_ID,
        "carrier_instance_count": carrier["value"],
        "schema": TARGET_SCHEMA,
        "shape_instance_count": shape["value"],
        "shape_multiset": shape_multiset,
    }


def _primitive_counts(target: Mapping[str, Any]) -> tuple[int, int]:
    line = 0
    arc = 0
    for shape in target["shape_multiset"]:
        shape_multiplicity = shape["multiplicity"]
        for action in shape["action_multiset"]:
            amount = shape_multiplicity * action["multiplicity"]
            if action["primitive"] == "line":
                line += amount
            elif action["primitive"] == "arc":
                arc += amount
            else:
                raise DuplicateTargetAuditError("target contains another primitive")
    if line + arc != target["carrier_instance_count"]:
        raise DuplicateTargetAuditError("target carrier count differs from primitive counts")
    return line, arc


def _line_digest(values: Sequence[str]) -> str:
    return _address("".join(f"{value}\n" for value in values).encode("utf-8"))


def _conflicting_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    projector: Callable[[Mapping[str, Any]], object],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for png_digest, rows in sorted(groups.items()):
        values = {canonical_json(projector(row)) for row in rows}
        if len(values) <= 1:
            continue
        result.append(
            {
                "occurrence_count": len(rows),
                "panel_ids": sorted(str(row["panel_id"]) for row in rows),
                "png_sha256": png_digest,
                "value_digests": sorted(_address(value) for value in values),
                "value_count": len(values),
            }
        )
    return result


def _conflict_summary(groups: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    return {
        "group_count": len(groups),
        "groups": list(groups),
        "groups_digest": "sha256:" + canonical_digest(list(groups)),
        "occurrence_count": sum(int(group["occurrence_count"]) for group in groups),
    }


def _summarize_cohort(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    by_png: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_png[str(row["png_sha256"])].append(row)
    duplicates = {digest: values for digest, values in by_png.items() if len(values) > 1}
    target_conflicts = _conflicting_groups(
        by_png, lambda row: str(row["pose_free_target_digest"])
    )
    straight_conflicts = _conflicting_groups(
        by_png, lambda row: int(row["label_triple"][0])
    )
    arc_conflicts = _conflicting_groups(
        by_png, lambda row: int(row["label_triple"][1])
    )
    action_pair_conflicts = _conflicting_groups(
        by_png, lambda row: list(row["label_triple"][:2])
    )
    catalog_conflicts = _conflicting_groups(
        by_png, lambda row: int(row["label_triple"][2])
    )

    eligible_index: list[dict[str, object]] = []
    eligible_panels: list[str] = []
    ineligible_occurrences = 0
    cardinalities: Counter[int] = Counter()
    for png_digest, group_rows in sorted(by_png.items()):
        targets = sorted({str(row["pose_free_target_digest"]) for row in group_rows})
        cardinalities[len(targets)] += 1
        if len(targets) != 1:
            ineligible_occurrences += len(group_rows)
            continue
        panel_ids = sorted(str(row["panel_id"]) for row in group_rows)
        eligible_panels.extend(panel_ids)
        eligible_index.append(
            {
                "group_normalized_weight_denominator": len(group_rows),
                "multiplicity": len(group_rows),
                "pose_free_target_digest": targets[0],
                "png_sha256": png_digest,
                "representative_panel_id": panel_ids[0],
            }
        )
    return {
        "action_count_conflicts": {
            "arc": _conflict_summary(arc_conflicts),
            "straight": _conflict_summary(straight_conflicts),
            "straight_arc_pair": _conflict_summary(action_pair_conflicts),
        },
        "catalog_convexity_conflicts": _conflict_summary(catalog_conflicts),
        "descriptor_loss_eligibility": {
            "criterion": (
                "eligible_iff_exact_png_digest_has_exactly_one_pose_free_target_digest_"
                "within_effective_cohort"
            ),
            "eligible_group_count": len(eligible_index),
            "eligible_index_digest": "sha256:" + canonical_digest(eligible_index),
            "eligible_occurrence_count": len(eligible_panels),
            "eligible_panel_ids_digest": _line_digest(sorted(eligible_panels)),
            "ineligible_group_count": len(by_png) - len(eligible_index),
            "ineligible_occurrence_count": ineligible_occurrences,
            "loss_weighting": (
                "one_over_png_group_multiplicity_per_occurrence_or_one_"
                "lexicographic_representative_per_group"
            ),
            "representative_panel_ids_digest": _line_digest(
                sorted(str(row["representative_panel_id"]) for row in eligible_index)
            ),
        },
        "duplicate_png_digest_group_count": len(duplicates),
        "duplicate_png_occurrence_count": sum(len(values) for values in duplicates.values()),
        "occurrence_count": len(rows),
        "png_digest_group_count": len(by_png),
        "pose_free_target_cardinality_histogram_by_png_group": {
            str(key): value for key, value in sorted(cardinalities.items())
        },
        "pose_free_target_conflicts": _conflict_summary(target_conflicts),
        "singleton_png_digest_group_count": len(by_png) - len(duplicates),
    }


def _effective_rows(
    *, repository_root: Path, precommit: Mapping[str, Any]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    authority = load_development_authority(repository_root=repository_root)
    if (
        authority.record_digest != LOCAL_AUTHORITY_RECORD_DIGEST
        or authority.to_record().get("bindings", {}).get("module_source_sha256")
        != LOCAL_AUTHORITY_SOURCE_SHA256
    ):
        raise DuplicateTargetAuditError("pose-free authority binding differs")
    observations = precommit.get("exact_png_observations")
    removed_record = precommit.get("validation_removed_due_exact_train_duplicate")
    if type(observations) is not list or type(removed_record) is not dict:
        raise DuplicateTargetAuditError("fit observation/removal records differ")
    removed = removed_record.get("panel_ids")
    if type(removed) is not list or len(removed) != 8 or len(set(removed)) != 8:
        raise DuplicateTargetAuditError("validation removal set differs")
    removed_set = set(removed)
    expected_panels = {
        panel
        for _cohort, panels in authority.cohort_panel_ids
        for panel in panels
    }
    seen: set[str] = set()
    original_counts: Counter[str] = Counter()
    effective_counts: Counter[str] = Counter()
    result: list[dict[str, object]] = []
    for source_row in observations:
        if type(source_row) is not dict:
            raise DuplicateTargetAuditError("fit observation is not an object")
        panel_id = source_row.get("panel_id")
        cohort = source_row.get("fit_cohort")
        png_digest = source_row.get("png_sha256")
        label = source_row.get("label_triple")
        if (
            type(panel_id) is not str
            or panel_id in seen
            or cohort not in ("train", "validation")
            or type(png_digest) is not str
            or _ADDRESS.fullmatch(png_digest) is None
            or type(label) is not list
            or len(label) != 3
            or any(type(value) is not int for value in label)
        ):
            raise DuplicateTargetAuditError("fit observation fields differ")
        seen.add(panel_id)
        if authority.cohort_for_panel(panel_id) != cohort:
            raise DuplicateTargetAuditError(
                f"fit cohort differs from development authority for {panel_id}"
            )
        original_counts[cohort] += 1
        if cohort == "validation" and panel_id in removed_set:
            continue
        supervision = compile_pose_free_panel(authority, panel_id).to_data()
        target = _pose_free_target(supervision)
        straight, arc = _primitive_counts(target)
        if (straight, arc) != (label[0], label[1]):
            raise DuplicateTargetAuditError(
                f"pose-free primitive counts differ from fit labels for {panel_id}"
            )
        target_digest = _address(canonical_json(target))
        result.append(
            {
                "fit_cohort": cohort,
                "label_triple": list(label),
                "panel_id": panel_id,
                "png_sha256": png_digest,
                "pose_free_target_digest": target_digest,
            }
        )
        effective_counts[cohort] += 1
    if seen != expected_panels or dict(original_counts) != EXPECTED_ORIGINAL_COUNTS:
        raise DuplicateTargetAuditError("fit observations differ from development authority")
    if dict(effective_counts) != EXPECTED_EFFECTIVE_COUNTS:
        raise DuplicateTargetAuditError("effective cohort counts differ")
    return result, dict(effective_counts)


def build_duplicate_target_conflict_audit(
    *, repository_root: str | Path | None = None
) -> dict[str, object]:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    precommit_path = root / (
        "downloads/ShapeBongard_V2_full/panel_action_count_cnn_fit_20260810_v3/"
        "fit_pixel_precommit.json"
    )
    precommit, _raw = _load_precommit(precommit_path)
    rows, effective_counts = _effective_rows(repository_root=root, precommit=precommit)
    summaries = {
        cohort: _summarize_cohort(
            [row for row in rows if row["fit_cohort"] == cohort]
        )
        for cohort in ("train", "validation")
    }
    body: dict[str, object] = {
        "algorithm_id": ALGORITHM_ID,
        "bindings": {
            "audit_source_sha256": _module_source_sha256(),
            "fit_precommit_record_digest": FIT_PRECOMMIT_RECORD_DIGEST,
            "fit_precommit_source_sha256": FIT_PRECOMMIT_SOURCE_SHA256,
            "pose_free_authority_commit": LOCAL_AUTHORITY_COMMIT,
            "pose_free_authority_record_digest": LOCAL_AUTHORITY_RECORD_DIGEST,
            "pose_free_authority_source_sha256": LOCAL_AUTHORITY_SOURCE_SHA256,
        },
        "cohorts": summaries,
        "custody": {
            "action_program_scope": "frozen_v3_train_and_validation_only",
            "calibration_evaluation_family_or_target_identifiers_opened": 0,
            "fit_precommit_exact_png_digest_metadata_read": len(
                precommit["exact_png_observations"]
            ),
            "label_source": "label_triples_already_frozen_in_fit_precommit",
            "png_bytes_read": 0,
            "pose_free_target_source": "committed_development_authority_8fd4de9a",
            "removed_validation_occurrences_not_joined": (
                EXPECTED_ORIGINAL_COUNTS["validation"]
                - effective_counts["validation"]
            ),
        },
        "effective_rows_digest": "sha256:" + canonical_digest(rows),
        "result": {
            "all_effective_png_groups_descriptor_loss_eligible": all(
                summary["descriptor_loss_eligibility"]["ineligible_group_count"] == 0
                for summary in summaries.values()
            ),
            "catalog_convexity_conflict_group_count": sum(
                summary["catalog_convexity_conflicts"]["group_count"]
                for summary in summaries.values()
            ),
            "pose_free_target_conflict_group_count": sum(
                summary["pose_free_target_conflicts"]["group_count"]
                for summary in summaries.values()
            ),
            "scope": "training_and_effective_validation_only",
            "straight_arc_pair_conflict_group_count": sum(
                summary["action_count_conflicts"]["straight_arc_pair"]["group_count"]
                for summary in summaries.values()
            ),
        },
        "schema": AUDIT_SCHEMA,
        "target_digest_semantics": {
            "excluded": [
                "panel_id",
                "cohort",
                "source_action_program_digest",
                "pixel_coordinates",
                "pixel_masks",
                "style",
                "signed_traversal_direction",
            ],
            "included": [
                "action_carrier_instance_count",
                "shape_instance_count",
                "unordered_shape_multiset",
                "per_shape_action_multiset_with_rounding_intervals",
                "per_shape_internal_junction_multiset_with_rounding_intervals",
            ],
            "schema": TARGET_SCHEMA,
        },
    }
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def verify_duplicate_target_conflict_audit(
    artifact: Mapping[str, object], *, repository_root: str | Path | None = None
) -> None:
    if type(artifact) is not dict:
        raise DuplicateTargetAuditError("audit artifact must be an exact dict")
    body = dict(artifact)
    digest = body.pop("record_digest", None)
    if digest != "sha256:" + canonical_digest(body):
        raise DuplicateTargetAuditError("audit artifact record digest differs")
    replay = build_duplicate_target_conflict_audit(repository_root=repository_root)
    if artifact != replay:
        raise DuplicateTargetAuditError("audit artifact differs from metadata-only replay")


__all__ = [
    "ALGORITHM_ID",
    "AUDIT_SCHEMA",
    "DuplicateTargetAuditError",
    "FIT_PRECOMMIT_RECORD_DIGEST",
    "FIT_PRECOMMIT_SOURCE_SHA256",
    "TARGET_SCHEMA",
    "build_duplicate_target_conflict_audit",
    "verify_duplicate_target_conflict_audit",
]
