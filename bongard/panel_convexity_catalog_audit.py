"""Metadata-only catalog labels for the HD turning/convexity observer.

The ShapeBongard archive contains action programs but not the shape-name row
used to create each HD panel.  Most rows can be recovered exactly by removing
only randomized stroke style from the serialized action program and matching
the remaining geometry/turn sequence to ``human_designed_shapes.tsv``.

Four archive signatures were produced from stale geometry.  They are not
fuzzy-matched.  A compatibility alias is admitted only when the archive's own
Basic singleton task named for one catalog row repeats that exact signature in
all seven positive panels.  The alias target must have raw convexity label
``-1``.  Raw ``-1`` is trained as ``catalog_unresolved`` and is never a
certified nonconvex/not-applicable value; a downstream calibrated set that
contains it must become a typed GAP.

This module reads TSV/JSON metadata only.  It has no image decoder and never
opens a PNG.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from collections import Counter, defaultdict
import csv
from dataclasses import InitVar, dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from bongard.canonical import canonical_digest


AUDIT_SCHEMA = "gkm.bongard-panel-convexity-catalog-audit.v1"
ALGORITHM_SCHEMA = "gkm.bongard-panel-convexity-catalog-algorithm.v1"
ALGORITHM_ID = "bongard.panel-convexity/exact-style-stripped-plus-bd-singleton-alias-v1"
RAW_LABEL_TO_CLASS: Mapping[str, str] = MappingProxyType({
    "1": "convex",
    "0": "nonconvex",
    "-1": "catalog_unresolved",
})


class ConvexityCatalogError(RuntimeError):
    """Catalog metadata cannot support a unique, fail-closed label."""


Signature = tuple[str, ...]
_CATALOG_BINDING_AUTHORITY = object()


@dataclass(frozen=True)
class CatalogBinding:
    direct_by_signature: Mapping[Signature, str]
    raw_label_by_name: Mapping[str, str]
    alias_by_signature: Mapping[Signature, str]
    alias_proofs: tuple[Mapping[str, Any], ...]
    hd_missing_signature_counts: Mapping[Signature, int]
    _authority: InitVar[object | None] = None
    _seal: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self, _authority: object | None) -> None:
        if _authority is not _CATALOG_BINDING_AUTHORITY:
            raise ConvexityCatalogError(
                "CatalogBinding must be reconstructed by build_catalog_binding"
            )
        direct = dict(self.direct_by_signature)
        labels = dict(self.raw_label_by_name)
        aliases = dict(self.alias_by_signature)
        missing = dict(self.hd_missing_signature_counts)
        if (
            any(
                not isinstance(signature, tuple)
                or not signature
                or any(not isinstance(token, str) or not token for token in signature)
                or not isinstance(name, str)
                or not name
                for signature, name in (*direct.items(), *aliases.items())
            )
            or any(
                not isinstance(name, str)
                or not name
                or raw_label not in RAW_LABEL_TO_CLASS
                for name, raw_label in labels.items()
            )
            or len(set(direct.values())) != len(direct)
            or set(direct.values()) != set(labels)
            or set(direct).intersection(aliases)
            or set(missing) != set(aliases)
            or any(type(count) is not int or count <= 0 for count in missing.values())
            or any(labels.get(name) != "-1" for name in aliases.values())
        ):
            raise ConvexityCatalogError("CatalogBinding inventories are inconsistent")

        frozen_proofs: list[Mapping[str, Any]] = []
        proven_aliases: dict[Signature, str] = {}
        expected_fields = {
            "bd_singleton_task_id",
            "current_table_signature",
            "hd_occurrence_count",
            "raw_convexity_label",
            "release_signature",
            "shape_function_name",
            "singleton_positive_panel_count",
        }
        for proof in self.alias_proofs:
            if not isinstance(proof, Mapping) or set(proof) != expected_fields:
                raise ConvexityCatalogError("compatibility proof fields differ")
            name = proof["shape_function_name"]
            release_signature = tuple(proof["release_signature"])
            current_signature = tuple(proof["current_table_signature"])
            if (
                not isinstance(name, str)
                or aliases.get(release_signature) != name
                or direct.get(current_signature) != name
                or proof["bd_singleton_task_id"] != f"bd_{name}_0000"
                or proof["hd_occurrence_count"] != missing.get(release_signature)
                or proof["raw_convexity_label"] != "-1"
                or proof["singleton_positive_panel_count"] != 7
                or release_signature in proven_aliases
            ):
                raise ConvexityCatalogError("compatibility proof is inconsistent")
            proven_aliases[release_signature] = name
            frozen_proofs.append(
                MappingProxyType(
                    {
                        **dict(proof),
                        "current_table_signature": current_signature,
                        "release_signature": release_signature,
                    }
                )
            )
        if proven_aliases != aliases:
            raise ConvexityCatalogError("compatibility proofs do not cover aliases")

        object.__setattr__(self, "direct_by_signature", MappingProxyType(direct))
        object.__setattr__(self, "raw_label_by_name", MappingProxyType(labels))
        object.__setattr__(self, "alias_by_signature", MappingProxyType(aliases))
        object.__setattr__(self, "hd_missing_signature_counts", MappingProxyType(missing))
        object.__setattr__(self, "alias_proofs", tuple(frozen_proofs))
        object.__setattr__(self, "_seal", _CATALOG_BINDING_AUTHORITY)


@dataclass(frozen=True)
class PanelCatalogLabel:
    match_kind: str
    raw_label: str
    shape_function_name: str
    supervised_class: str

    def to_data(self) -> dict[str, str]:
        return {
            "match_kind": self.match_kind,
            "raw_label": self.raw_label,
            "shape_function_name": self.shape_function_name,
            "supervised_class": self.supervised_class,
        }


def convexity_catalog_source_digest() -> str:
    """Return the import-time source seal while on-disk bytes still agree."""

    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def convexity_catalog_algorithm_digest() -> str:
    """Bind the exact matching, compatibility, and claim-limiting policy."""

    return "sha256:" + canonical_digest(
        {
            "schema": ALGORITHM_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "implementation_source_sha256": convexity_catalog_source_digest(),
            "signature_policy": (
                "preserve ordered action kind, length/radius, arc angle, and turn; "
                "remove only the randomized style token; serialize to three decimals"
            ),
            "direct_match_policy": "exact_tuple_equality_only",
            "compatibility_policy": (
                "for_each_HD_signature_absent_from_the_current_table_require_exactly_"
                "one_BD_<shape_function_name>_0000_task_whose_seven_positive_panels_"
                "are_single_object_and_share_that_exact_signature"
            ),
            "compatibility_target_raw_label_required": "-1",
            "catalog_binding_external_construction_allowed": False,
            "catalog_binding_mappings_mutable": False,
            "fuzzy_matching": False,
            "raw_label_to_supervised_class": dict(RAW_LABEL_TO_CLASS),
            "catalog_unresolved_downstream_policy": (
                "any_calibrated_set_containing_catalog_unresolved_becomes_typed_GAP"
            ),
            "catalog_unresolved_certifies_convex_absence": False,
            "catalog_unresolved_populates_typed_not_applicable": False,
            "mixed_turning_supervision_available": False,
            "pixels_read": 0,
        }
    )


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_tsv(path: Path, *, label: str) -> tuple[list[dict[str, str]], bytes]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ConvexityCatalogError(f"cannot read {label}: {exc}") from exc
    if not rows:
        raise ConvexityCatalogError(f"{label} has no data rows")
    if any(any(key is None or value is None for key, value in row.items()) for row in rows):
        raise ConvexityCatalogError(f"{label} has a malformed row")
    return rows, raw


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConvexityCatalogError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConvexityCatalogError(f"{label} must be a JSON object")
    return value, raw


def signature_from_shape_row(row: Mapping[str, str]) -> Signature:
    """Canonical geometry/turn signature emitted by the upstream serializer."""

    try:
        base_actions = [item.strip() for item in row["set of base actions"].split(",")]
        turn_angles = [item.strip() for item in row["turn angles"].split("--")]
    except KeyError as exc:
        raise ConvexityCatalogError("shape row lacks geometry columns") from exc
    if len(base_actions) != len(turn_angles) or not base_actions:
        raise ConvexityCatalogError("shape row action/turn cardinality differs")
    tokens: list[str] = []
    for action, turn_text in zip(base_actions, turn_angles):
        if len(turn_text) < 2 or turn_text[0] not in {"L", "R"}:
            raise ConvexityCatalogError(f"invalid turn angle {turn_text!r}")
        try:
            angle = float(turn_text[1:])
            turn = (
                (angle + 180.0) / 360.0
                if turn_text[0] == "L"
                else (180.0 - angle) / 360.0
            )
            parts = action.split("_")
            if len(parts) == 2 and parts[0] == "line":
                tokens.append(f"line:{float(parts[1]):.3f}:{turn:.3f}")
            elif len(parts) == 3 and parts[0] == "arc":
                arc_angle = (float(parts[2]) + 360.0) / 720.0
                tokens.append(
                    f"arc:{float(parts[1]):.3f}:{arc_angle:.3f}:{turn:.3f}"
                )
            else:
                raise ConvexityCatalogError(f"unsupported base action {action!r}")
        except ValueError as exc:
            raise ConvexityCatalogError(f"invalid shape action {action!r}") from exc
    return tuple(tokens)


def signature_from_actions(actions: Sequence[str]) -> Signature:
    """Remove only stroke style from a serialized release action sequence."""

    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)) or not actions:
        raise ConvexityCatalogError("panel action sequence must be nonempty")
    tokens: list[str] = []
    for action in actions:
        if not isinstance(action, str):
            raise ConvexityCatalogError("panel action token must be a string")
        try:
            movement, turn_text = action.split("-")
            parts = movement.split("_")
            turn = float(turn_text)
            if len(parts) == 3 and parts[0] == "line" and parts[1]:
                tokens.append(f"line:{float(parts[2]):.3f}:{turn:.3f}")
            elif len(parts) == 4 and parts[0] == "arc" and parts[1]:
                tokens.append(
                    f"arc:{float(parts[2]):.3f}:{float(parts[3]):.3f}:{turn:.3f}"
                )
            else:
                raise ConvexityCatalogError(f"unsupported release action {action!r}")
        except ValueError as exc:
            raise ConvexityCatalogError(f"invalid release action {action!r}") from exc
    return tuple(tokens)


def _task_panels(
    programs: Mapping[str, Any], task_id: str
) -> Iterable[tuple[int, int, Sequence[str]]]:
    value = programs.get(task_id)
    if not isinstance(value, list) or len(value) != 2:
        raise ConvexityCatalogError(f"{task_id}: task must have two sides")
    for side_index, side in enumerate(value):
        if not isinstance(side, list) or len(side) != 7:
            raise ConvexityCatalogError(f"{task_id}: side must have seven panels")
        for panel_index, panel in enumerate(side):
            if not isinstance(panel, list) or len(panel) != 1:
                raise ConvexityCatalogError(
                    f"{task_id}: panel must contain exactly one shape program"
                )
            actions = panel[0]
            if not isinstance(actions, list):
                raise ConvexityCatalogError(f"{task_id}: shape program must be a list")
            yield side_index, panel_index, actions


def build_catalog_binding(
    *,
    shape_rows: Sequence[Mapping[str, str]],
    attribute_rows: Sequence[Mapping[str, str]],
    hd_programs: Mapping[str, Any],
    bd_programs: Mapping[str, Any],
) -> CatalogBinding:
    """Build the collision-free direct index and exact BD-backed aliases."""

    direct: dict[Signature, str] = {}
    shape_signature_by_name: dict[str, Signature] = {}
    for row in shape_rows:
        name = row.get("shape function name")
        if not isinstance(name, str) or not name or name in shape_signature_by_name:
            raise ConvexityCatalogError("shape function names must be unique strings")
        signature = signature_from_shape_row(row)
        if signature in direct:
            raise ConvexityCatalogError(
                f"direct signature collision: {direct[signature]} and {name}"
            )
        direct[signature] = name
        shape_signature_by_name[name] = signature

    raw_label_by_name: dict[str, str] = {}
    for row in attribute_rows:
        name = row.get("shape function name")
        raw_label = row.get("convex")
        if not isinstance(name, str) or not name or name in raw_label_by_name:
            raise ConvexityCatalogError("attribute shape names must be unique strings")
        if raw_label not in RAW_LABEL_TO_CLASS:
            raise ConvexityCatalogError(f"{name}: unsupported raw convexity label")
        raw_label_by_name[name] = raw_label
    if set(raw_label_by_name) != set(shape_signature_by_name):
        raise ConvexityCatalogError("shape and attribute name inventories differ")

    missing_counts: Counter[Signature] = Counter()
    for task_id in sorted(hd_programs):
        for _, _, actions in _task_panels(hd_programs, task_id):
            signature = signature_from_actions(actions)
            if signature not in direct:
                missing_counts[signature] += 1

    singleton_candidates: defaultdict[Signature, list[str]] = defaultdict(list)
    for name in sorted(shape_signature_by_name):
        task_id = f"bd_{name}_0000"
        if task_id not in bd_programs:
            continue
        positive_signatures = {
            signature_from_actions(actions)
            for side_index, _, actions in _task_panels(bd_programs, task_id)
            if side_index == 0
        }
        if len(positive_signatures) != 1:
            raise ConvexityCatalogError(
                f"{task_id}: singleton positives do not share one signature"
            )
        signature = next(iter(positive_signatures))
        if signature in missing_counts:
            singleton_candidates[signature].append(name)

    aliases: dict[Signature, str] = {}
    proofs: list[Mapping[str, Any]] = []
    for signature in sorted(missing_counts):
        names = singleton_candidates.get(signature, [])
        if len(names) != 1:
            raise ConvexityCatalogError(
                "HD missing signature lacks exactly one BD singleton identity"
            )
        name = names[0]
        if raw_label_by_name[name] != "-1":
            raise ConvexityCatalogError(
                f"{name}: compatibility aliases may target only raw -1"
            )
        if signature in direct or signature in aliases:
            raise ConvexityCatalogError("compatibility alias is not disjoint and unique")
        aliases[signature] = name
        proofs.append(
            {
                "bd_singleton_task_id": f"bd_{name}_0000",
                "current_table_signature": list(shape_signature_by_name[name]),
                "hd_occurrence_count": missing_counts[signature],
                "raw_convexity_label": "-1",
                "release_signature": list(signature),
                "shape_function_name": name,
                "singleton_positive_panel_count": 7,
            }
        )

    return CatalogBinding(
        direct_by_signature=direct,
        raw_label_by_name=raw_label_by_name,
        alias_by_signature=aliases,
        alias_proofs=tuple(proofs),
        hd_missing_signature_counts=dict(missing_counts),
        _authority=_CATALOG_BINDING_AUTHORITY,
    )


def catalog_label_for_actions(
    actions: Sequence[str], binding: CatalogBinding
) -> PanelCatalogLabel:
    """Return the exact catalog target for one single-object panel."""

    if (
        type(binding) is not CatalogBinding
        or binding._seal is not _CATALOG_BINDING_AUTHORITY
        or not isinstance(binding.direct_by_signature, MappingProxyType)
        or not isinstance(binding.raw_label_by_name, MappingProxyType)
        or not isinstance(binding.alias_by_signature, MappingProxyType)
    ):
        raise ConvexityCatalogError("catalog binding is not sealed builder output")
    signature = signature_from_actions(actions)
    if signature in binding.direct_by_signature:
        name = binding.direct_by_signature[signature]
        match_kind = "direct_exact_signature"
    elif signature in binding.alias_by_signature:
        name = binding.alias_by_signature[signature]
        match_kind = "bd_singleton_compatibility_alias"
    else:
        raise ConvexityCatalogError("panel signature has no catalog identity")
    raw_label = binding.raw_label_by_name[name]
    return PanelCatalogLabel(
        match_kind=match_kind,
        raw_label=raw_label,
        shape_function_name=name,
        supervised_class=RAW_LABEL_TO_CLASS[raw_label],
    )


def audit_cohorts(
    *,
    programs: Mapping[str, Any],
    cohorts: Mapping[str, Sequence[str]],
    binding: CatalogBinding,
) -> dict[str, Any]:
    """Audit exact panel targets and whole-task completeness for frozen cohorts."""

    result: dict[str, Any] = {}
    seen_tasks: set[str] = set()
    for cohort_name, task_ids in cohorts.items():
        if not isinstance(cohort_name, str) or not cohort_name:
            raise ConvexityCatalogError("cohort names must be nonempty strings")
        if any(not isinstance(task_id, str) for task_id in task_ids):
            raise ConvexityCatalogError(f"{cohort_name}: task IDs must be strings")
        if len(task_ids) != len(set(task_ids)) or seen_tasks.intersection(task_ids):
            raise ConvexityCatalogError("cohort task IDs overlap")
        seen_tasks.update(task_ids)
        label_counts: Counter[str] = Counter()
        match_counts: Counter[str] = Counter()
        rows: list[dict[str, Any]] = []
        strict_complete = compatibility_complete = binary_complete = 0
        for task_id in task_ids:
            task_rows: list[dict[str, Any]] = []
            for side_index, panel_index, actions in _task_panels(programs, task_id):
                label = catalog_label_for_actions(actions, binding)
                row = {
                    **label.to_data(),
                    "panel_id": f"hd/{task_id}/{1 - side_index}/{panel_index}.png",
                }
                task_rows.append(row)
                label_counts[label.supervised_class] += 1
                match_counts[label.match_kind] += 1
            if len(task_rows) != 14:
                raise ConvexityCatalogError(f"{task_id}: task does not have 14 panels")
            strict_complete += all(
                row["match_kind"] == "direct_exact_signature" for row in task_rows
            )
            compatibility_complete += 1
            binary_complete += all(row["raw_label"] in {"0", "1"} for row in task_rows)
            rows.extend(task_rows)
        result[cohort_name] = {
            "all_14_binary_0_or_1_task_count": binary_complete,
            "all_14_catalog_labelled_with_compatibility_task_count": compatibility_complete,
            "all_14_direct_exact_signature_task_count": strict_complete,
            "label_counts": dict(sorted(label_counts.items())),
            "match_counts": dict(sorted(match_counts.items())),
            "panel_count": len(rows),
            "panel_label_rows_digest": "sha256:" + canonical_digest(rows),
            "task_count": len(task_ids),
        }
    return result


def build_live_audit(
    *,
    shape_rows_path: Path,
    attribute_rows_path: Path,
    hd_programs_path: Path,
    bd_programs_path: Path,
    cohort_sources: Mapping[str, Sequence[str]],
    cohort_source_bindings: Mapping[str, str],
    additional_source_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Build a compact, source-bound audit without reading image bytes."""

    shape_rows, shape_raw = _read_tsv(shape_rows_path, label="shape table")
    attribute_rows, attribute_raw = _read_tsv(
        attribute_rows_path, label="attribute table"
    )
    hd_programs, hd_raw = _read_json_object(hd_programs_path, label="HD programs")
    bd_programs, bd_raw = _read_json_object(bd_programs_path, label="BD programs")
    additional_bindings: dict[str, dict[str, str]] = {}
    for name, path in sorted((additional_source_paths or {}).items()):
        if not isinstance(name, str) or not name:
            raise ConvexityCatalogError("additional source names must be nonempty")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ConvexityCatalogError(f"cannot read additional source {name}: {exc}") from exc
        additional_bindings[name] = {
            "path": str(path),
            "sha256": _sha256_bytes(raw),
        }
        if name == "catalog_semantics_readme" and (
            b"-1" not in raw or b"might not be applicable" not in raw
        ):
            raise ConvexityCatalogError("catalog README does not bind -1 applicability wording")
    binding = build_catalog_binding(
        shape_rows=shape_rows,
        attribute_rows=attribute_rows,
        hd_programs=hd_programs,
        bd_programs=bd_programs,
    )
    raw_labels = Counter(binding.raw_label_by_name.values())
    hd_panel_count = sum(
        1
        for task_id in sorted(hd_programs)
        for _ in _task_panels(hd_programs, task_id)
    )
    direct_hd_panel_count = hd_panel_count - sum(
        binding.hd_missing_signature_counts.values()
    )
    body: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "algorithm": {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_digest": convexity_catalog_algorithm_digest(),
            "source_path": "bongard/panel_convexity_catalog_audit.py",
            "source_sha256": convexity_catalog_source_digest(),
        },
        "claim_limits": {
            "catalog_binding_external_construction_allowed": False,
            "catalog_binding_mappings_mutable": False,
            "catalog_labels_are_semantic_pixel_truth": False,
            "catalog_unresolved_certifies_convex_absence": False,
            "catalog_unresolved_downstream_disposition": "GAP",
            "catalog_unresolved_populates_typed_not_applicable": False,
            "fuzzy_matching_used": False,
            "mixed_turning_supervision_available": False,
            "official_validation_or_test_authorized": False,
            "pixels_read": 0,
        },
        "cohort_source_bindings": dict(sorted(cohort_source_bindings.items())),
        "cohorts": audit_cohorts(
            programs=hd_programs,
            cohorts=cohort_sources,
            binding=binding,
        ),
        "compatibility": {
            "alias_count": len(binding.alias_by_signature),
            "alias_proofs": [
                {
                    **dict(proof),
                    "current_table_signature": list(proof["current_table_signature"]),
                    "release_signature": list(proof["release_signature"]),
                }
                for proof in binding.alias_proofs
            ],
            "all_alias_targets_are_catalog_unresolved": all(
                binding.raw_label_by_name[name] == "-1"
                for name in binding.alias_by_signature.values()
            ),
            "construction": (
                "exact_HD_no-match_signature_to_unique_BD_named-singleton_"
                "seven-positive-panel_signature"
            ),
            "source_version_diagnosis": (
                "the_archive_BD_singletons_and_HD_panels_share_four_geometry_"
                "signatures_absent_from_the_bound_upstream_table_rows_named_by_"
                "those_singletons;_the_exact_historical_table_revision_is_not_"
                "identified_by_the_bound_sources"
            ),
        },
        "inventory": {
            "attribute_row_count": len(attribute_rows),
            "direct_signature_collision_count": 0,
            "direct_signature_count": len(binding.direct_by_signature),
            "hd_direct_exact_panel_count": direct_hd_panel_count,
            "hd_panel_count": hd_panel_count,
            "hd_panel_count_with_compatibility": direct_hd_panel_count
            + sum(binding.hd_missing_signature_counts.values()),
            "hd_task_count": len(hd_programs),
            "raw_convexity_label_counts": dict(sorted(raw_labels.items())),
            "shape_row_count": len(shape_rows),
        },
        "source_bindings": {
            **additional_bindings,
            "attribute_rows": {
                "path": str(attribute_rows_path),
                "sha256": _sha256_bytes(attribute_raw),
            },
            "bd_action_programs": {
                "path": str(bd_programs_path),
                "sha256": _sha256_bytes(bd_raw),
            },
            "hd_action_programs": {
                "path": str(hd_programs_path),
                "sha256": _sha256_bytes(hd_raw),
            },
            "shape_rows": {
                "path": str(shape_rows_path),
                "sha256": _sha256_bytes(shape_raw),
            },
        },
    }
    body["record_digest"] = "sha256:" + canonical_digest(body)
    return body


__all__ = [
    "ALGORITHM_ID",
    "AUDIT_SCHEMA",
    "CatalogBinding",
    "ConvexityCatalogError",
    "PanelCatalogLabel",
    "RAW_LABEL_TO_CLASS",
    "audit_cohorts",
    "build_catalog_binding",
    "build_live_audit",
    "catalog_label_for_actions",
    "convexity_catalog_algorithm_digest",
    "convexity_catalog_source_digest",
    "signature_from_actions",
    "signature_from_shape_row",
]
