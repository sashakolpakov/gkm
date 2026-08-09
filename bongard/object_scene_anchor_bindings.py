"""Closed Python semantics for binding soft claims to salient scene anchors.

The module deliberately separates deterministic anchor eligibility from visual
residual judgments.  A verified salience artifact supplies one decision-bearing
selected graph.  Python freezes the exhaustive eligible binding catalog before
any witness state is accepted, then compiles witnesses on each binding and an
error-dominant existential over the complete catalog.

Raw graphs and audit sentinels are verification provenance only.  They are not
included in decision-manifest, binding, catalog, or evaluation identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_scene_anchor_catalog import (
    ObjectSceneAnchorCatalogEntry,
    ObjectSceneAnchorDecisionManifest,
)
from bongard.object_scene_anchor_salience import ANCHOR_SALIENCE_HARD_COMPLETE_CAP
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


ANCHOR_BINDING_SPEC_SCHEMA = "gkm.object-scene-anchor-binding-spec.v1"
ANCHOR_BINDING_SCHEMA = "gkm.object-scene-resolved-anchor-binding.v1"
ANCHOR_BINDING_CATALOG_SCHEMA = "gkm.object-scene-anchor-binding-catalog.v1"
ANCHOR_BINDING_WITNESS_CELL_SCHEMA = (
    "gkm.object-scene-anchor-binding-witness-cell.v1"
)
ANCHOR_BINDING_EVALUATION_SCHEMA = (
    "gkm.object-scene-anchor-binding-evaluation.v1"
)
ANCHOR_BINDING_CATALOG_EVALUATION_SCHEMA = (
    "gkm.object-scene-anchor-binding-catalog-evaluation.v1"
)

_ANCHOR_KINDS = ("entity", "part", "frame")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_BINDING_ID = re.compile(r"binding_[0-9]{3}\Z")
_PART_ID = re.compile(r"part-[0-9]{8}\Z")
_COMPACT_ID = re.compile(r"compact-[0-9]{8}\Z")
_FRAME_ID = re.compile(r"frame-[0-9]{8}\Z")
_WITNESS_ID = re.compile(r"witness_[0-9]{2}\Z")
_CATALOG_REASONS = frozenset(
    (
        "complete_nonempty",
        "complete_empty",
        "salience_indeterminate",
        "salience_error",
        "salience_verification_error",
        "foreign_object",
    )
)


class ObjectSceneAnchorBindingError(ValueError):
    """A binding definition, catalog, or evaluation is not canonical."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
    }


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorBindingError(f"{label} must be a lowercase sha256")
    return value


def _optional_digest(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _digest(value, label)


def _object_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise ObjectSceneAnchorBindingError("object_id must be bounded printable text")
    return value


def _exact_fields(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ObjectSceneAnchorBindingError(f"{label} fields differ")
    return value


def _disposition(value: object, label: str) -> Disposition:
    if not isinstance(value, Disposition):
        raise ObjectSceneAnchorBindingError(f"{label} disposition differs")
    return value


def _disposition_from_value(value: object, label: str) -> Disposition:
    try:
        return Disposition(value)
    except (TypeError, ValueError) as exc:
        raise ObjectSceneAnchorBindingError(f"{label} disposition differs") from exc


def _scene_and(values: Sequence[Disposition]) -> Disposition:
    row = tuple(values)
    if not row:
        return Disposition.PRESENT
    if Disposition.ERROR in row:
        return Disposition.ERROR
    if Disposition.CERTIFIED_ABSENT in row:
        return Disposition.CERTIFIED_ABSENT
    if all(item is Disposition.PRESENT for item in row):
        return Disposition.PRESENT
    return Disposition.INDETERMINATE


def _scene_or(values: Sequence[Disposition]) -> Disposition:
    row = tuple(values)
    if not row:
        return Disposition.CERTIFIED_ABSENT
    if Disposition.ERROR in row:
        return Disposition.ERROR
    if Disposition.PRESENT in row:
        return Disposition.PRESENT
    if all(item is Disposition.CERTIFIED_ABSENT for item in row):
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def _merge_repeated_disposition(
    first: Disposition | None, second: Disposition | None
) -> Disposition:
    if first is None or second is None or Disposition.ERROR in (first, second):
        return Disposition.ERROR
    if first is second is Disposition.PRESENT:
        return Disposition.PRESENT
    if first is second is Disposition.CERTIFIED_ABSENT:
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def _binding_spec_content(value: "ObjectSceneAnchorBindingSpec") -> dict[str, object]:
    interval = value.incident_part_count
    return {
        "schema": ANCHOR_BINDING_SPEC_SCHEMA,
        "anchor_kind": value.anchor_kind,
        "incident_part_count": (
            None
            if interval is None
            else {"lower": interval[0], "upper": interval[1]}
        ),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBindingSpec:
    """A closed reusable hard prerequisite shared by every card witness."""

    anchor_kind: str
    incident_part_count: tuple[int, int] | None
    spec_digest: str

    def __post_init__(self) -> None:
        if self.anchor_kind not in _ANCHOR_KINDS:
            raise ObjectSceneAnchorBindingError("anchor kind differs")
        interval = self.incident_part_count
        if self.anchor_kind == "frame":
            if (
                type(interval) is not tuple
                or len(interval) != 2
                or any(type(item) is not int for item in interval)
                or not 3
                <= interval[0]
                <= interval[1]
                <= ANCHOR_SALIENCE_HARD_COMPLETE_CAP
            ):
                raise ObjectSceneAnchorBindingError(
                    "frame incident-part interval differs"
                )
        elif interval is not None:
            raise ObjectSceneAnchorBindingError(
                "only frame bindings admit an incident-part interval"
            )
        _digest(self.spec_digest, "binding spec digest")
        if self.spec_digest != canonical_digest(_binding_spec_content(self)):
            raise ObjectSceneAnchorBindingError("binding spec digest differs")

    @classmethod
    def create(
        cls,
        anchor_kind: str,
        incident_part_count: tuple[int, int] | None = None,
    ) -> "ObjectSceneAnchorBindingSpec":
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "anchor_kind", anchor_kind)
        object.__setattr__(provisional, "incident_part_count", incident_part_count)
        return cls(
            anchor_kind,
            incident_part_count,
            canonical_digest(_binding_spec_content(provisional)),
        )

    @classmethod
    def entity(cls) -> "ObjectSceneAnchorBindingSpec":
        return cls.create("entity")

    @classmethod
    def part(cls) -> "ObjectSceneAnchorBindingSpec":
        return cls.create("part")

    @classmethod
    def frame(
        cls, lower: int = 3, upper: int = ANCHOR_SALIENCE_HARD_COMPLETE_CAP
    ) -> "ObjectSceneAnchorBindingSpec":
        return cls.create("frame", (lower, upper))

    def to_data(self) -> dict[str, object]:
        return {**_binding_spec_content(self), "spec_digest": self.spec_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorBindingSpec":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "anchor_kind",
                    "incident_part_count",
                    "predicate_authority_id",
                    "python_is_canonical_authority",
                    "spec_digest",
                )
            ),
            "binding spec",
        )
        if (
            raw["schema"] != ANCHOR_BINDING_SPEC_SCHEMA
            or raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID
            or raw["python_is_canonical_authority"] is not True
        ):
            raise ObjectSceneAnchorBindingError("binding spec policy differs")
        interval = raw["incident_part_count"]
        if interval is not None:
            interval_raw = _exact_fields(
                interval, frozenset(("lower", "upper")), "incident-part interval"
            )
            interval = (interval_raw["lower"], interval_raw["upper"])
        result = cls(raw["anchor_kind"], interval, raw["spec_digest"])
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBindingError("binding spec is not canonical")
        return result


def _resolved_binding_content(
    value: "ObjectSceneResolvedAnchorBinding",
) -> dict[str, object]:
    return {
        "schema": ANCHOR_BINDING_SCHEMA,
        "binding_id": value.binding_id,
        "object_id": value.object_id,
        "decision_manifest_digest": value.decision_manifest_digest,
        "spec_digest": value.spec_digest,
        "anchor_kind": value.anchor_kind,
        "anchor_id": value.anchor_id,
        "anchor_digest": value.anchor_digest,
        "selected_graph_digest": value.selected_graph_digest,
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneResolvedAnchorBinding:
    """One exact local anchor resolved inside a selected graph."""

    binding_id: str
    object_id: str
    decision_manifest_digest: str
    spec_digest: str
    anchor_kind: str
    anchor_id: str
    anchor_digest: str
    selected_graph_digest: str
    binding_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.binding_id, str) or _BINDING_ID.fullmatch(
            self.binding_id
        ) is None:
            raise ObjectSceneAnchorBindingError("binding ID differs")
        _object_id(self.object_id)
        for label, value in (
            ("decision manifest digest", self.decision_manifest_digest),
            ("binding spec digest", self.spec_digest),
            ("anchor digest", self.anchor_digest),
            ("selected graph digest", self.selected_graph_digest),
            ("binding digest", self.binding_digest),
        ):
            _digest(value, label)
        if self.anchor_kind not in _ANCHOR_KINDS:
            raise ObjectSceneAnchorBindingError("resolved anchor kind differs")
        if (
            (self.anchor_kind == "entity" and self.anchor_id != "entity")
            or (
                self.anchor_kind == "part"
                and _PART_ID.fullmatch(self.anchor_id) is None
                and _COMPACT_ID.fullmatch(self.anchor_id) is None
            )
            or (
                self.anchor_kind == "frame"
                and _FRAME_ID.fullmatch(self.anchor_id) is None
            )
        ):
            raise ObjectSceneAnchorBindingError("resolved anchor ID differs")
        if self.binding_digest != canonical_digest(_resolved_binding_content(self)):
            raise ObjectSceneAnchorBindingError("resolved binding digest differs")

    @property
    def binding_alias(self) -> str:
        return self.binding_id

    @classmethod
    def create(
        cls,
        *,
        binding_id: str,
        object_id: str,
        decision_manifest_digest: str,
        spec_digest: str,
        anchor_kind: str,
        anchor_id: str,
        anchor_digest: str,
        selected_graph_digest: str,
    ) -> "ObjectSceneResolvedAnchorBinding":
        values = {
            "binding_id": binding_id,
            "object_id": object_id,
            "decision_manifest_digest": decision_manifest_digest,
            "spec_digest": spec_digest,
            "anchor_kind": anchor_kind,
            "anchor_id": anchor_id,
            "anchor_digest": anchor_digest,
            "selected_graph_digest": selected_graph_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            binding_digest=canonical_digest(_resolved_binding_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_resolved_binding_content(self), "binding_digest": self.binding_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneResolvedAnchorBinding":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "binding_id",
                    "object_id",
                    "decision_manifest_digest",
                    "spec_digest",
                    "anchor_kind",
                    "anchor_id",
                    "anchor_digest",
                    "selected_graph_digest",
                    "predicate_authority_id",
                    "python_is_canonical_authority",
                    "binding_digest",
                )
            ),
            "resolved binding",
        )
        if (
            raw["schema"] != ANCHOR_BINDING_SCHEMA
            or raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID
            or raw["python_is_canonical_authority"] is not True
        ):
            raise ObjectSceneAnchorBindingError("resolved binding policy differs")
        result = cls(
            binding_id=raw["binding_id"],
            object_id=raw["object_id"],
            decision_manifest_digest=raw["decision_manifest_digest"],
            spec_digest=raw["spec_digest"],
            anchor_kind=raw["anchor_kind"],
            anchor_id=raw["anchor_id"],
            anchor_digest=raw["anchor_digest"],
            selected_graph_digest=raw["selected_graph_digest"],
            binding_digest=raw["binding_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBindingError("resolved binding is not canonical")
        return result


def _catalog_content(value: "ObjectSceneAnchorBindingCatalog") -> dict[str, object]:
    return {
        "schema": ANCHOR_BINDING_CATALOG_SCHEMA,
        "object_id": value.object_id,
        "decision_manifest_digest": value.decision_manifest_digest,
        "binding_spec": value.binding_spec.to_data(),
        "selected_graph_digest": value.selected_graph_digest,
        "hard_disposition": value.hard_disposition.value,
        "reason": value.reason,
        "catalog_complete_under_spec": value.catalog_complete_under_spec,
        "bindings": [item.to_data() for item in value.bindings],
        "selected_graph_is_only_decision_graph": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBindingCatalog:
    object_id: str
    decision_manifest_digest: str | None
    binding_spec: ObjectSceneAnchorBindingSpec
    selected_graph_digest: str | None
    hard_disposition: Disposition
    reason: str
    catalog_complete_under_spec: bool
    bindings: tuple[ObjectSceneResolvedAnchorBinding, ...]
    catalog_digest: str

    def __post_init__(self) -> None:
        _object_id(self.object_id)
        _optional_digest(
            self.decision_manifest_digest, "catalog decision manifest digest"
        )
        if type(self.binding_spec) is not ObjectSceneAnchorBindingSpec:
            raise ObjectSceneAnchorBindingError("catalog binding spec differs")
        _optional_digest(self.selected_graph_digest, "catalog selected graph digest")
        _disposition(self.hard_disposition, "catalog hard")
        if self.reason not in _CATALOG_REASONS:
            raise ObjectSceneAnchorBindingError("catalog reason differs")
        if type(self.catalog_complete_under_spec) is not bool:
            raise ObjectSceneAnchorBindingError("catalog completeness differs")
        if (
            type(self.bindings) is not tuple
            or any(type(item) is not ObjectSceneResolvedAnchorBinding for item in self.bindings)
            or tuple(item.binding_id for item in self.bindings)
            != tuple(f"binding_{index:03d}" for index in range(len(self.bindings)))
            or tuple((item.anchor_kind, item.anchor_id) for item in self.bindings)
            != tuple(sorted((item.anchor_kind, item.anchor_id) for item in self.bindings))
        ):
            raise ObjectSceneAnchorBindingError("catalog binding order differs")
        for item in self.bindings:
            if (
                item.object_id != self.object_id
                or item.decision_manifest_digest != self.decision_manifest_digest
                or item.spec_digest != self.binding_spec.spec_digest
                or item.anchor_kind != self.binding_spec.anchor_kind
                or item.selected_graph_digest != self.selected_graph_digest
            ):
                raise ObjectSceneAnchorBindingError("catalog binding provenance differs")
        expected_policy = {
            "complete_nonempty": (Disposition.PRESENT, True, True, True),
            "complete_empty": (Disposition.CERTIFIED_ABSENT, True, False, True),
            "salience_indeterminate": (Disposition.INDETERMINATE, False, False, False),
            "salience_error": (Disposition.ERROR, False, False, False),
            "salience_verification_error": (Disposition.ERROR, False, False, False),
            "foreign_object": (Disposition.ERROR, False, False, False),
        }[self.reason]
        expected_disposition, expected_complete, expected_nonempty, graph_present = (
            expected_policy
        )
        if (
            self.hard_disposition is not expected_disposition
            or self.catalog_complete_under_spec is not expected_complete
            or bool(self.bindings) is not expected_nonempty
            or (self.selected_graph_digest is not None) is not graph_present
            or (
                (self.reason == "salience_verification_error" or self.reason == "foreign_object")
                != (self.decision_manifest_digest is None)
            )
        ):
            raise ObjectSceneAnchorBindingError("catalog state policy differs")
        _digest(self.catalog_digest, "binding catalog digest")
        if self.catalog_digest != canonical_digest(_catalog_content(self)):
            raise ObjectSceneAnchorBindingError("binding catalog digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_catalog_content(self), "catalog_digest": self.catalog_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorBindingCatalog":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "object_id",
                    "decision_manifest_digest",
                    "binding_spec",
                    "selected_graph_digest",
                    "hard_disposition",
                    "reason",
                    "catalog_complete_under_spec",
                    "bindings",
                    "selected_graph_is_only_decision_graph",
                    "predicate_authority_id",
                    "python_is_canonical_authority",
                    "catalog_digest",
                )
            ),
            "binding catalog",
        )
        if (
            raw["schema"] != ANCHOR_BINDING_CATALOG_SCHEMA
            or raw["selected_graph_is_only_decision_graph"] is not True
            or raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID
            or raw["python_is_canonical_authority"] is not True
            or not isinstance(raw["binding_spec"], Mapping)
            or not isinstance(raw["bindings"], list)
        ):
            raise ObjectSceneAnchorBindingError("binding catalog policy differs")
        result = cls(
            object_id=raw["object_id"],
            decision_manifest_digest=raw["decision_manifest_digest"],
            binding_spec=ObjectSceneAnchorBindingSpec.from_data(
                raw["binding_spec"]
            ),
            selected_graph_digest=raw["selected_graph_digest"],
            hard_disposition=_disposition_from_value(
                raw["hard_disposition"], "catalog hard"
            ),
            reason=raw["reason"],
            catalog_complete_under_spec=raw["catalog_complete_under_spec"],
            bindings=tuple(
                ObjectSceneResolvedAnchorBinding.from_data(item)
                for item in raw["bindings"]
            ),
            catalog_digest=raw["catalog_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBindingError("binding catalog is not canonical")
        return result


def _construct_catalog(
    *,
    object_id: str,
    decision_manifest_digest: str | None,
    binding_spec: ObjectSceneAnchorBindingSpec,
    selected_graph_digest: str | None,
    hard_disposition: Disposition,
    reason: str,
    catalog_complete_under_spec: bool,
    bindings: tuple[ObjectSceneResolvedAnchorBinding, ...],
) -> ObjectSceneAnchorBindingCatalog:
    values = {
        "object_id": object_id,
        "decision_manifest_digest": decision_manifest_digest,
        "binding_spec": binding_spec,
        "selected_graph_digest": selected_graph_digest,
        "hard_disposition": hard_disposition,
        "reason": reason,
        "catalog_complete_under_spec": catalog_complete_under_spec,
        "bindings": bindings,
    }
    provisional = object.__new__(ObjectSceneAnchorBindingCatalog)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorBindingCatalog(
        **values, catalog_digest=canonical_digest(_catalog_content(provisional))
    )


def _failure_catalog(
    *,
    object_id: str,
    binding_spec: ObjectSceneAnchorBindingSpec,
    reason: str,
) -> ObjectSceneAnchorBindingCatalog:
    return _construct_catalog(
        object_id=object_id,
        decision_manifest_digest=None,
        binding_spec=binding_spec,
        selected_graph_digest=None,
        hard_disposition=Disposition.ERROR,
        reason=reason,
        catalog_complete_under_spec=False,
        bindings=(),
    )


def _entity_anchor_digest(*, object_id: str, manifest_digest: str) -> str:
    return canonical_digest(
        {
            "schema": "gkm.object-scene-entity-anchor.v1",
            "object_id": object_id,
            "decision_manifest_digest": manifest_digest,
        }
    )


def _restore_decision_manifest(
    source: object,
) -> ObjectSceneAnchorDecisionManifest | None:
    """Validate a full entry when supplied, then expose only its decision view."""

    try:
        if type(source) is ObjectSceneAnchorCatalogEntry:
            restored_entry = ObjectSceneAnchorCatalogEntry.from_data(source.to_data())
            if restored_entry != source:
                return None
            return restored_entry.decision_manifest
        if type(source) is ObjectSceneAnchorDecisionManifest:
            restored_manifest = ObjectSceneAnchorDecisionManifest.from_data(
                source.to_data()
            )
            if restored_manifest != source:
                return None
            return restored_manifest
    except Exception:
        return None
    return None


def build_object_scene_anchor_binding_catalog(
    decision_source: object,
    binding_spec: ObjectSceneAnchorBindingSpec,
    *,
    expected_object_id: str,
) -> ObjectSceneAnchorBindingCatalog:
    """Freeze an exhaustive selected-anchor catalog or a typed hard gap."""

    object_id = _object_id(expected_object_id)
    if type(binding_spec) is not ObjectSceneAnchorBindingSpec:
        raise TypeError("binding_spec must be exact ObjectSceneAnchorBindingSpec")
    if ObjectSceneAnchorBindingSpec.from_data(binding_spec.to_data()) != binding_spec:
        raise ObjectSceneAnchorBindingError("binding spec is not canonical")
    manifest = _restore_decision_manifest(decision_source)
    if manifest is None:
        return _failure_catalog(
            object_id=object_id,
            binding_spec=binding_spec,
            reason="salience_verification_error",
        )
    if manifest.object_id != object_id:
        return _failure_catalog(
            object_id=object_id,
            binding_spec=binding_spec,
            reason="foreign_object",
        )
    if manifest.salience_state == "indeterminate":
        return _construct_catalog(
            object_id=object_id,
            decision_manifest_digest=manifest.manifest_digest,
            binding_spec=binding_spec,
            selected_graph_digest=None,
            hard_disposition=Disposition.INDETERMINATE,
            reason="salience_indeterminate",
            catalog_complete_under_spec=False,
            bindings=(),
        )
    if manifest.salience_state == "error":
        return _construct_catalog(
            object_id=object_id,
            decision_manifest_digest=manifest.manifest_digest,
            binding_spec=binding_spec,
            selected_graph_digest=None,
            hard_disposition=Disposition.ERROR,
            reason="salience_error",
            catalog_complete_under_spec=False,
            bindings=(),
        )
    graph = manifest.selected_graph
    if graph is None or graph.status.state != "clean":
        return _failure_catalog(
            object_id=object_id,
            binding_spec=binding_spec,
            reason="salience_verification_error",
        )
    graph_digest = graph.artifact_digest
    if graph_digest != manifest.selected_graph_artifact_digest:
        return _failure_catalog(
            object_id=object_id,
            binding_spec=binding_spec,
            reason="salience_verification_error",
        )
    candidates: list[tuple[str, str]]
    if binding_spec.anchor_kind == "entity":
        candidates = [
            (
                "entity",
                _entity_anchor_digest(
                    object_id=object_id,
                    manifest_digest=manifest.manifest_digest,
                ),
            )
        ]
    elif binding_spec.anchor_kind == "part":
        candidates = [
            (item.part_id, item.digest()) for item in graph.parts
        ] + [
            (item.compact_id, item.digest())
            for item in graph.compact_components
        ]
    else:
        assert binding_spec.incident_part_count is not None
        lower, upper = binding_spec.incident_part_count
        candidates = [
            (item.frame_id, item.digest())
            for item in graph.cyclic_frames
            if lower <= len(item.clockwise_incident_part_ids) <= upper
        ]
    candidates.sort(key=lambda item: item[0])
    bindings = tuple(
        ObjectSceneResolvedAnchorBinding.create(
            binding_id=f"binding_{index:03d}",
            object_id=object_id,
            decision_manifest_digest=manifest.manifest_digest,
            spec_digest=binding_spec.spec_digest,
            anchor_kind=binding_spec.anchor_kind,
            anchor_id=anchor_id,
            anchor_digest=anchor_digest,
            selected_graph_digest=graph_digest,
        )
        for index, (anchor_id, anchor_digest) in enumerate(candidates)
    )
    return _construct_catalog(
        object_id=object_id,
        decision_manifest_digest=manifest.manifest_digest,
        binding_spec=binding_spec,
        selected_graph_digest=graph_digest,
        hard_disposition=(
            Disposition.PRESENT if bindings else Disposition.CERTIFIED_ABSENT
        ),
        reason="complete_nonempty" if bindings else "complete_empty",
        catalog_complete_under_spec=True,
        bindings=bindings,
    )


@dataclass(frozen=True, order=True, slots=True)
class ObjectSceneAnchorWitnessSpec:
    witness_id: str
    witness_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.witness_id, str) or _WITNESS_ID.fullmatch(
            self.witness_id
        ) is None:
            raise ObjectSceneAnchorBindingError("witness ID differs")
        _digest(self.witness_digest, "witness digest")

    def to_data(self) -> dict[str, str]:
        return {
            "witness_id": self.witness_id,
            "witness_digest": self.witness_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorWitnessSpec":
        raw = _exact_fields(
            value,
            frozenset(("witness_id", "witness_digest")),
            "anchor witness spec",
        )
        result = cls(raw["witness_id"], raw["witness_digest"])
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBindingError("anchor witness spec is not canonical")
        return result


def _witness_cell_content(
    value: "ObjectSceneAnchorWitnessCell",
) -> dict[str, object]:
    return {
        "schema": ANCHOR_BINDING_WITNESS_CELL_SCHEMA,
        "binding_digest": value.binding_digest,
        "witness_id": value.witness_id,
        "witness_digest": value.witness_digest,
        "disposition": value.disposition.value,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorWitnessCell:
    binding_digest: str
    witness_id: str
    witness_digest: str
    disposition: Disposition
    cell_digest: str

    def __post_init__(self) -> None:
        _digest(self.binding_digest, "witness binding digest")
        ObjectSceneAnchorWitnessSpec(self.witness_id, self.witness_digest)
        _disposition(self.disposition, "witness")
        _digest(self.cell_digest, "witness cell digest")
        if self.cell_digest != canonical_digest(_witness_cell_content(self)):
            raise ObjectSceneAnchorBindingError("witness cell digest differs")

    @classmethod
    def create(
        cls,
        binding: ObjectSceneResolvedAnchorBinding,
        witness: ObjectSceneAnchorWitnessSpec,
        disposition: Disposition,
    ) -> "ObjectSceneAnchorWitnessCell":
        if type(binding) is not ObjectSceneResolvedAnchorBinding:
            raise TypeError("binding must be exact ObjectSceneResolvedAnchorBinding")
        if type(witness) is not ObjectSceneAnchorWitnessSpec:
            raise TypeError("witness must be exact ObjectSceneAnchorWitnessSpec")
        values = {
            "binding_digest": binding.binding_digest,
            "witness_id": witness.witness_id,
            "witness_digest": witness.witness_digest,
            "disposition": _disposition(disposition, "witness"),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values, cell_digest=canonical_digest(_witness_cell_content(provisional))
        )

    def to_data(self) -> dict[str, object]:
        return {**_witness_cell_content(self), "cell_digest": self.cell_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorWitnessCell":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "binding_digest",
                    "witness_id",
                    "witness_digest",
                    "disposition",
                    "predicate_authority_id",
                    "python_is_canonical_authority",
                    "cell_digest",
                )
            ),
            "anchor witness cell",
        )
        if (
            raw["schema"] != ANCHOR_BINDING_WITNESS_CELL_SCHEMA
            or raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID
            or raw["python_is_canonical_authority"] is not True
        ):
            raise ObjectSceneAnchorBindingError("anchor witness cell policy differs")
        result = cls(
            binding_digest=raw["binding_digest"],
            witness_id=raw["witness_id"],
            witness_digest=raw["witness_digest"],
            disposition=_disposition_from_value(
                raw["disposition"], "anchor witness"
            ),
            cell_digest=raw["cell_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBindingError("anchor witness cell is not canonical")
        return result


def _canonical_witness_specs(
    value: object,
) -> tuple[ObjectSceneAnchorWitnessSpec, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ObjectSceneAnchorBindingError("required witnesses must be a sequence")
    result = tuple(value)
    if (
        not result
        or len(result) > 4
        or any(type(item) is not ObjectSceneAnchorWitnessSpec for item in result)
        or tuple(item.witness_id for item in result)
        != tuple(f"witness_{index:02d}" for index in range(len(result)))
        or len({item.witness_digest for item in result}) != len(result)
    ):
        raise ObjectSceneAnchorBindingError("required witness catalog differs")
    return result


def _binding_evaluation_content(
    value: "ObjectSceneAnchorBindingEvaluation",
) -> dict[str, object]:
    return {
        "schema": ANCHOR_BINDING_EVALUATION_SCHEMA,
        "binding_digest": value.binding_digest,
        "required_witnesses": [item.to_data() for item in value.required_witnesses],
        "witness_cells": [item.to_data() for item in value.witness_cells],
        "structurally_valid": value.structurally_valid,
        "disposition": value.disposition.value,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBindingEvaluation:
    binding_digest: str
    required_witnesses: tuple[ObjectSceneAnchorWitnessSpec, ...]
    witness_cells: tuple[ObjectSceneAnchorWitnessCell, ...]
    structurally_valid: bool
    disposition: Disposition
    evaluation_digest: str

    def __post_init__(self) -> None:
        _digest(self.binding_digest, "binding evaluation binding digest")
        required = _canonical_witness_specs(self.required_witnesses)
        if type(self.witness_cells) is not tuple or any(
            type(item) is not ObjectSceneAnchorWitnessCell for item in self.witness_cells
        ):
            raise ObjectSceneAnchorBindingError("binding witness cells differ")
        valid = (
            tuple((item.witness_id, item.witness_digest) for item in self.witness_cells)
            == tuple((item.witness_id, item.witness_digest) for item in required)
            and all(item.binding_digest == self.binding_digest for item in self.witness_cells)
        )
        if type(self.structurally_valid) is not bool or self.structurally_valid is not valid:
            raise ObjectSceneAnchorBindingError("binding structural validity differs")
        expected = (
            _scene_and(tuple(item.disposition for item in self.witness_cells))
            if valid
            else Disposition.ERROR
        )
        if self.disposition is not expected:
            raise ObjectSceneAnchorBindingError("binding evaluation disposition differs")
        _digest(self.evaluation_digest, "binding evaluation digest")
        if self.evaluation_digest != canonical_digest(_binding_evaluation_content(self)):
            raise ObjectSceneAnchorBindingError("binding evaluation digest differs")

    def to_data(self) -> dict[str, object]:
        return {
            **_binding_evaluation_content(self),
            "evaluation_digest": self.evaluation_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorBindingEvaluation":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "binding_digest",
                    "required_witnesses",
                    "witness_cells",
                    "structurally_valid",
                    "disposition",
                    "predicate_authority_id",
                    "python_is_canonical_authority",
                    "evaluation_digest",
                )
            ),
            "anchor binding evaluation",
        )
        if (
            raw["schema"] != ANCHOR_BINDING_EVALUATION_SCHEMA
            or raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID
            or raw["python_is_canonical_authority"] is not True
            or not isinstance(raw["required_witnesses"], list)
            or not isinstance(raw["witness_cells"], list)
        ):
            raise ObjectSceneAnchorBindingError(
                "anchor binding evaluation policy differs"
            )
        result = cls(
            binding_digest=raw["binding_digest"],
            required_witnesses=tuple(
                ObjectSceneAnchorWitnessSpec.from_data(item)
                for item in raw["required_witnesses"]
            ),
            witness_cells=tuple(
                ObjectSceneAnchorWitnessCell.from_data(item)
                for item in raw["witness_cells"]
            ),
            structurally_valid=raw["structurally_valid"],
            disposition=_disposition_from_value(
                raw["disposition"], "anchor binding evaluation"
            ),
            evaluation_digest=raw["evaluation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBindingError(
                "anchor binding evaluation is not canonical"
            )
        return result


def compile_object_scene_anchor_binding(
    binding: ObjectSceneResolvedAnchorBinding,
    required_witnesses: Sequence[ObjectSceneAnchorWitnessSpec],
    witness_cells: Sequence[ObjectSceneAnchorWitnessCell],
) -> ObjectSceneAnchorBindingEvaluation:
    if type(binding) is not ObjectSceneResolvedAnchorBinding:
        raise TypeError("binding must be exact ObjectSceneResolvedAnchorBinding")
    required = _canonical_witness_specs(required_witnesses)
    if isinstance(witness_cells, (str, bytes)) or not isinstance(
        witness_cells, Sequence
    ):
        raise ObjectSceneAnchorBindingError("witness cells must be a sequence")
    cells = tuple(witness_cells)
    if any(type(item) is not ObjectSceneAnchorWitnessCell for item in cells):
        raise ObjectSceneAnchorBindingError("witness cells have the wrong type")
    valid = (
        tuple((item.witness_id, item.witness_digest) for item in cells)
        == tuple((item.witness_id, item.witness_digest) for item in required)
        and all(item.binding_digest == binding.binding_digest for item in cells)
    )
    values = {
        "binding_digest": binding.binding_digest,
        "required_witnesses": required,
        "witness_cells": cells,
        "structurally_valid": valid,
        "disposition": (
            _scene_and(tuple(item.disposition for item in cells))
            if valid
            else Disposition.ERROR
        ),
    }
    provisional = object.__new__(ObjectSceneAnchorBindingEvaluation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorBindingEvaluation(
        **values,
        evaluation_digest=canonical_digest(_binding_evaluation_content(provisional)),
    )


def _catalog_evaluation_content(
    value: "ObjectSceneAnchorCatalogEvaluation",
) -> dict[str, object]:
    return {
        "schema": ANCHOR_BINDING_CATALOG_EVALUATION_SCHEMA,
        "catalog_digest": value.catalog_digest,
        "hard_disposition": value.hard_disposition.value,
        "expected_binding_digests": list(value.expected_binding_digests),
        "binding_evaluations": [item.to_data() for item in value.binding_evaluations],
        "structurally_valid": value.structurally_valid,
        "disposition": value.disposition.value,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorCatalogEvaluation:
    catalog_digest: str
    hard_disposition: Disposition
    expected_binding_digests: tuple[str, ...]
    binding_evaluations: tuple[ObjectSceneAnchorBindingEvaluation, ...]
    structurally_valid: bool
    disposition: Disposition
    evaluation_digest: str

    def __post_init__(self) -> None:
        _digest(self.catalog_digest, "catalog evaluation catalog digest")
        _disposition(self.hard_disposition, "catalog evaluation hard")
        if type(self.expected_binding_digests) is not tuple:
            raise ObjectSceneAnchorBindingError("expected binding digests differ")
        for item in self.expected_binding_digests:
            _digest(item, "expected binding digest")
        if len(set(self.expected_binding_digests)) != len(
            self.expected_binding_digests
        ):
            raise ObjectSceneAnchorBindingError("expected binding digests repeat")
        if type(self.binding_evaluations) is not tuple or any(
            type(item) is not ObjectSceneAnchorBindingEvaluation
            for item in self.binding_evaluations
        ):
            raise ObjectSceneAnchorBindingError("catalog binding evaluations differ")
        valid = (
            self.hard_disposition is Disposition.PRESENT
            and tuple(item.binding_digest for item in self.binding_evaluations)
            == self.expected_binding_digests
            and all(item.structurally_valid for item in self.binding_evaluations)
        ) or (
            self.hard_disposition is not Disposition.PRESENT
            and not self.expected_binding_digests
            and not self.binding_evaluations
        )
        if type(self.structurally_valid) is not bool or self.structurally_valid is not valid:
            raise ObjectSceneAnchorBindingError("catalog structural validity differs")
        if not valid:
            expected = Disposition.ERROR
        elif self.hard_disposition is Disposition.PRESENT:
            expected = _scene_or(
                tuple(item.disposition for item in self.binding_evaluations)
            )
        else:
            expected = self.hard_disposition
        if self.disposition is not expected:
            raise ObjectSceneAnchorBindingError("catalog evaluation disposition differs")
        _digest(self.evaluation_digest, "catalog evaluation digest")
        if self.evaluation_digest != canonical_digest(_catalog_evaluation_content(self)):
            raise ObjectSceneAnchorBindingError("catalog evaluation digest differs")

    def to_data(self) -> dict[str, object]:
        return {
            **_catalog_evaluation_content(self),
            "evaluation_digest": self.evaluation_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneAnchorCatalogEvaluation":
        raw = _exact_fields(
            value,
            frozenset(
                (
                    "schema",
                    "catalog_digest",
                    "hard_disposition",
                    "expected_binding_digests",
                    "binding_evaluations",
                    "structurally_valid",
                    "disposition",
                    "predicate_authority_id",
                    "python_is_canonical_authority",
                    "evaluation_digest",
                )
            ),
            "anchor catalog evaluation",
        )
        if (
            raw["schema"] != ANCHOR_BINDING_CATALOG_EVALUATION_SCHEMA
            or raw["predicate_authority_id"] != PYTHON_PREDICATE_AUTHORITY_ID
            or raw["python_is_canonical_authority"] is not True
            or not isinstance(raw["expected_binding_digests"], list)
            or not isinstance(raw["binding_evaluations"], list)
        ):
            raise ObjectSceneAnchorBindingError(
                "anchor catalog evaluation policy differs"
            )
        result = cls(
            catalog_digest=raw["catalog_digest"],
            hard_disposition=_disposition_from_value(
                raw["hard_disposition"], "catalog evaluation hard"
            ),
            expected_binding_digests=tuple(raw["expected_binding_digests"]),
            binding_evaluations=tuple(
                ObjectSceneAnchorBindingEvaluation.from_data(item)
                for item in raw["binding_evaluations"]
            ),
            structurally_valid=raw["structurally_valid"],
            disposition=_disposition_from_value(
                raw["disposition"], "anchor catalog evaluation"
            ),
            evaluation_digest=raw["evaluation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectSceneAnchorBindingError(
                "anchor catalog evaluation is not canonical"
            )
        return result


def compile_object_scene_anchor_catalog(
    catalog: ObjectSceneAnchorBindingCatalog,
    binding_evaluations: Sequence[ObjectSceneAnchorBindingEvaluation] = (),
) -> ObjectSceneAnchorCatalogEvaluation:
    if type(catalog) is not ObjectSceneAnchorBindingCatalog:
        raise TypeError("catalog must be exact ObjectSceneAnchorBindingCatalog")
    if isinstance(binding_evaluations, (str, bytes)) or not isinstance(
        binding_evaluations, Sequence
    ):
        raise ObjectSceneAnchorBindingError("binding evaluations must be a sequence")
    rows = tuple(binding_evaluations)
    if any(type(item) is not ObjectSceneAnchorBindingEvaluation for item in rows):
        raise ObjectSceneAnchorBindingError("binding evaluations have the wrong type")
    expected = tuple(item.binding_digest for item in catalog.bindings)
    valid = (
        catalog.hard_disposition is Disposition.PRESENT
        and tuple(item.binding_digest for item in rows) == expected
        and all(item.structurally_valid for item in rows)
    ) or (
        catalog.hard_disposition is not Disposition.PRESENT
        and not expected
        and not rows
    )
    if not valid:
        disposition = Disposition.ERROR
    elif catalog.hard_disposition is Disposition.PRESENT:
        disposition = _scene_or(tuple(item.disposition for item in rows))
    else:
        disposition = catalog.hard_disposition
    values = {
        "catalog_digest": catalog.catalog_digest,
        "hard_disposition": catalog.hard_disposition,
        "expected_binding_digests": expected,
        "binding_evaluations": rows,
        "structurally_valid": valid,
        "disposition": disposition,
    }
    provisional = object.__new__(ObjectSceneAnchorCatalogEvaluation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorCatalogEvaluation(
        **values,
        evaluation_digest=canonical_digest(_catalog_evaluation_content(provisional)),
    )


def _merge_binding_evaluations(
    binding: ObjectSceneResolvedAnchorBinding,
    required_witnesses: tuple[ObjectSceneAnchorWitnessSpec, ...],
    first: ObjectSceneAnchorBindingEvaluation | None,
    second: ObjectSceneAnchorBindingEvaluation | None,
) -> ObjectSceneAnchorBindingEvaluation:
    expected = tuple(
        (item.witness_id, item.witness_digest) for item in required_witnesses
    )
    if (
        type(first) is not ObjectSceneAnchorBindingEvaluation
        or type(second) is not ObjectSceneAnchorBindingEvaluation
        or first.binding_digest != binding.binding_digest
        or second.binding_digest != binding.binding_digest
        or not first.structurally_valid
        or not second.structurally_valid
        or tuple(
            (item.witness_id, item.witness_digest)
            for item in first.required_witnesses
        )
        != expected
        or tuple(
            (item.witness_id, item.witness_digest)
            for item in second.required_witnesses
        )
        != expected
    ):
        return compile_object_scene_anchor_binding(binding, required_witnesses, ())
    merged_cells = tuple(
        ObjectSceneAnchorWitnessCell.create(
            binding,
            witness,
            _merge_repeated_disposition(first_cell.disposition, second_cell.disposition),
        )
        for witness, first_cell, second_cell in zip(
            required_witnesses,
            first.witness_cells,
            second.witness_cells,
            strict=True,
        )
    )
    return compile_object_scene_anchor_binding(
        binding, required_witnesses, merged_cells
    )


def _structural_error_binding_evaluation(
    binding_digest: str,
    required_witnesses: tuple[ObjectSceneAnchorWitnessSpec, ...],
) -> ObjectSceneAnchorBindingEvaluation:
    """Construct a serializable error witness for a malformed outer merge."""

    values = {
        "binding_digest": _digest(binding_digest, "structural error binding digest"),
        "required_witnesses": required_witnesses,
        "witness_cells": (),
        "structurally_valid": False,
        "disposition": Disposition.ERROR,
    }
    provisional = object.__new__(ObjectSceneAnchorBindingEvaluation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSceneAnchorBindingEvaluation(
        **values,
        evaluation_digest=canonical_digest(_binding_evaluation_content(provisional)),
    )


def merge_repeated_object_scene_anchor_catalog_evaluations(
    catalog: ObjectSceneAnchorBindingCatalog,
    required_witnesses: Sequence[ObjectSceneAnchorWitnessSpec],
    first: ObjectSceneAnchorCatalogEvaluation,
    second: ObjectSceneAnchorCatalogEvaluation,
) -> ObjectSceneAnchorCatalogEvaluation:
    """Merge witness states per exact binding before existential compilation."""

    if type(catalog) is not ObjectSceneAnchorBindingCatalog:
        raise TypeError("catalog must be exact ObjectSceneAnchorBindingCatalog")
    required = _canonical_witness_specs(required_witnesses)
    expected_binding_digests = tuple(
        item.binding_digest for item in catalog.bindings
    )
    if (
        type(first) is not ObjectSceneAnchorCatalogEvaluation
        or type(second) is not ObjectSceneAnchorCatalogEvaluation
        or first.catalog_digest != catalog.catalog_digest
        or second.catalog_digest != catalog.catalog_digest
        or first.hard_disposition is not catalog.hard_disposition
        or second.hard_disposition is not catalog.hard_disposition
        or first.expected_binding_digests != expected_binding_digests
        or second.expected_binding_digests != expected_binding_digests
        or tuple(
            item.binding_digest for item in first.binding_evaluations
        )
        != expected_binding_digests
        or tuple(
            item.binding_digest for item in second.binding_evaluations
        )
        != expected_binding_digests
        or not first.structurally_valid
        or not second.structurally_valid
    ):
        rows: tuple[ObjectSceneAnchorBindingEvaluation, ...] = ()
        if catalog.hard_disposition is not Disposition.PRESENT:
            rows = (
                _structural_error_binding_evaluation(
                    catalog.catalog_digest, required
                ),
            )
        return compile_object_scene_anchor_catalog(
            catalog,
            rows,
        )
    if catalog.hard_disposition is not Disposition.PRESENT:
        return compile_object_scene_anchor_catalog(catalog)
    merged = tuple(
        _merge_binding_evaluations(binding, required, first_row, second_row)
        for binding, first_row, second_row in zip(
            catalog.bindings,
            first.binding_evaluations,
            second.binding_evaluations,
            strict=True,
        )
    )
    return compile_object_scene_anchor_catalog(catalog, merged)


__all__ = (
    "ANCHOR_BINDING_CATALOG_EVALUATION_SCHEMA",
    "ANCHOR_BINDING_CATALOG_SCHEMA",
    "ANCHOR_BINDING_EVALUATION_SCHEMA",
    "ANCHOR_BINDING_SCHEMA",
    "ANCHOR_BINDING_SPEC_SCHEMA",
    "ANCHOR_BINDING_WITNESS_CELL_SCHEMA",
    "ObjectSceneAnchorBindingCatalog",
    "ObjectSceneAnchorBindingError",
    "ObjectSceneAnchorBindingEvaluation",
    "ObjectSceneAnchorBindingSpec",
    "ObjectSceneAnchorCatalogEvaluation",
    "ObjectSceneAnchorWitnessCell",
    "ObjectSceneAnchorWitnessSpec",
    "ObjectSceneResolvedAnchorBinding",
    "build_object_scene_anchor_binding_catalog",
    "compile_object_scene_anchor_binding",
    "compile_object_scene_anchor_catalog",
    "merge_repeated_object_scene_anchor_catalog_evaluations",
)
