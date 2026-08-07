"""Immutable, additive relational scene graphs over visual witness bundles.

This module is deliberately an attachment layer, not another image extractor.
The initial snapshot names the exact :class:`VisualWitnessBundle` and exposes
its scenario-qualified components and holes as entities.  Later visual legs
propose immutable :class:`SceneFragment` values.  ``glue_scene_fragment``
either validates and adds the whole fragment or raises a typed failure without
changing the parent snapshot.

Facts retain ordered arguments.  Thus ``smaller_than(a, b)`` is distinct from
``smaller_than(b, a)`` and a downstream synthesizer cannot accidentally erase
object roles.  Scalar observations are closed intervals with units from the
closed primary-track unit vocabulary.  Failed or uncertain observations use
the same four dispositions as the rest of the Bongard evidence layer; an
extractor error is never converted into absence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition, Provenance
from bongard.legs.contracts import Unit
from bongard.visual_witness_bundle import VisualWitnessBundle


SCENE_SNAPSHOT_SCHEMA = "gkm.bongard-relational-scene-snapshot.v2"
SCENE_FRAGMENT_SCHEMA = "gkm.bongard-relational-scene-fragment.v2"
SCENE_ENTITY_SCHEMA = "gkm.bongard-relational-scene-entity.v1"
SCENE_FACT_SCHEMA = "gkm.bongard-relational-scene-fact.v2"
SCENE_FRAME_Q16 = "panel.q16.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z")
_TYPE_NAME = re.compile(r"[a-z][a-z0-9_.-]*\Z")


def _exact_fields(
    data: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if not isinstance(data, Mapping) or set(data) != expected:
        raise ValueError(f"{label} fields differ from the static schema")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} is not a canonical identifier")
    return value


def _type_name(value: object, label: str) -> str:
    if not isinstance(value, str) or _TYPE_NAME.fullmatch(value) is None:
        raise ValueError(f"{label} is not a canonical type name")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _provenance_to_data(value: Provenance) -> dict[str, object]:
    return {
        "producer": value.producer,
        "version": value.version,
        "method": value.method,
        "input_digests": list(value.input_digests),
        "artifact_digest": value.artifact_digest,
        "run_id": value.run_id,
        "details": [list(item) for item in value.details],
    }


def _provenance_from_data(data: Mapping[str, Any]) -> Provenance:
    _exact_fields(
        data,
        frozenset(
            {
                "producer",
                "version",
                "method",
                "input_digests",
                "artifact_digest",
                "run_id",
                "details",
            }
        ),
        "scene provenance",
    )
    inputs = data["input_digests"]
    details = data["details"]
    if not isinstance(inputs, list) or any(not isinstance(item, str) for item in inputs):
        raise TypeError("scene provenance input_digests must be a string list")
    if not isinstance(details, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or any(not isinstance(part, str) for part in item)
        for item in details
    ):
        raise TypeError("scene provenance details must be string pairs")
    artifact = data["artifact_digest"]
    run_id = data["run_id"]
    if artifact is not None and not isinstance(artifact, str):
        raise TypeError("scene provenance artifact_digest must be a string or null")
    if run_id is not None and not isinstance(run_id, str):
        raise TypeError("scene provenance run_id must be a string or null")
    result = Provenance(
        producer=data["producer"],
        version=data["version"],
        method=data["method"],
        input_digests=tuple(inputs),
        artifact_digest=artifact,
        run_id=run_id,
        details=tuple((item[0], item[1]) for item in details),
    )
    if _provenance_to_data(result) != dict(data):
        raise ValueError("scene provenance is not canonically represented")
    return result


@dataclass(frozen=True, slots=True)
class ScalarInterval:
    """A finite closed interval in one unit from the closed unit vocabulary."""

    lower: float
    upper: float
    unit: Unit

    def __post_init__(self) -> None:
        if type(self.lower) is not float or type(self.upper) is not float:
            raise TypeError("scene interval bounds must be literal canonical floats")
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("scene interval bounds must be finite")
        if self.lower > self.upper:
            raise ValueError("scene interval lower bound exceeds upper bound")
        if not isinstance(self.unit, Unit):
            raise TypeError("scene interval unit must be a Unit")
        if self.unit is Unit.NONE:
            raise ValueError("a scalar interval cannot have the non-scalar unit none")

    @classmethod
    def point(cls, value: float, unit: Unit) -> "ScalarInterval":
        return cls(float(value), float(value), unit)

    def to_data(self) -> dict[str, object]:
        return {"lower": self.lower, "upper": self.upper, "unit": self.unit.value}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ScalarInterval":
        _exact_fields(data, frozenset({"lower", "upper", "unit"}), "scene interval")
        if not isinstance(data["unit"], str):
            raise TypeError("scene interval unit must be a string")
        result = cls(data["lower"], data["upper"], Unit(data["unit"]))
        if result.to_data() != dict(data):
            raise ValueError("scene interval is not canonically represented")
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True, slots=True)
class SceneEntity:
    """One typed, scenario-local entity bound to an exact source witness."""

    entity_id: str
    entity_type: str
    scenario_id: str
    frame_id: str
    source_witness_digest: str
    source_region_digest: str
    provenance_digest: str
    owner_entity_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.entity_id, "scene entity_id")
        _type_name(self.entity_type, "scene entity_type")
        _identifier(self.scenario_id, "scene entity scenario_id")
        _identifier(self.frame_id, "scene entity frame_id")
        _digest(self.source_witness_digest, "scene entity source_witness_digest")
        _digest(self.source_region_digest, "scene entity source_region_digest")
        _digest(self.provenance_digest, "scene entity provenance_digest")
        if self.owner_entity_id is not None:
            _identifier(self.owner_entity_id, "scene entity owner_entity_id")
            if self.owner_entity_id == self.entity_id:
                raise ValueError("a scene entity cannot own itself")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SCENE_ENTITY_SCHEMA,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "scenario_id": self.scenario_id,
            "frame_id": self.frame_id,
            "source_witness_digest": self.source_witness_digest,
            "source_region_digest": self.source_region_digest,
            "provenance_digest": self.provenance_digest,
            "owner_entity_id": self.owner_entity_id,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SceneEntity":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "entity_id",
                    "entity_type",
                    "scenario_id",
                    "frame_id",
                    "source_witness_digest",
                    "source_region_digest",
                    "provenance_digest",
                    "owner_entity_id",
                }
            ),
            "scene entity",
        )
        if data["schema"] != SCENE_ENTITY_SCHEMA:
            raise ValueError("unsupported scene entity schema")
        owner = data["owner_entity_id"]
        if owner is not None and not isinstance(owner, str):
            raise TypeError("scene entity owner_entity_id must be a string or null")
        result = cls(
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            scenario_id=data["scenario_id"],
            frame_id=data["frame_id"],
            source_witness_digest=data["source_witness_digest"],
            source_region_digest=data["source_region_digest"],
            provenance_digest=data["provenance_digest"],
            owner_entity_id=owner,
        )
        if result.to_data() != dict(data):
            raise ValueError("scene entity is not canonically represented")
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True, slots=True)
class SceneFact:
    """A typed observation whose argument order is semantically significant."""

    fact_id: str
    predicate: str
    arguments: tuple[str, ...]
    argument_types: tuple[str, ...]
    scenario_id: str
    frame_id: str
    disposition: Disposition
    provenance_digest: str
    source_region_digests: tuple[str, ...]
    unit: Unit = Unit.NONE
    interval: ScalarInterval | None = None
    certificate: str | None = None
    reason: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.fact_id, "scene fact_id")
        _type_name(self.predicate, "scene predicate")
        if not isinstance(self.arguments, tuple) or not self.arguments:
            raise ValueError("scene fact arguments must be a non-empty tuple")
        if any(_IDENTIFIER.fullmatch(item) is None for item in self.arguments):
            raise ValueError("scene fact contains a non-canonical argument ID")
        if not isinstance(self.argument_types, tuple) or len(self.argument_types) != len(
            self.arguments
        ):
            raise ValueError("scene fact argument types must align with arguments")
        for value in self.argument_types:
            _type_name(value, "scene fact argument type")
        _identifier(self.scenario_id, "scene fact scenario_id")
        _identifier(self.frame_id, "scene fact frame_id")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("scene fact disposition must be a Disposition")
        _digest(self.provenance_digest, "scene fact provenance_digest")
        if not isinstance(self.source_region_digests, tuple) or not self.source_region_digests:
            raise ValueError("scene fact requires source-region digests")
        for value in self.source_region_digests:
            _digest(value, "scene fact source_region_digest")
        if len(self.source_region_digests) != len(set(self.source_region_digests)):
            raise ValueError("scene fact source-region digests must be unique")
        if not isinstance(self.unit, Unit):
            raise TypeError("scene fact unit must be a Unit")
        if self.interval is not None and not isinstance(self.interval, ScalarInterval):
            raise TypeError("scene fact interval must be a ScalarInterval or null")
        if self.interval is not None and self.interval.unit is not self.unit:
            raise ValueError("scene fact interval unit differs from its declared unit")
        if self.unit is Unit.NONE and self.interval is not None:
            raise ValueError("a non-scalar scene fact cannot carry an interval")

        if self.disposition is Disposition.PRESENT:
            if any(value is not None for value in (self.certificate, self.reason, self.error_type)):
                raise ValueError("present scene fact cannot carry failure fields")
            if self.unit is not Unit.NONE and self.interval is None:
                raise ValueError("present scalar scene fact requires an interval")
        elif self.disposition is Disposition.CERTIFIED_ABSENT:
            if self.certificate is None or not self.certificate.strip():
                raise ValueError("certified-absent scene fact requires a certificate")
            if self.interval is not None or self.reason is not None or self.error_type is not None:
                raise ValueError("certified-absent scene fact has incompatible fields")
        elif self.disposition is Disposition.INDETERMINATE:
            if self.reason is None or not self.reason.strip():
                raise ValueError("indeterminate scene fact requires a reason")
            if self.certificate is not None or self.error_type is not None:
                raise ValueError("indeterminate scene fact has incompatible fields")
        elif self.disposition is Disposition.ERROR:
            if self.reason is None or not self.reason.strip():
                raise ValueError("error scene fact requires a reason")
            if self.error_type is None or not self.error_type.strip():
                raise ValueError("error scene fact requires an error_type")
            if self.interval is not None or self.certificate is not None:
                raise ValueError("error scene fact has incompatible fields")

    @property
    def logical_key(self) -> tuple[str, tuple[str, ...], str, str]:
        return (self.predicate, self.arguments, self.scenario_id, self.frame_id)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SCENE_FACT_SCHEMA,
            "fact_id": self.fact_id,
            "predicate": self.predicate,
            "arguments": list(self.arguments),
            "argument_types": list(self.argument_types),
            "scenario_id": self.scenario_id,
            "frame_id": self.frame_id,
            "disposition": self.disposition.value,
            "provenance_digest": self.provenance_digest,
            "source_region_digests": list(self.source_region_digests),
            "unit": self.unit.value,
            "interval": None if self.interval is None else self.interval.to_data(),
            "certificate": self.certificate,
            "reason": self.reason,
            "error_type": self.error_type,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SceneFact":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "fact_id",
                    "predicate",
                    "arguments",
                    "argument_types",
                    "scenario_id",
                    "frame_id",
                    "disposition",
                    "provenance_digest",
                    "source_region_digests",
                    "unit",
                    "interval",
                    "certificate",
                    "reason",
                    "error_type",
                }
            ),
            "scene fact",
        )
        if data["schema"] != SCENE_FACT_SCHEMA:
            raise ValueError("unsupported scene fact schema")
        arguments = data["arguments"]
        argument_types = data["argument_types"]
        regions = data["source_region_digests"]
        if not isinstance(arguments, list) or any(not isinstance(x, str) for x in arguments):
            raise TypeError("scene fact arguments must be a string list")
        if not isinstance(argument_types, list) or any(
            not isinstance(x, str) for x in argument_types
        ):
            raise TypeError("scene fact argument_types must be a string list")
        if not isinstance(regions, list) or any(not isinstance(x, str) for x in regions):
            raise TypeError("scene fact source_region_digests must be a string list")
        interval_data = data["interval"]
        if interval_data is not None and not isinstance(interval_data, Mapping):
            raise TypeError("scene fact interval must be an object or null")
        disposition = data["disposition"]
        if not isinstance(disposition, str):
            raise TypeError("scene fact disposition must be a string")
        if not isinstance(data["unit"], str):
            raise TypeError("scene fact unit must be a string")
        for name in ("certificate", "reason", "error_type"):
            if data[name] is not None and not isinstance(data[name], str):
                raise TypeError(f"scene fact {name} must be a string or null")
        result = cls(
            fact_id=data["fact_id"],
            predicate=data["predicate"],
            arguments=tuple(arguments),
            argument_types=tuple(argument_types),
            scenario_id=data["scenario_id"],
            frame_id=data["frame_id"],
            disposition=Disposition(disposition),
            provenance_digest=data["provenance_digest"],
            source_region_digests=tuple(regions),
            unit=Unit(data["unit"]),
            interval=(
                None if interval_data is None else ScalarInterval.from_data(interval_data)
            ),
            certificate=data["certificate"],
            reason=data["reason"],
            error_type=data["error_type"],
        )
        if result.to_data() != dict(data):
            raise ValueError("scene fact is not canonically represented")
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _validate_ordered_provenances(values: tuple[Provenance, ...], label: str) -> None:
    if not isinstance(values, tuple) or any(not isinstance(item, Provenance) for item in values):
        raise TypeError(f"{label} provenances must be a typed tuple")
    digests = tuple(item.digest() for item in values)
    if digests != tuple(sorted(digests)) or len(digests) != len(set(digests)):
        raise ValueError(f"{label} provenances must be unique and digest-sorted")


def _validate_container_order(
    entities: tuple[SceneEntity, ...], facts: tuple[SceneFact, ...], label: str
) -> None:
    if not isinstance(entities, tuple) or any(not isinstance(item, SceneEntity) for item in entities):
        raise TypeError(f"{label} entities must be a typed tuple")
    if not isinstance(facts, tuple) or any(not isinstance(item, SceneFact) for item in facts):
        raise TypeError(f"{label} facts must be a typed tuple")
    if tuple(item.entity_id for item in entities) != tuple(
        sorted(item.entity_id for item in entities)
    ):
        raise ValueError(f"{label} entities must be entity-ID sorted")
    if tuple(item.fact_id for item in facts) != tuple(sorted(item.fact_id for item in facts)):
        raise ValueError(f"{label} facts must be fact-ID sorted")


def _fact_unit(fact: SceneFact) -> Unit:
    return fact.unit


def _validate_complete_graph(
    *,
    scenario_ids: tuple[str, ...],
    frame_id: str,
    entities: tuple[SceneEntity, ...],
    facts: tuple[SceneFact, ...],
    provenances: tuple[Provenance, ...],
) -> None:
    provenance_ids = {item.digest() for item in provenances}
    entity_by_id: dict[str, SceneEntity] = {}
    for entity in entities:
        if entity.entity_id in entity_by_id:
            raise ValueError("scene snapshot has duplicate entity IDs")
        entity_by_id[entity.entity_id] = entity
        if entity.scenario_id not in scenario_ids:
            raise ValueError("scene entity names a scenario outside the snapshot")
        if entity.frame_id != frame_id:
            raise ValueError("scene entity frame differs from its snapshot")
        if entity.provenance_digest not in provenance_ids:
            raise ValueError("scene entity names missing provenance")
    for entity in entities:
        if entity.owner_entity_id is None:
            continue
        owner = entity_by_id.get(entity.owner_entity_id)
        if owner is None:
            raise ValueError("scene entity owner is missing")
        if owner.scenario_id != entity.scenario_id or owner.frame_id != entity.frame_id:
            raise ValueError("scene entity owner crosses a scenario or frame")

    fact_ids: set[str] = set()
    logical_keys: set[tuple[str, tuple[str, ...], str, str]] = set()
    predicate_signatures: dict[str, tuple[tuple[str, ...], Unit]] = {}
    for fact in facts:
        if fact.fact_id in fact_ids or fact.logical_key in logical_keys:
            raise ValueError("scene snapshot has duplicate/conflicting facts")
        fact_ids.add(fact.fact_id)
        logical_keys.add(fact.logical_key)
        if fact.scenario_id not in scenario_ids:
            raise ValueError("scene fact names a scenario outside the snapshot")
        if fact.frame_id != frame_id:
            raise ValueError("scene fact frame differs from its snapshot")
        if fact.provenance_digest not in provenance_ids:
            raise ValueError("scene fact names missing provenance")
        for argument, expected_type in zip(
            fact.arguments, fact.argument_types, strict=True
        ):
            entity = entity_by_id.get(argument)
            if entity is None:
                raise ValueError("scene fact argument entity is missing")
            if entity.entity_type != expected_type:
                raise ValueError("scene fact argument type differs from its entity")
            if entity.scenario_id != fact.scenario_id:
                raise ValueError("scene fact argument crosses a scenario")
        signature = (fact.argument_types, _fact_unit(fact))
        previous = predicate_signatures.setdefault(fact.predicate, signature)
        if previous != signature:
            raise ValueError("scene predicate type/unit signature is inconsistent")


@dataclass(frozen=True, slots=True)
class SceneSnapshot:
    """A content-addressed immutable scene state."""

    panel_digest: str
    parent_bundle_digest: str
    graph_schema_digest: str
    frame_id: str
    scenario_ids: tuple[str, ...]
    generation: int
    previous_snapshot_digest: str | None
    applied_fragment_digest: str | None
    provenances: tuple[Provenance, ...]
    entities: tuple[SceneEntity, ...]
    facts: tuple[SceneFact, ...]

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "scene panel_digest")
        _digest(self.parent_bundle_digest, "scene parent_bundle_digest")
        _digest(self.graph_schema_digest, "scene graph_schema_digest")
        _identifier(self.frame_id, "scene frame_id")
        if not isinstance(self.scenario_ids, tuple) or not self.scenario_ids:
            raise ValueError("scene scenario_ids must be a non-empty tuple")
        for scenario_id in self.scenario_ids:
            _identifier(scenario_id, "scene scenario_id")
        if self.scenario_ids != tuple(sorted(set(self.scenario_ids))):
            raise ValueError("scene scenario_ids must be unique and sorted")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise TypeError("scene generation must be an integer")
        if self.generation < 0:
            raise ValueError("scene generation cannot be negative")
        if self.generation == 0:
            if self.previous_snapshot_digest is not None:
                raise ValueError("initial scene cannot name a previous snapshot")
            if self.applied_fragment_digest is not None:
                raise ValueError("initial scene cannot name an applied fragment")
        else:
            _digest(self.previous_snapshot_digest, "scene previous_snapshot_digest")
            _digest(self.applied_fragment_digest, "scene applied_fragment_digest")
        _validate_ordered_provenances(self.provenances, "scene snapshot")
        _validate_container_order(self.entities, self.facts, "scene snapshot")
        _validate_complete_graph(
            scenario_ids=self.scenario_ids,
            frame_id=self.frame_id,
            entities=self.entities,
            facts=self.facts,
            provenances=self.provenances,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SCENE_SNAPSHOT_SCHEMA,
            "panel_digest": self.panel_digest,
            "parent_bundle_digest": self.parent_bundle_digest,
            "graph_schema_digest": self.graph_schema_digest,
            "frame_id": self.frame_id,
            "scenario_ids": list(self.scenario_ids),
            "generation": self.generation,
            "previous_snapshot_digest": self.previous_snapshot_digest,
            "applied_fragment_digest": self.applied_fragment_digest,
            "provenances": [_provenance_to_data(item) for item in self.provenances],
            "entities": [item.to_data() for item in self.entities],
            "facts": [item.to_data() for item in self.facts],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SceneSnapshot":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "panel_digest",
                    "parent_bundle_digest",
                    "graph_schema_digest",
                    "frame_id",
                    "scenario_ids",
                    "generation",
                    "previous_snapshot_digest",
                    "applied_fragment_digest",
                    "provenances",
                    "entities",
                    "facts",
                }
            ),
            "scene snapshot",
        )
        if data["schema"] != SCENE_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported scene snapshot schema")
        scenarios = data["scenario_ids"]
        provenances = data["provenances"]
        entities = data["entities"]
        facts = data["facts"]
        if not isinstance(scenarios, list) or any(not isinstance(x, str) for x in scenarios):
            raise TypeError("scene snapshot scenario_ids must be a string list")
        for values, label in (
            (provenances, "provenances"),
            (entities, "entities"),
            (facts, "facts"),
        ):
            if not isinstance(values, list) or any(not isinstance(x, Mapping) for x in values):
                raise TypeError(f"scene snapshot {label} must be an object list")
        previous = data["previous_snapshot_digest"]
        if previous is not None and not isinstance(previous, str):
            raise TypeError("scene previous_snapshot_digest must be a string or null")
        applied = data["applied_fragment_digest"]
        if applied is not None and not isinstance(applied, str):
            raise TypeError("scene applied_fragment_digest must be a string or null")
        result = cls(
            panel_digest=data["panel_digest"],
            parent_bundle_digest=data["parent_bundle_digest"],
            graph_schema_digest=data["graph_schema_digest"],
            frame_id=data["frame_id"],
            scenario_ids=tuple(scenarios),
            generation=data["generation"],
            previous_snapshot_digest=previous,
            applied_fragment_digest=applied,
            provenances=tuple(_provenance_from_data(item) for item in provenances),
            entities=tuple(SceneEntity.from_data(item) for item in entities),
            facts=tuple(SceneFact.from_data(item) for item in facts),
        )
        if result.to_data() != dict(data):
            raise ValueError("scene snapshot is not canonically represented")
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True, slots=True)
class SceneFragment:
    """One proposed additive attachment against an exact parent snapshot."""

    panel_digest: str
    parent_bundle_digest: str
    parent_snapshot_digest: str
    graph_schema_digest: str
    frame_id: str
    scenario_ids: tuple[str, ...]
    producer_leg: str
    producer_leg_digest: str
    provenances: tuple[Provenance, ...] = ()
    entities: tuple[SceneEntity, ...] = ()
    facts: tuple[SceneFact, ...] = ()
    leg_error_type: str | None = None
    leg_error_reason: str | None = None

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "fragment panel_digest")
        _digest(self.parent_bundle_digest, "fragment parent_bundle_digest")
        _digest(self.parent_snapshot_digest, "fragment parent_snapshot_digest")
        _digest(self.graph_schema_digest, "fragment graph_schema_digest")
        _identifier(self.frame_id, "fragment frame_id")
        if not isinstance(self.scenario_ids, tuple) or not self.scenario_ids:
            raise ValueError("fragment scenario_ids must be a non-empty tuple")
        for scenario_id in self.scenario_ids:
            _identifier(scenario_id, "fragment scenario_id")
        if self.scenario_ids != tuple(sorted(set(self.scenario_ids))):
            raise ValueError("fragment scenario_ids must be unique and sorted")
        _type_name(self.producer_leg, "fragment producer_leg")
        _digest(self.producer_leg_digest, "fragment producer_leg_digest")
        _validate_ordered_provenances(self.provenances, "scene fragment")
        _validate_container_order(self.entities, self.facts, "scene fragment")
        _optional_string(self.leg_error_type, "fragment leg_error_type")
        _optional_string(self.leg_error_reason, "fragment leg_error_reason")
        if (self.leg_error_type is None) != (self.leg_error_reason is None):
            raise ValueError("fragment leg error requires both type and reason")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SCENE_FRAGMENT_SCHEMA,
            "panel_digest": self.panel_digest,
            "parent_bundle_digest": self.parent_bundle_digest,
            "parent_snapshot_digest": self.parent_snapshot_digest,
            "graph_schema_digest": self.graph_schema_digest,
            "frame_id": self.frame_id,
            "scenario_ids": list(self.scenario_ids),
            "producer_leg": self.producer_leg,
            "producer_leg_digest": self.producer_leg_digest,
            "provenances": [_provenance_to_data(item) for item in self.provenances],
            "entities": [item.to_data() for item in self.entities],
            "facts": [item.to_data() for item in self.facts],
            "leg_error_type": self.leg_error_type,
            "leg_error_reason": self.leg_error_reason,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SceneFragment":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "panel_digest",
                    "parent_bundle_digest",
                    "parent_snapshot_digest",
                    "graph_schema_digest",
                    "frame_id",
                    "scenario_ids",
                    "producer_leg",
                    "producer_leg_digest",
                    "provenances",
                    "entities",
                    "facts",
                    "leg_error_type",
                    "leg_error_reason",
                }
            ),
            "scene fragment",
        )
        if data["schema"] != SCENE_FRAGMENT_SCHEMA:
            raise ValueError("unsupported scene fragment schema")
        scenarios = data["scenario_ids"]
        provenances = data["provenances"]
        entities = data["entities"]
        facts = data["facts"]
        if not isinstance(scenarios, list) or any(not isinstance(x, str) for x in scenarios):
            raise TypeError("scene fragment scenario_ids must be a string list")
        for values, label in (
            (provenances, "provenances"),
            (entities, "entities"),
            (facts, "facts"),
        ):
            if not isinstance(values, list) or any(not isinstance(x, Mapping) for x in values):
                raise TypeError(f"scene fragment {label} must be an object list")
        for name in ("leg_error_type", "leg_error_reason"):
            if data[name] is not None and not isinstance(data[name], str):
                raise TypeError(f"scene fragment {name} must be a string or null")
        result = cls(
            panel_digest=data["panel_digest"],
            parent_bundle_digest=data["parent_bundle_digest"],
            parent_snapshot_digest=data["parent_snapshot_digest"],
            graph_schema_digest=data["graph_schema_digest"],
            frame_id=data["frame_id"],
            scenario_ids=tuple(scenarios),
            producer_leg=data["producer_leg"],
            producer_leg_digest=data["producer_leg_digest"],
            provenances=tuple(_provenance_from_data(item) for item in provenances),
            entities=tuple(SceneEntity.from_data(item) for item in entities),
            facts=tuple(SceneFact.from_data(item) for item in facts),
            leg_error_type=data["leg_error_type"],
            leg_error_reason=data["leg_error_reason"],
        )
        if result.to_data() != dict(data):
            raise ValueError("scene fragment is not canonically represented")
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


class SceneGlueFailureCode(str, Enum):
    PANEL_MISMATCH = "panel_mismatch"
    PARENT_MISMATCH = "parent_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    FRAME_MISMATCH = "frame_mismatch"
    ENTITY_CONFLICT = "entity_conflict"
    OWNER_CONFLICT = "owner_conflict"
    DUPLICATE_FACT = "duplicate_fact"
    CONFLICTING_FACT = "conflicting_fact"
    UNIT_MISMATCH = "unit_mismatch"
    MISSING_ENTITY = "missing_entity"
    MISSING_PROVENANCE = "missing_provenance"
    LEG_ERROR = "leg_error"


class SceneGlueError(RuntimeError):
    """Typed, fail-closed rejection of a proposed scene attachment."""

    def __init__(self, code: SceneGlueFailureCode, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


def _raise(code: SceneGlueFailureCode, detail: str) -> None:
    raise SceneGlueError(code, detail)


def _bundle_provenance(bundle: VisualWitnessBundle) -> Provenance:
    return Provenance(
        producer="bongard.visual_witness_bundle",
        version="1",
        method="scene-foundation",
        input_digests=(bundle.panel_digest, bundle.digest()),
        artifact_digest=bundle.assembler_artifact_digest,
        details=(
            ("frame_id", SCENE_FRAME_Q16),
            ("scenario_count", str(len(bundle.base_packet.scenarios))),
        ),
    )


def _foundation_entities(
    bundle: VisualWitnessBundle, provenance_digest: str, frame_id: str
) -> tuple[SceneEntity, ...]:
    entities: list[SceneEntity] = []
    for scenario in bundle.base_packet.scenarios:
        component_ids: dict[str, str] = {}
        for component in scenario.components:
            entity_id = f"{scenario.scenario_id}/component/{component.component_id}"
            component_ids[component.component_id] = entity_id
            entities.append(
                SceneEntity(
                    entity_id=entity_id,
                    entity_type="component",
                    scenario_id=scenario.scenario_id,
                    frame_id=frame_id,
                    source_witness_digest=canonical_digest(component.to_data()),
                    source_region_digest=component.mask_digest,
                    provenance_digest=provenance_digest,
                )
            )
        for hole in scenario.holes:
            entities.append(
                SceneEntity(
                    entity_id=f"{scenario.scenario_id}/hole/{hole.hole_id}",
                    entity_type="hole",
                    scenario_id=scenario.scenario_id,
                    frame_id=frame_id,
                    source_witness_digest=canonical_digest(hole.to_data()),
                    source_region_digest=hole.mask_digest,
                    provenance_digest=provenance_digest,
                    owner_entity_id=(
                        None
                        if hole.owner_component_id is None
                        else component_ids[hole.owner_component_id]
                    ),
                )
            )
    return tuple(sorted(entities, key=lambda item: item.entity_id))


def start_scene_snapshot(
    bundle: VisualWitnessBundle,
    graph_schema_digest: str,
    *,
    frame_id: str = SCENE_FRAME_Q16,
) -> SceneSnapshot:
    """Create generation zero from one exact immutable visual bundle."""

    if not isinstance(bundle, VisualWitnessBundle):
        raise TypeError("scene foundation must be a VisualWitnessBundle")
    # Cold-decode the bundle representation.  This detects malformed in-memory
    # values without depending on mutable image bytes or running extraction.
    if VisualWitnessBundle.from_data(bundle.to_data()) != bundle:
        raise ValueError("scene foundation bundle is not canonically represented")
    _digest(graph_schema_digest, "scene graph_schema_digest")
    _identifier(frame_id, "scene frame_id")
    provenance = _bundle_provenance(bundle)
    return SceneSnapshot(
        panel_digest=bundle.panel_digest,
        parent_bundle_digest=bundle.digest(),
        graph_schema_digest=graph_schema_digest,
        frame_id=frame_id,
        scenario_ids=tuple(item.scenario_id for item in bundle.base_packet.scenarios),
        generation=0,
        previous_snapshot_digest=None,
        applied_fragment_digest=None,
        provenances=(provenance,),
        entities=_foundation_entities(bundle, provenance.digest(), frame_id),
        facts=(),
    )


def glue_scene_fragment(parent: SceneSnapshot, fragment: SceneFragment) -> SceneSnapshot:
    """Atomically attach ``fragment`` or raise :class:`SceneGlueError`.

    All validation is performed against local dictionaries and tuples.  The
    new :class:`SceneSnapshot` is constructed only after every check passes;
    the immutable ``parent`` can therefore never contain a partial update.
    """

    if not isinstance(parent, SceneSnapshot) or not isinstance(fragment, SceneFragment):
        raise TypeError("scene gluing requires a SceneSnapshot and SceneFragment")
    if fragment.panel_digest != parent.panel_digest:
        _raise(SceneGlueFailureCode.PANEL_MISMATCH, "fragment names another panel")
    if (
        fragment.parent_snapshot_digest != parent.digest()
        or fragment.parent_bundle_digest != parent.parent_bundle_digest
    ):
        _raise(SceneGlueFailureCode.PARENT_MISMATCH, "fragment names another parent")
    if fragment.graph_schema_digest != parent.graph_schema_digest:
        _raise(SceneGlueFailureCode.SCHEMA_MISMATCH, "graph schema digests differ")
    if fragment.frame_id != parent.frame_id:
        _raise(SceneGlueFailureCode.FRAME_MISMATCH, "coordinate frames differ")
    if fragment.scenario_ids != parent.scenario_ids:
        _raise(SceneGlueFailureCode.SCHEMA_MISMATCH, "scenario inventories differ")
    if fragment.leg_error_type is not None:
        _raise(
            SceneGlueFailureCode.LEG_ERROR,
            f"{fragment.leg_error_type}: {fragment.leg_error_reason}",
        )

    provenance_by_digest = {item.digest(): item for item in parent.provenances}
    for provenance in fragment.provenances:
        provenance_by_digest.setdefault(provenance.digest(), provenance)

    entity_by_id = {item.entity_id: item for item in parent.entities}
    for entity in fragment.entities:
        previous = entity_by_id.get(entity.entity_id)
        if previous is not None:
            if previous.owner_entity_id != entity.owner_entity_id:
                _raise(
                    SceneGlueFailureCode.OWNER_CONFLICT,
                    f"entity {entity.entity_id} changes owner",
                )
            _raise(
                SceneGlueFailureCode.ENTITY_CONFLICT,
                f"entity {entity.entity_id} already exists",
            )
        entity_by_id[entity.entity_id] = entity

    for entity in fragment.entities:
        if entity.scenario_id not in parent.scenario_ids:
            _raise(SceneGlueFailureCode.SCHEMA_MISMATCH, "entity scenario is undeclared")
        if entity.frame_id != parent.frame_id:
            _raise(SceneGlueFailureCode.FRAME_MISMATCH, "entity frame differs")
        if entity.provenance_digest not in provenance_by_digest:
            _raise(
                SceneGlueFailureCode.MISSING_PROVENANCE,
                f"entity {entity.entity_id} provenance is missing",
            )
        if entity.owner_entity_id is not None:
            owner = entity_by_id.get(entity.owner_entity_id)
            if owner is None:
                _raise(
                    SceneGlueFailureCode.MISSING_ENTITY,
                    f"owner {entity.owner_entity_id} is missing",
                )
            if owner.scenario_id != entity.scenario_id or owner.frame_id != entity.frame_id:
                _raise(
                    SceneGlueFailureCode.OWNER_CONFLICT,
                    f"owner of {entity.entity_id} crosses scenario/frame",
                )

    fact_by_id = {item.fact_id: item for item in parent.facts}
    fact_by_key = {item.logical_key: item for item in parent.facts}
    predicate_signatures: dict[str, tuple[tuple[str, ...], Unit]] = {
        item.predicate: (item.argument_types, _fact_unit(item)) for item in parent.facts
    }
    for fact in fragment.facts:
        same_id = fact_by_id.get(fact.fact_id)
        same_key = fact_by_key.get(fact.logical_key)
        if same_id is not None or same_key is not None:
            previous = same_id if same_id is not None else same_key
            assert previous is not None
            if previous == fact:
                _raise(
                    SceneGlueFailureCode.DUPLICATE_FACT,
                    f"fact {fact.fact_id} is already present",
                )
            _raise(
                SceneGlueFailureCode.CONFLICTING_FACT,
                f"fact {fact.fact_id} conflicts with an existing logical fact",
            )
        if fact.scenario_id not in parent.scenario_ids:
            _raise(SceneGlueFailureCode.SCHEMA_MISMATCH, "fact scenario is undeclared")
        if fact.frame_id != parent.frame_id:
            _raise(SceneGlueFailureCode.FRAME_MISMATCH, "fact frame differs")
        if fact.provenance_digest not in provenance_by_digest:
            _raise(
                SceneGlueFailureCode.MISSING_PROVENANCE,
                f"fact {fact.fact_id} provenance is missing",
            )
        for argument, expected_type in zip(
            fact.arguments, fact.argument_types, strict=True
        ):
            entity = entity_by_id.get(argument)
            if entity is None:
                _raise(
                    SceneGlueFailureCode.MISSING_ENTITY,
                    f"fact {fact.fact_id} argument {argument} is missing",
                )
            if entity.entity_type != expected_type or entity.scenario_id != fact.scenario_id:
                _raise(
                    SceneGlueFailureCode.SCHEMA_MISMATCH,
                    f"fact {fact.fact_id} argument type/scenario differs",
                )
        signature = (fact.argument_types, _fact_unit(fact))
        previous_signature = predicate_signatures.get(fact.predicate)
        if previous_signature is not None:
            if previous_signature[0] != signature[0]:
                _raise(
                    SceneGlueFailureCode.SCHEMA_MISMATCH,
                    f"predicate {fact.predicate} argument types differ",
                )
            if previous_signature[1] != signature[1]:
                _raise(
                    SceneGlueFailureCode.UNIT_MISMATCH,
                    f"predicate {fact.predicate} unit differs",
                )
        predicate_signatures[fact.predicate] = signature
        fact_by_id[fact.fact_id] = fact
        fact_by_key[fact.logical_key] = fact

    return SceneSnapshot(
        panel_digest=parent.panel_digest,
        parent_bundle_digest=parent.parent_bundle_digest,
        graph_schema_digest=parent.graph_schema_digest,
        frame_id=parent.frame_id,
        scenario_ids=parent.scenario_ids,
        generation=parent.generation + 1,
        previous_snapshot_digest=parent.digest(),
        applied_fragment_digest=fragment.digest(),
        provenances=tuple(sorted(provenance_by_digest.values(), key=lambda item: item.digest())),
        entities=tuple(sorted(entity_by_id.values(), key=lambda item: item.entity_id)),
        facts=tuple(sorted(fact_by_id.values(), key=lambda item: item.fact_id)),
    )


def verify_scene_snapshot(
    snapshot: SceneSnapshot,
    bundle: VisualWitnessBundle,
    *,
    previous_snapshot: SceneSnapshot | None = None,
    applied_fragment: SceneFragment | None = None,
) -> SceneSnapshot:
    """Cold-check canonical bytes, foundation binding, and optional lineage."""

    if not isinstance(snapshot, SceneSnapshot):
        raise TypeError("snapshot must be a SceneSnapshot")
    if not isinstance(bundle, VisualWitnessBundle):
        raise TypeError("bundle must be a VisualWitnessBundle")
    if SceneSnapshot.from_data(snapshot.to_data()) != snapshot:
        raise ValueError("scene snapshot is not canonically represented")
    if VisualWitnessBundle.from_data(bundle.to_data()) != bundle:
        raise ValueError("parent visual bundle is not canonically represented")
    if snapshot.panel_digest != bundle.panel_digest:
        raise ValueError("scene snapshot panel differs from parent bundle")
    if snapshot.parent_bundle_digest != bundle.digest():
        raise ValueError("scene snapshot does not bind the exact parent bundle")
    if snapshot.scenario_ids != tuple(
        item.scenario_id for item in bundle.base_packet.scenarios
    ):
        raise ValueError("scene snapshot scenario inventory differs from parent bundle")

    foundation_provenance = _bundle_provenance(bundle)
    provenance_by_digest = {item.digest(): item for item in snapshot.provenances}
    if provenance_by_digest.get(foundation_provenance.digest()) != foundation_provenance:
        raise ValueError("scene snapshot is missing its exact bundle provenance")
    entity_by_id = {item.entity_id: item for item in snapshot.entities}
    for expected in _foundation_entities(
        bundle, foundation_provenance.digest(), snapshot.frame_id
    ):
        if entity_by_id.get(expected.entity_id) != expected:
            raise ValueError("scene snapshot changed a foundation witness entity")

    if snapshot.generation == 0:
        expected = start_scene_snapshot(
            bundle, snapshot.graph_schema_digest, frame_id=snapshot.frame_id
        )
        if snapshot != expected:
            raise ValueError("generation-zero scene differs from its exact foundation")
        if previous_snapshot is not None:
            raise ValueError("generation-zero scene cannot have a previous snapshot")
        if applied_fragment is not None:
            raise ValueError("generation-zero scene cannot have an applied fragment")
    elif previous_snapshot is not None:
        if snapshot.previous_snapshot_digest != previous_snapshot.digest():
            raise ValueError("scene lineage does not bind the supplied previous snapshot")
        if snapshot.generation != previous_snapshot.generation + 1:
            raise ValueError("scene generation is not the previous generation plus one")
        for name in (
            "panel_digest",
            "parent_bundle_digest",
            "graph_schema_digest",
            "frame_id",
            "scenario_ids",
        ):
            if getattr(snapshot, name) != getattr(previous_snapshot, name):
                raise ValueError(f"scene lineage changed immutable boundary {name}")
        current_provenances = {item.digest(): item for item in snapshot.provenances}
        if any(
            current_provenances.get(item.digest()) != item
            for item in previous_snapshot.provenances
        ):
            raise ValueError("scene lineage removed or changed provenance")
        current_entities = {item.entity_id: item for item in snapshot.entities}
        if any(current_entities.get(item.entity_id) != item for item in previous_snapshot.entities):
            raise ValueError("scene lineage removed or changed an entity")
        current_facts = {item.fact_id: item for item in snapshot.facts}
        if any(current_facts.get(item.fact_id) != item for item in previous_snapshot.facts):
            raise ValueError("scene lineage removed or changed a fact")
        if applied_fragment is not None:
            if snapshot.applied_fragment_digest != applied_fragment.digest():
                raise ValueError("scene lineage does not bind the supplied fragment")
            if glue_scene_fragment(previous_snapshot, applied_fragment) != snapshot:
                raise ValueError("scene snapshot differs from deterministic fragment replay")
    elif applied_fragment is not None:
        raise ValueError("fragment replay requires the previous snapshot")
    return snapshot
