"""Cold-replayable loop scene packet attached to one visual witness bundle.

This orchestrator retains every bounded background region discovered by the
base extractor.  It never assumes that a panel contains exactly two objects
and never deletes small holes.  Each source ``HoleWitness`` receives one
scenario-local loop identity and one candidate-independent geometry witness.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition, Provenance
from bongard.legs.contracts import Unit, ValueType
from bongard.loop_geometry import (
    IntInterval,
    LoopGeometryWitness,
    extract_loop_geometry,
    loop_geometry_algorithm_digest,
)
from bongard.point_contact import (
    PairContactObservation,
    extract_pair_contact_observations,
    point_contact_algorithm_digest,
)
from bongard.relational_scene import (
    ScalarInterval,
    SceneEntity,
    SceneFact,
    SceneFragment,
    SceneSnapshot,
)
from bongard.visual_witness_bundle import (
    VisualWitnessBundle,
    extract_visual_witness_bundle,
    verify_visual_witness_bundle,
    visual_witness_bundle_extractor_digest,
)
from bongard import visual_witnesses as _base


LOOP_SCENE_PACKET = ValueType("loop_scene_packet")
LOOP_SCENE_PACKET_SCHEMA = "gkm.bongard-loop-scene-packet.v2"
LOOP_SCENE_SCENARIO_SCHEMA = "gkm.bongard-loop-scene-scenario.v2"
LOOP_SCENE_ALGORITHM_ID = "bongard.loop-scene-orchestrator/v2"
LOOP_SCENE_EXTRACTOR_ID = "bongard.loop_scene_witnesses"
LOOP_SCENE_EXTRACTOR_VERSION = "2"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _exact_fields(
    data: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if not isinstance(data, Mapping) or set(data) != expected:
        raise ValueError(f"{label} fields differ from the static schema")


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256")
    return value


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _artifact_digest(
    source_digest: str,
    parent_bundle_extractor_digest: str,
    geometry_algorithm_digest: str,
    contact_algorithm_digest: str,
) -> str:
    return canonical_digest(
        {
            "algorithm_id": LOOP_SCENE_ALGORITHM_ID,
            "source_digest": source_digest,
            "parent_bundle_extractor_digest": parent_bundle_extractor_digest,
            "loop_geometry_algorithm_digest": geometry_algorithm_digest,
            "point_contact_algorithm_digest": contact_algorithm_digest,
            "scenario_ids": list(_base.VISUAL_WITNESS_SCENARIO_IDS),
            "enumeration": "all bounded background holes; no hidden filtering",
            "identity": "scenario-local loop ordinal preserves source hole ordinal",
        }
    )


def loop_scene_extractor_digest() -> str:
    return _artifact_digest(
        _source_digest(),
        visual_witness_bundle_extractor_digest(),
        loop_geometry_algorithm_digest(),
        point_contact_algorithm_digest(),
    )


def loop_scene_catalog_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-loop-scene-catalog.v1",
            "extractor_id": LOOP_SCENE_EXTRACTOR_ID,
            "extractor_version": LOOP_SCENE_EXTRACTOR_VERSION,
            "packet_type": LOOP_SCENE_PACKET.to_data(),
            "entity_type": "loop",
            "predicate_ids": [
                "loop.area_pixels",
                "loop.edge_axis_obliqueness_millidegrees",
                "loop.polygon_side_count",
                "loop.substantive",
                "pair.point_contact",
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class LoopSceneScenarioWitness:
    scenario_id: str
    foreground_strength_threshold: int
    morphology: str
    loops: tuple[LoopGeometryWitness, ...]
    contacts: tuple[PairContactObservation, ...]

    def __post_init__(self) -> None:
        expected = {item[0]: item[1:] for item in _base._SCENARIOS}
        if self.scenario_id not in expected:
            raise ValueError("unknown loop scene scenario_id")
        if (self.foreground_strength_threshold, self.morphology) != expected[
            self.scenario_id
        ]:
            raise ValueError("loop scene parameters differ from frozen scenario")
        if not isinstance(self.loops, tuple) or any(
            not isinstance(item, LoopGeometryWitness) for item in self.loops
        ):
            raise TypeError("loop scene loops must be a typed tuple")
        expected_ids = tuple(f"loop-{index:08d}" for index in range(len(self.loops)))
        if tuple(item.loop_id for item in self.loops) != expected_ids:
            raise ValueError("loop IDs must preserve canonical hole order")
        if not isinstance(self.contacts, tuple) or any(
            not isinstance(item, PairContactObservation) for item in self.contacts
        ):
            raise TypeError("loop scene contacts must be a typed tuple")
        expected_pairs = tuple(
            (first, second)
            for index, first in enumerate(expected_ids)
            for second in expected_ids[index + 1 :]
        )
        if tuple(item.loop_ids for item in self.contacts) != expected_pairs:
            raise ValueError("loop scene must observe every loop pair in order")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": LOOP_SCENE_SCENARIO_SCHEMA,
            "scenario_id": self.scenario_id,
            "foreground_strength_threshold": self.foreground_strength_threshold,
            "morphology": self.morphology,
            "loops": [item.to_data() for item in self.loops],
            "contacts": [item.to_data() for item in self.contacts],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "LoopSceneScenarioWitness":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "scenario_id",
                    "foreground_strength_threshold",
                    "morphology",
                    "loops",
                    "contacts",
                }
            ),
            "loop scene scenario",
        )
        if data["schema"] != LOOP_SCENE_SCENARIO_SCHEMA:
            raise ValueError("unsupported loop scene scenario")
        loops = data["loops"]
        contacts = data["contacts"]
        if not isinstance(loops, list) or any(
            not isinstance(item, Mapping) for item in loops
        ):
            raise TypeError("loop scene loops must be an object list")
        if not isinstance(contacts, list) or any(
            not isinstance(item, Mapping) for item in contacts
        ):
            raise TypeError("loop scene contacts must be an object list")
        return cls(
            scenario_id=data["scenario_id"],
            foreground_strength_threshold=data["foreground_strength_threshold"],
            morphology=data["morphology"],
            loops=tuple(LoopGeometryWitness.from_data(item) for item in loops),
            contacts=tuple(PairContactObservation.from_data(item) for item in contacts),
        )


@dataclass(frozen=True, slots=True)
class LoopScenePacket:
    panel_digest: str
    width_pixels: int
    height_pixels: int
    parent_bundle_digest: str
    extractor_source_digest: str
    parent_bundle_extractor_digest: str
    loop_geometry_algorithm_digest: str
    point_contact_algorithm_digest: str
    extractor_artifact_digest: str
    scenarios: tuple[LoopSceneScenarioWitness, ...]

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "loop scene panel_digest")
        _integer(self.width_pixels, "loop scene width_pixels", minimum=2)
        _integer(self.height_pixels, "loop scene height_pixels", minimum=2)
        _digest(self.parent_bundle_digest, "loop scene parent_bundle_digest")
        source = _digest(
            self.extractor_source_digest, "loop scene extractor_source_digest"
        )
        parent = _digest(
            self.parent_bundle_extractor_digest,
            "loop scene parent_bundle_extractor_digest",
        )
        geometry = _digest(
            self.loop_geometry_algorithm_digest,
            "loop scene loop_geometry_algorithm_digest",
        )
        contact = _digest(
            self.point_contact_algorithm_digest,
            "loop scene point_contact_algorithm_digest",
        )
        _digest(self.extractor_artifact_digest, "loop scene extractor_artifact_digest")
        if self.extractor_artifact_digest != _artifact_digest(
            source, parent, geometry, contact
        ):
            raise ValueError("loop scene artifact digest does not bind dependencies")
        if not isinstance(self.scenarios, tuple) or any(
            not isinstance(item, LoopSceneScenarioWitness) for item in self.scenarios
        ):
            raise TypeError("loop scene scenarios must be a typed tuple")
        if tuple(item.scenario_id for item in self.scenarios) != (
            _base.VISUAL_WITNESS_SCENARIO_IDS
        ):
            raise ValueError("loop scene must retain all scenarios in canonical order")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": LOOP_SCENE_PACKET_SCHEMA,
            "algorithm_id": LOOP_SCENE_ALGORITHM_ID,
            "panel_digest": self.panel_digest,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "parent_bundle_digest": self.parent_bundle_digest,
            "extractor_source_digest": self.extractor_source_digest,
            "parent_bundle_extractor_digest": self.parent_bundle_extractor_digest,
            "loop_geometry_algorithm_digest": self.loop_geometry_algorithm_digest,
            "point_contact_algorithm_digest": self.point_contact_algorithm_digest,
            "extractor_artifact_digest": self.extractor_artifact_digest,
            "scenarios": [item.to_data() for item in self.scenarios],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "LoopScenePacket":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "algorithm_id",
                    "panel_digest",
                    "width_pixels",
                    "height_pixels",
                    "parent_bundle_digest",
                    "extractor_source_digest",
                    "parent_bundle_extractor_digest",
                    "loop_geometry_algorithm_digest",
                    "point_contact_algorithm_digest",
                    "extractor_artifact_digest",
                    "scenarios",
                }
            ),
            "loop scene packet",
        )
        if (
            data["schema"] != LOOP_SCENE_PACKET_SCHEMA
            or data["algorithm_id"] != LOOP_SCENE_ALGORITHM_ID
        ):
            raise ValueError("unsupported loop scene packet")
        scenarios = data["scenarios"]
        if not isinstance(scenarios, list) or any(
            not isinstance(item, Mapping) for item in scenarios
        ):
            raise TypeError("loop scene scenarios must be an object list")
        return cls(
            panel_digest=data["panel_digest"],
            width_pixels=data["width_pixels"],
            height_pixels=data["height_pixels"],
            parent_bundle_digest=data["parent_bundle_digest"],
            extractor_source_digest=data["extractor_source_digest"],
            parent_bundle_extractor_digest=data["parent_bundle_extractor_digest"],
            loop_geometry_algorithm_digest=data["loop_geometry_algorithm_digest"],
            point_contact_algorithm_digest=data["point_contact_algorithm_digest"],
            extractor_artifact_digest=data["extractor_artifact_digest"],
            scenarios=tuple(LoopSceneScenarioWitness.from_data(item) for item in scenarios),
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _ordered_hole_masks(
    mask: np.ndarray,
) -> tuple[np.ndarray, ...]:
    labels, count = ndimage.label(~mask, structure=_base._BACKGROUND_STRUCTURE)
    border_labels = set(int(item) for item in labels[0, :])
    border_labels.update(int(item) for item in labels[-1, :])
    border_labels.update(int(item) for item in labels[:, 0])
    border_labels.update(int(item) for item in labels[:, -1])
    border_labels.discard(0)
    raw: list[tuple[tuple[object, ...], np.ndarray]] = []
    for label in range(1, count + 1):
        if label in border_labels:
            continue
        hole_mask = np.ascontiguousarray(labels == label, dtype=bool)
        x0, y0, x1, y1 = _base._bbox(hole_mask)
        area = int(np.count_nonzero(hole_mask))
        digest = _base._mask_digest(hole_mask)
        raw.append(((x0, y0, x1, y1, area, digest), hole_mask))
    raw.sort(key=lambda item: item[0])
    return tuple(item[1] for item in raw)


def attach_loop_scene_witnesses(
    png_bytes: bytes, bundle: VisualWitnessBundle
) -> LoopScenePacket:
    """Attach loop geometry to an already-extracted exact visual bundle."""

    if not isinstance(png_bytes, bytes):
        raise TypeError("loop scene input must be exact PNG bytes")
    verify_visual_witness_bundle(bundle, expected_png_bytes=png_bytes)
    strength = _base._decode_png(png_bytes)
    scenarios: list[LoopSceneScenarioWitness] = []
    for base_scenario in bundle.base_packet.scenarios:
        mask = _base._scenario_mask(
            strength,
            base_scenario.foreground_strength_threshold,
            base_scenario.morphology,
        )
        hole_masks = _ordered_hole_masks(mask)
        if len(hole_masks) != len(base_scenario.holes):
            raise ValueError("loop scene hole enumeration differs from parent bundle")
        loops = tuple(
            extract_loop_geometry(
                hole_mask,
                hole,
                width_pixels=bundle.width_pixels,
                height_pixels=bundle.height_pixels,
            )
            for hole_mask, hole in zip(
                hole_masks, base_scenario.holes, strict=True
            )
        )
        contacts = extract_pair_contact_observations(
            mask,
            hole_masks,
            loops,
            width_pixels=bundle.width_pixels,
            height_pixels=bundle.height_pixels,
            source_owner_component_ids=tuple(
                item.owner_component_id for item in base_scenario.holes
            ),
        )
        scenarios.append(
            LoopSceneScenarioWitness(
                scenario_id=base_scenario.scenario_id,
                foreground_strength_threshold=(
                    base_scenario.foreground_strength_threshold
                ),
                morphology=base_scenario.morphology,
                loops=loops,
                contacts=contacts,
            )
        )
    source = _source_digest()
    parent_extractor = bundle.assembler_artifact_digest
    geometry = loop_geometry_algorithm_digest()
    contact = point_contact_algorithm_digest()
    return LoopScenePacket(
        panel_digest=bundle.panel_digest,
        width_pixels=bundle.width_pixels,
        height_pixels=bundle.height_pixels,
        parent_bundle_digest=bundle.digest(),
        extractor_source_digest=source,
        parent_bundle_extractor_digest=parent_extractor,
        loop_geometry_algorithm_digest=geometry,
        point_contact_algorithm_digest=contact,
        extractor_artifact_digest=_artifact_digest(
            source, parent_extractor, geometry, contact
        ),
        scenarios=tuple(scenarios),
    )


def extract_loop_scene_witnesses(png_bytes: bytes) -> LoopScenePacket:
    if not isinstance(png_bytes, bytes):
        raise TypeError("loop scene input must be exact PNG bytes")
    bundle = extract_visual_witness_bundle(png_bytes)
    return attach_loop_scene_witnesses(png_bytes, bundle)


def verify_loop_scene_packet(
    packet: LoopScenePacket,
    *,
    expected_png_bytes: bytes,
    expected_bundle: VisualWitnessBundle | None = None,
) -> LoopScenePacket:
    if not isinstance(packet, LoopScenePacket):
        raise TypeError("packet must be a LoopScenePacket")
    if not isinstance(expected_png_bytes, bytes):
        raise TypeError("loop scene replay input must be exact PNG bytes")
    bundle = (
        extract_visual_witness_bundle(expected_png_bytes)
        if expected_bundle is None
        else expected_bundle
    )
    verify_visual_witness_bundle(bundle, expected_png_bytes=expected_png_bytes)
    replay = attach_loop_scene_witnesses(expected_png_bytes, bundle)
    if replay != packet:
        raise ValueError("loop scene packet differs from exact PNG replay")
    return packet


def _scene_provenance(
    packet: LoopScenePacket, snapshot: SceneSnapshot, scenario_id: str
) -> Provenance:
    return Provenance(
        producer=LOOP_SCENE_EXTRACTOR_ID,
        version=LOOP_SCENE_EXTRACTOR_VERSION,
        method="candidate-independent-loop-geometry-and-contact-attachment",
        input_digests=(
            packet.panel_digest,
            packet.parent_bundle_digest,
            packet.digest(),
            snapshot.digest(),
        ),
        artifact_digest=packet.extractor_artifact_digest,
        details=(
            ("scenario_id", scenario_id),
            ("serialized_numeric_domain", "integers-with-outward-intervals"),
        ),
    )


def _fact_interval(interval: IntInterval, unit: Unit) -> ScalarInterval:
    # The generic scene core uses canonical JSON floats.  Exact integer packet
    # values remain authoritative and convert without rounding for these small
    # bounded measurements.
    return ScalarInterval(float(interval.lower), float(interval.upper), unit)


def loop_scene_fragment(
    packet: LoopScenePacket, snapshot: SceneSnapshot
) -> SceneFragment:
    """Translate the packet into one transactional additive scene fragment."""

    if not isinstance(packet, LoopScenePacket):
        raise TypeError("packet must be a LoopScenePacket")
    if not isinstance(snapshot, SceneSnapshot):
        raise TypeError("snapshot must be a SceneSnapshot")
    if packet.panel_digest != snapshot.panel_digest:
        raise ValueError("loop packet and scene snapshot name different panels")
    if packet.parent_bundle_digest != snapshot.parent_bundle_digest:
        raise ValueError("loop packet and scene snapshot bind different bundles")
    if tuple(item.scenario_id for item in packet.scenarios) != snapshot.scenario_ids:
        raise ValueError("loop packet and scene scenario inventories differ")

    provenances = tuple(
        sorted(
            (
                _scene_provenance(packet, snapshot, scenario.scenario_id)
                for scenario in packet.scenarios
            ),
            key=lambda item: item.digest(),
        )
    )
    provenance_by_scenario = {
        dict(item.details)["scenario_id"]: item for item in provenances
    }
    entities: list[SceneEntity] = []
    facts: list[SceneFact] = []
    foundation = {item.entity_id: item for item in snapshot.entities}
    for scenario in packet.scenarios:
        provenance = provenance_by_scenario[scenario.scenario_id]
        for loop in scenario.loops:
            hole_entity_id = (
                f"{scenario.scenario_id}/hole/{loop.source_hole_id}"
            )
            hole_entity = foundation.get(hole_entity_id)
            if hole_entity is None or hole_entity.entity_type != "hole":
                raise ValueError("loop source hole is absent from scene foundation")
            if hole_entity.source_region_digest != loop.source_mask_digest:
                raise ValueError("loop source mask differs from foundation hole")
            loop_entity_id = f"{scenario.scenario_id}/loop/{loop.loop_id}"
            entities.append(
                SceneEntity(
                    entity_id=loop_entity_id,
                    entity_type="loop",
                    scenario_id=scenario.scenario_id,
                    frame_id=snapshot.frame_id,
                    source_witness_digest=loop.digest(),
                    source_region_digest=loop.boundary_digest,
                    provenance_digest=provenance.digest(),
                    owner_entity_id=hole_entity.owner_entity_id,
                )
            )
            common = {
                "arguments": (loop_entity_id,),
                "argument_types": ("loop",),
                "scenario_id": scenario.scenario_id,
                "frame_id": snapshot.frame_id,
                "provenance_digest": provenance.digest(),
                "source_region_digests": tuple(
                    sorted((loop.source_mask_digest, loop.boundary_digest))
                ),
            }
            facts.append(
                SceneFact(
                    fact_id=f"{scenario.scenario_id}/fact/area/{loop.loop_id}",
                    predicate="loop.area_pixels",
                    disposition=Disposition.PRESENT,
                    unit=Unit.PIXEL_AREA,
                    interval=ScalarInterval.point(
                        float(loop.area_pixels), Unit.PIXEL_AREA
                    ),
                    **common,
                )
            )
            substantive = loop.substantiveness
            facts.append(
                SceneFact(
                    fact_id=(
                        f"{scenario.scenario_id}/fact/substantive/{loop.loop_id}"
                    ),
                    predicate="loop.substantive",
                    disposition=substantive.disposition,
                    unit=Unit.NONE,
                    certificate=substantive.certificate,
                    **common,
                )
            )
            polygon = loop.polygon
            polygon_interval = (
                None
                if polygon.side_count is None
                else _fact_interval(polygon.side_count, Unit.COUNT)
            )
            facts.append(
                SceneFact(
                    fact_id=f"{scenario.scenario_id}/fact/sides/{loop.loop_id}",
                    predicate="loop.polygon_side_count",
                    disposition=polygon.disposition,
                    unit=Unit.COUNT,
                    interval=polygon_interval,
                    reason=(
                        None
                        if polygon.disposition is Disposition.PRESENT
                        else polygon.reason_code
                    ),
                    **common,
                )
            )
            oblique = loop.edge_obliqueness
            oblique_interval = (
                None
                if oblique.minimum_millidegrees is None
                else _fact_interval(
                    oblique.minimum_millidegrees, Unit.MILLIDEGREES
                )
            )
            facts.append(
                SceneFact(
                    fact_id=f"{scenario.scenario_id}/fact/obliqueness/{loop.loop_id}",
                    predicate="loop.edge_axis_obliqueness_millidegrees",
                    disposition=oblique.disposition,
                    unit=Unit.MILLIDEGREES,
                    interval=oblique_interval,
                    reason=(
                        None
                        if oblique.disposition is Disposition.PRESENT
                        else oblique.reason_code
                    ),
                    **common,
                )
            )
        loops_by_id = {item.loop_id: item for item in scenario.loops}
        for contact in scenario.contacts:
            first, second = (loops_by_id[item] for item in contact.loop_ids)
            arguments = tuple(
                f"{scenario.scenario_id}/loop/{item}" for item in contact.loop_ids
            )
            fact_kwargs: dict[str, object] = {}
            if contact.disposition is Disposition.CERTIFIED_ABSENT:
                fact_kwargs["certificate"] = contact.certificate
            elif contact.disposition is Disposition.INDETERMINATE:
                fact_kwargs["reason"] = contact.reason_code
            elif contact.disposition is Disposition.ERROR:
                fact_kwargs["reason"] = contact.reason_code
                fact_kwargs["error_type"] = contact.error_type
            facts.append(
                SceneFact(
                    fact_id=(
                        f"{scenario.scenario_id}/fact/point-contact/"
                        f"{contact.loop_ids[0]}:{contact.loop_ids[1]}"
                    ),
                    predicate="pair.point_contact",
                    arguments=arguments,
                    argument_types=("loop", "loop"),
                    scenario_id=scenario.scenario_id,
                    frame_id=snapshot.frame_id,
                    disposition=contact.disposition,
                    provenance_digest=provenance.digest(),
                    unit=Unit.NONE,
                    source_region_digests=tuple(
                        sorted((first.boundary_digest, second.boundary_digest))
                    ),
                    **fact_kwargs,
                )
            )
    return SceneFragment(
        panel_digest=packet.panel_digest,
        parent_bundle_digest=packet.parent_bundle_digest,
        parent_snapshot_digest=snapshot.digest(),
        graph_schema_digest=snapshot.graph_schema_digest,
        frame_id=snapshot.frame_id,
        scenario_ids=snapshot.scenario_ids,
        producer_leg="loop_scene_extractor",
        producer_leg_digest=packet.extractor_artifact_digest,
        provenances=provenances,
        entities=tuple(sorted(entities, key=lambda item: item.entity_id)),
        facts=tuple(sorted(facts, key=lambda item: item.fact_id)),
    )


__all__ = [
    "LOOP_SCENE_ALGORITHM_ID",
    "LOOP_SCENE_PACKET",
    "LOOP_SCENE_PACKET_SCHEMA",
    "LoopScenePacket",
    "LoopSceneScenarioWitness",
    "attach_loop_scene_witnesses",
    "extract_loop_scene_witnesses",
    "loop_scene_catalog_digest",
    "loop_scene_extractor_digest",
    "loop_scene_fragment",
    "verify_loop_scene_packet",
]
