"""Candidate-independent bundle of base and contour witnesses for one PNG.

The bundle is the executable visual boundary.  Both child packets are derived
from the same exact bytes, retain the same three preprocessing scenarios, and
are verified together.  No task, side, label, candidate, or prose enters this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.contour_witnesses import (
    CONTOUR_WITNESS_SCENARIO_IDS,
    ContourWitnessPacket,
    contour_witness_catalog_digest,
    contour_witness_extractor_digest,
    extract_contour_witnesses,
    verify_contour_witness_packet,
)
from bongard.legs.contracts import ValueType
from bongard.visual_witnesses import (
    VISUAL_WITNESS_SCENARIO_IDS,
    VisualWitnessPacket,
    extract_visual_witnesses,
    verify_visual_witness_packet,
    visual_witness_catalog_digest,
    visual_witness_extractor_digest,
)


VISUAL_WITNESS_BUNDLE = ValueType("visual_witness_bundle")
VISUAL_WITNESS_BUNDLE_EXTRACTOR_ID = "visual-witness-bundle"
VISUAL_WITNESS_BUNDLE_SCHEMA = "gkm.bongard-visual-witness-bundle.v1"
VISUAL_WITNESS_BUNDLE_ALGORITHM_ID = "bongard.visual-witness-bundle/v1"
VISUAL_WITNESS_BUNDLE_VERSION = "1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _exact_fields(
    data: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if not isinstance(data, Mapping) or set(data) != expected:
        raise ValueError(f"{label} fields differ from the static schema")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256")
    return value


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _artifact_digest(
    source_digest: str,
    base_extractor_digest: str,
    contour_extractor_digest: str,
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-visual-witness-bundle-artifact.v1",
            "algorithm_id": VISUAL_WITNESS_BUNDLE_ALGORITHM_ID,
            "source_digest": source_digest,
            "base_extractor_artifact_digest": base_extractor_digest,
            "contour_extractor_artifact_digest": contour_extractor_digest,
            "same_exact_panel_digest_required": True,
            "same_dimensions_required": True,
            "same_ordered_scenarios_required": True,
        }
    )


def visual_witness_bundle_extractor_digest() -> str:
    """Return the source- and child-extractor-bound bundle identity."""

    return _artifact_digest(
        _source_digest(),
        visual_witness_extractor_digest(),
        contour_witness_extractor_digest(),
    )


def visual_witness_bundle_catalog_digest() -> str:
    """Bind both finite witness inventories and the bundle boundary type."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-visual-witness-bundle-catalog.v1",
            "bundle_type": VISUAL_WITNESS_BUNDLE.to_data(),
            "base_catalog_digest": visual_witness_catalog_digest(),
            "contour_catalog_digest": contour_witness_catalog_digest(),
            "scenario_ids": list(VISUAL_WITNESS_SCENARIO_IDS),
        }
    )


@dataclass(frozen=True, slots=True)
class VisualWitnessBundle:
    base_packet: VisualWitnessPacket
    contour_packet: ContourWitnessPacket
    assembler_source_digest: str
    base_extractor_artifact_digest: str
    contour_extractor_artifact_digest: str
    assembler_artifact_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.base_packet, VisualWitnessPacket):
            raise TypeError("base_packet must be a VisualWitnessPacket")
        if not isinstance(self.contour_packet, ContourWitnessPacket):
            raise TypeError("contour_packet must be a ContourWitnessPacket")
        source = _digest(self.assembler_source_digest, "assembler_source_digest")
        base = _digest(
            self.base_extractor_artifact_digest,
            "base_extractor_artifact_digest",
        )
        contour = _digest(
            self.contour_extractor_artifact_digest,
            "contour_extractor_artifact_digest",
        )
        _digest(self.assembler_artifact_digest, "assembler_artifact_digest")
        if self.assembler_artifact_digest != _artifact_digest(source, base, contour):
            raise ValueError("bundle artifact digest does not bind source/dependencies")
        if base != self.base_packet.extractor_artifact_digest:
            raise ValueError("bundle base dependency differs from its packet")
        if contour != self.contour_packet.extractor_artifact_digest:
            raise ValueError("bundle contour dependency differs from its packet")
        if self.contour_packet.base_visual_extractor_digest != base:
            raise ValueError("contour packet was not derived against the bundled base extractor")
        if self.base_packet.panel_digest != self.contour_packet.panel_digest:
            raise ValueError("bundle child packets do not commit the same exact PNG")
        if (
            self.base_packet.width_pixels,
            self.base_packet.height_pixels,
        ) != (
            self.contour_packet.width_pixels,
            self.contour_packet.height_pixels,
        ):
            raise ValueError("bundle child packet dimensions differ")
        if VISUAL_WITNESS_SCENARIO_IDS != CONTOUR_WITNESS_SCENARIO_IDS:
            raise RuntimeError("base and contour scenario vocabularies drifted")
        for base_scenario, contour_scenario in zip(
            self.base_packet.scenarios,
            self.contour_packet.scenarios,
            strict=True,
        ):
            if (
                base_scenario.scenario_id,
                base_scenario.foreground_strength_threshold,
                base_scenario.morphology,
            ) != (
                contour_scenario.scenario_id,
                contour_scenario.foreground_strength_threshold,
                contour_scenario.morphology,
            ):
                raise ValueError("bundle child preprocessing scenarios differ")
            if len(base_scenario.components) != len(contour_scenario.contours):
                raise ValueError("bundle component and contour ownership counts differ")
            if tuple(item.component_id for item in base_scenario.components) != tuple(
                item.owner_component_id for item in contour_scenario.contours
            ):
                raise ValueError("bundle contour owners do not align with base components")

    @property
    def panel_digest(self) -> str:
        return self.base_packet.panel_digest

    @property
    def width_pixels(self) -> int:
        return self.base_packet.width_pixels

    @property
    def height_pixels(self) -> int:
        return self.base_packet.height_pixels

    def to_data(self) -> dict[str, object]:
        return {
            "schema": VISUAL_WITNESS_BUNDLE_SCHEMA,
            "algorithm_id": VISUAL_WITNESS_BUNDLE_ALGORITHM_ID,
            "base_packet": self.base_packet.to_data(),
            "contour_packet": self.contour_packet.to_data(),
            "assembler_source_digest": self.assembler_source_digest,
            "base_extractor_artifact_digest": self.base_extractor_artifact_digest,
            "contour_extractor_artifact_digest": self.contour_extractor_artifact_digest,
            "assembler_artifact_digest": self.assembler_artifact_digest,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "VisualWitnessBundle":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "algorithm_id",
                    "base_packet",
                    "contour_packet",
                    "assembler_source_digest",
                    "base_extractor_artifact_digest",
                    "contour_extractor_artifact_digest",
                    "assembler_artifact_digest",
                }
            ),
            "visual witness bundle",
        )
        if (
            data["schema"] != VISUAL_WITNESS_BUNDLE_SCHEMA
            or data["algorithm_id"] != VISUAL_WITNESS_BUNDLE_ALGORITHM_ID
        ):
            raise ValueError("unsupported visual witness bundle")
        base = data["base_packet"]
        contour = data["contour_packet"]
        if not isinstance(base, Mapping) or not isinstance(contour, Mapping):
            raise TypeError("bundle child packets must be JSON objects")
        result = cls(
            base_packet=VisualWitnessPacket.from_data(base),
            contour_packet=ContourWitnessPacket.from_data(contour),
            assembler_source_digest=data["assembler_source_digest"],
            base_extractor_artifact_digest=data["base_extractor_artifact_digest"],
            contour_extractor_artifact_digest=data[
                "contour_extractor_artifact_digest"
            ],
            assembler_artifact_digest=data["assembler_artifact_digest"],
        )
        if result.to_data() != dict(data):
            raise ValueError("visual witness bundle is not canonically represented")
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def extract_visual_witness_bundle(png_bytes: bytes) -> VisualWitnessBundle:
    """Extract both candidate-independent packets from the same exact bytes."""

    if not isinstance(png_bytes, bytes):
        raise TypeError("visual witness bundle input must be exact PNG bytes")
    base_packet = extract_visual_witnesses(png_bytes)
    contour_packet = extract_contour_witnesses(png_bytes)
    source = _source_digest()
    base_digest = visual_witness_extractor_digest()
    contour_digest = contour_witness_extractor_digest()
    return VisualWitnessBundle(
        base_packet=base_packet,
        contour_packet=contour_packet,
        assembler_source_digest=source,
        base_extractor_artifact_digest=base_digest,
        contour_extractor_artifact_digest=contour_digest,
        assembler_artifact_digest=_artifact_digest(
            source, base_digest, contour_digest
        ),
    )


def verify_visual_witness_bundle(
    bundle: VisualWitnessBundle,
    expected_png_bytes: bytes | None = None,
) -> VisualWitnessBundle:
    """Verify child coherence and optionally cold-replay the exact PNG."""

    if not isinstance(bundle, VisualWitnessBundle):
        raise TypeError("bundle must be a VisualWitnessBundle")
    if VisualWitnessBundle.from_data(bundle.to_data()) != bundle:
        raise ValueError("visual witness bundle is not canonically represented")
    current_source = _source_digest()
    current_base = visual_witness_extractor_digest()
    current_contour = contour_witness_extractor_digest()
    if (
        bundle.assembler_source_digest != current_source
        or bundle.base_extractor_artifact_digest != current_base
        or bundle.contour_extractor_artifact_digest != current_contour
        or bundle.assembler_artifact_digest
        != _artifact_digest(current_source, current_base, current_contour)
    ):
        raise ValueError("visual witness bundle source or dependency has drifted")
    verify_visual_witness_packet(
        bundle.base_packet, expected_png_bytes=expected_png_bytes
    )
    verify_contour_witness_packet(
        bundle.contour_packet, expected_png_bytes=expected_png_bytes
    )
    if expected_png_bytes is not None:
        replayed = extract_visual_witness_bundle(expected_png_bytes)
        if replayed != bundle:
            raise ValueError("visual witness bundle differs from exact PNG replay")
    return bundle


__all__ = [
    "VISUAL_WITNESS_BUNDLE",
    "VISUAL_WITNESS_BUNDLE_ALGORITHM_ID",
    "VISUAL_WITNESS_BUNDLE_EXTRACTOR_ID",
    "VISUAL_WITNESS_BUNDLE_SCHEMA",
    "VISUAL_WITNESS_BUNDLE_VERSION",
    "VisualWitnessBundle",
    "extract_visual_witness_bundle",
    "verify_visual_witness_bundle",
    "visual_witness_bundle_catalog_digest",
    "visual_witness_bundle_extractor_digest",
]
