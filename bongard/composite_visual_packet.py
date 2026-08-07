"""Backend-neutral exact-panel packet for the closed visual predicate union.

The packet is assembled from exact PNG bytes only.  It coherently binds the
existing direct/contour witness bundle, the loop-scene packet, and a new
integer-valued bilateral-reflection witness over the *same* three frozen
preprocessing scenarios.  Task identity, labels, prose, and candidate formulas
are deliberately absent from every extractor entry point.

The bilateral score is reflected-ink coverage in parts per million.  Its
positive residual, ``1_000_000 - coverage``, is used by the closed predicate
layer for reflection-mismatch claims; it is not a logical negation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

from bongard import visual_witnesses as _base
from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.legs.bilateral_symmetry import (
    _MaskScore,
    _score_mask,
    operation_digest as bilateral_symmetry_operation_digest,
)
from bongard.loop_scene_witnesses import (
    LoopScenePacket,
    attach_loop_scene_witnesses,
    loop_scene_extractor_digest,
    verify_loop_scene_packet,
)
from bongard.visual_witness_bundle import (
    VisualWitnessBundle,
    extract_visual_witness_bundle,
    verify_visual_witness_bundle,
    visual_witness_bundle_extractor_digest,
)


PPM_SCALE = 1_000_000
BILATERAL_SCENARIO_SCHEMA = "gkm.bongard-bilateral-scenario-witness.v1"
BILATERAL_PACKET_SCHEMA = "gkm.bongard-bilateral-scenario-packet.v1"
BILATERAL_ALGORITHM_ID = "bongard.bilateral-scenario-reflection-coverage/v1"
EXACT_PANEL_PACKET_SCHEMA = "gkm.bongard-exact-panel-witness-packet.v1"
EXACT_PANEL_ALGORITHM_ID = "bongard.exact-panel-witness-composite/v1"
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


def _integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f"..{maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{label} must lie in {minimum}{suffix}")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be null or non-empty stripped text")
    return value


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def composite_visual_packet_source_digest() -> str:
    """Return the exact source identity for both new packet assemblers."""

    return _source_digest()


def _bilateral_artifact_digest(
    source_digest: str,
    base_extractor_digest: str,
    bilateral_operation_digest: str,
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-bilateral-scenario-artifact.v1",
            "algorithm_id": BILATERAL_ALGORITHM_ID,
            "source_digest": source_digest,
            "base_visual_extractor_digest": base_extractor_digest,
            "bilateral_operation_digest": bilateral_operation_digest,
            "scenario_ids": list(_base.VISUAL_WITNESS_SCENARIO_IDS),
            "source_masks": "exact frozen visual-witness scenario masks",
            "coverage_unit": "parts_per_million",
            "quantization": "round(binary-mask reflection coverage * 1000000)",
        }
    )


def bilateral_symmetry_witness_extractor_digest() -> str:
    return _bilateral_artifact_digest(
        _source_digest(),
        _base.visual_witness_extractor_digest(),
        bilateral_symmetry_operation_digest(),
    )


@dataclass(frozen=True, order=True, slots=True)
class PpmInterval:
    """Closed integer fraction interval in parts per million."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        _integer(self.lower, "ppm interval lower", maximum=PPM_SCALE)
        _integer(self.upper, "ppm interval upper", maximum=PPM_SCALE)
        if self.lower > self.upper:
            raise ValueError("ppm interval lower exceeds upper")

    @property
    def exact(self) -> bool:
        return self.lower == self.upper

    def to_data(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PpmInterval":
        _exact_fields(data, frozenset({"lower", "upper"}), "ppm interval")
        return cls(data["lower"], data["upper"])


def _scenario_provenance_digest(
    *,
    scenario_id: str,
    panel_digest: str,
    source_mask_digest: str,
    disposition: Disposition,
    coverage_ppm: PpmInterval | None,
    best_axis_millidegrees: int | None,
    foreground_pixels: int | None,
    reason: str | None,
    certificate: str | None,
    error_type: str | None,
    extractor_artifact_digest: str,
) -> str:
    return canonical_digest(
        {
            "algorithm_id": BILATERAL_ALGORITHM_ID,
            "scenario_id": scenario_id,
            "panel_digest": panel_digest,
            "source_mask_digest": source_mask_digest,
            "disposition": disposition.value,
            "coverage_ppm": None if coverage_ppm is None else coverage_ppm.to_data(),
            "best_axis_millidegrees": best_axis_millidegrees,
            "foreground_pixels": foreground_pixels,
            "reason": reason,
            "certificate": certificate,
            "error_type": error_type,
            "extractor_artifact_digest": extractor_artifact_digest,
        }
    )


@dataclass(frozen=True, slots=True)
class BilateralSymmetryScenarioWitness:
    """One four-state bilateral measurement in a correlated raster scenario."""

    scenario_id: str
    foreground_strength_threshold: int
    morphology: str
    panel_digest: str
    source_mask_digest: str
    disposition: Disposition
    coverage_ppm: PpmInterval | None
    best_axis_millidegrees: int | None
    foreground_pixels: int | None
    reason: str | None
    certificate: str | None
    error_type: str | None
    extractor_artifact_digest: str
    provenance_digest: str

    def __post_init__(self) -> None:
        expected = {item[0]: item[1:] for item in _base._SCENARIOS}
        if self.scenario_id not in expected:
            raise ValueError("unknown bilateral scenario_id")
        if (self.foreground_strength_threshold, self.morphology) != expected[
            self.scenario_id
        ]:
            raise ValueError("bilateral scenario parameters differ from frozen grid")
        _digest(self.panel_digest, "bilateral scenario panel_digest")
        _digest(self.source_mask_digest, "bilateral scenario source_mask_digest")
        _digest(self.extractor_artifact_digest, "bilateral extractor_artifact_digest")
        _digest(self.provenance_digest, "bilateral scenario provenance_digest")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("bilateral disposition must be a Disposition")
        reason = _optional_text(self.reason, "bilateral reason")
        certificate = _optional_text(self.certificate, "bilateral certificate")
        error_type = _optional_text(self.error_type, "bilateral error_type")

        if self.disposition is Disposition.PRESENT:
            if not isinstance(self.coverage_ppm, PpmInterval):
                raise ValueError("present bilateral witness requires a ppm interval")
            _integer(
                self.best_axis_millidegrees,
                "best_axis_millidegrees",
                maximum=179_999,
            )
            _integer(self.foreground_pixels, "foreground_pixels", minimum=1)
            if any(value is not None for value in (reason, certificate, error_type)):
                raise ValueError("present bilateral witness cannot carry failure fields")
        elif self.disposition is Disposition.CERTIFIED_ABSENT:
            if self.coverage_ppm is not None or any(
                value is not None
                for value in (self.best_axis_millidegrees, self.foreground_pixels)
            ):
                raise ValueError("certified absence cannot carry a measured score")
            if certificate is None or reason is not None or error_type is not None:
                raise ValueError("certified bilateral absence requires only a certificate")
        elif self.disposition is Disposition.INDETERMINATE:
            if self.coverage_ppm != PpmInterval(0, PPM_SCALE):
                raise ValueError("indeterminate bilateral witness requires full ppm interval")
            if self.best_axis_millidegrees is not None:
                raise ValueError("indeterminate bilateral witness cannot select an axis")
            if self.foreground_pixels is not None:
                _integer(self.foreground_pixels, "foreground_pixels")
            if reason is None or certificate is not None or error_type is not None:
                raise ValueError("indeterminate bilateral witness requires only a reason")
        elif self.disposition is Disposition.ERROR:
            if self.coverage_ppm is not None or any(
                value is not None
                for value in (self.best_axis_millidegrees, self.foreground_pixels)
            ):
                raise ValueError("bilateral error cannot carry a measured score")
            if reason is None or error_type is None or certificate is not None:
                raise ValueError("bilateral error requires reason and error_type")

        expected_provenance = _scenario_provenance_digest(
            scenario_id=self.scenario_id,
            panel_digest=self.panel_digest,
            source_mask_digest=self.source_mask_digest,
            disposition=self.disposition,
            coverage_ppm=self.coverage_ppm,
            best_axis_millidegrees=self.best_axis_millidegrees,
            foreground_pixels=self.foreground_pixels,
            reason=self.reason,
            certificate=self.certificate,
            error_type=self.error_type,
            extractor_artifact_digest=self.extractor_artifact_digest,
        )
        if self.provenance_digest != expected_provenance:
            raise ValueError("bilateral scenario provenance does not bind its contents")

    @property
    def mismatch_ppm(self) -> PpmInterval | None:
        if self.coverage_ppm is None:
            return None
        return PpmInterval(
            PPM_SCALE - self.coverage_ppm.upper,
            PPM_SCALE - self.coverage_ppm.lower,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": BILATERAL_SCENARIO_SCHEMA,
            "scenario_id": self.scenario_id,
            "foreground_strength_threshold": self.foreground_strength_threshold,
            "morphology": self.morphology,
            "panel_digest": self.panel_digest,
            "source_mask_digest": self.source_mask_digest,
            "disposition": self.disposition.value,
            "coverage_ppm": (
                None if self.coverage_ppm is None else self.coverage_ppm.to_data()
            ),
            "best_axis_millidegrees": self.best_axis_millidegrees,
            "foreground_pixels": self.foreground_pixels,
            "reason": self.reason,
            "certificate": self.certificate,
            "error_type": self.error_type,
            "extractor_artifact_digest": self.extractor_artifact_digest,
            "provenance_digest": self.provenance_digest,
        }

    @classmethod
    def from_data(
        cls, data: Mapping[str, Any]
    ) -> "BilateralSymmetryScenarioWitness":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "scenario_id",
                    "foreground_strength_threshold",
                    "morphology",
                    "panel_digest",
                    "source_mask_digest",
                    "disposition",
                    "coverage_ppm",
                    "best_axis_millidegrees",
                    "foreground_pixels",
                    "reason",
                    "certificate",
                    "error_type",
                    "extractor_artifact_digest",
                    "provenance_digest",
                }
            ),
            "bilateral scenario witness",
        )
        if data["schema"] != BILATERAL_SCENARIO_SCHEMA:
            raise ValueError("unsupported bilateral scenario witness")
        interval = data["coverage_ppm"]
        if interval is not None and not isinstance(interval, Mapping):
            raise TypeError("bilateral coverage_ppm must be null or an object")
        result = cls(
            scenario_id=data["scenario_id"],
            foreground_strength_threshold=data["foreground_strength_threshold"],
            morphology=data["morphology"],
            panel_digest=data["panel_digest"],
            source_mask_digest=data["source_mask_digest"],
            disposition=Disposition(data["disposition"]),
            coverage_ppm=(None if interval is None else PpmInterval.from_data(interval)),
            best_axis_millidegrees=data["best_axis_millidegrees"],
            foreground_pixels=data["foreground_pixels"],
            reason=data["reason"],
            certificate=data["certificate"],
            error_type=data["error_type"],
            extractor_artifact_digest=data["extractor_artifact_digest"],
            provenance_digest=data["provenance_digest"],
        )
        if result.to_data() != dict(data):
            raise ValueError("bilateral scenario witness is not canonical")
        return result


def _make_scenario_witness(
    *,
    scenario_id: str,
    foreground_strength_threshold: int,
    morphology: str,
    panel_digest: str,
    source_mask_digest: str,
    disposition: Disposition,
    coverage_ppm: PpmInterval | None = None,
    best_axis_millidegrees: int | None = None,
    foreground_pixels: int | None = None,
    reason: str | None = None,
    certificate: str | None = None,
    error_type: str | None = None,
    extractor_artifact_digest: str,
) -> BilateralSymmetryScenarioWitness:
    provenance = _scenario_provenance_digest(
        scenario_id=scenario_id,
        panel_digest=panel_digest,
        source_mask_digest=source_mask_digest,
        disposition=disposition,
        coverage_ppm=coverage_ppm,
        best_axis_millidegrees=best_axis_millidegrees,
        foreground_pixels=foreground_pixels,
        reason=reason,
        certificate=certificate,
        error_type=error_type,
        extractor_artifact_digest=extractor_artifact_digest,
    )
    return BilateralSymmetryScenarioWitness(
        scenario_id=scenario_id,
        foreground_strength_threshold=foreground_strength_threshold,
        morphology=morphology,
        panel_digest=panel_digest,
        source_mask_digest=source_mask_digest,
        disposition=disposition,
        coverage_ppm=coverage_ppm,
        best_axis_millidegrees=best_axis_millidegrees,
        foreground_pixels=foreground_pixels,
        reason=reason,
        certificate=certificate,
        error_type=error_type,
        extractor_artifact_digest=extractor_artifact_digest,
        provenance_digest=provenance,
    )


@dataclass(frozen=True, slots=True)
class BilateralSymmetryWitnessPacket:
    panel_digest: str
    width_pixels: int
    height_pixels: int
    parent_visual_bundle_digest: str
    extractor_source_digest: str
    base_visual_extractor_digest: str
    bilateral_operation_digest: str
    extractor_artifact_digest: str
    scenarios: tuple[BilateralSymmetryScenarioWitness, ...]

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "bilateral packet panel_digest")
        _integer(self.width_pixels, "bilateral width_pixels", minimum=2)
        _integer(self.height_pixels, "bilateral height_pixels", minimum=2)
        _digest(
            self.parent_visual_bundle_digest,
            "bilateral parent_visual_bundle_digest",
        )
        source = _digest(self.extractor_source_digest, "bilateral extractor source")
        base = _digest(
            self.base_visual_extractor_digest,
            "bilateral base visual extractor",
        )
        operation = _digest(
            self.bilateral_operation_digest, "bilateral operation digest"
        )
        _digest(self.extractor_artifact_digest, "bilateral extractor artifact")
        if self.extractor_artifact_digest != _bilateral_artifact_digest(
            source, base, operation
        ):
            raise ValueError("bilateral packet artifact does not bind dependencies")
        if not isinstance(self.scenarios, tuple) or tuple(
            item.scenario_id for item in self.scenarios
        ) != _base.VISUAL_WITNESS_SCENARIO_IDS:
            raise ValueError("bilateral packet must retain canonical scenarios")
        for scenario in self.scenarios:
            if not isinstance(scenario, BilateralSymmetryScenarioWitness):
                raise TypeError("bilateral packet scenarios must be typed witnesses")
            if scenario.panel_digest != self.panel_digest:
                raise ValueError("bilateral scenario binds a different exact panel")
            if scenario.extractor_artifact_digest != self.extractor_artifact_digest:
                raise ValueError("bilateral scenario extractor identity differs")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": BILATERAL_PACKET_SCHEMA,
            "algorithm_id": BILATERAL_ALGORITHM_ID,
            "panel_digest": self.panel_digest,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "parent_visual_bundle_digest": self.parent_visual_bundle_digest,
            "extractor_source_digest": self.extractor_source_digest,
            "base_visual_extractor_digest": self.base_visual_extractor_digest,
            "bilateral_operation_digest": self.bilateral_operation_digest,
            "extractor_artifact_digest": self.extractor_artifact_digest,
            "scenarios": [item.to_data() for item in self.scenarios],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "BilateralSymmetryWitnessPacket":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "algorithm_id",
                    "panel_digest",
                    "width_pixels",
                    "height_pixels",
                    "parent_visual_bundle_digest",
                    "extractor_source_digest",
                    "base_visual_extractor_digest",
                    "bilateral_operation_digest",
                    "extractor_artifact_digest",
                    "scenarios",
                }
            ),
            "bilateral packet",
        )
        if (
            data["schema"] != BILATERAL_PACKET_SCHEMA
            or data["algorithm_id"] != BILATERAL_ALGORITHM_ID
        ):
            raise ValueError("unsupported bilateral packet")
        scenarios = data["scenarios"]
        if not isinstance(scenarios, list) or any(
            not isinstance(item, Mapping) for item in scenarios
        ):
            raise TypeError("bilateral packet scenarios must be an object list")
        result = cls(
            panel_digest=data["panel_digest"],
            width_pixels=data["width_pixels"],
            height_pixels=data["height_pixels"],
            parent_visual_bundle_digest=data["parent_visual_bundle_digest"],
            extractor_source_digest=data["extractor_source_digest"],
            base_visual_extractor_digest=data["base_visual_extractor_digest"],
            bilateral_operation_digest=data["bilateral_operation_digest"],
            extractor_artifact_digest=data["extractor_artifact_digest"],
            scenarios=tuple(
                BilateralSymmetryScenarioWitness.from_data(item) for item in scenarios
            ),
        )
        if result.to_data() != dict(data):
            raise ValueError("bilateral packet is not canonically represented")
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def extract_bilateral_symmetry_witnesses(
    png_bytes: bytes,
    visual_bundle: VisualWitnessBundle,
) -> BilateralSymmetryWitnessPacket:
    """Measure bilateral coverage on every frozen exact-byte scenario mask."""

    if not isinstance(png_bytes, bytes):
        raise TypeError("bilateral witness input must be exact PNG bytes")
    verify_visual_witness_bundle(visual_bundle, expected_png_bytes=png_bytes)
    strength = _base._decode_png(png_bytes)
    panel_digest = hashlib.sha256(png_bytes).hexdigest()
    source = _source_digest()
    base_digest = _base.visual_witness_extractor_digest()
    operation = bilateral_symmetry_operation_digest()
    artifact = _bilateral_artifact_digest(source, base_digest, operation)
    scenarios: list[BilateralSymmetryScenarioWitness] = []
    for base_scenario in visual_bundle.base_packet.scenarios:
        mask = _base._scenario_mask(
            strength,
            base_scenario.foreground_strength_threshold,
            base_scenario.morphology,
        )
        source_mask_digest = _base._mask_digest(mask)
        try:
            measured = _score_mask(mask)
        except Exception as exc:  # noqa: BLE001 - four-disposition boundary.
            witness = _make_scenario_witness(
                scenario_id=base_scenario.scenario_id,
                foreground_strength_threshold=(
                    base_scenario.foreground_strength_threshold
                ),
                morphology=base_scenario.morphology,
                panel_digest=panel_digest,
                source_mask_digest=source_mask_digest,
                disposition=Disposition.ERROR,
                reason=str(exc) or repr(exc),
                error_type=type(exc).__name__,
                extractor_artifact_digest=artifact,
            )
        else:
            if isinstance(measured, _MaskScore):
                ppm = min(PPM_SCALE, max(0, int(round(measured.score * PPM_SCALE))))
                witness = _make_scenario_witness(
                    scenario_id=base_scenario.scenario_id,
                    foreground_strength_threshold=(
                        base_scenario.foreground_strength_threshold
                    ),
                    morphology=base_scenario.morphology,
                    panel_digest=panel_digest,
                    source_mask_digest=source_mask_digest,
                    disposition=Disposition.PRESENT,
                    coverage_ppm=PpmInterval(ppm, ppm),
                    best_axis_millidegrees=min(
                        179_999, int(round(measured.axis_degrees * 1000.0))
                    ),
                    foreground_pixels=measured.foreground_pixels,
                    extractor_artifact_digest=artifact,
                )
            elif measured == "absent":
                witness = _make_scenario_witness(
                    scenario_id=base_scenario.scenario_id,
                    foreground_strength_threshold=(
                        base_scenario.foreground_strength_threshold
                    ),
                    morphology=base_scenario.morphology,
                    panel_digest=panel_digest,
                    source_mask_digest=source_mask_digest,
                    disposition=Disposition.CERTIFIED_ABSENT,
                    certificate="frozen scenario mask contains exactly zero ink pixels",
                    extractor_artifact_digest=artifact,
                )
            else:
                witness = _make_scenario_witness(
                    scenario_id=base_scenario.scenario_id,
                    foreground_strength_threshold=(
                        base_scenario.foreground_strength_threshold
                    ),
                    morphology=base_scenario.morphology,
                    panel_digest=panel_digest,
                    source_mask_digest=source_mask_digest,
                    disposition=Disposition.INDETERMINATE,
                    coverage_ppm=PpmInterval(0, PPM_SCALE),
                    reason=f"bilateral measurement guard: {measured}",
                    extractor_artifact_digest=artifact,
                )
        scenarios.append(witness)
    return BilateralSymmetryWitnessPacket(
        panel_digest=panel_digest,
        width_pixels=visual_bundle.width_pixels,
        height_pixels=visual_bundle.height_pixels,
        parent_visual_bundle_digest=visual_bundle.digest(),
        extractor_source_digest=source,
        base_visual_extractor_digest=base_digest,
        bilateral_operation_digest=operation,
        extractor_artifact_digest=artifact,
        scenarios=tuple(scenarios),
    )


def verify_bilateral_symmetry_witness_packet(
    packet: BilateralSymmetryWitnessPacket,
    *,
    expected_png_bytes: bytes,
    expected_visual_bundle: VisualWitnessBundle,
) -> BilateralSymmetryWitnessPacket:
    if not isinstance(packet, BilateralSymmetryWitnessPacket):
        raise TypeError("packet must be a BilateralSymmetryWitnessPacket")
    if BilateralSymmetryWitnessPacket.from_data(packet.to_data()) != packet:
        raise ValueError("bilateral packet fails strict round trip")
    replay = extract_bilateral_symmetry_witnesses(
        expected_png_bytes, expected_visual_bundle
    )
    if replay != packet:
        raise ValueError("bilateral packet differs from exact PNG replay")
    return packet


def _composite_artifact_digest(
    source_digest: str,
    visual_extractor_digest: str,
    loop_extractor_digest: str,
    bilateral_extractor_digest: str,
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-exact-panel-composite-artifact.v1",
            "algorithm_id": EXACT_PANEL_ALGORITHM_ID,
            "source_digest": source_digest,
            "visual_bundle_extractor_digest": visual_extractor_digest,
            "loop_scene_extractor_digest": loop_extractor_digest,
            "bilateral_extractor_digest": bilateral_extractor_digest,
            "same_exact_png_required": True,
            "same_dimensions_required": True,
            "same_ordered_scenarios_required": True,
        }
    )


def exact_panel_witness_extractor_digest() -> str:
    return _composite_artifact_digest(
        _source_digest(),
        visual_witness_bundle_extractor_digest(),
        loop_scene_extractor_digest(),
        bilateral_symmetry_witness_extractor_digest(),
    )


@dataclass(frozen=True, slots=True)
class ExactPanelWitnessPacket:
    """One strict, cold-replayable packet for all closed predicate variants."""

    visual_bundle: VisualWitnessBundle
    loop_scene: LoopScenePacket
    bilateral_symmetry: BilateralSymmetryWitnessPacket
    assembler_source_digest: str
    visual_bundle_extractor_digest: str
    loop_scene_extractor_digest: str
    bilateral_extractor_digest: str
    assembler_artifact_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.visual_bundle, VisualWitnessBundle):
            raise TypeError("visual_bundle must be a VisualWitnessBundle")
        if not isinstance(self.loop_scene, LoopScenePacket):
            raise TypeError("loop_scene must be a LoopScenePacket")
        if not isinstance(
            self.bilateral_symmetry, BilateralSymmetryWitnessPacket
        ):
            raise TypeError("bilateral_symmetry must be a typed packet")
        source = _digest(self.assembler_source_digest, "composite assembler source")
        visual = _digest(
            self.visual_bundle_extractor_digest, "composite visual extractor"
        )
        loop = _digest(self.loop_scene_extractor_digest, "composite loop extractor")
        bilateral = _digest(
            self.bilateral_extractor_digest, "composite bilateral extractor"
        )
        _digest(self.assembler_artifact_digest, "composite assembler artifact")
        if self.assembler_artifact_digest != _composite_artifact_digest(
            source, visual, loop, bilateral
        ):
            raise ValueError("composite artifact digest does not bind dependencies")
        if visual != self.visual_bundle.assembler_artifact_digest:
            raise ValueError("composite visual dependency differs from child packet")
        if loop != self.loop_scene.extractor_artifact_digest:
            raise ValueError("composite loop dependency differs from child packet")
        if bilateral != self.bilateral_symmetry.extractor_artifact_digest:
            raise ValueError("composite bilateral dependency differs from child packet")
        if self.loop_scene.parent_bundle_digest != self.visual_bundle.digest():
            raise ValueError("loop scene does not bind the bundled visual witnesses")
        if (
            self.bilateral_symmetry.parent_visual_bundle_digest
            != self.visual_bundle.digest()
        ):
            raise ValueError("bilateral packet does not bind the visual bundle")
        if (
            self.bilateral_symmetry.base_visual_extractor_digest
            != self.visual_bundle.base_packet.extractor_artifact_digest
        ):
            raise ValueError(
                "bilateral packet base extractor differs from the visual bundle"
            )
        panel_digests = {
            self.visual_bundle.panel_digest,
            self.loop_scene.panel_digest,
            self.bilateral_symmetry.panel_digest,
        }
        if len(panel_digests) != 1:
            raise ValueError("composite children do not bind the same exact PNG")
        dimensions = {
            (self.visual_bundle.width_pixels, self.visual_bundle.height_pixels),
            (self.loop_scene.width_pixels, self.loop_scene.height_pixels),
            (
                self.bilateral_symmetry.width_pixels,
                self.bilateral_symmetry.height_pixels,
            ),
        }
        if len(dimensions) != 1:
            raise ValueError("composite child dimensions differ")
        visual_scenarios = tuple(
            (
                item.scenario_id,
                item.foreground_strength_threshold,
                item.morphology,
            )
            for item in self.visual_bundle.base_packet.scenarios
        )
        loop_scenarios = tuple(
            (item.scenario_id, item.foreground_strength_threshold, item.morphology)
            for item in self.loop_scene.scenarios
        )
        bilateral_scenarios = tuple(
            (item.scenario_id, item.foreground_strength_threshold, item.morphology)
            for item in self.bilateral_symmetry.scenarios
        )
        if visual_scenarios != loop_scenarios or visual_scenarios != bilateral_scenarios:
            raise ValueError("composite child preprocessing scenarios differ")

    @property
    def panel_digest(self) -> str:
        return self.visual_bundle.panel_digest

    def to_data(self) -> dict[str, object]:
        return {
            "schema": EXACT_PANEL_PACKET_SCHEMA,
            "algorithm_id": EXACT_PANEL_ALGORITHM_ID,
            "visual_bundle": self.visual_bundle.to_data(),
            "loop_scene": self.loop_scene.to_data(),
            "bilateral_symmetry": self.bilateral_symmetry.to_data(),
            "assembler_source_digest": self.assembler_source_digest,
            "visual_bundle_extractor_digest": self.visual_bundle_extractor_digest,
            "loop_scene_extractor_digest": self.loop_scene_extractor_digest,
            "bilateral_extractor_digest": self.bilateral_extractor_digest,
            "assembler_artifact_digest": self.assembler_artifact_digest,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ExactPanelWitnessPacket":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "algorithm_id",
                    "visual_bundle",
                    "loop_scene",
                    "bilateral_symmetry",
                    "assembler_source_digest",
                    "visual_bundle_extractor_digest",
                    "loop_scene_extractor_digest",
                    "bilateral_extractor_digest",
                    "assembler_artifact_digest",
                }
            ),
            "exact panel witness packet",
        )
        if (
            data["schema"] != EXACT_PANEL_PACKET_SCHEMA
            or data["algorithm_id"] != EXACT_PANEL_ALGORITHM_ID
        ):
            raise ValueError("unsupported exact panel witness packet")
        visual = data["visual_bundle"]
        loop = data["loop_scene"]
        bilateral = data["bilateral_symmetry"]
        if any(not isinstance(item, Mapping) for item in (visual, loop, bilateral)):
            raise TypeError("composite child packets must be JSON objects")
        result = cls(
            visual_bundle=VisualWitnessBundle.from_data(visual),
            loop_scene=LoopScenePacket.from_data(loop),
            bilateral_symmetry=BilateralSymmetryWitnessPacket.from_data(bilateral),
            assembler_source_digest=data["assembler_source_digest"],
            visual_bundle_extractor_digest=data[
                "visual_bundle_extractor_digest"
            ],
            loop_scene_extractor_digest=data["loop_scene_extractor_digest"],
            bilateral_extractor_digest=data["bilateral_extractor_digest"],
            assembler_artifact_digest=data["assembler_artifact_digest"],
        )
        if result.to_data() != dict(data):
            raise ValueError("exact panel witness packet is not canonical")
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def extract_exact_panel_witness_packet(png_bytes: bytes) -> ExactPanelWitnessPacket:
    """Extract every child from the same exact bytes, without candidate input."""

    if not isinstance(png_bytes, bytes):
        raise TypeError("exact panel packet input must be exact PNG bytes")
    visual = extract_visual_witness_bundle(png_bytes)
    loop = attach_loop_scene_witnesses(png_bytes, visual)
    bilateral = extract_bilateral_symmetry_witnesses(png_bytes, visual)
    source = _source_digest()
    visual_digest = visual.assembler_artifact_digest
    loop_digest = loop.extractor_artifact_digest
    bilateral_digest = bilateral.extractor_artifact_digest
    return ExactPanelWitnessPacket(
        visual_bundle=visual,
        loop_scene=loop,
        bilateral_symmetry=bilateral,
        assembler_source_digest=source,
        visual_bundle_extractor_digest=visual_digest,
        loop_scene_extractor_digest=loop_digest,
        bilateral_extractor_digest=bilateral_digest,
        assembler_artifact_digest=_composite_artifact_digest(
            source, visual_digest, loop_digest, bilateral_digest
        ),
    )


def verify_exact_panel_witness_packet(
    packet: ExactPanelWitnessPacket,
    *,
    expected_png_bytes: bytes,
) -> ExactPanelWitnessPacket:
    """Cold-replay every child and the complete composite from exact bytes."""

    if not isinstance(packet, ExactPanelWitnessPacket):
        raise TypeError("packet must be an ExactPanelWitnessPacket")
    if ExactPanelWitnessPacket.from_data(packet.to_data()) != packet:
        raise ValueError("exact panel packet fails strict round trip")
    verify_visual_witness_bundle(
        packet.visual_bundle, expected_png_bytes=expected_png_bytes
    )
    verify_loop_scene_packet(
        packet.loop_scene,
        expected_png_bytes=expected_png_bytes,
        expected_bundle=packet.visual_bundle,
    )
    verify_bilateral_symmetry_witness_packet(
        packet.bilateral_symmetry,
        expected_png_bytes=expected_png_bytes,
        expected_visual_bundle=packet.visual_bundle,
    )
    replay = extract_exact_panel_witness_packet(expected_png_bytes)
    if replay != packet:
        raise ValueError("exact panel witness packet differs from cold replay")
    return packet


__all__ = [
    "BILATERAL_ALGORITHM_ID",
    "BILATERAL_PACKET_SCHEMA",
    "BILATERAL_SCENARIO_SCHEMA",
    "EXACT_PANEL_ALGORITHM_ID",
    "EXACT_PANEL_PACKET_SCHEMA",
    "PPM_SCALE",
    "BilateralSymmetryScenarioWitness",
    "BilateralSymmetryWitnessPacket",
    "ExactPanelWitnessPacket",
    "PpmInterval",
    "bilateral_symmetry_witness_extractor_digest",
    "composite_visual_packet_source_digest",
    "exact_panel_witness_extractor_digest",
    "extract_bilateral_symmetry_witnesses",
    "extract_exact_panel_witness_packet",
    "verify_bilateral_symmetry_witness_packet",
    "verify_exact_panel_witness_packet",
]
