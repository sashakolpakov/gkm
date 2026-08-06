"""Candidate-independent component and owned-hole witnesses for one PNG.

The extractor sees exact panel bytes only.  It retains three preprocessing
scenarios as separate alternatives; callers, not this module, decide whether
scenario agreement warrants present, absence, or indeterminate evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy import ndimage

from bongard.artifacts import canonical_digest, canonical_json
from bongard.legs.contracts import ValueType


VISUAL_WITNESS_CAPABILITY_IDS = ("component.count", "hole.owner_count")
VISUAL_WITNESS_SCENARIO_IDS = tuple(
    sorted(
        (
            "threshold032.raw",
            "threshold064.close-cross-1",
            "threshold096.raw",
        )
    )
)
VISUAL_WITNESS_PACKET = ValueType("visual_witness_packet")
VISUAL_WITNESS_EXTRACTOR_ID = "bongard.visual_witnesses"
VISUAL_WITNESS_EXTRACTOR_VERSION = "1"

ALGORITHM_ID = "bongard.visual-witness-extractor/v1"
PACKET_SCHEMA = "gkm.bongard-visual-witness-packet.v1"
PREDICATE_RESULT_SCHEMA = "gkm.bongard-visual-scenario-predicate-result.v1"
_MAX_PANEL_PIXELS = 4096 * 4096
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_ID = re.compile(r"component-[0-9]{8}\Z")
_HOLE_ID = re.compile(r"hole-[0-9]{8}\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_FOREGROUND_STRUCTURE = np.ones((3, 3), dtype=bool)
_BACKGROUND_STRUCTURE = np.asarray(
    ((False, True, False), (True, True, True), (False, True, False)),
    dtype=bool,
)

# Sorted by ID, exactly like the public scenario constant.
_SCENARIOS = (
    ("threshold032.raw", 32, "raw"),
    ("threshold064.close-cross-1", 64, "close-cross-1"),
    ("threshold096.raw", 96, "raw"),
)
assert tuple(item[0] for item in _SCENARIOS) == VISUAL_WITNESS_SCENARIO_IDS


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


def _artifact_digest(source_digest: str) -> str:
    return canonical_digest(
        {
            "algorithm_id": ALGORITHM_ID,
            "source_digest": source_digest,
            "decoder": {
                "format": "PNG",
                "frames": 1,
                "rgba_background": 255,
                "white_distance": "255-min(composited-rgb)",
            },
            "connectivity": {"foreground": 8, "background": 4},
            "q16_bbox": "half-open coordinates; round(coord*65535/extent)",
            "scenarios": [
                {
                    "scenario_id": scenario_id,
                    "foreground_strength_threshold": threshold,
                    "morphology": morphology,
                }
                for scenario_id, threshold, morphology in _SCENARIOS
            ],
        }
    )


def visual_witness_extractor_digest() -> str:
    """Return the current source-bound extractor artifact identity."""

    return _artifact_digest(_source_digest())


def visual_witness_catalog_digest() -> str:
    """Return the static capability/scenario/type catalog identity."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-visual-witness-catalog.v1",
            "extractor_id": VISUAL_WITNESS_EXTRACTOR_ID,
            "extractor_version": VISUAL_WITNESS_EXTRACTOR_VERSION,
            "packet_type": VISUAL_WITNESS_PACKET.to_data(),
            "capability_ids": list(VISUAL_WITNESS_CAPABILITY_IDS),
            "scenario_ids": list(VISUAL_WITNESS_SCENARIO_IDS),
        }
    )


def _mask_digest(mask: np.ndarray) -> str:
    height, width = mask.shape
    prefix = canonical_json(
        {
            "schema": "gkm.bongard-binary-mask.v1",
            "height_pixels": height,
            "width_pixels": width,
            "packing": "numpy.packbits-axis-none-bitorder-big",
        }
    )
    packed = np.packbits(mask.reshape(-1), bitorder="big").tobytes()
    return hashlib.sha256(prefix + b"\x00" + packed).hexdigest()


def _q16(coordinate: int, extent: int) -> int:
    return (coordinate * 65535 + extent // 2) // extent


@dataclass(frozen=True, order=True)
class Q16BBox:
    """Normalized half-open raster bounding box in unsigned Q16 coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        for name, value in (
            ("x0", self.x0),
            ("y0", self.y0),
            ("x1", self.x1),
            ("y1", self.y1),
        ):
            _integer(value, name)
            if value > 65535:
                raise ValueError(f"{name} exceeds unsigned Q16 range")
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("Q16 bounding boxes must have positive extent")

    def to_data(self) -> dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "Q16BBox":
        _exact_fields(data, frozenset({"x0", "y0", "x1", "y1"}), "Q16 bbox")
        return cls(x0=data["x0"], y0=data["y0"], x1=data["x1"], y1=data["y1"])


@dataclass(frozen=True)
class ComponentWitness:
    component_id: str
    bbox_q16: Q16BBox
    area_pixels: int
    mask_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or _COMPONENT_ID.fullmatch(
            self.component_id
        ) is None:
            raise ValueError("component_id is not canonical")
        if not isinstance(self.bbox_q16, Q16BBox):
            raise TypeError("component bbox_q16 must be a Q16BBox")
        _integer(self.area_pixels, "component area_pixels", minimum=1)
        _digest(self.mask_digest, "component mask_digest")

    def to_data(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "bbox_q16": self.bbox_q16.to_data(),
            "area_pixels": self.area_pixels,
            "mask_digest": self.mask_digest,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ComponentWitness":
        _exact_fields(
            data,
            frozenset({"component_id", "bbox_q16", "area_pixels", "mask_digest"}),
            "component witness",
        )
        bbox = data["bbox_q16"]
        if not isinstance(bbox, Mapping):
            raise TypeError("component bbox_q16 must be an object")
        return cls(
            component_id=data["component_id"],
            bbox_q16=Q16BBox.from_data(bbox),
            area_pixels=data["area_pixels"],
            mask_digest=data["mask_digest"],
        )


@dataclass(frozen=True)
class HoleWitness:
    hole_id: str
    bbox_q16: Q16BBox
    area_pixels: int
    mask_digest: str
    owner_component_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.hole_id, str) or _HOLE_ID.fullmatch(
            self.hole_id
        ) is None:
            raise ValueError("hole_id is not canonical")
        if not isinstance(self.bbox_q16, Q16BBox):
            raise TypeError("hole bbox_q16 must be a Q16BBox")
        _integer(self.area_pixels, "hole area_pixels", minimum=1)
        _digest(self.mask_digest, "hole mask_digest")
        if self.owner_component_id is not None and (
            not isinstance(self.owner_component_id, str)
            or _COMPONENT_ID.fullmatch(self.owner_component_id) is None
        ):
            raise ValueError("hole owner_component_id is not canonical or null")

    def to_data(self) -> dict[str, object]:
        return {
            "hole_id": self.hole_id,
            "bbox_q16": self.bbox_q16.to_data(),
            "area_pixels": self.area_pixels,
            "mask_digest": self.mask_digest,
            "owner_component_id": self.owner_component_id,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "HoleWitness":
        _exact_fields(
            data,
            frozenset(
                {
                    "hole_id",
                    "bbox_q16",
                    "area_pixels",
                    "mask_digest",
                    "owner_component_id",
                }
            ),
            "hole witness",
        )
        bbox = data["bbox_q16"]
        if not isinstance(bbox, Mapping):
            raise TypeError("hole bbox_q16 must be an object")
        return cls(
            hole_id=data["hole_id"],
            bbox_q16=Q16BBox.from_data(bbox),
            area_pixels=data["area_pixels"],
            mask_digest=data["mask_digest"],
            owner_component_id=data["owner_component_id"],
        )


@dataclass(frozen=True)
class ScenarioWitness:
    scenario_id: str
    foreground_strength_threshold: int
    morphology: str
    components: tuple[ComponentWitness, ...]
    holes: tuple[HoleWitness, ...]

    def __post_init__(self) -> None:
        expected = {item[0]: item[1:] for item in _SCENARIOS}
        if self.scenario_id not in expected:
            raise ValueError("unknown visual witness scenario_id")
        if (
            self.foreground_strength_threshold,
            self.morphology,
        ) != expected[self.scenario_id]:
            raise ValueError("scenario parameters do not match the frozen scenario ID")
        if not isinstance(self.components, tuple) or any(
            not isinstance(item, ComponentWitness) for item in self.components
        ):
            raise TypeError("scenario components must be a typed tuple")
        if not isinstance(self.holes, tuple) or any(
            not isinstance(item, HoleWitness) for item in self.holes
        ):
            raise TypeError("scenario holes must be a typed tuple")
        expected_components = tuple(
            f"component-{index:08d}" for index in range(len(self.components))
        )
        if tuple(item.component_id for item in self.components) != expected_components:
            raise ValueError("scenario component IDs must be consecutive and ordered")
        expected_holes = tuple(f"hole-{index:08d}" for index in range(len(self.holes)))
        if tuple(item.hole_id for item in self.holes) != expected_holes:
            raise ValueError("scenario hole IDs must be consecutive and ordered")
        component_ids = set(expected_components)
        if any(
            item.owner_component_id is not None
            and item.owner_component_id not in component_ids
            for item in self.holes
        ):
            raise ValueError("hole owner does not name a scenario component")

    def to_data(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "foreground_strength_threshold": self.foreground_strength_threshold,
            "morphology": self.morphology,
            "components": [item.to_data() for item in self.components],
            "holes": [item.to_data() for item in self.holes],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ScenarioWitness":
        _exact_fields(
            data,
            frozenset(
                {
                    "scenario_id",
                    "foreground_strength_threshold",
                    "morphology",
                    "components",
                    "holes",
                }
            ),
            "scenario witness",
        )
        components = data["components"]
        holes = data["holes"]
        if not isinstance(components, list) or not isinstance(holes, list):
            raise TypeError("scenario components and holes must be JSON lists")
        if any(not isinstance(item, Mapping) for item in components + holes):
            raise TypeError("scenario witness entries must be JSON objects")
        return cls(
            scenario_id=data["scenario_id"],
            foreground_strength_threshold=data["foreground_strength_threshold"],
            morphology=data["morphology"],
            components=tuple(ComponentWitness.from_data(item) for item in components),
            holes=tuple(HoleWitness.from_data(item) for item in holes),
        )


@dataclass(frozen=True)
class VisualWitnessPacket:
    panel_digest: str
    width_pixels: int
    height_pixels: int
    extractor_source_digest: str
    extractor_artifact_digest: str
    scenarios: tuple[ScenarioWitness, ...]

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "panel_digest")
        _integer(self.width_pixels, "width_pixels", minimum=2)
        _integer(self.height_pixels, "height_pixels", minimum=2)
        if self.width_pixels * self.height_pixels > _MAX_PANEL_PIXELS:
            raise ValueError("packet dimensions exceed the fixed pixel guard")
        source_digest = _digest(
            self.extractor_source_digest, "extractor_source_digest"
        )
        _digest(self.extractor_artifact_digest, "extractor_artifact_digest")
        if self.extractor_artifact_digest != _artifact_digest(source_digest):
            raise ValueError("extractor artifact digest does not bind its source")
        if not isinstance(self.scenarios, tuple) or any(
            not isinstance(item, ScenarioWitness) for item in self.scenarios
        ):
            raise TypeError("packet scenarios must be a typed tuple")
        if tuple(item.scenario_id for item in self.scenarios) != (
            VISUAL_WITNESS_SCENARIO_IDS
        ):
            raise ValueError("packet must retain all scenarios in canonical order")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PACKET_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "panel_digest": self.panel_digest,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "extractor_source_digest": self.extractor_source_digest,
            "extractor_artifact_digest": self.extractor_artifact_digest,
            "scenarios": [item.to_data() for item in self.scenarios],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "VisualWitnessPacket":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "algorithm_id",
                    "panel_digest",
                    "width_pixels",
                    "height_pixels",
                    "extractor_source_digest",
                    "extractor_artifact_digest",
                    "scenarios",
                }
            ),
            "visual witness packet",
        )
        if data["schema"] != PACKET_SCHEMA or data["algorithm_id"] != ALGORITHM_ID:
            raise ValueError("unsupported visual witness packet")
        scenarios = data["scenarios"]
        if not isinstance(scenarios, list) or any(
            not isinstance(item, Mapping) for item in scenarios
        ):
            raise TypeError("packet scenarios must be a JSON object list")
        return cls(
            panel_digest=data["panel_digest"],
            width_pixels=data["width_pixels"],
            height_pixels=data["height_pixels"],
            extractor_source_digest=data["extractor_source_digest"],
            extractor_artifact_digest=data["extractor_artifact_digest"],
            scenarios=tuple(ScenarioWitness.from_data(item) for item in scenarios),
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class ScenarioPredicateObservation:
    scenario_id: str
    observed_count: int
    matches: bool

    def __post_init__(self) -> None:
        if self.scenario_id not in VISUAL_WITNESS_SCENARIO_IDS:
            raise ValueError("unknown predicate observation scenario_id")
        _integer(self.observed_count, "observed_count")
        if not isinstance(self.matches, bool):
            raise TypeError("matches must be boolean")

    def to_data(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "observed_count": self.observed_count,
            "matches": self.matches,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ScenarioPredicateObservation":
        _exact_fields(
            data,
            frozenset({"scenario_id", "observed_count", "matches"}),
            "scenario predicate observation",
        )
        return cls(
            scenario_id=data["scenario_id"],
            observed_count=data["observed_count"],
            matches=data["matches"],
        )


@dataclass(frozen=True)
class ScenarioPredicateResult:
    capability_id: str
    expected_count: int
    packet_digest: str
    observations: tuple[ScenarioPredicateObservation, ...]

    def __post_init__(self) -> None:
        if self.capability_id not in VISUAL_WITNESS_CAPABILITY_IDS:
            raise ValueError("unknown visual witness capability_id")
        _integer(self.expected_count, "expected_count")
        _digest(self.packet_digest, "predicate packet_digest")
        if not isinstance(self.observations, tuple) or any(
            not isinstance(item, ScenarioPredicateObservation)
            for item in self.observations
        ):
            raise TypeError("predicate observations must be a typed tuple")
        if tuple(item.scenario_id for item in self.observations) != (
            VISUAL_WITNESS_SCENARIO_IDS
        ):
            raise ValueError("predicate observations must retain canonical scenarios")
        if any(
            item.matches != (item.observed_count == self.expected_count)
            for item in self.observations
        ):
            raise ValueError("predicate match flag disagrees with the exact count")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PREDICATE_RESULT_SCHEMA,
            "capability_id": self.capability_id,
            "expected_count": self.expected_count,
            "packet_digest": self.packet_digest,
            "observations": [item.to_data() for item in self.observations],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ScenarioPredicateResult":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "capability_id",
                    "expected_count",
                    "packet_digest",
                    "observations",
                }
            ),
            "scenario predicate result",
        )
        if data["schema"] != PREDICATE_RESULT_SCHEMA:
            raise ValueError("unsupported scenario predicate result")
        observations = data["observations"]
        if not isinstance(observations, list) or any(
            not isinstance(item, Mapping) for item in observations
        ):
            raise TypeError("predicate observations must be a JSON object list")
        return cls(
            capability_id=data["capability_id"],
            expected_count=data["expected_count"],
            packet_digest=data["packet_digest"],
            observations=tuple(
                ScenarioPredicateObservation.from_data(item)
                for item in observations
            ),
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _decode_png(png_bytes: bytes) -> np.ndarray:
    if not isinstance(png_bytes, bytes):
        raise TypeError("visual witness input must be exact PNG bytes")
    if not png_bytes.startswith(_PNG_SIGNATURE):
        raise ValueError("visual witness input must have the PNG signature")
    try:
        with Image.open(BytesIO(png_bytes)) as encoded:
            if encoded.format != "PNG":
                raise ValueError("visual witness input must decode as PNG")
            if getattr(encoded, "n_frames", 1) != 1:
                raise ValueError("visual witness PNG must have exactly one frame")
            width, height = encoded.size
            if width < 2 or height < 2:
                raise ValueError("visual witness PNG dimensions are too small")
            if width * height > _MAX_PANEL_PIXELS:
                raise ValueError("visual witness PNG exceeds the fixed pixel guard")
            rgba = np.asarray(encoded.convert("RGBA"), dtype=np.uint8)
    except (TypeError, ValueError):
        raise
    except Exception as exc:  # Pillow exposes decoder-specific exception types.
        raise ValueError(
            f"visual witness PNG decoding failed: {type(exc).__name__}: {exc}"
        ) from exc
    values = rgba.astype(np.uint32, copy=False)
    alpha = values[..., 3:4]
    rgb = (values[..., :3] * alpha + 255 * (255 - alpha) + 127) // 255
    return np.ascontiguousarray((255 - np.min(rgb, axis=2)).astype(np.uint8))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _q16_bbox(
    pixel_bbox: tuple[int, int, int, int], width: int, height: int
) -> Q16BBox:
    x0, y0, x1, y1 = pixel_bbox
    return Q16BBox(_q16(x0, width), _q16(y0, height), _q16(x1, width), _q16(y1, height))


def _scenario_mask(
    strength: np.ndarray, threshold: int, morphology: str
) -> np.ndarray:
    mask = strength >= threshold
    if morphology == "close-cross-1":
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        padded = ndimage.binary_closing(
            padded, structure=_BACKGROUND_STRUCTURE, iterations=1
        )
        mask = padded[1:-1, 1:-1]
    elif morphology != "raw":
        raise AssertionError("unreachable frozen morphology")
    return np.ascontiguousarray(mask, dtype=bool)


def _extract_scenario(
    strength: np.ndarray, scenario_id: str, threshold: int, morphology: str
) -> ScenarioWitness:
    mask = _scenario_mask(strength, threshold, morphology)
    height, width = mask.shape
    labels, label_count = ndimage.label(mask, structure=_FOREGROUND_STRUCTURE)

    raw_components: list[tuple[tuple[object, ...], int, np.ndarray]] = []
    for label in range(1, label_count + 1):
        component_mask = labels == label
        pixel_bbox = _bbox(component_mask)
        area = int(np.count_nonzero(component_mask))
        digest = _mask_digest(component_mask)
        x0, y0, x1, y1 = pixel_bbox
        key = (x0, y0, x1, y1, area, digest)
        raw_components.append((key, label, component_mask))
    raw_components.sort(key=lambda item: item[0])

    label_to_id: dict[int, str] = {}
    components: list[ComponentWitness] = []
    for index, (key, label, component_mask) in enumerate(raw_components):
        component_id = f"component-{index:08d}"
        label_to_id[label] = component_id
        x0, y0, x1, y1, area, digest = key
        components.append(
            ComponentWitness(
                component_id=component_id,
                bbox_q16=_q16_bbox((x0, y0, x1, y1), width, height),
                area_pixels=area,
                mask_digest=digest,
            )
        )

    background_labels, background_count = ndimage.label(
        ~mask, structure=_BACKGROUND_STRUCTURE
    )
    border_labels = set(int(item) for item in background_labels[0, :])
    border_labels.update(int(item) for item in background_labels[-1, :])
    border_labels.update(int(item) for item in background_labels[:, 0])
    border_labels.update(int(item) for item in background_labels[:, -1])
    border_labels.discard(0)

    raw_holes: list[
        tuple[tuple[object, ...], str | None]
    ] = []
    for label in range(1, background_count + 1):
        if label in border_labels:
            continue
        hole_mask = background_labels == label
        pixel_bbox = _bbox(hole_mask)
        area = int(np.count_nonzero(hole_mask))
        digest = _mask_digest(hole_mask)
        boundary = ndimage.binary_dilation(
            hole_mask, structure=_BACKGROUND_STRUCTURE, iterations=1
        ) & mask
        owner_labels = set(int(item) for item in labels[boundary])
        owner_labels.discard(0)
        owner = (
            label_to_id[next(iter(owner_labels))]
            if len(owner_labels) == 1
            else None
        )
        x0, y0, x1, y1 = pixel_bbox
        raw_holes.append(((x0, y0, x1, y1, area, digest), owner))
    raw_holes.sort(key=lambda item: item[0])

    holes = tuple(
        HoleWitness(
            hole_id=f"hole-{index:08d}",
            bbox_q16=_q16_bbox(
                (key[0], key[1], key[2], key[3]), width, height
            ),
            area_pixels=key[4],
            mask_digest=key[5],
            owner_component_id=owner,
        )
        for index, (key, owner) in enumerate(raw_holes)
    )
    return ScenarioWitness(
        scenario_id=scenario_id,
        foreground_strength_threshold=threshold,
        morphology=morphology,
        components=tuple(components),
        holes=holes,
    )


def extract_visual_witnesses(png_bytes: bytes) -> VisualWitnessPacket:
    """Extract the frozen joint witness scenarios from exact PNG bytes only."""

    strength = _decode_png(png_bytes)
    height, width = strength.shape
    source_digest = _source_digest()
    return VisualWitnessPacket(
        panel_digest=hashlib.sha256(png_bytes).hexdigest(),
        width_pixels=width,
        height_pixels=height,
        extractor_source_digest=source_digest,
        extractor_artifact_digest=_artifact_digest(source_digest),
        scenarios=tuple(
            _extract_scenario(strength, scenario_id, threshold, morphology)
            for scenario_id, threshold, morphology in _SCENARIOS
        ),
    )


def verify_visual_witness_packet(
    packet: VisualWitnessPacket, expected_png_bytes: bytes | None = None
) -> VisualWitnessPacket:
    """Validate a packet and optionally replay it from its claimed exact bytes."""

    if not isinstance(packet, VisualWitnessPacket):
        raise TypeError("packet must be a VisualWitnessPacket")
    # Reconstructing through the strict JSON boundary catches forged nested
    # values even if a caller bypassed frozen dataclass assignment controls.
    if VisualWitnessPacket.from_data(packet.to_data()) != packet:
        raise ValueError("visual witness packet is not canonically represented")
    current_source = _source_digest()
    if packet.extractor_source_digest != current_source or (
        packet.extractor_artifact_digest != _artifact_digest(current_source)
    ):
        raise ValueError("visual witness extractor source or artifact has drifted")
    if expected_png_bytes is not None:
        if not isinstance(expected_png_bytes, bytes):
            raise TypeError("expected_png_bytes must be exact bytes or null")
        replayed = extract_visual_witnesses(expected_png_bytes)
        if replayed != packet:
            raise ValueError("visual witness packet differs from exact PNG replay")
    return packet


def _count_result(
    packet: VisualWitnessPacket, expected: int, capability_id: str
) -> ScenarioPredicateResult:
    verify_visual_witness_packet(packet)
    expected_count = _integer(expected, "expected count")
    if capability_id == "component.count":
        counts = tuple(len(scenario.components) for scenario in packet.scenarios)
    elif capability_id == "hole.owner_count":
        counts = tuple(
            sum(hole.owner_component_id is not None for hole in scenario.holes)
            for scenario in packet.scenarios
        )
    else:  # The two public wrappers make this unreachable.
        raise ValueError("unknown visual witness capability")
    return ScenarioPredicateResult(
        capability_id=capability_id,
        expected_count=expected_count,
        packet_digest=packet.digest(),
        observations=tuple(
            ScenarioPredicateObservation(
                scenario_id=scenario.scenario_id,
                observed_count=count,
                matches=count == expected_count,
            )
            for scenario, count in zip(packet.scenarios, counts, strict=True)
        ),
    )


def component_count_by_scenario(
    packet: VisualWitnessPacket, expected: int
) -> ScenarioPredicateResult:
    """Return, without consensus collapse, each scenario's component count."""

    return _count_result(packet, expected, "component.count")


def owned_hole_count_by_scenario(
    packet: VisualWitnessPacket, expected: int
) -> ScenarioPredicateResult:
    """Return each scenario's count of holes with exactly one component owner."""

    return _count_result(packet, expected, "hole.owner_count")
