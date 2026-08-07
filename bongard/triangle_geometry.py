"""Candidate-independent triangle measurements over typed loop geometry.

The loop extractor already retains two independently fitted polygon variants.
This module turns those retained vertices into conservative integer angle and
side-ratio intervals.  It does not reopen pixels and it never receives a task,
support label, candidate, formula, or natural-language description.

The measurements are deliberately useful even when they do not decide a
class.  A stable large triangle can be certified as equilateral/right/obtuse;
an unstable tiny raster triangle remains ``indeterminate`` and can be handed
to a separately calibrated vision observer as a verifier-owned witness.  A
failed or unstable polygon fit is never converted into semantic absence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.loop_geometry import IntInterval, PolygonVariantWitness
from bongard.loop_scene_witnesses import LoopScenePacket


TRIANGLE_GEOMETRY_PACKET_SCHEMA = "gkm.bongard-triangle-geometry-packet.v1"
TRIANGLE_GEOMETRY_OBSERVATION_SCHEMA = (
    "gkm.bongard-triangle-geometry-observation.v1"
)
TRIANGLE_VARIANT_MEASUREMENT_SCHEMA = (
    "gkm.bongard-triangle-variant-measurement.v1"
)
TRIANGLE_CLASS_RESULT_SCHEMA = "gkm.bongard-triangle-class-result.v1"
TRIANGLE_GEOMETRY_ALGORITHM_ID = (
    "bongard.triangle-geometry/polygon-variant-intervals-v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OBJECT_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}/loop/loop-[0-9]{8}\Z"
)
_MILLIDEGREES_180 = 180_000
_PPM = 1_000_000

# These are operational class bands, not claims about an ideal mathematical
# ontology.  They intentionally leave guard bands so borderline raster fits
# become indeterminate rather than being forced into a class.
_EQUILATERAL_MAX_SIDE_RATIO_PPM = 1_200_000
_EQUILATERAL_MIN_ANGLE_MDEG = 45_000
_EQUILATERAL_MAX_ANGLE_MDEG = 75_000
_RIGHT_MIN_MDEG = 80_000
_RIGHT_MAX_MDEG = 100_000
_OBTUSE_MIN_MDEG = 100_000
_ACUTE_MAX_MDEG = 80_000
# Below two tenths of one percent of the panel, raster quantization can turn a
# visible three-corner loop into a stable four-to-six-vertex polygon.  At that
# scale a positive hard fit remains useful, but a negative hard fit is not an
# absence certificate.  The calibrated observer owns that fallback.
_MIN_CERTIFIED_ABSENCE_AREA_PPM = 2_000
_MIN_CERTIFIED_ABSENCE_PIXELS = 16


class TriangleGeometryError(ValueError):
    """A triangle packet or one of its exact bindings is malformed."""


class TriangleClass(str, Enum):
    EQUILATERAL = "triangle.equilateral"
    RIGHT = "triangle.right"
    OBTUSE = "triangle.obtuse"
    ACUTE = "triangle.acute"


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TriangleGeometryError(f"{label} fields differ from schema")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TriangleGeometryError(f"{label} must be a lowercase sha256")
    return value


def _object_id(value: object) -> str:
    if not isinstance(value, str) or _OBJECT_ID.fullmatch(value) is None:
        raise TriangleGeometryError(
            "triangle object_id must be a scenario-qualified loop identity"
        )
    return value


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def triangle_geometry_algorithm_digest() -> str:
    """Bind the pure-Python measurement and all operational class bands."""

    return canonical_digest(
        {
            "algorithm_id": TRIANGLE_GEOMETRY_ALGORITHM_ID,
            "source_digest": _source_digest(),
            "input": "retained-frozen-polygon-variants-only",
            "numeric_boundary": "integer-ppm-and-millidegree-closed-intervals",
            "angle_uncertainty": (
                "atan(polygon-residual-fraction)+q16-quarter-degree-guard"
            ),
            "class_bands": {
                "equilateral": {
                    "maximum_side_ratio_ppm": (
                        _EQUILATERAL_MAX_SIDE_RATIO_PPM
                    ),
                    "minimum_angle_millidegrees": (
                        _EQUILATERAL_MIN_ANGLE_MDEG
                    ),
                    "maximum_angle_millidegrees": (
                        _EQUILATERAL_MAX_ANGLE_MDEG
                    ),
                },
                "right": [_RIGHT_MIN_MDEG, _RIGHT_MAX_MDEG],
                "obtuse_strictly_above_millidegrees": _OBTUSE_MIN_MDEG,
                "acute_strictly_below_millidegrees": _ACUTE_MAX_MDEG,
            },
            "failed_or_unstable_polygon_fit": "indeterminate-never-absence",
            "small_loop_absence_guard": {
                "minimum_area_ppm_of_panel": _MIN_CERTIFIED_ABSENCE_AREA_PPM,
                "absolute_floor_pixels": _MIN_CERTIFIED_ABSENCE_PIXELS,
                "present_evidence_retained": True,
                "certified_absence_downgraded_to_indeterminate": True,
            },
            "python_is_authority": True,
        }
    )


@dataclass(frozen=True, slots=True)
class TriangleVariantMeasurement:
    """One polygon variant measured with outward integer uncertainty."""

    variant_id: str
    polygon_variant_digest: str
    minimum_interior_angle_millidegrees: IntInterval
    maximum_interior_angle_millidegrees: IntInterval
    maximum_to_minimum_side_ratio_ppm: IntInterval

    def __post_init__(self) -> None:
        if not isinstance(self.variant_id, str) or not self.variant_id:
            raise TriangleGeometryError("triangle variant_id must be text")
        _digest(self.polygon_variant_digest, "polygon_variant_digest")
        for label, interval, maximum in (
            (
                "minimum interior angle",
                self.minimum_interior_angle_millidegrees,
                _MILLIDEGREES_180,
            ),
            (
                "maximum interior angle",
                self.maximum_interior_angle_millidegrees,
                _MILLIDEGREES_180,
            ),
            (
                "side ratio",
                self.maximum_to_minimum_side_ratio_ppm,
                None,
            ),
        ):
            if not isinstance(interval, IntInterval):
                raise TypeError(f"{label} must be an IntInterval")
            if maximum is not None and interval.upper > maximum:
                raise TriangleGeometryError(f"{label} exceeds its physical range")
        if self.maximum_to_minimum_side_ratio_ppm.lower < _PPM:
            raise TriangleGeometryError("maximum/minimum side ratio is below one")
        if (
            self.minimum_interior_angle_millidegrees.lower
            > self.maximum_interior_angle_millidegrees.upper
        ):
            raise TriangleGeometryError("minimum angle exceeds maximum angle")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": TRIANGLE_VARIANT_MEASUREMENT_SCHEMA,
            "variant_id": self.variant_id,
            "polygon_variant_digest": self.polygon_variant_digest,
            "minimum_interior_angle_millidegrees": (
                self.minimum_interior_angle_millidegrees.to_data()
            ),
            "maximum_interior_angle_millidegrees": (
                self.maximum_interior_angle_millidegrees.to_data()
            ),
            "maximum_to_minimum_side_ratio_ppm": (
                self.maximum_to_minimum_side_ratio_ppm.to_data()
            ),
        }

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "TriangleVariantMeasurement":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema",
                    "variant_id",
                    "polygon_variant_digest",
                    "minimum_interior_angle_millidegrees",
                    "maximum_interior_angle_millidegrees",
                    "maximum_to_minimum_side_ratio_ppm",
                }
            ),
            "triangle variant measurement",
        )
        if value["schema"] != TRIANGLE_VARIANT_MEASUREMENT_SCHEMA:
            raise TriangleGeometryError("unsupported triangle variant measurement")
        intervals = (
            value["minimum_interior_angle_millidegrees"],
            value["maximum_interior_angle_millidegrees"],
            value["maximum_to_minimum_side_ratio_ppm"],
        )
        if any(not isinstance(item, Mapping) for item in intervals):
            raise TriangleGeometryError("triangle intervals must be objects")
        result = cls(
            variant_id=value["variant_id"],
            polygon_variant_digest=value["polygon_variant_digest"],
            minimum_interior_angle_millidegrees=IntInterval.from_data(
                intervals[0]
            ),
            maximum_interior_angle_millidegrees=IntInterval.from_data(
                intervals[1]
            ),
            maximum_to_minimum_side_ratio_ppm=IntInterval.from_data(
                intervals[2]
            ),
        )
        if result.to_data() != dict(value):
            raise TriangleGeometryError(
                "triangle variant measurement is not canonical"
            )
        return result


@dataclass(frozen=True, order=True, slots=True)
class TriangleClassResult:
    triangle_class: TriangleClass
    disposition: Disposition
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.triangle_class, TriangleClass):
            raise TypeError("triangle_class must be a registered enum")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("triangle class disposition must be typed")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise TriangleGeometryError("triangle class reason_code must be text")
        if self.disposition is Disposition.ERROR:
            raise TriangleGeometryError(
                "deterministic triangle classification has no error constructor"
            )

    def to_data(self) -> dict[str, str]:
        return {
            "schema": TRIANGLE_CLASS_RESULT_SCHEMA,
            "triangle_class": self.triangle_class.value,
            "disposition": self.disposition.value,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "TriangleClassResult":
        _exact_fields(
            value,
            frozenset(
                {"schema", "triangle_class", "disposition", "reason_code"}
            ),
            "triangle class result",
        )
        if value["schema"] != TRIANGLE_CLASS_RESULT_SCHEMA:
            raise TriangleGeometryError("unsupported triangle class result")
        result = cls(
            TriangleClass(value["triangle_class"]),
            Disposition(value["disposition"]),
            value["reason_code"],
        )
        if result.to_data() != dict(value):
            raise TriangleGeometryError("triangle class result is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class TriangleGeometryObservation:
    object_id: str
    source_loop_digest: str
    polygon_disposition: Disposition
    variants: tuple[TriangleVariantMeasurement, ...]
    classes: tuple[TriangleClassResult, ...]

    def __post_init__(self) -> None:
        _object_id(self.object_id)
        _digest(self.source_loop_digest, "source_loop_digest")
        if not isinstance(self.polygon_disposition, Disposition):
            raise TypeError("polygon_disposition must be typed")
        if self.polygon_disposition not in {
            Disposition.PRESENT,
            Disposition.INDETERMINATE,
        }:
            raise TriangleGeometryError(
                "candidate-independent polygon fit must be present or indeterminate"
            )
        if not isinstance(self.variants, tuple) or any(
            not isinstance(item, TriangleVariantMeasurement)
            for item in self.variants
        ):
            raise TypeError("triangle variants must be a typed tuple")
        if not isinstance(self.classes, tuple) or tuple(
            item.triangle_class for item in self.classes
        ) != tuple(TriangleClass):
            raise TriangleGeometryError(
                "triangle observation must retain every class in enum order"
            )
        if self.polygon_disposition is Disposition.INDETERMINATE:
            if self.variants or any(
                item.disposition is not Disposition.INDETERMINATE
                for item in self.classes
            ):
                raise TriangleGeometryError(
                    "unstable polygon geometry must remain wholly indeterminate"
                )
        elif self.variants:
            if len(self.variants) < 2:
                raise TriangleGeometryError(
                    "stable triangle geometry must retain the frozen variant ladder"
                )
        else:
            dispositions = {item.disposition for item in self.classes}
            reasons = {item.reason_code for item in self.classes}
            if dispositions == {Disposition.CERTIFIED_ABSENT}:
                if reasons != {"stable_polygon_side_count_is_not_three"}:
                    raise TriangleGeometryError(
                        "stable non-triangle absence reason differs"
                    )
            elif dispositions == {Disposition.INDETERMINATE}:
                if reasons != {"small_loop_below_absence_resolution_guard"}:
                    raise TriangleGeometryError(
                        "small stable non-triangle guard reason differs"
                    )
            else:
                raise TriangleGeometryError(
                    "stable non-triangle must be uniformly absent or guarded"
                )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": TRIANGLE_GEOMETRY_OBSERVATION_SCHEMA,
            "object_id": self.object_id,
            "source_loop_digest": self.source_loop_digest,
            "polygon_disposition": self.polygon_disposition.value,
            "variants": [item.to_data() for item in self.variants],
            "classes": [item.to_data() for item in self.classes],
        }

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "TriangleGeometryObservation":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema",
                    "object_id",
                    "source_loop_digest",
                    "polygon_disposition",
                    "variants",
                    "classes",
                }
            ),
            "triangle geometry observation",
        )
        if value["schema"] != TRIANGLE_GEOMETRY_OBSERVATION_SCHEMA:
            raise TriangleGeometryError("unsupported triangle observation")
        raw_variants = value["variants"]
        raw_classes = value["classes"]
        if not isinstance(raw_variants, list) or any(
            not isinstance(item, Mapping) for item in raw_variants
        ):
            raise TriangleGeometryError("triangle variants must be an object list")
        if not isinstance(raw_classes, list) or any(
            not isinstance(item, Mapping) for item in raw_classes
        ):
            raise TriangleGeometryError("triangle classes must be an object list")
        result = cls(
            object_id=value["object_id"],
            source_loop_digest=value["source_loop_digest"],
            polygon_disposition=Disposition(value["polygon_disposition"]),
            variants=tuple(
                TriangleVariantMeasurement.from_data(item)
                for item in raw_variants
            ),
            classes=tuple(TriangleClassResult.from_data(item) for item in raw_classes),
        )
        if result.to_data() != dict(value):
            raise TriangleGeometryError("triangle observation is not canonical")
        return result

    def class_result(self, triangle_class: TriangleClass) -> TriangleClassResult:
        if not isinstance(triangle_class, TriangleClass):
            raise TypeError("triangle_class must be registered")
        return self.classes[tuple(TriangleClass).index(triangle_class)]


@dataclass(frozen=True, slots=True)
class TriangleGeometryPacket:
    panel_digest: str
    loop_scene_packet_digest: str
    algorithm_digest: str
    observations: tuple[TriangleGeometryObservation, ...]

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "triangle panel_digest")
        _digest(self.loop_scene_packet_digest, "loop_scene_packet_digest")
        _digest(self.algorithm_digest, "triangle algorithm_digest")
        if self.algorithm_digest != triangle_geometry_algorithm_digest():
            raise TriangleGeometryError("triangle algorithm identity drifted")
        if not isinstance(self.observations, tuple) or any(
            not isinstance(item, TriangleGeometryObservation)
            for item in self.observations
        ):
            raise TypeError("triangle observations must be a typed tuple")
        object_ids = tuple(item.object_id for item in self.observations)
        if object_ids != tuple(sorted(set(object_ids))):
            raise TriangleGeometryError(
                "triangle object inventory must be unique and sorted"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": TRIANGLE_GEOMETRY_PACKET_SCHEMA,
            "algorithm_id": TRIANGLE_GEOMETRY_ALGORITHM_ID,
            "panel_digest": self.panel_digest,
            "loop_scene_packet_digest": self.loop_scene_packet_digest,
            "algorithm_digest": self.algorithm_digest,
            "observations": [item.to_data() for item in self.observations],
        }

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "TriangleGeometryPacket":
        _exact_fields(
            value,
            frozenset(
                {
                    "schema",
                    "algorithm_id",
                    "panel_digest",
                    "loop_scene_packet_digest",
                    "algorithm_digest",
                    "observations",
                }
            ),
            "triangle geometry packet",
        )
        if (
            value["schema"] != TRIANGLE_GEOMETRY_PACKET_SCHEMA
            or value["algorithm_id"] != TRIANGLE_GEOMETRY_ALGORITHM_ID
        ):
            raise TriangleGeometryError("unsupported triangle geometry packet")
        raw = value["observations"]
        if not isinstance(raw, list) or any(
            not isinstance(item, Mapping) for item in raw
        ):
            raise TriangleGeometryError("triangle observations must be an object list")
        result = cls(
            panel_digest=value["panel_digest"],
            loop_scene_packet_digest=value["loop_scene_packet_digest"],
            algorithm_digest=value["algorithm_digest"],
            observations=tuple(
                TriangleGeometryObservation.from_data(item) for item in raw
            ),
        )
        if result.to_data() != dict(value):
            raise TriangleGeometryError("triangle packet is not canonical")
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _outward_interval(value: float, uncertainty: int, maximum: int) -> IntInterval:
    lower = max(0, math.floor(value) - uncertainty)
    upper = min(maximum, math.ceil(value) + uncertainty)
    return IntInterval(lower, upper)


def _triangle_angles_and_sides(
    variant: PolygonVariantWitness,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    points = tuple((item.x, item.y) for item in variant.vertices_q16)
    if len(points) != 3:
        raise TriangleGeometryError("triangle measurement requires three vertices")
    angles: list[float] = []
    sides: list[float] = []
    for index, middle in enumerate(points):
        previous = points[index - 1]
        following = points[(index + 1) % len(points)]
        first = (previous[0] - middle[0], previous[1] - middle[1])
        second = (following[0] - middle[0], following[1] - middle[1])
        first_norm = math.hypot(*first)
        second_norm = math.hypot(*second)
        if first_norm == 0.0 or second_norm == 0.0:
            raise TriangleGeometryError("polygon variant repeats a triangle vertex")
        cosine = (first[0] * second[0] + first[1] * second[1]) / (
            first_norm * second_norm
        )
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
        sides.append(
            math.hypot(
                following[0] - middle[0], following[1] - middle[1]
            )
        )
    return tuple(angles), tuple(sides)


def _measure_variant(
    variant: PolygonVariantWitness,
) -> TriangleVariantMeasurement:
    angles, sides = _triangle_angles_and_sides(variant)
    # The polygon residual is normalized to panel extent.  atan(residual)
    # gives a conservative directional error; a fixed quarter-degree guard
    # covers Q16 endpoint quantization and outward integer conversion.
    angle_uncertainty = math.ceil(
        math.degrees(math.atan(variant.residual_ppm_upper / _PPM)) * 1000
    ) + 250
    angle_intervals = tuple(
        _outward_interval(angle * 1000.0, angle_uncertainty, _MILLIDEGREES_180)
        for angle in angles
    )
    minimum_angle = IntInterval(
        min(item.lower for item in angle_intervals),
        min(item.upper for item in angle_intervals),
    )
    maximum_angle = IntInterval(
        max(item.lower for item in angle_intervals),
        max(item.upper for item in angle_intervals),
    )
    ratio = max(sides) / min(sides) * _PPM
    ratio_uncertainty = max(
        1,
        math.ceil(ratio * (2 * variant.residual_ppm_upper / _PPM)),
    )
    side_ratio = IntInterval(
        max(_PPM, math.floor(ratio) - ratio_uncertainty),
        math.ceil(ratio) + ratio_uncertainty,
    )
    return TriangleVariantMeasurement(
        variant_id=variant.variant_id,
        polygon_variant_digest=canonical_digest(variant.to_data()),
        minimum_interior_angle_millidegrees=minimum_angle,
        maximum_interior_angle_millidegrees=maximum_angle,
        maximum_to_minimum_side_ratio_ppm=side_ratio,
    )


def _envelope(
    values: Sequence[TriangleVariantMeasurement], attribute: str
) -> IntInterval:
    intervals = tuple(getattr(item, attribute) for item in values)
    return IntInterval(
        min(item.lower for item in intervals),
        max(item.upper for item in intervals),
    )


def _band_result(
    triangle_class: TriangleClass,
    interval: IntInterval,
    *,
    lower: int,
    upper: int,
) -> TriangleClassResult:
    if interval.lower >= lower and interval.upper <= upper:
        return TriangleClassResult(
            triangle_class, Disposition.PRESENT, "interval_inside_class_band"
        )
    if interval.upper < lower or interval.lower > upper:
        return TriangleClassResult(
            triangle_class,
            Disposition.CERTIFIED_ABSENT,
            "interval_disjoint_from_class_band",
        )
    return TriangleClassResult(
        triangle_class, Disposition.INDETERMINATE, "interval_overlaps_class_guard"
    )


def _classify(
    measurements: tuple[TriangleVariantMeasurement, ...]
) -> tuple[TriangleClassResult, ...]:
    minimum_angle = _envelope(
        measurements, "minimum_interior_angle_millidegrees"
    )
    maximum_angle = _envelope(
        measurements, "maximum_interior_angle_millidegrees"
    )
    side_ratio = _envelope(
        measurements, "maximum_to_minimum_side_ratio_ppm"
    )

    if (
        side_ratio.upper <= _EQUILATERAL_MAX_SIDE_RATIO_PPM
        and minimum_angle.lower >= _EQUILATERAL_MIN_ANGLE_MDEG
        and maximum_angle.upper <= _EQUILATERAL_MAX_ANGLE_MDEG
    ):
        equilateral = TriangleClassResult(
            TriangleClass.EQUILATERAL,
            Disposition.PRESENT,
            "angle_and_side_intervals_inside_equilateral_band",
        )
    elif (
        side_ratio.lower > _EQUILATERAL_MAX_SIDE_RATIO_PPM
        or minimum_angle.upper < _EQUILATERAL_MIN_ANGLE_MDEG
        or maximum_angle.lower > _EQUILATERAL_MAX_ANGLE_MDEG
    ):
        equilateral = TriangleClassResult(
            TriangleClass.EQUILATERAL,
            Disposition.CERTIFIED_ABSENT,
            "angle_or_side_interval_disjoint_from_equilateral_band",
        )
    else:
        equilateral = TriangleClassResult(
            TriangleClass.EQUILATERAL,
            Disposition.INDETERMINATE,
            "equilateral_interval_overlaps_guard",
        )

    right = _band_result(
        TriangleClass.RIGHT,
        maximum_angle,
        lower=_RIGHT_MIN_MDEG,
        upper=_RIGHT_MAX_MDEG,
    )
    if maximum_angle.lower > _OBTUSE_MIN_MDEG:
        obtuse = TriangleClassResult(
            TriangleClass.OBTUSE,
            Disposition.PRESENT,
            "maximum_angle_strictly_above_obtuse_guard",
        )
    elif maximum_angle.upper <= _OBTUSE_MIN_MDEG:
        obtuse = TriangleClassResult(
            TriangleClass.OBTUSE,
            Disposition.CERTIFIED_ABSENT,
            "maximum_angle_at_or_below_obtuse_guard",
        )
    else:
        obtuse = TriangleClassResult(
            TriangleClass.OBTUSE,
            Disposition.INDETERMINATE,
            "maximum_angle_overlaps_obtuse_guard",
        )
    if maximum_angle.upper < _ACUTE_MAX_MDEG:
        acute = TriangleClassResult(
            TriangleClass.ACUTE,
            Disposition.PRESENT,
            "maximum_angle_strictly_below_acute_guard",
        )
    elif maximum_angle.lower >= _ACUTE_MAX_MDEG:
        acute = TriangleClassResult(
            TriangleClass.ACUTE,
            Disposition.CERTIFIED_ABSENT,
            "maximum_angle_at_or_above_acute_guard",
        )
    else:
        acute = TriangleClassResult(
            TriangleClass.ACUTE,
            Disposition.INDETERMINATE,
            "maximum_angle_overlaps_acute_guard",
        )
    by_class = {
        item.triangle_class: item
        for item in (equilateral, right, obtuse, acute)
    }
    return tuple(by_class[item] for item in TriangleClass)


def _guard_small_loop_absence(
    classes: tuple[TriangleClassResult, ...],
    *,
    absence_eligible: bool,
) -> tuple[TriangleClassResult, ...]:
    if absence_eligible:
        return classes
    return tuple(
        TriangleClassResult(
            item.triangle_class,
            Disposition.INDETERMINATE,
            "small_loop_below_absence_resolution_guard",
        )
        if item.disposition is Disposition.CERTIFIED_ABSENT
        else item
        for item in classes
    )


def extract_triangle_geometry(packet: LoopScenePacket) -> TriangleGeometryPacket:
    """Measure every loop in canonical scenario/object order."""

    if not isinstance(packet, LoopScenePacket):
        raise TypeError("packet must be a LoopScenePacket")
    if LoopScenePacket.from_data(packet.to_data()) != packet:
        raise TriangleGeometryError("loop scene packet is not canonical")
    minimum_absence_area = max(
        _MIN_CERTIFIED_ABSENCE_PIXELS,
        (
            packet.width_pixels
            * packet.height_pixels
            * _MIN_CERTIFIED_ABSENCE_AREA_PPM
            + _PPM
            - 1
        )
        // _PPM,
    )
    observations: list[TriangleGeometryObservation] = []
    for scenario in packet.scenarios:
        for loop in scenario.loops:
            object_id = f"{scenario.scenario_id}/loop/{loop.loop_id}"
            polygon = loop.polygon
            absence_eligible = loop.area_pixels >= minimum_absence_area
            if polygon.disposition is Disposition.INDETERMINATE:
                classes = tuple(
                    TriangleClassResult(
                        item,
                        Disposition.INDETERMINATE,
                        "polygon_fit_indeterminate",
                    )
                    for item in TriangleClass
                )
                measurements: tuple[TriangleVariantMeasurement, ...] = ()
            elif polygon.side_count is None or polygon.side_count.lower != 3:
                classes = tuple(
                    TriangleClassResult(
                        item,
                        (
                            Disposition.CERTIFIED_ABSENT
                            if absence_eligible
                            else Disposition.INDETERMINATE
                        ),
                        (
                            "stable_polygon_side_count_is_not_three"
                            if absence_eligible
                            else "small_loop_below_absence_resolution_guard"
                        ),
                    )
                    for item in TriangleClass
                )
                measurements = ()
            else:
                measurements = tuple(
                    _measure_variant(item) for item in polygon.variants
                )
                classes = _guard_small_loop_absence(
                    _classify(measurements),
                    absence_eligible=absence_eligible,
                )
            observations.append(
                TriangleGeometryObservation(
                    object_id=object_id,
                    source_loop_digest=loop.digest(),
                    polygon_disposition=polygon.disposition,
                    variants=measurements,
                    classes=classes,
                )
            )
    return TriangleGeometryPacket(
        panel_digest=packet.panel_digest,
        loop_scene_packet_digest=packet.digest(),
        algorithm_digest=triangle_geometry_algorithm_digest(),
        observations=tuple(sorted(observations, key=lambda item: item.object_id)),
    )


def verify_triangle_geometry(
    value: TriangleGeometryPacket, packet: LoopScenePacket
) -> TriangleGeometryPacket:
    """Cold-recompute a triangle packet without pixels or a model."""

    if not isinstance(value, TriangleGeometryPacket):
        raise TypeError("value must be a TriangleGeometryPacket")
    replay = extract_triangle_geometry(packet)
    if replay != value:
        raise TriangleGeometryError("triangle geometry differs from replay")
    return value


__all__ = (
    "TRIANGLE_GEOMETRY_ALGORITHM_ID",
    "TRIANGLE_GEOMETRY_OBSERVATION_SCHEMA",
    "TRIANGLE_GEOMETRY_PACKET_SCHEMA",
    "TRIANGLE_VARIANT_MEASUREMENT_SCHEMA",
    "TriangleClass",
    "TriangleClassResult",
    "TriangleGeometryError",
    "TriangleGeometryObservation",
    "TriangleGeometryPacket",
    "TriangleVariantMeasurement",
    "extract_triangle_geometry",
    "triangle_geometry_algorithm_digest",
    "verify_triangle_geometry",
)
