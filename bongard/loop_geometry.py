"""Candidate-independent geometry for one enclosed raster loop.

The base visual witness extractor already discovers every bounded background
region (``HoleWitness``), but it intentionally discards the exact mask after
committing its digest.  This module is the next additive observation cell: it
reconstructs one exact hole mask, emits its directed unit-lattice boundary,
and measures polygon side count and edge obliqueness under a small frozen
tolerance ladder.

Nothing here receives a task, label, support side, candidate sentence, or
predicate threshold.  All serialized quantities are integers.  Floating
point is confined to the private numerical fit and is outward-rounded before
it crosses the witness boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from bongard.artifacts import canonical_digest
from bongard.contour_witnesses import Q16Point
from bongard.evidence import Disposition
from bongard.visual_witnesses import HoleWitness
from bongard import visual_witnesses as _base


LOOP_GEOMETRY_SCHEMA = "gkm.bongard-loop-geometry-witness.v1"
POLYGON_VARIANT_SCHEMA = "gkm.bongard-loop-polygon-variant.v1"
POLYGON_OBSERVATION_SCHEMA = "gkm.bongard-loop-polygon-observation.v1"
OBLIQUENESS_OBSERVATION_SCHEMA = "gkm.bongard-loop-obliqueness-observation.v1"
SUBSTANTIVENESS_OBSERVATION_SCHEMA = (
    "gkm.bongard-loop-substantiveness-observation.v1"
)
LOOP_GEOMETRY_ALGORITHM_ID = "bongard.loop-geometry/unit-boundary-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LOOP_ID = re.compile(r"loop-[0-9]{8}\Z")
_HOLE_ID = re.compile(r"hole-[0-9]{8}\Z")

# A loop below this exact sampling floor remains enumerated, but its semantic
# geometry is unresolved.  It is never silently deleted as a nuisance object.
_MIN_HOLE_PIXELS_FOR_GEOMETRY = 9
_MIN_BOUNDARY_EDGES_FOR_GEOMETRY = 20
_RESAMPLE_COUNT = 128
_MAX_POLYGON_RESIDUAL_PPM = 50_000

# These two variants were inherited from the already-tested crack-lab turning
# detector, reduced to the stable middle/conservative fits.  They are part of
# the source-bound algorithm identity, not candidate-proposed thresholds.
_POLYGON_VARIANTS = (
    ("turn033-step08-window10", 33_000, 8, 10),
    ("turn038-step09-window11", 38_000, 9, 11),
)

_POLYGON_REASONS = frozenset(
    {
        "stable_frozen_ladder",
        "variant_disagreement",
        "variant_unavailable",
        "undersampled_loop",
        "non_simple_boundary",
        "no_admissible_polygon_fit",
    }
)
_OBLIQUENESS_REASONS = frozenset(
    {
        "stable_polygon_edges",
        "polygon_fit_indeterminate",
        "no_qualifying_edges",
    }
)


@dataclass(frozen=True, slots=True)
class SubstantivenessObservation:
    """Explicit membership in the frozen geometry-resolution domain.

    Every hole remains in the packet.  This observation only decides whether
    the exact raster support is large enough to enter semantic role
    enumeration; it does not claim that a smaller hole does not exist.
    """

    disposition: Disposition
    minimum_area_pixels: int
    minimum_boundary_edges: int
    reason_code: str
    certificate: str | None = None

    def __post_init__(self) -> None:
        if self.disposition not in {
            Disposition.PRESENT,
            Disposition.CERTIFIED_ABSENT,
        }:
            raise ValueError(
                "exact substantiveness is present or certified absent"
            )
        if self.minimum_area_pixels != _MIN_HOLE_PIXELS_FOR_GEOMETRY:
            raise ValueError("substantiveness area floor differs from protocol")
        if self.minimum_boundary_edges != _MIN_BOUNDARY_EDGES_FOR_GEOMETRY:
            raise ValueError("substantiveness boundary floor differs from protocol")
        if self.disposition is Disposition.PRESENT:
            if self.reason_code != "meets_geometry_resolution_floor":
                raise ValueError("present substantiveness requires the meet reason")
            if self.certificate is not None:
                raise ValueError("present substantiveness cannot carry a certificate")
        else:
            if self.reason_code != "below_geometry_resolution_floor":
                raise ValueError("absent substantiveness requires the below-floor reason")
            if self.certificate != (
                "exact area or boundary support is below the frozen semantic-role floor"
            ):
                raise ValueError("substantiveness absence certificate differs")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SUBSTANTIVENESS_OBSERVATION_SCHEMA,
            "disposition": self.disposition.value,
            "minimum_area_pixels": self.minimum_area_pixels,
            "minimum_boundary_edges": self.minimum_boundary_edges,
            "reason_code": self.reason_code,
            "certificate": self.certificate,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SubstantivenessObservation":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "disposition",
                    "minimum_area_pixels",
                    "minimum_boundary_edges",
                    "reason_code",
                    "certificate",
                }
            ),
            "substantiveness observation",
        )
        if data["schema"] != SUBSTANTIVENESS_OBSERVATION_SCHEMA:
            raise ValueError("unsupported substantiveness observation")
        certificate = data["certificate"]
        if certificate is not None and not isinstance(certificate, str):
            raise TypeError("substantiveness certificate must be a string or null")
        return cls(
            disposition=Disposition(data["disposition"]),
            minimum_area_pixels=data["minimum_area_pixels"],
            minimum_boundary_edges=data["minimum_boundary_edges"],
            reason_code=data["reason_code"],
            certificate=certificate,
        )


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


def loop_geometry_source_digest() -> str:
    """Return the exact implementation-file digest."""

    return _source_digest()


def loop_geometry_algorithm_digest() -> str:
    """Return the source/dependency/constant-bound algorithm identity."""

    return canonical_digest(
        {
            "algorithm_id": LOOP_GEOMETRY_ALGORITHM_ID,
            "source_digest": _source_digest(),
            "base_visual_extractor_digest": _base.visual_witness_extractor_digest(),
            "boundary": {
                "representation": "directed-unit-lattice-edges",
                "cell_connectivity": 4,
                "junction_rule": "right-straight-left-back",
                "cycle_selection": "exactly-one-cycle-required",
            },
            "sampling_floor": {
                "hole_pixels": _MIN_HOLE_PIXELS_FOR_GEOMETRY,
                "boundary_edges": _MIN_BOUNDARY_EDGES_FOR_GEOMETRY,
            },
            "polygon": {
                "resample_count": _RESAMPLE_COUNT,
                "max_residual_ppm": _MAX_POLYGON_RESIDUAL_PPM,
                "variants": [
                    {
                        "variant_id": variant_id,
                        "turn_threshold_millidegrees": threshold,
                        "arc_step": step,
                        "exclusion_window": window,
                    }
                    for variant_id, threshold, step, window in _POLYGON_VARIANTS
                ],
            },
            "obliqueness": {
                "definition": "minimum edge distance to panel axes modulo 90deg",
                "range_millidegrees": [0, 45_000],
                "uncertainty": "atan(polygon-residual) outward-rounded",
            },
        }
    )


@dataclass(frozen=True, order=True, slots=True)
class IntInterval:
    """Closed integer interval used at the serialized geometry boundary."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        _integer(self.lower, "integer interval lower")
        _integer(self.upper, "integer interval upper")
        if self.lower > self.upper:
            raise ValueError("integer interval lower exceeds upper")

    @property
    def exact(self) -> bool:
        return self.lower == self.upper

    @classmethod
    def point(cls, value: int) -> "IntInterval":
        checked = _integer(value, "integer interval point")
        return cls(checked, checked)

    def to_data(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "IntInterval":
        _exact_fields(data, frozenset({"lower", "upper"}), "integer interval")
        return cls(data["lower"], data["upper"])


@dataclass(frozen=True, slots=True)
class PolygonVariantWitness:
    """One frozen polygon fit, retained instead of hiding model selection."""

    variant_id: str
    side_count: int
    residual_ppm_upper: int
    vertices_q16: tuple[Q16Point, ...]
    minimum_edge_obliqueness_millidegrees: IntInterval | None

    def __post_init__(self) -> None:
        if self.variant_id not in {item[0] for item in _POLYGON_VARIANTS}:
            raise ValueError("unknown polygon variant_id")
        _integer(self.side_count, "polygon side_count")
        _integer(self.residual_ppm_upper, "polygon residual_ppm_upper")
        if not isinstance(self.vertices_q16, tuple) or any(
            not isinstance(item, Q16Point) for item in self.vertices_q16
        ):
            raise TypeError("polygon vertices_q16 must be a typed tuple")
        if len(self.vertices_q16) != self.side_count:
            raise ValueError("polygon vertex count disagrees with side_count")
        if (
            self.minimum_edge_obliqueness_millidegrees is not None
            and not isinstance(
                self.minimum_edge_obliqueness_millidegrees, IntInterval
            )
        ):
            raise TypeError("polygon obliqueness must be an IntInterval or null")
        if self.minimum_edge_obliqueness_millidegrees is not None and (
            self.minimum_edge_obliqueness_millidegrees.upper > 45_000
        ):
            raise ValueError("edge obliqueness exceeds 45 degrees")

    @property
    def admissible(self) -> bool:
        return (
            self.side_count >= 3
            and self.residual_ppm_upper <= _MAX_POLYGON_RESIDUAL_PPM
            and self.minimum_edge_obliqueness_millidegrees is not None
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": POLYGON_VARIANT_SCHEMA,
            "variant_id": self.variant_id,
            "side_count": self.side_count,
            "residual_ppm_upper": self.residual_ppm_upper,
            "vertices_q16": [item.to_data() for item in self.vertices_q16],
            "minimum_edge_obliqueness_millidegrees": (
                None
                if self.minimum_edge_obliqueness_millidegrees is None
                else self.minimum_edge_obliqueness_millidegrees.to_data()
            ),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PolygonVariantWitness":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "variant_id",
                    "side_count",
                    "residual_ppm_upper",
                    "vertices_q16",
                    "minimum_edge_obliqueness_millidegrees",
                }
            ),
            "polygon variant witness",
        )
        if data["schema"] != POLYGON_VARIANT_SCHEMA:
            raise ValueError("unsupported polygon variant witness")
        vertices = data["vertices_q16"]
        obliqueness = data["minimum_edge_obliqueness_millidegrees"]
        if not isinstance(vertices, list) or any(
            not isinstance(item, Mapping) for item in vertices
        ):
            raise TypeError("polygon vertices_q16 must be an object list")
        if obliqueness is not None and not isinstance(obliqueness, Mapping):
            raise TypeError("polygon obliqueness must be an object or null")
        return cls(
            variant_id=data["variant_id"],
            side_count=data["side_count"],
            residual_ppm_upper=data["residual_ppm_upper"],
            vertices_q16=tuple(Q16Point.from_data(item) for item in vertices),
            minimum_edge_obliqueness_millidegrees=(
                None if obliqueness is None else IntInterval.from_data(obliqueness)
            ),
        )


@dataclass(frozen=True, slots=True)
class PolygonFitObservation:
    disposition: Disposition
    side_count: IntInterval | None
    residual_ppm_upper: IntInterval | None
    variants: tuple[PolygonVariantWitness, ...]
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, Disposition):
            raise TypeError("polygon disposition must be a Disposition")
        if self.disposition not in {Disposition.PRESENT, Disposition.INDETERMINATE}:
            raise ValueError("candidate-independent polygon fit is present or indeterminate")
        if self.side_count is not None and not isinstance(self.side_count, IntInterval):
            raise TypeError("polygon side_count must be an IntInterval or null")
        if self.residual_ppm_upper is not None and not isinstance(
            self.residual_ppm_upper, IntInterval
        ):
            raise TypeError("polygon residual must be an IntInterval or null")
        if not isinstance(self.variants, tuple) or any(
            not isinstance(item, PolygonVariantWitness) for item in self.variants
        ):
            raise TypeError("polygon variants must be a typed tuple")
        if self.variants and tuple(item.variant_id for item in self.variants) != tuple(
            item[0] for item in _POLYGON_VARIANTS
        ):
            raise ValueError("polygon variants must retain the complete frozen order")
        if self.reason_code not in _POLYGON_REASONS:
            raise ValueError("unknown polygon reason_code")
        admissible = tuple(item for item in self.variants if item.admissible)
        if self.side_count is None:
            if self.residual_ppm_upper is not None or admissible:
                raise ValueError("unavailable polygon interval disagrees with variants")
        else:
            if not admissible or self.residual_ppm_upper is None:
                raise ValueError("polygon interval requires admissible variants")
            counts = tuple(item.side_count for item in admissible)
            residuals = tuple(item.residual_ppm_upper for item in admissible)
            if self.side_count != IntInterval(min(counts), max(counts)):
                raise ValueError("polygon side interval does not envelope variants")
            if self.residual_ppm_upper != IntInterval(min(residuals), max(residuals)):
                raise ValueError("polygon residual interval does not envelope variants")
        if self.disposition is Disposition.PRESENT:
            if self.side_count is None or not self.side_count.exact:
                raise ValueError("present polygon fit requires an exact side count")
            if len(admissible) != len(_POLYGON_VARIANTS):
                raise ValueError(
                    "present polygon fit requires every frozen variant to be admissible"
                )
            if self.reason_code != "stable_frozen_ladder":
                raise ValueError("present polygon fit requires the stable reason")
        elif self.reason_code == "stable_frozen_ladder":
            raise ValueError("indeterminate polygon fit cannot claim stable ladder")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": POLYGON_OBSERVATION_SCHEMA,
            "disposition": self.disposition.value,
            "side_count": None if self.side_count is None else self.side_count.to_data(),
            "residual_ppm_upper": (
                None
                if self.residual_ppm_upper is None
                else self.residual_ppm_upper.to_data()
            ),
            "variants": [item.to_data() for item in self.variants],
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PolygonFitObservation":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "disposition",
                    "side_count",
                    "residual_ppm_upper",
                    "variants",
                    "reason_code",
                }
            ),
            "polygon fit observation",
        )
        if data["schema"] != POLYGON_OBSERVATION_SCHEMA:
            raise ValueError("unsupported polygon fit observation")
        variants = data["variants"]
        side_count = data["side_count"]
        residual = data["residual_ppm_upper"]
        if not isinstance(variants, list) or any(
            not isinstance(item, Mapping) for item in variants
        ):
            raise TypeError("polygon variants must be an object list")
        for value, label in ((side_count, "side_count"), (residual, "residual")):
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"polygon {label} must be an object or null")
        if not isinstance(data["disposition"], str):
            raise TypeError("polygon disposition must be a string")
        return cls(
            disposition=Disposition(data["disposition"]),
            side_count=None if side_count is None else IntInterval.from_data(side_count),
            residual_ppm_upper=(
                None if residual is None else IntInterval.from_data(residual)
            ),
            variants=tuple(PolygonVariantWitness.from_data(item) for item in variants),
            reason_code=data["reason_code"],
        )


@dataclass(frozen=True, slots=True)
class EdgeObliquenessObservation:
    disposition: Disposition
    minimum_millidegrees: IntInterval | None
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, Disposition):
            raise TypeError("obliqueness disposition must be a Disposition")
        if self.disposition not in {Disposition.PRESENT, Disposition.INDETERMINATE}:
            raise ValueError("candidate-independent obliqueness is present or indeterminate")
        if self.minimum_millidegrees is not None:
            if not isinstance(self.minimum_millidegrees, IntInterval):
                raise TypeError("obliqueness must be an IntInterval or null")
            if self.minimum_millidegrees.upper > 45_000:
                raise ValueError("obliqueness exceeds 45 degrees")
        if self.reason_code not in _OBLIQUENESS_REASONS:
            raise ValueError("unknown obliqueness reason_code")
        if self.disposition is Disposition.PRESENT:
            if self.minimum_millidegrees is None:
                raise ValueError("present obliqueness requires an interval")
            if self.reason_code != "stable_polygon_edges":
                raise ValueError("present obliqueness requires stable polygon edges")
        elif self.reason_code == "stable_polygon_edges":
            raise ValueError("indeterminate obliqueness cannot claim stable edges")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": OBLIQUENESS_OBSERVATION_SCHEMA,
            "disposition": self.disposition.value,
            "minimum_millidegrees": (
                None
                if self.minimum_millidegrees is None
                else self.minimum_millidegrees.to_data()
            ),
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "EdgeObliquenessObservation":
        _exact_fields(
            data,
            frozenset(
                {"schema", "disposition", "minimum_millidegrees", "reason_code"}
            ),
            "edge obliqueness observation",
        )
        if data["schema"] != OBLIQUENESS_OBSERVATION_SCHEMA:
            raise ValueError("unsupported edge obliqueness observation")
        interval = data["minimum_millidegrees"]
        if interval is not None and not isinstance(interval, Mapping):
            raise TypeError("obliqueness interval must be an object or null")
        if not isinstance(data["disposition"], str):
            raise TypeError("obliqueness disposition must be a string")
        return cls(
            disposition=Disposition(data["disposition"]),
            minimum_millidegrees=(
                None if interval is None else IntInterval.from_data(interval)
            ),
            reason_code=data["reason_code"],
        )


@dataclass(frozen=True, slots=True)
class LoopGeometryWitness:
    loop_id: str
    source_hole_id: str
    owner_component_id: str | None
    source_hole_digest: str
    source_mask_digest: str
    area_pixels: int
    boundary_digest: str
    boundary_edge_count: int
    boundary_cycle_count: int
    substantiveness: SubstantivenessObservation
    polygon: PolygonFitObservation
    edge_obliqueness: EdgeObliquenessObservation

    def __post_init__(self) -> None:
        if not isinstance(self.loop_id, str) or _LOOP_ID.fullmatch(self.loop_id) is None:
            raise ValueError("loop_id is not canonical")
        if (
            not isinstance(self.source_hole_id, str)
            or _HOLE_ID.fullmatch(self.source_hole_id) is None
        ):
            raise ValueError("source_hole_id is not canonical")
        if self.loop_id != self.source_hole_id.replace("hole-", "loop-", 1):
            raise ValueError("loop_id must preserve its source-hole ordinal")
        if self.owner_component_id is not None and not isinstance(
            self.owner_component_id, str
        ):
            raise TypeError("owner_component_id must be a string or null")
        _digest(self.source_hole_digest, "source_hole_digest")
        _digest(self.source_mask_digest, "source_mask_digest")
        _integer(self.area_pixels, "loop area_pixels", minimum=1)
        _digest(self.boundary_digest, "boundary_digest")
        _integer(self.boundary_edge_count, "boundary_edge_count", minimum=4)
        _integer(self.boundary_cycle_count, "boundary_cycle_count", minimum=1)
        if not isinstance(self.substantiveness, SubstantivenessObservation):
            raise TypeError("loop substantiveness must be an observation")
        if not isinstance(self.polygon, PolygonFitObservation):
            raise TypeError("loop polygon must be a PolygonFitObservation")
        if not isinstance(self.edge_obliqueness, EdgeObliquenessObservation):
            raise TypeError("loop edge_obliqueness must be an observation")
        if self.polygon.disposition is Disposition.PRESENT:
            if self.edge_obliqueness.disposition is not Disposition.PRESENT:
                raise ValueError("stable polygon must expose stable edge obliqueness")
        elif self.edge_obliqueness.disposition is Disposition.PRESENT:
            raise ValueError("unstable polygon cannot expose present obliqueness")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": LOOP_GEOMETRY_SCHEMA,
            "loop_id": self.loop_id,
            "source_hole_id": self.source_hole_id,
            "owner_component_id": self.owner_component_id,
            "source_hole_digest": self.source_hole_digest,
            "source_mask_digest": self.source_mask_digest,
            "area_pixels": self.area_pixels,
            "boundary_digest": self.boundary_digest,
            "boundary_edge_count": self.boundary_edge_count,
            "boundary_cycle_count": self.boundary_cycle_count,
            "substantiveness": self.substantiveness.to_data(),
            "polygon": self.polygon.to_data(),
            "edge_obliqueness": self.edge_obliqueness.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "LoopGeometryWitness":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "loop_id",
                    "source_hole_id",
                    "owner_component_id",
                    "source_hole_digest",
                    "source_mask_digest",
                    "area_pixels",
                    "boundary_digest",
                    "boundary_edge_count",
                    "boundary_cycle_count",
                    "substantiveness",
                    "polygon",
                    "edge_obliqueness",
                }
            ),
            "loop geometry witness",
        )
        if data["schema"] != LOOP_GEOMETRY_SCHEMA:
            raise ValueError("unsupported loop geometry witness")
        polygon = data["polygon"]
        substantiveness = data["substantiveness"]
        obliqueness = data["edge_obliqueness"]
        if (
            not isinstance(substantiveness, Mapping)
            or not isinstance(polygon, Mapping)
            or not isinstance(obliqueness, Mapping)
        ):
            raise TypeError("loop geometry nested observations must be objects")
        owner = data["owner_component_id"]
        if owner is not None and not isinstance(owner, str):
            raise TypeError("loop owner_component_id must be a string or null")
        return cls(
            loop_id=data["loop_id"],
            source_hole_id=data["source_hole_id"],
            owner_component_id=owner,
            source_hole_digest=data["source_hole_digest"],
            source_mask_digest=data["source_mask_digest"],
            area_pixels=data["area_pixels"],
            boundary_digest=data["boundary_digest"],
            boundary_edge_count=data["boundary_edge_count"],
            boundary_cycle_count=data["boundary_cycle_count"],
            substantiveness=SubstantivenessObservation.from_data(substantiveness),
            polygon=PolygonFitObservation.from_data(polygon),
            edge_obliqueness=EdgeObliquenessObservation.from_data(obliqueness),
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


_DIRECTION_INDEX = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
_TURN_PRIORITY = {1: 0, 0: 1, 3: 2, 2: 3}


def _directed_boundary_edges(
    region: np.ndarray,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Emit cell edges clockwise in screen coordinates (interior on right)."""

    height, width = region.shape
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for raw_y, raw_x in np.argwhere(region):
        y, x = int(raw_y), int(raw_x)
        if y == 0 or not region[y - 1, x]:
            edges.append(((x, y), (x + 1, y)))
        if x == width - 1 or not region[y, x + 1]:
            edges.append(((x + 1, y), (x + 1, y + 1)))
        if y == height - 1 or not region[y + 1, x]:
            edges.append(((x + 1, y + 1), (x, y + 1)))
        if x == 0 or not region[y, x - 1]:
            edges.append(((x, y + 1), (x, y)))
    return tuple(sorted(edges))


def _stitch_boundary_cycles(
    edges: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
) -> tuple[np.ndarray, ...]:
    unused = set(edges)
    cycles: list[np.ndarray] = []
    while unused:
        first = min(unused)
        unused.remove(first)
        previous, current = first
        points = [previous, current]
        while current != points[0]:
            candidates = [edge for edge in unused if edge[0] == current]
            if not candidates:
                raise ValueError("directed loop boundary does not close")
            incoming = _DIRECTION_INDEX[
                (current[0] - previous[0], current[1] - previous[1])
            ]

            def key(
                edge: tuple[tuple[int, int], tuple[int, int]]
            ) -> tuple[int, tuple[tuple[int, int], tuple[int, int]]]:
                outgoing = _DIRECTION_INDEX[
                    (edge[1][0] - edge[0][0], edge[1][1] - edge[0][1])
                ]
                return _TURN_PRIORITY[(outgoing - incoming) % 4], edge

            chosen = min(candidates, key=key)
            unused.remove(chosen)
            previous, current = chosen
            points.append(current)
            if len(points) > len(edges) + 2:
                raise RuntimeError("loop boundary stitching exceeded edge guard")
        cycles.append(np.asarray(points[:-1], dtype=float))
    cycles.sort(
        key=lambda points: (
            -abs(_signed_twice_area(points)),
            tuple((int(x), int(y)) for x, y in points),
        )
    )
    return tuple(cycles)


def _signed_twice_area(points: np.ndarray) -> float:
    return float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
    )


def _boundary_digest(
    edges: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
) -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-directed-unit-boundary.v1",
            "coordinate_frame": "pixel-cell-lattice/screen-xy",
            "edges": [[list(start), list(end)] for start, end in edges],
        }
    )


def _resample_closed(points: np.ndarray, count: int = _RESAMPLE_COUNT) -> np.ndarray:
    ring = np.vstack((points, points[:1]))
    segments = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segments)))
    if not math.isfinite(float(arc[-1])) or arc[-1] <= np.finfo(float).tiny:
        return np.empty((0, 2), dtype=float)
    positions = np.linspace(0.0, float(arc[-1]), count, endpoint=False)
    return np.stack(
        [np.interp(positions, arc, ring[:, axis]) for axis in range(2)], axis=1
    )


def _turning_profile(points: np.ndarray, step: int) -> np.ndarray:
    result = np.zeros(len(points), dtype=float)
    for index in range(len(points)):
        first = points[index] - points[(index - step) % len(points)]
        second = points[(index + step) % len(points)] - points[index]
        cross = first[0] * second[1] - first[1] * second[0]
        result[index] = abs(math.atan2(float(cross), float(first @ second)))
    return result


def _circular_distance(first: int, second: int, count: int) -> int:
    direct = abs(first - second)
    return min(direct, count - direct)


def _polygon_residual(points: np.ndarray, vertices: np.ndarray) -> float:
    distances = np.full(len(points), np.inf)
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        chord = second - first
        length_squared = float(chord @ chord)
        if length_squared <= np.finfo(float).tiny:
            continue
        projection = np.clip(((points - first) @ chord) / length_squared, 0.0, 1.0)
        projected = first + projection[:, None] * chord
        distances = np.minimum(distances, np.linalg.norm(points - projected, axis=1))
    extent = max(
        float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0
    )
    return float(np.mean(distances) / extent)


def _q16_coordinate(value: float, extent: int) -> int:
    return max(0, min(65_535, int(round(value * 65_535 / extent))))


def _q16_vertices(
    vertices: np.ndarray, width: int, height: int
) -> tuple[Q16Point, ...]:
    return tuple(
        Q16Point(_q16_coordinate(float(x), width), _q16_coordinate(float(y), height))
        for x, y in vertices
    )


def _minimum_edge_obliqueness(
    vertices: np.ndarray, residual: float
) -> IntInterval | None:
    values: list[float] = []
    extent = max(
        float(np.ptp(vertices[:, 0])), float(np.ptp(vertices[:, 1])), 1.0
    )
    minimum_edge_length = max(2.0, 0.08 * extent)
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        vector = second - first
        length = float(np.linalg.norm(vector))
        if not math.isfinite(length) or length < minimum_edge_length:
            continue
        angle = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 90.0
        values.append(min(angle, 90.0 - angle))
    if not values:
        return None
    nominal = min(values) * 1_000.0
    uncertainty = math.degrees(math.atan(max(0.0, residual))) * 1_000.0
    return IntInterval(
        max(0, int(math.floor(nominal - uncertainty))),
        min(45_000, int(math.ceil(nominal + uncertainty))),
    )


def _fit_variant(
    boundary: np.ndarray,
    *,
    variant_id: str,
    threshold_millidegrees: int,
    step: int,
    exclusion_window: int,
    width: int,
    height: int,
) -> PolygonVariantWitness:
    resampled = _resample_closed(boundary)
    if len(resampled) != _RESAMPLE_COUNT:
        return PolygonVariantWitness(variant_id, 0, 1_000_000, (), None)
    turning = _turning_profile(resampled, step)
    accepted: list[int] = []
    threshold = math.radians(threshold_millidegrees / 1_000.0)
    for raw_index in np.argsort(turning)[::-1]:
        if turning[raw_index] < threshold:
            break
        index = int(raw_index)
        if all(
            _circular_distance(index, other, len(resampled)) > exclusion_window
            for other in accepted
        ):
            accepted.append(index)
    accepted.sort()
    vertices = resampled[np.asarray(accepted, dtype=int)] if accepted else np.empty((0, 2))
    residual = (
        _polygon_residual(boundary, vertices)
        if len(vertices) >= 2
        else 1.0
    )
    if not math.isfinite(residual) or residual < 0.0:
        residual = 1.0
    residual_ppm = int(math.ceil(residual * 1_000_000.0))
    obliqueness = (
        _minimum_edge_obliqueness(vertices, residual)
        if len(vertices) >= 3
        else None
    )
    return PolygonVariantWitness(
        variant_id=variant_id,
        side_count=len(vertices),
        residual_ppm_upper=residual_ppm,
        vertices_q16=_q16_vertices(vertices, width, height),
        minimum_edge_obliqueness_millidegrees=obliqueness,
    )


def _unavailable_polygon(reason: str) -> PolygonFitObservation:
    return PolygonFitObservation(
        disposition=Disposition.INDETERMINATE,
        side_count=None,
        residual_ppm_upper=None,
        variants=(),
        reason_code=reason,
    )


def _polygon_observation(
    boundary: np.ndarray, width: int, height: int
) -> PolygonFitObservation:
    variants = tuple(
        _fit_variant(
            boundary,
            variant_id=variant_id,
            threshold_millidegrees=threshold,
            step=step,
            exclusion_window=window,
            width=width,
            height=height,
        )
        for variant_id, threshold, step, window in _POLYGON_VARIANTS
    )
    admissible = tuple(item for item in variants if item.admissible)
    if not admissible:
        return PolygonFitObservation(
            disposition=Disposition.INDETERMINATE,
            side_count=None,
            residual_ppm_upper=None,
            variants=variants,
            reason_code="no_admissible_polygon_fit",
        )
    counts = tuple(item.side_count for item in admissible)
    residuals = tuple(item.residual_ppm_upper for item in admissible)
    interval = IntInterval(min(counts), max(counts))
    complete_ladder = len(admissible) == len(_POLYGON_VARIANTS)
    stable_ladder = complete_ladder and interval.exact
    return PolygonFitObservation(
        disposition=(
            Disposition.PRESENT if stable_ladder else Disposition.INDETERMINATE
        ),
        side_count=interval,
        residual_ppm_upper=IntInterval(min(residuals), max(residuals)),
        variants=variants,
        reason_code=(
            "stable_frozen_ladder"
            if stable_ladder
            else (
                "variant_unavailable"
                if not complete_ladder
                else "variant_disagreement"
            )
        ),
    )


def _obliqueness_observation(
    polygon: PolygonFitObservation,
) -> EdgeObliquenessObservation:
    admissible = tuple(item for item in polygon.variants if item.admissible)
    intervals = tuple(
        item.minimum_edge_obliqueness_millidegrees
        for item in admissible
        if item.minimum_edge_obliqueness_millidegrees is not None
    )
    if polygon.disposition is not Disposition.PRESENT:
        envelope = (
            None
            if not intervals
            else IntInterval(
                min(item.lower for item in intervals),
                max(item.upper for item in intervals),
            )
        )
        return EdgeObliquenessObservation(
            disposition=Disposition.INDETERMINATE,
            minimum_millidegrees=envelope,
            reason_code="polygon_fit_indeterminate",
        )
    if not intervals:
        return EdgeObliquenessObservation(
            disposition=Disposition.INDETERMINATE,
            minimum_millidegrees=None,
            reason_code="no_qualifying_edges",
        )
    return EdgeObliquenessObservation(
        disposition=Disposition.PRESENT,
        minimum_millidegrees=IntInterval(
            min(item.lower for item in intervals),
            max(item.upper for item in intervals),
        ),
        reason_code="stable_polygon_edges",
    )


def extract_loop_geometry(
    hole_mask: np.ndarray,
    source_hole: HoleWitness,
    *,
    width_pixels: int,
    height_pixels: int,
) -> LoopGeometryWitness:
    """Extract one source-bound loop geometry witness from an exact hole mask."""

    if not isinstance(source_hole, HoleWitness):
        raise TypeError("source_hole must be a HoleWitness")
    _integer(width_pixels, "width_pixels", minimum=2)
    _integer(height_pixels, "height_pixels", minimum=2)
    mask = np.asarray(hole_mask)
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise TypeError("hole_mask must be a two-dimensional Boolean array")
    if mask.shape != (height_pixels, width_pixels):
        raise ValueError("hole_mask dimensions differ from the panel")
    if not mask.any():
        raise ValueError("hole_mask cannot be empty")
    mask = np.ascontiguousarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask))
    mask_digest = _base._mask_digest(mask)
    if area != source_hole.area_pixels or mask_digest != source_hole.mask_digest:
        raise ValueError("hole_mask differs from its exact source HoleWitness")

    edges = _directed_boundary_edges(mask)
    cycles = _stitch_boundary_cycles(edges)
    meets_floor = area >= _MIN_HOLE_PIXELS_FOR_GEOMETRY and len(edges) >= (
        _MIN_BOUNDARY_EDGES_FOR_GEOMETRY
    )
    substantiveness = SubstantivenessObservation(
        disposition=(
            Disposition.PRESENT if meets_floor else Disposition.CERTIFIED_ABSENT
        ),
        minimum_area_pixels=_MIN_HOLE_PIXELS_FOR_GEOMETRY,
        minimum_boundary_edges=_MIN_BOUNDARY_EDGES_FOR_GEOMETRY,
        reason_code=(
            "meets_geometry_resolution_floor"
            if meets_floor
            else "below_geometry_resolution_floor"
        ),
        certificate=(
            None
            if meets_floor
            else (
                "exact area or boundary support is below the frozen "
                "semantic-role floor"
            )
        ),
    )
    if not meets_floor:
        polygon = _unavailable_polygon("undersampled_loop")
    elif len(cycles) != 1:
        polygon = _unavailable_polygon("non_simple_boundary")
    else:
        polygon = _polygon_observation(cycles[0], width_pixels, height_pixels)
    obliqueness = _obliqueness_observation(polygon)
    return LoopGeometryWitness(
        loop_id=source_hole.hole_id.replace("hole-", "loop-", 1),
        source_hole_id=source_hole.hole_id,
        owner_component_id=source_hole.owner_component_id,
        source_hole_digest=canonical_digest(source_hole.to_data()),
        source_mask_digest=mask_digest,
        area_pixels=area,
        boundary_digest=_boundary_digest(edges),
        boundary_edge_count=len(edges),
        boundary_cycle_count=len(cycles),
        substantiveness=substantiveness,
        polygon=polygon,
        edge_obliqueness=obliqueness,
    )


def boundary_cycles_for_mask(hole_mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Diagnostic exact boundary cycles; callers must not serialize floats."""

    mask = np.asarray(hole_mask)
    if mask.dtype != np.bool_ or mask.ndim != 2 or not mask.any():
        raise TypeError("hole_mask must be a nonempty two-dimensional Boolean array")
    return _stitch_boundary_cycles(_directed_boundary_edges(mask))


__all__ = [
    "EdgeObliquenessObservation",
    "IntInterval",
    "LOOP_GEOMETRY_ALGORITHM_ID",
    "LOOP_GEOMETRY_SCHEMA",
    "LoopGeometryWitness",
    "PolygonFitObservation",
    "PolygonVariantWitness",
    "SubstantivenessObservation",
    "boundary_cycles_for_mask",
    "extract_loop_geometry",
    "loop_geometry_algorithm_digest",
    "loop_geometry_source_digest",
]
