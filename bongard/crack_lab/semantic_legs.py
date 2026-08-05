"""Typed visual legs for semantic Bongard cones.

The unrestricted predicate path still lives in ``bongard_arena.py`` and
``bongard_legs.py``.  This module is the semantic-pure basis: every arrow has
an auditable contract and returns either a typed witness or a scalar
measurement derived from such witnesses.

Runtime dispositions are explicit: success is present evidence,
``WitnessAbsent`` is established semantic non-membership,
``WitnessIndeterminate`` is unresolved evidence, and ordinary exceptions are
implementation errors.  Only the first two may participate in a Boolean
semantic decision.

Honesty invariant: a witness-producing leg must verify the structure it
claims.  ``detect_contact`` returns a ContactWitness only when parts actually
meet at a junction; when the relation is absent it raises ``WitnessAbsent`` instead of
fabricating evidence.  Absence claims are expressed through the honest
counting measurements (``contact_count``, ``intersection_count``,
``part_count``) which return 0 rather than raising.
"""
from __future__ import annotations

import math
import re
from numbers import Real
from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable

import numpy as np

from soft_semantics import (
    OBLIQUENESS_CALIBRATOR,
    SoftAbsent,
    SoftError,
    SoftEvidence,
    SoftEvidenceSet,
    SoftResult,
    fuzzy_all,
    fuzzy_any,
    fuzzy_max,
    fuzzy_mean,
    fuzzy_min,
    fuzzy_not,
    soft_add,
    soft_pair,
)
from visual_witnesses import (
    AngleWitness,
    ArcWitness,
    CircleIntersectionWitness,
    CirclePairWitness,
    CircleWitness,
    ContactWitness,
    ContourWitness,
    CurveWitness,
    ExteriorGapWitness,
    IncidentRayWitness,
    IntersectionWitness,
    LineSegmentWitness,
    PairWitness,
    PartGraphWitness,
    PartWitness,
    PointWitness,
    PointContactSignature,
    PolygonWitness,
    PrototypeWitness,
    QuadrilateralWitness,
    RadialArrangementWitness,
    SkeletonGraphWitness,
    SymmetryWitness,
    TriangleWitness,
    Witness,
)


class WitnessAbsent(ValueError):
    """Expected failure to construct a claimed semantic witness.

    A negative Bongard panel often lacks the structure named by a candidate
    cone.  That is evidence for a negative decision, not a predicate crash.
    Witness-producing legs raise this exception for an honestly absent
    structure; numerical failures and implementation defects continue to
    raise ordinary exceptions and are counted by the verifier.  The machine
    readable ``failure_mode`` must be advertised by the leg's contract or the
    compiler treats the exception as an implementation error.
    """

    def __init__(self, failure_mode: str, message: str | None = None) -> None:
        self.failure_mode = str(failure_mode)
        super().__init__(message if message is not None else self.failure_mode)


class WitnessIndeterminate(ValueError):
    """Expected inability to decide whether a semantic witness is present.

    This is epistemically different from :class:`WitnessAbsent`: the
    implementation had insufficient or poor-quality evidence, so the panel
    must not be classified as though the requested structure were absent.
    The machine-readable mode must be declared in the leg contract's
    ``indeterminate_modes``.
    """

    def __init__(self, failure_mode: str, message: str | None = None) -> None:
        self.failure_mode = str(failure_mode)
        super().__init__(message if message is not None else self.failure_mode)


@dataclass(frozen=True)
class ObjectMask:
    mask: np.ndarray
    object_id: str = "object"


@dataclass(frozen=True)
class Scene:
    panel: np.ndarray
    objects: tuple[ObjectMask, ...]


@dataclass(frozen=True)
class LegContract:
    name: str
    domain: tuple[str, ...]
    codomain: str
    implementation: Callable
    complexity: int = 1
    invariances: frozenset[str] = frozenset()
    equivariances: frozenset[str] = frozenset()
    # ``failure_modes`` are constructive semantic absences: the requested
    # witness is known not to exist.  ``indeterminate_modes`` mean the
    # extractor could not decide.  Keeping these static and disjoint prevents
    # a parser limitation from becoming negative evidence.
    failure_modes: tuple[str, ...] = ()
    indeterminate_modes: tuple[str, ...] = ()
    version: str = "0.1"
    proxy_for: tuple[str, ...] = ()
    measurement_kind: str | None = None
    proxy_directions: tuple[tuple[str, str], ...] = ()


class LegRegistry:
    def __init__(self) -> None:
        self._legs: dict[str, LegContract] = {}

    def register(self, contract: LegContract) -> None:
        if contract.name in self._legs:
            raise ValueError(f"duplicate leg {contract.name}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", contract.name):
            raise ValueError(f"invalid leg name {contract.name!r}")
        if not contract.domain or any(not str(item).strip()
                                      for item in contract.domain):
            raise ValueError(f"{contract.name}: domain types must be non-empty")
        if not contract.codomain.strip():
            raise ValueError(f"{contract.name}: codomain must be non-empty")
        measurement_kinds = {"continuous", "count", "binary"}
        if contract.codomain == "Measurement":
            if contract.measurement_kind not in measurement_kinds:
                raise ValueError(
                    f"{contract.name}: Measurement codomain requires one of "
                    + ", ".join(sorted(measurement_kinds)))
        elif contract.measurement_kind is not None:
            raise ValueError(
                f"{contract.name}: measurement_kind is only valid for "
                "Measurement codomains")
        if isinstance(contract.complexity, bool) \
                or not isinstance(contract.complexity, int) \
                or contract.complexity <= 0:
            raise ValueError(f"{contract.name}: complexity must be a positive integer")
        if not callable(contract.implementation):
            raise TypeError(f"{contract.name}: implementation must be callable")
        try:
            signature(contract.implementation).bind(
                *([None] * len(contract.domain)))
        except TypeError as exc:
            raise ValueError(
                f"{contract.name}: implementation does not accept declared "
                f"arity {len(contract.domain)}: {exc}") from exc
        actions = {"translation", "uniform_scale", "rotation", "reflection"}
        unknown_actions = (
            set(contract.invariances) | set(contract.equivariances)) - actions
        if unknown_actions:
            raise ValueError(
                f"{contract.name}: unknown transform contracts "
                + ", ".join(sorted(unknown_actions)))
        overlap = set(contract.invariances) & set(contract.equivariances)
        if overlap:
            raise ValueError(
                f"{contract.name}: transforms cannot be both invariant and "
                "equivariant: " + ", ".join(sorted(overlap)))
        for label, values in (
                ("failure mode", contract.failure_modes),
                ("indeterminate mode", contract.indeterminate_modes),
                ("proxy term", contract.proxy_for)):
            if len(values) != len(set(values)):
                raise ValueError(f"{contract.name}: duplicate {label}")
            if any(not str(value).strip() for value in values):
                raise ValueError(f"{contract.name}: empty {label}")
        disposition_overlap = (
            set(contract.failure_modes) & set(contract.indeterminate_modes))
        if disposition_overlap:
            raise ValueError(
                f"{contract.name}: modes cannot be both semantic absence and "
                "indeterminate: " + ", ".join(sorted(disposition_overlap)))
        directional_terms = [term for term, _ in contract.proxy_directions]
        if len(directional_terms) != len(set(directional_terms)):
            raise ValueError(f"{contract.name}: duplicate directional proxy term")
        if any(not term.strip() or direction not in {"low", "high"}
               for term, direction in contract.proxy_directions):
            raise ValueError(
                f"{contract.name}: directional proxies require a nonempty "
                "term and low/high direction")
        if any(term not in contract.proxy_for
               for term in directional_terms):
            raise ValueError(
                f"{contract.name}: directional proxy must also appear in "
                "proxy_for")
        self._legs[contract.name] = contract

    def get(self, name: str) -> LegContract:
        try:
            return self._legs[name]
        except KeyError as exc:
            raise KeyError(f"missing semantic leg {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._legs))

    def contracts(self) -> tuple[LegContract, ...]:
        return tuple(self._legs[name] for name in self.names())

    def terminal_types(self) -> tuple[str, ...]:
        return tuple(sorted({leg.codomain for leg in self._legs.values()}))


_RESULT_TYPES: dict[str, type] = {
    "Scene": Scene,
    "Object": ObjectMask,
    "Witness": Witness,
    "PairWitness": PairWitness,
    "PointWitness": PointWitness,
    "ContourWitness": ContourWitness,
    "SkeletonGraphWitness": SkeletonGraphWitness,
    "CurveWitness": CurveWitness,
    "LineSegmentWitness": LineSegmentWitness,
    "IncidentRayWitness": IncidentRayWitness,
    "ExteriorGapWitness": ExteriorGapWitness,
    "PointContactSignature": PointContactSignature,
    "AngleWitness": AngleWitness,
    "ArcWitness": ArcWitness,
    "CircleWitness": CircleWitness,
    "CirclePairWitness": CirclePairWitness,
    "PolygonWitness": PolygonWitness,
    "TriangleWitness": TriangleWitness,
    "QuadrilateralWitness": QuadrilateralWitness,
    "PartGraphWitness": PartGraphWitness,
    "PartWitness": PartWitness,
    "ContactWitness": ContactWitness,
    "IntersectionWitness": IntersectionWitness,
    "CircleIntersectionWitness": CircleIntersectionWitness,
    "RadialArrangementWitness": RadialArrangementWitness,
    "SymmetryWitness": SymmetryWitness,
    "PrototypeWitness": PrototypeWitness,
    "SoftResult": SoftResult,
    "SoftEvidenceSet": SoftEvidenceSet,
}


def result_type_for_codomain(codomain: str) -> type | None:
    """Runtime class backing a typed non-scalar registry codomain."""
    return _RESULT_TYPES.get(codomain)


def is_witness_codomain(codomain: str) -> bool:
    """Whether a registry codomain is backed by a :class:`Witness` class.

    Witness semantics are a runtime type relation, not a naming convention:
    valid witness types such as ``PointContactSignature`` need not end in the
    word ``Witness``.
    """
    result_type = result_type_for_codomain(codomain)
    return isinstance(result_type, type) and issubclass(result_type, Witness)


def is_pair_witness_codomain(codomain: str) -> bool:
    """Whether a codomain constructively certifies exactly two peers."""
    result_type = result_type_for_codomain(codomain)
    return isinstance(result_type, type) and issubclass(result_type, PairWitness)


def result_contract_issue(contract: LegContract, value: Any) -> str | None:
    """Return a diagnostic when an implementation violates its codomain.

    Static diagram typing is meaningful only if registered implementations
    actually return the values their contracts name.  This check lives beside
    the registry's type vocabulary and is run for every cone edge.
    """
    codomain = contract.codomain
    if codomain == "Measurement":
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            return f"expected finite Measurement, got {type(value).__name__}"
        if not math.isfinite(float(value)):
            return "expected finite Measurement, got non-finite value"
        numeric = float(value)
        if contract.measurement_kind == "count" and (
                numeric < 0.0 or not numeric.is_integer()):
            return f"expected nonnegative integer count, got {numeric!r}"
        if contract.measurement_kind == "binary" and numeric not in {0.0, 1.0}:
            return f"expected binary Measurement in {{0, 1}}, got {numeric!r}"
        return None
    if codomain in {"Panel", "BinaryPanel"}:
        if not isinstance(value, np.ndarray):
            return f"expected {codomain}, got {type(value).__name__}"
        if codomain == "BinaryPanel" and not np.all(
                (value == 0) | (value == 1)):
            return "BinaryPanel contains values outside {0, 1}"
        return None
    expected = _RESULT_TYPES.get(codomain)
    if expected is not None:
        if not isinstance(value, expected):
            return f"expected {codomain}, got {type(value).__name__}"
        return None
    # Extensible witness libraries may introduce a type not yet imported by
    # this module.  Exact runtime names still fail closed without a hard-coded
    # concept table.
    if type(value).__name__ != codomain:
        return f"expected {codomain}, got {type(value).__name__}"
    return None


# ---------------------------------------------------------------------------
# Pixel-graph helpers.  Panels are 1-px stroke drawings, so the mask itself
# is the curve; degree analysis on the 8-neighbourhood gives endpoints,
# junctions and cycles.
# ---------------------------------------------------------------------------

_NEIGHBOR_OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                          if (dy, dx) != (0, 0))


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Thin a stroke mask to a topology-preserving 1-px skeleton.

    Rasterized strokes are ~2 px thick with orientation-dependent staircase
    doublings, which make 8-neighbourhood degree analysis (endpoints,
    junctions, cycles) and contour ordering sensitive to rotation.  Thinning
    to a 1-px skeleton restores the true curve topology so those measurements
    become rotation-invariant.  scikit-image's default ``skeletonize`` is the
    primary path.  On sparse stroke masks only, a guarded Guo-Hall ``thin``
    fallback preserves a three-way diagonal junction when the primary path
    erases most of the stroke and loses that branch.  Filled objects and
    already-thin part masks never take that fallback.
    """
    binary = np.asarray(mask) > 0
    if not binary.any():
        return binary
    from skimage.morphology import skeletonize, thin
    primary = np.asarray(skeletonize(binary), dtype=bool)
    points = np.argwhere(binary)
    if not len(points):
        return primary
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0)
    bbox_area = max(1, int((y1 - y0 + 1) * (x1 - x0 + 1)))
    occupancy = float(np.count_nonzero(binary)) / bbox_area
    retention = float(np.count_nonzero(primary)) / max(
        1, int(np.count_nonzero(binary)))
    if occupancy < 0.35 and retention < 0.6:
        alternative = np.asarray(thin(binary), dtype=bool)
        primary_topology = _degrees(primary)
        alternative_topology = _degrees(alternative)
        if primary_topology[1] == 0 < alternative_topology[1] \
                and alternative_topology[0] >= 3:
            return alternative
    return primary


_THIN_CACHE: dict[tuple[tuple[int, ...], bytes], np.ndarray] = {}


def _thinned(mask: np.ndarray) -> np.ndarray:
    """Thinned copy of a stroke mask; solid/tiny blobs are left intact."""
    binary = np.ascontiguousarray(np.asarray(mask) > 0)
    # Raw bytes alone are ambiguous across shapes (for example 2x3 and 3x2
    # masks can have identical payloads).  Omitting shape made results depend
    # on which panel happened to populate the cache first.
    key = (binary.shape, binary.tobytes())
    cached = _THIN_CACHE.get(key)
    if cached is not None:
        return cached
    thin = _skeletonize(binary)
    # Never erase a real component: a filled region whose skeleton collapses
    # below the min-stroke size keeps its raw mask.
    if thin.sum() < 3 <= int(binary.sum()):
        thin = binary
    if len(_THIN_CACHE) > 1024:
        _THIN_CACHE.clear()
    _THIN_CACHE[key] = thin
    return thin


def _topo(obj: "ObjectMask") -> np.ndarray:
    return _thinned(obj.mask)


def _degree_map(mask: np.ndarray) -> dict[tuple[int, int], int]:
    coords = {(int(y), int(x)) for y, x in np.argwhere(mask)}
    return {
        p: sum((p[0] + dy, p[1] + dx) in coords for dy, dx in _NEIGHBOR_OFFSETS)
        for p in coords
    }


def _degrees(mask: np.ndarray) -> tuple[int, int, int]:
    """Return endpoint, junction, and first-Betti counts for a stroke mask.

    Pixelwise 8-neighbour degree is useful for locating candidates, but its
    raw Euler formula invents diagonal edges around an ordinary T junction
    and counts every pixel in a thick junction as a separate branch.  Cluster
    endpoint/junction pixels into semantic vertices and obtain the cycle count
    from digital topology instead.
    """
    deg = _degree_map(mask)
    endpoints = len(_cluster_points({p for p, d in deg.items() if d <= 1}))
    branches = len(_cluster_points({p for p, d in deg.items() if d >= 3}))
    if not deg:
        return endpoints, branches, 0
    from skimage.measure import euler_number, label
    binary = np.asarray(mask, dtype=bool)
    components = int(label(binary, connectivity=2).max())
    cycles = max(
        0, components - int(euler_number(binary, connectivity=2)))
    return endpoints, branches, cycles


def _cluster_points(points: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    clusters: list[set[tuple[int, int]]] = []
    remaining = set(points)
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        cluster = {seed}
        stack = [seed]
        while stack:
            y, x = stack.pop()
            for dy, dx in _NEIGHBOR_OFFSETS:
                p = (y + dy, x + dx)
                if p in remaining:
                    remaining.discard(p)
                    cluster.add(p)
                    stack.append(p)
        clusters.append(cluster)
    return clusters


_JUMP_OFFSETS = tuple((dy, dx) for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)
                      if (dy, dx) != (0, 0) and max(abs(dy), abs(dx)) == 2)


def _walk_order(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Order stroke pixels along the curve (endpoint-first, straightest-next).

    Rasterized strokes contain 2-px staircase doublings; when the walk gets
    locally stuck it may hop to an unvisited pixel at Chebyshev distance 2,
    which keeps the ordering monotone along the curve without any thinning.
    """
    if not coords:
        return []
    deg = {
        p: sum((p[0] + dy, p[1] + dx) in coords for dy, dx in _NEIGHBOR_OFFSETS)
        for p in coords
    }
    endpoints = sorted(p for p, d in deg.items() if d <= 1)
    current = endpoints[0] if endpoints else min(coords)
    visited = {current}
    path = [current]
    prev_dir: tuple[float, float] | None = None
    while True:
        options = [
            (current[0] + dy, current[1] + dx)
            for dy, dx in _NEIGHBOR_OFFSETS
            if (current[0] + dy, current[1] + dx) in coords
            and (current[0] + dy, current[1] + dx) not in visited
        ]
        if not options:
            options = [
                (current[0] + dy, current[1] + dx)
                for dy, dx in _JUMP_OFFSETS
                if (current[0] + dy, current[1] + dx) in coords
                and (current[0] + dy, current[1] + dx) not in visited
            ]
        if not options:
            break
        if prev_dir is None:
            nxt = options[0]
        else:
            def straightness(p: tuple[int, int]) -> float:
                vy, vx = p[0] - current[0], p[1] - current[1]
                norm = math.hypot(vy, vx) or 1.0
                return (vy * prev_dir[0] + vx * prev_dir[1]) / norm
            nxt = max(options, key=straightness)
        vy, vx = nxt[0] - current[0], nxt[1] - current[1]
        norm = math.hypot(vy, vx) or 1.0
        prev_dir = (vy / norm, vx / norm)
        visited.add(nxt)
        path.append(nxt)
        current = nxt
    return path


def _mask_from_points(points: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if points:
        ys, xs = zip(*points)
        mask[list(ys), list(xs)] = True
    return mask


def _component_masks(panel: np.ndarray, min_pixels: int = 3) -> tuple[ObjectMask, ...]:
    ink = np.asarray(panel, dtype=np.uint8) > 0
    coords = {(int(y), int(x)) for y, x in np.argwhere(ink)}
    objects: list[ObjectMask] = []
    for cluster in _cluster_points(coords):
        if len(cluster) >= min_pixels:
            objects.append(ObjectMask(_mask_from_points(cluster, ink.shape), "tmp"))
    objects.sort(key=lambda o: int(o.mask.sum()), reverse=True)
    return tuple(ObjectMask(o.mask, f"object_{i}") for i, o in enumerate(objects))


def binarize_panel(panel: np.ndarray) -> np.ndarray:
    return (np.asarray(panel) > 0).astype(np.uint8)


def parse_scene(panel: np.ndarray) -> Scene:
    return Scene(np.asarray(panel, dtype=np.uint8), _component_masks(panel))


def extract_connected_components(panel: np.ndarray) -> Scene:
    return parse_scene(panel)


def object_count(scene: Scene) -> float:
    return float(len(scene.objects))


def total_ink(panel: np.ndarray) -> float:
    return float(np.count_nonzero(np.asarray(panel) > 0))


def select_all_objects(scene: Scene) -> Scene:
    return scene


def select_largest(scene: Scene) -> ObjectMask:
    if not scene.objects:
        raise WitnessAbsent("no_objects", "scene contains no selectable objects")
    return max(scene.objects, key=lambda obj: int(np.count_nonzero(obj.mask)))


def select_largest_object(scene: Scene) -> ObjectMask:
    return select_largest(scene)


def select_smallest_object(scene: Scene) -> ObjectMask:
    if not scene.objects:
        raise WitnessAbsent("no_objects", "scene contains no selectable objects")
    return min(scene.objects, key=lambda obj: int(np.count_nonzero(obj.mask)))


def select_principal_objects(scene: Scene) -> Scene:
    ordered = tuple(sorted(
        scene.objects,
        key=lambda obj: (-int(np.count_nonzero(obj.mask)), obj.object_id),
    ))
    return Scene(scene.panel, ordered[: max(1, min(4, len(ordered)))])


def select_inner_object(scene: Scene) -> ObjectMask:
    return select_smallest_object(scene)


def select_outer_object(scene: Scene) -> ObjectMask:
    return select_largest(scene)


def select_parts(graph: PartGraphWitness) -> PartGraphWitness:
    return graph


def select_largest_part(graph: PartGraphWitness) -> PartWitness:
    """Select a concrete part already carried by a part graph.

    This is the generic target-to-source projection used by executable
    gluing declarations; it never fabricates a part when the graph is empty.
    """
    if not graph.parts:
        raise WitnessAbsent("no_parts", "part graph contains no parts")
    return max(
        graph.parts,
        key=lambda part: len(part.contour.points) if part.contour else 0,
    )


def largest_area(scene: Scene) -> float:
    return float(scene.objects[0].mask.sum()) if scene.objects else 0.0


def largest_ink(scene: Scene) -> float:
    """Foreground occupancy of the largest connected component.

    This intentionally does not call the quantity geometric area: on a stroke
    drawing, foreground-pixel count tracks stroke length/width and says
    nothing about the region enclosed by a contour.
    """
    return float(max(
        (np.count_nonzero(obj.mask) for obj in scene.objects), default=0))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    y0, x0 = pts.min(axis=0)
    y1, x1 = pts.max(axis=0)
    return int(y0), int(x0), int(y1), int(x1)


def bbox_aspect(obj: ObjectMask) -> float:
    box = _bbox(obj.mask)
    if box is None:
        return 0.0
    y0, x0, y1, x1 = box
    h = max(1, y1 - y0 + 1)
    w = max(1, x1 - x0 + 1)
    return float(max(w / h, h / w))


def bbox_fill(obj: ObjectMask) -> float:
    box = _bbox(obj.mask)
    if box is None:
        return 0.0
    y0, x0, y1, x1 = box
    area = max(1, (y1 - y0 + 1) * (x1 - x0 + 1))
    return float(obj.mask.sum() / area)


def bbox_occupancy(obj: ObjectMask) -> float:
    """Foreground-pixel density inside an object's bounding box.

    The public name deliberately describes a metric, not the categorical
    claim that a region is filled.  A learned relative threshold can compare
    occupancy values; it cannot establish an absolute filled/sparse class.
    """
    return bbox_fill(obj)


def _covariance_eigs(mask: np.ndarray) -> tuple[float, float]:
    pts = np.argwhere(mask).astype(float)
    if len(pts) < 2:
        return 0.0, 0.0
    cov = np.cov(pts.T)
    eigs = np.linalg.eigvalsh(cov)
    return float(max(eigs, default=0.0)), float(min(eigs, default=0.0))


def elongation(obj: ObjectMask) -> float:
    """Rotation-invariant elongation: sqrt(major/minor) of the point cloud.

    The axis-aligned bounding box makes ``bbox_aspect`` swing under rotation
    (a diagonal bar looks square).  The ratio of the covariance eigenvalues
    is orientation-free, so a thin/elongated shape scores high at any angle.
    """
    major, minor = _covariance_eigs(obj.mask)
    if major <= 0:
        return 1.0
    return float(math.sqrt(major / max(minor, 1e-6)))


# ---------------------------------------------------------------------------
# Contours and curve geometry.
# ---------------------------------------------------------------------------

def _has_long_terminal_branch(mask: np.ndarray, max_spur_length: int = 3) -> bool:
    """Whether a skeleton has a substantive endpoint-to-junction branch."""
    degrees = _degree_map(mask)
    for endpoint, degree in degrees.items():
        if degree > 1:
            continue
        previous = None
        current = endpoint
        length = 1
        while True:
            neighbours = [
                (current[0] + dy, current[1] + dx)
                for dy, dx in _NEIGHBOR_OFFSETS
                if (current[0] + dy, current[1] + dx) in degrees
                and (current[0] + dy, current[1] + dx) != previous
            ]
            if len(neighbours) != 1:
                break
            previous, current = current, neighbours[0]
            if degrees[current] != 2:
                if degrees[current] >= 3 and length > max_spur_length:
                    return True
                break
            length += 1
    return False


def _regularized_closed_contour(
        obj: ObjectMask, skeleton: np.ndarray) -> ContourWitness | None:
    """Recover a simple closed stroke when thinning leaves tiny corner spurs.

    Bilinear off-grid rotation can create a two-pixel spur or one-pixel hole
    at an acute corner.  Those artifacts made a closed triangle look branched
    even though its raster still has exactly one macroscopic interior.  The
    fallback is deliberately narrow: real endpoint-to-junction branches are
    rejected, tiny holes are filled, and the regularized mask must have one
    and only one hole.  Figure-eights and lollipops therefore remain outside
    the simple-contour codomain.
    """
    if _has_long_terminal_branch(skeleton):
        return None
    from scipy import ndimage
    from skimage.measure import euler_number, find_contours, label

    regularized = ndimage.binary_dilation(
        np.asarray(obj.mask, dtype=bool), iterations=1)
    # Foreground uses 8-connectivity, so use the complementary 4-connected
    # background when identifying bounded holes.  Fill only sampling-scale
    # holes; the macroscopic interior must survive.
    background_labels, count = ndimage.label(~regularized)
    border_labels = set(np.unique(np.concatenate((
        background_labels[0], background_labels[-1],
        background_labels[:, 0], background_labels[:, -1],
    ))))
    for component_id in range(1, count + 1):
        if component_id in border_labels:
            continue
        component = background_labels == component_id
        if int(np.count_nonzero(component)) <= 16:
            regularized[component] = True
    components = int(label(regularized, connectivity=2).max())
    cycles = max(
        0, components - int(euler_number(regularized, connectivity=2)))
    if components != 1 or cycles != 1:
        return None
    boundaries = find_contours(
        np.pad(regularized.astype(float), 1), level=0.5)
    if not boundaries:
        return None
    # With exactly one hole, find_contours yields the outer stroke boundary
    # and the shorter inner boundary.  The inner offset retains acute corners;
    # the outer offset is rounded by dilation and can turn a correct triangle
    # into a high-residual polygon at large stroke widths.
    boundary = min(boundaries, key=len) - 1.0
    if len(boundary) < 5:
        return None
    # find_contours repeats the first point at the end for a closed boundary.
    if np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]
    return ContourWitness(
        source_component_id=obj.object_id,
        points=tuple((float(x), float(y)) for y, x in boundary),
        is_closed=True,
        confidence=0.9,
        provenance=("extract_contours", "regularized_closed_boundary"),
    )

def extract_contours(obj: ObjectMask) -> ContourWitness:
    topo = _topo(obj)
    coords = {(int(y), int(x)) for y, x in np.argwhere(topo)}
    if not coords:
        raise WitnessAbsent(
            "empty_component", "cannot extract a contour from an empty object")
    endpoint_count_value, branch_count_value, cycle_count_value = _degrees(topo)
    if branch_count_value or endpoint_count_value not in {0, 2}:
        recovered = _regularized_closed_contour(obj, topo)
        if recovered is not None:
            return recovered
        raise WitnessAbsent(
            "not_simple_curve",
            "a single contour requires an unbranched open path or closed loop")
    if (endpoint_count_value == 0 and cycle_count_value != 1) \
            or (endpoint_count_value == 2 and cycle_count_value != 0):
        recovered = _regularized_closed_contour(obj, topo)
        if recovered is not None:
            return recovered
        raise WitnessAbsent(
            "not_simple_curve",
            "stroke topology is inconsistent with one simple contour")
    deg = _degree_map(topo)
    is_closed = endpoint_count_value == 0
    path = _walk_order(coords)
    covered = len(path) / len(coords)
    # Completeness is topological, not a raw coverage fraction: doubled
    # raster pixels are legitimately skipped by the walk.
    complete = False
    if len(path) >= 2 and covered >= 0.6:
        start, end = path[0], path[-1]
        if is_closed:
            complete = max(abs(start[0] - end[0]), abs(start[1] - end[1])) <= 2
        else:
            complete = deg[start] <= 1 and deg[end] <= 1 and start != end
    if complete:
        ordered = tuple((float(x), float(y)) for y, x in path)
        confidence = 1.0
    elif covered >= 0.8:
        ordered = tuple((float(x), float(y)) for y, x in path)
        confidence = covered
    else:
        pts = np.asarray([(x, y) for y, x in sorted(coords)], dtype=float)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        ordered = tuple((float(x), float(y)) for x, y in pts[np.argsort(angles)])
        confidence = 0.5
    return ContourWitness(
        source_component_id=obj.object_id,
        points=ordered,
        is_closed=is_closed,
        confidence=confidence,
        provenance=("extract_contours",),
    )


def contour_closedness(contour: ContourWitness) -> float:
    return 1.0 if contour.is_closed else 0.0


def build_skeleton_graph(obj: ObjectMask) -> SkeletonGraphWitness:
    topo = _topo(obj)
    endpoints, branches, cycles = _degrees(topo)
    coords = tuple(sorted((int(y), int(x)) for y, x in np.argwhere(topo)))
    node_index = {coord: index for index, coord in enumerate(coords)}
    nodes = tuple(
        PointWitness(x=float(x), y=float(y), source_id=obj.object_id)
        for y, x in coords
    )
    edges = tuple(sorted(
        (index, node_index[neighbor])
        for (y, x), index in node_index.items()
        for dy, dx in _NEIGHBOR_OFFSETS
        if (neighbor := (y + dy, x + dx)) in node_index
        and index < node_index[neighbor]
    ))
    return SkeletonGraphWitness(
        source_component_id=obj.object_id,
        nodes=nodes,
        edges=edges,
        endpoint_count=endpoints,
        branch_count=branches,
        cycle_count=cycles,
        provenance=("build_skeleton_graph",),
    )


def skeletonize_component(obj: ObjectMask) -> SkeletonGraphWitness:
    return build_skeleton_graph(obj)


def endpoint_count(graph: SkeletonGraphWitness) -> float:
    return float(graph.endpoint_count)


def branch_count(graph: SkeletonGraphWitness) -> float:
    return float(graph.branch_count)


def cycle_count(graph: SkeletonGraphWitness) -> float:
    return float(graph.cycle_count)


def closure_ratio(obj: ObjectMask) -> float:
    topo = _topo(obj)
    pts = np.argwhere(topo)
    if len(pts) == 0:
        return 1.0
    endpoints, _, _ = _degrees(topo)
    return float(endpoints / max(1, len(pts)))


def estimate_tangents(contour: ContourWitness) -> CurveWitness:
    return CurveWitness(
        source_component_id=contour.source_component_id,
        points=contour.points,
        confidence=contour.confidence,
        provenance=contour.provenance + ("estimate_tangents",),
    )


def estimate_curvature(contour: ContourWitness) -> CurveWitness:
    return estimate_tangents(contour)


def curvature_extrema(curve: CurveWitness) -> SkeletonGraphWitness:
    pts = tuple(PointWitness(x=x, y=y, source_id=curve.source_component_id)
                for x, y in curve.points[:: max(1, len(curve.points) // 8)])
    return SkeletonGraphWitness(
        source_component_id=curve.source_component_id,
        nodes=pts,
        endpoint_count=len(curve.endpoints),
        branch_count=0,
        cycle_count=0,
        provenance=curve.provenance + ("curvature_extrema",),
    )


# ---------------------------------------------------------------------------
# Polygon fitting via Ramer-Douglas-Peucker on the path-ordered contour.
# ---------------------------------------------------------------------------

def _rdp_indices(pts: np.ndarray, lo: int, hi: int, eps: float,
                 keep: set[int]) -> None:
    if hi <= lo + 1:
        return
    a, b = pts[lo], pts[hi]
    chord = b - a
    norm = float(np.hypot(*chord)) or 1.0
    rel = pts[lo + 1:hi] - a
    dist = np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / norm
    k = int(np.argmax(dist))
    if float(dist[k]) > eps:
        mid = lo + 1 + k
        keep.add(mid)
        _rdp_indices(pts, lo, mid, eps, keep)
        _rdp_indices(pts, mid, hi, eps, keep)


def _simplify_polyline(pts: np.ndarray, eps: float) -> list[int]:
    keep: set[int] = {0, len(pts) - 1}
    _rdp_indices(pts, 0, len(pts) - 1, eps, keep)
    return sorted(keep)


def _turn_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = b - a, c - b
    n1, n2 = float(np.hypot(*v1)), float(np.hypot(*v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def _resample_contour(pts: np.ndarray, n: int, closed: bool) -> np.ndarray:
    """Uniform arc-length resampling (rotation- and density-invariant)."""
    ring = np.vstack([pts, pts[:1]]) if closed else pts
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(arc[-1])
    if total <= np.finfo(float).tiny:
        return pts[:1]
    u = np.linspace(0.0, total, n, endpoint=not closed)
    xs = np.interp(u, arc, ring[:, 0])
    ys = np.interp(u, arc, ring[:, 1])
    return np.stack([xs, ys], axis=1)


def _turning_profile(rs: np.ndarray, closed: bool, k: int) -> np.ndarray:
    """Absolute turning angle (radians) at each sample over an arc window k."""
    n = len(rs)
    ang = np.zeros(n)
    indices = range(n) if closed else range(k, n - k)
    for i in indices:
        a, b, c = rs[(i - k) % n], rs[i], rs[(i + k) % n]
        v1, v2 = b - a, c - b
        n1, n2 = float(np.hypot(*v1)), float(np.hypot(*v2))
        if n1 <= np.finfo(float).tiny or n2 <= np.finfo(float).tiny:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = float(v1 @ v2)
        ang[i] = abs(math.atan2(cross, dot))
    return ang


def _circular_distance(a: int, b: int, n: int) -> int:
    d = abs(a - b)
    return min(d, n - d)


def _polygon_vertices(contour: ContourWitness) -> tuple[tuple[PointWitness, ...], float]:
    """Detect corners as turning-angle peaks on an arc-length-resampled curve.

    Turning angle and arc length are both rotation-invariant, so the corner
    set (and hence the side count) no longer depends on panel orientation or
    raster density.  A smooth curve (circle/arc) has turning spread evenly
    below threshold and yields no corners; a polygon concentrates turning at
    its vertices.
    """
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 4:
        return (), 1.0
    closed = contour.is_closed
    extent = max(
        float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-12)
    # The semantic fit must not change merely because every coordinate is
    # expressed in a different length unit.  Deriving the resampling density
    # from perimeter made the normalized residual (and therefore confidence)
    # drift under exact uniform scaling.  A fixed arc-length grid keeps the
    # geometric approximation independent of absolute scale while remaining
    # dense enough for the small Bongard contours.
    n = 128
    rs = _resample_contour(pts, n, closed)
    if len(rs) < 4:
        return (), 1.0
    n = len(rs)
    k = max(2, n // 16)
    window = max(2, n // 12)
    threshold = math.radians(33.0)
    ang = _turning_profile(rs, closed, k)

    accepted: list[int] = []
    for i in np.argsort(ang)[::-1]:
        if ang[i] < threshold:
            break
        i = int(i)
        if closed:
            if all(_circular_distance(i, j, n) > window for j in accepted):
                accepted.append(i)
        elif all(abs(i - j) > window for j in accepted):
            accepted.append(i)
    accepted.sort()

    corner_pts = [rs[i] for i in accepted]
    if closed:
        verts = corner_pts
        side_verts = corner_pts
    else:
        verts = [rs[0]] + corner_pts + [rs[-1]]
        side_verts = verts
    residual = _polygon_residual(pts, side_verts, closed) / extent if len(side_verts) >= 2 \
        else 1.0
    vertices = tuple(
        PointWitness(x=float(v[0]), y=float(v[1]),
                     source_id=contour.source_component_id)
        for v in verts
    )
    return vertices, residual


def _polygon_residual(pts: np.ndarray, verts: list[np.ndarray],
                      closed: bool) -> float:
    if len(verts) < 2:
        return float("inf")
    edges = list(zip(verts, verts[1:]))
    if closed:
        edges.append((verts[-1], verts[0]))
    dists = np.full(len(pts), np.inf)
    for a, b in edges:
        chord = b - a
        length2 = float(chord @ chord) or 1.0
        t = np.clip(((pts - a) @ chord) / length2, 0.0, 1.0)
        proj = a + t[:, None] * chord
        dists = np.minimum(dists, np.linalg.norm(pts - proj, axis=1))
    return float(np.mean(dists))


MAX_POLYGON_RESIDUAL = 0.025


def fit_polygon(contour: ContourWitness) -> PolygonWitness:
    if not contour.is_closed:
        raise WitnessAbsent(
            "open_contour", "a polygon witness requires a closed contour")
    vertices, residual = _polygon_vertices(contour)
    side_count = len(vertices) if contour.is_closed else max(0, len(vertices) - 1)
    if side_count < 3:
        raise WitnessAbsent(
            "too_few_sides",
            f"polygon fit found {side_count} sides; at least three are required")
    if not math.isfinite(residual) or residual > MAX_POLYGON_RESIDUAL:
        raise WitnessAbsent(
            "high_residual",
            f"polygon fit residual {residual:.4g} exceeds "
            f"{MAX_POLYGON_RESIDUAL:.4g}")
    confidence = max(0.0, 1.0 - 10.0 * residual) if len(vertices) >= 2 else 0.1
    return PolygonWitness(
        source_component_id=contour.source_component_id,
        vertices=vertices,
        side_count=side_count,
        residual=residual,
        confidence=confidence,
        provenance=contour.provenance + ("fit_polygon",),
    )


def detect_corners(contour: ContourWitness) -> PolygonWitness:
    return fit_polygon(contour)


def decompose_into_line_segments(contour: ContourWitness) -> PolygonWitness:
    return fit_polygon(contour)


def polygon_side_count(poly: PolygonWitness) -> float:
    return float(poly.side_count)


def classify_triangle(poly: PolygonWitness) -> TriangleWitness:
    if poly.side_count != 3:
        raise WitnessAbsent(
            "wrong_side_count",
            f"polygon has {poly.side_count} sides, not a triangle")
    return TriangleWitness(
        source_component_id=poly.source_component_id,
        vertices=poly.vertices,
        confidence=poly.confidence,
        residual=poly.residual,
        provenance=poly.provenance + ("classify_triangle",),
    )


def classify_quadrilateral(poly: PolygonWitness) -> QuadrilateralWitness:
    if poly.side_count != 4:
        raise WitnessAbsent(
            "wrong_side_count",
            f"polygon has {poly.side_count} sides, not a quadrilateral")
    return QuadrilateralWitness(
        source_component_id=poly.source_component_id,
        vertices=poly.vertices,
        confidence=poly.confidence,
        residual=poly.residual,
        provenance=poly.provenance + ("classify_quadrilateral",),
    )


MAX_LINE_RESIDUAL = 0.04


def fit_line_segment(contour: ContourWitness) -> LineSegmentWitness:
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 2:
        raise WitnessAbsent(
            "not_enough_points", "need at least two points for a line segment")
    if contour.is_closed:
        raise WitnessAbsent(
            "closed_contour", "a line segment requires an open contour")
    if not np.isfinite(pts).all():
        raise ValueError("line contour contains non-finite coordinates")
    center = pts.mean(axis=0)
    centered = pts - center
    covariance = centered.T @ centered / len(pts)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    along = centered @ direction
    a = center + float(along.min()) * direction
    b = center + float(along.max()) * direction
    length = float(along.max() - along.min())
    if length <= np.finfo(float).tiny:
        raise WitnessAbsent(
            "degenerate_segment", "line support has zero extent")
    orthogonal = centered - along[:, None] * direction
    orthogonal_distance = np.linalg.norm(orthogonal, axis=1)
    rms_residual = float(np.sqrt(np.mean(orthogonal_distance ** 2)) / length)
    max_residual = float(np.max(orthogonal_distance) / length)
    # A global RMS alone lets one conspicuous localized bend hide inside a
    # long otherwise-straight path.  A line witness must satisfy both the
    # average and worst-supported deviation; storing the stricter residual
    # also keeps downstream scoring honest.
    residual = max(rms_residual, max_residual)
    if residual > MAX_LINE_RESIDUAL:
        raise WitnessAbsent(
            "high_residual",
            f"line fit residual {residual:.4g} exceeds {MAX_LINE_RESIDUAL:.4g}")
    start = PointWitness(
        x=float(a[0]), y=float(a[1]),
        source_id=contour.source_component_id)
    end = PointWitness(
        x=float(b[0]), y=float(b[1]),
        source_id=contour.source_component_id)
    return LineSegmentWitness(
        source_component_id=contour.source_component_id,
        points=contour.points,
        endpoints=(start, end),
        start=start,
        end=end,
        length=length,
        residual=residual,
        confidence=max(0.0, 1.0 - residual),
        provenance=contour.provenance + ("fit_line_segment",),
    )


def angle_between_segments(
        first: LineSegmentWitness,
        second: LineSegmentWitness) -> AngleWitness:
    """Construct an intrinsic unsigned angle from two meeting segments.

    The closest endpoint pair must meet within a scale-relative tolerance;
    unrelated lines do not fabricate an angle.  Fit residuals become explicit
    angular uncertainty carried by the witness.
    """
    endpoints_a = (first.start, first.end)
    endpoints_b = (second.start, second.end)
    coordinates_a = tuple(
        np.asarray((point.x, point.y), dtype=float) for point in endpoints_a)
    coordinates_b = tuple(
        np.asarray((point.x, point.y), dtype=float) for point in endpoints_b)
    if not all(np.isfinite(point).all()
               for point in coordinates_a + coordinates_b):
        raise ValueError("segment endpoint coordinates must be finite")
    lengths = []
    for coordinates in (coordinates_a, coordinates_b):
        lengths.append(float(np.linalg.norm(coordinates[1] - coordinates[0])))
    if any(not math.isfinite(length) or length <= np.finfo(float).tiny
           for length in lengths):
        raise WitnessAbsent(
            "degenerate_segment", "angle support contains a zero-length segment")
    candidates = [
        (float(np.linalg.norm(left - right)), index_a, index_b)
        for index_a, left in enumerate(coordinates_a)
        for index_b, right in enumerate(coordinates_b)
    ]
    gap, index_a, index_b = min(candidates)
    # This relative gate is invariant to translation and uniform scale.  It
    # tolerates only the small endpoint mismatch introduced by independent
    # line fits, not arbitrary near-passing segments.
    maximum_gap = 0.08 * min(lengths)
    if gap > maximum_gap:
        raise WitnessAbsent(
            "segments_do_not_meet",
            f"closest segment endpoints differ by {gap:.4g}, above "
            f"scale-relative tolerance {maximum_gap:.4g}")
    vertex_xy = (coordinates_a[index_a] + coordinates_b[index_b]) / 2.0
    ray_a = coordinates_a[1 - index_a] - vertex_xy
    ray_b = coordinates_b[1 - index_b] - vertex_xy
    norms = (float(np.linalg.norm(ray_a)), float(np.linalg.norm(ray_b)))
    if any(norm <= np.finfo(float).tiny for norm in norms):
        raise WitnessAbsent(
            "degenerate_segment", "angle ray has zero extent")
    cosine = float(np.dot(ray_a, ray_b) / (norms[0] * norms[1]))
    degrees = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
    residuals = []
    for name, segment in (("first", first), ("second", second)):
        residual = float(segment.residual)
        if not math.isfinite(residual) or residual < 0.0:
            raise ValueError(f"{name} segment residual must be finite and nonnegative")
        residuals.append(residual)
    uncertainty = math.degrees(
        math.atan(residuals[0]) + math.atan(residuals[1]))
    confidence = max(0.0, 1.0 - uncertainty / 45.0)
    source_a = first.source_component_id or first.start.source_id or "segment-a"
    source_b = second.source_component_id or second.start.source_id or "segment-b"
    return AngleWitness(
        source_a=source_a,
        source_b=source_b,
        vertex=PointWitness(
            x=float(vertex_xy[0]), y=float(vertex_xy[1]),
            source_id=f"{source_a}+{source_b}"),
        degrees=degrees,
        uncertainty_degrees=uncertainty,
        reference_frame="interior",
        residual=min(1.0, uncertainty / 180.0),
        confidence=confidence,
        provenance=first.provenance + second.provenance
        + ("angle_between_segments",),
    )


def angle_degrees(angle: AngleWitness) -> float:
    """Expose the fitted unsigned angle without changing its semantics.

    The uncertainty remains carried by ``AngleWitness`` and is therefore
    available in the execution trace.  This leg is deliberately distinct
    from ``angle_noncardinality_degrees``: angles equally far from a cardinal
    direction (for example 30 and 60 degrees) must not be collapsed when the
    literal claim concerns angle magnitude.
    """
    degrees = float(angle.degrees)
    uncertainty = float(angle.uncertainty_degrees)
    if not math.isfinite(degrees) or not 0.0 <= degrees <= 180.0:
        raise ValueError("angle degrees must be finite and in [0, 180]")
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise ValueError("angle uncertainty must be finite and nonnegative")
    return degrees


def angle_noncardinality_degrees(angle: AngleWitness) -> float:
    """Conservative distance from the cardinal set {0, 90, 180}."""
    degrees = float(angle.degrees)
    uncertainty = float(angle.uncertainty_degrees)
    if not math.isfinite(degrees) or not 0.0 <= degrees <= 180.0:
        raise ValueError("angle degrees must be finite and in [0, 180]")
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise ValueError("angle uncertainty must be finite and nonnegative")
    distance = min(abs(degrees - cardinal) for cardinal in (0.0, 90.0, 180.0))
    # Uncertainty cannot make the evidence stronger.  The lower end of the
    # admissible distance interval is the honest membership input.
    return max(0.0, distance - uncertainty)


def angle_obliqueness_evidence(angle: AngleWitness) -> SoftResult:
    raw = angle_noncardinality_degrees(angle)
    return OBLIQUENESS_CALIBRATOR.apply(
        raw,
        "angle-obliqueness",
        provenance=angle.provenance + ("angle_noncardinality_degrees",),
        components=(("fit-certainty", angle.confidence),),
    )


def angle_obliqueness_membership(angle: AngleWitness) -> float:
    """Bounded [0,1] analytic obliqueness membership."""
    result = angle_obliqueness_evidence(angle)
    if isinstance(result, SoftEvidence):
        return result.membership
    if isinstance(result, SoftAbsent):
        raise WitnessAbsent("soft_evidence_absent", result.detail)
    if isinstance(result, SoftError):
        raise ValueError(f"{result.error_code}: {result.detail}")
    raise TypeError("angle obliqueness returned an unknown soft-result type")


def soft_membership_value(result: SoftResult) -> float:
    """Expose present evidence to numeric selection without sentinels."""
    if isinstance(result, SoftEvidence):
        return result.membership
    if isinstance(result, SoftAbsent):
        raise WitnessAbsent("soft_evidence_absent", result.detail)
    if isinstance(result, SoftError):
        raise ValueError(f"{result.error_code}: {result.detail}")
    raise TypeError("unknown soft-result subtype")


def _taubin_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Taubin algebraic circle fit (Chernov).

    Kåsa's fit is heavily biased for small/partial arcs (Chernov, "Circular
    and Linear Regression"); Taubin removes most of that essential bias while
    staying a closed-form algebraic fit, so arc residuals are meaningful for
    the partial arcs common in these panels.
    """
    pts = np.asarray(pts, dtype=float)
    if not np.isfinite(pts).all():
        raise ValueError("circle support contains non-finite coordinates")
    origin = pts.mean(axis=0)
    normalizer = float(np.sqrt(np.mean(np.sum((pts - origin) ** 2, axis=1))))
    if not math.isfinite(normalizer) or normalizer <= np.finfo(float).tiny:
        raise ValueError("degenerate circle fit")
    # Solve in dimensionless coordinates.  The previous absolute determinant
    # cutoff rejected a perfectly valid small circle while accepting its 2x
    # rescaling, contradicting the advertised scale equivariance.
    normalized = (pts - origin) / normalizer
    x, y = normalized[:, 0], normalized[:, 1]
    mx, my = float(x.mean()), float(y.mean())
    u, v = x - mx, y - my
    z = u * u + v * v
    mz = float(z.mean())
    mxx, myy, mxy = float((u * u).mean()), float((v * v).mean()), float((u * v).mean())
    mxz, myz, mzz = float((u * z).mean()), float((v * z).mean()), float((z * z).mean())
    cov_xy = mxx * myy - mxy * mxy
    var_z = mzz - mz * mz
    a3 = 4.0 * mz
    a2 = -3.0 * mz * mz - mzz
    a1 = var_z * mz + 4.0 * cov_xy * mz - mxz * mxz - myz * myz
    a0 = mxz * (mxz * myy - myz * mxy) + myz * (myz * mxx - mxz * mxy) - var_z * cov_xy
    a22, a33 = a2 + a2, a3 + a3 + a3
    xnew, ynew = 0.0, a0
    for _ in range(99):
        dy = a1 + xnew * (a22 + a33 * xnew)
        if dy == 0.0:
            break
        step = ynew / dy
        cand = xnew - step
        if cand == xnew or not math.isfinite(cand):
            break
        yval = a0 + cand * (a1 + cand * (a2 + cand * a3))
        if abs(yval) >= abs(ynew):
            break
        xnew, ynew = cand, yval
    det = xnew * xnew - xnew * mz + cov_xy
    if abs(det) < 1e-12:
        raise ValueError("degenerate circle fit")
    xc = (mxz * (myy - xnew) - myz * mxy) / det / 2.0
    yc = (myz * (mxx - xnew) - mxz * mxy) / det / 2.0
    radius_sq = xc * xc + yc * yc + mz
    if not math.isfinite(radius_sq) or radius_sq <= 0:
        raise ValueError("degenerate circle fit")
    center = (np.array([xc + mx, yc + my]) * normalizer) + origin
    return center, math.sqrt(radius_sq) * normalizer


def _fit_circle_raw(contour: ContourWitness) -> CircleWitness:
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 3:
        raise ValueError("not enough contour points for circle fit")
    center, radius = _taubin_circle(pts)
    radii = np.linalg.norm(pts - center, axis=1)
    residual = float(np.sqrt(np.mean((radii - radius) ** 2)) / radius)
    return CircleWitness(
        source_component_id=contour.source_component_id,
        center=PointWitness(x=float(center[0]), y=float(center[1]),
                            source_id=contour.source_component_id),
        radius=radius,
        support_points=tuple((float(x), float(y)) for x, y in pts[:: max(1, len(pts) // 32)]),
        residual=residual,
        confidence=max(0.0, 1.0 - residual),
        provenance=contour.provenance + ("fit_circle",),
    )


MAX_CIRCLE_RESIDUAL = 0.04
MAX_ARC_RESIDUAL = 0.08
MAX_CIRCLE_RADIAL_Q95 = 0.04
MAX_CIRCLE_RADIAL_MAX = 0.10
MAX_ARC_RADIAL_Q95 = 0.08
MAX_ARC_RADIAL_MAX = 0.18


def _radial_deviation_tail(
        contour: ContourWitness,
        circle: CircleWitness,
        ) -> tuple[float, float]:
    """Scale-free upper-tail and maximum radial deviations from a fit.

    RMS residual is intentionally retained as the ordinary fit score, but it
    can dilute a short, deep dent among hundreds of otherwise perfect support
    points.  These local statistics close that admission hole without making
    the contract depend on absolute pixel size.
    """
    pts = np.asarray(contour.points, dtype=float)
    center = np.array([circle.center.x, circle.center.y], dtype=float)
    deviations = np.abs(np.linalg.norm(pts - center, axis=1) - circle.radius)
    normalized = deviations / circle.radius
    return float(np.quantile(normalized, 0.95)), float(np.max(normalized))


def fit_circle(contour: ContourWitness) -> CircleWitness:
    if len(contour.points) < 3:
        raise WitnessAbsent(
            "not_enough_points", "not enough contour points for a circle")
    if not contour.is_closed:
        raise WitnessAbsent(
            "open_contour", "contour is open; a circle is a closed curve")
    circle = _fit_circle_raw(contour)
    radial_q95, radial_max = _radial_deviation_tail(contour, circle)
    if circle.residual > MAX_CIRCLE_RESIDUAL \
            or radial_q95 > MAX_CIRCLE_RADIAL_Q95 \
            or radial_max > MAX_CIRCLE_RADIAL_MAX:
        raise WitnessAbsent(
            "high_residual",
            "circle radial residuals exceed the admissible envelope: "
            f"rms={circle.residual:.4g}/{MAX_CIRCLE_RESIDUAL:.4g}, "
            f"q95={radial_q95:.4g}/{MAX_CIRCLE_RADIAL_Q95:.4g}, "
            f"max={radial_max:.4g}/{MAX_CIRCLE_RADIAL_MAX:.4g}")
    return circle


def fit_arc(contour: ContourWitness) -> ArcWitness:
    if len(contour.points) < 8:
        raise WitnessAbsent(
            "not_enough_points",
            "need at least eight ordered support points for an arc")
    if contour.is_closed:
        raise WitnessAbsent(
            "closed_contour", "a closed circle is not an open arc witness")
    pts = np.asarray(contour.points, dtype=float)
    if not np.isfinite(pts).all():
        raise ValueError("arc support contains non-finite coordinates")
    singular_values = np.linalg.svd(
        pts - pts.mean(axis=0), compute_uv=False)
    if len(singular_values) < 2 or singular_values[0] <= 0.0 \
            or singular_values[1] / singular_values[0] < 1e-3:
        raise WitnessAbsent(
            "degenerate_fit", "arc support is effectively collinear")
    circle = _fit_circle_raw(contour)
    radial_q95, radial_max = _radial_deviation_tail(contour, circle)
    if circle.residual > MAX_ARC_RESIDUAL \
            or radial_q95 > MAX_ARC_RADIAL_Q95 \
            or radial_max > MAX_ARC_RADIAL_MAX:
        raise WitnessAbsent(
            "high_residual",
            "arc radial residuals exceed the admissible envelope: "
            f"rms={circle.residual:.4g}/{MAX_ARC_RESIDUAL:.4g}, "
            f"q95={radial_q95:.4g}/{MAX_ARC_RADIAL_Q95:.4g}, "
            f"max={radial_max:.4g}/{MAX_ARC_RADIAL_MAX:.4g}")
    center = np.array([circle.center.x, circle.center.y])
    angles = np.unwrap(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    steps = np.diff(angles)
    meaningful = steps[np.abs(steps) > 1e-8]
    if not len(meaningful):
        raise WitnessAbsent(
            "insufficient_angular_support", "arc has no measurable angular sweep")
    direction = 1.0 if float(np.median(meaningful)) > 0.0 else -1.0
    if np.any(direction * meaningful < -math.radians(2.0)):
        raise WitnessAbsent(
            "direction_reversal",
            "arc support reverses angular direction around its fitted center")
    if float(np.max(np.abs(meaningful))) > math.radians(20.0):
        raise WitnessAbsent(
            "insufficient_angular_support",
            "arc support is too sparse to distinguish a curve from chords")
    swept = math.degrees(abs(float(angles[-1] - angles[0])))
    if swept < 15.0:
        raise WitnessAbsent(
            "insufficient_angular_support",
            f"arc sweep {swept:.4g} degrees is below 15 degrees")
    return ArcWitness(
        source_component_id=contour.source_component_id,
        center=circle.center,
        radius=circle.radius,
        angle_degrees=min(360.0, swept),
        residual=circle.residual,
        confidence=circle.confidence,
        provenance=circle.provenance + ("fit_arc",),
    )


def arc_angle_degrees(arc: ArcWitness) -> float:
    return float(arc.angle_degrees)


def line_length(line: LineSegmentWitness) -> float:
    return float(line.length)


def _signed_turning_profile(rs: np.ndarray, closed: bool, k: int) -> np.ndarray:
    """Signed turning angle (radians) per sample; + is left/convex for CCW."""
    n = len(rs)
    ang = np.zeros(n)
    indices = range(n) if closed else range(k, n - k)
    for i in indices:
        a, b, c = rs[(i - k) % n], rs[i], rs[(i + k) % n]
        v1, v2 = b - a, c - b
        n1, n2 = float(np.hypot(*v1)), float(np.hypot(*v2))
        if n1 <= np.finfo(float).tiny or n2 <= np.finfo(float).tiny:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = float(v1 @ v2)
        ang[i] = math.atan2(cross, dot)
    return ang


def _signed_area(rs: np.ndarray) -> float:
    x, y = rs[:, 0], rs[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def count_curve_parts(contour: ContourWitness) -> float:
    """Number of parts by the minima rule (Hoffman & Richards).

    A shape's boundary is segmented at negative minima of curvature — the
    concave creases where transversality says two parts join.  The contour
    is oriented counter-clockwise first so "negative" (concave) is defined
    independent of walk direction, and curvature is measured on an
    arc-length resampling, so the part count is rotation-invariant.  A convex
    blob has one part; a k-lobed/petalled shape has k concave notches → k
    parts.
    """
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 6:
        return 1.0
    closed = contour.is_closed
    if not closed:
        # For an open curve, the stable semantic pieces are its maximal
        # same-turning-direction runs.  Counting negative curvature minima
        # directly on a raster staircase made the answer depend on angle and
        # stroke width; the persistence-filtered reversal count below gives
        # one part for a line/arc and k+1 parts for k genuine reversals.
        return count_inflections(contour) + 1.0
    # Use a dimensionless sampling grid.  Perimeter-derived density made the
    # number of detected concavity minima change when coordinates were merely
    # multiplied by a constant.
    n = 128
    rs = _resample_contour(pts, n, closed)
    if len(rs) < 6:
        return 1.0
    n = len(rs)
    if closed and _signed_area(rs) < 0:
        rs = rs[::-1]
    k = max(2, n // 16)
    window = max(2, n // 12)
    ang = _signed_turning_profile(rs, closed, k)
    concavity = -ang  # positive where the boundary is concave (a notch)
    threshold = math.radians(25.0)
    notches: list[int] = []
    for i in np.argsort(concavity)[::-1]:
        if concavity[i] < threshold:
            break
        i = int(i)
        if closed:
            if all(_circular_distance(i, j, n) > window for j in notches):
                notches.append(i)
        elif all(abs(i - j) > window for j in notches):
            notches.append(i)
    if closed:
        return float(max(1, len(notches)))
    return float(len(notches) + 1)


def count_inflections(contour: ContourWitness) -> float:
    """Number of turning-direction reversals along the curve.

    A general, rotation-invariant shape primitive: resample the ordered
    contour by arc length, take the signed turn (cross product of successive
    tangents) with a deadband to ignore raster noise, and count sign
    changes.  A simple arc turns monotonically (0 reversals); a wavy /
    scalloped / S-shaped curve reverses several times.  This is curve
    geometry, not a named concept.
    """
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 5:
        return 0.0

    # Cross-leg coherence is load-bearing: a contour that passes the honest
    # line or circular-arc fit cannot simultaneously contain an inflection.
    # These fits are normalized and residual-gated, so this is a semantic
    # classification, not a permissive numerical shortcut.
    for primitive_fit in (fit_line_segment, fit_arc):
        try:
            primitive_fit(contour)
            return 0.0
        except WitnessAbsent:
            pass

    # Work on a dimensionless arc-length grid.  Raw raster tangents alternate
    # on every staircase pixel: after an off-grid rotation a perfectly straight
    # line previously produced 30+ false reversals.  Gaussian smoothing on the
    # fixed grid, followed by a minimum-persistence sign run, removes that
    # sampling frequency while retaining broad S-curve reversals.
    from scipy.ndimage import gaussian_filter1d

    n = 128
    rs = _resample_contour(pts, n, contour.is_closed)
    if len(rs) < 5:
        return 0.0
    mode = "wrap" if contour.is_closed else "nearest"
    smooth = np.stack([
        gaussian_filter1d(rs[:, axis], 4.0, mode=mode)
        for axis in range(2)
    ], axis=1)
    turning = _signed_turning_profile(smooth, contour.is_closed, 2)
    turning = gaussian_filter1d(turning, 4.0, mode=mode)
    signs = np.where(
        turning > 0.02, 1, np.where(turning < -0.02, -1, 0))

    runs: list[list[int]] = []
    current = 0
    start = 0
    for index, value in enumerate(signs):
        value = int(value)
        if value == current:
            continue
        if current:
            runs.append([current, index - start])
        current = value
        start = index
    if current:
        runs.append([current, len(signs) - start])

    # A genuine curvature regime must occupy at least 3/128 of the curve.
    # Merge equal-sign regimes separated only by a discarded deadband/noise
    # run before counting transitions.
    persistent = [run for run in runs if run[1] >= 3]
    merged: list[list[int]] = []
    for sign, length in persistent:
        if merged and merged[-1][0] == sign:
            merged[-1][1] += length
        else:
            merged.append([sign, length])
    if contour.is_closed and len(merged) > 1 \
            and merged[0][0] == merged[-1][0]:
        merged[0][1] += merged[-1][1]
        merged.pop()
    if len(merged) < 2:
        return 0.0
    transitions = len(merged) if contour.is_closed else len(merged) - 1
    return float(transitions)


def _same_circle_candidate(a: CircleWitness, b: CircleWitness) -> bool:
    scale = max(1.0, min(a.radius, b.radius))
    center_gap = math.hypot(a.center.x - b.center.x,
                            a.center.y - b.center.y)
    return center_gap <= 0.12 * scale \
        and abs(a.radius - b.radius) <= 0.12 * scale


def _hough_circle_candidates(obj: ObjectMask) -> list[CircleWitness]:
    """Recover full circle supports inside a merged/branched component.

    The Hough transform only proposes geometry.  Admission is determined from
    the actual component pixels after Taubin refinement: every angular bin
    must support the circumference and normalized RMS/q95/max radial errors
    must satisfy the same circle envelope as a simple-contour fit.
    """
    mask = np.asarray(obj.mask, dtype=bool)
    yx = np.argwhere(mask)
    if len(yx) < 24:
        return []
    y0, x0 = yx.min(axis=0)
    y1, x1 = yx.max(axis=0)
    extent = max(int(y1 - y0 + 1), int(x1 - x0 + 1))
    max_radius = min(max(mask.shape) // 2, max(5, int(math.ceil(0.7 * extent))))
    if max_radius < 4:
        return []

    from skimage.transform import hough_circle, hough_circle_peaks

    radii = np.arange(4, max_radius + 1, dtype=int)
    accumulator = hough_circle(mask, radii, normalize=False)
    strengths, centers_x, centers_y, peak_radii = hough_circle_peaks(
        accumulator,
        radii,
        total_num_peaks=24,
        min_xdistance=3,
        min_ydistance=3,
        normalize=False,
    )
    points = yx[:, ::-1].astype(float)  # (x, y)
    candidates: list[CircleWitness] = []
    for peak_index, (strength, cx, cy, radius) in enumerate(zip(
            strengths, centers_x, centers_y, peak_radii)):
        if strength <= 0.0:
            continue
        center = np.array([float(cx), float(cy)])
        fitted_radius = float(radius)
        # Two robust refinements shed the other circle except near the genuine
        # crossings.  The seed band is deliberately wider than admission;
        # angular-bin validation below is the non-circularity gate.
        for _ in range(2):
            radial_error = np.abs(
                np.linalg.norm(points - center, axis=1) - fitted_radius)
            seed_tolerance = max(2.0, 0.12 * fitted_radius)
            support = points[radial_error <= seed_tolerance]
            if len(support) < 16:
                break
            try:
                center, fitted_radius = _taubin_circle(support)
            except ValueError:
                support = np.empty((0, 2), dtype=float)
                break
        if len(support) < 16 or fitted_radius < 3.0:
            continue

        vectors = points - center
        radial_error = np.abs(np.linalg.norm(vectors, axis=1) - fitted_radius)
        angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]),
                        2.0 * math.pi)
        bin_count = max(16, min(48, int(round(2.0 * math.pi * fitted_radius / 3.0))))
        bins = np.minimum(
            (angles * bin_count / (2.0 * math.pi)).astype(int),
            bin_count - 1,
        )
        best_error = np.full(bin_count, np.inf)
        best_index = np.full(bin_count, -1, dtype=int)
        for point_index, bin_index in enumerate(bins):
            if radial_error[point_index] < best_error[bin_index]:
                best_error[bin_index] = radial_error[point_index]
                best_index[bin_index] = point_index
        normalized = best_error / fitted_radius
        # Requiring the maximum bin error also requires full angular support:
        # an empty bin remains +inf and cannot pass.
        if not np.isfinite(normalized).all():
            continue
        rms = float(np.sqrt(np.mean(normalized ** 2)))
        q95 = float(np.quantile(normalized, 0.95))
        maximum = float(np.max(normalized))
        if rms > MAX_CIRCLE_RESIDUAL \
                or q95 > MAX_CIRCLE_RADIAL_Q95 \
                or maximum > MAX_CIRCLE_RADIAL_MAX:
            continue
        source_id = f"{obj.object_id}:circle_{peak_index}"
        candidate = CircleWitness(
            source_component_id=source_id,
            center=PointWitness(
                x=float(center[0]), y=float(center[1]), source_id=source_id),
            radius=float(fitted_radius),
            support_points=tuple(
                (float(points[index, 0]), float(points[index, 1]))
                for index in best_index
            ),
            residual=rms,
            confidence=max(0.0, 1.0 - rms),
            provenance=("fit_multiple_circles", "hough_full_support"),
        )
        if not any(_same_circle_candidate(candidate, old)
                   for old in candidates):
            candidates.append(candidate)
    return candidates


def _joint_circle_coverage(
        obj: ObjectMask,
        first: CircleWitness,
        second: CircleWitness,
        ) -> tuple[float, float]:
    """Fraction of component ink jointly and uniquely explained by a pair."""
    points = np.argwhere(obj.mask)[:, ::-1].astype(float)
    if not len(points):
        return 0.0, 0.0

    def normalized_error(circle: CircleWitness) -> np.ndarray:
        center = np.array([circle.center.x, circle.center.y])
        return np.abs(np.linalg.norm(points - center, axis=1) - circle.radius) \
            / circle.radius

    first_error = normalized_error(first)
    second_error = normalized_error(second)
    nearest = np.minimum(first_error, second_error)
    covered = nearest <= MAX_CIRCLE_RADIAL_MAX
    coverage = float(np.mean(covered))
    if not covered.any():
        return coverage, 0.0
    assigned_first = covered & (first_error <= second_error)
    assigned_second = covered & (second_error < first_error)
    balance = float(min(assigned_first.sum(), assigned_second.sum())
                    / covered.sum())
    return coverage, balance


def fit_multiple_circles(scene: Scene) -> CirclePairWitness:
    candidates: list[tuple[CircleWitness, ObjectMask]] = []
    rejected_high_residual = 0
    for obj in scene.objects:
        simple_circle: CircleWitness | None = None
        try:
            contour = extract_contours(obj)
            simple_circle = fit_circle(contour)
        except WitnessAbsent as exc:
            if exc.failure_mode == "high_residual":
                rejected_high_residual += 1
        if simple_circle is not None:
            candidates.append((simple_circle, obj))
            continue
        hough_candidates = _hough_circle_candidates(obj)
        if not hough_candidates:
            rejected_high_residual += 1
        candidates.extend((candidate, obj) for candidate in hough_candidates)

    distinct: list[tuple[CircleWitness, ObjectMask]] = []
    for candidate, obj in sorted(candidates, key=lambda item: item[0].residual):
        if not any(_same_circle_candidate(candidate, old)
                   for old, _old_obj in distinct):
            distinct.append((candidate, obj))

    best: tuple[float, CircleWitness, CircleWitness] | None = None
    for first_index, (first, first_obj) in enumerate(distinct):
        for second, second_obj in distinct[first_index + 1:]:
            if _same_circle_candidate(first, second):
                continue
            coverage = 1.0
            balance = 0.5
            if first_obj is second_obj:
                coverage, balance = _joint_circle_coverage(
                    first_obj, first, second)
                if coverage < 0.90 or balance < 0.18:
                    continue
            score = max(first.residual, second.residual) \
                + (1.0 - coverage) + max(0.0, 0.25 - balance)
            if best is None or score < best[0]:
                best = (score, first, second)

    if best is None:
        if rejected_high_residual or candidates:
            raise WitnessAbsent(
                "high_residual",
                "fewer than two distinct, fully supported circle candidates "
                "passed residual and joint-coverage gates")
        raise WitnessAbsent(
            "fewer_than_two_candidates", "need at least two circle candidates")
    _score, a, b = best
    d = math.hypot(a.center.x - b.center.x, a.center.y - b.center.y)
    return CirclePairWitness(
        first=a,
        second=b,
        center_distance=d,
        confidence=min(a.confidence, b.confidence),
        residual=max(a.residual, b.residual),
        provenance=a.provenance + b.provenance + ("fit_multiple_circles",),
    )


# ---------------------------------------------------------------------------
# Part decomposition at stroke junctions.  A junction is a pixel cluster of
# degree >= 3; removing it splits the stroke into parts.  Clusters touching
# only two parts are raster artifacts and get merged back.  Real junctions
# become honest contact/intersection evidence: attachment for 3 incident
# branches, crossing for 4 or more.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Junction:
    center: tuple[float, float]           # (x, y)
    part_indices: tuple[int, ...]
    branchiness: int


def _decompose_mask(mask: np.ndarray, source_id: str,
                    min_part_pixels: int = 2
                    ) -> tuple[list[set[tuple[int, int]]], list[_Junction]]:
    coords = {(int(y), int(x)) for y, x in np.argwhere(mask)}
    if not coords:
        return [], []
    deg = _degree_map(mask)
    junction_pixels = {p for p, d in deg.items() if d >= 3}
    clusters = _cluster_points(junction_pixels)
    remainder = coords - junction_pixels
    raw_parts = [c for c in _cluster_points(remainder) if len(c) >= min_part_pixels]

    def adjacent_parts(cluster: set[tuple[int, int]],
                       parts: list[set[tuple[int, int]]]) -> list[int]:
        found = []
        for i, part in enumerate(parts):
            if any((y + dy, x + dx) in part
                   for y, x in cluster for dy, dx in _NEIGHBOR_OFFSETS):
                found.append(i)
        return found

    # Sub-minimum isolated fragments are sampling artifacts around a removed
    # junction cluster.  Attaching one to the nearest part by centroid can
    # make that otherwise simple path disconnected (three apparent endpoints)
    # and destroy the downstream contour.  Real short arms have at least
    # ``min_part_pixels`` support and remain explicit parts.

    # Merge parts across artifact clusters (only two incident branches means
    # the stroke merely continues through a raster-thick spot).
    retained_remainder = set().union(*raw_parts) if raw_parts else set()
    parent = list(range(len(raw_parts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    real_clusters: list[tuple[set[tuple[int, int]], list[int], int]] = []
    for cluster in clusters:
        adj = adjacent_parts(cluster, raw_parts)
        # Distinct remainder components are not the same as incident
        # half-edges: a loop may leave and re-enter one junction through the
        # same component.  Classify the vertex by its localized boundary
        # support so figure-eights (4 half-edges, 2 components) and lollipops
        # (3 half-edges, 2 components) are not merged as raster artifacts.
        # Sampling-scale fragments were deliberately excluded above; letting
        # an isolated one-pixel remnant contribute here turns a three-ray hub
        # into a false four-way intersection.
        boundary = {
            point for point in retained_remainder
            if any((point[0] + dy, point[1] + dx) in cluster
                   for dy, dx in _NEIGHBOR_OFFSETS)
        }
        branchiness = len(boundary)
        if branchiness <= 2:
            for i in adj[1:]:
                union(adj[0], i)
            if adj:
                raw_parts[adj[0]] |= cluster
            continue
        real_clusters.append((cluster, adj, branchiness))

    merged: dict[int, set[tuple[int, int]]] = {}
    for i, part in enumerate(raw_parts):
        merged.setdefault(find(i), set()).update(part)
    part_list = sorted(merged.values(), key=len, reverse=True)
    index_of = {find(i): None for i in range(len(raw_parts))}
    for root in index_of:
        for k, part in enumerate(part_list):
            if merged[root] is part:
                index_of[root] = k
                break

    junctions: list[_Junction] = []
    for cluster, adj, branchiness in real_clusters:
        ys, xs = zip(*cluster)
        center = (float(sum(xs)) / len(xs), float(sum(ys)) / len(ys))
        parts_idx = tuple(sorted({index_of[find(i)] for i in adj
                                  if index_of.get(find(i)) is not None}))
        junctions.append(_Junction(center, parts_idx, branchiness))
    if not part_list:
        part_list = [coords]
    return part_list, junctions


def _part_witness(points: set[tuple[int, int]], shape: tuple[int, int],
                  part_id: str, source_id: str, provenance: tuple[str, ...]
                  ) -> PartWitness:
    if len(points) <= 2:
        # A legitimate short arm can be only two skeleton pixels after the
        # junction cluster is removed.  Re-thinning that isolated fragment
        # collapses it to one pixel, so preserve the already-thinned support
        # directly instead of routing it through extract_contours again.
        ordered = tuple(
            (float(x), float(y)) for y, x in sorted(points))
        contour = ContourWitness(
            source_component_id=source_id,
            points=ordered,
            is_closed=False,
            confidence=1.0,
            provenance=provenance,
        )
    else:
        sub = ObjectMask(_mask_from_points(points, shape), part_id)
        extracted = extract_contours(sub)
        contour = ContourWitness(
            source_component_id=source_id,
            points=extracted.points,
            is_closed=extracted.is_closed,
            confidence=extracted.confidence,
            provenance=provenance,
        )
    return PartWitness(
        part_id=part_id,
        role="stroke",
        source_component_id=source_id,
        contour=contour,
        provenance=provenance,
    )


def _graph_from_mask(mask: np.ndarray, source_id: str,
                     provenance: tuple[str, ...]) -> PartGraphWitness:
    parts_pts, junctions = _decompose_mask(mask, source_id)
    try:
        parts = tuple(
            _part_witness(
                pts, mask.shape, f"{source_id}_part_{i}", source_id,
                provenance)
            for i, pts in enumerate(parts_pts)
        )
    except WitnessAbsent as exc:
        if exc.failure_mode != "not_simple_curve":
            raise
        raise WitnessIndeterminate(
            "not_simple_curve",
            "part-graph extraction could not order a decomposed stroke as "
            "a simple curve",
        ) from exc
    contacts = []
    adjacency = []
    for j in junctions:
        relation = "intersection" if j.branchiness >= 4 else "attachment"
        point = PointWitness(x=j.center[0], y=j.center[1], source_id=source_id)
        ids = [parts[i].part_id for i in j.part_indices if i < len(parts)]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                adjacency.append((ids[a], ids[b]))
        if len(ids) >= 2:
            witness_cls = IntersectionWitness if relation == "intersection" else ContactWitness
            contacts.append(witness_cls(
                source_a=ids[0], source_b=ids[1], points=(point,),
                relation=relation, confidence=1.0,
                provenance=provenance,
            ))
    return PartGraphWitness(
        parts=parts,
        contacts=tuple(contacts),
        adjacency=tuple(adjacency),
        confidence=1.0,
        provenance=provenance,
    )


def decompose_component_into_parts(obj: ObjectMask) -> PartGraphWitness:
    return _graph_from_mask(_topo(obj), obj.object_id,
                            ("decompose_component_into_parts",))


def build_part_graph(value: Scene | ObjectMask) -> PartGraphWitness:
    if isinstance(value, ObjectMask):
        return _graph_from_mask(_topo(value), value.object_id, ("build_part_graph",))
    parts: list[PartWitness] = []
    contacts: list[ContactWitness] = []
    adjacency: list[tuple[str, str]] = []
    for obj in value.objects:
        sub = _graph_from_mask(_topo(obj), obj.object_id, ("build_part_graph",))
        parts.extend(sub.parts)
        contacts.extend(sub.contacts)
        adjacency.extend(sub.adjacency)
    return PartGraphWitness(
        parts=tuple(parts),
        contacts=tuple(contacts),
        adjacency=tuple(adjacency),
        confidence=1.0,
        provenance=("build_part_graph",),
    )


_INCIDENT_ENDPOINT_GAP_FRACTION = 0.08
_INCIDENT_RAY_WINDOW_FRACTION = 0.11
_INCIDENT_RAY_MIN_POINTS = 5
_MAX_INCIDENT_RAY_RESIDUAL = 0.08


def _fit_incident_ray(
        part: PartWitness, junction: PointWitness, *, reverse: bool,
        ) -> LineSegmentWitness | None:
    """Fit one local ray from a witnessed junction into a graph part."""
    contour = part.contour
    endpoint_name = "end" if reverse else "start"
    if contour is None or contour.is_closed \
            or len(contour.points) < _INCIDENT_RAY_MIN_POINTS:
        raise WitnessIndeterminate(
            "insufficient_incident_rays",
            f"part {part.part_id!r} has no usable open {endpoint_name} contour",
        )
    points = np.asarray(contour.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 \
            or not np.isfinite(points).all():
        raise ValueError("incident-ray contour points must be finite xy pairs")
    if reverse:
        points = points[::-1]
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(np.sum(steps))
    if not math.isfinite(total_length) \
            or total_length <= np.finfo(float).tiny:
        raise WitnessIndeterminate(
            "insufficient_incident_rays",
            f"part {part.part_id!r} has degenerate contour length",
        )
    junction_xy = np.asarray((junction.x, junction.y), dtype=float)
    if not np.isfinite(junction_xy).all():
        raise ValueError("junction coordinates must be finite")
    endpoint_gap = float(np.linalg.norm(points[0] - junction_xy))
    if endpoint_gap > _INCIDENT_ENDPOINT_GAP_FRACTION * total_length:
        return None

    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    stop = int(np.searchsorted(
        cumulative,
        _INCIDENT_RAY_WINDOW_FRACTION * total_length,
        side="right",
    ))
    stop = min(len(points), max(_INCIDENT_RAY_MIN_POINTS, stop))
    local = points[:stop]
    center = np.mean(local, axis=0)
    centered = local - center
    covariance = centered.T @ centered / len(local)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if float(np.dot(axis, center - junction_xy)) < 0.0:
        axis = -axis
    projections = centered @ axis
    span = float(np.ptp(projections))
    scale_floor = np.finfo(float).eps * max(1.0, total_length)
    if not math.isfinite(span) or span <= scale_floor:
        raise WitnessIndeterminate(
            "insufficient_incident_rays",
            f"part {part.part_id!r} {endpoint_name} ray is degenerate",
        )
    normal = np.asarray((-axis[1], axis[0]))
    orthogonal = centered @ normal
    junction_offset = float(np.dot(junction_xy - center, normal))
    residual = math.sqrt(
        (float(np.dot(orthogonal, orthogonal)) + junction_offset ** 2)
        / (len(local) + 1)
    ) / span
    if not math.isfinite(residual):
        raise ValueError("incident-ray residual is non-finite")
    if residual > _MAX_INCIDENT_RAY_RESIDUAL:
        raise WitnessIndeterminate(
            "high_residual",
            f"part {part.part_id!r} {endpoint_name} ray residual "
            f"{residual:.4g} exceeds {_MAX_INCIDENT_RAY_RESIDUAL:.4g}",
        )
    length = float(np.max((local - junction_xy) @ axis))
    if not math.isfinite(length) or length <= scale_floor:
        raise WitnessIndeterminate(
            "insufficient_incident_rays",
            f"part {part.part_id!r} {endpoint_name} ray has no outward extent",
        )
    ray_id = f"{part.part_id}:{endpoint_name}:incident-ray"
    start = PointWitness(
        x=float(junction_xy[0]), y=float(junction_xy[1]), source_id=ray_id)
    end_xy = junction_xy + length * axis
    end = PointWitness(
        x=float(end_xy[0]), y=float(end_xy[1]), source_id=ray_id)
    confidence = min(
        float(part.confidence), float(contour.confidence),
        max(0.0, 1.0 - residual),
    )
    return LineSegmentWitness(
        source_component_id=ray_id,
        points=tuple((float(x), float(y)) for x, y in local),
        endpoints=(start, end),
        start=start,
        end=end,
        length=length,
        residual=residual,
        confidence=confidence,
        provenance=part.provenance + contour.provenance
        + (f"fit_incident_ray:{endpoint_name}",),
    )


_POINT_CONTACT_MIN_HOLE_PIXELS = 10
_POINT_CONTACT_DILATION_ITERATIONS = 1
_POINT_CONTACT_MAX_NORMALIZED_GAP = 0.30
_POINT_CONTACT_MAX_NORMALIZED_INTERFACE_SPREAD = 0.30
_POINT_CONTACT_RAY_WINDOW_FRACTION = 0.13
_POINT_CONTACT_RAY_MIN_POINTS = 7
_POINT_CONTACT_MAX_RAY_RESIDUAL = 0.09


def _ordered_provenance(*items: tuple[str, ...] | str) -> tuple[str, ...]:
    result: list[str] = []
    for item in items:
        values = (item,) if isinstance(item, str) else item
        for value in values:
            if value and value not in result:
                result.append(value)
    return tuple(result)


def _point_contact_indeterminate(detail: str) -> WitnessIndeterminate:
    # This mode is deliberately distinct from semantic non-membership.  The
    # verifier can therefore invalidate an unresolved fit instead of treating
    # it as evidence that the panel lacks the concept.
    return WitnessIndeterminate("point_contact_fit_indeterminate", detail)


def _owned_incident_ray(
        segment: LineSegmentWitness, owner_id: str, endpoint_name: str,
        ) -> IncidentRayWitness:
    vector = np.asarray(
        (segment.end.x - segment.start.x, segment.end.y - segment.start.y),
        dtype=float,
    )
    length = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not math.isfinite(length) \
            or length <= np.finfo(float).tiny:
        raise _point_contact_indeterminate(
            f"{owner_id!r} has a degenerate fitted incident ray")
    residual = float(segment.residual)
    if not math.isfinite(residual) or residual < 0.0:
        raise _point_contact_indeterminate(
            f"{owner_id!r} has a non-finite incident-ray residual")
    direction = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
    uncertainty = math.degrees(math.atan(residual))
    ray_id = segment.source_component_id \
        or f"{owner_id}:{endpoint_name}:incident-ray"
    return IncidentRayWitness(
        ray_id=ray_id,
        owner_id=owner_id,
        endpoint_name=endpoint_name,
        segment=segment,
        direction_degrees=direction,
        uncertainty_degrees=uncertainty,
        confidence=float(segment.confidence),
        residual=residual,
        provenance=_ordered_provenance(
            segment.provenance, "owner_labelled_incident_ray"),
    )


def _assemble_point_contact_signature(
        vertex: PointWitness,
        part_ids: tuple[str, str],
        rays: tuple[IncidentRayWitness, ...],
        provenance: tuple[str, ...],
        *,
        confidence: float,
        ) -> PointContactSignature:
    """Certify cyclic ownership and retain both exterior angular gaps."""
    if len(rays) != 4:
        raise _point_contact_indeterminate(
            f"expected four fitted rays, obtained {len(rays)}")
    ordered = tuple(sorted(
        rays,
        key=lambda ray: (ray.direction_degrees, ray.owner_id, ray.ray_id),
    ))
    cyclic_owners = tuple(ray.owner_id for ray in ordered)
    transitions = sum(
        cyclic_owners[index] != cyclic_owners[(index + 1) % 4]
        for index in range(4)
    )
    if transitions != 2:
        raise WitnessAbsent(
            "no_point_contact_signature",
            "incident-ray owners interleave cyclically instead of forming "
            "two loop blocks",
        )

    gaps: list[ExteriorGapWitness] = []
    for index, ray in enumerate(ordered):
        following = ordered[(index + 1) % 4]
        degrees = (
            float(following.direction_degrees) - float(ray.direction_degrees)
        ) % 360.0
        uncertainty = (
            float(ray.uncertainty_degrees)
            + float(following.uncertainty_degrees)
        )
        if not math.isfinite(degrees) or degrees <= uncertainty:
            raise _point_contact_indeterminate(
                "incident-ray uncertainty does not certify a strict cyclic "
                "ordering")
        if ray.owner_id == following.owner_id:
            continue
        gaps.append(ExteriorGapWitness(
            ray_a_id=ray.ray_id,
            ray_b_id=following.ray_id,
            owner_a=ray.owner_id,
            owner_b=following.owner_id,
            degrees=degrees,
            uncertainty_degrees=uncertainty,
            confidence=min(ray.confidence, following.confidence),
            residual=max(ray.residual, following.residual),
            provenance=_ordered_provenance(
                ray.provenance, following.provenance,
                "cyclic_cross_owner_exterior_gap"),
        ))
    if len(gaps) != 2:
        raise _point_contact_indeterminate(
            f"expected two exterior gaps, obtained {len(gaps)}")
    gaps.sort(key=lambda gap: (
        gap.degrees, gap.uncertainty_degrees,
        gap.ray_a_id, gap.ray_b_id,
    ))
    residual = max(ray.residual for ray in ordered)
    return PointContactSignature(
        vertex=vertex,
        part_ids=part_ids,
        contact_count=1,
        loop_incidence=tuple(
            (part_id, True, True) for part_id in part_ids),
        rays=ordered,
        cyclic_owners=cyclic_owners,
        exterior_gaps=(gaps[0], gaps[1]),
        confidence=min(confidence, *(ray.confidence for ray in ordered)),
        residual=residual,
        provenance=_ordered_provenance(
            provenance,
            *(ray.provenance for ray in ordered),
            "assemble_point_contact_signature",
        ),
    )


def _fit_loop_boundary_ray(
        points: np.ndarray,
        contact_index: int,
        *,
        step: int,
        owner_id: str,
        conceptual_vertex: PointWitness,
        ) -> IncidentRayWitness:
    """Fit one scale-relative local ray on a closed enclosed-region boundary."""
    if points.ndim != 2 or points.shape[1] != 2 \
            or len(points) < 2 * _POINT_CONTACT_RAY_MIN_POINTS \
            or not np.isfinite(points).all():
        raise _point_contact_indeterminate(
            f"{owner_id!r} has insufficient finite loop-boundary support")
    count = max(
        _POINT_CONTACT_RAY_MIN_POINTS,
        int(round(len(points) * _POINT_CONTACT_RAY_WINDOW_FRACTION)),
    )
    count = min(count, max(_POINT_CONTACT_RAY_MIN_POINTS, len(points) // 3))
    indices = tuple(
        (contact_index + step * offset) % len(points)
        for offset in range(count)
    )
    local = points[np.asarray(indices, dtype=int)]
    center = np.mean(local, axis=0)
    centered = local - center
    covariance = centered.T @ centered / len(local)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    origin = points[contact_index]
    if float(np.dot(axis, center - origin)) < 0.0:
        axis = -axis
    projections = centered @ axis
    span = float(np.ptp(projections))
    if not math.isfinite(span) or span <= np.finfo(float).tiny:
        raise _point_contact_indeterminate(
            f"{owner_id!r} has a degenerate loop-boundary ray")
    normal = np.asarray((-axis[1], axis[0]))
    orthogonal = centered @ normal
    residual = math.sqrt(float(np.dot(orthogonal, orthogonal)) / len(local)) \
        / span
    if not math.isfinite(residual) or residual > _POINT_CONTACT_MAX_RAY_RESIDUAL:
        raise _point_contact_indeterminate(
            f"{owner_id!r} loop ray residual {residual:.4g} exceeds "
            f"{_POINT_CONTACT_MAX_RAY_RESIDUAL:.4g}")
    extent = float(np.max((local - origin) @ axis))
    if not math.isfinite(extent) or extent <= np.finfo(float).tiny:
        raise _point_contact_indeterminate(
            f"{owner_id!r} loop ray has no outward extent")
    endpoint_name = "start" if step > 0 else "end"
    ray_id = f"{owner_id}:{endpoint_name}:boundary-ray"
    start = PointWitness(
        x=float(origin[0]), y=float(origin[1]), source_id=ray_id)
    end_xy = origin + extent * axis
    end = PointWitness(
        x=float(end_xy[0]), y=float(end_xy[1]), source_id=ray_id)
    segment = LineSegmentWitness(
        source_component_id=ray_id,
        points=tuple((float(x), float(y)) for x, y in local),
        endpoints=(start, end),
        start=start,
        end=end,
        length=extent,
        confidence=max(0.0, 1.0 - residual),
        residual=residual,
        provenance=(
            "one_pixel_dilated_enclosed_region",
            f"fit_loop_boundary_ray:{endpoint_name}",
            f"conceptual_vertex:{conceptual_vertex.source_id}",
        ),
    )
    return _owned_incident_ray(segment, owner_id, endpoint_name)


def _enclosed_region_boundaries(
        panel: np.ndarray,
        ) -> tuple[tuple[np.ndarray, int], ...]:
    """Return substantive enclosed background components after dilation."""
    array = np.asarray(panel)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number) \
            or not np.isfinite(array).all():
        raise ValueError("point-contact extraction requires a finite 2-D panel")
    from scipy import ndimage
    from skimage.measure import find_contours

    ink = array > 0
    dilated = ndimage.binary_dilation(
        ink, iterations=_POINT_CONTACT_DILATION_ITERATIONS)
    labels, count = ndimage.label(
        ~dilated, structure=np.ones((3, 3), dtype=np.uint8))
    border_ids = set(np.unique(np.concatenate((
        labels[0], labels[-1], labels[:, 0], labels[:, -1],
    ))))
    regions: list[tuple[np.ndarray, int]] = []
    for region_id in range(1, int(count) + 1):
        if region_id in border_ids:
            continue
        region = labels == region_id
        area = int(np.count_nonzero(region))
        if area <= _POINT_CONTACT_MIN_HOLE_PIXELS:
            continue
        contours = find_contours(region.astype(float), level=0.5)
        if not contours:
            raise _point_contact_indeterminate(
                f"enclosed region {region_id} has no extractable boundary")
        boundary_yx = max(contours, key=len)
        if len(boundary_yx) >= 2 \
                and np.allclose(boundary_yx[0], boundary_yx[-1]):
            boundary_yx = boundary_yx[:-1]
        boundary_xy = np.asarray(boundary_yx[:, ::-1], dtype=float)
        if len(boundary_xy) < 2 * _POINT_CONTACT_RAY_MIN_POINTS:
            raise _point_contact_indeterminate(
                f"enclosed region {region_id} boundary is undersampled")
        regions.append((boundary_xy, area))
    # Area and perimeter are intrinsic tie-breakers for stable owner labels.
    # The centroid fallback affects IDs only in an exact descriptor tie; all
    # semantic measurements remain owner-swap invariant.
    regions.sort(key=lambda item: (
        -item[1], -len(item[0]),
        float(np.mean(item[0][:, 0])), float(np.mean(item[0][:, 1])),
    ))
    return tuple(regions)


def extract_point_contact_signature(panel: np.ndarray) -> PointContactSignature:
    """Extract the complete local geometry of two outlined loops touching.

    Skeleton degree is intentionally not the source of truth: raster corners
    can create or erase degree-three clusters under harmless rerendering.  A
    loop is instead witnessed as a substantive enclosed background region
    after one-pixel ink dilation.  Exactly two such regions must form one
    scale-relative, spatially local interface; their two boundary directions
    each then produce the four owner-labelled incident rays.
    """
    regions = _enclosed_region_boundaries(panel)
    if len(regions) != 2:
        raise WitnessAbsent(
            "no_point_contact_signature",
            f"expected exactly two enclosed loop regions, obtained {len(regions)}",
        )
    from scipy.spatial.distance import cdist, pdist

    first, second = regions
    distances = cdist(first[0], second[0])
    if not np.isfinite(distances).all() or distances.size == 0:
        raise _point_contact_indeterminate(
            "loop-boundary distance calculation is unresolved")
    minimum = float(np.min(distances))
    loop_scale = math.sqrt(float(min(first[1], second[1])))
    if not math.isfinite(loop_scale) or loop_scale <= np.finfo(float).tiny:
        raise _point_contact_indeterminate("enclosed-loop scale is degenerate")
    if minimum > _POINT_CONTACT_MAX_NORMALIZED_GAP * loop_scale:
        raise WitnessAbsent(
            "no_point_contact_signature",
            f"two loops are separated by {minimum / loop_scale:.4g} loop "
            "scales, above the point-contact gate",
        )

    # Certify that the near-minimum interface is one point-like cluster rather
    # than an extended shared boundary or two separate contact sites.
    near_pairs = np.argwhere(distances <= minimum + 1.0)
    midpoints = np.asarray([
        (first[0][int(i)] + second[0][int(j)]) / 2.0
        for i, j in near_pairs
    ])
    interface_spread = float(np.max(pdist(midpoints))) \
        if len(midpoints) > 1 else 0.0
    if interface_spread > (
            _POINT_CONTACT_MAX_NORMALIZED_INTERFACE_SPREAD * loop_scale):
        raise WitnessAbsent(
            "no_point_contact_signature",
            "the loop interface is extended or has multiple contact sites",
        )

    # Choose the exact closest pair nearest the centre of the certified local
    # interface.  This is deterministic even when rasterization creates a
    # plateau of equally close samples.
    interface_center = np.mean(midpoints, axis=0)
    candidate_pairs = np.argwhere(
        np.isclose(distances, minimum, rtol=0.0, atol=1e-12))
    chosen_i, chosen_j = min(
        ((int(i), int(j)) for i, j in candidate_pairs),
        key=lambda pair: (
            float(np.linalg.norm(
                (first[0][pair[0]] + second[0][pair[1]]) / 2.0
                - interface_center)),
            pair,
        ),
    )
    first_id, second_id = "enclosed-loop-0", "enclosed-loop-1"
    vertex_xy = (first[0][chosen_i] + second[0][chosen_j]) / 2.0
    vertex = PointWitness(
        x=float(vertex_xy[0]), y=float(vertex_xy[1]),
        source_id="point-contact-interface-midpoint",
        confidence=max(0.0, 1.0 - minimum / loop_scale),
        residual=minimum / loop_scale,
        provenance=(
            "one_pixel_dilated_enclosed_regions",
            "unique_near_contact_interface",
        ),
    )
    rays = tuple(
        _fit_loop_boundary_ray(
            boundary,
            contact_index,
            step=step,
            owner_id=owner_id,
            conceptual_vertex=vertex,
        )
        for owner_id, boundary, contact_index in (
            (first_id, first[0], chosen_i),
            (second_id, second[0], chosen_j),
        )
        for step in (-1, 1)
    )
    return _assemble_point_contact_signature(
        vertex,
        (first_id, second_id),
        rays,
        provenance=(
            "extract_point_contact_signature",
            "enclosed_region_topology",
            "scale_relative_near_contact",
            "noninterleaving_cyclic_ownership",
        ),
        confidence=vertex.confidence,
    )


def _extract_graph_point_contact_signature(
        graph: PartGraphWitness) -> PointContactSignature:
    """Diagnostic exact-graph realization of the same signature schema."""
    if len(graph.parts) != 2 or len(graph.contacts) != 1:
        raise WitnessAbsent(
            "no_point_contact_signature",
            "graph realization requires exactly two parts and one contact",
        )
    parts: dict[str, PartWitness] = {}
    for part in graph.parts:
        if not part.part_id or part.part_id in parts:
            raise ValueError("part graph must have unique nonempty part IDs")
        parts[part.part_id] = part
    contact = graph.contacts[0]
    if len(contact.points) != 1 or contact.source_a == contact.source_b \
            or {contact.source_a, contact.source_b} != set(parts):
        raise WitnessAbsent(
            "no_point_contact_signature",
            "the unique contact must bind both parts at exactly one point",
        )
    vertex = contact.points[0]
    rays: list[IncidentRayWitness] = []
    for part_id in (contact.source_a, contact.source_b):
        part = parts[part_id]
        contour = part.contour
        if contour is None or len(contour.points) \
                < _INCIDENT_RAY_MIN_POINTS:
            raise _point_contact_indeterminate(
                f"part {part_id!r} lacks usable loop-contour support")
        if contour.is_closed:
            raise WitnessAbsent(
                "no_point_contact_signature",
                f"part {part_id!r} is not cut into a loop path at the contact",
            )
        points = np.asarray(contour.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 \
                or not np.isfinite(points).all():
            raise _point_contact_indeterminate(
                f"part {part_id!r} has malformed loop-contour support")
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total_length = float(np.sum(steps))
        if not math.isfinite(total_length) \
                or total_length <= np.finfo(float).tiny:
            raise _point_contact_indeterminate(
                f"part {part_id!r} has degenerate loop-contour length")
        vertex_xy = np.asarray((vertex.x, vertex.y), dtype=float)
        endpoint_gaps = (
            float(np.linalg.norm(points[0] - vertex_xy)),
            float(np.linalg.norm(points[-1] - vertex_xy)),
        )
        if any(gap > _INCIDENT_ENDPOINT_GAP_FRACTION * total_length
               for gap in endpoint_gaps):
            raise WitnessAbsent(
                "no_point_contact_signature",
                f"part {part_id!r} does not start and end at the contact",
            )
        for reverse, endpoint_name in ((False, "start"), (True, "end")):
            try:
                segment = _fit_incident_ray(part, vertex, reverse=reverse)
            except (WitnessAbsent, WitnessIndeterminate) as exc:
                raise _point_contact_indeterminate(
                    f"part {part_id!r} incident-ray fit failed: "
                    f"{exc.failure_mode}") from exc
            if segment is None:
                raise WitnessAbsent(
                    "no_point_contact_signature",
                    f"part {part_id!r} endpoint is not incident to the contact",
                )
            rays.append(_owned_incident_ray(segment, part_id, endpoint_name))
    return _assemble_point_contact_signature(
        vertex,
        (contact.source_a, contact.source_b),
        tuple(rays),
        provenance=_ordered_provenance(
            graph.provenance, contact.provenance,
            "diagnostic_graph_point_contact_signature"),
        confidence=min(graph.confidence, contact.confidence),
    )


def point_contact_small_exterior_gap_degrees(
        signature: PointContactSignature) -> float:
    """Central fitted value of the smaller cross-owner exterior gap."""
    return float(signature.exterior_gaps[0].degrees)


def point_contact_large_exterior_gap_degrees(
        signature: PointContactSignature) -> float:
    """Central fitted value of the larger cross-owner exterior gap."""
    return float(signature.exterior_gaps[1].degrees)


def point_contact_exterior_gap_ratio(
        signature: PointContactSignature) -> float:
    """Ratio of large to small exterior gaps (always at least one)."""
    small = point_contact_small_exterior_gap_degrees(signature)
    large = point_contact_large_exterior_gap_degrees(signature)
    if small <= 0.0:
        raise ValueError("point-contact small exterior gap must be positive")
    return large / small


def point_contact_gap_ratio_lower_bound(
        signature: PointContactSignature) -> float:
    """Uncertainty-conservative lower bound on exterior-gap asymmetry."""
    small, large = signature.exterior_gaps
    denominator = float(small.degrees + small.uncertainty_degrees)
    numerator = max(0.0, float(
        large.degrees - large.uncertainty_degrees))
    if denominator <= 0.0:
        raise ValueError("point-contact conservative ratio is undefined")
    return numerator / denominator


def minimum_incident_angle(graph: PartGraphWitness) -> AngleWitness:
    """Return the smallest cross-part ray angle at a witnessed junction.

    Only the two part identities bound by each ``ContactWitness`` participate.
    Every endpoint close enough to that contact is fitted over a local,
    scale-relative arc-length window.  A junction is admitted only when every
    incident endpoint of both bound parts passes the line residual gate, so a
    curved or unresolved arm cannot silently disappear before minimization.
    """
    if not graph.contacts:
        raise WitnessAbsent("no_junction", "part graph has no witnessed junction")
    parts: dict[str, PartWitness] = {}
    for part in graph.parts:
        if not part.part_id or part.part_id in parts:
            raise ValueError("part graph must have unique nonempty part IDs")
        parts[part.part_id] = part

    candidates: list[AngleWitness] = []
    saw_unique_point = False
    saw_high_residual = False
    for contact in graph.contacts:
        if len(contact.points) != 1:
            continue
        saw_unique_point = True
        if contact.source_a == contact.source_b:
            continue
        bound_parts = (parts.get(contact.source_a), parts.get(contact.source_b))
        if any(part is None for part in bound_parts):
            continue
        ray_groups: list[list[LineSegmentWitness]] = []
        valid_junction = True
        for part in bound_parts:
            assert part is not None
            part_rays: list[LineSegmentWitness] = []
            for reverse in (False, True):
                try:
                    ray = _fit_incident_ray(
                        part, contact.points[0], reverse=reverse)
                    if ray is not None:
                        part_rays.append(ray)
                except WitnessIndeterminate as exc:
                    if exc.failure_mode not in {
                            "high_residual", "insufficient_incident_rays"}:
                        raise
                    saw_high_residual |= exc.failure_mode == "high_residual"
                    valid_junction = False
                    break
            if not valid_junction or not part_rays:
                valid_junction = False
                break
            ray_groups.append(part_rays)
        if not valid_junction or len(ray_groups) != 2:
            continue
        # The contact binds two parts.  Its angle is a separating relation
        # between them, not an interior angle between two endpoints of one
        # part.  Keeping the ray groups explicit prevents an acute same-part
        # corner from silently winning the minimization.
        for first in ray_groups[0]:
            for second in ray_groups[1]:
                angle = angle_between_segments(first, second)
                confidence = min(
                    float(graph.confidence), float(contact.confidence),
                    float(angle.confidence),
                )
                candidates.append(AngleWitness(
                    source_a=angle.source_a,
                    source_b=angle.source_b,
                    vertex=angle.vertex,
                    degrees=angle.degrees,
                    uncertainty_degrees=angle.uncertainty_degrees,
                    reference_frame=angle.reference_frame,
                    residual=angle.residual,
                    confidence=confidence,
                    provenance=graph.provenance + contact.provenance
                    + angle.provenance + ("minimum_incident_angle",),
                ))
    if candidates:
        return min(candidates, key=lambda angle: (
            angle.degrees,
            angle.uncertainty_degrees,
            angle.source_a,
            angle.source_b,
            angle.vertex.x,
            angle.vertex.y,
        ))
    if saw_high_residual:
        raise WitnessIndeterminate(
            "high_residual",
            "no witnessed junction has only well-fitted incident rays",
        )
    if not saw_unique_point:
        raise WitnessAbsent(
            "no_junction", "part graph has no unique witnessed junction point")
    raise WitnessIndeterminate(
        "insufficient_incident_rays",
        "no witnessed junction binds two parts with usable incident rays",
    )


def build_containment_tree(scene: Scene) -> PartGraphWitness:
    return build_part_graph(scene)


def decompose_curve_into_arcs_and_lines(contour: ContourWitness) -> PartGraphWitness:
    pts = np.asarray(contour.points, dtype=float)
    if len(pts) < 2:
        raise ValueError("contour too small to decompose")
    vertices, _residual = _polygon_vertices(contour)
    cuts = [0]
    vert_xy = [np.array([v.x, v.y]) for v in vertices]
    for v in vert_xy:
        idx = int(np.argmin(np.linalg.norm(pts - v, axis=1)))
        cuts.append(idx)
    cuts.append(len(pts) - 1)
    cuts = sorted(set(cuts))
    parts = []
    contacts = []
    adjacency = []
    prov = contour.provenance + ("decompose_curve_into_arcs_and_lines",)
    for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        if b - a < 2:
            continue
        seg_points = tuple((float(x), float(y)) for x, y in pts[a:b + 1])
        seg_contour = ContourWitness(
            source_component_id=contour.source_component_id,
            points=seg_points, is_closed=False,
            confidence=contour.confidence, provenance=prov,
        )
        parts.append(PartWitness(
            part_id=f"{contour.source_component_id}_seg_{k}",
            role="segment",
            source_component_id=contour.source_component_id,
            contour=seg_contour,
            provenance=prov,
        ))
    for prev, nxt in zip(parts[:-1], parts[1:]):
        adjacency.append((prev.part_id, nxt.part_id))
        joint = prev.contour.points[-1]
        contacts.append(ContactWitness(
            source_a=prev.part_id, source_b=nxt.part_id,
            points=(PointWitness(x=joint[0], y=joint[1],
                                 source_id=contour.source_component_id),),
            relation="shared_endpoint", confidence=1.0, provenance=prov,
        ))
    if not parts:
        raise ValueError("no curve segments found")
    return PartGraphWitness(parts=tuple(parts), contacts=tuple(contacts),
                            adjacency=tuple(adjacency), confidence=1.0,
                            provenance=prov)


# ---------------------------------------------------------------------------
# Honest relation witnesses.  These raise when the relation is absent; the
# counting measurements below return 0 instead.
# ---------------------------------------------------------------------------

def _part_centroid(part: PartWitness) -> PointWitness:
    if not part.contour or not part.contour.points:
        return PointWitness(source_id=part.part_id)
    pts = np.asarray(part.contour.points, dtype=float)
    c = pts.mean(axis=0)
    return PointWitness(x=float(c[0]), y=float(c[1]), source_id=part.part_id)


def detect_contact(graph: PartGraphWitness) -> ContactWitness:
    if len(graph.parts) < 2:
        raise WitnessAbsent(
            "no_contact", "need at least two parts to witness contact")
    if not graph.contacts:
        raise WitnessAbsent("no_contact", "no contact between parts")
    return max(graph.contacts, key=lambda c: c.confidence)


def detect_attachment(graph: PartGraphWitness) -> ContactWitness:
    attachments = [
        contact for contact in graph.contacts
        if contact.relation == "attachment"
    ]
    if not attachments:
        raise WitnessAbsent(
            "no_attachment", "no attachment junction between parts")
    return max(attachments, key=lambda contact: contact.confidence)


def detect_tangency(graph: PartGraphWitness) -> ContactWitness:
    c = detect_contact(graph)
    return ContactWitness(source_a=c.source_a, source_b=c.source_b, points=c.points,
                          relation="tangency", confidence=c.confidence,
                          provenance=c.provenance + ("detect_tangency",))


def detect_intersection(graph: PartGraphWitness) -> IntersectionWitness:
    crossings = [c for c in graph.contacts if c.relation == "intersection"]
    if not crossings:
        raise WitnessAbsent(
            "no_crossing", "no crossing junction between parts")
    best = max(crossings, key=lambda c: c.confidence)
    return IntersectionWitness(
        source_a=best.source_a, source_b=best.source_b, points=best.points,
        confidence=best.confidence,
        provenance=best.provenance + ("detect_intersection",),
    )


def detect_shared_endpoint(graph: PartGraphWitness) -> ContactWitness:
    shared = [c for c in graph.contacts if c.relation == "shared_endpoint"]
    if shared:
        return shared[0]
    c = detect_contact(graph)
    return ContactWitness(source_a=c.source_a, source_b=c.source_b, points=c.points,
                          relation="shared_endpoint", confidence=c.confidence,
                          provenance=c.provenance + ("detect_shared_endpoint",))


def detect_shared_point(graph: PartGraphWitness) -> ContactWitness:
    c = detect_contact(graph)
    return ContactWitness(source_a=c.source_a, source_b=c.source_b, points=c.points,
                          relation="shared_point", confidence=c.confidence,
                          provenance=c.provenance + ("detect_shared_point",))


def circle_pair_intersection(pair: CirclePairWitness) -> CircleIntersectionWitness:
    a, b = pair.first, pair.second
    geometry = (
        a.center.x, a.center.y, a.radius,
        b.center.x, b.center.y, b.radius,
        pair.center_distance,
    )
    if not all(math.isfinite(value) for value in geometry):
        raise ValueError("circle pair contains non-finite geometry")
    if a.radius <= 0.0 or b.radius <= 0.0:
        raise ValueError("circle radii must be positive")

    dx = b.center.x - a.center.x
    dy = b.center.y - a.center.y
    d = math.hypot(dx, dy)
    scale = max(a.radius, b.radius, d, np.finfo(float).tiny)
    distance_tolerance = 1e-9 * scale
    if not math.isclose(
            pair.center_distance, d, rel_tol=1e-9,
            abs_tol=distance_tolerance):
        raise ValueError(
            "circle pair center_distance is inconsistent with its centers")

    if d <= distance_tolerance:
        if abs(a.radius - b.radius) <= distance_tolerance:
            raise WitnessAbsent(
                "no_intersection",
                "coincident circles do not have isolated intersection points")
        raise WitnessAbsent(
            "no_intersection", "concentric circles do not intersect")

    minimum_distance = abs(a.radius - b.radius)
    maximum_distance = a.radius + b.radius
    if (d < minimum_distance - distance_tolerance
            or d > maximum_distance + distance_tolerance):
        raise WitnessAbsent(
            "no_intersection", "circle pair does not intersect")

    # Distance from a's center to the chord joining the intersection points.
    chord_offset = (
        a.radius * a.radius - b.radius * b.radius + d * d
    ) / (2.0 * d)
    half_chord_sq = a.radius * a.radius - chord_offset * chord_offset
    squared_tolerance = 1e-9 * scale * scale
    if half_chord_sq < -squared_tolerance:
        raise ValueError("circle intersection construction is inconsistent")

    base_x = a.center.x + chord_offset * dx / d
    base_y = a.center.y + chord_offset * dy / d
    source_prefix = (
        f"{a.source_component_id}:{b.source_component_id}:circle_intersection")
    if half_chord_sq <= squared_tolerance:
        points = (PointWitness(
            x=base_x,
            y=base_y,
            source_id=source_prefix,
        ),)
    else:
        half_chord = math.sqrt(half_chord_sq)
        offset_x = -dy * half_chord / d
        offset_y = dx * half_chord / d
        points = (
            PointWitness(
                x=base_x + offset_x,
                y=base_y + offset_y,
                source_id=f"{source_prefix}:0",
            ),
            PointWitness(
                x=base_x - offset_x,
                y=base_y - offset_y,
                source_id=f"{source_prefix}:1",
            ),
        )
    return CircleIntersectionWitness(
        source_a=a.source_component_id,
        source_b=b.source_component_id,
        points=points,
        pair=pair,
        confidence=pair.confidence,
        residual=pair.residual,
        provenance=pair.provenance + ("circle_pair_intersection",),
    )


def part_count(graph: PartGraphWitness) -> float:
    return float(len(graph.parts))


def contact_count(graph: PartGraphWitness) -> float:
    return float(len(graph.contacts))


def intersection_count(graph: PartGraphWitness) -> float:
    return float(sum(1 for c in graph.contacts if c.relation == "intersection"))


def reflection_symmetry(obj: ObjectMask) -> SymmetryWitness:
    score = symmetry_residual(obj)
    return SymmetryWitness(
        source_id=obj.object_id,
        kind="reflection",
        order=2,
        residual=score,
        confidence=max(0.0, 1.0 - score),
        provenance=("reflection_symmetry",),
    )


def rotational_symmetry_order(obj: ObjectMask) -> SymmetryWitness:
    score = symmetry_residual(obj)
    order = 4 if score < 0.08 else 2 if score < 0.18 else 1
    return SymmetryWitness(
        source_id=obj.object_id,
        kind="rotation",
        order=order,
        residual=score,
        confidence=max(0.0, 1.0 - score),
        provenance=("rotational_symmetry_order",),
    )


def symmetry_residual(obj: ObjectMask) -> float:
    box = _bbox(obj.mask)
    if box is None:
        return 1.0
    y0, x0, y1, x1 = box
    crop = obj.mask[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0:
        return 1.0
    residuals = [
        np.mean(crop != np.fliplr(crop)),
        np.mean(crop != np.flipud(crop)),
        np.mean(crop != np.rot90(crop, 2)),
    ]
    return float(min(residuals))


def detect_radial_arrangement(graph: PartGraphWitness) -> RadialArrangementWitness:
    if len(graph.parts) < 3:
        raise WitnessAbsent(
            "fewer_than_three_parts",
            "need at least three parts for a radial arrangement")
    part_ids = {part.part_id for part in graph.parts}
    incident_ids = {
        part_id
        for edge in graph.adjacency
        for part_id in edge
        if part_id in part_ids
    }
    contact_points = [
        point
        for contact in graph.contacts
        for point in contact.points
    ]
    if not contact_points or incident_ids != part_ids:
        raise WitnessAbsent(
            "no_shared_hub",
            "radial parts must all participate in a shared contact hub")
    centers = [_part_centroid(p) for p in graph.parts]
    cx = sum(point.x for point in contact_points) / len(contact_points)
    cy = sum(point.y for point in contact_points) / len(contact_points)
    radii = [math.hypot(p.x - cx, p.y - cy) for p in centers]
    mean_r = sum(radii) / len(radii)
    hub_spread = max(
        math.hypot(point.x - cx, point.y - cy)
        for point in contact_points
    ) / max(mean_r, np.finfo(float).tiny)
    if mean_r <= np.finfo(float).tiny or hub_spread > 0.15:
        raise WitnessAbsent(
            "no_shared_hub",
            "contact junctions do not identify one localized radial hub")
    angles = sorted(math.atan2(p.y - cy, p.x - cx) for p in centers)
    gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    gaps.append(2 * math.pi - (angles[-1] - angles[0]))
    mean_gap = sum(gaps) / len(gaps)
    gap_var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    uniformity = max(0.0, 1.0 - math.sqrt(gap_var) / max(mean_gap, 1e-9))
    radius_var = sum((r - mean_r) ** 2 for r in radii) / len(radii)
    evenness = max(
        0.0,
        1.0 - math.sqrt(radius_var) / max(mean_r, np.finfo(float).tiny),
    )
    confidence = min(uniformity, evenness)
    if confidence < 0.72:
        raise WitnessAbsent(
            "poor_radial_fit",
            f"radial angular/radius confidence {confidence:.4g} is below 0.72")
    return RadialArrangementWitness(
        center=PointWitness(x=cx, y=cy, source_id="radial_center"),
        parts=graph.parts,
        part_count=len(graph.parts),
        # Equal angular/radial placement does not prove that the part shapes
        # themselves are interchangeable under rotation.  A future symmetry
        # leg may establish a nonzero order; this detector does not fabricate
        # one.
        symmetry_order=0,
        confidence=confidence,
        residual=1.0 - confidence,
        provenance=graph.provenance + ("detect_radial_arrangement",),
    )


def pair_parts_by_symmetry(graph: PartGraphWitness) -> PartGraphWitness:
    return graph


def prototype_bird_like(graph: PartGraphWitness) -> PrototypeWitness:
    """Kept only as a template for future promoted macros; not registered."""
    roles = {}
    for idx, role in enumerate(("body", "left_appendage", "right_appendage")):
        if idx < len(graph.parts):
            roles[role] = graph.parts[idx].part_id
    missing = tuple(r for r in ("body", "left_appendage", "right_appendage") if r not in roles)
    if missing:
        raise ValueError(f"missing bird-like roles: {','.join(missing)}")
    return PrototypeWitness(
        prototype_name="bird_like",
        roles=roles,
        required_roles=("body", "left_appendage", "right_appendage"),
        confidence=0.6,
        provenance=graph.provenance + ("prototype_bird_like",),
    )


def witness_confidence(witness: Witness) -> float:
    return float(witness.confidence)


def witness_residual(witness: Witness) -> float:
    return float(witness.residual)


def radial_part_count(witness: RadialArrangementWitness) -> float:
    return float(witness.part_count)


def symmetry_order_score(witness: SymmetryWitness) -> float:
    return float(witness.order)


def _reg(name: str, domain: tuple[str, ...], codomain: str, fn: Callable,
         complexity: int = 1, invariances: tuple[str, ...] = (),
         equivariances: tuple[str, ...] = (), failure_modes: tuple[str, ...] = (),
         indeterminate_modes: tuple[str, ...] = (),
         proxy_for: tuple[str, ...] = (), measurement_kind: str | None = None,
         proxy_directions: tuple[tuple[str, str], ...] = ()) -> LegContract:
    if codomain == "Measurement" and measurement_kind is None:
        measurement_kind = "continuous"
    return LegContract(
        name=name,
        domain=domain,
        codomain=codomain,
        implementation=fn,
        complexity=complexity,
        invariances=frozenset(invariances),
        equivariances=frozenset(equivariances),
        failure_modes=failure_modes,
        indeterminate_modes=indeterminate_modes,
        proxy_for=proxy_for,
        measurement_kind=measurement_kind,
        proxy_directions=proxy_directions,
    )


def default_registry() -> LegRegistry:
    reg = LegRegistry()
    scalar_inv = ("translation", "uniform_scale")
    vector_equiv = ("translation", "uniform_scale")
    angular_actions = (
        "translation", "uniform_scale", "rotation", "reflection")
    # Raster feature thresholds are not scale-normalized yet.  Structural
    # extractors therefore advertise only the exact translation action.
    raster_equiv = ("translation",)
    for contract in (
        # Binarization remains an internal preprocessing helper until a typed
        # downstream consumer makes BinaryPanel load-bearing.
        _reg("parse_scene", ("Panel",), "Scene", parse_scene, 4,
             equivariances=raster_equiv,
             proxy_for=("connected component",)),
        _reg("extract_connected_components", ("Panel",), "Scene",
             extract_connected_components, 4, equivariances=raster_equiv,
             proxy_for=("connected component",)),
        _reg("extract_contours", ("Object",), "ContourWitness",
             extract_contours, 4, equivariances=raster_equiv,
             failure_modes=("empty_component", "not_simple_curve"),
             proxy_for=("curve", "contour", "stroke", "boundary",
                        "outline", "path")),
        _reg("contour_closedness", ("ContourWitness",), "Measurement",
             contour_closedness, 1, scalar_inv,
             proxy_for=("open", "closed", "closure", "openness",
                        "closedness", "loop"),
             measurement_kind="binary",
             proxy_directions=(("open", "low"), ("openness", "low"),
                               ("closed", "high"), ("closure", "high"),
                               ("closedness", "high"), ("loop", "high"))),
        _reg("skeletonize_component", ("Object",), "SkeletonGraphWitness",
             skeletonize_component, 3, equivariances=raster_equiv,
             proxy_for=("skeleton", "path", "graph")),
        _reg("build_skeleton_graph", ("Object",), "SkeletonGraphWitness",
             build_skeleton_graph, 3, equivariances=raster_equiv,
             proxy_for=("skeleton", "path", "graph")),
        _reg("endpoint_count", ("SkeletonGraphWitness",), "Measurement",
             endpoint_count, 1, scalar_inv,
             proxy_for=("endpoint", "end", "tip"),
             measurement_kind="count"),
        _reg("branch_count", ("SkeletonGraphWitness",), "Measurement",
             branch_count, 1, scalar_inv,
             proxy_for=("branch", "junction"), measurement_kind="count"),
        _reg("cycle_count", ("SkeletonGraphWitness",), "Measurement",
             cycle_count, 1, scalar_inv,
             proxy_for=("cycle", "loop"), measurement_kind="count"),
        # `estimate_tangents` currently transports only contour points; the
        # witness has no tangent field.  Do not expose the semantic name until
        # actual tangent vectors and residual checks are represented.
        # The current curve partitioner does not yet classify each produced
        # part as an arc or line.  Keep it internal until its name is backed by
        # explicit typed segment witnesses.
        _reg("fit_line_segment", ("ContourWitness",), "LineSegmentWitness",
             fit_line_segment, 2, equivariances=vector_equiv,
             failure_modes=("not_enough_points", "closed_contour",
                            "degenerate_segment", "high_residual"),
             proxy_for=("line", "segment", "straight")),
        _reg("line_residual", ("LineSegmentWitness",), "Measurement",
             witness_residual, 1, scalar_inv,
             proxy_for=("line", "straight", "residual"),
             proxy_directions=(("straight", "low"),)),
        _reg("line_length", ("LineSegmentWitness",), "Measurement",
             line_length, 1, ("translation",),
             proxy_for=("line", "segment", "length")),
        _reg("fit_arc", ("ContourWitness",), "ArcWitness", fit_arc, 3,
             equivariances=vector_equiv,
             failure_modes=("not_enough_points", "closed_contour",
                            "degenerate_fit", "high_residual",
                            "insufficient_angular_support",
                            "direction_reversal"),
             proxy_for=("arc", "curved", "smooth")),
        _reg("arc_angle_degrees", ("ArcWitness",), "Measurement",
             arc_angle_degrees, 1, scalar_inv,
             proxy_for=("angle", "sweep")),
        _reg("arc_residual", ("ArcWitness",), "Measurement",
             witness_residual, 1, scalar_inv, proxy_for=("smooth", "arc"),
             proxy_directions=(("smooth", "low"),)),
        _reg("count_inflections", ("ContourWitness",), "Measurement",
             count_inflections, 2, scalar_inv,
             proxy_for=("inflection", "turning reversal"),
             measurement_kind="count"),
        _reg("count_curve_parts", ("ContourWitness",), "Measurement",
             count_curve_parts, 3, scalar_inv,
             proxy_for=("curve part", "concavity", "notch"),
             measurement_kind="count"),
        _reg("fit_circle", ("ContourWitness",), "CircleWitness", fit_circle,
             4, equivariances=vector_equiv,
             failure_modes=("not_enough_points", "open_contour",
                            "high_residual")),
        # Raster rescaling changes stroke sampling enough to perturb the fitted
        # residual materially.  Translation is tested; uniform-scale
        # equivariance remains future work and is not advertised as an
        # invariant.
        _reg("fit_multiple_circles", ("Scene",), "CirclePairWitness",
             fit_multiple_circles, 8, equivariances=raster_equiv,
             failure_modes=("fewer_than_two_candidates", "high_residual")),
        _reg("detect_corners", ("ContourWitness",), "PolygonWitness",
             detect_corners, 4, equivariances=vector_equiv,
             failure_modes=("open_contour", "too_few_sides",
                            "high_residual")),
        # `decompose_into_line_segments` remains an internal compatibility
        # helper.  It currently aliases polygon fitting and does not construct
        # or validate explicit LineSegmentWitness values, so exposing its name
        # would let prose claim line segments without line evidence.
        _reg("fit_polygon", ("ContourWitness",), "PolygonWitness",
             fit_polygon, 5, equivariances=vector_equiv,
             failure_modes=("open_contour", "too_few_sides",
                            "high_residual"),
             proxy_for=("polygon", "corner", "side", "sides", "angular",
                        "vertex", "vertices", "bend")),
        _reg("polygon_side_count", ("PolygonWitness",), "Measurement",
             polygon_side_count, 1, scalar_inv,
             proxy_for=("side", "sides"), measurement_kind="count"),
        _reg("polygon_fit_residual", ("PolygonWitness",), "Measurement",
             witness_residual, 1, scalar_inv),
        _reg("classify_triangle", ("PolygonWitness",), "TriangleWitness",
             classify_triangle, 2, equivariances=vector_equiv,
             failure_modes=("wrong_side_count",)),
        _reg("classify_quadrilateral", ("PolygonWitness",),
             "QuadrilateralWitness", classify_quadrilateral, 2,
             equivariances=vector_equiv,
             failure_modes=("wrong_side_count",)),
        _reg("decompose_component_into_parts", ("Object",),
             "PartGraphWitness", decompose_component_into_parts, 5,
             equivariances=raster_equiv,
             indeterminate_modes=("not_simple_curve",)),
        _reg("build_part_graph", ("Scene",), "PartGraphWitness",
             build_part_graph, 5, equivariances=raster_equiv,
             indeterminate_modes=("not_simple_curve",)),
        _reg("build_object_part_graph", ("Object",), "PartGraphWitness",
             build_part_graph, 5, equivariances=raster_equiv,
             indeterminate_modes=("not_simple_curve",)),
        _reg("extract_point_contact_signature", ("Panel",),
             "PointContactSignature", extract_point_contact_signature, 8,
             equivariances=raster_equiv,
             failure_modes=("no_point_contact_signature",),
             indeterminate_modes=("point_contact_fit_indeterminate",),
             proxy_for=(
                 "point contact signature", "point contact",
                 "two loops touching", "touching loops",
                 "four incident rays", "noninterleaving contact",
                 "contact geometry")),
        _reg("point_contact_small_exterior_gap_degrees",
             ("PointContactSignature",), "Measurement",
             point_contact_small_exterior_gap_degrees, 1, angular_actions,
             proxy_for=(
                 "small exterior gap", "narrow exterior gap",
                 "small contact angle", "acute contact gap"),
             proxy_directions=(
                 ("small exterior gap", "low"),
                 ("narrow exterior gap", "low"),
                 ("small contact angle", "low"),
                 ("acute contact gap", "low"))),
        _reg("point_contact_large_exterior_gap_degrees",
             ("PointContactSignature",), "Measurement",
             point_contact_large_exterior_gap_degrees, 1, angular_actions,
             proxy_for=(
                 "large exterior gap", "wide exterior gap",
                 "large contact angle", "obtuse contact gap"),
             proxy_directions=(
                 ("large exterior gap", "high"),
                 ("wide exterior gap", "high"),
                 ("large contact angle", "high"),
                 ("obtuse contact gap", "high"))),
        _reg("point_contact_exterior_gap_ratio",
             ("PointContactSignature",), "Measurement",
             point_contact_exterior_gap_ratio, 2, angular_actions,
             proxy_for=(
                 "exterior gap ratio", "contact gap ratio",
                 "asymmetric contact angles", "contact angle asymmetry"),
             proxy_directions=(
                 ("exterior gap ratio", "high"),
                 ("contact gap ratio", "high"),
                 ("asymmetric contact angles", "high"),
                 ("contact angle asymmetry", "high"))),
        _reg("point_contact_gap_ratio_lower_bound",
             ("PointContactSignature",), "Measurement",
             point_contact_gap_ratio_lower_bound, 2, angular_actions,
             proxy_for=(
                 "conservative exterior gap ratio",
                 "certified contact angle asymmetry"),
             proxy_directions=(
                 ("conservative exterior gap ratio", "high"),
                 ("certified contact angle asymmetry", "high"))),
        _reg("minimum_incident_angle", ("PartGraphWitness",),
             "AngleWitness", minimum_incident_angle, 5,
             equivariances=angular_actions,
             failure_modes=("no_junction",),
             indeterminate_modes=(
                 "insufficient_incident_rays", "high_residual"),
             proxy_for=(
                 "angle", "junction angle", "incident angle", "local angle",
                 "turn angle", "acute angle")),
        _reg("angle_degrees", ("AngleWitness",), "Measurement",
             angle_degrees, 1, angular_actions,
             proxy_for=(
                 "angle", "angle degrees", "junction angle",
                 "incident angle", "local angle", "turn angle",
                 "acute angle")),
        _reg("detect_attachment", ("PartGraphWitness",), "ContactWitness",
             detect_attachment, 3, equivariances=vector_equiv,
             failure_modes=("no_attachment",),
             proxy_for=("attachment", "attached", "joined")),
        _reg("detect_contact", ("PartGraphWitness",), "ContactWitness",
             detect_contact, 3, equivariances=vector_equiv,
             failure_modes=("no_contact",),
             proxy_for=("contact", "touching", "touch")),
        _reg("detect_intersection", ("PartGraphWitness",),
             "IntersectionWitness", detect_intersection, 3,
             equivariances=vector_equiv, failure_modes=("no_crossing",),
             proxy_for=("intersect", "intersecting", "intersection",
                        "crossing", "cross")),
        _reg("circle_pair_intersection", ("CirclePairWitness",),
             "CircleIntersectionWitness", circle_pair_intersection, 3,
             equivariances=vector_equiv, failure_modes=("no_intersection",),
             proxy_for=("intersect", "intersecting", "intersection")),
        _reg("part_count", ("PartGraphWitness",), "Measurement", part_count,
             1, scalar_inv, proxy_for=("part", "parts"),
             measurement_kind="count"),
        _reg("contact_count", ("PartGraphWitness",), "Measurement",
             contact_count, 1, scalar_inv,
             proxy_for=("contact", "junction"), measurement_kind="count"),
        _reg("intersection_count", ("PartGraphWitness",), "Measurement",
             intersection_count, 1, scalar_inv,
             proxy_for=("intersection", "crossing"),
             measurement_kind="count"),
        _reg("detect_radial_arrangement", ("PartGraphWitness",),
             "RadialArrangementWitness", detect_radial_arrangement, 5,
             equivariances=vector_equiv,
             failure_modes=("fewer_than_three_parts", "no_shared_hub",
                            "poor_radial_fit")),
        _reg("select_all_objects", ("Scene",), "Scene", select_all_objects, 1,
             equivariances=vector_equiv),
        _reg("select_principal_objects", ("Scene",), "Scene",
             select_principal_objects, 1, equivariances=vector_equiv),
        _reg("select_largest", ("Scene",), "Object", select_largest, 1,
             equivariances=vector_equiv, failure_modes=("no_objects",)),
        _reg("select_largest_object", ("Scene",), "Object",
             select_largest_object, 1, equivariances=vector_equiv,
             failure_modes=("no_objects",)),
        _reg("select_smallest_object", ("Scene",), "Object",
             select_smallest_object, 1, equivariances=vector_equiv,
             failure_modes=("no_objects",)),
        _reg("select_parts", ("PartGraphWitness",), "PartGraphWitness",
             select_parts, 1, equivariances=vector_equiv),
        _reg("select_largest_part", ("PartGraphWitness",), "PartWitness",
             select_largest_part, 1, equivariances=vector_equiv,
             failure_modes=("no_parts",)),
        _reg("object_count", ("Scene",), "Measurement", object_count, 1,
             scalar_inv,
             proxy_for=("count", "component", "object", "objects"),
             measurement_kind="count"),
        _reg("total_ink", ("Panel",), "Measurement", total_ink, 1,
             ("translation",),
             proxy_for=("ink",), measurement_kind="count"),
        _reg("largest_ink", ("Scene",), "Measurement", largest_ink, 1,
             ("translation",),
             proxy_for=("ink",), measurement_kind="count"),
        _reg("bbox_aspect", ("Object",), "Measurement", bbox_aspect, 1,
             ("translation", "uniform_scale", "reflection"),
             proxy_for=("aspect",)),
        _reg("elongation", ("Object",), "Measurement", elongation, 2,
             ("translation", "reflection"),
             proxy_for=("elongation", "aspect")),
        _reg("bbox_occupancy", ("Object",), "Measurement", bbox_occupancy, 1,
             ("translation", "uniform_scale", "reflection"),
             proxy_for=("occupancy", "density")),
        # `closure_ratio` is endpoint density (and therefore length-confounded),
        # not evidence of closure.  Keep the helper internal; the honest
        # contour_closedness leg is the public open/closed path.
        _reg("witness_confidence", ("TriangleWitness",), "Measurement",
             witness_confidence, 1, scalar_inv),
        _reg("quadrilateral_confidence", ("QuadrilateralWitness",),
             "Measurement", witness_confidence, 1, scalar_inv),
        _reg("circle_residual", ("CircleWitness",), "Measurement",
             witness_residual, 1, scalar_inv),
        _reg("contact_confidence", ("ContactWitness",), "Measurement",
             witness_confidence, 1, scalar_inv),
        _reg("intersection_confidence", ("IntersectionWitness",),
             "Measurement", witness_confidence, 1, scalar_inv),
        _reg("circle_intersection_confidence",
             ("CircleIntersectionWitness",), "Measurement",
             witness_confidence, 1, scalar_inv),
        _reg("radial_part_count", ("RadialArrangementWitness",),
             "Measurement", radial_part_count, 1, scalar_inv,
             proxy_for=("part", "parts"), measurement_kind="count"),
        _reg("radial_uniformity", ("RadialArrangementWitness",),
             "Measurement", witness_confidence, 1, scalar_inv),
    ):
        reg.register(contract)
    return reg


def register_soft_semantic_legs(registry: LegRegistry) -> LegRegistry:
    """Opt a registry into deterministic soft geometry and fuzzy operators.

    The Phase-D default basis remains unchanged until this extension is
    explicitly selected and preregistered.  In particular, this function does
    not register the quarantined open-world ``prototype_bird_like`` helper.
    """
    if not isinstance(registry, LegRegistry):
        raise TypeError("soft semantic legs require a LegRegistry")
    angular_inv = (
        "translation", "uniform_scale", "rotation", "reflection")
    contracts = (
        _reg("angle_between_segments",
             ("LineSegmentWitness", "LineSegmentWitness"), "AngleWitness",
             angle_between_segments, 3, equivariances=angular_inv,
             failure_modes=("degenerate_segment", "segments_do_not_meet"),
             proxy_for=("angle", "interior angle", "segment angle")),
        _reg("angle_noncardinality_degrees", ("AngleWitness",),
             "Measurement", angle_noncardinality_degrees, 1, angular_inv,
             proxy_for=("angle", "non-cardinal angle", "oblique angle"),
             proxy_directions=(("non-cardinal angle", "high"),
                               ("oblique angle", "high"))),
        _reg("angle_obliqueness_evidence", ("AngleWitness",),
             "SoftResult", angle_obliqueness_evidence, 2, angular_inv,
             proxy_for=("angle", "non-cardinal angle", "oblique angle")),
        _reg("angle_obliqueness_membership", ("AngleWitness",),
             "Measurement", angle_obliqueness_membership, 2, angular_inv,
             proxy_for=("angle", "non-cardinal angle", "oblique angle",
                        "obliqueness membership"),
             proxy_directions=(("non-cardinal angle", "high"),
                               ("oblique angle", "high"),
                               ("obliqueness membership", "high"))),
        _reg("soft_fuzzy_min", ("SoftResult", "SoftResult"),
             "SoftResult", fuzzy_min, 1),
        _reg("soft_fuzzy_max", ("SoftResult", "SoftResult"),
             "SoftResult", fuzzy_max, 1),
        _reg("soft_fuzzy_not", ("SoftResult",),
             "SoftResult", fuzzy_not, 1),
        _reg("soft_pair", ("SoftResult", "SoftResult"),
             "SoftEvidenceSet", soft_pair, 1),
        _reg("soft_add", ("SoftEvidenceSet", "SoftResult"),
             "SoftEvidenceSet", soft_add, 1),
        _reg("soft_all", ("SoftEvidenceSet",),
             "SoftResult", fuzzy_all, 1),
        _reg("soft_any", ("SoftEvidenceSet",),
             "SoftResult", fuzzy_any, 1),
        _reg("soft_mean", ("SoftEvidenceSet",),
             "SoftResult", fuzzy_mean, 1),
        _reg("soft_membership_value", ("SoftResult",),
             "Measurement", soft_membership_value, 1,
             failure_modes=("soft_evidence_absent",),
             proxy_for=("membership", "similarity")),
    )
    existing = set(registry.names())
    overlap = sorted(contract.name for contract in contracts
                     if contract.name in existing)
    if overlap:
        raise ValueError(
            "soft semantic legs are already registered: "
            + ", ".join(overlap))
    for contract in contracts:
        registry.register(contract)
    return registry


def soft_semantic_registry() -> LegRegistry:
    """Return the default hard basis plus the opt-in soft extension."""
    return register_soft_semantic_legs(default_registry())
