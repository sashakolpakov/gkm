"""Pure synthetic connected line/arc fixtures and exact raster-cover targets.

The module owns no filesystem or official-data input.  It renders a bounded
catalog of connected paths, issues only those in-memory PNG bytes, and derives
its target from the PNG plus the same frozen primitive-mask inventory.  Hidden
program history is deliberately absent from :func:`exact_cover_target`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from typing import Final, Literal

import numpy as np

from bongard import panel_action_count_synthetic_identifiability as base
from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

IMAGE_SIZE: Final = 64
PIXEL_COUNT: Final = IMAGE_SIZE * IMAGE_SIZE
MAX_PRIMITIVES: Final = 9
_Kind = Literal["line", "arc"]
_Layout = Literal["single_shape", "two_shape"]

_FAMILY_VALUES: Final = (
    "lattice", "perimeter", "pinwheel", "radial", "staggered"
)
_NUISANCE_VALUES: Final = (
    ("identity", 2, 1000),
    ("r90", 3, 1000),
)
_PAIR_VALUES: Final = tuple(
    (straight, arc)
    for straight in range(10)
    for arc in range(10)
    if 1 <= straight + arc <= MAX_PRIMITIVES
)

_SINGLE_Y: Final = {
    "lattice": (220, 220, 340, 340, 460, 460, 580, 580, 700, 700),
    "perimeter": (160, 280, 160, 400, 240, 560, 320, 720, 480, 800),
    "pinwheel": (500, 300, 520, 260, 540, 220, 560, 180, 580, 140),
    "radial": (760, 580, 720, 500, 640, 420, 560, 340, 480, 260),
    "staggered": (180, 420, 240, 520, 300, 620, 360, 720, 420, 820),
}
_TWO_Y: Final = {
    "lattice": ((120, 120, 200, 200, 300, 300), (700, 700, 780, 780, 860, 860)),
    "perimeter": ((100, 220, 120, 300, 160, 380), (620, 760, 640, 840, 680, 900)),
    "pinwheel": ((300, 120, 320, 100, 340, 80), (820, 640, 840, 620, 860, 600)),
    "radial": ((360, 220, 340, 180, 300, 120), (900, 760, 860, 700, 820, 640)),
    "staggered": ((100, 300, 140, 340, 180, 380), (620, 840, 660, 880, 700, 900)),
}

_ISSUED_PNGS: dict[str, bytes] = {}


class ConnectedSyntheticError(ValueError):
    """A connected fixture, transport, or exact-cover invariant differs."""


def source_sha256() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _exact_text(value: object, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise TypeError(f"{name} must be a canonical exact string")
    return value


def _pixels(value: object, name: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or any(type(item) is not int for item in value)
        or value != tuple(sorted(set(value)))
        or (nonempty and not value)
        or (value and (value[0] < 0 or value[-1] >= PIXEL_COUNT))
    ):
        raise TypeError(f"{name} differs")
    return value


def _points(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if (
        type(value) is not tuple
        or any(
            type(point) is not tuple
            or len(point) != 2
            or any(type(coordinate) is not int for coordinate in point)
            or any(not 0 <= coordinate < IMAGE_SIZE for coordinate in point)
            for point in value
        )
        or value != tuple(sorted(set(value)))
    ):
        raise TypeError(f"{name} differs")
    return value


@dataclass(frozen=True, order=True, slots=True)
class CountPair:
    straight: int
    arc: int

    def __post_init__(self) -> None:
        if (
            type(self.straight) is not int
            or type(self.arc) is not int
            or self.straight < 0
            or self.arc < 0
            or not 1 <= self.straight + self.arc <= MAX_PRIMITIVES
        ):
            raise ValueError("count pair leaves the 54-cell universe")

    def as_tuple(self) -> tuple[int, int]:
        return self.straight, self.arc


@dataclass(frozen=True, slots=True)
class ConnectedNuisance:
    d4: str
    stroke_width: int
    scale_milli: int

    def __post_init__(self) -> None:
        if (
            type(self.d4) is not str
            or self.d4 not in base.D4_NAMES
            or type(self.stroke_width) is not int
            or type(self.scale_milli) is not int
            or not 1 <= self.stroke_width <= 4
            or not 750 <= self.scale_milli <= 1200
        ):
            raise ValueError("connected nuisance differs")

    @property
    def identity(self) -> str:
        return f"{self.d4}-w{self.stroke_width}-s{self.scale_milli}"

    @property
    def nuisance_id(self) -> str:
        return self.identity

    def as_base(self) -> base.Nuisance:
        return base.Nuisance(self.d4, self.stroke_width, self.scale_milli)


@dataclass(frozen=True, slots=True)
class CatalogPrimitive:
    primitive_id: str
    kind: _Kind
    ink_pixels: tuple[int, ...]
    endpoints_yx: tuple[tuple[int, int], ...]
    boundary_pixels: tuple[int, ...]

    def __post_init__(self) -> None:
        _exact_text(self.primitive_id, "primitive_id")
        if type(self.kind) is not str or self.kind not in ("line", "arc"):
            raise ValueError("catalog primitive kind differs")
        _pixels(self.ink_pixels, "catalog ink", nonempty=True)
        endpoints = _points(self.endpoints_yx, "catalog endpoints")
        if len(endpoints) != 2:
            raise ValueError("catalog primitive must have two endpoints")
        boundary = _pixels(self.boundary_pixels, "catalog boundary", nonempty=True)
        if not set(boundary).issubset(self.ink_pixels):
            raise ValueError("catalog boundary leaves primitive ink")


@dataclass(frozen=True, slots=True)
class ShapeProgram:
    shape_id: str
    primitive_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _exact_text(self.shape_id, "shape_id")
        if (
            type(self.primitive_ids) is not tuple
            or not self.primitive_ids
            or any(
                type(item) is not str
                or not item
                or item.strip() != item
                for item in self.primitive_ids
            )
            or len(self.primitive_ids) != len(set(self.primitive_ids))
        ):
            raise ValueError("shape primitive inventory differs")


@dataclass(frozen=True, slots=True)
class PanelProgram:
    carrier_family: str
    layout: _Layout
    shapes: tuple[ShapeProgram, ...]

    def __post_init__(self) -> None:
        if self.carrier_family not in _FAMILY_VALUES or type(self.carrier_family) is not str:
            raise ValueError("panel carrier differs")
        if type(self.layout) is not str or self.layout not in ("single_shape", "two_shape"):
            raise ValueError("panel layout differs")
        if (
            type(self.shapes) is not tuple
            or len(self.shapes) != (1 if self.layout == "single_shape" else 2)
            or any(type(item) is not ShapeProgram for item in self.shapes)
        ):
            raise ValueError("panel shape inventory differs")
        for shape in self.shapes:
            ShapeProgram.__post_init__(shape)
        if len({shape.shape_id for shape in self.shapes}) != len(self.shapes):
            raise ValueError("panel shape identifiers repeat")
        ids = tuple(item for shape in self.shapes for item in shape.primitive_ids)
        if not 1 <= len(ids) <= MAX_PRIMITIVES or len(ids) != len(set(ids)):
            raise ValueError("panel primitive inventory differs")


@dataclass(frozen=True, slots=True)
class BoundaryTruth:
    left_primitive_id: str
    right_primitive_id: str
    kind: Literal["AA", "AL", "LA", "LL"]
    adjacent: bool
    touching_pixels: tuple[int, ...]

    def __post_init__(self) -> None:
        _exact_text(self.left_primitive_id, "left boundary primitive")
        _exact_text(self.right_primitive_id, "right boundary primitive")
        if self.left_primitive_id == self.right_primitive_id:
            raise ValueError("boundary primitive identifiers repeat")
        if type(self.kind) is not str or self.kind not in ("AA", "AL", "LA", "LL"):
            raise ValueError("boundary kind differs")
        if type(self.adjacent) is not bool or self.adjacent is not True:
            raise ValueError("boundary must be an actual within-shape adjacency")
        touching = _pixels(self.touching_pixels, "touching pixels", nonempty=True)
        try:
            catalog = _catalog_by_id()
            left = catalog[self.left_primitive_id]
            right = catalog[self.right_primitive_id]
        except KeyError as exc:
            raise ValueError("boundary leaves the fixed primitive catalog") from exc
        expected_kind = left.kind[0].upper() + right.kind[0].upper()
        if self.kind != expected_kind:
            raise ValueError("boundary kind differs from catalog primitives")
        if touching != _expected_touching_pixels(left, right):
            raise ValueError("boundary evidence differs from exact raster contact")

    @property
    def boundary_kind(self) -> str:
        return self.kind


@dataclass(frozen=True, slots=True)
class ConnectedSyntheticSample:
    sample_id: str
    panel_program: PanelProgram
    png_bytes: bytes
    raster_digest: str
    nuisance: ConnectedNuisance
    declared_pair: CountPair
    layout_truth: _Layout
    shape_truth: tuple[CountPair, ...]
    boundary_truth: tuple[BoundaryTruth, ...]

    def __post_init__(self) -> None:
        _exact_text(self.sample_id, "sample_id")
        if type(self.panel_program) is not PanelProgram:
            raise TypeError("sample panel program differs")
        PanelProgram.__post_init__(self.panel_program)
        if type(self.png_bytes) is not bytes or not self.png_bytes:
            raise TypeError("sample PNG differs")
        digest = "sha256:" + hashlib.sha256(self.png_bytes).hexdigest()
        if type(self.raster_digest) is not str or self.raster_digest != digest:
            raise ValueError("sample PNG digest differs")
        if _ISSUED_PNGS.get(digest) != self.png_bytes:
            raise ValueError("sample PNG was not issued by the connected renderer")
        if type(self.nuisance) is not ConnectedNuisance:
            raise TypeError("sample nuisance differs")
        ConnectedNuisance.__post_init__(self.nuisance)
        if type(self.declared_pair) is not CountPair:
            raise TypeError("sample count pair differs")
        CountPair.__post_init__(self.declared_pair)
        if self.layout_truth != self.panel_program.layout or type(self.layout_truth) is not str:
            raise ValueError("layout truth differs")
        if (
            type(self.shape_truth) is not tuple
            or len(self.shape_truth) != len(self.panel_program.shapes)
            or any(type(item) is not CountPair for item in self.shape_truth)
        ):
            raise ValueError("shape truth differs")
        for pair in self.shape_truth:
            CountPair.__post_init__(pair)
        aggregate = (
            sum(pair.straight for pair in self.shape_truth),
            sum(pair.arc for pair in self.shape_truth),
        )
        if aggregate != self.declared_pair.as_tuple():
            raise ValueError("shape truths do not aggregate to panel truth")
        if (
            type(self.boundary_truth) is not tuple
            or any(type(item) is not BoundaryTruth for item in self.boundary_truth)
        ):
            raise ValueError("boundary truth differs")
        for row in self.boundary_truth:
            BoundaryTruth.__post_init__(row)
        primitive_ids = tuple(
            primitive_id
            for shape in self.panel_program.shapes
            for primitive_id in shape.primitive_ids
        )
        try:
            catalog = _catalog_by_id()
            kinds = tuple(catalog[item].kind for item in primitive_ids)
        except KeyError as exc:
            raise ValueError("sample program leaves the fixed catalog") from exc
        expected_pair = (
            sum(kind == "line" for kind in kinds),
            sum(kind == "arc" for kind in kinds),
        )
        if expected_pair != self.declared_pair.as_tuple():
            raise ValueError("sample truth differs from catalog program")
        expected_shape_truth = tuple(
            CountPair(
                sum(catalog[item].kind == "line" for item in shape.primitive_ids),
                sum(catalog[item].kind == "arc" for item in shape.primitive_ids),
            )
            for shape in self.panel_program.shapes
        )
        if self.shape_truth != expected_shape_truth:
            raise ValueError("shape truth differs from catalog program")
        expected_boundaries = tuple(
            row
            for shape in self.panel_program.shapes
            for row in _boundary_rows(shape)
        )
        if self.boundary_truth != expected_boundaries:
            raise ValueError("boundary truth differs from catalog program")
        if self.png_bytes != _png_for_ids(primitive_ids):
            raise ValueError("sample PNG differs from catalog program")
        try:
            nuisance_index = _NUISANCE_VALUES.index(
                (
                    self.nuisance.d4,
                    self.nuisance.stroke_width,
                    self.nuisance.scale_milli,
                )
            )
        except ValueError as exc:
            raise ValueError("sample nuisance leaves the frozen inventory") from exc
        if tuple(shape.shape_id for shape in self.panel_program.shapes) != tuple(
            f"shape-{index}" for index in range(len(self.panel_program.shapes))
        ):
            raise ValueError("sample shape identifiers differ")
        expected_primitive_ids = tuple(
            f"{self.panel_program.carrier_family}.{nuisance_index}."
            f"{self.panel_program.layout}.s{shape_index}.p{slot}.{catalog_id.rsplit('.', 1)[-1]}"
            for shape_index, shape in enumerate(self.panel_program.shapes)
            for slot, catalog_id in enumerate(shape.primitive_ids)
        )
        if primitive_ids != expected_primitive_ids:
            raise ValueError("sample catalog context differs")
        expected_sample_id = (
            f"{self.panel_program.carrier_family}.{nuisance_index}."
            f"{self.panel_program.layout}.l{self.declared_pair.straight}"
            f"a{self.declared_pair.arc}"
        )
        if self.sample_id != expected_sample_id:
            raise ValueError("sample identifier differs from bound context")


@dataclass(frozen=True, slots=True)
class ExactCoverHypothesis:
    count_pair: CountPair
    primitive_ids: tuple[str, ...]
    covered_pixels: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.count_pair) is not CountPair:
            raise TypeError("cover count pair differs")
        CountPair.__post_init__(self.count_pair)
        if (
            type(self.primitive_ids) is not tuple
            or len(self.primitive_ids) != sum(self.count_pair.as_tuple())
            or self.primitive_ids != tuple(sorted(set(self.primitive_ids)))
            or any(
                type(item) is not str or not item or item.strip() != item
                for item in self.primitive_ids
            )
        ):
            raise ValueError("cover primitive inventory differs")
        covered = _pixels(self.covered_pixels, "cover pixels", nonempty=True)
        try:
            catalog = _catalog_by_id()
            rows = tuple(catalog[item] for item in self.primitive_ids)
        except KeyError as exc:
            raise ValueError("cover leaves the fixed primitive catalog") from exc
        expected_pair = (
            sum(row.kind == "line" for row in rows),
            sum(row.kind == "arc" for row in rows),
        )
        if expected_pair != self.count_pair.as_tuple():
            raise ValueError("cover pair differs from primitive kinds")
        expected_covered = tuple(
            sorted({pixel for row in rows for pixel in row.ink_pixels})
        )
        if covered != expected_covered:
            raise ValueError("cover pixels differ from primitive union")


@dataclass(frozen=True, slots=True)
class ExactCoverTarget:
    png_digest: str
    minimum_primitive_count: int
    count_pairs: tuple[CountPair, ...]
    hypotheses: tuple[ExactCoverHypothesis, ...]

    def __post_init__(self) -> None:
        if type(self.png_digest) is not str or not self.png_digest.startswith("sha256:") or len(self.png_digest) != 71:
            raise ValueError("target PNG digest differs")
        if type(self.minimum_primitive_count) is not int or not 1 <= self.minimum_primitive_count <= MAX_PRIMITIVES:
            raise ValueError("target minimum differs")
        if (
            type(self.count_pairs) is not tuple
            or not self.count_pairs
            or any(type(item) is not CountPair for item in self.count_pairs)
            or self.count_pairs != tuple(sorted(set(self.count_pairs)))
        ):
            raise ValueError("target count-pair set differs")
        if (
            type(self.hypotheses) is not tuple
            or not self.hypotheses
            or any(type(item) is not ExactCoverHypothesis for item in self.hypotheses)
        ):
            raise ValueError("target hypotheses differ")
        for row in self.hypotheses:
            ExactCoverHypothesis.__post_init__(row)
            if len(row.primitive_ids) != self.minimum_primitive_count:
                raise ValueError("target hypothesis is not minimum")
        pairs = tuple(sorted({row.count_pair for row in self.hypotheses}))
        if pairs != self.count_pairs:
            raise ValueError("target pair projection differs")
        png_bytes = _ISSUED_PNGS.get(self.png_digest)
        if png_bytes is None:
            raise ValueError("target PNG was not issued by the connected renderer")
        expected_pixels = _foreground(png_bytes)
        if any(row.covered_pixels != expected_pixels for row in self.hypotheses):
            raise ValueError("target hypotheses do not reconstruct the PNG")
        expected_minimum, expected_pairs, expected_hypotheses = (
            _expected_target_data(png_bytes)
        )
        actual_hypotheses = tuple(
            (
                row.count_pair.as_tuple(),
                row.primitive_ids,
                row.covered_pixels,
            )
            for row in self.hypotheses
        )
        if (
            self.minimum_primitive_count != expected_minimum
            or tuple(row.as_tuple() for row in self.count_pairs) != expected_pairs
            or actual_hypotheses != expected_hypotheses
        ):
            raise ValueError("target is not the exhaustive canonical minimum cover")


def connected_carrier_families() -> tuple[str, ...]:
    source_sha256()
    return tuple(_FAMILY_VALUES)


def connected_nuisances() -> tuple[ConnectedNuisance, ...]:
    source_sha256()
    return tuple(ConnectedNuisance(*row) for row in _NUISANCE_VALUES)


def _base_points(family: str, layout: _Layout, shape_index: int) -> tuple[base.Point, ...]:
    if layout == "single_shape":
        xs = tuple(range(80, 945, 96))
        ys = _SINGLE_Y[family]
    else:
        xs = tuple(range(80 + 20 * shape_index, 881 + 20 * shape_index, 160))
        ys = _TWO_Y[family][shape_index]
    return tuple(base.Point(x, y) for x, y in zip(xs, ys, strict=True))


def _kind_positions(pair: CountPair, family: str, nuisance_index: int) -> frozenset[int]:
    total = pair.straight + pair.arc
    ranked = sorted(
        range(total),
        key=lambda index: (
            (index * 5 + nuisance_index * 3 + len(family)) % 17,
            index,
        ),
    )
    return frozenset(ranked[: pair.arc])


def _action(
    primitive_id: str,
    kind: _Kind,
    start: base.Point,
    end: base.Point,
) -> base.LineAction | base.ArcAction:
    if kind == "line":
        return base.LineAction(primitive_id, start, end)
    dx = end.x - start.x
    dy = end.y - start.y
    length = math.hypot(dx, dy)
    # Curve to a consistent side of the carrier path.  The sign follows dy so
    # consecutive segments retain shared endpoint ink under the rasterizer.
    sign = 1 if dy >= 0 else -1
    through = base.Point(
        round((start.x + end.x) / 2 - sign * dy / length * 48),
        round((start.y + end.y) / 2 + sign * dx / length * 48),
    )
    return base.ArcAction(primitive_id, start, through, end)


def _boundary_pixels(mask: np.ndarray) -> tuple[int, ...]:
    padded = np.pad(mask, 1, constant_values=False)
    interior = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            interior &= padded[dy : dy + IMAGE_SIZE, dx : dx + IMAGE_SIZE]
    return tuple(int(index) for index in np.flatnonzero(mask & ~interior))


def _endpoint_pixels(action: base.LineAction | base.ArcAction, nuisance: base.Nuisance) -> tuple[tuple[int, int], ...]:
    values = []
    for point in (action.start, action.end):
        x, y = base._transform(point, nuisance)  # noqa: SLF001
        values.append((max(0, min(63, int(y // 16))), max(0, min(63, int(x // 16)))))
    return tuple(sorted(set(values)))


@lru_cache(maxsize=1)
def _catalog_and_programs() -> tuple[
    tuple[CatalogPrimitive, ...],
    dict[tuple[str, int, str, int, int, str], str],
]:
    rows: list[CatalogPrimitive] = []
    lookup: dict[tuple[str, int, str, int, int, str], str] = {}
    for family in _FAMILY_VALUES:
        for nuisance_index, nuisance_record in enumerate(connected_nuisances()):
            nuisance = nuisance_record.as_base()
            for layout in ("single_shape", "two_shape"):
                shape_count = 1 if layout == "single_shape" else 2
                slots = 9 if layout == "single_shape" else 5
                for shape_index in range(shape_count):
                    points = _base_points(family, layout, shape_index)
                    for slot in range(slots):
                        for kind in ("line", "arc"):
                            primitive_id = (
                                f"{family}.{nuisance_index}.{layout}."
                                f"s{shape_index}.p{slot}.{kind}"
                            )
                            action = _action(
                                primitive_id, kind, points[slot], points[slot + 1]
                            )
                            mask = base._action_mask(action, nuisance)  # noqa: SLF001
                            ink = tuple(int(index) for index in np.flatnonzero(mask))
                            row = CatalogPrimitive(
                                primitive_id,
                                kind,
                                ink,
                                _endpoint_pixels(action, nuisance),
                                _boundary_pixels(mask),
                            )
                            rows.append(row)
                            lookup[
                                (family, nuisance_index, layout, shape_index, slot, kind)
                            ] = primitive_id
    # A fixed target-only ambiguity witness.  Both (A,B) and (C,D) are
    # materially different two-line programs with the same exact PNG union.
    # These rows are deliberately outside the scored 1,060-panel corpus.
    stress_nuisance = base.Nuisance("identity", 2, 1000)
    for label, start_x, end_x in (
        ("a", 192, 383),
        ("b", 385, 832),
        ("c", 192, 575),
        ("d", 577, 832),
    ):
        primitive_id = f"stress.ambiguity.{label}.line"
        action = base.LineAction(
            primitive_id,
            base.Point(start_x, 512),
            base.Point(end_x, 512),
        )
        mask = base._action_mask(action, stress_nuisance)  # noqa: SLF001
        rows.append(
            CatalogPrimitive(
                primitive_id,
                "line",
                tuple(int(index) for index in np.flatnonzero(mask)),
                _endpoint_pixels(action, stress_nuisance),
                _boundary_pixels(mask),
            )
        )
    rows.sort(key=lambda row: row.primitive_id)
    return tuple(rows), lookup


def primitive_catalog() -> tuple[CatalogPrimitive, ...]:
    source_sha256()
    rows, _ = _catalog_and_programs()
    return tuple(
        CatalogPrimitive(
            row.primitive_id, row.kind, tuple(row.ink_pixels),
            tuple(row.endpoints_yx), tuple(row.boundary_pixels),
        )
        for row in rows
    )


@lru_cache(maxsize=1)
def _catalog_by_id() -> dict[str, CatalogPrimitive]:
    return {row.primitive_id: row for row in _catalog_and_programs()[0]}


def _png_for_ids(primitive_ids: tuple[str, ...]) -> bytes:
    if (
        type(primitive_ids) is not tuple
        or not primitive_ids
        or any(type(item) is not str for item in primitive_ids)
        or len(primitive_ids) != len(set(primitive_ids))
    ):
        raise ConnectedSyntheticError("primitive ID transport differs")
    catalog = _catalog_by_id()
    try:
        masks = tuple(
            np.isin(np.arange(PIXEL_COUNT), catalog[item].ink_pixels).reshape(64, 64)
            for item in primitive_ids
        )
    except KeyError as exc:
        raise ConnectedSyntheticError("unknown catalog primitive") from exc
    return base._png(np.logical_or.reduce(masks))  # noqa: SLF001


def render_catalog_program(primitive_ids: tuple[str, ...]) -> bytes:
    source_sha256()
    if type(primitive_ids) is not tuple or not 1 <= len(primitive_ids) <= MAX_PRIMITIVES:
        raise ConnectedSyntheticError("catalog programs contain one to nine primitives")
    png = _png_for_ids(primitive_ids)
    digest = "sha256:" + hashlib.sha256(png).hexdigest()
    previous = _ISSUED_PNGS.setdefault(digest, png)
    if previous != png:  # pragma: no cover - cryptographic collision guard
        raise ConnectedSyntheticError("connected PNG digest collision")
    return png


def require_issued_connected_png(png_bytes: bytes) -> str:
    source_sha256()
    if type(png_bytes) is not bytes:
        raise ConnectedSyntheticError("connected PNG payload must be exact bytes")
    digest = "sha256:" + hashlib.sha256(png_bytes).hexdigest()
    if _ISSUED_PNGS.get(digest) != png_bytes:
        raise ConnectedSyntheticError("PNG was not issued by the connected renderer")
    return digest


def d4_raster_orbit_digest(png_bytes: bytes) -> str:
    require_issued_connected_png(png_bytes)
    from io import BytesIO
    from PIL import Image

    with Image.open(BytesIO(png_bytes)) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) < 128
    transforms = []
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        transforms.extend((rotated, np.fliplr(rotated)))
    canonical = min(np.ascontiguousarray(row).tobytes() for row in transforms)
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _shape_program(
    family: str,
    nuisance_index: int,
    layout: _Layout,
    shape_index: int,
    start_slot: int,
    kinds: tuple[_Kind, ...],
) -> ShapeProgram:
    lookup = _catalog_and_programs()[1]
    ids = tuple(
        lookup[(family, nuisance_index, layout, shape_index, slot, kind)]
        for slot, kind in enumerate(kinds, start=start_slot)
    )
    return ShapeProgram(f"shape-{shape_index}", ids)


def _expected_touching_pixels(
    left: CatalogPrimitive, right: CatalogPrimitive
) -> tuple[int, ...]:
    """Return the exact overlap or left-side 8-touch raster witness."""

    intersection = tuple(sorted(set(left.ink_pixels) & set(right.ink_pixels)))
    if intersection:
        return intersection
    right_boundary = set(right.boundary_pixels)
    return tuple(
        sorted(
            left_pixel
            for left_pixel in set(left.boundary_pixels)
            if any(
                max(
                    abs(left_pixel // IMAGE_SIZE - right_pixel // IMAGE_SIZE),
                    abs(left_pixel % IMAGE_SIZE - right_pixel % IMAGE_SIZE),
                ) <= 1
                for right_pixel in right_boundary
            )
        )
    )


def _boundary_rows(shape: ShapeProgram) -> tuple[BoundaryTruth, ...]:
    catalog = _catalog_by_id()
    result = []
    for left_id, right_id in zip(shape.primitive_ids, shape.primitive_ids[1:]):
        left, right = catalog[left_id], catalog[right_id]
        touching = _expected_touching_pixels(left, right)
        if not touching:
            raise ConnectedSyntheticError(
                "adjacent primitive masks do not intersect or 8-touch"
            )
        result.append(
            BoundaryTruth(
                left_id,
                right_id,
                left.kind[0].upper() + right.kind[0].upper(),
                True,
                touching,
            )
        )
    return tuple(result)


def _sample(
    family: str,
    nuisance_index: int,
    layout: _Layout,
    pair: CountPair,
) -> ConnectedSyntheticSample:
    total = pair.straight + pair.arc
    arc_positions = _kind_positions(pair, family, nuisance_index)
    kinds: tuple[_Kind, ...] = tuple(
        "arc" if index in arc_positions else "line" for index in range(total)
    )
    if layout == "single_shape":
        shapes = (_shape_program(family, nuisance_index, layout, 0, 0, kinds),)
    else:
        first_count = (total + 1) // 2
        shapes = (
            _shape_program(family, nuisance_index, layout, 0, 0, kinds[:first_count]),
            _shape_program(family, nuisance_index, layout, 1, 0, kinds[first_count:]),
        )
    program = PanelProgram(family, layout, shapes)
    ids = tuple(item for shape in shapes for item in shape.primitive_ids)
    png = render_catalog_program(ids)
    digest = "sha256:" + hashlib.sha256(png).hexdigest()
    catalog = _catalog_by_id()
    shape_truth = tuple(
        CountPair(
            sum(catalog[item].kind == "line" for item in shape.primitive_ids),
            sum(catalog[item].kind == "arc" for item in shape.primitive_ids),
        )
        for shape in shapes
    )
    boundaries = tuple(row for shape in shapes for row in _boundary_rows(shape))
    sample = ConnectedSyntheticSample(
        f"{family}.{nuisance_index}.{layout}.l{pair.straight}a{pair.arc}",
        program,
        png,
        digest,
        connected_nuisances()[nuisance_index],
        pair,
        layout,
        shape_truth,
        boundaries,
    )
    return sample


@lru_cache(maxsize=1)
def _corpus_cache() -> tuple[ConnectedSyntheticSample, ...]:
    rows = []
    for family in _FAMILY_VALUES:
        for nuisance_index in range(len(_NUISANCE_VALUES)):
            for straight, arc in _PAIR_VALUES:
                pair = CountPair(straight, arc)
                rows.append(_sample(family, nuisance_index, "single_shape", pair))
                if straight + arc >= 2:
                    rows.append(_sample(family, nuisance_index, "two_shape", pair))
    rows.sort(key=lambda row: row.sample_id)
    if len(rows) != 1060:
        raise ConnectedSyntheticError("connected corpus cardinality differs")
    return tuple(rows)


def _sample_fingerprint(sample: ConnectedSyntheticSample) -> tuple[object, ...]:
    return (
        sample.sample_id,
        sample.png_bytes,
        sample.raster_digest,
        sample.panel_program.carrier_family,
        sample.panel_program.layout,
        tuple((shape.shape_id, tuple(shape.primitive_ids)) for shape in sample.panel_program.shapes),
        (sample.nuisance.d4, sample.nuisance.stroke_width, sample.nuisance.scale_milli),
        sample.declared_pair.as_tuple(),
        tuple(pair.as_tuple() for pair in sample.shape_truth),
        tuple(
            (
                row.left_primitive_id,
                row.right_primitive_id,
                row.kind,
                row.adjacent,
                tuple(row.touching_pixels),
            )
            for row in sample.boundary_truth
        ),
    )


def build_connected_corpus() -> tuple[ConnectedSyntheticSample, ...]:
    source_sha256()
    result = []
    for original in _corpus_cache():
        # Rebuild every nested record so callers cannot mutate cached records.
        shapes = tuple(ShapeProgram(row.shape_id, tuple(row.primitive_ids)) for row in original.panel_program.shapes)
        sample = ConnectedSyntheticSample(
            original.sample_id,
            PanelProgram(original.panel_program.carrier_family, original.panel_program.layout, shapes),
            bytes(original.png_bytes),
            original.raster_digest,
            ConnectedNuisance(original.nuisance.d4, original.nuisance.stroke_width, original.nuisance.scale_milli),
            CountPair(*original.declared_pair.as_tuple()),
            original.layout_truth,
            tuple(CountPair(*row.as_tuple()) for row in original.shape_truth),
            tuple(
                BoundaryTruth(
                    row.left_primitive_id,
                    row.right_primitive_id,
                    row.kind,
                    row.adjacent,
                    tuple(row.touching_pixels),
                )
                for row in original.boundary_truth
            ),
        )
        result.append(sample)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _MaskRow:
    primitive_id: str
    kind: _Kind
    bits: int


@lru_cache(maxsize=1)
def _distinct_masks() -> tuple[_MaskRow, ...]:
    groups: dict[tuple[str, tuple[int, ...]], list[str]] = {}
    for row in _catalog_and_programs()[0]:
        groups.setdefault((row.kind, row.ink_pixels), []).append(row.primitive_id)
    result = []
    for (kind, pixels), ids in groups.items():
        bits = 0
        for pixel in pixels:
            bits |= 1 << pixel
        result.append(_MaskRow(min(ids), kind, bits))
    return tuple(sorted(result, key=lambda row: row.primitive_id))


def _foreground(png_bytes: bytes) -> tuple[int, ...]:
    from io import BytesIO
    from PIL import Image

    with Image.open(BytesIO(png_bytes)) as image:
        if image.format != "PNG" or image.size != (64, 64) or getattr(image, "n_frames", 1) != 1:
            raise ConnectedSyntheticError("issued PNG format differs")
        mask = np.asarray(image.convert("L"), dtype=np.uint8) < 128
    pixels = tuple(int(index) for index in np.flatnonzero(mask))
    if not pixels:
        raise ConnectedSyntheticError("issued PNG has no ink")
    return pixels


def _minimum_covers(target_bits: int, candidates: tuple[_MaskRow, ...]) -> tuple[tuple[int, ...], ...]:
    covering: dict[int, tuple[int, ...]] = {}
    remaining = target_bits
    while remaining:
        low = remaining & -remaining
        pixel = low.bit_length() - 1
        values = tuple(i for i, row in enumerate(candidates) if row.bits & low)
        if not values:
            return ()
        covering[pixel] = values
        remaining ^= low

    impossible = MAX_PRIMITIVES + 1

    def pivot(covered: int) -> int:
        missing = target_bits & ~covered
        values = []
        while missing:
            low = missing & -missing
            values.append(low.bit_length() - 1)
            missing ^= low
        return min(values, key=lambda pixel: (len(covering[pixel]), pixel))

    @lru_cache(maxsize=None)
    def minimum(covered: int, slots: int) -> int:
        if covered == target_bits:
            return 0
        if slots == 0:
            return impossible
        best = impossible
        for index in covering[pivot(covered)]:
            nxt = covered | candidates[index].bits
            if nxt != covered:
                tail = minimum(nxt, slots - 1)
                if tail < impossible:
                    best = min(best, 1 + tail)
        return best

    count = minimum(0, MAX_PRIMITIVES)
    if not 1 <= count <= MAX_PRIMITIVES:
        return ()
    result: set[tuple[int, ...]] = set()

    def collect(covered: int, chosen: frozenset[int]) -> None:
        if covered == target_bits:
            if len(chosen) == count:
                result.add(tuple(sorted(chosen)))
            return
        slots = count - len(chosen)
        if slots <= 0 or minimum(covered, slots) > slots:
            return
        for index in covering[pivot(covered)]:
            if index in chosen:
                continue
            nxt = covered | candidates[index].bits
            if minimum(nxt, slots - 1) <= slots - 1:
                collect(nxt, chosen | {index})

    collect(0, frozenset())
    return tuple(sorted(result))


@lru_cache(maxsize=4096)
def _expected_target_data(
    png_bytes: bytes,
) -> tuple[
    int,
    tuple[tuple[int, int], ...],
    tuple[tuple[tuple[int, int], tuple[str, ...], tuple[int, ...]], ...],
]:
    """Return the exhaustive minimum-cover target as immutable scalars."""

    pixels = _foreground(png_bytes)
    bits = 0
    for pixel in pixels:
        bits |= 1 << pixel
    eligible = tuple(row for row in _distinct_masks() if row.bits & ~bits == 0)
    covers = _minimum_covers(bits, eligible)
    if not covers:
        raise ConnectedSyntheticError("issued PNG has no bounded exact catalog cover")
    hypotheses: list[
        tuple[tuple[int, int], tuple[str, ...], tuple[int, ...]]
    ] = []
    for cover in covers:
        rows = tuple(eligible[index] for index in cover)
        pair = (
            sum(row.kind == "line" for row in rows),
            sum(row.kind == "arc" for row in rows),
        )
        hypotheses.append(
            (
                pair,
                tuple(sorted(row.primitive_id for row in rows)),
                pixels,
            )
        )
    hypotheses.sort(key=lambda row: (row[0], row[1]))
    payload = tuple(hypotheses)
    return (
        len(payload[0][1]),
        tuple(sorted({row[0] for row in payload})),
        payload,
    )


@lru_cache(maxsize=4096)
def _target_cache(png_bytes: bytes) -> ExactCoverTarget:
    minimum, pairs, hypothesis_data = _expected_target_data(png_bytes)
    hypotheses = tuple(
        ExactCoverHypothesis(CountPair(*pair), primitive_ids, pixels)
        for pair, primitive_ids, pixels in hypothesis_data
    )
    return ExactCoverTarget(
        "sha256:" + hashlib.sha256(png_bytes).hexdigest(),
        minimum,
        tuple(CountPair(*pair) for pair in pairs),
        hypotheses,
    )


def exact_cover_target(png_bytes: bytes) -> ExactCoverTarget:
    source_sha256()
    require_issued_connected_png(png_bytes)
    target = _target_cache(png_bytes)
    return ExactCoverTarget(
        target.png_digest,
        target.minimum_primitive_count,
        tuple(CountPair(*row.as_tuple()) for row in target.count_pairs),
        tuple(
            ExactCoverHypothesis(
                CountPair(*row.count_pair.as_tuple()),
                tuple(row.primitive_ids),
                tuple(row.covered_pixels),
            )
            for row in target.hypotheses
        ),
    )


__all__ = (
    "BoundaryTruth", "CatalogPrimitive", "ConnectedNuisance",
    "ConnectedSyntheticError", "ConnectedSyntheticSample", "CountPair",
    "ExactCoverHypothesis", "ExactCoverTarget", "PanelProgram", "ShapeProgram",
    "build_connected_corpus", "connected_carrier_families", "connected_nuisances",
    "d4_raster_orbit_digest", "exact_cover_target", "primitive_catalog",
    "render_catalog_program", "require_issued_connected_png", "source_sha256",
)
