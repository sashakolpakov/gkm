"""Synthetic-only line/arc rendering and bounded identifiability probes.

Nothing in this module reads ShapeBongard, its authorities, or any downloaded
artifact.  It is a small controlled grammar for asking which *declared*
primitive histories survive a deterministic rasterizer.  In particular,
``canonical_visible_pair`` is deliberately partial: it is a pure function of
the rendered pixels only when every connected component exactly matches one
bounded finite line or annular sector.  Otherwise it is ``None``.  It is not
a universal or semantic minimal-stroke decomposition, and it never falls back
to generator history.

The deliberately easy balanced corpus keeps actions disconnected and uses
four D4 variants at fixed stroke and scale so every row has an exact target.
It is suitable for a mechanistic representation control only.  Its carrier
split and complete 54-cell target grid confer neither generalization evidence
nor benchmark authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import itertools
import math
from typing import Final, Iterable, Literal, Sequence

import numpy as np
from PIL import Image

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)


LOGICAL_SIZE: Final = 1024
IMAGE_SIZE: Final = 64
PIXEL_DENOMINATOR: Final = IMAGE_SIZE * IMAGE_SIZE
SYNTHETIC_SCOPE: Final = "synthetic_only_no_official_data_or_claims"
D4_NAMES: Final = (
    "identity", "r90", "r180", "r270",
    "mirror_x", "mirror_x_r90", "mirror_x_r180", "mirror_x_r270",
)
CARRIER_FAMILIES: Final = (
    "lattice",
    "perimeter",
    "pinwheel",
    "radial",
    "staggered",
)
_ISSUED_SYNTHETIC_PNGS: dict[str, bytes] = {}
_ISSUED_RENDERED_PANELS: dict[
    int, tuple[object, tuple[object, ...]]
] = {}
_SINGLETON_NORMAL_FORM_CACHE: dict[bytes, tuple[int, int] | None] = {}


class SyntheticIdentifiabilityError(ValueError):
    """A synthetic-only source, transport, or bounded grammar invariant differs."""


def source_sha256() -> str:
    """Verify and return this module's import-time source address."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    return value


@dataclass(frozen=True, order=True)
class CountPair:
    straight: int
    arc: int

    def __post_init__(self) -> None:
        straight = _exact_int(self.straight, "straight")
        arc = _exact_int(self.arc, "arc")
        if straight < 0 or arc < 0 or not 1 <= straight + arc <= 9:
            raise ValueError("counts must be nonnegative with total in [1, 9]")

    def as_tuple(self) -> tuple[int, int]:
        return self.straight, self.arc

    def __iter__(self):
        return iter(self.as_tuple())

    def __getitem__(self, index: int) -> int:
        return self.as_tuple()[index]

    def __len__(self) -> int:
        return 2


_VALID_COUNT_PAIR_VALUES: Final = tuple(
    (straight, arc)
    for straight in range(10)
    for arc in range(10)
    if 1 <= straight + arc <= 9
)
def valid_count_pairs() -> tuple[CountPair, ...]:
    """Return fresh exact records in the deterministic 54-cell target order."""

    source_sha256()
    return tuple(
        CountPair(straight, arc) for straight, arc in _VALID_COUNT_PAIR_VALUES
    )


def _validated_program_copy(program: object) -> Program:
    """Rebuild an exact program tree so mutated frozen children cannot pass."""

    if type(program) is not Program:
        raise TypeError("renderer requires an exact Program record")
    Program.__post_init__(program)
    actions: list[PrimitiveAction] = []
    for action in program.actions:
        if type(action) is LineAction:
            LineAction.__post_init__(action)
            actions.append(
                LineAction(
                    action.action_id,
                    Point(action.start.x, action.start.y),
                    Point(action.end.x, action.end.y),
                    action.kind,
                )
            )
        elif type(action) is ArcAction:
            ArcAction.__post_init__(action)
            actions.append(
                ArcAction(
                    action.action_id,
                    Point(action.start.x, action.start.y),
                    Point(action.through.x, action.through.y),
                    Point(action.end.x, action.end.y),
                    action.kind,
                )
            )
        else:
            raise TypeError("program action type differs")
    return Program(program.carrier_family, tuple(actions))


@dataclass(frozen=True, order=True)
class Point:
    x: int
    y: int

    def __post_init__(self) -> None:
        x = _exact_int(self.x, "x")
        y = _exact_int(self.y, "y")
        if not 0 <= x <= LOGICAL_SIZE or not 0 <= y <= LOGICAL_SIZE:
            raise ValueError("point leaves the bounded logical canvas")


@dataclass(frozen=True)
class LineAction:
    action_id: str
    start: Point
    end: Point
    kind: Literal["line"] = "line"

    def __post_init__(self) -> None:
        if type(self.action_id) is not str or not self.action_id:
            raise ValueError("action_id must be nonempty")
        if type(self.start) is not Point or type(self.end) is not Point:
            raise TypeError("line endpoints must be exact Point records")
        if self.start == self.end:
            raise ValueError("a line must have distinct endpoints")
        if math.hypot(self.end.x - self.start.x, self.end.y - self.start.y) < 64:
            raise ValueError("a line must span at least four output pixels")
        if type(self.kind) is not str or self.kind != "line":
            raise ValueError("LineAction.kind must be 'line'")


@dataclass(frozen=True)
class ArcAction:
    """Circular arc selected by start, an on-sweep point, and end."""

    action_id: str
    start: Point
    through: Point
    end: Point
    kind: Literal["arc"] = "arc"

    def __post_init__(self) -> None:
        if type(self.action_id) is not str or not self.action_id:
            raise ValueError("action_id must be nonempty")
        if any(type(point) is not Point for point in (self.start, self.through, self.end)):
            raise TypeError("arc controls must be exact Point records")
        area2 = (
            (self.through.x - self.start.x) * (self.end.y - self.start.y)
            - (self.through.y - self.start.y) * (self.end.x - self.start.x)
        )
        if area2 == 0:
            raise ValueError("arc points must be distinct and non-collinear")
        chord = math.hypot(
            self.end.x - self.start.x, self.end.y - self.start.y
        )
        sagitta = abs(area2) / chord
        if chord < 64 or sagitta < 32:
            raise ValueError(
                "an arc chord and curvature must survive the output raster"
            )
        if type(self.kind) is not str or self.kind != "arc":
            raise ValueError("ArcAction.kind must be 'arc'")


PrimitiveAction = LineAction | ArcAction


@dataclass(frozen=True)
class Program:
    carrier_family: str
    actions: tuple[PrimitiveAction, ...]

    def __post_init__(self) -> None:
        if (
            type(self.carrier_family) is not str
            or not self.carrier_family
            or self.carrier_family.strip() != self.carrier_family
        ):
            raise ValueError("carrier_family must be a canonical nonempty identifier")
        if (
            type(self.actions) is not tuple
            or not 1 <= len(self.actions) <= 9
            or any(type(action) not in (LineAction, ArcAction) for action in self.actions)
        ):
            raise ValueError("programs must contain between one and nine actions")
        ids = tuple(action.action_id for action in self.actions)
        if len(ids) != len(set(ids)):
            raise ValueError("action identifiers must be unique")

    @property
    def declared_pair(self) -> CountPair:
        return CountPair(
            sum(isinstance(action, LineAction) for action in self.actions),
            sum(isinstance(action, ArcAction) for action in self.actions),
        )


@dataclass(frozen=True)
class Nuisance:
    d4: str = "identity"
    stroke_width: int = 2
    scale_milli: int = 1000

    def __post_init__(self) -> None:
        if type(self.d4) is not str or self.d4 not in D4_NAMES:
            raise ValueError(f"d4 must be one of {D4_NAMES!r}")
        width = _exact_int(self.stroke_width, "stroke_width")
        scale = _exact_int(self.scale_milli, "scale_milli")
        if not 1 <= width <= 4:
            raise ValueError("stroke_width must be in [1, 4]")
        if not 750 <= scale <= 1200:
            raise ValueError("scale_milli must be in [750, 1200]")

    @property
    def nuisance_id(self) -> str:
        return f"{self.d4}-w{self.stroke_width}-s{self.scale_milli}"

    @property
    def identity(self) -> str:
        """Compatibility name used by canonical benchmark manifests."""

        return self.nuisance_id


_DEFAULT_NUISANCE_VALUES: Final = (
    ("identity", 2, 1000),
    ("r90", 2, 1000),
    ("mirror_x", 2, 1000),
    ("mirror_x_r270", 2, 1000),
)


def default_nuisances() -> tuple[Nuisance, ...]:
    """Return fresh copies of the frozen four-regime nuisance inventory."""

    source_sha256()
    return tuple(
        Nuisance(d4, width, scale)
        for d4, width, scale in _DEFAULT_NUISANCE_VALUES
    )


@dataclass(frozen=True)
class ActionPixelProvenance:
    action_id: str
    kind: Literal["line", "arc"]
    ink_pixels: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.action_id) is not str or not self.action_id:
            raise ValueError("action_id must be nonempty")
        if type(self.kind) is not str or self.kind not in ("line", "arc"):
            raise ValueError("kind must be line or arc")
        if (
            type(self.ink_pixels) is not tuple
            or not self.ink_pixels
            or any(type(index) is not int for index in self.ink_pixels)
        ):
            raise ValueError("each action must render at least one pixel")
        if self.ink_pixels != tuple(sorted(set(self.ink_pixels))):
            raise ValueError("ink_pixels must be sorted and unique")
        if self.ink_pixels[0] < 0 or self.ink_pixels[-1] >= PIXEL_DENOMINATOR:
            raise ValueError("ink pixel outside the fixed canvas")


@dataclass(frozen=True)
class RenderedPanel:
    png_bytes: bytes
    raster_digest: str
    carrier_id: str
    carrier_family: str
    nuisance: Nuisance
    declared_pair: CountPair
    canonical_visible_pair: CountPair | None
    provenance: tuple[ActionPixelProvenance, ...]

    def __post_init__(self) -> None:
        if type(self.png_bytes) is not bytes or not self.png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("png_bytes is not a PNG transport")
        expected = "sha256:" + hashlib.sha256(self.png_bytes).hexdigest()
        if type(self.raster_digest) is not str or self.raster_digest != expected:
            raise ValueError("raster_digest does not address png_bytes")
        if (
            type(self.carrier_id) is not str
            or type(self.carrier_family) is not str
            or not self.carrier_id
        ):
            raise ValueError("rendered carrier identity differs")
        if self.carrier_id != self.carrier_family:
            raise ValueError("this grammar binds carrier_id to carrier_family")
        if (
            type(self.nuisance) is not Nuisance
            or type(self.declared_pair) is not CountPair
            or (
                self.canonical_visible_pair is not None
                and type(self.canonical_visible_pair) is not CountPair
            )
            or type(self.provenance) is not tuple
            or any(type(row) is not ActionPixelProvenance for row in self.provenance)
        ):
            raise TypeError("rendered panel record types differ")
        if len(self.provenance) != self.declared_pair.straight + self.declared_pair.arc:
            raise ValueError("provenance must contain exactly one row per declared action")
        ids = tuple(row.action_id for row in self.provenance)
        if len(ids) != len(set(ids)):
            raise ValueError("provenance action identifiers must be unique")
        if (
            sum(row.kind == "line" for row in self.provenance)
            != self.declared_pair.straight
            or sum(row.kind == "arc" for row in self.provenance)
            != self.declared_pair.arc
        ):
            raise ValueError("provenance kinds do not reconstruct the declared pair")
        if self.canonical_visible_pair is not None and (
            self.canonical_visible_pair.straight
            + self.canonical_visible_pair.arc
            > self.declared_pair.straight + self.declared_pair.arc
        ):
            raise ValueError("visible-support quotient exceeds declared history")
        try:
            with Image.open(BytesIO(self.png_bytes)) as image:
                if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("rendered transport must be one PNG frame")
                image.load()
                foreground = tuple(
                    int(index)
                    for index in np.flatnonzero(np.asarray(image.convert("L")) < 128)
                )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"cannot decode rendered PNG: {exc}") from exc
        provenance_union = tuple(
            sorted({index for row in self.provenance for index in row.ink_pixels})
        )
        if foreground != provenance_union:
            raise ValueError("provenance pixels do not reconstruct rendered foreground")

    @property
    def png_sha256(self) -> str:
        """Compatibility name for the exact raster transport address."""

        return self.raster_digest


def _rendered_panel_fingerprint(panel: RenderedPanel) -> tuple[object, ...]:
    def typed(value: object) -> tuple[type[object], object]:
        return type(value), value

    return (
        typed(panel.png_bytes),
        typed(panel.raster_digest),
        typed(panel.carrier_id),
        typed(panel.carrier_family),
        (
            type(panel.nuisance),
            typed(panel.nuisance.d4),
            typed(panel.nuisance.stroke_width),
            typed(panel.nuisance.scale_milli),
        ),
        (
            type(panel.declared_pair),
            typed(panel.declared_pair.straight),
            typed(panel.declared_pair.arc),
        ),
        None
        if panel.canonical_visible_pair is None
        else (
            type(panel.canonical_visible_pair),
            typed(panel.canonical_visible_pair.straight),
            typed(panel.canonical_visible_pair.arc),
        ),
        (
            type(panel.provenance),
            tuple(
                (
                    type(row),
                    typed(row.action_id),
                    typed(row.kind),
                    type(row.ink_pixels),
                    tuple(typed(index) for index in row.ink_pixels),
                )
                for row in panel.provenance
            ),
        ),
    )


def _register_issued_rendered_panel(panel: RenderedPanel) -> None:
    key = id(panel)
    previous = _ISSUED_RENDERED_PANELS.get(key)
    if previous is not None and previous[0] is not panel:
        raise SyntheticIdentifiabilityError("rendered-panel identity collision")
    _ISSUED_RENDERED_PANELS[key] = (panel, _rendered_panel_fingerprint(panel))


def require_issued_rendered_panel(panel: RenderedPanel) -> None:
    """Require the exact immutable record issued by :func:`render_program`."""

    source_sha256()
    if type(panel) is not RenderedPanel:
        raise SyntheticIdentifiabilityError(
            "panel was not issued by the synthetic renderer"
        )
    try:
        RenderedPanel.__post_init__(panel)
        Nuisance.__post_init__(panel.nuisance)
        CountPair.__post_init__(panel.declared_pair)
        if panel.canonical_visible_pair is not None:
            CountPair.__post_init__(panel.canonical_visible_pair)
        for row in panel.provenance:
            ActionPixelProvenance.__post_init__(row)
    except (TypeError, ValueError) as exc:
        raise SyntheticIdentifiabilityError(
            "panel was not issued by the synthetic renderer"
        ) from exc
    issued = _ISSUED_RENDERED_PANELS.get(id(panel))
    if (
        issued is None
        or issued[0] is not panel
        or issued[1] != _rendered_panel_fingerprint(panel)
    ):
        raise SyntheticIdentifiabilityError(
            "panel was not issued by the synthetic renderer"
        )


@dataclass(frozen=True)
class SyntheticSample:
    sample_id: str
    panel: RenderedPanel

    def __post_init__(self) -> None:
        if (
            type(self.sample_id) is not str
            or not self.sample_id
            or self.sample_id.strip() != self.sample_id
            or type(self.panel) is not RenderedPanel
        ):
            raise ValueError("sample_id must be a canonical nonempty identifier")
        require_issued_rendered_panel(self.panel)
        if self.panel.carrier_id != self.panel.carrier_family:
            raise ValueError("sample carrier identity may not be relabelled")

    @property
    def png_bytes(self) -> bytes:
        return self.panel.png_bytes

    @property
    def carrier_id(self) -> str:
        return self.panel.carrier_id

    @property
    def carrier_family(self) -> str:
        return self.panel.carrier_family

    @property
    def nuisance(self) -> Nuisance:
        return self.panel.nuisance

    @property
    def declared_pair(self) -> CountPair:
        return self.panel.declared_pair

    @property
    def canonical_visible_pair(self) -> CountPair | None:
        return self.panel.canonical_visible_pair

    @property
    def provenance(self) -> tuple[ActionPixelProvenance, ...]:
        return self.panel.provenance


@dataclass(frozen=True)
class CorpusSplit:
    train: tuple[SyntheticSample, ...]
    held_out: tuple[SyntheticSample, ...]
    held_out_carriers: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.train) is not tuple
            or type(self.held_out) is not tuple
            or type(self.held_out_carriers) is not tuple
            or any(type(row) is not SyntheticSample for row in self.train + self.held_out)
            or any(type(value) is not str or not value for value in self.held_out_carriers)
            or not self.train
            or not self.held_out
        ):
            raise ValueError("both split roles must be nonempty")
        for row in self.train + self.held_out:
            SyntheticSample.__post_init__(row)
        train_carriers = {row.carrier_id for row in self.train}
        held_carriers = {row.carrier_id for row in self.held_out}
        if train_carriers & held_carriers:
            raise ValueError("carrier_id leakage across split roles")
        if (
            self.held_out_carriers != tuple(sorted(set(self.held_out_carriers)))
            or held_carriers != set(self.held_out_carriers)
        ):
            raise ValueError("held_out_carriers must exactly describe held_out")
        if (
            {row.panel.raster_digest for row in self.train}
            & {row.panel.raster_digest for row in self.held_out}
        ):
            raise ValueError("exact raster bytes leak across split roles")
        if (
            {d4_raster_orbit_digest(row.panel.png_bytes) for row in self.train}
            & {d4_raster_orbit_digest(row.panel.png_bytes) for row in self.held_out}
        ):
            raise ValueError("D4 raster orbit leaks across split roles")
        ids = [row.sample_id for row in (*self.train, *self.held_out)]
        if len(ids) != len(set(ids)):
            raise ValueError("sample identifiers must be globally unique")

    @property
    def training(self) -> tuple[SyntheticSample, ...]:
        return self.train

    @property
    def evaluation(self) -> tuple[SyntheticSample, ...]:
        return self.held_out


def _transform(point: Point, nuisance: Nuisance) -> tuple[float, float]:
    x = (point.x - LOGICAL_SIZE / 2) * nuisance.scale_milli / 1000
    y = (point.y - LOGICAL_SIZE / 2) * nuisance.scale_milli / 1000
    mirror = nuisance.d4.startswith("mirror_x")
    if mirror:
        x = -x
    suffix = nuisance.d4.removeprefix("mirror_x_") if mirror else nuisance.d4
    if nuisance.d4 == "mirror_x":
        suffix = "identity"
    turns = {"identity": 0, "r90": 1, "r180": 2, "r270": 3}[suffix]
    for _ in range(turns):
        x, y = -y, x
    return x + LOGICAL_SIZE / 2, y + LOGICAL_SIZE / 2


_PIXEL_X, _PIXEL_Y = np.meshgrid(
    (np.arange(IMAGE_SIZE, dtype=np.float64) + 0.5) * LOGICAL_SIZE / IMAGE_SIZE,
    (np.arange(IMAGE_SIZE, dtype=np.float64) + 0.5) * LOGICAL_SIZE / IMAGE_SIZE,
)


def _line_mask(action: LineAction, nuisance: Nuisance) -> np.ndarray:
    ax, ay = _transform(action.start, nuisance)
    bx, by = _transform(action.end, nuisance)
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    t = np.clip(((_PIXEL_X - ax) * dx + (_PIXEL_Y - ay) * dy) / denominator, 0, 1)
    distance2 = (_PIXEL_X - (ax + t * dx)) ** 2 + (_PIXEL_Y - (ay + t * dy)) ** 2
    radius = nuisance.stroke_width * LOGICAL_SIZE / (2 * IMAGE_SIZE)
    return distance2 <= radius * radius


def _circle(points: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    (ax, ay), (bx, by), (cx, cy) = points
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        raise ValueError("transformed arc became degenerate")
    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / d
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


def _arc_geometry(
    action: ArcAction, nuisance: Nuisance
) -> tuple[float, float, float, float, float, bool]:
    points = tuple(_transform(point, nuisance) for point in (action.start, action.through, action.end))
    cx, cy, radius = _circle(points)
    angles = tuple(math.atan2(y - cy, x - cx) % math.tau for x, y in points)
    start, through, end = angles
    ccw_span = (end - start) % math.tau
    through_span = (through - start) % math.tau
    return cx, cy, radius, start, ccw_span, through_span <= ccw_span + 1e-10


def _arc_mask(action: ArcAction, nuisance: Nuisance) -> np.ndarray:
    cx, cy, radius, start, ccw_span, ccw = _arc_geometry(action, nuisance)
    angles = np.arctan2(_PIXEL_Y - cy, _PIXEL_X - cx) % math.tau
    if ccw:
        on_sweep = ((angles - start) % math.tau) <= ccw_span + 1e-10
    else:
        clockwise_span = math.tau - ccw_span
        on_sweep = ((start - angles) % math.tau) <= clockwise_span + 1e-10
    radial = np.hypot(_PIXEL_X - cx, _PIXEL_Y - cy)
    radius_tolerance = nuisance.stroke_width * LOGICAL_SIZE / (2 * IMAGE_SIZE)
    return on_sweep & (np.abs(radial - radius) <= radius_tolerance)


def _action_mask(action: PrimitiveAction, nuisance: Nuisance) -> np.ndarray:
    return _line_mask(action, nuisance) if isinstance(action, LineAction) else _arc_mask(action, nuisance)


def _same_support(
    left: PrimitiveAction, right: PrimitiveAction, nuisance: Nuisance
) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, LineAction) and isinstance(right, LineAction):
        a, b = _transform(left.start, nuisance), _transform(left.end, nuisance)
        c, d = _transform(right.start, nuisance), _transform(right.end, nuisance)
        cross_direction = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
        cross_offset = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return abs(cross_direction) < 1e-6 and abs(cross_offset) < 1e-6
    assert isinstance(left, ArcAction) and isinstance(right, ArcAction)
    lc = _arc_geometry(left, nuisance)[:3]
    rc = _arc_geometry(right, nuisance)[:3]
    return all(abs(a - b) < 1e-6 for a, b in zip(lc, rc))


def _touching(left: np.ndarray, right: np.ndarray) -> bool:
    if np.any(left & right):
        return True
    padded = np.pad(left, 1)
    dilated = np.zeros_like(left)
    for dy in range(3):
        for dx in range(3):
            dilated |= padded[dy:dy + IMAGE_SIZE, dx:dx + IMAGE_SIZE]
    return bool(np.any(dilated & right))


def _connected_component_masks(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return ()
    remaining = {tuple(int(value) for value in row) for row in coordinates}
    components: list[np.ndarray] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        frontier = [seed]
        members = [seed]
        while frontier:
            y, x = frontier.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbour = (y + dy, x + dx)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        frontier.append(neighbour)
                        members.append(neighbour)
        component = np.zeros_like(mask)
        for y, x in members:
            component[y, x] = True
        components.append(component)
    return tuple(components)


def _is_single_8_connected_component(mask: np.ndarray) -> bool:
    return len(_connected_component_masks(mask)) == 1


def _mask_has_bounded_exact_single_line_explanation(mask: np.ndarray) -> bool:
    """Return whether one finite constant-width capsule equals ``mask``.

    This is a bounded raster search, not a continuous-geometry theorem.  For
    each sampled axis and each renderer half-width, foreground pixels define
    the complete feasible offset interval.  A finite segment is then derived
    from the projected pixel-disc intervals and compared against *all* 4,096
    pixel centres.  Endpoint caps are therefore part of the exact comparison;
    no interior-only match is accepted.
    """

    if type(mask) is not np.ndarray or mask.dtype != np.bool_ or mask.shape != (
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise SyntheticIdentifiabilityError("single-line raster mask differs")
    foreground_y, foreground_x = np.nonzero(mask)
    if len(foreground_x) < 2:
        return False
    foreground = np.column_stack(
        (foreground_x.astype(np.float64) + 0.5, foreground_y.astype(np.float64) + 0.5)
    )
    covariance = np.cov(foreground.T, bias=True)
    _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    principal_angle = math.atan2(float(direction[1]), float(direction[0]))
    pixel_x = _PIXEL_X.ravel() * IMAGE_SIZE / LOGICAL_SIZE
    pixel_y = _PIXEL_Y.ravel() * IMAGE_SIZE / LOGICAL_SIZE
    flat_mask = mask.ravel()

    for angle in np.linspace(principal_angle - 0.15, principal_angle + 0.15, 121):
        cosine, sine = math.cos(float(angle)), math.sin(float(angle))
        along = pixel_x * cosine + pixel_y * sine
        across = -pixel_x * sine + pixel_y * cosine
        foreground_across = across[flat_mask]
        for radius in (0.5, 1.0, 1.5, 2.0):
            center_lower = float(np.max(foreground_across - radius))
            center_upper = float(np.min(foreground_across + radius))
            if center_lower > center_upper + 1e-12:
                continue
            centers = (
                center_lower,
                center_upper,
                (center_lower + center_upper) / 2,
            )
            for center in centers:
                perpendicular = np.abs(across - center)
                within_strip = perpendicular <= radius + 1e-12
                reach = np.zeros_like(perpendicular)
                reach[within_strip] = np.sqrt(
                    np.maximum(
                        0.0,
                        radius * radius
                        - np.square(perpendicular[within_strip]),
                    )
                )
                interval_lower = along - reach
                interval_upper = along + reach
                segment_start = float(np.min(interval_upper[flat_mask]))
                segment_end = float(np.max(interval_lower[flat_mask]))
                if segment_start > segment_end:
                    midpoint = (segment_start + segment_end) / 2
                    segment_start = segment_end = midpoint
                predicted = (
                    within_strip
                    & (interval_upper >= segment_start - 1e-12)
                    & (interval_lower <= segment_end + 1e-12)
                )
                if np.array_equal(predicted, flat_mask):
                    return True
    return False


def has_bounded_exact_single_line_explanation(png_bytes: bytes) -> bool:
    """Check an issued PNG against the bounded exact finite-line raster model."""

    require_issued_synthetic_png(png_bytes)
    return _mask_has_bounded_exact_single_line_explanation(
        _mask_from_png(png_bytes)
    )


def _angular_sector_exactly_selects(
    angles: np.ndarray,
    annulus: np.ndarray,
    foreground: np.ndarray,
) -> bool:
    if np.any(foreground & ~annulus):
        return False
    indices = np.flatnonzero(annulus)
    if not len(indices):
        return False
    order = indices[np.argsort(angles[indices], kind="stable")]
    ordered_angles = angles[order]
    ordered_foreground = foreground[order]
    starts = np.concatenate(
        (
            np.asarray((0,), dtype=np.int64),
            np.flatnonzero(np.diff(ordered_angles) > 1e-12) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.asarray((len(order),))))
    groups: list[bool] = []
    for start, end in zip(starts, ends, strict=True):
        values = ordered_foreground[start:end]
        if np.any(values) and not np.all(values):
            return False
        groups.append(bool(values[0]))
    labels = np.asarray(groups, dtype=np.bool_)
    return bool(
        np.any(labels)
        and np.count_nonzero(labels != np.roll(labels, 1)) <= 2
    )


def _mask_has_bounded_exact_single_arc_explanation(mask: np.ndarray) -> bool:
    """Return whether one finite annular sector equals the complete mask."""

    if type(mask) is not np.ndarray or mask.dtype != np.bool_ or mask.shape != (
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise SyntheticIdentifiabilityError("single-arc raster mask differs")
    foreground_y, foreground_x = np.nonzero(mask)
    if len(foreground_x) < 5:
        return False
    foreground_points = np.column_stack(
        (foreground_x.astype(np.float64) + 0.5, foreground_y.astype(np.float64) + 0.5)
    )
    design = np.column_stack(
        (
            2 * foreground_points[:, 0],
            2 * foreground_points[:, 1],
            np.ones(len(foreground_points)),
        )
    )
    target = (
        np.square(foreground_points[:, 0])
        + np.square(foreground_points[:, 1])
    )
    center_x, center_y, _constant = np.linalg.lstsq(
        design, target, rcond=None
    )[0]
    flat_x = _PIXEL_X.ravel() * IMAGE_SIZE / LOGICAL_SIZE
    flat_y = _PIXEL_Y.ravel() * IMAGE_SIZE / LOGICAL_SIZE
    foreground = mask.ravel()
    offsets = np.arange(-3.0, 3.01, 0.25)

    def candidate_centers(value: float) -> tuple[float, ...]:
        # Least-squares centres are biased by a clipped sweep.  Preserve that
        # deterministic local grid, but also test the nearest integer/half/
        # quarter-pixel centres used by symmetric members of the renderer
        # grammar.  The set remains small and explicitly bounded.
        values = {float(value + offset) for offset in offsets}
        values.update(
            float(round(value * denominator) / denominator)
            for denominator in (1, 2, 4)
        )
        return tuple(sorted(values))

    for candidate_x in candidate_centers(float(center_x)):
        for candidate_y in candidate_centers(float(center_y)):
            distances = np.hypot(
                flat_x - candidate_x, flat_y - candidate_y
            )
            angles = np.mod(
                np.arctan2(flat_y - candidate_y, flat_x - candidate_x),
                math.tau,
            )
            foreground_distances = distances[foreground]
            for half_width in (0.5, 1.0, 1.5, 2.0):
                radius_lower = float(
                    np.max(foreground_distances - half_width)
                )
                radius_upper = float(
                    np.min(foreground_distances + half_width)
                )
                if radius_lower > radius_upper + 1e-12:
                    continue
                for radius in (
                    radius_lower,
                    radius_upper,
                    (radius_lower + radius_upper) / 2,
                ):
                    annulus = (
                        np.abs(distances - radius) <= half_width + 1e-12
                    )
                    if _angular_sector_exactly_selects(
                        angles, annulus, foreground
                    ):
                        return True
    return False


def has_bounded_exact_single_arc_explanation(png_bytes: bytes) -> bool:
    """Check an issued PNG against the shared finite-arc raster quotient."""

    require_issued_synthetic_png(png_bytes)
    return _mask_has_bounded_exact_single_arc_explanation(
        _mask_from_png(png_bytes)
    )


def _mask_singleton_normal_form(mask: np.ndarray) -> CountPair | None:
    if type(mask) is not np.ndarray or mask.dtype != np.bool_ or mask.shape != (
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise SyntheticIdentifiabilityError("singleton normal-form mask differs")
    key = np.ascontiguousarray(mask).tobytes()
    if key in _SINGLETON_NORMAL_FORM_CACHE:
        cached = _SINGLETON_NORMAL_FORM_CACHE[key]
        return None if cached is None else CountPair(*cached)
    if not _is_single_8_connected_component(mask):
        _SINGLETON_NORMAL_FORM_CACHE[key] = None
        return None
    if _mask_has_bounded_exact_single_line_explanation(mask):
        result: tuple[int, int] | None = (1, 0)
    elif _mask_has_bounded_exact_single_arc_explanation(mask):
        result = (0, 1)
    else:
        result = None
    _SINGLETON_NORMAL_FORM_CACHE[key] = result
    return None if result is None else CountPair(*result)


def _mask_component_normal_form(mask: np.ndarray) -> CountPair | None:
    components = _connected_component_masks(mask)
    forms = tuple(_mask_singleton_normal_form(mask) for mask in components)
    if not components or any(form is None for form in forms):
        return None
    straight = sum(form.straight for form in forms if form is not None)
    arc = sum(form.arc for form in forms if form is not None)
    if not 1 <= straight + arc <= 9:
        return None
    return CountPair(straight, arc)


def visible_raster_component_normal_form(
    png_bytes: bytes,
) -> CountPair | None:
    """Return the pure-raster target, or ``None`` for unresolved components."""

    require_issued_synthetic_png(png_bytes)
    return _mask_component_normal_form(_mask_from_png(png_bytes))


def canonical_visible_pair(
    program: Program, nuisance: Nuisance | None = None
) -> CountPair | None:
    """Return this grammar's bounded visible-raster primitive normal form.

    Each 8-connected component must have a complete 4,096-pixel explanation as
    one finite line (the deterministic equal-complexity tie-break) or one
    finite annular sector. The target is the sum of those component primitives.
    If any connected component lacks such an explanation, the target is
    explicitly unresolved (``None``), never reconstructed from generator
    history. Thus identical PNGs necessarily have identical target states.
    """

    source_sha256()
    if nuisance is None:
        nuisance = Nuisance()
    if type(program) is not Program or type(nuisance) is not Nuisance:
        raise TypeError("visible-support quotient requires exact synthetic records")
    validated_program = _validated_program_copy(program)
    validated_nuisance = Nuisance(
        nuisance.d4, nuisance.stroke_width, nuisance.scale_milli
    )
    masks = tuple(
        _action_mask(action, validated_nuisance)
        for action in validated_program.actions
    )
    combined = np.logical_or.reduce(masks)
    return _mask_component_normal_form(combined)


def _png(mask: np.ndarray) -> bytes:
    image = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L")
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def render_program(
    program: Program, nuisance: Nuisance | None = None
) -> RenderedPanel:
    source_sha256()
    if nuisance is None:
        nuisance = Nuisance()
    if type(program) is not Program or type(nuisance) is not Nuisance:
        raise TypeError("renderer requires exact synthetic program and nuisance records")
    issued_program = _validated_program_copy(program)
    # Never retain the caller's nested record (including the function-default
    # instance) in an issued panel.  This prevents adversarial mutation of one
    # panel from poisoning later renders that share the original input object.
    issued_nuisance = Nuisance(
        nuisance.d4, nuisance.stroke_width, nuisance.scale_milli
    )
    masks = tuple(
        _action_mask(action, issued_nuisance) for action in issued_program.actions
    )
    if any(not np.any(mask) for mask in masks):
        raise ValueError("every action must leave visible raster support")
    combined = np.logical_or.reduce(masks)
    png_bytes = _png(combined)
    provenance = tuple(
        ActionPixelProvenance(
            action.action_id,
            action.kind,
            tuple(int(index) for index in np.flatnonzero(mask)),
        )
        for action, mask in zip(issued_program.actions, masks)
    )
    panel = RenderedPanel(
        png_bytes=png_bytes,
        raster_digest="sha256:" + hashlib.sha256(png_bytes).hexdigest(),
        carrier_id=issued_program.carrier_family,
        carrier_family=issued_program.carrier_family,
        nuisance=issued_nuisance,
        declared_pair=issued_program.declared_pair,
        canonical_visible_pair=canonical_visible_pair(
            issued_program, issued_nuisance
        ),
        provenance=provenance,
    )
    previous = _ISSUED_SYNTHETIC_PNGS.setdefault(panel.raster_digest, panel.png_bytes)
    if previous != panel.png_bytes:  # pragma: no cover - SHA-256 collision guard
        raise SyntheticIdentifiabilityError("synthetic raster digest collision")
    _register_issued_rendered_panel(panel)
    return panel


def require_issued_synthetic_png(png_bytes: bytes) -> str:
    """Require bytes issued by this process's bounded synthetic renderer."""

    source_sha256()
    if type(png_bytes) is not bytes:
        raise SyntheticIdentifiabilityError("synthetic PNG payload must be exact bytes")
    digest = "sha256:" + hashlib.sha256(png_bytes).hexdigest()
    if _ISSUED_SYNTHETIC_PNGS.get(digest) != png_bytes:
        raise SyntheticIdentifiabilityError(
            "PNG bytes were not issued by the in-process synthetic renderer"
        )
    return digest


def d4_raster_orbit_digest(png_bytes: bytes) -> str:
    """Address the exact foreground orbit under all eight square symmetries."""

    require_issued_synthetic_png(png_bytes)
    mask = _mask_from_png(png_bytes)
    transforms = []
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        transforms.extend((rotated, np.fliplr(rotated)))
    canonical = min(np.ascontiguousarray(value).tobytes() for value in transforms)
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


_CENTERS: Final = (
    (224, 224), (512, 224), (800, 224),
    (224, 512), (512, 512), (800, 512),
    (224, 800), (512, 800), (800, 800),
)


def _carrier_program(pair: CountPair, family: str) -> Program:
    if family not in CARRIER_FAMILIES:
        raise ValueError(f"unknown synthetic carrier family: {family!r}")
    if family == "lattice":
        order = tuple(range(9))
        phase = 0
    elif family == "radial":
        order = (4, 0, 2, 8, 6, 1, 5, 7, 3)
        phase = 1
    elif family == "staggered":
        order = (0, 3, 6, 7, 4, 1, 2, 5, 8)
        phase = 2
    elif family == "perimeter":
        order = (0, 1, 2, 5, 8, 7, 6, 3, 4)
        phase = 3
    else:
        order = (4, 1, 5, 7, 3, 0, 2, 8, 6)
        phase = 2
    extent = {
        "lattice": 80,
        "perimeter": 72,
        "pinwheel": 64,
        "radial": 88,
        "staggered": 96,
    }[family]
    diagonal_extent = round(extent * 0.8)
    kinds = ("line",) * pair.straight + ("arc",) * pair.arc
    actions: list[PrimitiveAction] = []
    for index, kind in enumerate(kinds):
        cx, cy = _CENTERS[order[index]]
        orientation = (index + phase) % 4
        if kind == "line":
            offsets = (
                ((-extent, 0), (extent, 0)),
                ((0, -extent), (0, extent)),
                ((-diagonal_extent, -diagonal_extent), (diagonal_extent, diagonal_extent)),
                ((-diagonal_extent, diagonal_extent), (diagonal_extent, -diagonal_extent)),
            )
            (sx, sy), (ex, ey) = offsets[orientation]
            actions.append(LineAction(f"a{index:02d}", Point(cx + sx, cy + sy), Point(cx + ex, cy + ey)))
        else:
            arc_offsets = (
                ((-extent, 0), (0, -extent), (extent, 0)),
                ((0, -extent), (extent, 0), (0, extent)),
                ((extent, 0), (0, extent), (-extent, 0)),
                ((0, extent), (-extent, 0), (0, -extent)),
            )
            start, through, end = arc_offsets[orientation]
            actions.append(ArcAction(
                f"a{index:02d}",
                Point(cx + start[0], cy + start[1]),
                Point(cx + through[0], cy + through[1]),
                Point(cx + end[0], cy + end[1]),
            ))
    return Program(family, tuple(actions))


def build_balanced_corpus(
    *,
    carrier_families: Sequence[str] = CARRIER_FAMILIES,
    nuisances: Sequence[Nuisance] | None = None,
    samples_per_pair_per_carrier: int = 1,
) -> tuple[SyntheticSample, ...]:
    """Build one declared==visible row for every family/nuisance/54-cell stratum."""

    source_sha256()
    families = tuple(carrier_families)
    if nuisances is None:
        nuisances = default_nuisances()
    styles = tuple(nuisances)
    if (
        type(carrier_families) not in (tuple, list)
        or any(type(family) is not str for family in families)
        or type(nuisances) not in (tuple, list)
        or any(type(style) is not Nuisance for style in styles)
    ):
        raise TypeError("corpus inventories require exact synthetic values")
    repeats = _exact_int(samples_per_pair_per_carrier, "samples_per_pair_per_carrier")
    if repeats != 1:
        raise ValueError(
            "the balanced foundation freezes exactly one sample per "
            "family/nuisance/target cell"
        )
    if len(families) < 2 or len(families) != len(set(families)):
        raise ValueError("at least two unique carrier families are required")
    if not styles or len(styles) != len(set(styles)):
        raise ValueError("nuisances must be nonempty and unique")
    rows: list[SyntheticSample] = []
    for family in families:
        for nuisance in styles:
            cell: list[SyntheticSample] = []
            for pair in valid_count_pairs():
                panel = render_program(_carrier_program(pair, family), nuisance)
                if panel.canonical_visible_pair != pair or panel.declared_pair != pair:
                    raise RuntimeError(
                        "balanced corpus geometry aliased a target; counterfactual "
                        "ambiguities belong only in ambiguity_cases()"
                    )
                sample_id = f"{family}:{nuisance.nuisance_id}:s{pair.straight}a{pair.arc}"
                cell.append(SyntheticSample(sample_id, panel))
            if {row.canonical_visible_pair for row in cell} != set(valid_count_pairs()):
                raise RuntimeError("family/nuisance stratum does not cover all 54 targets")
            rows.extend(cell)
    return tuple(rows)


def carrier_disjoint_split(
    samples: Sequence[SyntheticSample], *, held_out_families: Iterable[str]
) -> CorpusSplit:
    source_sha256()
    if (
        type(samples) not in (tuple, list)
        or not samples
        or any(type(row) is not SyntheticSample for row in samples)
    ):
        raise TypeError("split samples require exact synthetic records")
    for row in samples:
        SyntheticSample.__post_init__(row)
    held = tuple(sorted(set(held_out_families)))
    if any(type(value) is not str or not value for value in held):
        raise TypeError("held-out family identifiers differ")
    if not held:
        raise ValueError("held_out_families must be nonempty")
    known = {row.carrier_id for row in samples}
    if not set(held) < known:
        raise ValueError("held-out carriers must be a strict nonempty subset")
    train = tuple(row for row in samples if row.carrier_id not in held)
    test = tuple(row for row in samples if row.carrier_id in held)
    return CorpusSplit(train, test, held)


@dataclass(frozen=True)
class AmbiguityCase:
    case_id: str
    left: Program
    right: Program
    expected_relation: Literal["exact", "near"]

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("case_id must be nonempty")
        if (
            type(self.left) is not Program
            or type(self.right) is not Program
            or type(self.expected_relation) is not str
            or self.expected_relation not in ("exact", "near")
        ):
            raise ValueError("expected_relation must be exact or near")
        if self.left.declared_pair == self.right.declared_pair:
            raise ValueError("ambiguity cases require different declared histories")


def ambiguity_cases() -> tuple[AmbiguityCase, ...]:
    source_sha256()
    full_line = LineAction("full", Point(192, 512), Point(832, 512))
    full_arc = ArcAction("full", Point(192, 512), Point(512, 192), Point(832, 512))
    vertical_stem = LineAction("stem", Point(512, 160), Point(512, 864))
    disconnected_context = LineAction(
        "context", Point(128, 800), Point(320, 800)
    )
    return (
        AmbiguityCase(
            "one-line-vs-raster-aliased-split-collinear",
            Program("counterfactual-line", (full_line,)),
            Program("counterfactual-line", (
                LineAction("left", Point(192, 512), Point(511, 512)),
                LineAction("right", Point(513, 512), Point(832, 512)),
            )),
            "exact",
        ),
        AmbiguityCase(
            "one-arc-vs-split-cocircular",
            Program("counterfactual-arc", (full_arc,)),
            Program("counterfactual-arc", (
                ArcAction("left", Point(192, 512), Point(256, 320), Point(512, 192)),
                ArcAction("right", Point(512, 192), Point(768, 320), Point(832, 512)),
            )),
            "exact",
        ),
        AmbiguityCase(
            "full-arc-plus-contained-left-arc",
            Program("counterfactual-arc-containment", (full_arc,)),
            Program("counterfactual-arc-containment", (
                ArcAction("outer", Point(192, 512), Point(512, 192), Point(832, 512)),
                ArcAction("contained", Point(192, 512), Point(256, 320), Point(512, 192)),
            )),
            "exact",
        ),
        AmbiguityCase(
            "one-line-plus-raster-subsumed-near-parallel-line",
            Program("counterfactual-subsumed-line", (vertical_stem,)),
            Program("counterfactual-subsumed-line", (
                vertical_stem,
                LineAction("subsumed", Point(504, 416), Point(520, 608)),
            )),
            "exact",
        ),
        AmbiguityCase(
            "endpoint-branch-vs-one-raster-equivalent-line",
            Program("counterfactual-endpoint-branch", (
                LineAction("stem", Point(512, 160), Point(512, 864)),
                LineAction("branch", Point(504, 129), Point(520, 191)),
            )),
            Program("counterfactual-endpoint-branch", (
                LineAction("single", Point(504, 124), Point(508, 864)),
            )),
            "exact",
        ),
        AmbiguityCase(
            "endpoint-branch-alias-with-disconnected-context",
            Program("counterfactual-endpoint-context", (
                LineAction("stem", Point(512, 160), Point(512, 864)),
                LineAction("branch", Point(504, 129), Point(520, 191)),
                disconnected_context,
            )),
            Program("counterfactual-endpoint-context", (
                LineAction("single", Point(504, 124), Point(508, 864)),
                disconnected_context,
            )),
            "exact",
        ),
        AmbiguityCase(
            "endpoint-branch-alias-with-touching-context",
            Program("counterfactual-endpoint-touch", (
                LineAction("stem", Point(512, 160), Point(512, 864)),
                LineAction("branch", Point(504, 129), Point(520, 191)),
                LineAction("touch", Point(508, 512), Point(800, 512)),
            )),
            Program("counterfactual-endpoint-touch", (
                LineAction("single", Point(504, 124), Point(508, 864)),
                LineAction("touch", Point(508, 512), Point(800, 512)),
            )),
            "exact",
        ),
        AmbiguityCase(
            "three-line-chain-vs-one-raster-equivalent-arc",
            Program("counterfactual-line-chain-arc", (
                LineAction("line-0", Point(421, 421), Point(479, 388)),
                LineAction("line-1", Point(479, 388), Point(545, 388)),
                LineAction("line-2", Point(545, 388), Point(603, 421)),
            )),
            Program("counterfactual-line-chain-arc", (
                ArcAction(
                    "arc",
                    Point(404, 434),
                    Point(512, 387),
                    Point(620, 434),
                ),
            )),
            "exact",
        ),
        AmbiguityCase(
            "line-chain-arc-alias-with-disconnected-context",
            Program("counterfactual-line-chain-arc-context", (
                LineAction("line-0", Point(421, 421), Point(479, 388)),
                LineAction("line-1", Point(479, 388), Point(545, 388)),
                LineAction("line-2", Point(545, 388), Point(603, 421)),
                disconnected_context,
            )),
            Program("counterfactual-line-chain-arc-context", (
                ArcAction(
                    "arc",
                    Point(404, 434),
                    Point(512, 387),
                    Point(620, 434),
                ),
                disconnected_context,
            )),
            "exact",
        ),
        AmbiguityCase(
            "line-chain-arc-alias-with-touching-context",
            Program("counterfactual-line-chain-arc-touch", (
                LineAction("line-0", Point(421, 421), Point(479, 388)),
                LineAction("line-1", Point(479, 388), Point(545, 388)),
                LineAction("line-2", Point(545, 388), Point(603, 421)),
                LineAction("touch", Point(603, 421), Point(800, 421)),
            )),
            Program("counterfactual-line-chain-arc-touch", (
                ArcAction(
                    "arc",
                    Point(404, 434),
                    Point(512, 387),
                    Point(620, 434),
                ),
                LineAction("touch", Point(603, 421), Point(800, 421)),
            )),
            "exact",
        ),
        AmbiguityCase(
            "one-line-vs-visible-one-pixel-gap",
            Program("counterfactual-near-line", (full_line,)),
            Program("counterfactual-near-line", (
                LineAction("left", Point(192, 512), Point(487, 512)),
                LineAction("right", Point(537, 512), Point(832, 512)),
            )),
            "near",
        ),
    )


def build_identifiability_counterfactuals() -> tuple[SyntheticSample, ...]:
    """Return two representative exact history collisions as four rows."""

    selected = tuple(
        case
        for case in ambiguity_cases()
        if case.case_id
        in {
            "one-line-vs-raster-aliased-split-collinear",
            "one-arc-vs-split-cocircular",
        }
    )
    rows: list[SyntheticSample] = []
    for case in selected:
        rows.extend(
            (
                SyntheticSample(f"{case.case_id}:left", render_program(case.left)),
                SyntheticSample(f"{case.case_id}:right", render_program(case.right)),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class AuditCandidate:
    candidate_id: str
    panel: RenderedPanel

    def __post_init__(self) -> None:
        if (
            type(self.candidate_id) is not str
            or not self.candidate_id
            or self.candidate_id.strip() != self.candidate_id
            or type(self.panel) is not RenderedPanel
        ):
            raise ValueError("candidate_id must be canonical and nonempty")
        require_issued_rendered_panel(self.panel)


@dataclass(frozen=True)
class CollisionClass:
    kind: Literal["exact", "near"]
    candidate_ids: tuple[str, ...]
    raster_digests: tuple[str, ...]
    declared_pairs: tuple[CountPair, ...]
    canonical_visible_pairs: tuple[CountPair | None, ...]
    xor_pixels: int
    union_pixels: int
    denominator_pixels: int
    iou_millionths: int

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in ("exact", "near"):
            raise ValueError("collision kind must be exact or near")
        if (
            type(self.candidate_ids) is not tuple
            or type(self.raster_digests) is not tuple
            or type(self.declared_pairs) is not tuple
            or type(self.canonical_visible_pairs) is not tuple
            or any(type(value) is not str or not value for value in self.candidate_ids)
            or any(
                type(value) is not str
                or not value.startswith("sha256:")
                or len(value) != 71
                for value in self.raster_digests
            )
            or any(type(value) is not CountPair for value in self.declared_pairs)
            or any(
                value is not None and type(value) is not CountPair
                for value in self.canonical_visible_pairs
            )
        ):
            raise TypeError("collision field types differ")
        for pair in self.declared_pairs:
            CountPair.__post_init__(pair)
        for pair in self.canonical_visible_pairs:
            if pair is not None:
                CountPair.__post_init__(pair)
        count = len(self.candidate_ids)
        if count < 2 or any(len(values) != count for values in (
            self.raster_digests, self.declared_pairs, self.canonical_visible_pairs
        )):
            raise ValueError("collision fields must align and contain at least two members")
        if self.candidate_ids != tuple(sorted(set(self.candidate_ids))):
            raise ValueError("candidate_ids must be sorted and unique")
        denominator = _exact_int(self.denominator_pixels, "denominator_pixels")
        xor = _exact_int(self.xor_pixels, "xor_pixels")
        union = _exact_int(self.union_pixels, "union_pixels")
        iou = _exact_int(self.iou_millionths, "iou_millionths")
        if denominator != PIXEL_DENOMINATOR or not 0 <= xor <= denominator:
            raise ValueError("collision uses the wrong fixed denominator or XOR range")
        if not 1 <= union <= denominator or not 0 <= iou <= 1_000_000:
            raise ValueError("invalid union or IoU")
        if self.kind == "exact":
            if xor != 0 or len(set(self.raster_digests)) != 1 or iou != 1_000_000:
                raise ValueError("exact collisions require identical rasters and exact metrics")
        else:
            if count != 2 or xor == 0 or len(set(self.raster_digests)) != 2:
                raise ValueError("near collisions require two nonidentical rasters")
            expected = round((union - xor) * 1_000_000 / union)
            if iou != expected:
                raise ValueError("near-collision IoU is inconsistent with XOR and union")

    @property
    def declared_target_conflict(self) -> bool:
        return len(set(self.declared_pairs)) > 1

    @property
    def canonical_target_conflict(self) -> bool:
        return len(set(self.canonical_visible_pairs)) > 1


@dataclass(frozen=True)
class CollisionAudit:
    scope: Literal["bounded_synthetic_only_not_exhaustive"]
    examined_candidate_ids: tuple[str, ...]
    denominator_pixels: int
    possible_different_target_pairs: int
    compared_different_target_pairs: int
    max_near_comparisons: int
    qualifying_near_collision_count: int
    max_retained_near_collisions: int
    exact_collisions: tuple[CollisionClass, ...]
    near_collisions: tuple[CollisionClass, ...]
    exact_canonical_conflict_count: int

    def __post_init__(self) -> None:
        if (
            type(self.scope) is not str
            or self.scope != "bounded_synthetic_only_not_exhaustive"
        ):
            raise ValueError("audit scope must disclose its bounded synthetic status")
        if (
            type(self.examined_candidate_ids) is not tuple
            or any(
                type(value) is not str or not value
                for value in self.examined_candidate_ids
            )
            or type(self.exact_collisions) is not tuple
            or type(self.near_collisions) is not tuple
            or any(type(value) is not CollisionClass for value in self.exact_collisions)
            or any(type(value) is not CollisionClass for value in self.near_collisions)
        ):
            raise TypeError("collision audit field types differ")
        for row in self.exact_collisions + self.near_collisions:
            CollisionClass.__post_init__(row)
        if self.examined_candidate_ids != tuple(sorted(set(self.examined_candidate_ids))):
            raise ValueError("examined candidate IDs must be sorted and unique")
        denominator = _exact_int(self.denominator_pixels, "denominator pixels")
        conflict_count = _exact_int(
            self.exact_canonical_conflict_count, "exact canonical conflict count"
        )
        if denominator != PIXEL_DENOMINATOR:
            raise ValueError("audit denominator must be the fixed canvas area")
        possible = _exact_int(self.possible_different_target_pairs, "possible pairs")
        compared = _exact_int(self.compared_different_target_pairs, "compared pairs")
        maximum = _exact_int(self.max_near_comparisons, "max comparisons")
        qualifying = _exact_int(
            self.qualifying_near_collision_count, "qualifying near collisions"
        )
        retained_maximum = _exact_int(
            self.max_retained_near_collisions, "max retained near collisions"
        )
        if not 0 <= compared <= possible or compared > maximum or maximum < 1:
            raise ValueError("near comparison accounting is inconsistent")
        if (
            not 0 <= qualifying <= compared
            or retained_maximum < 1
            or len(self.near_collisions) != min(qualifying, retained_maximum)
        ):
            raise ValueError("near collision retention accounting is inconsistent")
        if any(row.kind != "exact" or row.denominator_pixels != self.denominator_pixels for row in self.exact_collisions):
            raise ValueError("exact collision collection contains an invalid class")
        if any(row.kind != "near" or row.denominator_pixels != self.denominator_pixels for row in self.near_collisions):
            raise ValueError("near collision collection contains an invalid class")
        expected_conflicts = sum(row.canonical_target_conflict for row in self.exact_collisions)
        if conflict_count != expected_conflicts:
            raise ValueError("exact canonical conflict count is inconsistent")


def _mask_from_png(raw: bytes) -> np.ndarray:
    with Image.open(BytesIO(raw)) as image:
        array = np.asarray(image.convert("L"))
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("audit candidates must use the fixed renderer canvas")
    return array < 128


def audit_collisions(
    candidates: Sequence[AuditCandidate | SyntheticSample],
    *,
    max_near_comparisons: int = 20_000,
    max_near_results: int = 16,
    near_xor_limit: int = 32,
) -> CollisionAudit:
    """Audit exact classes and a bounded prefix of different-history pairs."""

    source_sha256()
    if (
        type(candidates) not in (tuple, list)
        or not 2 <= len(candidates) <= 2048
        or any(type(value) not in (AuditCandidate, SyntheticSample) for value in candidates)
    ):
        raise ValueError("audit accepts between 2 and 2048 synthetic candidates")
    maximum = _exact_int(max_near_comparisons, "max_near_comparisons")
    results_limit = _exact_int(max_near_results, "max_near_results")
    xor_limit = _exact_int(near_xor_limit, "near_xor_limit")
    if maximum < 1 or results_limit < 1 or not 1 <= xor_limit <= PIXEL_DENOMINATOR:
        raise ValueError("near-audit bounds must be positive and in range")
    normalized = []
    for value in candidates:
        if type(value) is AuditCandidate:
            AuditCandidate.__post_init__(value)
            candidate = value
        else:
            SyntheticSample.__post_init__(value)
            candidate = AuditCandidate(value.sample_id, value.panel)
        require_issued_rendered_panel(candidate.panel)
        require_issued_synthetic_png(candidate.panel.png_bytes)
        normalized.append(candidate)
    normalized.sort(key=lambda row: row.candidate_id)
    ids = tuple(row.candidate_id for row in normalized)
    if len(ids) != len(set(ids)):
        raise ValueError("audit candidate identifiers must be unique")
    masks = {row.candidate_id: _mask_from_png(row.panel.png_bytes) for row in normalized}

    by_digest: dict[str, list[AuditCandidate]] = {}
    for row in normalized:
        by_digest.setdefault(row.panel.raster_digest, []).append(row)
    exact: list[CollisionClass] = []
    for members in by_digest.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda row: row.candidate_id)
        union = int(masks[members[0].candidate_id].sum())
        exact.append(CollisionClass(
            "exact",
            tuple(row.candidate_id for row in members),
            tuple(row.panel.raster_digest for row in members),
            tuple(row.panel.declared_pair for row in members),
            tuple(row.panel.canonical_visible_pair for row in members),
            0, union, PIXEL_DENOMINATOR, 1_000_000,
        ))
    exact.sort(key=lambda row: row.candidate_ids)

    different = [
        (left, right)
        for left, right in itertools.combinations(normalized, 2)
        if left.panel.declared_pair != right.panel.declared_pair
    ]
    selected = different[:maximum]
    near: list[CollisionClass] = []
    for left, right in selected:
        left_mask, right_mask = masks[left.candidate_id], masks[right.candidate_id]
        xor = int(np.logical_xor(left_mask, right_mask).sum())
        if xor == 0 or xor > xor_limit:
            continue
        union = int(np.logical_or(left_mask, right_mask).sum())
        near.append(CollisionClass(
            "near",
            (left.candidate_id, right.candidate_id),
            (left.panel.raster_digest, right.panel.raster_digest),
            (left.panel.declared_pair, right.panel.declared_pair),
            (left.panel.canonical_visible_pair, right.panel.canonical_visible_pair),
            xor, union, PIXEL_DENOMINATOR,
            round((union - xor) * 1_000_000 / union),
        ))
    near.sort(key=lambda row: (row.xor_pixels, -row.iou_millionths, row.candidate_ids))
    qualifying_near_count = len(near)
    near = near[:results_limit]
    return CollisionAudit(
        "bounded_synthetic_only_not_exhaustive",
        ids,
        PIXEL_DENOMINATOR,
        len(different),
        len(selected),
        maximum,
        qualifying_near_count,
        results_limit,
        tuple(exact),
        tuple(near),
        sum(row.canonical_target_conflict for row in exact),
    )


def ambiguity_audit(nuisance: Nuisance | None = None) -> CollisionAudit:
    if nuisance is None:
        nuisance = Nuisance()
    candidates: list[AuditCandidate] = []
    for case in ambiguity_cases():
        candidates.extend((
            AuditCandidate(f"{case.case_id}:left", render_program(case.left, nuisance)),
            AuditCandidate(f"{case.case_id}:right", render_program(case.right, nuisance)),
        ))
    # This named fixture inventory is small enough to compare exhaustively;
    # the separate balanced-corpus audit keeps its preregistered prefix bound.
    return audit_collisions(candidates, max_near_comparisons=512, near_xor_limit=64)


__all__ = (
    "ActionPixelProvenance", "AmbiguityCase", "ArcAction", "AuditCandidate",
    "CARRIER_FAMILIES", "CollisionAudit", "CollisionClass", "CorpusSplit",
    "CountPair", "D4_NAMES", "LineAction", "Nuisance",
    "Point", "Program", "RenderedPanel", "SYNTHETIC_SCOPE", "SyntheticSample",
    "ambiguity_audit", "ambiguity_cases", "audit_collisions",
    "build_balanced_corpus", "build_identifiability_counterfactuals",
    "canonical_visible_pair", "carrier_disjoint_split", "default_nuisances",
    "d4_raster_orbit_digest", "has_bounded_exact_single_arc_explanation",
    "has_bounded_exact_single_line_explanation",
    "render_program", "require_issued_rendered_panel",
    "require_issued_synthetic_png", "source_sha256", "valid_count_pairs",
    "visible_raster_component_normal_form",
)
