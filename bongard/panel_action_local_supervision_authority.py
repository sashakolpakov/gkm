"""Pose-free local supervision from the exposed ShapeBongard action programs.

The official ``*_action_programs.json`` values contain the ordered drawing
actions, but not the pose, scale, painter RNG state, retry history, or
PostScript rasterization state used for the released PNG.  Consequently this
module certifies only pose-free construction metadata.  In particular it does
*not* manufacture pixel masks or pixel coordinates.

The public loader is deliberately bound to the already-exposed v3 development
cohort (TRAIN/validation only).  It scans the pinned HD action-program file but
materializes values only for that allowlist.  Calibration, evaluation, target
family, query pixels, PNGs, and label manifests are not reachable here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json


AUTHORITY_SCHEMA = "gkm.bongard-pose-free-local-action-authority.v1"
SUPERVISION_SCHEMA = "gkm.bongard-pose-free-local-action-supervision.v1"
ALGORITHM_ID = "pose-free-action-and-internal-junction-multisets/v1"

DEVELOPMENT_SCHEMA = "gkm.bongard-action-count-cnn-development-panel-ids.v3"
DEVELOPMENT_CLAIM = "exact-v2-train-and-validation-panel-identifiers-only"
DEVELOPMENT_RECORD_DIGEST = (
    "sha256:ee02e48ea3e07dd4804ad24e5c1c9228addc4a0fe658efe821993451bc749fde"
)
DEVELOPMENT_SOURCE_SHA256 = (
    "sha256:9f0c8957bd1be7885022c0bf12d8104c531eea36b1680b902406c1b5e39923db"
)
HD_ACTION_PROGRAM_SOURCE_SHA256 = (
    "sha256:190f3f850d98fa9df0f85cbbafa05fbbaf6d8845586c186ce062af8812ba7e7c"
)
UPSTREAM_BONGARD_SOURCE_SHA256 = (
    "sha256:71454672264d99fc50f44042854922ce9f39a63b4adba5c42535106162ab2961"
)
UPSTREAM_PAINTER_SOURCE_SHA256 = (
    "sha256:aa006b814863f25057caceb3a12fc67d2691bca88acf8e70cd9beae8e606ab02"
)
UPSTREAM_COMMIT = "9df7c78ee9c6a2ff041b48d9ed407359aac259c3"

EXPECTED_COHORT_COUNTS = {
    "train": (800, 11_200),
    "validation": (100, 1_400),
}
EXPECTED_ACTION_COUNT_HISTOGRAM = {
    "train": {1: 78, 2: 357, 3: 640, 4: 1981, 5: 1618, 6: 2679,
              7: 1660, 8: 1743, 9: 444},
    "validation": {1: 7, 2: 46, 3: 89, 4: 241, 5: 182, 6: 368,
                   7: 199, 8: 213, 9: 55},
}

TARGET_FAMILY_PREFIX = "hd_convex-has_four_straight_lines_"
KNOWN_STYLES = frozenset({"normal", "zigzag", "circle", "square", "triangle"})
MAX_ACTION_FILE_BYTES = 32 * 1024 * 1024
MAX_ACTIONS_PER_SHAPE = 9
MAX_SHAPES_PER_PANEL = 2

_DECIMAL = r"(?:0\.[0-9]{3}|1\.000)"
_STYLE = r"(?P<style>[a-z]+)"
_LINE = re.compile(
    rf"line_{_STYLE}_(?P<length>{_DECIMAL})-(?P<turn>{_DECIMAL})\Z"
)
_ARC = re.compile(
    rf"arc_{_STYLE}_(?P<radius>{_DECIMAL})_(?P<sweep>{_DECIMAL})-"
    rf"(?P<turn>{_DECIMAL})\Z"
)
_PANEL_ID = re.compile(
    r"hd/(?P<task>hd_[a-z0-9_-]+_[0-9]{4})/(?P<folder>[01])/"
    r"(?P<panel>[0-6])\.png\Z"
)
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class LocalSupervisionError(RuntimeError):
    """A source/custody invariant differs from the frozen authority."""


class Disposition(str, Enum):
    CERTIFIED = "CERTIFIED"
    GAP = "GAP"


@dataclass(frozen=True, slots=True)
class IntegerInterval:
    lower: int
    upper: int
    unit: str

    def __post_init__(self) -> None:
        if type(self.lower) is not int or type(self.upper) is not int:
            raise TypeError("interval endpoints must be exact integers")
        if self.lower > self.upper:
            raise ValueError("interval lower exceeds upper")
        if not self.unit:
            raise ValueError("interval unit is empty")

    def to_data(self) -> dict[str, object]:
        return {"lower": self.lower, "upper": self.upper, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class TypedGap:
    code: str
    detail: str

    def to_data(self) -> dict[str, object]:
        return {
            "code": self.code,
            "detail": self.detail,
            "disposition": Disposition.GAP.value,
        }


@dataclass(frozen=True, slots=True)
class CarrierToken:
    primitive: str
    length_source_normalized_milli: int | None = None
    length_normalized_micro_interval: IntegerInterval | None = None
    radius_source_normalized_milli: int | None = None
    radius_normalized_micro_interval: IntegerInterval | None = None
    sweep_magnitude_source_degrees_milli: int | None = None
    sweep_magnitude_degrees_milli_interval: IntegerInterval | None = None

    def __post_init__(self) -> None:
        line_fields = (
            self.length_source_normalized_milli,
            self.length_normalized_micro_interval,
        )
        arc_fields = (
            self.radius_source_normalized_milli,
            self.radius_normalized_micro_interval,
            self.sweep_magnitude_source_degrees_milli,
            self.sweep_magnitude_degrees_milli_interval,
        )
        if self.primitive == "line":
            if any(value is None for value in line_fields) or any(
                value is not None for value in arc_fields
            ):
                raise ValueError("line carrier fields differ")
        elif self.primitive == "arc":
            if any(value is None for value in arc_fields) or any(
                value is not None for value in line_fields
            ):
                raise ValueError("arc carrier fields differ")
        else:
            raise ValueError("unsupported carrier primitive")

    def to_data(self) -> dict[str, object]:
        if self.primitive == "line":
            assert self.length_normalized_micro_interval is not None
            return {
                "length_normalized_micro_interval": (
                    self.length_normalized_micro_interval.to_data()
                ),
                "length_source_normalized_milli": (
                    self.length_source_normalized_milli
                ),
                "primitive": "line",
            }
        assert self.radius_normalized_micro_interval is not None
        assert self.sweep_magnitude_degrees_milli_interval is not None
        return {
            "primitive": "arc",
            "radius_normalized_micro_interval": (
                self.radius_normalized_micro_interval.to_data()
            ),
            "radius_source_normalized_milli": self.radius_source_normalized_milli,
            "sweep_magnitude_degrees_milli_interval": (
                self.sweep_magnitude_degrees_milli_interval.to_data()
            ),
            "sweep_magnitude_source_degrees_milli": (
                self.sweep_magnitude_source_degrees_milli
            ),
        }

    def sort_key(self) -> bytes:
        return canonical_json(self.to_data())


@dataclass(frozen=True, slots=True)
class JunctionToken:
    carriers: tuple[CarrierToken, CarrierToken]
    turn_magnitude_source_degrees_milli: int
    turn_magnitude_degrees_milli_interval: IntegerInterval

    def __post_init__(self) -> None:
        if len(self.carriers) != 2:
            raise ValueError("junction must bind two carriers")
        if self.carriers[0].sort_key() > self.carriers[1].sort_key():
            raise ValueError("junction carrier pair is not canonical")

    def to_data(self) -> dict[str, object]:
        return {
            "carriers": [carrier.to_data() for carrier in self.carriers],
            "turn_magnitude_degrees_milli_interval": (
                self.turn_magnitude_degrees_milli_interval.to_data()
            ),
            "turn_magnitude_source_degrees_milli": (
                self.turn_magnitude_source_degrees_milli
            ),
        }

    def sort_key(self) -> bytes:
        return canonical_json(self.to_data())


@dataclass(frozen=True, slots=True)
class ShapeToken:
    action_multiset: tuple[tuple[CarrierToken, int], ...]
    internal_junction_multiset: tuple[tuple[JunctionToken, int], ...]
    action_count: int

    def to_data(self) -> dict[str, object]:
        return {
            "action_count": self.action_count,
            "action_multiset": [
                {**token.to_data(), "multiplicity": multiplicity}
                for token, multiplicity in self.action_multiset
            ],
            "internal_junction_multiset": [
                {**token.to_data(), "multiplicity": multiplicity}
                for token, multiplicity in self.internal_junction_multiset
            ],
        }

    def sort_key(self) -> bytes:
        return canonical_json(self.to_data())


@dataclass(frozen=True, slots=True)
class PoseFreePanelSupervision:
    panel_id: str
    cohort: str
    action_program_sha256: str
    authority_record_digest: str
    disposition: Disposition
    shape_multiset: tuple[tuple[ShapeToken, int], ...] = ()
    carrier_instance_count: int | None = None
    shape_instance_count: int | None = None
    gap: TypedGap | None = None

    def __post_init__(self) -> None:
        if _ADDRESS.fullmatch(self.action_program_sha256) is None:
            raise ValueError("action-program address differs")
        if _ADDRESS.fullmatch(self.authority_record_digest) is None:
            raise ValueError("authority address differs")
        if self.disposition is Disposition.CERTIFIED:
            if (
                self.gap is not None
                or self.carrier_instance_count is None
                or self.shape_instance_count is None
            ):
                raise ValueError("certified supervision carries a gap")
        elif (
            self.gap is None
            or self.shape_multiset
            or self.carrier_instance_count is not None
            or self.shape_instance_count is not None
        ):
            raise ValueError("gap supervision carries certified values")

    def to_data(self) -> dict[str, object]:
        pixel_gap = TypedGap(
            "official_pixel_registration_unavailable",
            "action strings omit official pose, scale, painter RNG/retry state, "
            "and PostScript rasterization",
        ).to_data()
        endpoint_gap = TypedGap(
            "sequence_endpoints_not_visually_identifiable",
            "start/traversal choice is not identifiable under rotation, reversal, "
            "and cyclic closed-shape representations",
        ).to_data()
        result: dict[str, object] = {
            "action_program_sha256": self.action_program_sha256,
            "algorithm_id": ALGORITHM_ID,
            "authority_record_digest": self.authority_record_digest,
            "cohort": self.cohort,
            "disposition": self.disposition.value,
            "panel_id": self.panel_id,
            "pixel_instance_assignment": pixel_gap,
            "pixel_registration": pixel_gap,
            "schema": SUPERVISION_SCHEMA,
            "sequence_endpoint_localization": endpoint_gap,
        }
        if self.disposition is Disposition.GAP:
            assert self.gap is not None
            result["gap"] = self.gap.to_data()
            return result
        result.update(
            {
                "carrier_instance_count": {
                    "disposition": Disposition.CERTIFIED.value,
                    "value": self.carrier_instance_count,
                },
                "shape_instance_count": {
                    "disposition": Disposition.CERTIFIED.value,
                    "value": self.shape_instance_count,
                },
                "shape_multiset": [
                    {**shape.to_data(), "multiplicity": multiplicity}
                    for shape, multiplicity in self.shape_multiset
                ],
                "supervision_semantics": {
                    "action_order": "discarded",
                    "closing_boundary": "not_claimed",
                    "internal_junctions": "unordered_multiset_within_shape",
                    "shape_order": "discarded",
                    "signed_turn_and_sweep": "discarded_as_traversal_dependent",
                    "stored_token_centers": "exact_operational_labels",
                    "underlying_geometry": "conservative_export_rounding_intervals",
                },
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class DevelopmentActionAuthority:
    cohort_panel_ids: tuple[tuple[str, tuple[str, ...]], ...]
    cohort_task_ids: tuple[tuple[str, tuple[str, ...]], ...]
    selected_programs: tuple[tuple[str, str, bytes], ...]
    record_json: bytes
    record_digest: str
    _seal: "_AuthoritySeal | None" = None

    def program_for(self, task_id: str) -> object:
        if self._seal is None:
            raise LocalSupervisionError("authority lacks its builder seal")
        try:
            address, payload = self._seal.program_by_task[task_id]
        except KeyError as exc:
            raise LocalSupervisionError(
                f"task {task_id!r} is outside development authority"
            ) from exc
        # Stored canonical bytes are immutable.  Every consumer gets a fresh
        # parsed value, so mutation cannot alter later labels.
        if _sha256(payload) != address:
            raise LocalSupervisionError("selected program bytes differ from seal")
        return json.loads(payload)

    def cohort_for_panel(self, panel_id: str) -> str:
        if self._seal is not None:
            if panel_id in self._seal.train_panels:
                return "train"
            if panel_id in self._seal.validation_panels:
                return "validation"
        for cohort, values in self.cohort_panel_ids:
            if panel_id in values:
                return cohort
        raise LocalSupervisionError(f"panel {panel_id!r} is outside development authority")

    def to_record(self) -> dict[str, object]:
        value = json.loads(self.record_json)
        if type(value) is not dict:
            raise LocalSupervisionError("internal authority record differs")
        return value


_AUTHORITY_BUILDER_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _AuthoritySeal:
    builder_token: object
    cohort_task_rows_digest: str
    cohort_panel_rows_digest: str
    program_inventory_digest: str
    program_inventory: frozenset[tuple[str, str]]
    program_by_task: Mapping[str, tuple[str, bytes]]
    record_digest: str
    record_json_sha256: str
    train_panels: frozenset[str]
    validation_panels: frozenset[str]


def _cohort_rows_digest(rows: object) -> str:
    return "sha256:" + canonical_digest(rows)


def _program_inventory_digest(
    rows: Sequence[tuple[str, str, bytes]],
) -> str:
    return "sha256:" + canonical_digest(
        [[task_id, address] for task_id, address, _payload in rows]
    )


def verify_development_authority(
    authority: DevelopmentActionAuthority, *, panel_id: str | None = None
) -> None:
    """Reject naked, replaced, or target-bearing authority objects.

    The large program values are immutable bytes.  The seal binds their ordered
    content-address inventory; ``program_for`` checks the requested bytes
    against that address before parsing.  This keeps per-panel verification
    small without trusting a caller-supplied dataclass.
    """

    if type(authority) is not DevelopmentActionAuthority:
        raise LocalSupervisionError("authority has the wrong exact type")
    seal = authority._seal
    if seal is None or seal.builder_token is not _AUTHORITY_BUILDER_TOKEN:
        raise LocalSupervisionError("authority was not produced by the frozen loader")
    if (
        seal.record_digest != authority.record_digest
        or seal.record_json_sha256 != _sha256(authority.record_json)
    ):
        raise LocalSupervisionError("authority record differs from its builder seal")
    if panel_id is not None:
        if panel_id not in seal.train_panels and panel_id not in seal.validation_panels:
            raise LocalSupervisionError(
                f"panel {panel_id!r} is outside sealed development authority"
            )
        return
    if (
        seal.cohort_task_rows_digest
        != _cohort_rows_digest(authority.cohort_task_ids)
        or seal.cohort_panel_rows_digest
        != _cohort_rows_digest(authority.cohort_panel_ids)
        or seal.program_inventory_digest
        != _program_inventory_digest(authority.selected_programs)
        or seal.program_inventory != frozenset(
            (task, address) for task, address, _payload in authority.selected_programs
        )
        or any(
            seal.program_by_task.get(task) != (address, payload)
            for task, address, payload in authority.selected_programs
        )
    ):
        raise LocalSupervisionError("authority content differs from its builder seal")
    try:
        record = json.loads(authority.record_json)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LocalSupervisionError(f"authority record cannot be decoded: {exc}") from exc
    if type(record) is not dict or canonical_json(record) != authority.record_json:
        raise LocalSupervisionError("authority record is not canonical JSON")
    body = dict(record)
    digest = body.pop("record_digest", None)
    if (
        digest != authority.record_digest
        or digest != "sha256:" + canonical_digest(body)
        or record.get("schema") != AUTHORITY_SCHEMA
        or record.get("algorithm_id") != ALGORITHM_ID
    ):
        raise LocalSupervisionError("authority record digest/schema differs")
    bindings = record.get("bindings")
    if type(bindings) is not dict or bindings != {
        "development_manifest_record_digest": DEVELOPMENT_RECORD_DIGEST,
        "development_manifest_source_sha256": DEVELOPMENT_SOURCE_SHA256,
        "hd_action_program_source_sha256": HD_ACTION_PROGRAM_SOURCE_SHA256,
        "module_source_sha256": _module_source_sha256(),
        "upstream_bongard_commit": UPSTREAM_COMMIT,
        "upstream_bongard_source_sha256": UPSTREAM_BONGARD_SOURCE_SHA256,
        "upstream_painter_source_sha256": UPSTREAM_PAINTER_SOURCE_SHA256,
    }:
        raise LocalSupervisionError("authority source bindings differ")
    if tuple(name for name, _values in authority.cohort_task_ids) != (
        "train", "validation"
    ) or tuple(name for name, _values in authority.cohort_panel_ids) != (
        "train", "validation"
    ):
        raise LocalSupervisionError("authority cohort order/scope differs")
    task_map = dict(authority.cohort_task_ids)
    panel_map = dict(authority.cohort_panel_ids)
    all_tasks: set[str] = set()
    all_panels: set[str] = set()
    for cohort in ("train", "validation"):
        tasks = task_map[cohort]
        panels = panel_map[cohort]
        expected_task_count, expected_panel_count = EXPECTED_COHORT_COUNTS[cohort]
        if (
            len(tasks) != expected_task_count
            or len(panels) != expected_panel_count
            or len(set(tasks)) != len(tasks)
            or len(set(panels)) != len(panels)
            or all_tasks.intersection(tasks)
            or all_panels.intersection(panels)
            or any(task.startswith(TARGET_FAMILY_PREFIX) for task in tasks)
        ):
            raise LocalSupervisionError("authority identifiers differ")
        expected_panels = tuple(
            f"hd/{task}/{folder}/{panel}.png"
            for task in tasks
            for folder in (1, 0)
            for panel in range(7)
        )
        if panels != expected_panels:
            raise LocalSupervisionError("authority panel expansion differs")
        all_tasks.update(tasks)
        all_panels.update(panels)
    program_keys = tuple(task for task, _address, _payload in authority.selected_programs)
    if (
        len(set(program_keys)) != len(program_keys)
        or set(program_keys) != all_tasks
        or program_keys != tuple(sorted(program_keys))
        or any(_ADDRESS.fullmatch(address) is None
               for _task, address, _payload in authority.selected_programs)
        or any(type(payload) is not bytes
               for _task, _address, payload in authority.selected_programs)
    ):
        raise LocalSupervisionError("authority selected-program inventory differs")


@dataclass(frozen=True, slots=True)
class _ParsedAction:
    carrier: CarrierToken
    turn_source_degrees_milli: int
    turn_degrees_milli_interval: tuple[int, int]


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _module_source_sha256() -> str:
    return _sha256(Path(__file__).resolve().read_bytes())


def _milli(value: str) -> int:
    whole, fraction = value.split(".")
    return int(whole) * 1000 + int(fraction)


def _normalized_micro_interval(source_milli: int) -> IntegerInterval:
    return IntegerInterval(
        max(0, source_milli * 1000 - 500),
        min(1_000_000, source_milli * 1000 + 500),
        "normalized_micro",
    )


def _floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def _ceil_fraction(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _signed_degree_milli_interval(
    source_milli: int, *, span_degrees: int
) -> tuple[int, int]:
    normalized = _normalized_micro_interval(source_milli)
    lower = Fraction(normalized.lower * span_degrees, 1000) - (
        span_degrees * 500
    )
    upper = Fraction(normalized.upper * span_degrees, 1000) - (
        span_degrees * 500
    )
    return _floor_fraction(lower), _ceil_fraction(upper)


def _magnitude_interval(lower: int, upper: int) -> IntegerInterval:
    if lower <= 0 <= upper:
        magnitude_lower = 0
    else:
        magnitude_lower = min(abs(lower), abs(upper))
    return IntegerInterval(
        magnitude_lower,
        max(abs(lower), abs(upper)),
        "degree_milli",
    )


def _parse_action(value: object) -> _ParsedAction | TypedGap:
    if type(value) is not str:
        return TypedGap("unsupported_action_value", "action is not a string")
    match = _LINE.fullmatch(value)
    if match is not None:
        if match.group("style") not in KNOWN_STYLES:
            return TypedGap("unsupported_style", match.group("style"))
        length = _milli(match.group("length"))
        if length == 0:
            return TypedGap("degenerate_line", "zero-length line carrier")
        turn = _milli(match.group("turn"))
        turn_bounds = _signed_degree_milli_interval(turn, span_degrees=360)
        return _ParsedAction(
            CarrierToken(
                "line",
                length_source_normalized_milli=length,
                length_normalized_micro_interval=_normalized_micro_interval(length),
            ),
            turn * 360 - 180_000,
            turn_bounds,
        )
    match = _ARC.fullmatch(value)
    if match is not None:
        if match.group("style") not in KNOWN_STYLES:
            return TypedGap("unsupported_style", match.group("style"))
        radius = _milli(match.group("radius"))
        sweep = _milli(match.group("sweep"))
        sweep_source = sweep * 720 - 360_000
        if radius == 0 or sweep_source == 0:
            return TypedGap("degenerate_arc", "zero-radius or zero-sweep arc carrier")
        sweep_bounds = _signed_degree_milli_interval(sweep, span_degrees=720)
        turn = _milli(match.group("turn"))
        turn_bounds = _signed_degree_milli_interval(turn, span_degrees=360)
        return _ParsedAction(
            CarrierToken(
                "arc",
                radius_source_normalized_milli=radius,
                radius_normalized_micro_interval=_normalized_micro_interval(radius),
                sweep_magnitude_source_degrees_milli=abs(sweep_source),
                sweep_magnitude_degrees_milli_interval=_magnitude_interval(
                    *sweep_bounds
                ),
            ),
            turn * 360 - 180_000,
            turn_bounds,
        )
    return TypedGap(
        "unsupported_action_syntax",
        "only pinned line/arc actions with known styles and three decimals are supported",
    )


def _compress_tokens(tokens: Iterable[Any]) -> tuple[tuple[Any, int], ...]:
    by_key: dict[bytes, Any] = {}
    counts: Counter[bytes] = Counter()
    for token in tokens:
        key = token.sort_key()
        by_key[key] = token
        counts[key] += 1
    return tuple((by_key[key], counts[key]) for key in sorted(counts))


def _compile_shape(value: object) -> ShapeToken | TypedGap:
    if type(value) is not list or not 1 <= len(value) <= MAX_ACTIONS_PER_SHAPE:
        return TypedGap(
            "unsupported_action_capacity",
            f"shape action count must be 1..{MAX_ACTIONS_PER_SHAPE}",
        )
    parsed: list[_ParsedAction] = []
    for action in value:
        item = _parse_action(action)
        if isinstance(item, TypedGap):
            return item
        parsed.append(item)
    junctions: list[JunctionToken] = []
    for previous, current in zip(parsed, parsed[1:]):
        pair = tuple(sorted((previous.carrier, current.carrier), key=lambda x: x.sort_key()))
        junctions.append(
            JunctionToken(
                (pair[0], pair[1]),
                abs(current.turn_source_degrees_milli),
                _magnitude_interval(*current.turn_degrees_milli_interval),
            )
        )
    return ShapeToken(
        action_multiset=_compress_tokens(item.carrier for item in parsed),
        internal_junction_multiset=_compress_tokens(junctions),
        action_count=len(parsed),
    )


def _compile_image_program(
    *, panel_id: str, cohort: str, image_program: object, authority_record_digest: str
) -> PoseFreePanelSupervision:
    program_address = _sha256(canonical_json(image_program))
    if type(image_program) is not list or not 1 <= len(image_program) <= MAX_SHAPES_PER_PANEL:
        return PoseFreePanelSupervision(
            panel_id,
            cohort,
            program_address,
            authority_record_digest,
            Disposition.GAP,
            gap=TypedGap(
                "unsupported_shape_capacity",
                f"panel shape count must be 1..{MAX_SHAPES_PER_PANEL}",
            ),
        )
    shapes: list[ShapeToken] = []
    for shape in image_program:
        compiled = _compile_shape(shape)
        if isinstance(compiled, TypedGap):
            return PoseFreePanelSupervision(
                panel_id,
                cohort,
                program_address,
                authority_record_digest,
                Disposition.GAP,
                gap=compiled,
            )
        shapes.append(compiled)
    return PoseFreePanelSupervision(
        panel_id,
        cohort,
        program_address,
        authority_record_digest,
        Disposition.CERTIFIED,
        shape_multiset=_compress_tokens(shapes),
        carrier_instance_count=sum(shape.action_count for shape in shapes),
        shape_instance_count=len(shapes),
    )


def compile_pose_free_panel(
    authority: DevelopmentActionAuthority, panel_id: str
) -> PoseFreePanelSupervision:
    """Compile one allowlisted development panel without reading its PNG."""

    verify_development_authority(authority, panel_id=panel_id)
    assert authority._seal is not None
    cohort = "train" if panel_id in authority._seal.train_panels else "validation"
    match = _PANEL_ID.fullmatch(panel_id)
    if match is None:
        raise LocalSupervisionError("development panel ID syntax differs")
    task_id = match.group("task")
    folder = int(match.group("folder"))
    panel_index = int(match.group("panel"))
    value = authority.program_for(task_id)
    try:
        image_program = value[0 if folder == 1 else 1][panel_index]  # type: ignore[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise LocalSupervisionError(
            f"action-program structure differs for {panel_id}"
        ) from exc
    return _compile_image_program(
        panel_id=panel_id,
        cohort=cohort,
        image_program=image_program,
        authority_record_digest=authority.record_digest,
    )


def _stable_regular_bytes(path: Path, *, maximum: int) -> bytes:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise LocalSupervisionError(f"source is not a regular nonsymlink file: {path}")
    if before.st_size <= 0 or before.st_size > maximum:
        raise LocalSupervisionError(f"source byte size is outside its bound: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            payload = handle.read(maximum + 1)
            after_read = os.fstat(handle.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    after = path.lstat()
    fingerprint = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    if not (fingerprint(before) == fingerprint(opened) == fingerprint(after_read)
            == fingerprint(after)):
        raise LocalSupervisionError(f"source changed during read: {path}")
    if len(payload) != before.st_size or len(payload) > maximum:
        raise LocalSupervisionError(f"source byte count differs: {path}")
    return payload


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _skip_json_value(text: str, index: int) -> int:
    """Lexically skip a JSON value without materializing its nested content."""

    index = _skip_ws(text, index)
    if index >= len(text):
        raise LocalSupervisionError("truncated action-program JSON")
    opening = text[index]
    if opening not in "[{\"":
        while index < len(text) and text[index] not in ",}":
            index += 1
        return index
    if opening == '"':
        index += 1
        escaped = False
        while index < len(text):
            character = text[index]
            index += 1
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                return index
        raise LocalSupervisionError("unterminated JSON string")
    stack = [opening]
    index += 1
    in_string = False
    escaped = False
    while index < len(text) and stack:
        character = text[index]
        index += 1
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}":
            expected = "[" if character == "]" else "{"
            if not stack or stack.pop() != expected:
                raise LocalSupervisionError("unbalanced action-program JSON")
    if stack or in_string:
        raise LocalSupervisionError("truncated action-program JSON value")
    return index


def _select_top_level_values(payload: bytes, allowed: set[str]) -> dict[str, object]:
    """Materialize only allowlisted values from the pinned top-level object."""

    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise LocalSupervisionError(f"action programs are not UTF-8: {exc}") from exc
    decoder = json.JSONDecoder()
    index = _skip_ws(text, 0)
    if index >= len(text) or text[index] != "{":
        raise LocalSupervisionError("action-program root is not an object")
    index += 1
    seen: set[str] = set()
    selected: dict[str, object] = {}
    while True:
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == "}":
            index += 1
            break
        try:
            key, index = decoder.raw_decode(text, index)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise LocalSupervisionError(f"cannot decode action-program key: {exc}") from exc
        if type(key) is not str or key in seen:
            raise LocalSupervisionError("action-program key is nonstring or duplicated")
        seen.add(key)
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            raise LocalSupervisionError("action-program key lacks colon")
        index = _skip_ws(text, index + 1)
        if key in allowed:
            try:
                value, index = decoder.raw_decode(text, index)
            except (json.JSONDecodeError, RecursionError) as exc:
                raise LocalSupervisionError(
                    f"cannot decode selected action program {key}: {exc}"
                ) from exc
            selected[key] = value
        else:
            index = _skip_json_value(text, index)
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            index += 1
            break
        raise LocalSupervisionError("action-program object separator differs")
    if _skip_ws(text, index) != len(text):
        raise LocalSupervisionError("trailing action-program JSON content")
    missing = allowed.difference(selected)
    if missing:
        raise LocalSupervisionError(
            f"{len(missing)} development tasks are missing from action programs"
        )
    return selected


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _stable_regular_bytes(path, maximum=4 * 1024 * 1024)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LocalSupervisionError(f"cannot decode development manifest: {exc}") from exc
    if type(value) is not dict or raw != canonical_json(value) + b"\n":
        raise LocalSupervisionError("development manifest is not canonical JSON plus newline")
    body = dict(value)
    digest = body.pop("record_digest", None)
    if (
        value.get("schema") != DEVELOPMENT_SCHEMA
        or value.get("claim") != DEVELOPMENT_CLAIM
        or digest != DEVELOPMENT_RECORD_DIGEST
        or digest != "sha256:" + canonical_digest(body)
        or _sha256(raw) != DEVELOPMENT_SOURCE_SHA256
    ):
        raise LocalSupervisionError("development manifest binding differs")
    return value, raw


def _verify_manifest_scope(
    manifest: Mapping[str, Any],
) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    cohorts = manifest.get("cohorts")
    if type(cohorts) is not dict or set(cohorts) != set(EXPECTED_COHORT_COUNTS):
        raise LocalSupervisionError("only train and validation cohorts are authorized")
    task_rows: list[tuple[str, tuple[str, ...]]] = []
    panel_rows: list[tuple[str, tuple[str, ...]]] = []
    all_tasks: set[str] = set()
    all_panels: set[str] = set()
    for cohort in ("train", "validation"):
        row = cohorts[cohort]
        if type(row) is not dict or set(row) != {"task_ids", "panel_ids"}:
            raise LocalSupervisionError(f"{cohort} manifest row differs")
        tasks = tuple(row["task_ids"])
        panels = tuple(row["panel_ids"])
        expected_tasks, expected_panels = EXPECTED_COHORT_COUNTS[cohort]
        if len(tasks) != expected_tasks or len(panels) != expected_panels:
            raise LocalSupervisionError(f"{cohort} count differs")
        if any(type(value) is not str for value in tasks + panels):
            raise LocalSupervisionError(f"{cohort} identifier type differs")
        if len(set(tasks)) != len(tasks) or len(set(panels)) != len(panels):
            raise LocalSupervisionError(f"{cohort} identifiers are duplicated")
        if any(task.startswith(TARGET_FAMILY_PREFIX) for task in tasks):
            raise LocalSupervisionError("sealed target-family task entered authority")
        expected = tuple(
            f"hd/{task}/{folder}/{panel}.png"
            for task in tasks
            for folder in (1, 0)
            for panel in range(7)
        )
        if panels != expected:
            raise LocalSupervisionError(f"{cohort} panel expansion/order differs")
        if all_tasks.intersection(tasks) or all_panels.intersection(panels):
            raise LocalSupervisionError("development cohorts overlap")
        all_tasks.update(tasks)
        all_panels.update(panels)
        task_rows.append((cohort, tasks))
        panel_rows.append((cohort, panels))
    return tuple(task_rows), tuple(panel_rows)


def _program_histogram(
    programs: Mapping[str, object], cohort_tasks: Sequence[tuple[str, tuple[str, ...]]]
) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for cohort, tasks in cohort_tasks:
        counts: Counter[int] = Counter()
        for task in tasks:
            value = programs[task]
            if type(value) is not list or len(value) != 2:
                raise LocalSupervisionError(f"task program structure differs: {task}")
            for side in value:
                if type(side) is not list or len(side) != 7:
                    raise LocalSupervisionError(f"task side structure differs: {task}")
                for image in side:
                    if type(image) is not list or not 1 <= len(image) <= MAX_SHAPES_PER_PANEL:
                        raise LocalSupervisionError(f"panel shape capacity differs: {task}")
                    action_count = 0
                    for shape in image:
                        if type(shape) is not list:
                            raise LocalSupervisionError(f"shape program type differs: {task}")
                        action_count += len(shape)
                    counts[action_count] += 1
        result[cohort] = dict(sorted(counts.items()))
    return result


def _load_development_authority_bound(
    *,
    manifest_path: Path,
    action_program_path: Path,
    upstream_bongard_path: Path,
    upstream_painter_path: Path,
    expected_manifest_source: str,
    expected_action_source: str,
    expected_bongard_source: str,
    expected_painter_source: str,
    verify_fixed_histogram: bool,
) -> DevelopmentActionAuthority:
    manifest, _manifest_raw = _load_manifest(manifest_path)
    cohort_tasks, cohort_panels = _verify_manifest_scope(manifest)
    all_tasks = {task for _cohort, tasks in cohort_tasks for task in tasks}

    action_raw = _stable_regular_bytes(action_program_path, maximum=MAX_ACTION_FILE_BYTES)
    if _sha256(action_raw) != expected_action_source:
        raise LocalSupervisionError("HD action-program source binding differs")
    programs = _select_top_level_values(action_raw, all_tasks)

    for path, expected, label in (
        (upstream_bongard_path, expected_bongard_source, "upstream action semantics"),
        (upstream_painter_path, expected_painter_source, "upstream painter semantics"),
    ):
        raw = _stable_regular_bytes(path, maximum=1024 * 1024)
        if _sha256(raw) != expected:
            raise LocalSupervisionError(f"{label} source binding differs")
    if expected_manifest_source != DEVELOPMENT_SOURCE_SHA256:
        raise LocalSupervisionError("test override cannot change manifest authority")

    histogram = _program_histogram(programs, cohort_tasks)
    if verify_fixed_histogram and histogram != EXPECTED_ACTION_COUNT_HISTOGRAM:
        raise LocalSupervisionError("development action-capacity histogram differs")
    selected_program_values = tuple(sorted(programs.items()))
    body: dict[str, object] = {
        "algorithm_id": ALGORITHM_ID,
        "bindings": {
            "development_manifest_record_digest": DEVELOPMENT_RECORD_DIGEST,
            "development_manifest_source_sha256": DEVELOPMENT_SOURCE_SHA256,
            "hd_action_program_source_sha256": expected_action_source,
            "module_source_sha256": _module_source_sha256(),
            "upstream_bongard_commit": UPSTREAM_COMMIT,
            "upstream_bongard_source_sha256": expected_bongard_source,
            "upstream_painter_source_sha256": expected_painter_source,
        },
        "cohorts": {
            cohort: {
                "action_count_histogram": {
                    str(key): value for key, value in histogram[cohort].items()
                },
                "panel_count": len(dict(cohort_panels)[cohort]),
                "task_count": len(tasks),
            }
            for cohort, tasks in cohort_tasks
        },
        "custody": {
            "action_program_raw_bytes_scanned_for_digest_and_key_selection": True,
            "authorized_cohorts": ["train", "validation"],
            "calibration_or_evaluation_identifiers_opened": 0,
            "label_manifests_opened": 0,
            "nonselected_action_program_values_materialized": 0,
            "png_files_opened": 0,
            "query_or_target_pixels_opened": 0,
            "target_family_prefix_forbidden": TARGET_FAMILY_PREFIX,
        },
        "schema": AUTHORITY_SCHEMA,
        "selected_programs_digest": "sha256:" + canonical_digest(
            {key: value for key, value in selected_program_values}
        ),
        "selected_task_ids_digest": _sha256(
            "".join(f"{task}\n" for task in sorted(all_tasks)).encode("utf-8")
        ),
        "semantics": {
            "certified": [
                "exact_serialized_action_token_centers",
                "conservative_export_rounding_intervals",
                "pose_free_action_multisets_per_shape",
                "pose_free_internal_junction_multisets_per_shape",
                "action_carrier_instance_count_from_total_action_count",
                "shape_instance_count_from_shape_program_count",
            ],
            "explicit_gaps": [
                "official_pixel_registration",
                "pixel_instance_assignment",
                "sequence_endpoint_localization",
                "closing_boundary",
            ],
            "never_claimed": [
                "official_pixel_coordinates",
                "official_pixel_masks",
                "signed_direction_under_unobservable_traversal",
            ],
        },
    }
    record_digest = "sha256:" + canonical_digest(body)
    record = {**body, "record_digest": record_digest}
    selected_programs = tuple(
        (key, _sha256(canonical_json(value)), canonical_json(value))
        for key, value in selected_program_values
    )
    record_json = canonical_json(record)
    seal = _AuthoritySeal(
        builder_token=_AUTHORITY_BUILDER_TOKEN,
        cohort_task_rows_digest=_cohort_rows_digest(cohort_tasks),
        cohort_panel_rows_digest=_cohort_rows_digest(cohort_panels),
        program_inventory_digest=_program_inventory_digest(selected_programs),
        program_inventory=frozenset(
            (task, address) for task, address, _payload in selected_programs
        ),
        program_by_task=MappingProxyType(
            {
                task: (address, payload)
                for task, address, payload in selected_programs
            }
        ),
        record_digest=record_digest,
        record_json_sha256=_sha256(record_json),
        train_panels=frozenset(dict(cohort_panels)["train"]),
        validation_panels=frozenset(dict(cohort_panels)["validation"]),
    )
    return DevelopmentActionAuthority(
        cohort_panel_ids=cohort_panels,
        cohort_task_ids=cohort_tasks,
        selected_programs=selected_programs,
        record_json=record_json,
        record_digest=record_digest,
        _seal=seal,
    )


def load_development_authority(
    *,
    repository_root: str | Path | None = None,
) -> DevelopmentActionAuthority:
    """Load the frozen development-only action authority.

    No path to calibration/evaluation manifests, labels, or PNGs is accepted by
    this API.  A caller may relocate the repository root, but cannot override
    any content address.
    """

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    return _load_development_authority_bound(
        manifest_path=root
        / "bongard/data/panel_action_count_cnn_development_panels_20260810_v3.json",
        action_program_path=root
        / "downloads/ShapeBongard_V2_full/ShapeBongard_V2/hd/hd_action_programs.json",
        upstream_bongard_path=root / "downloads/Bongard-LOGO/bongard/bongard.py",
        upstream_painter_path=root
        / "downloads/Bongard-LOGO/bongard/bongard_painter.py",
        expected_manifest_source=DEVELOPMENT_SOURCE_SHA256,
        expected_action_source=HD_ACTION_PROGRAM_SOURCE_SHA256,
        expected_bongard_source=UPSTREAM_BONGARD_SOURCE_SHA256,
        expected_painter_source=UPSTREAM_PAINTER_SOURCE_SHA256,
        verify_fixed_histogram=True,
    )


__all__ = [
    "ALGORITHM_ID",
    "AUTHORITY_SCHEMA",
    "CarrierToken",
    "DevelopmentActionAuthority",
    "Disposition",
    "IntegerInterval",
    "JunctionToken",
    "LocalSupervisionError",
    "PoseFreePanelSupervision",
    "SUPERVISION_SCHEMA",
    "ShapeToken",
    "TypedGap",
    "compile_pose_free_panel",
    "load_development_authority",
    "verify_development_authority",
]
