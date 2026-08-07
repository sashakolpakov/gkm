"""Complete owner-labelled point-contact observations for loop pairs.

This is the production promotion of the useful crack-lab idea, not a wrapper
around its API.  Inputs are exact candidate-independent scenario masks and
source-bound loop witnesses.  Outputs use integer fixed-point measurements,
retain four incident rays and both exterior gaps, and distinguish certified
non-contact from unresolved fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from bongard.artifacts import canonical_digest
from bongard.contour_witnesses import Q16Point
from bongard.evidence import Disposition
from bongard.loop_geometry import (
    IntInterval,
    LoopGeometryWitness,
    boundary_cycles_for_mask,
    loop_geometry_algorithm_digest,
)
from bongard import visual_witnesses as _base


PAIR_CONTACT_SCHEMA = "gkm.bongard-loop-pair-contact-observation.v1"
POINT_CONTACT_SIGNATURE_SCHEMA = "gkm.bongard-point-contact-signature.v1"
INCIDENT_RAY_SCHEMA = "gkm.bongard-point-contact-incident-ray.v1"
EXTERIOR_GAP_SCHEMA = "gkm.bongard-point-contact-exterior-gap.v1"
POINT_CONTACT_ALGORITHM_ID = "bongard.point-contact/four-owned-rays-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LOOP_ID = re.compile(r"loop-[0-9]{8}\Z")
_COMPONENT_ID = re.compile(r"component-[0-9]{8}\Z")
_RAY_ID = re.compile(r"loop-[0-9]{8}:(start|end):boundary-ray\Z")

_DILATION_ITERATIONS = 1
_MAX_NORMALIZED_GAP_PPM = 300_000
_MAX_INTERFACE_SPREAD_PPM = 300_000
_RAY_WINDOW_NUMERATOR = 13
_RAY_WINDOW_DENOMINATOR = 100
_MIN_RAY_POINTS = 7
_MAX_RAY_RESIDUAL_PPM = 90_000
_MAX_NEAR_INTERFACE_PAIRS = 8_192
_SEPARATE_CERTIFICATE = (
    "the two source loops are owned by distinct exact foreground components"
)
_INTERLEAVING_CERTIFICATE = (
    "four uncertainty-ordered ray owners alternate cyclically rather than "
    "forming two loop blocks"
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


def point_contact_source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def point_contact_algorithm_digest() -> str:
    return canonical_digest(
        {
            "algorithm_id": POINT_CONTACT_ALGORITHM_ID,
            "source_digest": point_contact_source_digest(),
            "loop_geometry_algorithm_digest": loop_geometry_algorithm_digest(),
            "source_regions": {
                "background_connectivity": 4,
                "foreground_connectivity": 8,
                "loop_region": "one-pixel foreground-dilated source hole",
                "mapping": "exactly one region per source loop; no overwrite",
                "owner": (
                    "canonical exact foreground component touching hole; "
                    "must equal parent hole owner"
                ),
            },
            "point_gate": {
                "maximum_normalized_gap_ppm": _MAX_NORMALIZED_GAP_PPM,
                "maximum_interface_spread_ppm": _MAX_INTERFACE_SPREAD_PPM,
                "near_pair_additive_pixels": 1,
                "maximum_near_interface_pairs": _MAX_NEAR_INTERFACE_PAIRS,
                "nearest_search": "deterministic scipy cKDTree workers=1",
                "spread": "axis-aligned midpoint-bbox diagonal upper bound",
                "distinct_owner_components": "certified separate",
                "same_owner_gap_failure": "indeterminate",
                "same_owner_spread_failure": "indeterminate",
            },
            "ray_fit": {
                "window_fraction": [
                    _RAY_WINDOW_NUMERATOR,
                    _RAY_WINDOW_DENOMINATOR,
                ],
                "minimum_points": _MIN_RAY_POINTS,
                "maximum_residual_ppm": _MAX_RAY_RESIDUAL_PPM,
            },
            "signature": {
                "loop_count": 2,
                "contact_count": 1,
                "ray_count": 4,
                "rays_per_owner": 2,
                "cyclic_owner_transitions": 2,
                "exterior_gap_count": 2,
            },
        }
    )


class ContactKind(str, Enum):
    POINT = "point"
    SEPARATE = "separate"
    EXTENDED_OR_MULTIPLE = "extended_or_multiple"
    INTERLEAVING = "interleaving"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class IncidentRayWitness:
    ray_id: str
    owner_loop_id: str
    endpoint_name: str
    direction_millidegrees: int
    uncertainty_millidegrees: int
    residual_ppm_upper: int
    endpoint_q16: Q16Point
    source_boundary_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.ray_id, str) or _RAY_ID.fullmatch(self.ray_id) is None:
            raise ValueError("incident ray_id is not canonical")
        if (
            not isinstance(self.owner_loop_id, str)
            or _LOOP_ID.fullmatch(self.owner_loop_id) is None
        ):
            raise ValueError("incident ray owner_loop_id is not canonical")
        if self.endpoint_name not in {"start", "end"}:
            raise ValueError("incident ray endpoint_name must be start or end")
        if self.ray_id != (
            f"{self.owner_loop_id}:{self.endpoint_name}:boundary-ray"
        ):
            raise ValueError("incident ray_id must bind owner and endpoint")
        _integer(self.direction_millidegrees, "ray direction_millidegrees")
        if self.direction_millidegrees >= 360_000:
            raise ValueError("ray direction must be in [0, 360000)")
        _integer(self.uncertainty_millidegrees, "ray uncertainty_millidegrees")
        _integer(self.residual_ppm_upper, "ray residual_ppm_upper")
        if self.residual_ppm_upper > _MAX_RAY_RESIDUAL_PPM:
            raise ValueError("incident ray residual exceeds the frozen gate")
        if not isinstance(self.endpoint_q16, Q16Point):
            raise TypeError("incident ray endpoint_q16 must be a Q16Point")
        _digest(self.source_boundary_digest, "ray source_boundary_digest")
        minimum_uncertainty = int(
            math.ceil(
                math.degrees(math.atan(self.residual_ppm_upper / 1_000_000.0))
                * 1_000.0
            )
        )
        if self.uncertainty_millidegrees < minimum_uncertainty:
            raise ValueError("incident ray uncertainty understates residual")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": INCIDENT_RAY_SCHEMA,
            "ray_id": self.ray_id,
            "owner_loop_id": self.owner_loop_id,
            "endpoint_name": self.endpoint_name,
            "direction_millidegrees": self.direction_millidegrees,
            "uncertainty_millidegrees": self.uncertainty_millidegrees,
            "residual_ppm_upper": self.residual_ppm_upper,
            "endpoint_q16": self.endpoint_q16.to_data(),
            "source_boundary_digest": self.source_boundary_digest,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "IncidentRayWitness":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "ray_id",
                    "owner_loop_id",
                    "endpoint_name",
                    "direction_millidegrees",
                    "uncertainty_millidegrees",
                    "residual_ppm_upper",
                    "endpoint_q16",
                    "source_boundary_digest",
                }
            ),
            "incident ray witness",
        )
        if data["schema"] != INCIDENT_RAY_SCHEMA:
            raise ValueError("unsupported incident ray witness")
        endpoint = data["endpoint_q16"]
        if not isinstance(endpoint, Mapping):
            raise TypeError("incident ray endpoint_q16 must be an object")
        return cls(
            ray_id=data["ray_id"],
            owner_loop_id=data["owner_loop_id"],
            endpoint_name=data["endpoint_name"],
            direction_millidegrees=data["direction_millidegrees"],
            uncertainty_millidegrees=data["uncertainty_millidegrees"],
            residual_ppm_upper=data["residual_ppm_upper"],
            endpoint_q16=Q16Point.from_data(endpoint),
            source_boundary_digest=data["source_boundary_digest"],
        )


@dataclass(frozen=True, slots=True)
class ExteriorGapWitness:
    ray_a_id: str
    ray_b_id: str
    owner_a: str
    owner_b: str
    nominal_millidegrees: int
    interval_millidegrees: IntInterval

    def __post_init__(self) -> None:
        for value, label in (
            (self.ray_a_id, "gap ray_a_id"),
            (self.ray_b_id, "gap ray_b_id"),
        ):
            if not isinstance(value, str) or _RAY_ID.fullmatch(value) is None:
                raise ValueError(f"{label} is not canonical")
        for value, label in ((self.owner_a, "gap owner_a"), (self.owner_b, "gap owner_b")):
            if not isinstance(value, str) or _LOOP_ID.fullmatch(value) is None:
                raise ValueError(f"{label} is not canonical")
        if self.owner_a == self.owner_b:
            raise ValueError("exterior gap must cross two owners")
        _integer(self.nominal_millidegrees, "gap nominal_millidegrees", minimum=1)
        if self.nominal_millidegrees >= 360_000:
            raise ValueError("exterior gap nominal must be below 360 degrees")
        if not isinstance(self.interval_millidegrees, IntInterval):
            raise TypeError("exterior gap interval must be an IntInterval")
        if not (
            0 < self.interval_millidegrees.lower
            <= self.nominal_millidegrees
            <= self.interval_millidegrees.upper
            < 360_000
        ):
            raise ValueError("exterior gap interval does not contain a strict gap")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": EXTERIOR_GAP_SCHEMA,
            "ray_a_id": self.ray_a_id,
            "ray_b_id": self.ray_b_id,
            "owner_a": self.owner_a,
            "owner_b": self.owner_b,
            "nominal_millidegrees": self.nominal_millidegrees,
            "interval_millidegrees": self.interval_millidegrees.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ExteriorGapWitness":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "ray_a_id",
                    "ray_b_id",
                    "owner_a",
                    "owner_b",
                    "nominal_millidegrees",
                    "interval_millidegrees",
                }
            ),
            "exterior gap witness",
        )
        if data["schema"] != EXTERIOR_GAP_SCHEMA:
            raise ValueError("unsupported exterior gap witness")
        interval = data["interval_millidegrees"]
        if not isinstance(interval, Mapping):
            raise TypeError("exterior gap interval must be an object")
        return cls(
            ray_a_id=data["ray_a_id"],
            ray_b_id=data["ray_b_id"],
            owner_a=data["owner_a"],
            owner_b=data["owner_b"],
            nominal_millidegrees=data["nominal_millidegrees"],
            interval_millidegrees=IntInterval.from_data(interval),
        )


@dataclass(frozen=True, slots=True)
class PointContactSignature:
    contact_id: str
    loop_ids: tuple[str, str]
    contact_count: int
    vertex_q16: Q16Point
    normalized_gap_ppm_upper: int
    interface_spread_ppm_upper: int
    rays: tuple[IncidentRayWitness, ...]
    cyclic_owners: tuple[str, ...]
    exterior_gaps: tuple[ExteriorGapWitness, ExteriorGapWitness]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.loop_ids, tuple)
            or len(self.loop_ids) != 2
            or self.loop_ids != tuple(sorted(set(self.loop_ids)))
            or any(_LOOP_ID.fullmatch(item) is None for item in self.loop_ids)
        ):
            raise ValueError("point contact requires two sorted distinct loop IDs")
        if self.contact_id != f"contact:{self.loop_ids[0]}:{self.loop_ids[1]}":
            raise ValueError("point-contact ID must bind its ordered loop pair")
        if isinstance(self.contact_count, bool) or self.contact_count != 1:
            raise ValueError("point-contact signature requires exactly one contact")
        if not isinstance(self.vertex_q16, Q16Point):
            raise TypeError("point-contact vertex_q16 must be a Q16Point")
        _integer(self.normalized_gap_ppm_upper, "normalized gap ppm")
        _integer(self.interface_spread_ppm_upper, "interface spread ppm")
        if self.normalized_gap_ppm_upper > _MAX_NORMALIZED_GAP_PPM:
            raise ValueError("point-contact gap exceeds its validated support")
        if self.interface_spread_ppm_upper > _MAX_INTERFACE_SPREAD_PPM:
            raise ValueError("point-contact interface spread exceeds its gate")
        if not isinstance(self.rays, tuple) or len(self.rays) != 4 or any(
            not isinstance(item, IncidentRayWitness) for item in self.rays
        ):
            raise TypeError("point-contact signature requires four typed rays")
        directions = tuple(item.direction_millidegrees for item in self.rays)
        if any(right <= left for left, right in zip(directions, directions[1:])):
            raise ValueError("point-contact rays must have strict cyclic order")
        for index, ray in enumerate(self.rays):
            following = self.rays[(index + 1) % 4]
            nominal_gap = (
                following.direction_millidegrees - ray.direction_millidegrees
            ) % 360_000
            uncertainty = (
                ray.uncertainty_millidegrees
                + following.uncertainty_millidegrees
            )
            if nominal_gap <= uncertainty:
                raise ValueError(
                    "point-contact cyclic ray order is not uncertainty-certified"
                )
        if len({item.ray_id for item in self.rays}) != 4:
            raise ValueError("point-contact ray IDs must be unique")
        owner_counts = {
            loop_id: sum(item.owner_loop_id == loop_id for item in self.rays)
            for loop_id in self.loop_ids
        }
        if owner_counts != {item: 2 for item in self.loop_ids}:
            raise ValueError("each point-contact loop must own exactly two rays")
        owners = tuple(item.owner_loop_id for item in self.rays)
        if self.cyclic_owners != owners:
            raise ValueError("cyclic_owners must match the ordered rays")
        transitions = sum(
            owners[index] != owners[(index + 1) % 4] for index in range(4)
        )
        if transitions != 2:
            raise ValueError("point-contact owners must form two cyclic blocks")
        if not isinstance(self.exterior_gaps, tuple) or len(self.exterior_gaps) != 2:
            raise TypeError("point-contact signature requires two exterior gaps")
        ray_by_id = {item.ray_id: item for item in self.rays}
        expected_pairs: set[tuple[str, str]] = set()
        for index, ray in enumerate(self.rays):
            following = self.rays[(index + 1) % 4]
            if ray.owner_loop_id != following.owner_loop_id:
                expected_pairs.add((ray.ray_id, following.ray_id))
        if {(item.ray_a_id, item.ray_b_id) for item in self.exterior_gaps} != expected_pairs:
            raise ValueError("exterior gaps do not retain both cross-owner gaps")
        for gap in self.exterior_gaps:
            first, second = ray_by_id[gap.ray_a_id], ray_by_id[gap.ray_b_id]
            if (
                gap.owner_a != first.owner_loop_id
                or gap.owner_b != second.owner_loop_id
            ):
                raise ValueError("exterior gap owners differ from its incident rays")
            nominal = (
                second.direction_millidegrees - first.direction_millidegrees
            ) % 360_000
            uncertainty = (
                first.uncertainty_millidegrees
                + second.uncertainty_millidegrees
            )
            if gap.nominal_millidegrees != nominal or gap.interval_millidegrees != (
                IntInterval(nominal - uncertainty, nominal + uncertainty)
            ):
                raise ValueError("exterior gap does not match its incident rays")
        if self.exterior_gaps != tuple(
            sorted(
                self.exterior_gaps,
                key=lambda item: (
                    item.nominal_millidegrees,
                    item.ray_a_id,
                    item.ray_b_id,
                ),
            )
        ):
            raise ValueError("exterior gaps must be ordered small then large")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": POINT_CONTACT_SIGNATURE_SCHEMA,
            "contact_id": self.contact_id,
            "loop_ids": list(self.loop_ids),
            "contact_count": self.contact_count,
            "vertex_q16": self.vertex_q16.to_data(),
            "normalized_gap_ppm_upper": self.normalized_gap_ppm_upper,
            "interface_spread_ppm_upper": self.interface_spread_ppm_upper,
            "rays": [item.to_data() for item in self.rays],
            "cyclic_owners": list(self.cyclic_owners),
            "exterior_gaps": [item.to_data() for item in self.exterior_gaps],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PointContactSignature":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "contact_id",
                    "loop_ids",
                    "contact_count",
                    "vertex_q16",
                    "normalized_gap_ppm_upper",
                    "interface_spread_ppm_upper",
                    "rays",
                    "cyclic_owners",
                    "exterior_gaps",
                }
            ),
            "point-contact signature",
        )
        if data["schema"] != POINT_CONTACT_SIGNATURE_SCHEMA:
            raise ValueError("unsupported point-contact signature")
        loops = data["loop_ids"]
        rays = data["rays"]
        owners = data["cyclic_owners"]
        gaps = data["exterior_gaps"]
        vertex = data["vertex_q16"]
        if not isinstance(loops, list) or any(not isinstance(item, str) for item in loops):
            raise TypeError("point-contact loop_ids must be a string list")
        if not isinstance(rays, list) or any(not isinstance(item, Mapping) for item in rays):
            raise TypeError("point-contact rays must be an object list")
        if not isinstance(owners, list) or any(not isinstance(item, str) for item in owners):
            raise TypeError("point-contact cyclic_owners must be a string list")
        if not isinstance(gaps, list) or any(not isinstance(item, Mapping) for item in gaps):
            raise TypeError("point-contact exterior_gaps must be an object list")
        if not isinstance(vertex, Mapping):
            raise TypeError("point-contact vertex_q16 must be an object")
        return cls(
            contact_id=data["contact_id"],
            loop_ids=tuple(loops),  # type: ignore[arg-type]
            contact_count=data["contact_count"],
            vertex_q16=Q16Point.from_data(vertex),
            normalized_gap_ppm_upper=data["normalized_gap_ppm_upper"],
            interface_spread_ppm_upper=data["interface_spread_ppm_upper"],
            rays=tuple(IncidentRayWitness.from_data(item) for item in rays),
            cyclic_owners=tuple(owners),
            exterior_gaps=tuple(ExteriorGapWitness.from_data(item) for item in gaps),  # type: ignore[arg-type]
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True, slots=True)
class PairContactObservation:
    loop_ids: tuple[str, str]
    owner_component_ids: tuple[str | None, str | None]
    disposition: Disposition
    contact_kind: ContactKind
    normalized_gap_ppm_upper: int | None
    interface_spread_ppm_upper: int | None
    signature: PointContactSignature | None
    reason_code: str
    certificate: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.loop_ids, tuple)
            or len(self.loop_ids) != 2
            or self.loop_ids != tuple(sorted(set(self.loop_ids)))
            or any(_LOOP_ID.fullmatch(item) is None for item in self.loop_ids)
        ):
            raise ValueError("pair contact requires sorted distinct loop IDs")
        if not isinstance(self.owner_component_ids, tuple) or len(
            self.owner_component_ids
        ) != 2:
            raise TypeError("pair contact owner_component_ids must be a pair")
        if any(
            item is not None
            and (not isinstance(item, str) or _COMPONENT_ID.fullmatch(item) is None)
            for item in self.owner_component_ids
        ):
            raise ValueError("pair contact owner component ID is not canonical or null")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("pair contact disposition must be a Disposition")
        if not isinstance(self.contact_kind, ContactKind):
            raise TypeError("pair contact kind must be a ContactKind")
        for value, label in (
            (self.normalized_gap_ppm_upper, "normalized gap"),
            (self.interface_spread_ppm_upper, "interface spread"),
        ):
            if value is not None:
                _integer(value, label)
        if self.signature is not None and not isinstance(
            self.signature, PointContactSignature
        ):
            raise TypeError("pair contact signature must be typed or null")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("pair contact reason_code must be nonempty")
        for value, label in (
            (self.certificate, "pair contact certificate"),
            (self.error_type, "pair contact error_type"),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{label} must be a nonempty string or null")
        if self.disposition is Disposition.PRESENT:
            if self.contact_kind is not ContactKind.POINT or self.signature is None:
                raise ValueError("present pair contact requires a point signature")
            if self.signature.loop_ids != self.loop_ids:
                raise ValueError("pair contact signature names another pair")
            if self.certificate is not None or self.error_type is not None:
                raise ValueError("present pair contact cannot carry failure fields")
            if self.reason_code != "complete_four_ray_signature":
                raise ValueError("present pair contact reason differs")
            if (
                self.normalized_gap_ppm_upper
                != self.signature.normalized_gap_ppm_upper
                or self.interface_spread_ppm_upper
                != self.signature.interface_spread_ppm_upper
            ):
                raise ValueError("present pair telemetry differs from its signature")
            if (
                self.owner_component_ids[0] is None
                or self.owner_component_ids[0] != self.owner_component_ids[1]
            ):
                raise ValueError("present pair contact requires one shared owner")
        elif self.disposition is Disposition.CERTIFIED_ABSENT:
            if self.contact_kind not in {
                ContactKind.SEPARATE,
                ContactKind.INTERLEAVING,
            }:
                raise ValueError("certified absence requires a negative contact kind")
            if self.signature is not None or not self.certificate:
                raise ValueError("contact absence requires a certificate and no signature")
            if self.error_type is not None:
                raise ValueError("contact absence cannot carry an error type")
            if self.contact_kind is ContactKind.SEPARATE and (
                None in self.owner_component_ids
                or self.owner_component_ids[0] == self.owner_component_ids[1]
            ):
                raise ValueError("separate certificate requires two distinct owners")
            if self.contact_kind is ContactKind.SEPARATE and (
                self.reason_code != "distinct_source_foreground_components"
                or self.certificate != _SEPARATE_CERTIFICATE
                or self.normalized_gap_ppm_upper is not None
                or self.interface_spread_ppm_upper is not None
            ):
                raise ValueError("separate certificate fields are not canonical")
            if self.contact_kind is not ContactKind.SEPARATE and (
                self.owner_component_ids[0] is None
                or self.owner_component_ids[0] != self.owner_component_ids[1]
            ):
                raise ValueError("geometric non-point certificate requires one owner")
            if self.contact_kind is ContactKind.INTERLEAVING and (
                self.reason_code != "cyclic_owners_interleave"
                or self.certificate != _INTERLEAVING_CERTIFICATE
                or self.normalized_gap_ppm_upper is None
                or self.interface_spread_ppm_upper is None
                or self.normalized_gap_ppm_upper > _MAX_NORMALIZED_GAP_PPM
                or self.interface_spread_ppm_upper > _MAX_INTERFACE_SPREAD_PPM
            ):
                raise ValueError("interleaving certificate fields are not canonical")
        elif self.disposition is Disposition.INDETERMINATE:
            if self.contact_kind is not ContactKind.INDETERMINATE:
                raise ValueError("indeterminate contact requires indeterminate kind")
            if self.signature is not None or self.certificate is not None or self.error_type is not None:
                raise ValueError("indeterminate contact has incompatible fields")
        elif self.disposition is Disposition.ERROR:
            if self.contact_kind is not ContactKind.INDETERMINATE:
                raise ValueError("contact error uses indeterminate contact kind")
            if self.signature is not None or self.certificate is not None or not self.error_type:
                raise ValueError("contact error requires only an error type")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PAIR_CONTACT_SCHEMA,
            "loop_ids": list(self.loop_ids),
            "owner_component_ids": list(self.owner_component_ids),
            "disposition": self.disposition.value,
            "contact_kind": self.contact_kind.value,
            "normalized_gap_ppm_upper": self.normalized_gap_ppm_upper,
            "interface_spread_ppm_upper": self.interface_spread_ppm_upper,
            "signature": None if self.signature is None else self.signature.to_data(),
            "reason_code": self.reason_code,
            "certificate": self.certificate,
            "error_type": self.error_type,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PairContactObservation":
        _exact_fields(
            data,
            frozenset(
                {
                    "schema",
                    "loop_ids",
                    "owner_component_ids",
                    "disposition",
                    "contact_kind",
                    "normalized_gap_ppm_upper",
                    "interface_spread_ppm_upper",
                    "signature",
                    "reason_code",
                    "certificate",
                    "error_type",
                }
            ),
            "pair contact observation",
        )
        if data["schema"] != PAIR_CONTACT_SCHEMA:
            raise ValueError("unsupported pair contact observation")
        loops = data["loop_ids"]
        owners = data["owner_component_ids"]
        signature = data["signature"]
        if not isinstance(loops, list) or any(not isinstance(item, str) for item in loops):
            raise TypeError("pair contact loop_ids must be a string list")
        if not isinstance(owners, list) or any(
            item is not None and not isinstance(item, str) for item in owners
        ):
            raise TypeError("pair contact owner_component_ids must be strings or null")
        if signature is not None and not isinstance(signature, Mapping):
            raise TypeError("pair contact signature must be an object or null")
        return cls(
            loop_ids=tuple(loops),  # type: ignore[arg-type]
            owner_component_ids=tuple(owners),  # type: ignore[arg-type]
            disposition=Disposition(data["disposition"]),
            contact_kind=ContactKind(data["contact_kind"]),
            normalized_gap_ppm_upper=data["normalized_gap_ppm_upper"],
            interface_spread_ppm_upper=data["interface_spread_ppm_upper"],
            signature=(
                None if signature is None else PointContactSignature.from_data(signature)
            ),
            reason_code=data["reason_code"],
            certificate=data["certificate"],
            error_type=data["error_type"],
        )


@dataclass(frozen=True, slots=True)
class _DilatedRegion:
    loop_id: str
    area_pixels: int
    boundary: np.ndarray
    boundary_digest: str


class _UnresolvedContact(RuntimeError):
    pass


def _q16(value: float, extent: int) -> int:
    return max(0, min(65_535, int(round(value * 65_535 / extent))))


def _source_owner_component_ids(
    foreground_mask: np.ndarray, hole_masks: tuple[np.ndarray, ...]
) -> tuple[str | None, ...]:
    labels, count = ndimage.label(
        foreground_mask, structure=_base._FOREGROUND_STRUCTURE
    )
    raw: list[tuple[tuple[object, ...], int]] = []
    for label in range(1, count + 1):
        component = np.ascontiguousarray(labels == label, dtype=bool)
        x0, y0, x1, y1 = _base._bbox(component)
        raw.append(
            (
                (
                    x0,
                    y0,
                    x1,
                    y1,
                    int(np.count_nonzero(component)),
                    _base._mask_digest(component),
                ),
                label,
            )
        )
    raw.sort(key=lambda item: item[0])
    label_to_id = {
        label: f"component-{index:08d}"
        for index, (_key, label) in enumerate(raw)
    }
    owners: list[str | None] = []
    for hole in hole_masks:
        boundary = ndimage.binary_dilation(
            hole, structure=_base._BACKGROUND_STRUCTURE, iterations=1
        ) & foreground_mask
        owner_labels = set(int(item) for item in labels[boundary])
        owner_labels.discard(0)
        owners.append(
            label_to_id[next(iter(owner_labels))]
            if len(owner_labels) == 1
            else None
        )
    return tuple(owners)


def _region_mapping(
    foreground_mask: np.ndarray,
    hole_masks: tuple[np.ndarray, ...],
    loops: tuple[LoopGeometryWitness, ...],
) -> dict[str, _DilatedRegion]:
    dilated = ndimage.binary_dilation(
        foreground_mask, iterations=_DILATION_ITERATIONS
    )
    labels, count = ndimage.label(
        ~dilated, structure=_base._BACKGROUND_STRUCTURE
    )
    border = set(int(item) for item in labels[0, :])
    border.update(int(item) for item in labels[-1, :])
    border.update(int(item) for item in labels[:, 0])
    border.update(int(item) for item in labels[:, -1])
    border.discard(0)
    candidates: dict[str, list[_DilatedRegion]] = {}
    for label in range(1, count + 1):
        if label in border:
            continue
        region = np.ascontiguousarray(labels == label, dtype=bool)
        overlaps = [
            (int(np.count_nonzero(region & source)), loop.loop_id)
            for source, loop in zip(hole_masks, loops, strict=True)
        ]
        positive = tuple(item for item in overlaps if item[0] > 0)
        if len(positive) != 1:
            continue
        _overlap, loop_id = positive[0]
        cycles = boundary_cycles_for_mask(region)
        if len(cycles) != 1:
            continue
        boundary = cycles[0]
        candidates.setdefault(loop_id, []).append(_DilatedRegion(
            loop_id=loop_id,
            area_pixels=int(np.count_nonzero(region)),
            boundary=boundary,
            boundary_digest=canonical_digest(
                {
                    "schema": "gkm.bongard-dilated-loop-boundary.v1",
                    "loop_id": loop_id,
                    "mask_digest": _base._mask_digest(region),
                    "points": [
                        [int(point[0]), int(point[1])] for point in boundary
                    ],
                }
            ),
        ))
    return {
        loop_id: regions[0]
        for loop_id, regions in candidates.items()
        if len(regions) == 1
    }


def _fit_ray(
    region: _DilatedRegion,
    contact_index: int,
    *,
    step: int,
    width: int,
    height: int,
) -> IncidentRayWitness:
    points = region.boundary
    if len(points) < 2 * _MIN_RAY_POINTS:
        raise _UnresolvedContact("loop boundary is undersampled for two rays")
    count = max(
        _MIN_RAY_POINTS,
        int(round(len(points) * _RAY_WINDOW_NUMERATOR / _RAY_WINDOW_DENOMINATOR)),
    )
    count = min(count, max(_MIN_RAY_POINTS, len(points) // 3))
    indices = tuple(
        (contact_index + step * offset) % len(points) for offset in range(count)
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
        raise _UnresolvedContact("loop boundary ray is degenerate")
    normal = np.asarray((-axis[1], axis[0]))
    orthogonal = centered @ normal
    residual = math.sqrt(float(orthogonal @ orthogonal) / len(local)) / span
    residual_ppm = int(math.ceil(residual * 1_000_000.0))
    if not math.isfinite(residual) or residual_ppm > _MAX_RAY_RESIDUAL_PPM:
        raise _UnresolvedContact("loop boundary ray exceeds residual gate")
    extent = float(np.max((local - origin) @ axis))
    if not math.isfinite(extent) or extent <= np.finfo(float).tiny:
        raise _UnresolvedContact("loop boundary ray has no outward extent")
    endpoint = origin + extent * axis
    endpoint_name = "start" if step > 0 else "end"
    direction = int(
        round((math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 360.0) * 1_000.0)
    ) % 360_000
    uncertainty = int(
        math.ceil(math.degrees(math.atan(residual_ppm / 1_000_000.0)) * 1_000.0)
    )
    return IncidentRayWitness(
        ray_id=f"{region.loop_id}:{endpoint_name}:boundary-ray",
        owner_loop_id=region.loop_id,
        endpoint_name=endpoint_name,
        direction_millidegrees=direction,
        uncertainty_millidegrees=uncertainty,
        residual_ppm_upper=residual_ppm,
        endpoint_q16=Q16Point(_q16(float(endpoint[0]), width), _q16(float(endpoint[1]), height)),
        source_boundary_digest=region.boundary_digest,
    )


def _indeterminate(
    loop_ids: tuple[str, str],
    owner_component_ids: tuple[str | None, str | None],
    reason: str,
    *,
    gap: int | None = None,
    spread: int | None = None,
) -> PairContactObservation:
    return PairContactObservation(
        loop_ids=loop_ids,
        owner_component_ids=owner_component_ids,
        disposition=Disposition.INDETERMINATE,
        contact_kind=ContactKind.INDETERMINATE,
        normalized_gap_ppm_upper=gap,
        interface_spread_ppm_upper=spread,
        signature=None,
        reason_code=reason,
    )


def _absent(
    loop_ids: tuple[str, str],
    owner_component_ids: tuple[str | None, str | None],
    kind: ContactKind,
    reason: str,
    certificate: str,
    *,
    gap: int | None,
    spread: int | None,
) -> PairContactObservation:
    return PairContactObservation(
        loop_ids=loop_ids,
        owner_component_ids=owner_component_ids,
        disposition=Disposition.CERTIFIED_ABSENT,
        contact_kind=kind,
        normalized_gap_ppm_upper=gap,
        interface_spread_ppm_upper=spread,
        signature=None,
        reason_code=reason,
        certificate=certificate,
    )


def _pair_observation(
    first: _DilatedRegion,
    second: _DilatedRegion,
    *,
    owner_component_ids: tuple[str | None, str | None],
    width: int,
    height: int,
) -> PairContactObservation:
    loop_ids = tuple(sorted((first.loop_id, second.loop_id)))
    if not len(first.boundary) or not len(second.boundary):
        return _indeterminate(
            loop_ids, owner_component_ids, "boundary_distance_unresolved"
        )
    try:
        second_tree = cKDTree(second.boundary)
        nearest, _nearest_indices = second_tree.query(
            first.boundary, k=1, workers=1
        )
    except (ValueError, RuntimeError):
        return _indeterminate(
            loop_ids, owner_component_ids, "boundary_distance_unresolved"
        )
    nearest_values = np.asarray(nearest, dtype=float).reshape(-1)
    if not nearest_values.size or not np.isfinite(nearest_values).all():
        return _indeterminate(
            loop_ids, owner_component_ids, "boundary_distance_unresolved"
        )
    minimum = float(np.min(nearest_values))
    scale = math.sqrt(float(min(first.area_pixels, second.area_pixels)))
    if not math.isfinite(scale) or scale <= np.finfo(float).tiny:
        return _indeterminate(loop_ids, owner_component_ids, "degenerate_loop_scale")
    gap_ppm = int(math.ceil(minimum * 1_000_000.0 / scale))
    if gap_ppm > _MAX_NORMALIZED_GAP_PPM:
        return _indeterminate(
            loop_ids,
            owner_component_ids,
            "normalized_gap_outside_validated_support",
            gap=gap_ppm,
        )
    near_pairs: list[tuple[int, int, float]] = []
    try:
        neighbor_lists = second_tree.query_ball_point(
            first.boundary, r=minimum + 1.0, workers=1
        )
    except (ValueError, RuntimeError):
        return _indeterminate(
            loop_ids,
            owner_component_ids,
            "near_interface_unresolved",
            gap=gap_ppm,
        )
    for first_index, neighbors in enumerate(neighbor_lists):
        for second_index in sorted(int(item) for item in neighbors):
            distance = float(
                np.linalg.norm(
                    first.boundary[first_index] - second.boundary[second_index]
                )
            )
            if not math.isfinite(distance) or distance > minimum + 1.0:
                continue
            near_pairs.append((first_index, second_index, distance))
            if len(near_pairs) > _MAX_NEAR_INTERFACE_PAIRS:
                return _indeterminate(
                    loop_ids,
                    owner_component_ids,
                    "near_interface_pair_limit_exceeded",
                    gap=gap_ppm,
                )
    if not near_pairs:
        return _indeterminate(
            loop_ids,
            owner_component_ids,
            "near_interface_unresolved",
            gap=gap_ppm,
        )
    midpoints = np.asarray(
        [
            (first.boundary[first_index] + second.boundary[second_index]) / 2.0
            for first_index, second_index, _distance in near_pairs
        ]
    )
    spans = np.ptp(midpoints, axis=0) if len(midpoints) > 1 else np.zeros(2)
    spread = float(np.linalg.norm(spans))
    spread_ppm = int(math.ceil(spread * 1_000_000.0 / scale))
    if spread_ppm > _MAX_INTERFACE_SPREAD_PPM:
        return _indeterminate(
            loop_ids,
            owner_component_ids,
            "interface_spread_outside_validated_support",
            gap=gap_ppm,
            spread=spread_ppm,
        )
    interface_center = np.mean(midpoints, axis=0)
    true_minimum = min(item[2] for item in near_pairs)
    candidates = tuple(
        (first_index, second_index)
        for first_index, second_index, distance in near_pairs
        if math.isclose(distance, true_minimum, rel_tol=0.0, abs_tol=1e-12)
    )
    chosen_i, chosen_j = min(
        candidates,
        key=lambda pair: (
            float(
                np.linalg.norm(
                    (first.boundary[pair[0]] + second.boundary[pair[1]]) / 2.0
                    - interface_center
                )
            ),
            pair,
        ),
    )
    vertex = (first.boundary[chosen_i] + second.boundary[chosen_j]) / 2.0
    try:
        rays = tuple(
            _fit_ray(region, index, step=step, width=width, height=height)
            for region, index in ((first, chosen_i), (second, chosen_j))
            for step in (-1, 1)
        )
    except _UnresolvedContact as exc:
        return _indeterminate(
            loop_ids,
            owner_component_ids,
            f"ray_fit_unresolved:{exc}",
            gap=gap_ppm,
            spread=spread_ppm,
        )
    ordered = tuple(
        sorted(
            rays,
            key=lambda item: (
                item.direction_millidegrees,
                item.owner_loop_id,
                item.ray_id,
            ),
        )
    )
    if len({item.direction_millidegrees for item in ordered}) != 4:
        return _indeterminate(
            loop_ids,
            owner_component_ids,
            "ray_cyclic_order_unresolved",
            gap=gap_ppm,
            spread=spread_ppm,
        )
    for index, ray in enumerate(ordered):
        following = ordered[(index + 1) % 4]
        nominal_gap = (
            following.direction_millidegrees - ray.direction_millidegrees
        ) % 360_000
        uncertainty = (
            ray.uncertainty_millidegrees + following.uncertainty_millidegrees
        )
        if nominal_gap <= uncertainty:
            return _indeterminate(
                loop_ids,
                owner_component_ids,
                "ray_cyclic_order_uncertain",
                gap=gap_ppm,
                spread=spread_ppm,
            )
    owners = tuple(item.owner_loop_id for item in ordered)
    transitions = sum(owners[index] != owners[(index + 1) % 4] for index in range(4))
    if transitions != 2:
        return _absent(
            loop_ids,
            owner_component_ids,
            ContactKind.INTERLEAVING,
            "cyclic_owners_interleave",
            _INTERLEAVING_CERTIFICATE,
            gap=gap_ppm,
            spread=spread_ppm,
        )
    gaps: list[ExteriorGapWitness] = []
    for index, ray in enumerate(ordered):
        following = ordered[(index + 1) % 4]
        if ray.owner_loop_id == following.owner_loop_id:
            continue
        nominal = (
            following.direction_millidegrees - ray.direction_millidegrees
        ) % 360_000
        uncertainty = (
            ray.uncertainty_millidegrees + following.uncertainty_millidegrees
        )
        if nominal <= uncertainty or nominal + uncertainty >= 360_000:
            return _indeterminate(
                loop_ids,
                owner_component_ids,
                "exterior_gap_order_unresolved",
                gap=gap_ppm,
                spread=spread_ppm,
            )
        gaps.append(
            ExteriorGapWitness(
                ray_a_id=ray.ray_id,
                ray_b_id=following.ray_id,
                owner_a=ray.owner_loop_id,
                owner_b=following.owner_loop_id,
                nominal_millidegrees=nominal,
                interval_millidegrees=IntInterval(
                    nominal - uncertainty, nominal + uncertainty
                ),
            )
        )
    if len(gaps) != 2:
        return _indeterminate(
            loop_ids,
            owner_component_ids,
            "exterior_gap_count_unresolved",
            gap=gap_ppm,
            spread=spread_ppm,
        )
    gaps.sort(
        key=lambda item: (
            item.nominal_millidegrees,
            item.ray_a_id,
            item.ray_b_id,
        )
    )
    signature = PointContactSignature(
        contact_id=f"contact:{loop_ids[0]}:{loop_ids[1]}",
        loop_ids=loop_ids,
        contact_count=1,
        vertex_q16=Q16Point(_q16(float(vertex[0]), width), _q16(float(vertex[1]), height)),
        normalized_gap_ppm_upper=gap_ppm,
        interface_spread_ppm_upper=spread_ppm,
        rays=ordered,
        cyclic_owners=owners,
        exterior_gaps=(gaps[0], gaps[1]),
    )
    return PairContactObservation(
        loop_ids=loop_ids,
        owner_component_ids=owner_component_ids,
        disposition=Disposition.PRESENT,
        contact_kind=ContactKind.POINT,
        normalized_gap_ppm_upper=gap_ppm,
        interface_spread_ppm_upper=spread_ppm,
        signature=signature,
        reason_code="complete_four_ray_signature",
    )


def extract_pair_contact_observations(
    foreground_mask: np.ndarray,
    hole_masks: tuple[np.ndarray, ...],
    loops: tuple[LoopGeometryWitness, ...],
    *,
    width_pixels: int,
    height_pixels: int,
    source_owner_component_ids: tuple[str | None, ...],
) -> tuple[PairContactObservation, ...]:
    """Exhaustively observe every unordered loop pair in one scenario."""

    mask = np.asarray(foreground_mask)
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise TypeError("foreground_mask must be a two-dimensional Boolean array")
    if mask.shape != (height_pixels, width_pixels):
        raise ValueError("foreground mask dimensions differ from the panel")
    if not isinstance(hole_masks, tuple) or not isinstance(loops, tuple):
        raise TypeError("hole masks and loops must be tuples")
    if len(hole_masks) != len(loops):
        raise ValueError("hole masks and loops differ in length")
    if not isinstance(source_owner_component_ids, tuple) or len(
        source_owner_component_ids
    ) != len(loops):
        raise TypeError("source owner component IDs must align with loops")
    if any(
        item is not None
        and (not isinstance(item, str) or _COMPONENT_ID.fullmatch(item) is None)
        for item in source_owner_component_ids
    ):
        raise ValueError("source owner component ID is not canonical or null")
    for source, loop in zip(hole_masks, loops, strict=True):
        if _base._mask_digest(np.asarray(source, dtype=bool)) != loop.source_mask_digest:
            raise ValueError("point-contact source mask differs from loop witness")
    mapping = _region_mapping(mask, hole_masks, loops)
    owners = _source_owner_component_ids(mask, hole_masks)
    if owners != source_owner_component_ids:
        raise ValueError("point-contact owner replay differs from parent holes")
    observations: list[PairContactObservation] = []
    for first_index, first_loop in enumerate(loops):
        for second_index, second_loop in enumerate(
            loops[first_index + 1 :], start=first_index + 1
        ):
            loop_ids = tuple(sorted((first_loop.loop_id, second_loop.loop_id)))
            owner_component_ids = (owners[first_index], owners[second_index])
            if None in owner_component_ids:
                observations.append(
                    _indeterminate(
                        loop_ids,
                        owner_component_ids,
                        "source_owner_component_unresolved",
                    )
                )
                continue
            if owner_component_ids[0] != owner_component_ids[1]:
                observations.append(
                    _absent(
                        loop_ids,
                        owner_component_ids,
                        ContactKind.SEPARATE,
                        "distinct_source_foreground_components",
                        _SEPARATE_CERTIFICATE,
                        gap=None,
                        spread=None,
                    )
                )
                continue
            if any(
                item.substantiveness.disposition is not Disposition.PRESENT
                for item in (first_loop, second_loop)
            ):
                observations.append(
                    _indeterminate(
                        loop_ids,
                        owner_component_ids,
                        "pair_below_geometry_resolution_floor",
                    )
                )
                continue
            first = mapping.get(first_loop.loop_id)
            second = mapping.get(second_loop.loop_id)
            if first is None or second is None:
                observations.append(
                    _indeterminate(
                        loop_ids,
                        owner_component_ids,
                        "dilated_loop_mapping_unresolved",
                    )
                )
                continue
            observations.append(
                _pair_observation(
                    first,
                    second,
                    owner_component_ids=owner_component_ids,
                    width=width_pixels,
                    height=height_pixels,
                )
            )
    return tuple(sorted(observations, key=lambda item: item.loop_ids))


__all__ = [
    "ContactKind",
    "ExteriorGapWitness",
    "IncidentRayWitness",
    "PAIR_CONTACT_SCHEMA",
    "POINT_CONTACT_ALGORITHM_ID",
    "PairContactObservation",
    "PointContactSignature",
    "extract_pair_contact_observations",
    "point_contact_algorithm_digest",
    "point_contact_source_digest",
]
