"""Candidate-independent deterministic raster features for Bongard panels.

The raw extractor accepts only exact PNG bytes or a path whose exact bytes are
hashed.  It has no task, side, query-role, prose, proposal, or formula input.
Semantic decisions belong to the separately frozen support-prototype layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
import hashlib
import json
import math
import platform
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

from bongard.evidence import Disposition, Evidence, Provenance, Uncertainty
from bongard.support_prototypes import (
    FeatureDimension,
    FeatureInterval,
    FrozenFeatureSpace,
    FrozenPanelFeatures,
)


ALGORITHM_ID = "neutral-white-distance-raster-features/v1"
EXTRACTOR_VERSION = "1.0.0"
RECEIPT_SCHEMA = "bongard.neutral-raster-feature-receipt/v1"
PACKET_COMMITMENT_SCHEMA = "bongard.neutral-raster-packet-commitment/v1"
FEATURE_GROUP_IDS = (
    "prototype.topology",
    "prototype.global_geometry",
    "prototype.moments_symmetry",
    "prototype.boundary_angle",
)

_FOREGROUND_STRENGTH_THRESHOLDS = (32, 64, 96)
_MIN_FOREGROUND_PIXELS = 24
_MIN_BOUNDING_DIAGONAL_PIXELS = 6.0
_MAX_PANEL_PIXELS = 4096 * 4096
_MAX_RUN_COUNT = 65536
_RECEIPT_PROTOCOL = {
    "schema": RECEIPT_SCHEMA,
    "packet_commitment_schema": PACKET_COMMITMENT_SCHEMA,
    "packet_receipt_binding": (
        "sha256(receipt canonical JSON) is inserted into the otherwise "
        "committed packet as extractor_receipt_digest"
    ),
    "four_dispositions": [item.value for item in Disposition],
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_exact_fields(
    data: Mapping[str, Any], fields: frozenset[str], label: str
) -> None:
    if not isinstance(data, Mapping) or set(data) != fields:
        raise ValueError(f"{label} fields differ from schema")


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty exact string")
    return value


def _require_digest(name: str, value: object) -> str:
    result = _require_text(name, value)
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise ValueError(f"{name} must be a lowercase sha256")
    return result


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _environment() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                ("numpy", np.__version__),
                ("pillow", PILLOW_VERSION),
                ("python_implementation", platform.python_implementation()),
                ("python_version", platform.python_version()),
            )
        )
    )


def _preprocessing_data(group_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "schema": "bongard.neutral-raster-preprocessing/v1",
        "decoder": {
            "format": "PNG",
            "frames": 1,
            "rgba_background": 255,
            "white_distance": "max(255-r,255-g,255-b)",
        },
        "foreground_strength_thresholds": list(
            _FOREGROUND_STRENGTH_THRESHOLDS
        ),
        "topology_connectivity": {
            "foreground": 8,
            "background_holes": 4,
        },
        "guards": {
            "minimum_foreground_pixels": _MIN_FOREGROUND_PIXELS,
            "minimum_bounding_diagonal_pixels": _MIN_BOUNDING_DIAGONAL_PIXELS,
            "maximum_panel_pixels": _MAX_PANEL_PIXELS,
            "maximum_run_count": _MAX_RUN_COUNT,
            "reject_border_clipping": True,
        },
        "feature_group_ids": list(group_ids),
    }


def _receipt_protocol_digest() -> str:
    return _digest(_RECEIPT_PROTOCOL)


_DIMENSIONS = {
    "bbox_fill_fraction": FeatureDimension(
        "bbox_fill_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "bbox_height_fraction": FeatureDimension(
        "bbox_height_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "bbox_isotropy_fraction": FeatureDimension(
        "bbox_isotropy_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "bbox_width_fraction": FeatureDimension(
        "bbox_width_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "boundary_fraction": FeatureDimension(
        "boundary_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "centroid_x_fraction": FeatureDimension(
        "centroid_x_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "centroid_y_fraction": FeatureDimension(
        "centroid_y_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "component_count": FeatureDimension(
        "component_count", "count", 0.0, float(_MAX_RUN_COUNT), 4.0
    ),
    "diagonal_gradient_fraction": FeatureDimension(
        "diagonal_gradient_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "euler_characteristic": FeatureDimension(
        "euler_characteristic",
        "count",
        -float(_MAX_RUN_COUNT),
        float(_MAX_RUN_COUNT),
        4.0,
    ),
    "foreground_area_fraction": FeatureDimension(
        "foreground_area_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "half_turn_agreement_fraction": FeatureDimension(
        "half_turn_agreement_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "hole_count": FeatureDimension(
        "hole_count", "count", 0.0, float(_MAX_RUN_COUNT), 4.0
    ),
    "largest_component_fraction": FeatureDimension(
        "largest_component_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "mirror_axis_aligned_agreement_fraction": FeatureDimension(
        "mirror_axis_aligned_agreement_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "principal_axis_anisotropy_fraction": FeatureDimension(
        "principal_axis_anisotropy_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "principal_axis_obliqueness_fraction": FeatureDimension(
        "principal_axis_obliqueness_fraction", "fraction", 0.0, 1.0, 1.0
    ),
    "two_by_two_corner_fraction": FeatureDimension(
        "two_by_two_corner_fraction", "fraction", 0.0, 1.0, 1.0
    ),
}

_GROUPS = (
    (
        "prototype.topology",
        "Foreground components, enclosed background holes, Euler count, and component dominance.",
        (
            "component_count",
            "euler_characteristic",
            "hole_count",
            "largest_component_fraction",
        ),
    ),
    (
        "prototype.global_geometry",
        "Ink area and translation-normalized bounding-box extent, fill, and isotropy.",
        (
            "bbox_fill_fraction",
            "bbox_height_fraction",
            "bbox_isotropy_fraction",
            "bbox_width_fraction",
            "foreground_area_fraction",
        ),
    ),
    (
        "prototype.moments_symmetry",
        "Normalized centroid and second-moment shape with fixed half-turn and axis-aligned mirror agreement.",
        (
            "centroid_x_fraction",
            "centroid_y_fraction",
            "half_turn_agreement_fraction",
            "mirror_axis_aligned_agreement_fraction",
            "principal_axis_anisotropy_fraction",
            "principal_axis_obliqueness_fraction",
        ),
    ),
    (
        "prototype.boundary_angle",
        "Raster boundary density plus fixed binary corner and oblique-gradient proxies.",
        (
            "boundary_fraction",
            "diagonal_gradient_fraction",
            "two_by_two_corner_fraction",
        ),
    ),
)


def _normalize_group_ids(group_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(group_ids, tuple) or not group_ids:
        raise TypeError("feature group IDs must be a non-empty tuple")
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("feature group IDs must be unique")
    if any(group_id not in FEATURE_GROUP_IDS for group_id in group_ids):
        raise ValueError("unknown neutral feature group ID")
    return group_ids


def _dimension_names(group_ids: tuple[str, ...]) -> tuple[str, ...]:
    wanted = set(group_ids)
    return tuple(
        sorted(
            name
            for group_id, _, names in _GROUPS
            if group_id in wanted
            for name in names
        )
    )


def _space_for_groups_with_source(
    group_ids: tuple[str, ...], source_digest: str
) -> FrozenFeatureSpace:
    group_ids = _normalize_group_ids(group_ids)
    preprocessing_digest = _digest(_preprocessing_data(group_ids))
    return FrozenFeatureSpace(
        extractor_id="bongard.neutral_raster_features",
        extractor_version=EXTRACTOR_VERSION,
        extractor_artifact_digest=source_digest,
        preprocessing_digest=preprocessing_digest,
        receipt_protocol_digest=_receipt_protocol_digest(),
        dimensions=tuple(_DIMENSIONS[name] for name in _dimension_names(group_ids)),
    )


def _operation_digest_from(
    feature_space: FrozenFeatureSpace,
    group_ids: tuple[str, ...],
    environment_digest: str,
) -> str:
    return _digest(
        {
            "algorithm_id": ALGORITHM_ID,
            "feature_group_ids": list(group_ids),
            "feature_space": feature_space.to_data(),
            "environment_digest": environment_digest,
        }
    )


def _provenance_data(provenance: Provenance) -> dict[str, object]:
    return {
        "producer": provenance.producer,
        "version": provenance.version,
        "method": provenance.method,
        "input_digests": list(provenance.input_digests),
        "artifact_digest": provenance.artifact_digest,
        "run_id": provenance.run_id,
        "details": [list(item) for item in provenance.details],
    }


def _provenance_from_data(data: Mapping[str, Any]) -> Provenance:
    _require_exact_fields(
        data,
        frozenset(
            {
                "producer",
                "version",
                "method",
                "input_digests",
                "artifact_digest",
                "run_id",
                "details",
            }
        ),
        "neutral receipt provenance",
    )
    input_digests = data["input_digests"]
    details = data["details"]
    if not isinstance(input_digests, list) or any(
        not isinstance(item, str) for item in input_digests
    ):
        raise TypeError("provenance input digests must be a JSON string list")
    if not isinstance(details, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or any(not isinstance(value, str) for value in item)
        for item in details
    ):
        raise TypeError("provenance details must be JSON string pairs")
    return Provenance(
        producer=data["producer"],
        version=data["version"],
        method=data["method"],
        input_digests=tuple(input_digests),
        artifact_digest=data["artifact_digest"],
        run_id=data["run_id"],
        details=tuple((item[0], item[1]) for item in details),
    )


def _uncertainty_data(value: Uncertainty | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "lower": value.lower,
        "upper": value.upper,
        "confidence_level": value.confidence_level,
        "causes": list(value.causes),
    }


def _uncertainty_from_data(value: object) -> Uncertainty | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("receipt uncertainty must be an object or null")
    _require_exact_fields(
        value,
        frozenset({"lower", "upper", "confidence_level", "causes"}),
        "receipt uncertainty",
    )
    causes = value["causes"]
    if not isinstance(causes, list) or any(not isinstance(item, str) for item in causes):
        raise TypeError("uncertainty causes must be a JSON string list")
    return Uncertainty(
        value["lower"],
        value["upper"],
        value["confidence_level"],
        tuple(causes),
    )


@dataclass(frozen=True)
class NeutralFeatureGroup:
    """One closed, verifier-owned proposer catalog entry."""

    group_id: str
    description: str
    dimension_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.group_id not in FEATURE_GROUP_IDS:
            raise ValueError("unknown neutral feature group")
        _require_text("feature group description", self.description)
        if (
            not isinstance(self.dimension_names, tuple)
            or not self.dimension_names
            or self.dimension_names != tuple(sorted(self.dimension_names))
            or len(self.dimension_names) != len(set(self.dimension_names))
            or any(name not in _DIMENSIONS for name in self.dimension_names)
        ):
            raise ValueError("feature group dimensions must be known, unique, and sorted")

    def to_data(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "description": self.description,
            "dimensions": [
                _DIMENSIONS[name].to_data() for name in self.dimension_names
            ],
        }


@dataclass(frozen=True)
class NeutralInputIdentity:
    """Exact byte identity, or an explicit identity for an unreadable input."""

    kind: str
    sha256: str
    byte_count: int | None
    media_type: str | None

    def __post_init__(self) -> None:
        if self.kind not in {
            "png_bytes",
            "png_path_bytes",
            "encoded_bytes",
            "unreadable_path",
            "unsupported_input",
        }:
            raise ValueError("unknown neutral input identity kind")
        _require_digest("input sha256", self.sha256)
        if self.byte_count is not None and (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count < 0
        ):
            raise ValueError("input byte_count must be a nonnegative integer or null")
        if self.media_type not in (None, "image/png", "application/octet-stream"):
            raise ValueError("unsupported input media type")
        has_bytes = self.kind in {
            "png_bytes",
            "png_path_bytes",
            "encoded_bytes",
        }
        if has_bytes != (self.byte_count is not None):
            raise ValueError("byte-bearing input identity has inconsistent byte count")
        if has_bytes != (self.media_type is not None):
            raise ValueError("byte-bearing input identity has inconsistent media type")

    def to_data(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "NeutralInputIdentity":
        _require_exact_fields(
            data,
            frozenset({"kind", "sha256", "byte_count", "media_type"}),
            "neutral input identity",
        )
        return cls(
            kind=data["kind"],
            sha256=data["sha256"],
            byte_count=data["byte_count"],
            media_type=data["media_type"],
        )


@dataclass(frozen=True)
class NeutralPacketCommitment:
    """Unsigned packet preimage; omits the receipt digest to avoid a cycle."""

    panel_digest: str
    feature_space_digest: str
    values: tuple[FeatureInterval, ...]

    def __post_init__(self) -> None:
        _require_digest("packet panel_digest", self.panel_digest)
        _require_digest("packet feature_space_digest", self.feature_space_digest)
        if (
            not isinstance(self.values, tuple)
            or not self.values
            or any(not isinstance(item, FeatureInterval) for item in self.values)
        ):
            raise TypeError("packet commitment values must be a typed tuple")
        names = tuple(item.name for item in self.values)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("packet commitment values must be unique and sorted")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PACKET_COMMITMENT_SCHEMA,
            "panel_digest": self.panel_digest,
            "feature_space_digest": self.feature_space_digest,
            "values": [item.to_data() for item in self.values],
        }

    def digest(self) -> str:
        return _digest(self.to_data())

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "NeutralPacketCommitment":
        _require_exact_fields(
            data,
            frozenset(
                {"schema", "panel_digest", "feature_space_digest", "values"}
            ),
            "neutral packet commitment",
        )
        if data["schema"] != PACKET_COMMITMENT_SCHEMA:
            raise ValueError("unsupported neutral packet commitment schema")
        values = data["values"]
        if not isinstance(values, list):
            raise TypeError("packet commitment values must be a JSON list")
        return cls(
            panel_digest=data["panel_digest"],
            feature_space_digest=data["feature_space_digest"],
            values=tuple(FeatureInterval.from_data(item) for item in values),
        )


@dataclass(frozen=True)
class NeutralFeatureReceipt:
    """Canonical archived receipt preimage for every extractor disposition."""

    input_identity: NeutralInputIdentity
    feature_group_ids: tuple[str, ...]
    catalog_digest: str
    feature_space: FrozenFeatureSpace
    feature_space_digest: str
    source_digest: str
    environment: tuple[tuple[str, str], ...]
    environment_digest: str
    preprocessing_digest: str
    receipt_protocol_digest: str
    operation_digest: str
    disposition: Disposition
    packet_commitment: NeutralPacketCommitment | None
    packet_commitment_digest: str | None
    uncertainty: Uncertainty | None
    certificate: str | None
    reason: str | None
    error_type: str | None
    provenance: Provenance
    provenance_digest: str
    parent_receipt_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_identity, NeutralInputIdentity):
            raise TypeError("receipt input_identity is malformed")
        _normalize_group_ids(self.feature_group_ids)
        for name, value in (
            ("catalog_digest", self.catalog_digest),
            ("feature_space_digest", self.feature_space_digest),
            ("source_digest", self.source_digest),
            ("environment_digest", self.environment_digest),
            ("preprocessing_digest", self.preprocessing_digest),
            ("receipt_protocol_digest", self.receipt_protocol_digest),
            ("operation_digest", self.operation_digest),
            ("provenance_digest", self.provenance_digest),
        ):
            _require_digest(name, value)
        if self.parent_receipt_digest is not None:
            _require_digest("parent_receipt_digest", self.parent_receipt_digest)
        if not isinstance(self.feature_space, FrozenFeatureSpace):
            raise TypeError("receipt feature_space is malformed")
        if (
            not isinstance(self.environment, tuple)
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or any(not isinstance(value, str) or not value for value in item)
                for item in self.environment
            )
            or self.environment != tuple(sorted(self.environment))
            or len({key for key, _ in self.environment}) != len(self.environment)
        ):
            raise ValueError("receipt environment must be unique sorted string pairs")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("receipt disposition is malformed")
        if self.packet_commitment_digest is not None:
            _require_digest(
                "packet_commitment_digest", self.packet_commitment_digest
            )
        if not isinstance(self.provenance, Provenance):
            raise TypeError("receipt provenance is malformed")
        # Reuse Evidence's exhaustive state validation without manufacturing a
        # packet whose receipt digest has not yet been computed.
        if self.disposition is Disposition.PRESENT:
            if self.packet_commitment is None:
                raise ValueError("present receipt requires a packet commitment")
            if any(
                item is not None
                for item in (self.certificate, self.reason, self.error_type)
            ):
                raise ValueError("present receipt carries non-present fields")
        elif self.disposition is Disposition.CERTIFIED_ABSENT:
            if (
                self.packet_commitment is not None
                or not isinstance(self.certificate, str)
                or not self.certificate.strip()
                or self.reason is not None
                or self.error_type is not None
            ):
                raise ValueError("malformed certified-absent receipt")
        elif self.disposition is Disposition.INDETERMINATE:
            if (
                self.packet_commitment is not None
                or not isinstance(self.reason, str)
                or not self.reason.strip()
                or self.certificate is not None
                or self.error_type is not None
            ):
                raise ValueError("malformed indeterminate receipt")
        elif self.disposition is Disposition.ERROR:
            if (
                self.packet_commitment is not None
                or not isinstance(self.reason, str)
                or not self.reason.strip()
                or not isinstance(self.error_type, str)
                or not self.error_type.strip()
                or self.certificate is not None
            ):
                raise ValueError("malformed error receipt")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": RECEIPT_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "input_identity": self.input_identity.to_data(),
            "feature_group_ids": list(self.feature_group_ids),
            "catalog_digest": self.catalog_digest,
            "feature_space": self.feature_space.to_data(),
            "feature_space_digest": self.feature_space_digest,
            "source_digest": self.source_digest,
            "environment": [list(item) for item in self.environment],
            "environment_digest": self.environment_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "receipt_protocol_digest": self.receipt_protocol_digest,
            "operation_digest": self.operation_digest,
            "disposition": self.disposition.value,
            "packet_commitment": (
                self.packet_commitment.to_data()
                if self.packet_commitment is not None
                else None
            ),
            "packet_commitment_digest": self.packet_commitment_digest,
            "uncertainty": _uncertainty_data(self.uncertainty),
            "certificate": self.certificate,
            "reason": self.reason,
            "error_type": self.error_type,
            "provenance": _provenance_data(self.provenance),
            "provenance_digest": self.provenance_digest,
            "parent_receipt_digest": self.parent_receipt_digest,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "NeutralFeatureReceipt":
        fields = frozenset(
            {
                "schema",
                "algorithm_id",
                "input_identity",
                "feature_group_ids",
                "catalog_digest",
                "feature_space",
                "feature_space_digest",
                "source_digest",
                "environment",
                "environment_digest",
                "preprocessing_digest",
                "receipt_protocol_digest",
                "operation_digest",
                "disposition",
                "packet_commitment",
                "packet_commitment_digest",
                "uncertainty",
                "certificate",
                "reason",
                "error_type",
                "provenance",
                "provenance_digest",
                "parent_receipt_digest",
            }
        )
        _require_exact_fields(data, fields, "neutral feature receipt")
        if data["schema"] != RECEIPT_SCHEMA or data["algorithm_id"] != ALGORITHM_ID:
            raise ValueError("unsupported neutral feature receipt")
        group_ids = data["feature_group_ids"]
        environment = data["environment"]
        if not isinstance(group_ids, list) or any(
            not isinstance(item, str) for item in group_ids
        ):
            raise TypeError("receipt feature_group_ids must be a JSON string list")
        if not isinstance(environment, list) or any(
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(value, str) for value in item)
            for item in environment
        ):
            raise TypeError("receipt environment must be JSON string pairs")
        packet_data = data["packet_commitment"]
        if packet_data is not None and not isinstance(packet_data, Mapping):
            raise TypeError("packet commitment must be an object or null")
        provenance_data = data["provenance"]
        if not isinstance(provenance_data, Mapping):
            raise TypeError("receipt provenance must be an object")
        space_data = data["feature_space"]
        input_data = data["input_identity"]
        if not isinstance(space_data, Mapping) or not isinstance(input_data, Mapping):
            raise TypeError("receipt identity and feature space must be objects")
        result = cls(
            input_identity=NeutralInputIdentity.from_data(input_data),
            feature_group_ids=tuple(group_ids),
            catalog_digest=data["catalog_digest"],
            feature_space=FrozenFeatureSpace.from_data(space_data),
            feature_space_digest=data["feature_space_digest"],
            source_digest=data["source_digest"],
            environment=tuple((item[0], item[1]) for item in environment),
            environment_digest=data["environment_digest"],
            preprocessing_digest=data["preprocessing_digest"],
            receipt_protocol_digest=data["receipt_protocol_digest"],
            operation_digest=data["operation_digest"],
            disposition=Disposition(data["disposition"]),
            packet_commitment=(
                NeutralPacketCommitment.from_data(packet_data)
                if packet_data is not None
                else None
            ),
            packet_commitment_digest=data["packet_commitment_digest"],
            uncertainty=_uncertainty_from_data(data["uncertainty"]),
            certificate=data["certificate"],
            reason=data["reason"],
            error_type=data["error_type"],
            provenance=_provenance_from_data(provenance_data),
            provenance_digest=data["provenance_digest"],
            parent_receipt_digest=data["parent_receipt_digest"],
        )
        result.verify()
        return result

    def digest(self) -> str:
        return _digest(self.to_data())

    def verify(self) -> FrozenPanelFeatures | None:
        if self.catalog_digest != feature_group_catalog_digest():
            raise ValueError("neutral feature catalog digest drift")
        expected_space = _space_for_groups_with_source(
            self.feature_group_ids, self.source_digest
        )
        if self.feature_space != expected_space:
            raise ValueError("receipt feature space differs from its frozen groups")
        if self.feature_space_digest != self.feature_space.digest():
            raise ValueError("receipt feature-space digest drift")
        if self.source_digest != self.feature_space.extractor_artifact_digest:
            raise ValueError("receipt source digest drift")
        if self.preprocessing_digest != self.feature_space.preprocessing_digest:
            raise ValueError("receipt preprocessing digest drift")
        if self.receipt_protocol_digest != self.feature_space.receipt_protocol_digest:
            raise ValueError("receipt protocol digest drift")
        if self.environment_digest != _digest(dict(self.environment)):
            raise ValueError("receipt environment digest drift")
        expected_operation = _operation_digest_from(
            self.feature_space, self.feature_group_ids, self.environment_digest
        )
        if self.operation_digest != expected_operation:
            raise ValueError("receipt operation digest drift")
        expected_inputs = (self.input_identity.sha256,) + (
            (self.parent_receipt_digest,)
            if self.parent_receipt_digest is not None
            else ()
        )
        if self.provenance.input_digests != expected_inputs:
            raise ValueError("receipt provenance does not bind its inputs")
        if self.provenance.artifact_digest != self.feature_space_digest:
            raise ValueError("receipt provenance does not bind its feature space")
        if self.provenance_digest != self.provenance.digest():
            raise ValueError("receipt provenance digest drift")
        if self.disposition is not Disposition.ERROR and (
            self.input_identity.kind != "png_bytes"
            or self.input_identity.media_type != "image/png"
            or self.input_identity.byte_count is None
        ):
            raise ValueError("measured receipt does not bind validated PNG bytes")
        if self.disposition is not Disposition.PRESENT:
            if self.packet_commitment_digest is not None:
                raise ValueError("non-present receipt carries a packet digest")
            return None
        assert self.packet_commitment is not None
        if self.packet_commitment_digest != self.packet_commitment.digest():
            raise ValueError("receipt packet-commitment digest drift")
        if self.packet_commitment.panel_digest != self.input_identity.sha256:
            raise ValueError("packet commitment does not bind panel bytes")
        if (
            self.packet_commitment.feature_space_digest
            != self.feature_space_digest
        ):
            raise ValueError("packet commitment does not bind feature space")
        packet = FrozenPanelFeatures(
            panel_digest=self.packet_commitment.panel_digest,
            feature_space_digest=self.packet_commitment.feature_space_digest,
            extractor_receipt_digest=self.digest(),
            values=self.packet_commitment.values,
        )
        packet.validate(self.feature_space)
        return packet


class NeutralFeatureInputError(ValueError):
    """Input could not enter the exact PNG measurement boundary."""

    def __init__(
        self, message: str, *, identity: NeutralInputIdentity | None = None
    ) -> None:
        super().__init__(message)
        self.identity = identity


class _MeasurementGuard(ValueError):
    pass


def _fallback_identity(panel: object, kind: str) -> NeutralInputIdentity:
    identity_data = {
        "schema": "bongard.neutral-unavailable-input/v1",
        "kind": kind,
        "python_type": f"{type(panel).__module__}.{type(panel).__qualname__}",
        "path": str(panel) if isinstance(panel, (str, Path)) else None,
    }
    return NeutralInputIdentity(kind, _digest(identity_data), None, None)


def _read_exact_bytes(
    panel: bytes | str | Path,
) -> tuple[bytes, NeutralInputIdentity]:
    if isinstance(panel, bytes):
        raw = panel
        kind = "png_bytes"
    elif isinstance(panel, (str, Path)):
        try:
            raw = Path(panel).read_bytes()
        except OSError as exc:
            identity = _fallback_identity(panel, "unreadable_path")
            raise NeutralFeatureInputError(
                f"panel bytes could not be read: {type(exc).__name__}: {exc}",
                identity=identity,
            ) from exc
        # Transport is erased once bytes are available: the same exact PNG has
        # one content identity whether supplied directly or by filesystem path.
        kind = "png_bytes"
    else:
        identity = _fallback_identity(panel, "unsupported_input")
        raise NeutralFeatureInputError(
            "panel must be exact PNG bytes or a PNG filesystem path",
            identity=identity,
        )
    identity = NeutralInputIdentity(
        kind=kind,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
        media_type="application/octet-stream",
    )
    return raw, identity


def _decode_white_distance(raw: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(raw)) as encoded:
            if encoded.format != "PNG":
                raise NeutralFeatureInputError("encoded panel must be a PNG")
            if getattr(encoded, "n_frames", 1) != 1:
                raise NeutralFeatureInputError("encoded PNG must have exactly one frame")
            width, height = encoded.size
            if width < 2 or height < 2:
                raise NeutralFeatureInputError(
                    "PNG dimensions must both be at least two pixels"
                )
            if width * height > _MAX_PANEL_PIXELS:
                raise NeutralFeatureInputError(
                    "PNG exceeds the fixed maximum pixel count"
                )
            rgba = np.asarray(encoded.convert("RGBA"), dtype=np.uint8)
    except NeutralFeatureInputError:
        raise
    except Exception as exc:  # noqa: BLE001 - decode disposition boundary.
        raise NeutralFeatureInputError(
            f"PNG decoding failed: {type(exc).__name__}: {exc}"
        ) from exc
    values = rgba.astype(np.uint32, copy=False)
    alpha = values[..., 3:4]
    rgb = (values[..., :3] * alpha + 255 * (255 - alpha) + 127) // 255
    strength = 255 - np.min(rgb, axis=2)
    return np.ascontiguousarray(strength.astype(np.uint8))


def _component_roots(
    mask: np.ndarray, *, diagonal: bool
) -> tuple[tuple[int, bool], ...]:
    """Run-length union-find components as ``(area, touches_border)``."""

    height, width = mask.shape
    parents: list[int] = []
    sizes: list[int] = []
    touches: list[bool] = []

    def find(index: int) -> int:
        root = index
        while parents[root] != root:
            root = parents[root]
        while parents[index] != index:
            parent = parents[index]
            parents[index] = root
            index = parent
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root] or (
            sizes[left_root] == sizes[right_root] and left_root > right_root
        ):
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        sizes[left_root] += sizes[right_root]
        touches[left_root] = touches[left_root] or touches[right_root]

    previous: list[tuple[int, int, int]] = []
    run_count = 0
    expand = 1 if diagonal else 0
    for y in range(height):
        row = mask[y]
        padded = np.empty(width + 2, dtype=np.int8)
        padded[0] = 0
        padded[-1] = 0
        padded[1:-1] = row
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        current: list[tuple[int, int, int]] = []
        for start, stop in zip(changes[0::2], changes[1::2], strict=True):
            run_count += 1
            if run_count > _MAX_RUN_COUNT:
                raise _MeasurementGuard("run_count_limit")
            label = len(parents)
            parents.append(label)
            sizes.append(int(stop - start))
            touches.append(
                y == 0 or y == height - 1 or int(start) == 0 or int(stop) == width
            )
            current.append((int(start), int(stop - 1), label))
        previous_index = 0
        for start, stop, label in current:
            while (
                previous_index < len(previous)
                and previous[previous_index][1] < start - expand
            ):
                previous_index += 1
            candidate = previous_index
            while candidate < len(previous) and previous[candidate][0] <= stop + expand:
                other_start, other_stop, other_label = previous[candidate]
                if other_stop >= start - expand and other_start <= stop + expand:
                    union(label, other_label)
                candidate += 1
        previous = current
    roots: dict[int, tuple[int, bool]] = {}
    for label in range(len(parents)):
        root = find(label)
        if root not in roots:
            roots[root] = (sizes[root], touches[root])
    return tuple(roots[index] for index in sorted(roots))


def _agreement(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    if union == 0:
        return 1.0
    return float(1.0 - np.count_nonzero(left ^ right) / union)


def _mask_metrics(mask: np.ndarray) -> dict[str, float]:
    ys, xs = np.nonzero(mask)
    foreground = int(xs.size)
    if foreground == 0:
        raise _MeasurementGuard("zero_foreground")
    if (
        bool(mask[0].any())
        or bool(mask[-1].any())
        or bool(mask[:, 0].any())
        or bool(mask[:, -1].any())
    ):
        raise _MeasurementGuard("border_clipped")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    diagonal = math.hypot(width, height)
    if foreground < _MIN_FOREGROUND_PIXELS:
        raise _MeasurementGuard("insufficient_foreground")
    if diagonal < _MIN_BOUNDING_DIAGONAL_PIXELS:
        raise _MeasurementGuard("insufficient_extent")

    foreground_roots = _component_roots(mask, diagonal=True)
    background_roots = _component_roots(~mask, diagonal=False)
    component_sizes = tuple(area for area, _ in foreground_roots)
    component_count = len(component_sizes)
    hole_count = sum(1 for _, touches_border in background_roots if not touches_border)

    crop = mask[y0 : y1 + 1, x0 : x1 + 1]
    horizontal = _agreement(crop, np.fliplr(crop))
    vertical = _agreement(crop, np.flipud(crop))
    half_turn = _agreement(crop, np.flip(crop, axis=(0, 1)))

    normalized_x = (xs.astype(np.float64) - x0) / max(1, width - 1)
    normalized_y = (ys.astype(np.float64) - y0) / max(1, height - 1)
    centroid_x = float(np.mean(normalized_x)) if width > 1 else 0.5
    centroid_y = float(np.mean(normalized_y)) if height > 1 else 0.5
    centered_x = normalized_x - centroid_x
    centered_y = normalized_y - centroid_y
    var_x = float(np.mean(centered_x * centered_x))
    var_y = float(np.mean(centered_y * centered_y))
    covariance = float(np.mean(centered_x * centered_y))
    discriminant = math.hypot(var_x - var_y, 2.0 * covariance)
    trace = var_x + var_y
    anisotropy = discriminant / trace if trace > 0.0 else 0.0
    obliqueness = abs(2.0 * covariance) / discriminant if discriminant > 0.0 else 0.0

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = mask.copy()
    interior &= padded[:-2, 1:-1]
    interior &= padded[2:, 1:-1]
    interior &= padded[1:-1, :-2]
    interior &= padded[1:-1, 2:]
    boundary_fraction = float(np.count_nonzero(mask & ~interior) / foreground)

    binary = padded.astype(np.int16, copy=False)
    gradient_x = (
        binary[:-2, 2:]
        + 2 * binary[1:-1, 2:]
        + binary[2:, 2:]
        - binary[:-2, :-2]
        - 2 * binary[1:-1, :-2]
        - binary[2:, :-2]
    )
    gradient_y = (
        binary[2:, :-2]
        + 2 * binary[2:, 1:-1]
        + binary[2:, 2:]
        - binary[:-2, :-2]
        - 2 * binary[:-2, 1:-1]
        - binary[:-2, 2:]
    )
    energy = gradient_x * gradient_x + gradient_y * gradient_y
    total_energy = int(np.sum(energy, dtype=np.int64))
    diagonal_gradient = (
        float(
            2
            * np.sum(
                np.abs(gradient_x.astype(np.int32) * gradient_y),
                dtype=np.int64,
            )
            / total_energy
        )
        if total_energy
        else 0.0
    )
    block_sum = (
        mask[:-1, :-1].astype(np.uint8)
        + mask[1:, :-1]
        + mask[:-1, 1:]
        + mask[1:, 1:]
    )
    transitions = (block_sum > 0) & (block_sum < 4)
    transition_count = int(np.count_nonzero(transitions))
    corners = (block_sum == 1) | (block_sum == 3)
    corner_fraction = (
        float(np.count_nonzero(corners) / transition_count)
        if transition_count
        else 0.0
    )

    def fraction(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    return {
        "bbox_fill_fraction": fraction(foreground / (width * height)),
        "bbox_height_fraction": fraction(height / mask.shape[0]),
        "bbox_isotropy_fraction": fraction(min(width, height) / max(width, height)),
        "bbox_width_fraction": fraction(width / mask.shape[1]),
        "boundary_fraction": fraction(boundary_fraction),
        "centroid_x_fraction": fraction(centroid_x),
        "centroid_y_fraction": fraction(centroid_y),
        "component_count": float(component_count),
        "diagonal_gradient_fraction": fraction(diagonal_gradient),
        "euler_characteristic": float(component_count - hole_count),
        "foreground_area_fraction": fraction(foreground / mask.size),
        "half_turn_agreement_fraction": fraction(half_turn),
        "hole_count": float(hole_count),
        "largest_component_fraction": fraction(max(component_sizes) / foreground),
        "mirror_axis_aligned_agreement_fraction": fraction(max(horizontal, vertical)),
        "principal_axis_anisotropy_fraction": fraction(anisotropy),
        "principal_axis_obliqueness_fraction": fraction(obliqueness),
        "two_by_two_corner_fraction": fraction(corner_fraction),
    }


@dataclass(frozen=True)
class NeutralFeatureExtraction:
    """Four-disposition evidence paired with its complete receipt preimage."""

    evidence: Evidence[FrozenPanelFeatures]
    receipt: NeutralFeatureReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, Evidence) or not isinstance(
            self.receipt, NeutralFeatureReceipt
        ):
            raise TypeError("neutral extraction requires typed evidence and receipt")
        verify_neutral_feature_extraction(self)

    def verify(self) -> FrozenPanelFeatures | None:
        return verify_neutral_feature_extraction(self)

    def __bool__(self) -> bool:
        raise TypeError("neutral feature extraction has four dispositions")


def _make_receipt(
    *,
    identity: NeutralInputIdentity,
    group_ids: tuple[str, ...],
    disposition: Disposition,
    packet_commitment: NeutralPacketCommitment | None = None,
    uncertainty: Uncertainty | None = None,
    certificate: str | None = None,
    reason: str | None = None,
    error_type: str | None = None,
    parent_receipt_digest: str | None = None,
    method: str,
) -> NeutralFeatureReceipt:
    source_digest = _source_digest()
    space = _space_for_groups_with_source(group_ids, source_digest)
    environment = _environment()
    environment_digest = _digest(dict(environment))
    operation_digest = _operation_digest_from(space, group_ids, environment_digest)
    packet_digest = (
        packet_commitment.digest() if packet_commitment is not None else None
    )
    inputs = (identity.sha256,) + (
        (parent_receipt_digest,) if parent_receipt_digest is not None else ()
    )
    provenance = Provenance(
        producer="bongard.neutral_features",
        version=EXTRACTOR_VERSION,
        method=method,
        input_digests=inputs,
        artifact_digest=space.digest(),
        details=tuple(
            sorted(
                (
                    ("algorithm_id", ALGORITHM_ID),
                    ("catalog_digest", feature_group_catalog_digest()),
                    ("operation_digest", operation_digest),
                    ("packet_commitment_digest", packet_digest or "none"),
                    ("preprocessing_digest", space.preprocessing_digest),
                    ("receipt_protocol_digest", space.receipt_protocol_digest),
                )
            )
        ),
    )
    return NeutralFeatureReceipt(
        input_identity=identity,
        feature_group_ids=group_ids,
        catalog_digest=feature_group_catalog_digest(),
        feature_space=space,
        feature_space_digest=space.digest(),
        source_digest=source_digest,
        environment=environment,
        environment_digest=environment_digest,
        preprocessing_digest=space.preprocessing_digest,
        receipt_protocol_digest=space.receipt_protocol_digest,
        operation_digest=operation_digest,
        disposition=disposition,
        packet_commitment=packet_commitment,
        packet_commitment_digest=packet_digest,
        uncertainty=uncertainty,
        certificate=certificate,
        reason=reason,
        error_type=error_type,
        provenance=provenance,
        provenance_digest=provenance.digest(),
        parent_receipt_digest=parent_receipt_digest,
    )


def _evidence_from_receipt(
    receipt: NeutralFeatureReceipt,
) -> Evidence[FrozenPanelFeatures]:
    packet = receipt.verify()
    if receipt.disposition is Disposition.PRESENT:
        assert packet is not None
        return Evidence.present(packet, receipt.provenance, receipt.uncertainty)
    if receipt.disposition is Disposition.CERTIFIED_ABSENT:
        assert receipt.certificate is not None
        return Evidence.certified_absent(
            receipt.provenance, receipt.certificate, receipt.uncertainty
        )
    if receipt.disposition is Disposition.INDETERMINATE:
        assert receipt.reason is not None
        return Evidence.indeterminate(
            receipt.provenance, receipt.reason, receipt.uncertainty
        )
    assert receipt.reason is not None and receipt.error_type is not None
    return Evidence.error(receipt.provenance, receipt.error_type, receipt.reason)


def _extraction_from_receipt(
    receipt: NeutralFeatureReceipt,
) -> NeutralFeatureExtraction:
    return NeutralFeatureExtraction(_evidence_from_receipt(receipt), receipt)


def feature_group_catalog() -> tuple[NeutralFeatureGroup, ...]:
    return tuple(
        NeutralFeatureGroup(group_id, description, tuple(sorted(names)))
        for group_id, description, names in _GROUPS
    )


def feature_group_catalog_digest() -> str:
    return _digest(
        {
            "schema": "bongard.neutral-feature-group-catalog/v1",
            "groups": [item.to_data() for item in feature_group_catalog()],
        }
    )


def full_neutral_feature_space() -> FrozenFeatureSpace:
    return feature_space_for_groups(FEATURE_GROUP_IDS)


def feature_space_for_group(group_id: str) -> FrozenFeatureSpace:
    return feature_space_for_groups((group_id,))


def feature_space_for_groups(group_ids: tuple[str, ...]) -> FrozenFeatureSpace:
    return _space_for_groups_with_source(group_ids, _source_digest())


def extract_neutral_features(
    panel: bytes | str | Path,
) -> NeutralFeatureExtraction:
    """Extract the full fixed feature packet from one exact PNG input."""

    group_ids = FEATURE_GROUP_IDS
    try:
        raw, identity = _read_exact_bytes(panel)
    except Exception as exc:  # noqa: BLE001 - four-disposition boundary.
        identity = getattr(exc, "identity", None) or _fallback_identity(
            panel, "unsupported_input"
        )
        receipt = _make_receipt(
            identity=identity,
            group_ids=group_ids,
            disposition=Disposition.ERROR,
            reason=str(exc) or repr(exc),
            error_type=type(exc).__name__,
            method="input_error",
        )
        return _extraction_from_receipt(receipt)

    try:
        strength = _decode_white_distance(raw)
    except Exception as exc:  # noqa: BLE001 - four-disposition boundary.
        identity = replace(identity, kind="encoded_bytes")
        receipt = _make_receipt(
            identity=identity,
            group_ids=group_ids,
            disposition=Disposition.ERROR,
            reason=str(exc) or repr(exc),
            error_type=type(exc).__name__,
            method="decode_error",
        )
        return _extraction_from_receipt(receipt)

    identity = replace(identity, media_type="image/png")
    outcomes: list[tuple[int, dict[str, float] | str]] = []
    try:
        for threshold in _FOREGROUND_STRENGTH_THRESHOLDS:
            mask = strength >= threshold
            if not bool(mask.any()):
                outcomes.append((threshold, "zero_foreground"))
                continue
            try:
                outcomes.append((threshold, _mask_metrics(mask)))
            except _MeasurementGuard as exc:
                outcomes.append((threshold, str(exc)))
    except Exception as exc:  # noqa: BLE001 - measurement failure is ERROR.
        receipt = _make_receipt(
            identity=identity,
            group_ids=group_ids,
            disposition=Disposition.ERROR,
            reason=f"feature measurement failed: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
            method="measurement_error",
        )
        return _extraction_from_receipt(receipt)

    if all(outcome == "zero_foreground" for _, outcome in outcomes):
        operation = _operation_digest_from(
            full_neutral_feature_space(),
            group_ids,
            _digest(dict(_environment())),
        )
        receipt = _make_receipt(
            identity=identity,
            group_ids=group_ids,
            disposition=Disposition.CERTIFIED_ABSENT,
            certificate=(
                "all fixed white-distance thresholds contain exactly zero "
                f"foreground pixels; operation={operation}"
            ),
            method="foreground_absence",
        )
        return _extraction_from_receipt(receipt)

    if any(not isinstance(outcome, dict) for _, outcome in outcomes):
        states = ",".join(
            f"{threshold}:{'measured' if isinstance(outcome, dict) else outcome}"
            for threshold, outcome in outcomes
        )
        receipt = _make_receipt(
            identity=identity,
            group_ids=group_ids,
            disposition=Disposition.INDETERMINATE,
            reason=f"fixed preprocessing ensemble is not fully measurable ({states})",
            uncertainty=Uncertainty(
                0.0,
                1.0,
                causes=("preprocessing_guard_or_visibility",),
            ),
            method="measurement_guard",
        )
        return _extraction_from_receipt(receipt)

    measured = tuple(
        outcome for _, outcome in outcomes if isinstance(outcome, dict)
    )
    values = tuple(
        FeatureInterval(
            name,
            min(result[name] for result in measured),
            max(result[name] for result in measured),
        )
        for name in sorted(_DIMENSIONS)
    )
    space = full_neutral_feature_space()
    commitment = NeutralPacketCommitment(identity.sha256, space.digest(), values)
    receipt = _make_receipt(
        identity=identity,
        group_ids=group_ids,
        disposition=Disposition.PRESENT,
        packet_commitment=commitment,
        method="fixed_threshold_geometry_topology",
    )
    return _extraction_from_receipt(receipt)


def project_neutral_feature_extraction(
    extraction: NeutralFeatureExtraction,
    group_ids: str | tuple[str, ...],
) -> NeutralFeatureExtraction:
    """Project a full receipt onto one or more closed catalog groups.

    Pixel measurements are not rerun and no weights or thresholds are
    accepted.  The exact ordered group tuple and parent receipt digest are
    committed by the derived receipt.
    """

    verify_neutral_feature_extraction(extraction)
    selected = (group_ids,) if isinstance(group_ids, str) else group_ids
    selected = _normalize_group_ids(selected)
    if any(item not in extraction.receipt.feature_group_ids for item in selected):
        raise ValueError("projection selects a group absent from its parent receipt")
    parent_digest = extraction.receipt.digest()
    commitment: NeutralPacketCommitment | None = None
    if extraction.evidence.disposition is Disposition.PRESENT:
        parent_packet = extraction.evidence.unwrap()
        names = set(_dimension_names(selected))
        space = feature_space_for_groups(selected)
        values = tuple(item for item in parent_packet.values if item.name in names)
        commitment = NeutralPacketCommitment(
            parent_packet.panel_digest, space.digest(), values
        )
    receipt = _make_receipt(
        identity=extraction.receipt.input_identity,
        group_ids=selected,
        disposition=extraction.evidence.disposition,
        packet_commitment=commitment,
        uncertainty=extraction.evidence.uncertainty,
        certificate=extraction.evidence.certificate,
        reason=extraction.evidence.reason,
        error_type=extraction.evidence.error_type,
        parent_receipt_digest=parent_digest,
        method="verifier_owned_group_projection",
    )
    return _extraction_from_receipt(receipt)


def verify_neutral_feature_extraction(
    extraction: NeutralFeatureExtraction,
    expected_panel_bytes: bytes | None = None,
) -> FrozenPanelFeatures | None:
    """Cold-verify evidence against the complete canonical receipt preimage."""

    if not isinstance(extraction, NeutralFeatureExtraction):
        raise TypeError("expected NeutralFeatureExtraction")
    receipt = extraction.receipt
    packet = receipt.verify()
    expected_evidence = _evidence_from_receipt(receipt)
    if extraction.evidence != expected_evidence:
        raise ValueError("neutral evidence differs from its receipt preimage")
    if expected_panel_bytes is not None:
        if not isinstance(expected_panel_bytes, bytes):
            raise TypeError("expected_panel_bytes must be exact bytes")
        if receipt.input_identity.sha256 != hashlib.sha256(
            expected_panel_bytes
        ).hexdigest():
            raise ValueError("panel bytes differ from receipt digest")
        if receipt.input_identity.byte_count != len(expected_panel_bytes):
            raise ValueError("panel byte count differs from receipt")
    return packet


__all__ = (
    "ALGORITHM_ID",
    "EXTRACTOR_VERSION",
    "FEATURE_GROUP_IDS",
    "NeutralFeatureExtraction",
    "NeutralFeatureGroup",
    "NeutralFeatureReceipt",
    "NeutralInputIdentity",
    "NeutralPacketCommitment",
    "extract_neutral_features",
    "feature_group_catalog",
    "feature_group_catalog_digest",
    "feature_space_for_group",
    "feature_space_for_groups",
    "full_neutral_feature_space",
    "project_neutral_feature_extraction",
    "verify_neutral_feature_extraction",
)
