"""Cold-verifiable artifacts for support-relative visual prototypes.

The pre-query record contains every byte and JSON preimage needed to re-fit a
support prototype.  Query extraction is a separate record which points back
to the frozen digest.  This split is intentional: no query packet can enter
the object whose digest freezes the feature space, support sides, fit,
positive formula, and decision margin.

These are Python-authoritative canonical-JSON records.  They contain no Lean
identity and do not execute candidate-authored code.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, NoReturn

from bongard.artifacts import SupportCommitment
from bongard.evidence import Disposition, Evidence, Provenance, Uncertainty
from bongard.legs.neutral_features import (
    NeutralFeatureExtraction,
    NeutralFeatureReceipt,
)
from bongard.support_prototypes import (
    ALGORITHM_ID,
    ORIENTATION,
    ContrastiveMargin,
    FrozenFeatureSpace,
    FrozenPanelFeatures,
    FrozenSupportPrototypes,
    PositivePrototypeFormula,
    SupportPrototypePlan,
    contrastive_margin,
    evaluate_frozen_support_member,
    evaluate_support_prototype,
    panel_side_assignment_digest,
    validate_prototype_formula,
    verify_support_prototypes,
)


POLICY_SCHEMA = "bongard.support-prototype-freeze-policy/v1"
EXTRACTION_PREIMAGE_SCHEMA = "bongard.feature-extraction-preimage/v1"
PREQUERY_SCHEMA = "bongard.support-prototype-prequery-freeze/v1"
PREQUERY_COMMITMENT_SCHEMA = "bongard.support-prototype-prequery-commitment/v1"
QUERY_SCHEMA = "bongard.support-prototype-query-artifact/v1"
SUPPORT_REPLAY_SCHEMA = "bongard.support-prototype-support-replay/v1"
EVIDENCE_SCHEMA = "bongard.support-prototype-truth-evidence/v1"
REQUIRED_SUPPORT_PER_SIDE = 6


class PrototypeArtifactError(ValueError):
    """A prototype record is malformed or violates the frozen protocol."""


class PrototypeArtifactTamperError(PrototypeArtifactError):
    """A stored preimage no longer agrees with a committed identity."""


def _validate_json(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path}: JSON object keys must be strings")
        for key, item in value.items():
            _validate_json(item, f"{path}.{key}")
        return
    raise ValueError(f"{path}: unsupported JSON value {type(value).__name__}")


def canonical_json(data: object) -> bytes:
    _validate_json(data)
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(data: object) -> str:
    return hashlib.sha256(canonical_json(data)).hexdigest()


def _require_fields(data: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(data, Mapping) or set(data) != expected:
        raise ValueError(f"{label} JSON has missing or unknown fields")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty exact string")
    return value


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{label} must be a lowercase sha256")
    return text


def _identifier(value: object, label: str) -> str:
    text = _text(value, label)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", text):
        raise ValueError(f"invalid {label} {text!r}")
    return text


def _positive_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _provenance_to_data(value: Provenance) -> dict[str, object]:
    return {
        "producer": value.producer,
        "version": value.version,
        "method": value.method,
        "input_digests": list(value.input_digests),
        "artifact_digest": value.artifact_digest,
        "run_id": value.run_id,
        "details": [[key, item] for key, item in value.details],
    }


def _provenance_from_data(value: object) -> Provenance:
    data = _mapping(value, "provenance")
    _require_fields(
        data,
        {
            "producer",
            "version",
            "method",
            "input_digests",
            "artifact_digest",
            "run_id",
            "details",
        },
        "provenance",
    )
    inputs = _list(data["input_digests"], "provenance input_digests")
    details = _list(data["details"], "provenance details")
    if any(not isinstance(item, str) for item in inputs):
        raise TypeError("provenance input digests must be strings")
    parsed_details: list[tuple[str, str]] = []
    for item in details:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(part, str) for part in item)
        ):
            raise TypeError("provenance details must be string pairs")
        parsed_details.append((item[0], item[1]))
    artifact = data["artifact_digest"]
    run_id = data["run_id"]
    if artifact is not None and not isinstance(artifact, str):
        raise TypeError("provenance artifact_digest must be string or null")
    if run_id is not None and not isinstance(run_id, str):
        raise TypeError("provenance run_id must be string or null")
    return Provenance(
        producer=_text(data["producer"], "provenance producer"),
        version=_text(data["version"], "provenance version"),
        method=_text(data["method"], "provenance method"),
        input_digests=tuple(inputs),
        artifact_digest=artifact,
        run_id=run_id,
        details=tuple(parsed_details),
    )


def _uncertainty_to_data(value: Uncertainty | None) -> dict[str, object] | None:
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
    data = _mapping(value, "uncertainty")
    _require_fields(
        data, {"lower", "upper", "confidence_level", "causes"}, "uncertainty"
    )
    lower = data["lower"]
    upper = data["upper"]
    confidence = data["confidence_level"]
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in (lower, upper)):
        raise TypeError("uncertainty bounds must be real scalars")
    if confidence is not None and (
        isinstance(confidence, bool) or not isinstance(confidence, (int, float))
    ):
        raise TypeError("uncertainty confidence_level must be a real scalar or null")
    causes = _list(data["causes"], "uncertainty causes")
    if any(not isinstance(item, str) for item in causes):
        raise TypeError("uncertainty causes must be strings")
    return Uncertainty(
        float(lower),
        float(upper),
        float(confidence) if confidence is not None else None,
        tuple(causes),
    )


@dataclass(frozen=True)
class PrototypeTruthEvidence:
    """Strict JSON projection which preserves all four dispositions."""

    disposition: Disposition
    provenance: Provenance
    value: bool | None = None
    uncertainty: Uncertainty | None = None
    certificate: str | None = None
    reason: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        evidence = self.to_evidence()
        if evidence.disposition is Disposition.PRESENT and evidence.unwrap() is not True:
            raise ValueError("present prototype evidence must contain exactly true")

    @classmethod
    def from_evidence(cls, evidence: Evidence[bool]) -> "PrototypeTruthEvidence":
        if not isinstance(evidence, Evidence):
            raise TypeError("prototype result must be typed Evidence")
        if evidence.disposition is Disposition.PRESENT and evidence.unwrap() is not True:
            raise ValueError("present prototype evidence must contain exactly true")
        return cls(
            evidence.disposition,
            evidence.provenance,
            evidence.value,
            evidence.uncertainty,
            evidence.certificate,
            evidence.reason,
            evidence.error_type,
        )

    def to_evidence(self) -> Evidence[bool]:
        return Evidence(
            disposition=self.disposition,
            provenance=self.provenance,
            value=self.value,
            uncertainty=self.uncertainty,
            certificate=self.certificate,
            reason=self.reason,
            error_type=self.error_type,
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": EVIDENCE_SCHEMA,
            "disposition": self.disposition.value,
            "provenance": _provenance_to_data(self.provenance),
            "value": self.value,
            "uncertainty": _uncertainty_to_data(self.uncertainty),
            "certificate": self.certificate,
            "reason": self.reason,
            "error_type": self.error_type,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PrototypeTruthEvidence":
        _require_fields(
            data,
            {
                "schema",
                "disposition",
                "provenance",
                "value",
                "uncertainty",
                "certificate",
                "reason",
                "error_type",
            },
            "prototype evidence",
        )
        if data["schema"] != EVIDENCE_SCHEMA:
            raise ValueError("unsupported prototype-evidence schema")
        if not isinstance(data["disposition"], str):
            raise TypeError("evidence disposition must be a string")
        value = data["value"]
        if value is not None and value is not True:
            raise ValueError("prototype evidence value must be exactly true or null")
        for field in ("certificate", "reason", "error_type"):
            if data[field] is not None and not isinstance(data[field], str):
                raise TypeError(f"evidence {field} must be a string or null")
        return cls(
            disposition=Disposition(data["disposition"]),
            provenance=_provenance_from_data(data["provenance"]),
            value=value,
            uncertainty=_uncertainty_from_data(data["uncertainty"]),
            certificate=data["certificate"],
            reason=data["reason"],
            error_type=data["error_type"],
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def __bool__(self) -> NoReturn:
        raise TypeError("prototype evidence has four dispositions and cannot be bool")


@dataclass(frozen=True, order=True)
class AllowedPrototypeFeatureGroup:
    """One catalog entry whose space and threshold are verifier-owned."""

    feature_group_id: str
    feature_space_digest: str
    preprocessing_digest: str
    decision_margin: float

    def __post_init__(self) -> None:
        _identifier(self.feature_group_id, "feature_group_id")
        _digest(self.feature_space_digest, "feature_space_digest")
        _digest(self.preprocessing_digest, "preprocessing_digest")
        object.__setattr__(
            self,
            "decision_margin",
            _positive_real(self.decision_margin, "decision_margin"),
        )

    def to_data(self) -> dict[str, object]:
        return {
            "feature_group_id": self.feature_group_id,
            "feature_space_digest": self.feature_space_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "decision_margin": self.decision_margin,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "AllowedPrototypeFeatureGroup":
        _require_fields(
            data,
            {
                "feature_group_id",
                "feature_space_digest",
                "preprocessing_digest",
                "decision_margin",
            },
            "allowed feature group",
        )
        return cls(
            data["feature_group_id"],
            data["feature_space_digest"],
            data["preprocessing_digest"],
            data["decision_margin"],
        )


@dataclass(frozen=True)
class PrototypeFreezePolicy:
    """Verifier-owned, pre-support catalog and fixed group thresholds."""

    feature_catalog_digest: str
    allowed_feature_groups: tuple[AllowedPrototypeFeatureGroup, ...]
    extractor_id: str
    extractor_version: str
    extractor_artifact_digest: str
    receipt_protocol_digest: str
    minimum_per_side: int

    def __post_init__(self) -> None:
        for field in (
            "feature_catalog_digest",
            "extractor_artifact_digest",
            "receipt_protocol_digest",
        ):
            _digest(getattr(self, field), field)
        if (
            not isinstance(self.allowed_feature_groups, tuple)
            or not self.allowed_feature_groups
            or any(
                not isinstance(item, AllowedPrototypeFeatureGroup)
                for item in self.allowed_feature_groups
            )
        ):
            raise TypeError("allowed_feature_groups must be a nonempty typed tuple")
        if list(self.allowed_feature_groups) != sorted(self.allowed_feature_groups):
            raise ValueError("allowed feature groups must be sorted")
        ids = [item.feature_group_id for item in self.allowed_feature_groups]
        if len(ids) != len(set(ids)):
            raise ValueError("allowed feature group ids must be unique")
        _text(self.extractor_id, "extractor_id")
        _text(self.extractor_version, "extractor_version")
        if isinstance(self.minimum_per_side, bool) or self.minimum_per_side != REQUIRED_SUPPORT_PER_SIDE:
            raise ValueError("prototype policy requires exactly six panels per side")

    @classmethod
    def create(
        cls,
        *,
        feature_catalog_digest: str,
        allowed_groups: Mapping[str, tuple[FrozenFeatureSpace, float]],
    ) -> "PrototypeFreezePolicy":
        if not isinstance(allowed_groups, Mapping) or not allowed_groups:
            raise ValueError("allowed_groups must be a nonempty mapping")
        spaces = tuple(value[0] for value in allowed_groups.values())
        if any(not isinstance(space, FrozenFeatureSpace) for space in spaces):
            raise TypeError("allowed group spaces must be FrozenFeatureSpace records")
        identity = {
            (
                space.extractor_id,
                space.extractor_version,
                space.extractor_artifact_digest,
                space.receipt_protocol_digest,
            )
            for space in spaces
        }
        if len(identity) != 1:
            raise ValueError("allowed feature groups must share extractor identity")
        first = spaces[0]
        return cls(
            feature_catalog_digest=feature_catalog_digest,
            allowed_feature_groups=tuple(
                sorted(
                    AllowedPrototypeFeatureGroup(
                        group_id,
                        space.digest(),
                        space.preprocessing_digest,
                        margin,
                    )
                    for group_id, (space, margin) in allowed_groups.items()
                )
            ),
            extractor_id=first.extractor_id,
            extractor_version=first.extractor_version,
            extractor_artifact_digest=first.extractor_artifact_digest,
            receipt_protocol_digest=first.receipt_protocol_digest,
            minimum_per_side=REQUIRED_SUPPORT_PER_SIDE,
        )

    def select(
        self, feature_group_id: str, feature_space: FrozenFeatureSpace
    ) -> AllowedPrototypeFeatureGroup:
        matches = tuple(
            item
            for item in self.allowed_feature_groups
            if item.feature_group_id == feature_group_id
        )
        if not matches:
            raise PrototypeArtifactTamperError(
                "selected feature group was not precommitted by policy"
            )
        selected = matches[0]
        expected = (
            feature_space.digest(),
            feature_space.extractor_id,
            feature_space.extractor_version,
            feature_space.extractor_artifact_digest,
            feature_space.preprocessing_digest,
            feature_space.receipt_protocol_digest,
        )
        actual = (
            selected.feature_space_digest,
            self.extractor_id,
            self.extractor_version,
            self.extractor_artifact_digest,
            selected.preprocessing_digest,
            self.receipt_protocol_digest,
        )
        if actual != expected:
            raise PrototypeArtifactTamperError(
                "selected feature space differs from frozen policy"
            )
        return selected

    def to_data(self) -> dict[str, object]:
        return {
            "schema": POLICY_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "orientation": ORIENTATION,
            "feature_catalog_digest": self.feature_catalog_digest,
            "allowed_feature_groups": [
                item.to_data() for item in self.allowed_feature_groups
            ],
            "extractor_identity": {
                "extractor_id": self.extractor_id,
                "extractor_version": self.extractor_version,
                "extractor_artifact_digest": self.extractor_artifact_digest,
                "receipt_protocol_digest": self.receipt_protocol_digest,
            },
            "minimum_per_side": self.minimum_per_side,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PrototypeFreezePolicy":
        _require_fields(
            data,
            {
                "schema",
                "algorithm_id",
                "orientation",
                "feature_catalog_digest",
                "allowed_feature_groups",
                "extractor_identity",
                "minimum_per_side",
            },
            "prototype policy",
        )
        if data["schema"] != POLICY_SCHEMA or data["algorithm_id"] != ALGORITHM_ID:
            raise ValueError("unsupported prototype policy")
        if data["orientation"] != ORIENTATION:
            raise ValueError("prototype policy attempts a polarity change")
        groups = _list(data["allowed_feature_groups"], "allowed feature groups")
        if not groups:
            raise ValueError("prototype policy requires a nonempty feature catalog")
        identity = _mapping(data["extractor_identity"], "extractor identity")
        _require_fields(
            identity,
            {
                "extractor_id",
                "extractor_version",
                "extractor_artifact_digest",
                "receipt_protocol_digest",
            },
            "extractor identity",
        )
        return cls(
            feature_catalog_digest=data["feature_catalog_digest"],
            allowed_feature_groups=tuple(
                AllowedPrototypeFeatureGroup.from_data(
                    _mapping(item, "allowed feature group")
                )
                for item in groups
            ),
            extractor_id=identity["extractor_id"],
            extractor_version=identity["extractor_version"],
            extractor_artifact_digest=identity["extractor_artifact_digest"],
            receipt_protocol_digest=identity["receipt_protocol_digest"],
            minimum_per_side=data["minimum_per_side"],
        )

    def digest(self) -> str:
        return canonical_digest(self.to_data())


def _receipt_evidence(
    receipt: NeutralFeatureReceipt,
) -> Evidence[FrozenPanelFeatures]:
    """Reconstruct the typed extraction outcome from its verified receipt."""

    packet = receipt.verify()
    data = receipt.to_data()
    disposition_raw = data.get("disposition")
    if not isinstance(disposition_raw, str):
        raise PrototypeArtifactTamperError("neutral receipt lacks a disposition")
    disposition = Disposition(disposition_raw)
    provenance = _provenance_from_data(data.get("provenance"))
    uncertainty = _uncertainty_from_data(data.get("uncertainty"))
    certificate = data.get("certificate")
    reason = data.get("reason")
    error_type = data.get("error_type")
    if disposition is Disposition.PRESENT:
        if packet is None:
            raise PrototypeArtifactTamperError(
                "present neutral receipt did not reconstruct a feature packet"
            )
        return Evidence.present(packet, provenance, uncertainty)
    if packet is not None:
        raise PrototypeArtifactTamperError(
            "non-present neutral receipt reconstructed a feature packet"
        )
    if disposition is Disposition.CERTIFIED_ABSENT:
        if not isinstance(certificate, str):
            raise PrototypeArtifactTamperError(
                "certified-absent neutral receipt lacks a certificate"
            )
        return Evidence.certified_absent(provenance, certificate, uncertainty)
    if disposition is Disposition.INDETERMINATE:
        if not isinstance(reason, str):
            raise PrototypeArtifactTamperError(
                "indeterminate neutral receipt lacks a reason"
            )
        return Evidence.indeterminate(provenance, reason, uncertainty)
    if not isinstance(reason, str) or not isinstance(error_type, str):
        raise PrototypeArtifactTamperError(
            "error neutral receipt lacks an error type or reason"
        )
    return Evidence.error(provenance, error_type, reason)


@dataclass(frozen=True)
class FeatureExtractionPreimage:
    """Exact panel bytes and neutral receipt, with a packet only when present."""

    panel_bytes: bytes
    receipt: NeutralFeatureReceipt
    feature_packet: FrozenPanelFeatures | None

    def __post_init__(self) -> None:
        if not isinstance(self.panel_bytes, bytes) or not self.panel_bytes:
            raise TypeError("panel preimage must be nonempty exact bytes")
        if not isinstance(self.receipt, NeutralFeatureReceipt):
            raise TypeError("extraction preimage requires a NeutralFeatureReceipt")
        if self.feature_packet is not None and not isinstance(
            self.feature_packet, FrozenPanelFeatures
        ):
            raise TypeError("feature_packet must be FrozenPanelFeatures or null")
        self.verify()

    @classmethod
    def from_extraction(
        cls, panel_bytes: bytes, extraction: NeutralFeatureExtraction
    ) -> "FeatureExtractionPreimage":
        if not isinstance(extraction, NeutralFeatureExtraction):
            raise TypeError("expected NeutralFeatureExtraction")
        packet = (
            extraction.evidence.unwrap()
            if extraction.evidence.disposition is Disposition.PRESENT
            else None
        )
        return cls(panel_bytes, extraction.receipt, packet)

    @property
    def panel_digest(self) -> str:
        identity = _mapping(self.receipt.to_data().get("input_identity"), "input identity")
        return _digest(identity.get("sha256"), "input identity sha256")

    @property
    def feature_group_id(self) -> str:
        raw = _list(
            self.receipt.to_data().get("feature_group_ids"),
            "receipt feature_group_ids",
        )
        if len(raw) != 1:
            raise PrototypeArtifactError(
                "PURE prototype extraction must select exactly one feature group"
            )
        return _identifier(raw[0], "receipt feature_group_id")

    def extraction_evidence(self) -> Evidence[FrozenPanelFeatures]:
        return _receipt_evidence(self.receipt)

    def require_present(self) -> FrozenPanelFeatures:
        evidence = self.extraction_evidence()
        if evidence.disposition is not Disposition.PRESENT:
            raise PrototypeArtifactError(
                "support extraction must be present; failure is not a negative"
            )
        return evidence.unwrap()

    def verify(self, feature_space: FrozenFeatureSpace | None = None) -> None:
        receipt_data = self.receipt.to_data()
        identity = _mapping(receipt_data.get("input_identity"), "input identity")
        _require_fields(
            identity,
            {"kind", "sha256", "byte_count", "media_type"},
            "receipt input identity",
        )
        expected_sha = hashlib.sha256(self.panel_bytes).hexdigest()
        if identity["sha256"] != expected_sha:
            raise PrototypeArtifactTamperError(
                "panel bytes differ from neutral receipt input identity"
            )
        if (
            isinstance(identity["byte_count"], bool)
            or not isinstance(identity["byte_count"], int)
            or identity["byte_count"] != len(self.panel_bytes)
        ):
            raise PrototypeArtifactTamperError(
                "panel byte count differs from neutral receipt input identity"
            )
        reconstructed = self.receipt.verify()
        if reconstructed != self.feature_packet:
            raise PrototypeArtifactTamperError(
                "stored feature packet differs from neutral receipt preimage"
            )
        if reconstructed is not None:
            if reconstructed.extractor_receipt_digest != self.receipt.digest():
                raise PrototypeArtifactTamperError(
                    "feature packet does not bind the archived receipt"
                )
            if reconstructed.panel_digest != expected_sha:
                raise PrototypeArtifactTamperError(
                    "feature packet does not bind the archived panel bytes"
                )
            if feature_space is not None:
                reconstructed.validate(feature_space)
        elif feature_space is not None:
            receipt_space = _mapping(receipt_data.get("feature_space"), "receipt feature space")
            if FrozenFeatureSpace.from_data(receipt_space) != feature_space:
                raise PrototypeArtifactTamperError(
                    "failed extraction receipt names another feature space"
                )

    def to_data(self) -> dict[str, object]:
        packet_data = (
            self.feature_packet.to_data() if self.feature_packet is not None else None
        )
        encoded = base64.b64encode(self.panel_bytes).decode("ascii")
        return {
            "schema": EXTRACTION_PREIMAGE_SCHEMA,
            "panel_encoding": "base64",
            "panel_base64": encoded,
            "panel_sha256": hashlib.sha256(self.panel_bytes).hexdigest(),
            "panel_byte_count": len(self.panel_bytes),
            "feature_packet": packet_data,
            "feature_packet_digest": (
                self.feature_packet.digest()
                if self.feature_packet is not None
                else None
            ),
            "extractor_receipt": self.receipt.to_data(),
            "extractor_receipt_digest": self.receipt.digest(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "FeatureExtractionPreimage":
        _require_fields(
            data,
            {
                "schema",
                "panel_encoding",
                "panel_base64",
                "panel_sha256",
                "panel_byte_count",
                "feature_packet",
                "feature_packet_digest",
                "extractor_receipt",
                "extractor_receipt_digest",
            },
            "feature extraction preimage",
        )
        if data["schema"] != EXTRACTION_PREIMAGE_SCHEMA:
            raise ValueError("unsupported feature-extraction preimage schema")
        if data["panel_encoding"] != "base64" or not isinstance(
            data["panel_base64"], str
        ):
            raise ValueError("panel preimage must use canonical base64")
        try:
            panel_bytes = base64.b64decode(data["panel_base64"], validate=True)
        except Exception as exc:  # noqa: BLE001 - serialization boundary.
            raise ValueError("panel_base64 is invalid") from exc
        if base64.b64encode(panel_bytes).decode("ascii") != data["panel_base64"]:
            raise ValueError("panel_base64 is not canonical")
        if data["panel_sha256"] != hashlib.sha256(panel_bytes).hexdigest():
            raise PrototypeArtifactTamperError("panel preimage digest drift")
        if (
            isinstance(data["panel_byte_count"], bool)
            or not isinstance(data["panel_byte_count"], int)
            or data["panel_byte_count"] != len(panel_bytes)
        ):
            raise PrototypeArtifactTamperError("panel preimage byte-count drift")
        receipt = NeutralFeatureReceipt.from_data(
            _mapping(data["extractor_receipt"], "extractor receipt")
        )
        if data["extractor_receipt_digest"] != receipt.digest():
            raise PrototypeArtifactTamperError("extractor receipt digest drift")
        packet_raw = data["feature_packet"]
        if packet_raw is None:
            packet = None
            if data["feature_packet_digest"] is not None:
                raise PrototypeArtifactTamperError(
                    "absent feature packet carries a digest"
                )
        else:
            packet = FrozenPanelFeatures.from_data(
                _mapping(packet_raw, "feature packet")
            )
            if data["feature_packet_digest"] != packet.digest():
                raise PrototypeArtifactTamperError("feature packet digest drift")
        return cls(panel_bytes, receipt, packet)

    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def __bool__(self) -> NoReturn:
        raise TypeError("feature extraction has four dispositions and cannot be bool")


@dataclass(frozen=True)
class PrototypeCompilerInputs:
    feature_space: FrozenFeatureSpace
    prototypes: FrozenSupportPrototypes
    formula: PositivePrototypeFormula
    feature_group_id: str
    decision_margin: float
    semantic_proposal_digest: str


@dataclass(frozen=True, order=True)
class PrototypeSupportGateEvidence:
    """Replayed evidence for one exact archived support member."""

    panel_digest: str
    positive: bool
    evidence: PrototypeTruthEvidence

    def __post_init__(self) -> None:
        _digest(self.panel_digest, "support gate panel_digest")
        if not isinstance(self.positive, bool):
            raise TypeError("support gate side must be bool")
        if not isinstance(self.evidence, PrototypeTruthEvidence):
            raise TypeError("support gate evidence must preserve four dispositions")

    def to_data(self) -> dict[str, object]:
        return {
            "panel_digest": self.panel_digest,
            "positive": self.positive,
            "evidence": self.evidence.to_data(),
            "evidence_digest": self.evidence.digest(),
        }


@dataclass(frozen=True)
class PrototypePreQueryFreeze:
    """Complete support-derived preimage, frozen before query extraction."""

    support_commitment_digest: str
    policy: PrototypeFreezePolicy
    policy_digest: str
    selected_feature_group_id: str
    feature_space: FrozenFeatureSpace
    feature_space_digest: str
    positive_support: tuple[FeatureExtractionPreimage, ...]
    negative_support: tuple[FeatureExtractionPreimage, ...]
    support_assignment_digest: str
    fit_plan: SupportPrototypePlan
    fit_plan_digest: str
    prototypes: FrozenSupportPrototypes
    prototype_digest: str
    positive_formula: PositivePrototypeFormula
    positive_formula_digest: str
    fixed_decision_margin: float
    semantic_proposal_digest: str

    def __post_init__(self) -> None:
        for field in (
            "support_commitment_digest",
            "policy_digest",
            "feature_space_digest",
            "support_assignment_digest",
            "fit_plan_digest",
            "prototype_digest",
            "positive_formula_digest",
            "semantic_proposal_digest",
        ):
            _digest(getattr(self, field), field)
        _identifier(self.selected_feature_group_id, "selected_feature_group_id")
        object.__setattr__(
            self,
            "fixed_decision_margin",
            _positive_real(self.fixed_decision_margin, "fixed_decision_margin"),
        )
        self.verify()

    @classmethod
    def create(
        cls,
        *,
        support_commitment: SupportCommitment,
        policy: PrototypeFreezePolicy,
        selected_feature_group_id: str,
        feature_space: FrozenFeatureSpace,
        positive_support: tuple[FeatureExtractionPreimage, ...],
        negative_support: tuple[FeatureExtractionPreimage, ...],
        fit_plan: SupportPrototypePlan,
        prototypes: FrozenSupportPrototypes,
        positive_formula: PositivePrototypeFormula,
        semantic_proposal_digest: str,
    ) -> "PrototypePreQueryFreeze":
        selected = policy.select(selected_feature_group_id, feature_space)
        return cls(
            support_commitment_digest=support_commitment.digest(),
            policy=policy,
            policy_digest=policy.digest(),
            selected_feature_group_id=selected_feature_group_id,
            feature_space=feature_space,
            feature_space_digest=feature_space.digest(),
            positive_support=tuple(
                sorted(positive_support, key=lambda item: item.panel_digest)
            ),
            negative_support=tuple(
                sorted(negative_support, key=lambda item: item.panel_digest)
            ),
            support_assignment_digest=fit_plan.support_assignment_digest,
            fit_plan=fit_plan,
            fit_plan_digest=fit_plan.digest(),
            prototypes=prototypes,
            prototype_digest=prototypes.digest(),
            positive_formula=positive_formula,
            positive_formula_digest=positive_formula.digest(),
            fixed_decision_margin=selected.decision_margin,
            semantic_proposal_digest=semantic_proposal_digest,
        ).verified_against_support(support_commitment)

    def verified_against_support(
        self, support_commitment: SupportCommitment
    ) -> "PrototypePreQueryFreeze":
        self.verify(support_commitment)
        return self

    def verify(self, support_commitment: SupportCommitment | None = None) -> None:
        if self.policy.digest() != self.policy_digest:
            raise PrototypeArtifactTamperError("prototype policy digest drift")
        selected = self.policy.select(
            self.selected_feature_group_id, self.feature_space
        )
        if self.feature_space.digest() != self.feature_space_digest:
            raise PrototypeArtifactTamperError("feature-space digest drift")
        if len(self.positive_support) != REQUIRED_SUPPORT_PER_SIDE or len(
            self.negative_support
        ) != REQUIRED_SUPPORT_PER_SIDE:
            raise PrototypeArtifactError("pre-query freeze requires exactly 6+6 support")
        for label, records in (
            ("positive", self.positive_support),
            ("negative", self.negative_support),
        ):
            if not isinstance(records, tuple) or any(
                not isinstance(item, FeatureExtractionPreimage) for item in records
            ):
                raise TypeError(f"{label} support must be an immutable typed tuple")
            panels = [item.panel_digest for item in records]
            if panels != sorted(panels) or len(panels) != len(set(panels)):
                raise PrototypeArtifactError(
                    f"{label} support panel identities must be unique and sorted"
                )
            for item in records:
                item.verify(self.feature_space)
                if item.feature_group_id != self.selected_feature_group_id:
                    raise PrototypeArtifactTamperError(
                        "support receipt names another feature group"
                    )
                if (
                    item.receipt.to_data().get("catalog_digest")
                    != self.policy.feature_catalog_digest
                ):
                    raise PrototypeArtifactTamperError(
                        "support receipt names another feature catalog"
                    )
        positive_packets = tuple(item.require_present() for item in self.positive_support)
        negative_packets = tuple(item.require_present() for item in self.negative_support)
        positive_panels = tuple(item.panel_digest for item in self.positive_support)
        negative_panels = tuple(item.panel_digest for item in self.negative_support)
        if set(positive_panels) & set(negative_panels):
            raise PrototypeArtifactError("a support panel occurs on both sides")
        assignment = panel_side_assignment_digest(positive_panels, negative_panels)
        if assignment != self.support_assignment_digest:
            raise PrototypeArtifactTamperError("support-side assignment digest drift")
        if self.fit_plan.digest() != self.fit_plan_digest:
            raise PrototypeArtifactTamperError("fit-plan digest drift")
        if (
            self.fit_plan.feature_space_digest != self.feature_space_digest
            or self.fit_plan.support_assignment_digest != assignment
            or self.fit_plan.minimum_per_side != self.policy.minimum_per_side
        ):
            raise PrototypeArtifactTamperError("fit plan differs from frozen policy/support")
        if self.prototypes.digest() != self.prototype_digest:
            raise PrototypeArtifactTamperError("prototype digest drift")
        verify_support_prototypes(
            self.prototypes,
            self.fit_plan,
            self.feature_space,
            positive_packets,
            negative_packets,
        )
        if self.positive_formula.digest() != self.positive_formula_digest:
            raise PrototypeArtifactTamperError("positive-formula digest drift")
        validate_prototype_formula(
            self.positive_formula, self.prototypes, self.feature_space
        )
        if (
            self.fixed_decision_margin != selected.decision_margin
            or self.positive_formula.decision_margin != selected.decision_margin
        ):
            raise PrototypeArtifactTamperError(
                "formula margin differs from precommitted group threshold"
            )
        if support_commitment is not None:
            if support_commitment.digest() != self.support_commitment_digest:
                raise PrototypeArtifactTamperError("support commitment digest drift")
            expected = {
                (item.panel.sha256, item.positive)
                for item in support_commitment.support
            }
            observed = {
                *((digest, True) for digest in positive_panels),
                *((digest, False) for digest in negative_panels),
            }
            if expected != observed or len(expected) != 12:
                raise PrototypeArtifactTamperError(
                    "prototype support panels/sides differ from support commitment"
                )

    def compiler_inputs(self) -> PrototypeCompilerInputs:
        return PrototypeCompilerInputs(
            self.feature_space,
            self.prototypes,
            self.positive_formula,
            self.selected_feature_group_id,
            self.fixed_decision_margin,
            self.semantic_proposal_digest,
        )

    def replay_support_gate(self) -> tuple[PrototypeSupportGateEvidence, ...]:
        """Re-evaluate only exact fitted support members, never held-out queries."""

        records: list[PrototypeSupportGateEvidence] = []
        for positive, side in (
            (True, self.positive_support),
            (False, self.negative_support),
        ):
            for item in side:
                evidence = evaluate_frozen_support_member(
                    self.positive_formula,
                    self.prototypes,
                    self.feature_space,
                    item.extraction_evidence(),
                )
                records.append(
                    PrototypeSupportGateEvidence(
                        item.panel_digest,
                        positive,
                        PrototypeTruthEvidence.from_evidence(evidence),
                    )
                )
        return tuple(sorted(records))

    @property
    def support_panel_digests(self) -> frozenset[str]:
        return frozenset(
            item.panel_digest
            for item in self.positive_support + self.negative_support
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PREQUERY_SCHEMA,
            "support_commitment_digest": self.support_commitment_digest,
            "policy": self.policy.to_data(),
            "policy_digest": self.policy_digest,
            "selected_feature_group_id": self.selected_feature_group_id,
            "feature_space": self.feature_space.to_data(),
            "feature_space_digest": self.feature_space_digest,
            "positive_support": [item.to_data() for item in self.positive_support],
            "negative_support": [item.to_data() for item in self.negative_support],
            "support_assignment_digest": self.support_assignment_digest,
            "fit_plan": self.fit_plan.to_data(),
            "fit_plan_digest": self.fit_plan_digest,
            "prototypes": self.prototypes.to_data(),
            "prototype_digest": self.prototype_digest,
            "positive_formula": self.positive_formula.to_data(),
            "positive_formula_digest": self.positive_formula_digest,
            "fixed_decision_margin": self.fixed_decision_margin,
            "semantic_proposal_digest": self.semantic_proposal_digest,
        }

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        *,
        support_commitment: SupportCommitment | None = None,
    ) -> "PrototypePreQueryFreeze":
        _require_fields(
            data,
            {
                "schema",
                "support_commitment_digest",
                "policy",
                "policy_digest",
                "selected_feature_group_id",
                "feature_space",
                "feature_space_digest",
                "positive_support",
                "negative_support",
                "support_assignment_digest",
                "fit_plan",
                "fit_plan_digest",
                "prototypes",
                "prototype_digest",
                "positive_formula",
                "positive_formula_digest",
                "fixed_decision_margin",
                "semantic_proposal_digest",
            },
            "prototype pre-query freeze",
        )
        if data["schema"] != PREQUERY_SCHEMA:
            raise ValueError("unsupported prototype pre-query schema")

        def support_side(name: str) -> tuple[FeatureExtractionPreimage, ...]:
            return tuple(
                FeatureExtractionPreimage.from_data(
                    _mapping(item, f"{name} extraction preimage")
                )
                for item in _list(data[name], name)
            )

        result = cls(
            support_commitment_digest=data["support_commitment_digest"],
            policy=PrototypeFreezePolicy.from_data(_mapping(data["policy"], "policy")),
            policy_digest=data["policy_digest"],
            selected_feature_group_id=data["selected_feature_group_id"],
            feature_space=FrozenFeatureSpace.from_data(
                _mapping(data["feature_space"], "feature space")
            ),
            feature_space_digest=data["feature_space_digest"],
            positive_support=support_side("positive_support"),
            negative_support=support_side("negative_support"),
            support_assignment_digest=data["support_assignment_digest"],
            fit_plan=SupportPrototypePlan.from_data(
                _mapping(data["fit_plan"], "fit plan")
            ),
            fit_plan_digest=data["fit_plan_digest"],
            prototypes=FrozenSupportPrototypes.from_data(
                _mapping(data["prototypes"], "prototypes")
            ),
            prototype_digest=data["prototype_digest"],
            positive_formula=PositivePrototypeFormula.from_data(
                _mapping(data["positive_formula"], "positive formula")
            ),
            positive_formula_digest=data["positive_formula_digest"],
            fixed_decision_margin=data["fixed_decision_margin"],
            semantic_proposal_digest=data["semantic_proposal_digest"],
        )
        if support_commitment is not None:
            result.verify(support_commitment)
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())

    def committed_data(self) -> dict[str, object]:
        return {
            "schema": PREQUERY_COMMITMENT_SCHEMA,
            "pre_query_freeze": self.to_data(),
            "pre_query_freeze_digest": self.digest(),
        }

    @classmethod
    def from_committed_data(
        cls,
        data: Mapping[str, Any],
        *,
        support_commitment: SupportCommitment | None = None,
    ) -> "PrototypePreQueryFreeze":
        _require_fields(
            data,
            {"schema", "pre_query_freeze", "pre_query_freeze_digest"},
            "pre-query commitment",
        )
        if data["schema"] != PREQUERY_COMMITMENT_SCHEMA:
            raise ValueError("unsupported pre-query commitment schema")
        result = cls.from_data(
            _mapping(data["pre_query_freeze"], "pre-query freeze"),
            support_commitment=support_commitment,
        )
        if data["pre_query_freeze_digest"] != result.digest():
            raise PrototypeArtifactTamperError("pre-query freeze digest drift")
        return result


@dataclass(frozen=True)
class PrototypeQueryArtifact:
    """One post-freeze query extraction and its replayable predicate evidence."""

    query_id: str
    pre_query_freeze_digest: str
    query_panel_digest: str
    extraction: FeatureExtractionPreimage
    extraction_digest: str
    margin: ContrastiveMargin | None
    margin_digest: str | None
    evidence: PrototypeTruthEvidence
    evidence_digest: str

    def __post_init__(self) -> None:
        _identifier(self.query_id, "query_id")
        for field in (
            "pre_query_freeze_digest",
            "query_panel_digest",
            "extraction_digest",
            "evidence_digest",
        ):
            _digest(getattr(self, field), field)
        if not isinstance(self.extraction, FeatureExtractionPreimage):
            raise TypeError("query extraction must be a typed preimage")
        if not isinstance(self.evidence, PrototypeTruthEvidence):
            raise TypeError("query result must preserve four dispositions")
        if self.extraction.digest() != self.extraction_digest:
            raise PrototypeArtifactTamperError("query extraction digest drift")
        if self.extraction.panel_digest != self.query_panel_digest:
            raise PrototypeArtifactTamperError("query panel digest drift")
        if self.evidence.digest() != self.evidence_digest:
            raise PrototypeArtifactTamperError("query evidence digest drift")
        if self.margin is None:
            if self.margin_digest is not None:
                raise PrototypeArtifactTamperError(
                    "query without a margin carries a margin digest"
                )
        else:
            if not isinstance(self.margin, ContrastiveMargin):
                raise TypeError("query margin must be ContrastiveMargin or null")
            if self.margin_digest != self.margin.digest():
                raise PrototypeArtifactTamperError("query margin digest drift")

    @classmethod
    def capture(
        cls,
        *,
        query_id: str,
        freeze: PrototypePreQueryFreeze,
        extraction: FeatureExtractionPreimage,
    ) -> "PrototypeQueryArtifact":
        freeze.verify()
        extraction.verify(freeze.feature_space)
        if extraction.feature_group_id != freeze.selected_feature_group_id:
            raise PrototypeArtifactTamperError(
                "query receipt names another feature group"
            )
        if (
            extraction.receipt.to_data().get("catalog_digest")
            != freeze.policy.feature_catalog_digest
        ):
            raise PrototypeArtifactTamperError(
                "query receipt names another feature catalog"
            )
        if extraction.panel_digest in freeze.support_panel_digests:
            raise PrototypeArtifactTamperError("query panel overlaps frozen support")
        upstream = extraction.extraction_evidence()
        margin = (
            contrastive_margin(
                upstream.unwrap(), freeze.prototypes, freeze.feature_space
            )
            if upstream.disposition is Disposition.PRESENT
            else None
        )
        evidence = evaluate_support_prototype(
            freeze.positive_formula,
            freeze.prototypes,
            freeze.feature_space,
            upstream,
        )
        record = PrototypeTruthEvidence.from_evidence(evidence)
        return cls(
            query_id=query_id,
            pre_query_freeze_digest=freeze.digest(),
            query_panel_digest=extraction.panel_digest,
            extraction=extraction,
            extraction_digest=extraction.digest(),
            margin=margin,
            margin_digest=margin.digest() if margin is not None else None,
            evidence=record,
            evidence_digest=record.digest(),
        )

    def verify(self, freeze: PrototypePreQueryFreeze) -> None:
        freeze.verify()
        if self.pre_query_freeze_digest != freeze.digest():
            raise PrototypeArtifactTamperError(
                "query artifact names another pre-query freeze"
            )
        rebuilt = self.capture(
            query_id=self.query_id,
            freeze=freeze,
            extraction=self.extraction,
        )
        if rebuilt != self:
            raise PrototypeArtifactTamperError(
                "query margin/evidence differs from model-free replay"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": QUERY_SCHEMA,
            "query_id": self.query_id,
            "pre_query_freeze_digest": self.pre_query_freeze_digest,
            "query_panel_digest": self.query_panel_digest,
            "extraction": self.extraction.to_data(),
            "extraction_digest": self.extraction_digest,
            "margin": self.margin.to_data() if self.margin is not None else None,
            "margin_digest": self.margin_digest,
            "evidence": self.evidence.to_data(),
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        *,
        freeze: PrototypePreQueryFreeze | None = None,
    ) -> "PrototypeQueryArtifact":
        _require_fields(
            data,
            {
                "schema",
                "query_id",
                "pre_query_freeze_digest",
                "query_panel_digest",
                "extraction",
                "extraction_digest",
                "margin",
                "margin_digest",
                "evidence",
                "evidence_digest",
            },
            "prototype query artifact",
        )
        if data["schema"] != QUERY_SCHEMA:
            raise ValueError("unsupported prototype-query schema")
        margin_raw = data["margin"]
        margin = (
            None
            if margin_raw is None
            else ContrastiveMargin.from_data(_mapping(margin_raw, "query margin"))
        )
        result = cls(
            query_id=data["query_id"],
            pre_query_freeze_digest=data["pre_query_freeze_digest"],
            query_panel_digest=data["query_panel_digest"],
            extraction=FeatureExtractionPreimage.from_data(
                _mapping(data["extraction"], "query extraction")
            ),
            extraction_digest=data["extraction_digest"],
            margin=margin,
            margin_digest=data["margin_digest"],
            evidence=PrototypeTruthEvidence.from_data(
                _mapping(data["evidence"], "query evidence")
            ),
            evidence_digest=data["evidence_digest"],
        )
        if freeze is not None:
            result.verify(freeze)
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


@dataclass(frozen=True)
class PrototypeSupportReplayArtifact:
    """Fresh, label-blind extraction replay for one committed support panel."""

    pre_query_freeze_digest: str
    panel_digest: str
    extraction: FeatureExtractionPreimage
    extraction_digest: str
    margin: ContrastiveMargin | None
    margin_digest: str | None
    evidence: PrototypeTruthEvidence
    evidence_digest: str

    def __post_init__(self) -> None:
        for field in (
            "pre_query_freeze_digest",
            "panel_digest",
            "extraction_digest",
            "evidence_digest",
        ):
            _digest(getattr(self, field), field)
        if self.extraction.panel_digest != self.panel_digest:
            raise PrototypeArtifactTamperError("support replay panel digest drift")
        if self.extraction.digest() != self.extraction_digest:
            raise PrototypeArtifactTamperError("support extraction digest drift")
        if self.evidence.digest() != self.evidence_digest:
            raise PrototypeArtifactTamperError("support evidence digest drift")
        if self.margin is None:
            if self.margin_digest is not None:
                raise PrototypeArtifactTamperError(
                    "support replay without margin carries a margin digest"
                )
        elif self.margin_digest != self.margin.digest():
            raise PrototypeArtifactTamperError("support margin digest drift")

    @classmethod
    def capture(
        cls,
        *,
        freeze: PrototypePreQueryFreeze,
        extraction: FeatureExtractionPreimage,
    ) -> "PrototypeSupportReplayArtifact":
        freeze.verify()
        extraction.verify(freeze.feature_space)
        if extraction.feature_group_id != freeze.selected_feature_group_id:
            raise PrototypeArtifactTamperError(
                "support replay receipt names another feature group"
            )
        if (
            extraction.receipt.to_data().get("catalog_digest")
            != freeze.policy.feature_catalog_digest
        ):
            raise PrototypeArtifactTamperError(
                "support replay receipt names another feature catalog"
            )
        if extraction.panel_digest not in freeze.support_panel_digests:
            raise PrototypeArtifactTamperError(
                "support replay panel is not in the frozen support"
            )
        upstream = extraction.extraction_evidence()
        if upstream.disposition is Disposition.PRESENT:
            packet = upstream.unwrap()
            members = {
                (item.panel_digest, item.vector_digest)
                for item in (
                    freeze.prototypes.positive_members
                    + freeze.prototypes.negative_members
                )
            }
            if (packet.panel_digest, packet.digest()) not in members:
                raise PrototypeArtifactTamperError(
                    "fresh support vector differs from the frozen fitted member"
                )
        evidence = evaluate_frozen_support_member(
            freeze.positive_formula,
            freeze.prototypes,
            freeze.feature_space,
            upstream,
        )
        margin = None
        if upstream.disposition is Disposition.PRESENT:
            if evidence.uncertainty is None:
                raise PrototypeArtifactTamperError(
                    "support predicate evidence lacks its contrastive margin"
                )
            margin = ContrastiveMargin(
                upstream.unwrap().digest(),
                freeze.prototypes.digest(),
                evidence.uncertainty.lower,
                evidence.uncertainty.upper,
            )
        result = PrototypeTruthEvidence.from_evidence(evidence)
        return cls(
            pre_query_freeze_digest=freeze.digest(),
            panel_digest=extraction.panel_digest,
            extraction=extraction,
            extraction_digest=extraction.digest(),
            margin=margin,
            margin_digest=margin.digest() if margin is not None else None,
            evidence=result,
            evidence_digest=result.digest(),
        )

    def verify(self, freeze: PrototypePreQueryFreeze) -> None:
        if self.pre_query_freeze_digest != freeze.digest():
            raise PrototypeArtifactTamperError(
                "support replay names another pre-query freeze"
            )
        rebuilt = self.capture(freeze=freeze, extraction=self.extraction)
        if rebuilt != self:
            raise PrototypeArtifactTamperError(
                "support replay differs from fresh model-free evaluation"
            )

    def to_data(self) -> dict[str, object]:
        # There is deliberately no positive/negative side field here.  The
        # outer verifier joins this panel digest to its hidden commitment.
        return {
            "schema": SUPPORT_REPLAY_SCHEMA,
            "pre_query_freeze_digest": self.pre_query_freeze_digest,
            "panel_digest": self.panel_digest,
            "extraction": self.extraction.to_data(),
            "extraction_digest": self.extraction_digest,
            "margin": self.margin.to_data() if self.margin is not None else None,
            "margin_digest": self.margin_digest,
            "evidence": self.evidence.to_data(),
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        *,
        freeze: PrototypePreQueryFreeze | None = None,
    ) -> "PrototypeSupportReplayArtifact":
        _require_fields(
            data,
            {
                "schema",
                "pre_query_freeze_digest",
                "panel_digest",
                "extraction",
                "extraction_digest",
                "margin",
                "margin_digest",
                "evidence",
                "evidence_digest",
            },
            "prototype support replay",
        )
        if data["schema"] != SUPPORT_REPLAY_SCHEMA:
            raise ValueError("unsupported prototype-support replay schema")
        margin_raw = data["margin"]
        result = cls(
            pre_query_freeze_digest=data["pre_query_freeze_digest"],
            panel_digest=data["panel_digest"],
            extraction=FeatureExtractionPreimage.from_data(
                _mapping(data["extraction"], "support extraction")
            ),
            extraction_digest=data["extraction_digest"],
            margin=(
                None
                if margin_raw is None
                else ContrastiveMargin.from_data(
                    _mapping(margin_raw, "support margin")
                )
            ),
            margin_digest=data["margin_digest"],
            evidence=PrototypeTruthEvidence.from_data(
                _mapping(data["evidence"], "support evidence")
            ),
            evidence_digest=data["evidence_digest"],
        )
        if freeze is not None:
            result.verify(freeze)
        return result

    def digest(self) -> str:
        return canonical_digest(self.to_data())


__all__ = [
    "AllowedPrototypeFeatureGroup",
    "FeatureExtractionPreimage",
    "PrototypeArtifactError",
    "PrototypeArtifactTamperError",
    "PrototypeCompilerInputs",
    "PrototypeFreezePolicy",
    "PrototypePreQueryFreeze",
    "PrototypeQueryArtifact",
    "PrototypeSupportReplayArtifact",
    "PrototypeSupportGateEvidence",
    "PrototypeTruthEvidence",
    "canonical_digest",
    "canonical_json",
]
