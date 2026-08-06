"""Candidate-independent support prototypes for Bongard predicates.

This module supplies the missing *task-relative* scoring layer between frozen
panel features and the closed predicate IR.  A feature packet is extracted
from one panel without a task identifier, side label, prose claim, formula, or
query role.  Only after those packets are frozen does this module fit separate
positive and negative support centroids.  Query packets are scored against the
frozen artifact by a single, fixed orientation::

    margin = distance(query, negative) - distance(query, positive)

Thus a larger margin always means "more like the positive support".  There is
no polarity parameter and no operation that swaps the two sides after seeing
query outcomes.

The serialized records, not this Python implementation, are the scientific
contract.  The reference algorithm uses interval-valued, weighted normalized
L1 distance so every query result is an enclosure and can be replayed by
another backend.  This module does not extract pixels, authenticate extractor
receipts, prove that extraction was candidate-independent, or calibrate a
geometric margin into semantic truth.  Those are explicit outer-runner duties.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from bongard.evidence import Disposition, Evidence, Provenance, Uncertainty
from bongard.legs import (
    BOOLEAN_WITNESS,
    InvarianceContract,
    LegContract,
    LegReference,
    LegRegistry,
    LegSemantics,
    ValueType,
)


FEATURE_SPACE_SCHEMA = "bongard.support_prototype.feature_space.v1"
FEATURE_VECTOR_SCHEMA = "bongard.support_prototype.feature_vector.v1"
FIT_PLAN_SCHEMA = "bongard.support_prototype.fit_plan.v1"
SUPPORT_ASSIGNMENT_SCHEMA = "bongard.support_prototype.panel_side_assignment.v1"
PROTOTYPE_SCHEMA = "bongard.support_prototype.artifact.v1"
FORMULA_SCHEMA = "bongard.support_prototype.formula.v1"
MARGIN_SCHEMA = "bongard.support_prototype.margin.v1"
ALGORITHM_ID = "interval_weighted_normalized_l1_centroids_v1"
INPUT_CONTRACT = "panel_bytes_only_no_task_candidate_side_or_role_context_v1"
ORIENTATION = "negative_distance_minus_positive_distance"
SUPPORT_PROTOTYPE_FEATURES = ValueType("support_prototype_features")


class SupportPrototypeError(ValueError):
    """Base class for malformed or inadmissible prototype records."""


class SupportPrototypeIntegrityError(SupportPrototypeError):
    """A runtime record differs from a frozen content identity."""


def _canonical_digest(data: object) -> str:
    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_fields(
    data: Mapping[str, Any], fields: frozenset[str], label: str
) -> None:
    if not isinstance(data, Mapping) or set(data) != fields:
        raise ValueError(f"{label} fields differ from schema")


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty exact string")
    return value


def _require_name(name: str, value: object) -> str:
    text = _require_text(name, value)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", text):
        raise ValueError(f"{name} must be lower snake case")
    return text


def _require_digest(name: str, value: object) -> str:
    text = _require_text(name, value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{name} must be a lowercase sha256")
    return text


def _require_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _sequence(name: str, value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list")
    return value


def panel_side_assignment_digest(
    positive_panel_digests: Sequence[str],
    negative_panel_digests: Sequence[str],
) -> str:
    """Commit the exact panel-only support orientation.

    The preimage intentionally has no dataset or task identifier.  Sorting
    makes transport order irrelevant, while the two named arrays make swapping
    positive and negative support a different commitment.
    """

    positive = tuple(sorted(positive_panel_digests))
    negative = tuple(sorted(negative_panel_digests))
    if not positive or not negative:
        raise ValueError("panel-side assignment requires both support sides")
    for item in positive + negative:
        _require_digest("support panel digest", item)
    if len(positive) != len(set(positive)) or len(negative) != len(set(negative)):
        raise ValueError("panel-side assignment contains a duplicate panel")
    if set(positive) & set(negative):
        raise ValueError("a panel cannot occur on both support sides")
    return _canonical_digest(
        {
            "schema": SUPPORT_ASSIGNMENT_SCHEMA,
            "positive_panel_digests": list(positive),
            "negative_panel_digests": list(negative),
        }
    )


@dataclass(frozen=True, order=True)
class FeatureDimension:
    """One preregistered coordinate in a candidate-independent feature space."""

    name: str
    unit: str
    lower_bound: float
    upper_bound: float
    scale: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        _require_name("feature name", self.name)
        _require_text("feature unit", self.unit)
        lower = _require_real("feature lower_bound", self.lower_bound)
        upper = _require_real("feature upper_bound", self.upper_bound)
        scale = _require_real("feature scale", self.scale)
        weight = _require_real("feature weight", self.weight)
        if lower >= upper:
            raise ValueError("feature lower_bound must be below upper_bound")
        if scale <= 0.0 or weight <= 0.0:
            raise ValueError("feature scale and weight must be positive")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "weight", weight)

    def to_data(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "scale": self.scale,
            "weight": self.weight,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "FeatureDimension":
        fields = frozenset(
            {"name", "unit", "lower_bound", "upper_bound", "scale", "weight"}
        )
        _require_fields(data, fields, "feature dimension")
        return cls(
            name=data["name"],
            unit=data["unit"],
            lower_bound=data["lower_bound"],
            upper_bound=data["upper_bound"],
            scale=data["scale"],
            weight=data["weight"],
        )


@dataclass(frozen=True, order=True)
class FeatureInterval:
    """A closed enclosure for one named feature coordinate."""

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        _require_name("feature interval name", self.name)
        lower = _require_real("feature interval lower", self.lower)
        upper = _require_real("feature interval upper", self.upper)
        if lower > upper:
            raise ValueError("feature interval lower exceeds upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_data(self) -> dict[str, object]:
        return {"name": self.name, "lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "FeatureInterval":
        _require_fields(
            data, frozenset({"name", "lower", "upper"}), "feature interval"
        )
        return cls(data["name"], data["lower"], data["upper"])


@dataclass(frozen=True)
class FrozenFeatureSpace:
    """Content-addressed extractor and coordinate contract.

    The fixed input contract deliberately excludes all task-relative context.
    An outer verifier must freeze this record before candidate generation and
    authenticate that the declared extractor actually honored it.
    """

    extractor_id: str
    extractor_version: str
    extractor_artifact_digest: str
    preprocessing_digest: str
    receipt_protocol_digest: str
    dimensions: tuple[FeatureDimension, ...]

    def __post_init__(self) -> None:
        _require_text("extractor_id", self.extractor_id)
        _require_text("extractor_version", self.extractor_version)
        _require_digest("extractor_artifact_digest", self.extractor_artifact_digest)
        _require_digest("preprocessing_digest", self.preprocessing_digest)
        _require_digest("receipt_protocol_digest", self.receipt_protocol_digest)
        if not isinstance(self.dimensions, tuple) or not self.dimensions:
            raise ValueError("feature dimensions must be a non-empty immutable tuple")
        if any(not isinstance(item, FeatureDimension) for item in self.dimensions):
            raise TypeError("feature space contains a malformed dimension")
        names = [item.name for item in self.dimensions]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("feature dimensions must be unique and name-sorted")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_SPACE_SCHEMA,
            "input_contract": INPUT_CONTRACT,
            "extractor_id": self.extractor_id,
            "extractor_version": self.extractor_version,
            "extractor_artifact_digest": self.extractor_artifact_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "receipt_protocol_digest": self.receipt_protocol_digest,
            "dimensions": [item.to_data() for item in self.dimensions],
        }

    def digest(self) -> str:
        return _canonical_digest(self.to_data())

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "FrozenFeatureSpace":
        fields = frozenset(
            {
                "schema",
                "input_contract",
                "extractor_id",
                "extractor_version",
                "extractor_artifact_digest",
                "preprocessing_digest",
                "receipt_protocol_digest",
                "dimensions",
            }
        )
        _require_fields(data, fields, "feature space")
        if data["schema"] != FEATURE_SPACE_SCHEMA:
            raise ValueError("unsupported feature-space schema")
        if data["input_contract"] != INPUT_CONTRACT:
            raise ValueError("feature space admits task-relative extractor input")
        dimensions = _sequence("dimensions", data["dimensions"])
        return cls(
            extractor_id=data["extractor_id"],
            extractor_version=data["extractor_version"],
            extractor_artifact_digest=data["extractor_artifact_digest"],
            preprocessing_digest=data["preprocessing_digest"],
            receipt_protocol_digest=data["receipt_protocol_digest"],
            dimensions=tuple(
                FeatureDimension.from_data(item) for item in dimensions
            ),
        )


@dataclass(frozen=True)
class FrozenPanelFeatures:
    """One panel-only feature packet admitted by an outer verifier."""

    panel_digest: str
    feature_space_digest: str
    extractor_receipt_digest: str
    values: tuple[FeatureInterval, ...]

    def __post_init__(self) -> None:
        _require_digest("panel_digest", self.panel_digest)
        _require_digest("feature_space_digest", self.feature_space_digest)
        _require_digest("extractor_receipt_digest", self.extractor_receipt_digest)
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("feature values must be a non-empty immutable tuple")
        if any(not isinstance(item, FeatureInterval) for item in self.values):
            raise TypeError("feature packet contains a malformed interval")
        names = [item.name for item in self.values]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("feature values must be unique and name-sorted")

    def validate(self, space: FrozenFeatureSpace) -> None:
        if self.feature_space_digest != space.digest():
            raise SupportPrototypeIntegrityError(
                "feature packet belongs to another feature space"
            )
        dimensions = {item.name: item for item in space.dimensions}
        if tuple(item.name for item in self.values) != tuple(dimensions):
            raise SupportPrototypeIntegrityError(
                "feature packet coordinates differ from feature space"
            )
        for value in self.values:
            dimension = dimensions[value.name]
            if (
                value.lower < dimension.lower_bound
                or value.upper > dimension.upper_bound
            ):
                raise SupportPrototypeIntegrityError(
                    f"feature {value.name} lies outside its frozen bounds"
                )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_VECTOR_SCHEMA,
            "panel_digest": self.panel_digest,
            "feature_space_digest": self.feature_space_digest,
            "extractor_receipt_digest": self.extractor_receipt_digest,
            "values": [item.to_data() for item in self.values],
        }

    def digest(self) -> str:
        return _canonical_digest(self.to_data())

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "FrozenPanelFeatures":
        fields = frozenset(
            {
                "schema",
                "panel_digest",
                "feature_space_digest",
                "extractor_receipt_digest",
                "values",
            }
        )
        _require_fields(data, fields, "feature packet")
        if data["schema"] != FEATURE_VECTOR_SCHEMA:
            raise ValueError("unsupported feature-vector schema")
        values = _sequence("values", data["values"])
        return cls(
            panel_digest=data["panel_digest"],
            feature_space_digest=data["feature_space_digest"],
            extractor_receipt_digest=data["extractor_receipt_digest"],
            values=tuple(FeatureInterval.from_data(item) for item in values),
        )


@dataclass(frozen=True)
class SupportPrototypePlan:
    """Verifier-frozen fit choices and panel-only support-side commitment."""

    feature_space_digest: str
    support_assignment_digest: str
    minimum_per_side: int = 2

    def __post_init__(self) -> None:
        _require_digest("feature_space_digest", self.feature_space_digest)
        _require_digest("support_assignment_digest", self.support_assignment_digest)
        if (
            isinstance(self.minimum_per_side, bool)
            or not isinstance(self.minimum_per_side, int)
            or self.minimum_per_side < 1
        ):
            raise ValueError("minimum_per_side must be a positive integer")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FIT_PLAN_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "orientation": ORIENTATION,
            "feature_space_digest": self.feature_space_digest,
            "support_assignment_digest": self.support_assignment_digest,
            "minimum_per_side": self.minimum_per_side,
        }

    def digest(self) -> str:
        return _canonical_digest(self.to_data())

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SupportPrototypePlan":
        fields = frozenset(
            {
                "schema",
                "algorithm_id",
                "orientation",
                "feature_space_digest",
                "support_assignment_digest",
                "minimum_per_side",
            }
        )
        _require_fields(data, fields, "prototype fit plan")
        if data["schema"] != FIT_PLAN_SCHEMA or data["algorithm_id"] != ALGORITHM_ID:
            raise ValueError("unsupported prototype fit plan")
        if data["orientation"] != ORIENTATION:
            raise ValueError("prototype fit plan attempts a polarity change")
        return cls(
            feature_space_digest=data["feature_space_digest"],
            support_assignment_digest=data["support_assignment_digest"],
            minimum_per_side=data["minimum_per_side"],
        )


@dataclass(frozen=True, order=True)
class SupportMember:
    """Panel/vector identity pair; deliberately contains no dataset identity."""

    panel_digest: str
    vector_digest: str

    def __post_init__(self) -> None:
        _require_digest("support panel_digest", self.panel_digest)
        _require_digest("support vector_digest", self.vector_digest)

    def to_data(self) -> dict[str, str]:
        return {
            "panel_digest": self.panel_digest,
            "vector_digest": self.vector_digest,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "SupportMember":
        _require_fields(
            data, frozenset({"panel_digest", "vector_digest"}), "support member"
        )
        return cls(data["panel_digest"], data["vector_digest"])


@dataclass(frozen=True)
class FrozenSupportPrototypes:
    """Support-derived centroids frozen before any query feature is scored."""

    plan_digest: str
    feature_space_digest: str
    support_assignment_digest: str
    positive_members: tuple[SupportMember, ...]
    negative_members: tuple[SupportMember, ...]
    positive_centroid: tuple[FeatureInterval, ...]
    negative_centroid: tuple[FeatureInterval, ...]

    def __post_init__(self) -> None:
        for name in (
            "plan_digest",
            "feature_space_digest",
            "support_assignment_digest",
        ):
            _require_digest(name, getattr(self, name))
        if not self.positive_members or not self.negative_members:
            raise ValueError("prototype artifact requires both support sides")
        for label, members in (
            ("positive", self.positive_members),
            ("negative", self.negative_members),
        ):
            if not isinstance(members, tuple) or any(
                not isinstance(item, SupportMember) for item in members
            ):
                raise TypeError(f"{label} members must be an immutable typed tuple")
            if list(members) != sorted(members) or len(members) != len(set(members)):
                raise ValueError(f"{label} members must be unique and sorted")
        positive_panels = {item.panel_digest for item in self.positive_members}
        negative_panels = {item.panel_digest for item in self.negative_members}
        if len(positive_panels) != len(self.positive_members) or len(
            negative_panels
        ) != len(self.negative_members):
            raise ValueError("a panel cannot occur twice on one support side")
        if positive_panels & negative_panels:
            raise ValueError("a panel cannot occur on both support sides")
        for label, centroid in (
            ("positive", self.positive_centroid),
            ("negative", self.negative_centroid),
        ):
            if not isinstance(centroid, tuple) or not centroid or any(
                not isinstance(item, FeatureInterval) for item in centroid
            ):
                raise TypeError(f"{label} centroid must be an immutable interval tuple")
            names = [item.name for item in centroid]
            if names != sorted(names) or len(names) != len(set(names)):
                raise ValueError(f"{label} centroid coordinates must be sorted")
        if tuple(item.name for item in self.positive_centroid) != tuple(
            item.name for item in self.negative_centroid
        ):
            raise ValueError("positive and negative centroid coordinates differ")

    @property
    def support_panel_digests(self) -> frozenset[str]:
        return frozenset(
            item.panel_digest
            for item in self.positive_members + self.negative_members
        )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": PROTOTYPE_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "orientation": ORIENTATION,
            "plan_digest": self.plan_digest,
            "feature_space_digest": self.feature_space_digest,
            "support_assignment_digest": self.support_assignment_digest,
            "positive_members": [item.to_data() for item in self.positive_members],
            "negative_members": [item.to_data() for item in self.negative_members],
            "positive_centroid": [item.to_data() for item in self.positive_centroid],
            "negative_centroid": [item.to_data() for item in self.negative_centroid],
        }

    def digest(self) -> str:
        return _canonical_digest(self.to_data())

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "FrozenSupportPrototypes":
        fields = frozenset(
            {
                "schema",
                "algorithm_id",
                "orientation",
                "plan_digest",
                "feature_space_digest",
                "support_assignment_digest",
                "positive_members",
                "negative_members",
                "positive_centroid",
                "negative_centroid",
            }
        )
        _require_fields(data, fields, "prototype artifact")
        if data["schema"] != PROTOTYPE_SCHEMA or data["algorithm_id"] != ALGORITHM_ID:
            raise ValueError("unsupported prototype artifact")
        if data["orientation"] != ORIENTATION:
            raise ValueError("prototype artifact attempts a polarity change")

        def members(name: str) -> tuple[SupportMember, ...]:
            return tuple(
                SupportMember.from_data(item) for item in _sequence(name, data[name])
            )

        def centroid(name: str) -> tuple[FeatureInterval, ...]:
            return tuple(
                FeatureInterval.from_data(item)
                for item in _sequence(name, data[name])
            )

        return cls(
            plan_digest=data["plan_digest"],
            feature_space_digest=data["feature_space_digest"],
            support_assignment_digest=data["support_assignment_digest"],
            positive_members=members("positive_members"),
            negative_members=members("negative_members"),
            positive_centroid=centroid("positive_centroid"),
            negative_centroid=centroid("negative_centroid"),
        )


@dataclass(frozen=True)
class PositivePrototypeFormula:
    """An affirmative claim bound to one exact frozen support artifact."""

    claim: str
    feature_space_digest: str
    prototype_digest: str
    support_assignment_digest: str
    decision_margin: float

    def __post_init__(self) -> None:
        _require_text("prototype formula claim", self.claim)
        _require_digest("feature_space_digest", self.feature_space_digest)
        _require_digest("prototype_digest", self.prototype_digest)
        _require_digest("support_assignment_digest", self.support_assignment_digest)
        margin = _require_real("decision_margin", self.decision_margin)
        if margin <= 0.0:
            raise ValueError("decision_margin must be strictly positive")
        object.__setattr__(self, "decision_margin", margin)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FORMULA_SCHEMA,
            "algorithm_id": ALGORITHM_ID,
            "orientation": ORIENTATION,
            "claim": self.claim,
            "feature_space_digest": self.feature_space_digest,
            "prototype_digest": self.prototype_digest,
            "support_assignment_digest": self.support_assignment_digest,
            "decision_margin": self.decision_margin,
        }

    def digest(self) -> str:
        return _canonical_digest(self.to_data())

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "PositivePrototypeFormula":
        fields = frozenset(
            {
                "schema",
                "algorithm_id",
                "orientation",
                "claim",
                "feature_space_digest",
                "prototype_digest",
                "support_assignment_digest",
                "decision_margin",
            }
        )
        _require_fields(data, fields, "prototype formula")
        if data["schema"] != FORMULA_SCHEMA or data["algorithm_id"] != ALGORITHM_ID:
            raise ValueError("unsupported prototype formula")
        if data["orientation"] != ORIENTATION:
            raise ValueError("prototype formula attempts a polarity change")
        return cls(
            claim=data["claim"],
            feature_space_digest=data["feature_space_digest"],
            prototype_digest=data["prototype_digest"],
            support_assignment_digest=data["support_assignment_digest"],
            decision_margin=data["decision_margin"],
        )


@dataclass(frozen=True)
class ContrastiveMargin:
    """Closed interval for the fixed positive-support contrastive margin."""

    query_vector_digest: str
    prototype_digest: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        _require_digest("query_vector_digest", self.query_vector_digest)
        _require_digest("prototype_digest", self.prototype_digest)
        lower = _require_real("contrastive margin lower", self.lower)
        upper = _require_real("contrastive margin upper", self.upper)
        if lower > upper:
            raise ValueError("contrastive margin lower exceeds upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": MARGIN_SCHEMA,
            "orientation": ORIENTATION,
            "query_vector_digest": self.query_vector_digest,
            "prototype_digest": self.prototype_digest,
            "lower": self.lower,
            "upper": self.upper,
        }

    def digest(self) -> str:
        return _canonical_digest(self.to_data())

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ContrastiveMargin":
        _require_fields(
            data,
            frozenset(
                {
                    "schema",
                    "orientation",
                    "query_vector_digest",
                    "prototype_digest",
                    "lower",
                    "upper",
                }
            ),
            "contrastive margin",
        )
        if data["schema"] != MARGIN_SCHEMA:
            raise ValueError("unsupported contrastive-margin schema")
        if data["orientation"] != ORIENTATION:
            raise ValueError("contrastive margin attempts a polarity change")
        return cls(
            query_vector_digest=data["query_vector_digest"],
            prototype_digest=data["prototype_digest"],
            lower=data["lower"],
            upper=data["upper"],
        )


def _centroid(vectors: Sequence[FrozenPanelFeatures]) -> tuple[FeatureInterval, ...]:
    count = len(vectors)
    return tuple(
        FeatureInterval(
            name=first.name,
            lower=math.fsum(vector.values[index].lower for vector in vectors) / count,
            upper=math.fsum(vector.values[index].upper for vector in vectors) / count,
        )
        for index, first in enumerate(vectors[0].values)
    )


def fit_support_prototypes(
    plan: SupportPrototypePlan,
    feature_space: FrozenFeatureSpace,
    positive: Sequence[FrozenPanelFeatures],
    negative: Sequence[FrozenPanelFeatures],
    *,
    expected_plan_digest: str,
) -> FrozenSupportPrototypes:
    """Fit deterministic interval centroids from support-only feature packets."""

    _require_digest("expected_plan_digest", expected_plan_digest)
    if plan.digest() != expected_plan_digest:
        raise SupportPrototypeIntegrityError("fit plan differs from frozen commitment")
    if plan.feature_space_digest != feature_space.digest():
        raise SupportPrototypeIntegrityError("fit plan names another feature space")
    if len(positive) < plan.minimum_per_side or len(negative) < plan.minimum_per_side:
        raise SupportPrototypeError("insufficient feature packets on a support side")
    all_vectors = tuple(positive) + tuple(negative)
    if any(not isinstance(item, FrozenPanelFeatures) for item in all_vectors):
        raise TypeError("support inputs must be FrozenPanelFeatures")
    ordered_positive = tuple(sorted(positive, key=lambda item: item.digest()))
    ordered_negative = tuple(sorted(negative, key=lambda item: item.digest()))
    for item in ordered_positive + ordered_negative:
        item.validate(feature_space)
    panels = [item.panel_digest for item in ordered_positive + ordered_negative]
    if len(panels) != len(set(panels)):
        raise SupportPrototypeError(
            "support panel identities must be unique across both sides"
        )
    assignment_digest = panel_side_assignment_digest(
        tuple(item.panel_digest for item in ordered_positive),
        tuple(item.panel_digest for item in ordered_negative),
    )
    if assignment_digest != plan.support_assignment_digest:
        raise SupportPrototypeIntegrityError(
            "support sides differ from the frozen panel-side assignment"
        )
    return FrozenSupportPrototypes(
        plan_digest=expected_plan_digest,
        feature_space_digest=feature_space.digest(),
        support_assignment_digest=plan.support_assignment_digest,
        positive_members=tuple(
            sorted(SupportMember(item.panel_digest, item.digest()) for item in positive)
        ),
        negative_members=tuple(
            sorted(SupportMember(item.panel_digest, item.digest()) for item in negative)
        ),
        positive_centroid=_centroid(ordered_positive),
        negative_centroid=_centroid(ordered_negative),
    )


def verify_support_prototypes(
    artifact: FrozenSupportPrototypes,
    plan: SupportPrototypePlan,
    feature_space: FrozenFeatureSpace,
    positive: Sequence[FrozenPanelFeatures],
    negative: Sequence[FrozenPanelFeatures],
) -> None:
    """Re-fit an artifact from its committed support preimage."""

    rebuilt = fit_support_prototypes(
        plan,
        feature_space,
        positive,
        negative,
        expected_plan_digest=artifact.plan_digest,
    )
    if rebuilt != artifact:
        raise SupportPrototypeIntegrityError(
            "prototype artifact differs from committed support preimage"
        )


def validate_prototype_formula(
    formula: PositivePrototypeFormula,
    artifact: FrozenSupportPrototypes,
    feature_space: FrozenFeatureSpace,
) -> None:
    """Check every cross-layer identity before any query packet is used."""

    if formula.feature_space_digest != feature_space.digest():
        raise SupportPrototypeIntegrityError("formula names another feature space")
    if artifact.feature_space_digest != feature_space.digest():
        raise SupportPrototypeIntegrityError("prototype names another feature space")
    if formula.prototype_digest != artifact.digest():
        raise SupportPrototypeIntegrityError("formula names another prototype artifact")
    if formula.support_assignment_digest != artifact.support_assignment_digest:
        raise SupportPrototypeIntegrityError("formula/support assignment mismatch")


def _absolute_distance(
    left: FeatureInterval, right: FeatureInterval
) -> tuple[float, float]:
    minimum = max(0.0, left.lower - right.upper, right.lower - left.upper)
    maximum = max(
        abs(left.lower - right.upper), abs(left.upper - right.lower)
    )
    return minimum, maximum


def _distance(
    query: FrozenPanelFeatures,
    centroid: tuple[FeatureInterval, ...],
    feature_space: FrozenFeatureSpace,
) -> tuple[float, float]:
    total_weight = math.fsum(item.weight for item in feature_space.dimensions)
    coordinate_bounds = tuple(
        _absolute_distance(query.values[index], centroid[index])
        for index in range(len(centroid))
    )
    lower = math.fsum(
        bounds[0] * dimension.weight / dimension.scale
        for bounds, dimension in zip(
            coordinate_bounds, feature_space.dimensions, strict=True
        )
    ) / total_weight
    upper = math.fsum(
        bounds[1] * dimension.weight / dimension.scale
        for bounds, dimension in zip(
            coordinate_bounds, feature_space.dimensions, strict=True
        )
    ) / total_weight
    return lower, upper


def _contrastive_margin(
    query: FrozenPanelFeatures,
    artifact: FrozenSupportPrototypes,
    feature_space: FrozenFeatureSpace,
    *,
    frozen_support_member: bool,
) -> ContrastiveMargin:
    query.validate(feature_space)
    if artifact.feature_space_digest != feature_space.digest():
        raise SupportPrototypeIntegrityError("prototype names another feature space")
    member = SupportMember(query.panel_digest, query.digest())
    archived_members = frozenset(
        artifact.positive_members + artifact.negative_members
    )
    if frozen_support_member and member not in archived_members:
        raise SupportPrototypeIntegrityError(
            "support replay packet is not an exact frozen support member"
        )
    if not frozen_support_member and query.panel_digest in artifact.support_panel_digests:
        raise SupportPrototypeIntegrityError("query panel overlaps frozen support")
    expected_names = tuple(item.name for item in feature_space.dimensions)
    if tuple(item.name for item in artifact.positive_centroid) != expected_names:
        raise SupportPrototypeIntegrityError("prototype coordinates differ from space")
    positive_lower, positive_upper = _distance(
        query, artifact.positive_centroid, feature_space
    )
    negative_lower, negative_upper = _distance(
        query, artifact.negative_centroid, feature_space
    )
    return ContrastiveMargin(
        query_vector_digest=query.digest(),
        prototype_digest=artifact.digest(),
        lower=negative_lower - positive_upper,
        upper=negative_upper - positive_lower,
    )


def contrastive_margin(
    query: FrozenPanelFeatures,
    artifact: FrozenSupportPrototypes,
    feature_space: FrozenFeatureSpace,
) -> ContrastiveMargin:
    """Compute a held-out panel's fixed positive-support margin enclosure."""

    return _contrastive_margin(
        query,
        artifact,
        feature_space,
        frozen_support_member=False,
    )


def _base_provenance(
    formula: PositivePrototypeFormula,
    artifact: FrozenSupportPrototypes,
    query_digest: str | None = None,
) -> Provenance:
    inputs = [formula.digest(), artifact.digest()]
    if query_digest is not None:
        inputs.append(query_digest)
    return Provenance(
        producer="bongard.support_prototypes",
        version="1",
        method="interval_contrastive_margin",
        input_digests=tuple(inputs),
        artifact_digest=artifact.digest(),
        details=(("orientation", ORIENTATION),),
    )


def _evaluate_support_prototype(
    formula: PositivePrototypeFormula,
    artifact: FrozenSupportPrototypes,
    feature_space: FrozenFeatureSpace,
    query: FrozenPanelFeatures | Evidence[FrozenPanelFeatures],
    *,
    frozen_support_member: bool,
) -> Evidence[bool]:
    try:
        validate_prototype_formula(formula, artifact, feature_space)
    except (TypeError, ValueError) as exc:
        return Evidence.error(
            _base_provenance(formula, artifact),
            type(exc).__name__,
            str(exc) or repr(exc),
        )
    upstream: Provenance | None = None
    if isinstance(query, Evidence):
        upstream = query.provenance
        provenance = Provenance.composed(
            "bongard.support_prototypes",
            "1",
            "upstream_feature_disposition",
            (_base_provenance(formula, artifact), query.provenance),
        )
        if query.disposition is Disposition.CERTIFIED_ABSENT:
            return Evidence.indeterminate(
                provenance,
                "feature extraction absence is not predicate absence",
                query.uncertainty,
            )
        if query.disposition is Disposition.INDETERMINATE:
            return Evidence.indeterminate(
                provenance,
                query.reason or "query features are indeterminate",
                query.uncertainty,
            )
        if query.disposition is Disposition.ERROR:
            return Evidence.error(
                provenance,
                query.error_type or "FeatureExtractionError",
                query.reason or "query feature extraction failed",
            )
        query = query.unwrap()
    if not isinstance(query, FrozenPanelFeatures):
        return Evidence.error(
            _base_provenance(formula, artifact),
            "MalformedFeaturePacket",
            f"expected FrozenPanelFeatures, got {type(query).__name__}",
        )
    try:
        margin = _contrastive_margin(
            query,
            artifact,
            feature_space,
            frozen_support_member=frozen_support_member,
        )
    except (TypeError, ValueError) as exc:
        return Evidence.error(
            _base_provenance(formula, artifact, query.digest()),
            type(exc).__name__,
            str(exc) or repr(exc),
        )
    provenance = _base_provenance(formula, artifact, query.digest())
    if upstream is not None:
        provenance = Provenance.composed(
            "bongard.support_prototypes",
            "1",
            "interval_contrastive_margin",
            (provenance, upstream),
            details=(("margin_digest", margin.digest()),),
        )
    uncertainty = Uncertainty(
        margin.lower,
        margin.upper,
        causes=("interval_feature_enclosure",),
    )
    if margin.lower >= formula.decision_margin:
        return Evidence.present(True, provenance, uncertainty)
    if margin.upper <= -formula.decision_margin:
        certificate = "operational-contrastive-nonmatch:" + _canonical_digest(
            {
                "formula_digest": formula.digest(),
                "prototype_digest": artifact.digest(),
                "margin": margin.to_data(),
            }
        )
        return Evidence.certified_absent(provenance, certificate, uncertainty)
    return Evidence.indeterminate(
        provenance,
        "contrastive margin intersects the frozen abstention region",
        uncertainty,
    )


def evaluate_support_prototype(
    formula: PositivePrototypeFormula,
    artifact: FrozenSupportPrototypes,
    feature_space: FrozenFeatureSpace,
    query: FrozenPanelFeatures | Evidence[FrozenPanelFeatures],
) -> Evidence[bool]:
    """Evaluate one held-out packet with all four evidence dispositions.

    Certified absence means only that the exact operational predicate is
    safely below its negative margin.  It is not a proof of prose-level
    semantic absence.  Upstream feature absence is therefore indeterminate,
    never silently converted to a negative prediction.  A panel used for
    fitting is rejected here so it cannot masquerade as held-out evidence.
    """

    return _evaluate_support_prototype(
        formula,
        artifact,
        feature_space,
        query,
        frozen_support_member=False,
    )


def evaluate_frozen_support_member(
    formula: PositivePrototypeFormula,
    artifact: FrozenSupportPrototypes,
    feature_space: FrozenFeatureSpace,
    packet: FrozenPanelFeatures | Evidence[FrozenPanelFeatures],
) -> Evidence[bool]:
    """Replay the operational predicate on one exact fitted support vector.

    This is deliberately separate from :func:`evaluate_support_prototype`.
    It is a training-fit check for the strict support gate, not held-out
    calibration or query evidence.  A fresh extraction must reconstruct an
    exact ``(panel_digest, vector_digest)`` member of the frozen artifact;
    merely reusing the panel digest with changed features is an error.
    """

    return _evaluate_support_prototype(
        formula,
        artifact,
        feature_space,
        packet,
        frozen_support_member=True,
    )


def register_support_prototype_leg(
    registry: LegRegistry,
    formula: PositivePrototypeFormula,
    artifact: FrozenSupportPrototypes,
    feature_space: FrozenFeatureSpace,
) -> LegReference:
    """Attach one frozen support prototype to the ordinary closed IR.

    The callable receives only an already-frozen feature packet (or upstream
    four-disposition feature evidence).  Its source is fixed Python; the
    operational digest separately binds the exact feature space, support
    prototypes, and affirmative formula.  Consequently two task-relative
    prototypes cannot masquerade as the same registered leg.
    """

    if not isinstance(registry, LegRegistry):
        raise TypeError("registry must be a LegRegistry")
    validate_prototype_formula(formula, artifact, feature_space)
    operational_digest = _canonical_digest(
        {
            "schema": "bongard.support_prototype.registered_leg.v1",
            "algorithm": ALGORITHM_ID,
            "feature_space_digest": feature_space.digest(),
            "prototype_digest": artifact.digest(),
            "formula_digest": formula.digest(),
        }
    )

    def support_prototype_match(
        packet: FrozenPanelFeatures | Evidence[FrozenPanelFeatures],
    ) -> Evidence[bool]:
        return evaluate_support_prototype(formula, artifact, feature_space, packet)

    return registry.register(
        LegContract(
            name="support_prototype_match",
            version="prototype-" + formula.digest()[:16],
            domain=(SUPPORT_PROTOTYPE_FEATURES,),
            codomain=BOOLEAN_WITNESS,
            implementation=support_prototype_match,
            invariance=InvarianceContract(),
            semantics=LegSemantics.DERIVED,
            operational_digest=operational_digest,
        )
    )


__all__ = [
    "ALGORITHM_ID",
    "INPUT_CONTRACT",
    "ORIENTATION",
    "SUPPORT_PROTOTYPE_FEATURES",
    "ContrastiveMargin",
    "FeatureDimension",
    "FeatureInterval",
    "FrozenFeatureSpace",
    "FrozenPanelFeatures",
    "FrozenSupportPrototypes",
    "PositivePrototypeFormula",
    "SupportMember",
    "SupportPrototypeError",
    "SupportPrototypeIntegrityError",
    "SupportPrototypePlan",
    "contrastive_margin",
    "evaluate_frozen_support_member",
    "evaluate_support_prototype",
    "fit_support_prototypes",
    "panel_side_assignment_digest",
    "register_support_prototype_leg",
    "validate_prototype_formula",
    "verify_support_prototypes",
]
