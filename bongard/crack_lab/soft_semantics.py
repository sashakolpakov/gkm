"""Deterministic, content-addressed soft-semantic evidence.

This module provides a small trusted substrate for fuzzy semantic scores.  It
does not contain an open-world concept recognizer: concepts such as
``bird-like`` remain inadmissible until an explicit, replayable prototype and
matching legs are promoted into the semantic registry.

Soft results are deliberately typed.  Absence and evaluation failure never
masquerade as numeric membership values, so a downstream threshold selector
cannot accidentally learn from a sentinel.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable


SOFT_SEMANTICS_SCHEMA = "bongard.soft-semantics/v1"
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9._/-]*\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unique finite UTF-8 JSON representation of ``value``."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical identifier")


def _require_digest(value: str, field_name: str, *, optional: bool = False) -> None:
    if optional and value == "":
        return
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a sha256 content digest")


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite real number")
    return result


def _require_provenance(values: tuple[str, ...], field_name: str) -> None:
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{field_name} entries must be nonempty text")


@dataclass(frozen=True)
class PrototypeRoleSpec:
    """One typed role required or optionally supported by a prototype."""

    name: str
    witness_type: str
    required: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.name, "prototype role name")
        if not isinstance(self.witness_type, str) \
                or not self.witness_type.endswith("Witness"):
            raise ValueError("prototype role witness_type must name a Witness")
        if not isinstance(self.required, bool):
            raise ValueError("prototype role required must be boolean")


@dataclass(frozen=True)
class PrototypeRelationSpec:
    """A typed relation over named prototype roles."""

    name: str
    roles: tuple[str, ...]
    witness_type: str
    required: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.name, "prototype relation name")
        if not self.roles or len(self.roles) != len(set(self.roles)):
            raise ValueError("prototype relation roles must be unique and nonempty")
        for role in self.roles:
            _require_identifier(role, "prototype relation role")
        if not isinstance(self.witness_type, str) \
                or not self.witness_type.endswith("Witness"):
            raise ValueError("prototype relation witness_type must name a Witness")
        if not isinstance(self.required, bool):
            raise ValueError("prototype relation required must be boolean")


@dataclass(frozen=True)
class PrototypeSpec:
    """Replayable structural prototype; it is data, never an executable leg."""

    prototype_id: str
    roles: tuple[PrototypeRoleSpec, ...]
    relations: tuple[PrototypeRelationSpec, ...]
    source_manifest_digest: str
    version: str = SOFT_SEMANTICS_SCHEMA

    def __post_init__(self) -> None:
        _require_identifier(self.prototype_id, "prototype_id")
        if self.version != SOFT_SEMANTICS_SCHEMA:
            raise ValueError("unsupported soft-semantics prototype version")
        if not self.roles:
            raise ValueError("prototype must declare at least one role")
        role_names = tuple(role.name for role in self.roles)
        if len(role_names) != len(set(role_names)):
            raise ValueError("prototype role names must be unique")
        if not any(role.required for role in self.roles):
            raise ValueError("prototype must have at least one required role")
        known_roles = set(role_names)
        relation_names = tuple(relation.name for relation in self.relations)
        if len(relation_names) != len(set(relation_names)):
            raise ValueError("prototype relation names must be unique")
        for relation in self.relations:
            unknown = set(relation.roles) - known_roles
            if unknown:
                raise ValueError(
                    "prototype relation references unknown roles: "
                    + ", ".join(sorted(unknown)))
        _require_digest(self.source_manifest_digest, "source_manifest_digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return content_digest(self.to_dict())


class SoftResult:
    """Runtime base type for present, absent, or failed soft evidence."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def digest(self) -> str:
        return content_digest(self.to_dict())


@dataclass(frozen=True)
class SoftEvidence(SoftResult):
    """Present fuzzy membership with explicit producer provenance."""

    concept_id: str
    membership: float
    producer_digest: str
    raw_value: float | None = None
    components: tuple[tuple[str, float], ...] = ()
    prototype_digest: str = ""
    input_digests: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.concept_id, "soft evidence concept_id")
        membership = _finite_number(self.membership, "membership")
        if not 0.0 <= membership <= 1.0:
            raise ValueError("membership must be in [0, 1]")
        object.__setattr__(self, "membership", membership)
        _require_digest(self.producer_digest, "producer_digest")
        _require_digest(
            self.prototype_digest, "prototype_digest", optional=True)
        for digest in self.input_digests:
            _require_digest(digest, "soft evidence input digest")
        if self.raw_value is not None:
            object.__setattr__(
                self, "raw_value", _finite_number(self.raw_value, "raw_value"))
        names: list[str] = []
        normalized: list[tuple[str, float]] = []
        for name, value in self.components:
            _require_identifier(name, "soft evidence component name")
            numeric = _finite_number(value, f"component {name}")
            if not 0.0 <= numeric <= 1.0:
                raise ValueError("soft evidence components must be in [0, 1]")
            names.append(name)
            normalized.append((name, numeric))
        if len(names) != len(set(names)):
            raise ValueError("soft evidence component names must be unique")
        object.__setattr__(self, "components", tuple(normalized))
        _require_provenance(self.provenance, "soft evidence provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"state": "present", **asdict(self)}


@dataclass(frozen=True)
class SoftAbsent(SoftResult):
    """The requested semantic carrier is honestly absent."""

    concept_id: str
    reason_code: str
    detail: str = ""
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.concept_id, "soft absence concept_id")
        _require_identifier(self.reason_code, "soft absence reason_code")
        if not isinstance(self.detail, str):
            raise ValueError("soft absence detail must be text")
        _require_provenance(self.provenance, "soft absence provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"state": "absent", **asdict(self)}


@dataclass(frozen=True)
class SoftError(SoftResult):
    """Evaluation failed; unlike absence, this is an implementation error."""

    concept_id: str
    error_code: str
    detail: str = ""
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.concept_id, "soft error concept_id")
        _require_identifier(self.error_code, "soft error error_code")
        if not isinstance(self.detail, str):
            raise ValueError("soft error detail must be text")
        _require_provenance(self.provenance, "soft error provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"state": "error", **asdict(self)}


@dataclass(frozen=True)
class CalibratorContract:
    """Frozen monotone conversion from one raw metric to fuzzy membership.

    The source-manifest digest identifies the independent analytic or
    calibration evidence used to choose the bounds.  Test-panel labels are
    therefore not accepted as inline calibration parameters.
    """

    calibrator_id: str
    metric_id: str
    raw_low: float
    raw_high: float
    direction: str
    score_semantics: str
    source_manifest_digest: str
    fixed_cutoff: float | None = None
    version: str = SOFT_SEMANTICS_SCHEMA

    def __post_init__(self) -> None:
        _require_identifier(self.calibrator_id, "calibrator_id")
        _require_identifier(self.metric_id, "metric_id")
        if self.version != SOFT_SEMANTICS_SCHEMA:
            raise ValueError("unsupported soft-semantics calibrator version")
        low = _finite_number(self.raw_low, "raw_low")
        high = _finite_number(self.raw_high, "raw_high")
        if not high > low:
            raise ValueError("raw_high must be greater than raw_low")
        object.__setattr__(self, "raw_low", low)
        object.__setattr__(self, "raw_high", high)
        if self.direction not in {"high", "low"}:
            raise ValueError("calibrator direction must be high or low")
        if self.score_semantics not in {"membership", "similarity"}:
            raise ValueError(
                "score_semantics must be membership or similarity")
        _require_digest(self.source_manifest_digest, "source_manifest_digest")
        if self.fixed_cutoff is not None:
            cutoff = _finite_number(self.fixed_cutoff, "fixed_cutoff")
            if not 0.0 <= cutoff <= 1.0:
                raise ValueError("fixed_cutoff must be in [0, 1]")
            object.__setattr__(self, "fixed_cutoff", cutoff)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return content_digest(self.to_dict())

    def apply(
            self, raw_value: Any, concept_id: str, *,
            prototype_digest: str = "", provenance: tuple[str, ...] = (),
            components: tuple[tuple[str, float], ...] = ()) -> SoftResult:
        """Return evidence or a typed error; never a numeric sentinel."""
        try:
            raw = _finite_number(raw_value, "raw_value")
            _require_identifier(concept_id, "concept_id")
            _require_digest(
                prototype_digest, "prototype_digest", optional=True)
            _require_provenance(provenance, "calibrator provenance")
            scaled = (raw - self.raw_low) / (self.raw_high - self.raw_low)
            membership = min(1.0, max(0.0, scaled))
            if self.direction == "low":
                membership = 1.0 - membership
            return SoftEvidence(
                concept_id=concept_id,
                membership=membership,
                producer_digest=self.digest(),
                raw_value=raw,
                components=components,
                prototype_digest=prototype_digest,
                provenance=provenance + (self.calibrator_id,),
            )
        except ValueError as exc:
            safe_concept = concept_id if isinstance(concept_id, str) \
                and _IDENTIFIER_RE.fullmatch(concept_id) else "invalid-concept"
            safe_provenance = provenance if isinstance(provenance, tuple) \
                and all(isinstance(item, str) and item for item in provenance) \
                else ()
            return SoftError(
                safe_concept, "invalid-raw-measurement", str(exc),
                safe_provenance)


def _operator_digest(operator: str) -> str:
    return content_digest({
        "schema": SOFT_SEMANTICS_SCHEMA,
        "operator": operator,
    })


def _first_nonpresent(values: Iterable[SoftResult]) -> SoftResult | None:
    values = tuple(values)
    error = next((value for value in values if isinstance(value, SoftError)), None)
    if error is not None:
        return error
    return next((value for value in values if isinstance(value, SoftAbsent)), None)


def _combine(operator: str, values: tuple[SoftResult, ...], membership: float
             ) -> SoftResult:
    nonpresent = _first_nonpresent(values)
    if nonpresent is not None:
        return nonpresent
    evidence = tuple(value for value in values if isinstance(value, SoftEvidence))
    if len(evidence) != len(values):
        return SoftError(
            "soft-composition", "invalid-soft-result",
            "composition received an unknown soft-result subtype")
    concept = f"{operator}-" + "-".join(value.concept_id for value in evidence)
    return SoftEvidence(
        concept_id=concept,
        membership=membership,
        producer_digest=_operator_digest(operator),
        components=tuple(
            (f"operand-{index}", value.membership)
            for index, value in enumerate(evidence)),
        input_digests=tuple(value.digest() for value in evidence),
        provenance=tuple(
            item for value in evidence for item in value.provenance)
            + (f"soft-{operator}",),
    )


def fuzzy_min(left: SoftResult, right: SoftResult) -> SoftResult:
    nonpresent = _first_nonpresent((left, right))
    value = 0.0 if nonpresent is not None else min(
        left.membership, right.membership)  # type: ignore[attr-defined]
    return _combine("min", (left, right), value)


def fuzzy_max(left: SoftResult, right: SoftResult) -> SoftResult:
    nonpresent = _first_nonpresent((left, right))
    value = 0.0 if nonpresent is not None else max(
        left.membership, right.membership)  # type: ignore[attr-defined]
    return _combine("max", (left, right), value)


def fuzzy_not(value: SoftResult) -> SoftResult:
    nonpresent = _first_nonpresent((value,))
    membership = 0.0 if nonpresent is not None else \
        1.0 - value.membership  # type: ignore[attr-defined]
    return _combine("not", (value,), membership)


@dataclass(frozen=True)
class SoftEvidenceSet:
    """Immutable carrier for explicit fuzzy quantification."""

    values: tuple[SoftResult, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(value, SoftResult) for value in self.values):
            raise ValueError("SoftEvidenceSet accepts only typed soft results")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOFT_SEMANTICS_SCHEMA,
            "values": [value.to_dict() for value in self.values],
        }

    def digest(self) -> str:
        return content_digest(self.to_dict())


def soft_pair(left: SoftResult, right: SoftResult) -> SoftEvidenceSet:
    return SoftEvidenceSet((left, right))


def soft_add(values: SoftEvidenceSet, value: SoftResult) -> SoftEvidenceSet:
    return SoftEvidenceSet(values.values + (value,))


def _quantify(operator: str, values: SoftEvidenceSet) -> SoftResult:
    if not values.values:
        return SoftAbsent(
            f"soft-{operator}", "empty-carrier",
            "fuzzy quantification requires a nonempty carrier")
    nonpresent = _first_nonpresent(values.values)
    if nonpresent is not None:
        return nonpresent
    memberships = tuple(
        value.membership for value in values.values
        if isinstance(value, SoftEvidence))
    if operator == "all":
        result = min(memberships)
    elif operator == "any":
        result = max(memberships)
    elif operator == "mean":
        result = math.fsum(memberships) / len(memberships)
    else:  # pragma: no cover - callers are closed over the three operators.
        raise AssertionError(operator)
    return _combine(operator, values.values, result)


def fuzzy_all(values: SoftEvidenceSet) -> SoftResult:
    return _quantify("all", values)


def fuzzy_any(values: SoftEvidenceSet) -> SoftResult:
    return _quantify("any", values)


def fuzzy_mean(values: SoftEvidenceSet) -> SoftResult:
    return _quantify("mean", values)


# Analytic geometry, not a fit to Bongard labels: 45 degrees is the maximum
# possible distance from the cardinal set {0, 90, 180} on an unsigned angle.
_OBLIQUENESS_DEFINITION_DIGEST = content_digest({
    "schema": SOFT_SEMANTICS_SCHEMA,
    "definition": "distance-degrees-from-nearest-of-0-90-180-divided-by-45",
})
OBLIQUENESS_CALIBRATOR = CalibratorContract(
    calibrator_id="analytic-angle-obliqueness-v1",
    metric_id="angle-noncardinality-degrees",
    raw_low=0.0,
    raw_high=45.0,
    direction="high",
    score_semantics="membership",
    source_manifest_digest=_OBLIQUENESS_DEFINITION_DIGEST,
)


__all__ = [
    "SOFT_SEMANTICS_SCHEMA",
    "CalibratorContract",
    "OBLIQUENESS_CALIBRATOR",
    "PrototypeRelationSpec",
    "PrototypeRoleSpec",
    "PrototypeSpec",
    "SoftAbsent",
    "SoftError",
    "SoftEvidence",
    "SoftEvidenceSet",
    "SoftResult",
    "canonical_json_bytes",
    "content_digest",
    "fuzzy_all",
    "fuzzy_any",
    "fuzzy_max",
    "fuzzy_mean",
    "fuzzy_min",
    "fuzzy_not",
    "soft_add",
    "soft_pair",
]
