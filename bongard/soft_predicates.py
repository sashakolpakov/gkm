"""Calibrate frozen scorer records into predictive-support intervals.

This module begins *after* scoring.  It neither consumes pixels nor verifies a
pixels-to-score execution.  It binds an affirmative prose claim to an exact
scorer artifact and converts an externally admitted ``FROZEN_VISUAL_SCORE``
record into a cluster-calibrated predictive-support interval.  That interval
describes development-population predictive support under the preregistered
sampling assumptions.  It is never a proof that an individual panel truly has
the semantic property.

The integrity checks here are content-addressed data-contract checks.  They do
not provide timestamps, signatures, or proof that a purported verifier really
owned an identifier; an outer benchmark runner must authenticate and publish
the preregistration before query observations exist.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, NoReturn, Sequence

from bongard.evidence import (
    Disposition,
    Evidence,
    Provenance,
    SoftSemanticObservation,
    Uncertainty,
)
from bongard.ir import Atom, Quantity, Relation, StaticLegCall
from bongard.legs import (
    AffirmativeRelation,
    InvarianceContract,
    LegContract,
    LegReference,
    LegRegistry,
    LegSemantics,
    FROZEN_VISUAL_SCORE,
    SOFT_SEMANTIC,
    Unit,
)


_ALGORITHM_ID = "fixed_bin_cluster_hoeffding_monotone_v2"
_ESTIMAND_ID = "equal_weight_cluster_predictive_support_v1"
_CLAIM_SCHEMA = "bongard.soft_predicate_claim.v2"
_DESIGN_SCHEMA = "bongard.soft_predicate_calibration_design.v2"
_DEVELOPMENT_UNIT_SCHEMA = "bongard.soft_predicate_development_unit.v2"
_PLAN_SCHEMA = "bongard.soft_predicate_calibration_plan.v2"
_OBSERVATION_SCHEMA = "bongard.soft_predicate_development_observation.v2"
_ARTIFACT_SCHEMA = "bongard.soft_predicate_calibration.v2"
_SCORE_SCHEMA = "bongard.frozen_visual_score.v2"
_OPERATION_SCHEMA = "bongard.soft_predicate_operation.v2"

# The family scorer is deliberately a separate contract from the historical
# exact-claim calibration above.  A task-local claim is an input to a blind
# scorer record; it is never the identity of the calibration that interprets
# the score.  Keep these schemas versioned independently so old v2 artifacts
# retain their exact meaning.
_SOFT_SCORER_PROTOCOL_SCHEMA = "bongard.soft_scorer_protocol.v1"
_SOFT_FAMILY_SCHEMA = "bongard.soft_scorer_family.v2"
_SOFT_FAMILY_DEVELOPMENT_UNIT_SCHEMA = (
    "bongard.soft_scorer_family.development_unit.v2"
)
_SOFT_FAMILY_DEVELOPMENT_MANIFEST_SCHEMA = (
    "bongard.soft_scorer_family.development_manifest.v2"
)
_SOFT_CUE_JUDGMENT_SCHEMA = "bongard.soft_cue_judgment.v1"
_BLIND_SOFT_SCORE_SCHEMA = "bongard.blind_soft_score.v2"
_SOFT_FAMILY_CALIBRATION_ID = (
    "fixed_bin_cluster_family_raw_simultaneous_hoeffding_v2"
)
_SOFT_FAMILY_IDENTITY_SEMANTICS = (
    "protocol_precedes_development_and_claim_is_runtime_input_v1"
)
_SOFT_ORDINAL_MAP: tuple[tuple[str, float], ...] = (
    ("supported", 1.0),
    ("ambiguous", 0.5),
    ("unsupported", 0.0),
)
_SOFT_AGGREGATION = "min"
_REASONING_EFFORTS = frozenset(
    {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
)
_BLIND_SCORE_OUTCOMES = frozenset(
    {"present", "transport_error", "parser_error"}
)


class SoftPredicateError(ValueError):
    """Base class for rejected soft-predicate data or configuration."""


class CalibrationError(SoftPredicateError):
    """Development observations cannot produce the preregistered artifact."""


class SoftPredicateIntegrityError(SoftPredicateError):
    """A content identity differs from the verifier's frozen identity."""


def _canonical_sha256(data: object) -> str:
    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty exact string")
    return value


def _require_sha256(name: str, value: object) -> str:
    text = _require_nonempty(name, value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase sha256")
    return text


def _require_score(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return result


def _require_fields(
    data: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if not isinstance(data, Mapping) or set(data) != expected:
        missing = sorted(expected - set(data)) if isinstance(data, Mapping) else []
        extra = sorted(set(data) - expected) if isinstance(data, Mapping) else []
        detail = ""
        if missing:
            detail += "; missing " + ", ".join(missing)
        if extra:
            detail += "; unknown " + ", ".join(extra)
        raise ValueError(f"{label} JSON has missing or unknown fields{detail}")


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    return value


def _check_expected_digest(
    label: str, actual: str, expected: str | None
) -> None:
    if expected is None:
        return
    _require_sha256(f"expected {label} digest", expected)
    if actual != expected:
        raise SoftPredicateIntegrityError(f"{label} digest mismatch")


@dataclass(frozen=True)
class SoftPredicateClaim:
    """One content-addressed affirmative prose/scorer specification.

    ``affirmative_cues`` describe evidence *for* the phrase.  Complement cues
    and a polarity field are intentionally absent.  Exact model, prompt, and
    decoder identifiers are part of the claim identity rather than mutable
    runtime parameters.  These identifiers specify an intended scorer call;
    they are not evidence that such a call executed on any panel.
    """

    phrase: str
    affirmative_cues: tuple[str, ...]
    model_id: str
    prompt_id: str
    decoder_id: str

    def __post_init__(self) -> None:
        _require_nonempty("claim phrase", self.phrase)
        if not isinstance(self.affirmative_cues, tuple) or not self.affirmative_cues:
            raise ValueError("affirmative_cues must be a non-empty immutable tuple")
        for cue in self.affirmative_cues:
            _require_nonempty("affirmative cue", cue)
        if len(self.affirmative_cues) != len(set(self.affirmative_cues)):
            raise ValueError("affirmative cues must be unique")
        _require_nonempty("model_id", self.model_id)
        _require_nonempty("prompt_id", self.prompt_id)
        _require_nonempty("decoder_id", self.decoder_id)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": _CLAIM_SCHEMA,
            "phrase": self.phrase,
            "affirmative_cues": list(self.affirmative_cues),
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "decoder_id": self.decoder_id,
        }

    def digest(self) -> str:
        return _canonical_sha256(self.to_data())


@dataclass(frozen=True)
class CalibrationDesign:
    """Exact annotation, population, sampling, and scorer preregistration.

    Digests refer to archived artifact bytes owned by the outer verifier.  The
    identifiers are human-auditable names; both names and digests are bound.
    This DTO checks identity but does not authenticate the archives.
    """

    annotation_protocol_id: str
    annotation_protocol_digest: str
    annotation_ontology_id: str
    annotation_ontology_digest: str
    target_population_id: str
    population_manifest_digest: str
    sampling_design_id: str
    sampling_design_digest: str
    scorer_artifact_id: str
    scorer_artifact_digest: str
    score_admission_protocol_id: str
    score_admission_protocol_digest: str

    def __post_init__(self) -> None:
        for name in (
            "annotation_protocol_id",
            "annotation_ontology_id",
            "target_population_id",
            "sampling_design_id",
            "scorer_artifact_id",
            "score_admission_protocol_id",
        ):
            _require_nonempty(name, getattr(self, name))
        for name in (
            "annotation_protocol_digest",
            "annotation_ontology_digest",
            "population_manifest_digest",
            "sampling_design_digest",
            "scorer_artifact_digest",
            "score_admission_protocol_digest",
        ):
            _require_sha256(name, getattr(self, name))

    def to_data(self) -> dict[str, str]:
        return {
            "schema": _DESIGN_SCHEMA,
            "annotation_protocol_id": self.annotation_protocol_id,
            "annotation_protocol_digest": self.annotation_protocol_digest,
            "annotation_ontology_id": self.annotation_ontology_id,
            "annotation_ontology_digest": self.annotation_ontology_digest,
            "target_population_id": self.target_population_id,
            "population_manifest_digest": self.population_manifest_digest,
            "sampling_design_id": self.sampling_design_id,
            "sampling_design_digest": self.sampling_design_digest,
            "scorer_artifact_id": self.scorer_artifact_id,
            "scorer_artifact_digest": self.scorer_artifact_digest,
            "score_admission_protocol_id": self.score_admission_protocol_id,
            "score_admission_protocol_digest": self.score_admission_protocol_digest,
        }

    def digest(self) -> str:
        return _canonical_sha256(self.to_data())


@dataclass(frozen=True, order=True)
class DevelopmentUnit:
    """One exact preregistered development panel and dependence cluster."""

    observation_id: str
    task_id: str
    group_id: str
    model_call_id: str
    cluster_id: str
    panel_digest: str

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "task_id",
            "group_id",
            "model_call_id",
            "cluster_id",
        ):
            _require_nonempty(name, getattr(self, name))
        _require_sha256("panel_digest", self.panel_digest)

    def to_data(self) -> dict[str, str]:
        return {
            "schema": _DEVELOPMENT_UNIT_SCHEMA,
            "observation_id": self.observation_id,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "model_call_id": self.model_call_id,
            "cluster_id": self.cluster_id,
            "panel_digest": self.panel_digest,
        }


class ObservationRole(str, Enum):
    """Roles are explicit so query/support data cannot enter calibration."""

    DEVELOPMENT = "development"
    SUPPORT = "support"
    QUERY = "query"


@dataclass(frozen=True)
class CalibrationObservation:
    """A verifier-labelled observation admitted to calibration fitting."""

    observation_id: str
    task_id: str
    group_id: str
    model_call_id: str
    cluster_id: str
    panel_digest: str
    claim_digest: str
    model_id: str
    prompt_id: str
    decoder_id: str
    scorer_artifact_digest: str
    admitting_verifier_id: str
    score_admission_protocol_digest: str
    score_admission_receipt_digest: str
    annotation_protocol_digest: str
    annotation_ontology_digest: str
    annotation_receipt_digest: str
    role: ObservationRole
    score: float
    affirmative_label: bool

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "task_id",
            "group_id",
            "model_call_id",
            "cluster_id",
        ):
            _require_nonempty(name, getattr(self, name))
        _require_sha256("panel_digest", self.panel_digest)
        _require_sha256("claim_digest", self.claim_digest)
        _require_nonempty("model_id", self.model_id)
        _require_nonempty("prompt_id", self.prompt_id)
        _require_nonempty("decoder_id", self.decoder_id)
        _require_nonempty("admitting_verifier_id", self.admitting_verifier_id)
        for name in (
            "scorer_artifact_digest",
            "score_admission_protocol_digest",
            "score_admission_receipt_digest",
            "annotation_protocol_digest",
            "annotation_ontology_digest",
            "annotation_receipt_digest",
        ):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.role, ObservationRole):
            raise TypeError("calibration observation role is malformed")
        _require_score("calibration score", self.score)
        if type(self.affirmative_label) is not bool:
            raise TypeError("affirmative_label must be literal bool")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": _OBSERVATION_SCHEMA,
            "observation_id": self.observation_id,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "model_call_id": self.model_call_id,
            "cluster_id": self.cluster_id,
            "panel_digest": self.panel_digest,
            "claim_digest": self.claim_digest,
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "decoder_id": self.decoder_id,
            "scorer_artifact_digest": self.scorer_artifact_digest,
            "admitting_verifier_id": self.admitting_verifier_id,
            "score_admission_protocol_digest": self.score_admission_protocol_digest,
            "score_admission_receipt_digest": self.score_admission_receipt_digest,
            "annotation_protocol_digest": self.annotation_protocol_digest,
            "annotation_ontology_digest": self.annotation_ontology_digest,
            "annotation_receipt_digest": self.annotation_receipt_digest,
            "role": self.role.value,
            "score": float(self.score),
            "affirmative_label": self.affirmative_label,
        }


@dataclass(frozen=True)
class PreregisteredCalibrationPlan:
    """Verifier-owned choices frozen before any calibration fit or query.

    The only admitted orientation is ``AT_LEAST``.  Fixed score bins avoid
    choosing cut points after seeing labels.  ``expected_plan_digest`` at fit
    time is the outer verifier's pre-published commitment.
    """

    verifier_id: str
    registration_id: str
    claim_digest: str
    design: CalibrationDesign
    development_units: tuple[DevelopmentUnit, ...]
    bin_edges: tuple[float, ...]
    confidence_level: float
    minimum_clusters_per_bin: int
    affirmative_threshold: float
    algorithm_id: str = field(default=_ALGORITHM_ID, init=False)
    estimand_id: str = field(default=_ESTIMAND_ID, init=False)

    def __post_init__(self) -> None:
        _require_nonempty("verifier_id", self.verifier_id)
        _require_nonempty("registration_id", self.registration_id)
        _require_sha256("claim_digest", self.claim_digest)
        if not isinstance(self.design, CalibrationDesign):
            raise TypeError("calibration design is malformed")
        if not isinstance(self.development_units, tuple) or not self.development_units:
            raise ValueError(
                "development_units must be a non-empty immutable preregistration"
            )
        if any(not isinstance(unit, DevelopmentUnit) for unit in self.development_units):
            raise TypeError("development_units contains a malformed unit")
        if tuple(
            sorted(self.development_units, key=lambda unit: unit.observation_id)
        ) != self.development_units:
            raise ValueError("development_units must be sorted by observation_id")
        observation_ids = tuple(unit.observation_id for unit in self.development_units)
        panel_digests = tuple(unit.panel_digest for unit in self.development_units)
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("development observation ids must be unique")
        if len(panel_digests) != len(set(panel_digests)):
            raise ValueError("development panel digests must be unique")
        # Repeated panels, calls, tasks, and generation groups can be
        # arbitrarily dependent.  A preregistration may merge them into a
        # larger cluster, but it may never split one identity across clusters.
        for identity_name in ("task_id", "group_id", "model_call_id"):
            cluster_by_identity: dict[str, str] = {}
            for unit in self.development_units:
                identity = getattr(unit, identity_name)
                prior = cluster_by_identity.setdefault(identity, unit.cluster_id)
                if prior != unit.cluster_id:
                    raise ValueError(
                        f"{identity_name} {identity!r} is split across dependence clusters"
                    )
        if self.algorithm_id != _ALGORITHM_ID:  # defensive against object forgery
            raise ValueError("unsupported calibration algorithm")
        if self.estimand_id != _ESTIMAND_ID:
            raise ValueError("unsupported predictive-support estimand")
        if not isinstance(self.bin_edges, tuple) or len(self.bin_edges) < 3:
            raise ValueError("bin_edges must define at least two fixed bins")
        edges = tuple(_require_score("bin edge", edge) for edge in self.bin_edges)
        if edges[0] != 0.0 or edges[-1] != 1.0:
            raise ValueError("fixed calibration bins must cover exactly [0, 1]")
        if any(left >= right for left, right in zip(edges, edges[1:])):
            raise ValueError("bin_edges must be strictly increasing")
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not math.isfinite(float(self.confidence_level))
            or not 0.0 < float(self.confidence_level) < 1.0
        ):
            raise ValueError("confidence_level must be finite and lie in (0, 1)")
        if (
            isinstance(self.minimum_clusters_per_bin, bool)
            or not isinstance(self.minimum_clusters_per_bin, int)
            or self.minimum_clusters_per_bin < 2
        ):
            raise ValueError(
                "minimum_clusters_per_bin must be an integer of at least two"
            )
        _require_score("affirmative_threshold", self.affirmative_threshold)

    @property
    def affirmative_relation(self) -> AffirmativeRelation:
        return AffirmativeRelation.AT_LEAST

    def to_data(self) -> dict[str, object]:
        return {
            "schema": _PLAN_SCHEMA,
            "verifier_id": self.verifier_id,
            "registration_id": self.registration_id,
            "claim_digest": self.claim_digest,
            "design": self.design.to_data(),
            "design_digest": self.design.digest(),
            "development_units": [unit.to_data() for unit in self.development_units],
            "development_manifest_digest": self.development_manifest_digest,
            "algorithm_id": self.algorithm_id,
            "estimand_id": self.estimand_id,
            "bin_edges": [float(edge) for edge in self.bin_edges],
            "confidence_level": float(self.confidence_level),
            "minimum_clusters_per_bin": self.minimum_clusters_per_bin,
            "affirmative_relation": self.affirmative_relation.value,
            "affirmative_threshold": float(self.affirmative_threshold),
        }

    def digest(self) -> str:
        return _canonical_sha256(self.to_data())

    @property
    def development_manifest_digest(self) -> str:
        return _canonical_sha256(
            {
                "schema": _DEVELOPMENT_UNIT_SCHEMA + ".manifest",
                "units": [unit.to_data() for unit in self.development_units],
            }
        )


@dataclass(frozen=True)
class CalibrationBand:
    """One score bin with a cluster-level predictive-support band."""

    score_lower: float
    score_upper: float
    include_upper: bool
    panel_count: int
    cluster_ids: tuple[str, ...]
    cluster_support_mean: float
    support_lower: float
    support_upper: float

    def __post_init__(self) -> None:
        lower = _require_score("band score_lower", self.score_lower)
        upper = _require_score("band score_upper", self.score_upper)
        if lower >= upper:
            raise ValueError("calibration band score bounds must increase")
        if type(self.include_upper) is not bool:
            raise TypeError("include_upper must be literal bool")
        if (
            isinstance(self.panel_count, bool)
            or not isinstance(self.panel_count, int)
            or self.panel_count < 1
        ):
            raise ValueError("calibration band panel_count must be positive")
        if not isinstance(self.cluster_ids, tuple) or not self.cluster_ids:
            raise ValueError("calibration band requires cluster identities")
        if tuple(sorted(self.cluster_ids)) != self.cluster_ids:
            raise ValueError("calibration band cluster ids must be sorted")
        if len(self.cluster_ids) != len(set(self.cluster_ids)):
            raise ValueError("calibration band cluster ids must be unique")
        if self.panel_count < len(self.cluster_ids):
            raise ValueError("panel_count cannot be smaller than cluster_count")
        for cluster_id in self.cluster_ids:
            _require_nonempty("cluster_id", cluster_id)
        _require_score("cluster_support_mean", self.cluster_support_mean)
        support_lower = _require_score("support_lower", self.support_lower)
        support_upper = _require_score("support_upper", self.support_upper)
        if support_lower > support_upper:
            raise ValueError("calibration support interval is reversed")

    def contains(self, score: float) -> bool:
        return self.score_lower <= score and (
            score <= self.score_upper if self.include_upper else score < self.score_upper
        )

    @property
    def cluster_count(self) -> int:
        return len(self.cluster_ids)

    def to_data(self) -> dict[str, object]:
        return {
            "score_lower": float(self.score_lower),
            "score_upper": float(self.score_upper),
            "include_upper": self.include_upper,
            "panel_count": self.panel_count,
            "cluster_ids": list(self.cluster_ids),
            "cluster_count": self.cluster_count,
            "cluster_support_mean": float(self.cluster_support_mean),
            "support_lower": float(self.support_lower),
            "support_upper": float(self.support_upper),
        }


@dataclass(frozen=True)
class MonotoneCalibrationArtifact:
    """Content-addressed cluster-calibrated predictive-support fit."""

    plan: PreregisteredCalibrationPlan
    plan_digest: str
    development_observations_digest: str
    development_panel_digests: tuple[str, ...]
    bands: tuple[CalibrationBand, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PreregisteredCalibrationPlan):
            raise TypeError("calibration artifact plan is malformed")
        _require_sha256("plan_digest", self.plan_digest)
        if self.plan.digest() != self.plan_digest:
            raise SoftPredicateIntegrityError("calibration plan digest mismatch")
        _require_sha256(
            "development_observations_digest", self.development_observations_digest
        )
        if not isinstance(self.development_panel_digests, tuple):
            raise TypeError("development panel digests must be an immutable tuple")
        if tuple(sorted(self.development_panel_digests)) != self.development_panel_digests:
            raise ValueError("development panel digests must be sorted")
        if len(self.development_panel_digests) != len(set(self.development_panel_digests)):
            raise ValueError("development panel digests must be unique")
        for digest in self.development_panel_digests:
            _require_sha256("development panel digest", digest)
        expected_panels = tuple(
            sorted(unit.panel_digest for unit in self.plan.development_units)
        )
        if self.development_panel_digests != expected_panels:
            raise SoftPredicateIntegrityError(
                "artifact development panels differ from preregistration"
            )
        if not isinstance(self.bands, tuple) or len(self.bands) != len(self.plan.bin_edges) - 1:
            raise ValueError("artifact must contain exactly one band per fixed bin")
        registered_clusters = {
            unit.cluster_id for unit in self.plan.development_units
        }
        for index, band in enumerate(self.bands):
            if not isinstance(band, CalibrationBand):
                raise TypeError("calibration artifact contains a malformed band")
            if (
                band.score_lower != self.plan.bin_edges[index]
                or band.score_upper != self.plan.bin_edges[index + 1]
            ):
                raise ValueError("calibration band boundaries differ from preregistration")
            if band.include_upper != (index == len(self.bands) - 1):
                raise ValueError("only the final calibration band may include its upper edge")
            if band.cluster_count < self.plan.minimum_clusters_per_bin:
                raise ValueError(
                    "calibration band is below preregistered cluster minimum"
                )
            if not set(band.cluster_ids) <= registered_clusters:
                raise ValueError("calibration band cites an unregistered cluster")
        if sum(band.panel_count for band in self.bands) != len(
            self.plan.development_units
        ):
            raise ValueError("calibration bands do not account for every panel")
        lowers = tuple(band.support_lower for band in self.bands)
        uppers = tuple(band.support_upper for band in self.bands)
        if any(left > right for left, right in zip(lowers, lowers[1:])):
            raise ValueError("calibration lower band is not monotone")
        if any(left > right for left, right in zip(uppers, uppers[1:])):
            raise ValueError("calibration upper band is not monotone")

    @property
    def claim_digest(self) -> str:
        return self.plan.claim_digest

    @property
    def affirmative_threshold(self) -> float:
        return float(self.plan.affirmative_threshold)

    @property
    def affirmative_relation(self) -> AffirmativeRelation:
        return AffirmativeRelation.AT_LEAST

    def band_for(self, score: float) -> CalibrationBand:
        checked = _require_score("query score", score)
        index = bisect.bisect_right(self.plan.bin_edges, checked) - 1
        index = min(index, len(self.bands) - 1)
        band = self.bands[index]
        if not band.contains(checked):  # pragma: no cover - constructor invariant
            raise SoftPredicateIntegrityError("calibration bins do not cover score")
        return band

    def to_data(self) -> dict[str, object]:
        return {
            "schema": _ARTIFACT_SCHEMA,
            "plan": self.plan.to_data(),
            "plan_digest": self.plan_digest,
            "development_observations_digest": self.development_observations_digest,
            "development_panel_digests": list(self.development_panel_digests),
            "bands": [band.to_data() for band in self.bands],
        }

    def digest(self) -> str:
        return _canonical_sha256(self.to_data())


def _observation_bin(score: float, edges: tuple[float, ...]) -> int:
    index = bisect.bisect_right(edges, score) - 1
    return min(index, len(edges) - 2)


def fit_monotone_calibration(
    plan: PreregisteredCalibrationPlan,
    claim: SoftPredicateClaim,
    observations: Sequence[CalibrationObservation],
    *,
    expected_plan_digest: str,
) -> MonotoneCalibrationArtifact:
    """Fit fixed-bin bounds over preregistered dependence clusters.

    Panels within a task/group/model call may be arbitrarily dependent.  In
    each bin they are first reduced to one bounded mean per preregistered
    cluster.  If the outer sampling design justifies independence across those
    clusters, a Hoeffding radius with ``n = number of clusters`` and a union
    bound across fixed bins gives simultaneous finite-sample predictive-support
    bounds.  This code does not establish that sampling assumption.

    Monotonicity tightens lower bounds by prefix maxima and upper bounds by
    suffix minima.  Infeasible observations fail instead of flipping polarity.
    """

    if not isinstance(plan, PreregisteredCalibrationPlan):
        raise TypeError("plan must be a PreregisteredCalibrationPlan")
    if not isinstance(claim, SoftPredicateClaim):
        raise TypeError("claim must be a SoftPredicateClaim")
    _require_sha256("expected_plan_digest", expected_plan_digest)
    if plan.digest() != expected_plan_digest:
        raise SoftPredicateIntegrityError("plan differs from preregistered digest")
    claim_digest = claim.digest()
    if plan.claim_digest != claim_digest:
        raise SoftPredicateIntegrityError("plan is registered to a different claim")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise TypeError("observations must be a finite sequence")
    supplied = tuple(observations)
    if not supplied:
        raise CalibrationError("calibration requires development observations")
    if any(not isinstance(item, CalibrationObservation) for item in supplied):
        raise TypeError("calibration contains a malformed observation")
    ordered = tuple(sorted(supplied, key=lambda item: item.observation_id))
    observation_ids = tuple(item.observation_id for item in ordered)
    panel_digests = tuple(item.panel_digest for item in ordered)
    if len(observation_ids) != len(set(observation_ids)):
        raise CalibrationError("calibration observation ids must be unique")
    if len(panel_digests) != len(set(panel_digests)):
        raise CalibrationError("a development panel cannot be counted twice")
    expected_ids = tuple(unit.observation_id for unit in plan.development_units)
    if observation_ids != expected_ids:
        raise CalibrationError(
            "observations differ from the exact preregistered development manifest"
        )
    design = plan.design
    for expected, item in zip(plan.development_units, ordered, strict=True):
        if item.role is not ObservationRole.DEVELOPMENT:
            raise CalibrationError(
                f"{item.role.value} observation {item.observation_id!r} cannot fit calibration"
            )
        if (
            item.task_id != expected.task_id
            or item.group_id != expected.group_id
            or item.model_call_id != expected.model_call_id
            or item.cluster_id != expected.cluster_id
            or item.panel_digest != expected.panel_digest
        ):
            raise SoftPredicateIntegrityError(
                f"calibration observation {item.observation_id!r} differs "
                "from its preregistered task/group/call/cluster/panel identity"
            )
        if (
            item.claim_digest != claim_digest
            or item.model_id != claim.model_id
            or item.prompt_id != claim.prompt_id
            or item.decoder_id != claim.decoder_id
            or item.scorer_artifact_digest != design.scorer_artifact_digest
            or item.admitting_verifier_id != plan.verifier_id
            or item.score_admission_protocol_digest
            != design.score_admission_protocol_digest
            or item.annotation_protocol_digest
            != design.annotation_protocol_digest
            or item.annotation_ontology_digest
            != design.annotation_ontology_digest
        ):
            raise SoftPredicateIntegrityError(
                f"calibration observation {item.observation_id!r} has the "
                "wrong operational identity"
            )

    bin_count = len(plan.bin_edges) - 1
    labels_by_bin_cluster: list[dict[str, list[bool]]] = [
        {} for _ in range(bin_count)
    ]
    for item in ordered:
        index = _observation_bin(
            _require_score("calibration score", item.score), plan.bin_edges
        )
        labels_by_bin_cluster[index].setdefault(item.cluster_id, []).append(
            item.affirmative_label
        )
    sparse = [
        index
        for index, clusters in enumerate(labels_by_bin_cluster)
        if len(clusters) < plan.minimum_clusters_per_bin
    ]
    if sparse:
        raise CalibrationError(
            "insufficient independent calibration clusters in preregistered bins: "
            + ", ".join(str(index) for index in sparse)
        )

    alpha = 1.0 - float(plan.confidence_level)
    cluster_means_by_bin: list[tuple[float, ...]] = []
    raw_lower: list[float] = []
    raw_upper: list[float] = []
    for clusters in labels_by_bin_cluster:
        cluster_means = tuple(
            sum(labels) / len(labels)
            for _, labels in sorted(clusters.items())
        )
        cluster_means_by_bin.append(cluster_means)
        empirical = sum(cluster_means) / len(cluster_means)
        radius = math.sqrt(
            math.log((2.0 * bin_count) / alpha)
            / (2.0 * len(cluster_means))
        )
        raw_lower.append(max(0.0, empirical - radius))
        raw_upper.append(min(1.0, empirical + radius))

    monotone_lower: list[float] = []
    running_lower = 0.0
    for lower in raw_lower:
        running_lower = max(running_lower, lower)
        monotone_lower.append(running_lower)
    monotone_upper = [1.0] * bin_count
    running_upper = 1.0
    for index in range(bin_count - 1, -1, -1):
        running_upper = min(running_upper, raw_upper[index])
        monotone_upper[index] = running_upper
    if any(lower > upper for lower, upper in zip(monotone_lower, monotone_upper, strict=True)):
        raise CalibrationError(
            "development observations are incompatible with the preregistered monotone direction"
        )

    bands = tuple(
        CalibrationBand(
            score_lower=float(plan.bin_edges[index]),
            score_upper=float(plan.bin_edges[index + 1]),
            include_upper=index == bin_count - 1,
            panel_count=sum(
                len(labels) for labels in labels_by_bin_cluster[index].values()
            ),
            cluster_ids=tuple(sorted(labels_by_bin_cluster[index])),
            cluster_support_mean=(
                sum(cluster_means_by_bin[index])
                / len(cluster_means_by_bin[index])
            ),
            support_lower=monotone_lower[index],
            support_upper=monotone_upper[index],
        )
        for index in range(bin_count)
    )
    observations_digest = _canonical_sha256(
        {
            "schema": _OBSERVATION_SCHEMA + ".set",
            "observations": [item.to_data() for item in ordered],
        }
    )
    return MonotoneCalibrationArtifact(
        plan=plan,
        plan_digest=expected_plan_digest,
        development_observations_digest=observations_digest,
        development_panel_digests=tuple(sorted(panel_digests)),
        bands=bands,
    )


@dataclass(frozen=True)
class CalibratedPredictiveSupport(SoftSemanticObservation):
    """Population-calibrated support, explicitly not individual semantic truth."""

    target_population_id: str = ""
    sampling_design_id: str = ""
    calibration_digest: str = ""
    scorer_artifact_digest: str = ""
    effective_cluster_count: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_nonempty("target_population_id", self.target_population_id)
        _require_nonempty("sampling_design_id", self.sampling_design_id)
        _require_sha256("calibration_digest", self.calibration_digest)
        _require_sha256("scorer_artifact_digest", self.scorer_artifact_digest)
        if (
            isinstance(self.effective_cluster_count, bool)
            or not isinstance(self.effective_cluster_count, int)
            or self.effective_cluster_count < 1
        ):
            raise ValueError("effective_cluster_count must be positive")


@dataclass(frozen=True)
class FrozenVisualScore:
    """One externally admitted scorer record; not a pixels-to-score proof.

    ``score=None`` is an explicit missing observation and therefore becomes
    ``INDETERMINATE``.  A malformed scalar cannot be constructed and query
    labels do not appear in this schema.  The receipt fields are commitments
    for an outer verifier to authenticate.  This module checks their identity
    but cannot prove the scorer ran on the declared panel bytes.
    """

    task_id: str
    group_id: str
    model_call_id: str
    cluster_id: str
    panel_digest: str
    claim_digest: str
    model_id: str
    prompt_id: str
    decoder_id: str
    scorer_artifact_digest: str
    admitting_verifier_id: str
    score_admission_protocol_digest: str
    score_admission_receipt_digest: str
    score: float | None
    description: str = ""
    observed_cue_ids: tuple[str, ...] = ()
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "group_id",
            "model_call_id",
            "cluster_id",
            "admitting_verifier_id",
        ):
            _require_nonempty(name, getattr(self, name))
        _require_sha256("panel_digest", self.panel_digest)
        _require_sha256("claim_digest", self.claim_digest)
        _require_nonempty("model_id", self.model_id)
        _require_nonempty("prompt_id", self.prompt_id)
        _require_nonempty("decoder_id", self.decoder_id)
        _require_sha256("scorer_artifact_digest", self.scorer_artifact_digest)
        _require_sha256(
            "score_admission_protocol_digest",
            self.score_admission_protocol_digest,
        )
        _require_sha256(
            "score_admission_receipt_digest",
            self.score_admission_receipt_digest,
        )
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        if not isinstance(self.observed_cue_ids, tuple):
            raise TypeError("observed_cue_ids must be an immutable tuple")
        for cue in self.observed_cue_ids:
            _require_nonempty("observed cue id", cue)
        if len(self.observed_cue_ids) != len(set(self.observed_cue_ids)):
            raise ValueError("observed cue ids must be unique")
        if self.score is None:
            _require_nonempty("missing_reason", self.missing_reason)
            if self.observed_cue_ids:
                raise ValueError("a missing score cannot claim observed affirmative cues")
        else:
            _require_score("query score", self.score)
            if self.missing_reason is not None:
                raise ValueError("a present score cannot carry a missing reason")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": _SCORE_SCHEMA,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "model_call_id": self.model_call_id,
            "cluster_id": self.cluster_id,
            "panel_digest": self.panel_digest,
            "claim_digest": self.claim_digest,
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "decoder_id": self.decoder_id,
            "scorer_artifact_digest": self.scorer_artifact_digest,
            "admitting_verifier_id": self.admitting_verifier_id,
            "score_admission_protocol_digest": self.score_admission_protocol_digest,
            "score_admission_receipt_digest": self.score_admission_receipt_digest,
            "score": None if self.score is None else float(self.score),
            "description": self.description,
            "observed_cue_ids": list(self.observed_cue_ids),
            "missing_reason": self.missing_reason,
        }

    def digest(self) -> str:
        return _canonical_sha256(self.to_data())


def _operational_digest(
    claim: SoftPredicateClaim, calibration: MonotoneCalibrationArtifact
) -> str:
    return _canonical_sha256(
        {
            "schema": _OPERATION_SCHEMA,
            "claim": claim.to_data(),
            "claim_digest": claim.digest(),
            "calibration": calibration.to_data(),
            "calibration_digest": calibration.digest(),
            "result_semantics": "cluster_calibrated_predictive_support",
            "individual_semantic_truth_claimed": False,
            "affirmative_relation": AffirmativeRelation.AT_LEAST.value,
            "affirmative_threshold": calibration.affirmative_threshold,
        }
    )


def _integrity_provenance(
    claim: SoftPredicateClaim,
    *,
    claim_digest: str,
    calibration_digest: str,
    scorer_artifact_digest: str,
    calibration_design_digest: str,
    target_population_id: str,
    sampling_design_id: str,
    operational_digest: str,
) -> Provenance:
    return Provenance(
        producer="bongard.soft_predicate_bridge",
        version="2",
        method="cluster_calibrated_predictive_support",
        input_digests=(claim_digest, calibration_digest),
        artifact_digest=operational_digest,
        details=tuple(
            sorted(
                (
                    ("decoder_id", claim.decoder_id),
                    ("calibration_design_digest", calibration_design_digest),
                    ("model_id", claim.model_id),
                    ("prompt_id", claim.prompt_id),
                    (
                        "scorer_artifact_digest",
                        scorer_artifact_digest,
                    ),
                    ("sampling_design_id", sampling_design_id),
                    ("target_population_id", target_population_id),
                )
            )
        ),
    )


@dataclass(frozen=True)
class RegisteredSoftPredicate:
    """The exact registered leg plus its only admitted IR atom constructor."""

    claim: SoftPredicateClaim
    calibration: MonotoneCalibrationArtifact
    operational_digest: str
    contract: LegContract = field(repr=False, compare=False)
    reference: LegReference
    _claim_digest: str = field(repr=False)
    _calibration_digest: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_sha256("operational_digest", self.operational_digest)
        _require_sha256("claim digest", self._claim_digest)
        _require_sha256("calibration digest", self._calibration_digest)
        if self.contract.operational_digest != self.operational_digest:
            raise SoftPredicateIntegrityError("leg does not bind the soft operation")
        if self.contract.codomain != SOFT_SEMANTIC:
            raise SoftPredicateIntegrityError("soft leg has the wrong codomain")
        if self.contract.domain != (FROZEN_VISUAL_SCORE,):
            raise SoftPredicateIntegrityError(
                "calibration leg must consume only FROZEN_VISUAL_SCORE"
            )
        if self.contract.affirmative_relations != frozenset({AffirmativeRelation.AT_LEAST}):
            raise SoftPredicateIntegrityError("soft leg admits a polarity flip")
        if self.reference.contract_digest != self.contract.digest():
            raise SoftPredicateIntegrityError("registered leg reference is stale")

    @property
    def affirmative_threshold(self) -> float:
        return self.calibration.affirmative_threshold

    def atom(self, score_binding: str = "frozen_score") -> Atom:
        """Build the fixed positive atom over calibrated predictive support."""

        if not isinstance(score_binding, str) or not score_binding.strip():
            raise ValueError("frozen-score binding must be non-empty")
        if self.claim.digest() != self._claim_digest:
            raise SoftPredicateIntegrityError("soft claim changed after registration")
        if self.calibration.digest() != self._calibration_digest:
            raise SoftPredicateIntegrityError("calibration changed after registration")
        if _operational_digest(self.claim, self.calibration) != self.operational_digest:
            raise SoftPredicateIntegrityError("soft operation digest mismatch")
        return Atom(
            call=StaticLegCall(self.reference, (score_binding,)),
            relation=Relation.AT_LEAST,
            claim="calibrated predictive support for: " + self.claim.phrase,
            lower=Quantity(self.affirmative_threshold, Unit.PROBABILITY),
        )

    def __bool__(self) -> NoReturn:
        raise TypeError("a registered soft predicate is not itself a truth value")


def register_soft_predicate(
    registry: LegRegistry,
    *,
    name: str,
    version: str,
    claim: SoftPredicateClaim,
    calibration: MonotoneCalibrationArtifact,
    expected_claim_digest: str,
    expected_calibration_digest: str,
    invariance: InvarianceContract = InvarianceContract(),
    cost: int = 1,
) -> RegisteredSoftPredicate:
    """Register a derived score-to-predictive-support calibration leg.

    Both expected digests must come from the verifier's frozen configuration.
    The closure rechecks those identities on every call, so post-registration
    mutation through unsafe reflection becomes ``ERROR`` rather than evidence.
    This leg consumes no panel and makes no claim that the scorer receipt is
    authentic; the outer verifier must authenticate it before assigning the
    ``FROZEN_VISUAL_SCORE`` boundary type.
    """

    if not isinstance(registry, LegRegistry):
        raise TypeError("registry must be verifier-owned LegRegistry")
    if not isinstance(claim, SoftPredicateClaim):
        raise TypeError("claim must be a SoftPredicateClaim")
    if not isinstance(calibration, MonotoneCalibrationArtifact):
        raise TypeError("calibration must be a MonotoneCalibrationArtifact")
    _require_sha256("expected_claim_digest", expected_claim_digest)
    _require_sha256("expected_calibration_digest", expected_calibration_digest)
    if claim.digest() != expected_claim_digest:
        raise SoftPredicateIntegrityError("claim differs from verifier commitment")
    if calibration.digest() != expected_calibration_digest:
        raise SoftPredicateIntegrityError(
            "calibration differs from verifier commitment"
        )
    if calibration.claim_digest != expected_claim_digest:
        raise SoftPredicateIntegrityError("calibration belongs to another claim")
    if invariance != InvarianceContract():
        raise ValueError(
            "a frozen-score calibration leg cannot declare panel-transform invariance"
        )
    operation_digest = _operational_digest(claim, calibration)
    design = calibration.plan.design
    base_provenance = _integrity_provenance(
        claim,
        claim_digest=expected_claim_digest,
        calibration_digest=expected_calibration_digest,
        scorer_artifact_digest=design.scorer_artifact_digest,
        calibration_design_digest=design.digest(),
        target_population_id=design.target_population_id,
        sampling_design_id=design.sampling_design_id,
        operational_digest=operation_digest,
    )
    development_panels = frozenset(calibration.development_panel_digests)
    development_tasks = frozenset(
        unit.task_id for unit in calibration.plan.development_units
    )
    development_groups = frozenset(
        unit.group_id for unit in calibration.plan.development_units
    )
    development_calls = frozenset(
        unit.model_call_id for unit in calibration.plan.development_units
    )
    development_clusters = frozenset(
        unit.cluster_id for unit in calibration.plan.development_units
    )

    def calibrated_predictive_support_leg(
        record: object,
    ) -> Evidence[CalibratedPredictiveSupport]:
        if claim.digest() != expected_claim_digest:
            return Evidence.error(
                base_provenance,
                "SoftPredicateIntegrityError",
                "soft claim changed after registration",
            )
        if calibration.digest() != expected_calibration_digest:
            return Evidence.error(
                base_provenance,
                "SoftPredicateIntegrityError",
                "calibration changed after registration",
            )
        if _operational_digest(claim, calibration) != operation_digest:
            return Evidence.error(
                base_provenance,
                "SoftPredicateIntegrityError",
                "operational configuration changed after registration",
            )
        if record is None:
            return Evidence.indeterminate(
                base_provenance, "the externally admitted frozen score is missing"
            )

        upstream_provenance: Provenance | None = None
        if isinstance(record, Evidence):
            upstream_provenance = record.provenance
            if record.disposition is Disposition.CERTIFIED_ABSENT:
                # A missing calibrated score cannot bypass the calibration
                # threshold merely because an upstream producer called its
                # own prose an absence certificate.  Dedicated geometric
                # absence certifiers belong in separate registered legs.
                return Evidence.indeterminate(
                    Provenance.composed(
                        "bongard.soft_predicate_bridge",
                        "2",
                        "upstream_absence_without_calibrated_score",
                        (base_provenance, record.provenance),
                    ),
                    "upstream absence did not provide the preregistered calibrated score",
                    record.uncertainty,
                )
            if record.disposition is Disposition.INDETERMINATE:
                return Evidence.indeterminate(
                    Provenance.composed(
                        "bongard.soft_predicate_bridge",
                        "2",
                        "upstream_indeterminate",
                        (base_provenance, record.provenance),
                    ),
                    record.reason or "score admission was indeterminate",
                    record.uncertainty,
                )
            if record.disposition is Disposition.ERROR:
                return Evidence.error(
                    Provenance.composed(
                        "bongard.soft_predicate_bridge",
                        "2",
                        "upstream_error",
                        (base_provenance, record.provenance),
                    ),
                    record.error_type or "ScoreAdmissionError",
                    record.reason or "score admission failed",
                )
            record = record.unwrap()
        if not isinstance(record, FrozenVisualScore):
            return Evidence.error(
                base_provenance,
                "MalformedVisualScore",
                f"expected FrozenVisualScore, got {type(record).__name__}",
            )
        try:
            # Revalidate values in case unsafe reflection bypassed the frozen DTO.
            for identity_name in (
                "task_id",
                "group_id",
                "model_call_id",
                "cluster_id",
                "admitting_verifier_id",
            ):
                _require_nonempty(identity_name, getattr(record, identity_name))
            _require_sha256("panel_digest", record.panel_digest)
            _require_sha256("claim_digest", record.claim_digest)
            _require_sha256(
                "scorer_artifact_digest", record.scorer_artifact_digest
            )
            _require_sha256(
                "score_admission_protocol_digest",
                record.score_admission_protocol_digest,
            )
            _require_sha256(
                "score_admission_receipt_digest",
                record.score_admission_receipt_digest,
            )
            if record.score is not None:
                score = _require_score("query score", record.score)
            else:
                score = None
            if (
                record.claim_digest != expected_claim_digest
                or record.model_id != claim.model_id
                or record.prompt_id != claim.prompt_id
                or record.decoder_id != claim.decoder_id
                or record.scorer_artifact_digest
                != design.scorer_artifact_digest
                or record.admitting_verifier_id != calibration.plan.verifier_id
                or record.score_admission_protocol_digest
                != design.score_admission_protocol_digest
            ):
                raise SoftPredicateIntegrityError(
                    "frozen score has the wrong claim/scorer/admission identity"
                )
            unknown_cues = set(record.observed_cue_ids) - set(
                claim.affirmative_cues
            )
            if unknown_cues:
                raise SoftPredicateIntegrityError(
                    "query score cites unregistered affirmative cues: "
                    + ", ".join(sorted(unknown_cues))
                )
            leaked_identity = (
                record.panel_digest in development_panels
                or record.task_id in development_tasks
                or record.group_id in development_groups
                or record.model_call_id in development_calls
                or record.cluster_id in development_clusters
            )
            if leaked_identity:
                raise SoftPredicateIntegrityError(
                    "query score overlaps a preregistered development "
                    "panel/task/group/model-call/cluster"
                )
        except (SoftPredicateError, TypeError, ValueError) as exc:
            return Evidence.error(
                base_provenance, type(exc).__name__, str(exc) or repr(exc)
            )
        packet_digest = record.digest()
        parents = (base_provenance,) if upstream_provenance is None else (
            base_provenance,
            upstream_provenance,
        )
        provenance = Provenance.composed(
            "bongard.soft_predicate_bridge",
            "2",
            "cluster_calibrated_predictive_support",
            parents,
            details=tuple(
                sorted(
                    (
                        ("cluster_id", record.cluster_id),
                        ("group_id", record.group_id),
                        ("model_call_id", record.model_call_id),
                        ("panel_digest", record.panel_digest),
                        (
                            "score_admission_receipt_digest",
                            record.score_admission_receipt_digest,
                        ),
                        ("score_packet_digest", packet_digest),
                        ("task_id", record.task_id),
                    )
                )
            ),
        )
        if score is None:
            return Evidence.indeterminate(
                provenance,
                record.missing_reason or "the frozen score is missing",
            )
        band = calibration.band_for(score)
        support = Uncertainty(
            band.support_lower,
            band.support_upper,
            confidence_level=float(calibration.plan.confidence_level),
            causes=(
                "cluster_level_hoeffding",
                "preregistered_monotone_calibration",
            ),
        )
        observation = CalibratedPredictiveSupport(
            phrase=claim.phrase,
            support=support,
            provenance=provenance,
            description=record.description,
            witness_ids=record.observed_cue_ids,
            target_population_id=design.target_population_id,
            sampling_design_id=design.sampling_design_id,
            calibration_digest=expected_calibration_digest,
            scorer_artifact_digest=design.scorer_artifact_digest,
            effective_cluster_count=band.cluster_count,
        )
        return Evidence.present(observation, provenance, support)

    contract = LegContract(
        name=name,
        version=version,
        domain=(FROZEN_VISUAL_SCORE,),
        codomain=SOFT_SEMANTIC,
        implementation=calibrated_predictive_support_leg,
        affirmative_relations=frozenset({AffirmativeRelation.AT_LEAST}),
        invariance=invariance,
        semantics=LegSemantics.DERIVED,
        cost=cost,
        operational_digest=operation_digest,
    )
    reference = registry.register(contract)
    return RegisteredSoftPredicate(
        claim=claim,
        calibration=calibration,
        operational_digest=operation_digest,
        contract=contract,
        reference=reference,
        _claim_digest=expected_claim_digest,
        _calibration_digest=expected_calibration_digest,
    )


# ---------------------------------------------------------------------------
# Family-calibrated dynamic soft claims
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoftScorerProtocol:
    """All scorer-family choices frozen before development observations.

    This digest is the prospective identity bound by task-local claims and
    blind score records.  The proposer prompt is an exact policy-static
    prompt.  The scorer prompt identity is instead a template/procedure: its
    concrete call prompt necessarily contains the dynamic claim, cue rubric,
    and verifier witness summary, and is bound by that call's receipt.

    No development unit, development label, fitted interval, query claim, or
    query panel appears here.  Consequently the digest can be published
    before calibration starts without a placeholder or hash fixed point.
    """

    family_id: str
    version: str
    proposer_grammar_id: str
    proposer_grammar_digest: str
    proposer_model_id: str
    proposer_reasoning_effort: str
    proposer_prompt_id: str
    proposer_prompt_digest: str
    scorer_model_id: str
    scorer_reasoning_effort: str
    scorer_prompt_template_id: str
    scorer_prompt_template_digest: str
    scorer_decoder_id: str
    scorer_decoder_digest: str
    ordinal_map: tuple[tuple[str, float], ...]
    aggregation: str
    witness_extractor_id: str
    witness_extractor_digest: str
    support_gate_id: str
    support_gate_digest: str
    score_bin_edges: tuple[float, ...]
    affirmative_boundary: float
    confidence_level: float
    minimum_clusters_per_bin: int
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "family_id",
            "version",
            "proposer_grammar_id",
            "proposer_model_id",
            "proposer_prompt_id",
            "scorer_model_id",
            "scorer_prompt_template_id",
            "scorer_decoder_id",
            "witness_extractor_id",
            "support_gate_id",
        ):
            _require_nonempty(name, getattr(self, name))
        for name in ("proposer_reasoning_effort", "scorer_reasoning_effort"):
            if getattr(self, name) not in _REASONING_EFFORTS:
                raise ValueError(f"{name} is not an allowlisted reasoning effort")
        for name in (
            "proposer_grammar_digest",
            "proposer_prompt_digest",
            "scorer_prompt_template_digest",
            "scorer_decoder_digest",
            "witness_extractor_digest",
            "support_gate_digest",
        ):
            _require_sha256(name, getattr(self, name))
        if self.ordinal_map != _SOFT_ORDINAL_MAP:
            raise ValueError(
                "ordinal_map must be exactly supported=1, ambiguous=0.5, "
                "unsupported=0"
            )
        if self.aggregation != _SOFT_AGGREGATION:
            raise ValueError("soft cue aggregation must be the frozen minimum")
        if not isinstance(self.score_bin_edges, tuple) or len(self.score_bin_edges) < 2:
            raise ValueError("score_bin_edges must define at least one bin")
        edges = tuple(
            _require_score("soft-protocol score bin edge", edge)
            for edge in self.score_bin_edges
        )
        if edges[0] != 0.0 or edges[-1] != 1.0:
            raise ValueError("soft-protocol score bins must cover exactly [0, 1]")
        if any(left >= right for left, right in zip(edges, edges[1:])):
            raise ValueError("soft-protocol score bin edges must strictly increase")
        _require_score("affirmative_boundary", self.affirmative_boundary)
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not math.isfinite(float(self.confidence_level))
            or not 0.0 < float(self.confidence_level) < 1.0
        ):
            raise ValueError("confidence_level must lie in (0, 1)")
        if (
            isinstance(self.minimum_clusters_per_bin, bool)
            or not isinstance(self.minimum_clusters_per_bin, int)
            or self.minimum_clusters_per_bin < 2
        ):
            raise ValueError("minimum_clusters_per_bin must be an integer >= 2")
        object.__setattr__(self, "_sealed_digest", self.digest())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": _SOFT_SCORER_PROTOCOL_SCHEMA,
            "family_id": self.family_id,
            "version": self.version,
            "identity_semantics": _SOFT_FAMILY_IDENTITY_SEMANTICS,
            "proposer": {
                "grammar_id": self.proposer_grammar_id,
                "grammar_digest": self.proposer_grammar_digest,
                "model_id": self.proposer_model_id,
                "reasoning_effort": self.proposer_reasoning_effort,
                "prompt_id": self.proposer_prompt_id,
                "prompt_digest": self.proposer_prompt_digest,
            },
            "scorer": {
                "model_id": self.scorer_model_id,
                "reasoning_effort": self.scorer_reasoning_effort,
                "prompt_template_id": self.scorer_prompt_template_id,
                "prompt_template_digest": self.scorer_prompt_template_digest,
                "decoder_id": self.scorer_decoder_id,
                "decoder_digest": self.scorer_decoder_digest,
            },
            "ordinal_map": [
                {"judgment": judgment, "score": float(score)}
                for judgment, score in self.ordinal_map
            ],
            "aggregation": self.aggregation,
            "witness_extractor": {
                "id": self.witness_extractor_id,
                "digest": self.witness_extractor_digest,
            },
            "support_gate": {
                "id": self.support_gate_id,
                "digest": self.support_gate_digest,
            },
            "calibration_protocol": {
                "algorithm_id": _SOFT_FAMILY_CALIBRATION_ID,
                "estimand_id": _ESTIMAND_ID,
                "score_bin_edges": [float(edge) for edge in self.score_bin_edges],
                "affirmative_relation": AffirmativeRelation.AT_LEAST.value,
                "affirmative_boundary": float(self.affirmative_boundary),
                "confidence_level": float(self.confidence_level),
                "minimum_clusters_per_bin": self.minimum_clusters_per_bin,
            },
        }

    def digest(self) -> str:
        return _canonical_sha256(self.to_data())

    def assert_untampered(self) -> None:
        if self.digest() != self._sealed_digest:
            raise SoftPredicateIntegrityError("soft scorer protocol changed after sealing")

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> "SoftScorerProtocol":
        data = _require_mapping(value, "soft scorer protocol")
        _require_fields(
            data,
            {
                "schema",
                "family_id",
                "version",
                "identity_semantics",
                "proposer",
                "scorer",
                "ordinal_map",
                "aggregation",
                "witness_extractor",
                "support_gate",
                "calibration_protocol",
            },
            "soft scorer protocol",
        )
        if data["schema"] != _SOFT_SCORER_PROTOCOL_SCHEMA:
            raise ValueError("unsupported soft-scorer-protocol schema")
        if data["identity_semantics"] != _SOFT_FAMILY_IDENTITY_SEMANTICS:
            raise ValueError("soft-scorer-protocol identity semantics changed")
        proposer = _require_mapping(data["proposer"], "protocol proposer")
        _require_fields(
            proposer,
            {
                "grammar_id",
                "grammar_digest",
                "model_id",
                "reasoning_effort",
                "prompt_id",
                "prompt_digest",
            },
            "protocol proposer",
        )
        scorer = _require_mapping(data["scorer"], "protocol scorer")
        _require_fields(
            scorer,
            {
                "model_id",
                "reasoning_effort",
                "prompt_template_id",
                "prompt_template_digest",
                "decoder_id",
                "decoder_digest",
            },
            "protocol scorer",
        )
        witness_extractor = _require_mapping(
            data["witness_extractor"], "protocol witness_extractor"
        )
        _require_fields(
            witness_extractor, {"id", "digest"}, "protocol witness_extractor"
        )
        support_gate = _require_mapping(data["support_gate"], "protocol support_gate")
        _require_fields(support_gate, {"id", "digest"}, "protocol support_gate")
        raw_ordinal = _require_list(data["ordinal_map"], "protocol ordinal_map")
        ordinal: list[tuple[str, float]] = []
        for item in raw_ordinal:
            entry = _require_mapping(item, "protocol ordinal-map entry")
            _require_fields(
                entry, {"judgment", "score"}, "protocol ordinal-map entry"
            )
            if not isinstance(entry["judgment"], str):
                raise TypeError("ordinal judgment must be a string")
            ordinal.append(
                (
                    entry["judgment"],
                    _require_score("ordinal score", entry["score"]),
                )
            )
        calibration = _require_mapping(
            data["calibration_protocol"], "family calibration protocol"
        )
        _require_fields(
            calibration,
            {
                "algorithm_id",
                "estimand_id",
                "score_bin_edges",
                "affirmative_relation",
                "affirmative_boundary",
                "confidence_level",
                "minimum_clusters_per_bin",
            },
            "family calibration protocol",
        )
        if calibration["algorithm_id"] != _SOFT_FAMILY_CALIBRATION_ID:
            raise ValueError("unsupported soft-family calibration algorithm")
        if calibration["estimand_id"] != _ESTIMAND_ID:
            raise ValueError("unsupported soft-family calibration estimand")
        if calibration["affirmative_relation"] != AffirmativeRelation.AT_LEAST.value:
            raise ValueError("soft-family protocol admits a polarity flip")
        raw_edges = _require_list(
            calibration["score_bin_edges"], "protocol score_bin_edges"
        )
        result = cls(
            family_id=_require_nonempty("family_id", data["family_id"]),
            version=_require_nonempty("version", data["version"]),
            proposer_grammar_id=_require_nonempty(
                "proposer grammar_id", proposer["grammar_id"]
            ),
            proposer_grammar_digest=_require_sha256(
                "proposer grammar_digest", proposer["grammar_digest"]
            ),
            proposer_model_id=_require_nonempty(
                "proposer model_id", proposer["model_id"]
            ),
            proposer_reasoning_effort=_require_nonempty(
                "proposer reasoning_effort", proposer["reasoning_effort"]
            ),
            proposer_prompt_id=_require_nonempty(
                "proposer prompt_id", proposer["prompt_id"]
            ),
            proposer_prompt_digest=_require_sha256(
                "proposer prompt_digest", proposer["prompt_digest"]
            ),
            scorer_model_id=_require_nonempty("scorer model_id", scorer["model_id"]),
            scorer_reasoning_effort=_require_nonempty(
                "scorer reasoning_effort", scorer["reasoning_effort"]
            ),
            scorer_prompt_template_id=_require_nonempty(
                "scorer prompt_template_id", scorer["prompt_template_id"]
            ),
            scorer_prompt_template_digest=_require_sha256(
                "scorer prompt_template_digest", scorer["prompt_template_digest"]
            ),
            scorer_decoder_id=_require_nonempty(
                "scorer decoder_id", scorer["decoder_id"]
            ),
            scorer_decoder_digest=_require_sha256(
                "scorer decoder_digest", scorer["decoder_digest"]
            ),
            ordinal_map=tuple(ordinal),
            aggregation=data["aggregation"],
            witness_extractor_id=_require_nonempty(
                "witness extractor id", witness_extractor["id"]
            ),
            witness_extractor_digest=_require_sha256(
                "witness extractor digest", witness_extractor["digest"]
            ),
            support_gate_id=_require_nonempty("support gate id", support_gate["id"]),
            support_gate_digest=_require_sha256(
                "support gate digest", support_gate["digest"]
            ),
            score_bin_edges=tuple(
                _require_score("protocol score bin edge", edge) for edge in raw_edges
            ),
            affirmative_boundary=_require_score(
                "affirmative_boundary", calibration["affirmative_boundary"]
            ),
            confidence_level=calibration["confidence_level"],
            minimum_clusters_per_bin=calibration["minimum_clusters_per_bin"],
        )
        _check_expected_digest("soft scorer protocol", result.digest(), expected_digest)
        return result


@dataclass(frozen=True, order=True)
class SoftFamilyDevelopmentUnit:
    """One preregistered family-calibration observation.

    The development manifest contains exact task-local claims, because those
    are the observations on which the family was calibrated.  A query claim
    is deliberately absent from :class:`SoftScorerFamily`: repeating a rubric
    on a genuinely fresh task is allowed, while all development observations
    sharing that rubric must remain in one dependence cluster.  Dependence is
    attached to the task, proposer call, scorer call, and exact claim rather
    than guessed from a nominal panel count.
    """

    observation_id: str
    task_id: str
    panel_digest: str
    claim_digest: str
    scorer_protocol_digest: str
    proposer_call_id: str
    scorer_call_id: str
    dependence_cluster_id: str
    score_record_digest: str
    annotation_receipt_digest: str
    score: float
    affirmative_label: bool
    score_bin_index: int

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "task_id",
            "proposer_call_id",
            "scorer_call_id",
            "dependence_cluster_id",
        ):
            _require_nonempty(name, getattr(self, name))
        _require_sha256("development panel_digest", self.panel_digest)
        _require_sha256("development claim_digest", self.claim_digest)
        _require_sha256(
            "development scorer_protocol_digest", self.scorer_protocol_digest
        )
        _require_sha256("development score_record_digest", self.score_record_digest)
        _require_sha256(
            "development annotation_receipt_digest",
            self.annotation_receipt_digest,
        )
        if _require_score("development score", self.score) not in {
            score for _, score in _SOFT_ORDINAL_MAP
        }:
            raise ValueError("development score is outside the frozen ordinal map")
        if type(self.affirmative_label) is not bool:
            raise TypeError("development affirmative_label must be literal bool")
        if (
            isinstance(self.score_bin_index, bool)
            or not isinstance(self.score_bin_index, int)
            or self.score_bin_index < 0
        ):
            raise ValueError("score_bin_index must be a non-negative integer")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": _SOFT_FAMILY_DEVELOPMENT_UNIT_SCHEMA,
            "observation_id": self.observation_id,
            "task_id": self.task_id,
            "panel_digest": self.panel_digest,
            "claim_digest": self.claim_digest,
            "scorer_protocol_digest": self.scorer_protocol_digest,
            "proposer_call_id": self.proposer_call_id,
            "scorer_call_id": self.scorer_call_id,
            "dependence_cluster_id": self.dependence_cluster_id,
            "score_record_digest": self.score_record_digest,
            "annotation_receipt_digest": self.annotation_receipt_digest,
            "score": float(self.score),
            "affirmative_label": self.affirmative_label,
            "score_bin_index": self.score_bin_index,
        }

    def digest(self) -> str:
        return _canonical_sha256(self.to_data())

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> "SoftFamilyDevelopmentUnit":
        data = _require_mapping(value, "soft-family development unit")
        _require_fields(
            data,
            {
                "schema",
                "observation_id",
                "task_id",
                "panel_digest",
                "claim_digest",
                "scorer_protocol_digest",
                "proposer_call_id",
                "scorer_call_id",
                "dependence_cluster_id",
                "score_record_digest",
                "annotation_receipt_digest",
                "score",
                "affirmative_label",
                "score_bin_index",
            },
            "soft-family development unit",
        )
        if data["schema"] != _SOFT_FAMILY_DEVELOPMENT_UNIT_SCHEMA:
            raise ValueError("unsupported soft-family development-unit schema")
        result = cls(
            observation_id=_require_nonempty(
                "observation_id", data["observation_id"]
            ),
            task_id=_require_nonempty("task_id", data["task_id"]),
            panel_digest=_require_sha256("panel_digest", data["panel_digest"]),
            claim_digest=_require_sha256("claim_digest", data["claim_digest"]),
            scorer_protocol_digest=_require_sha256(
                "scorer_protocol_digest", data["scorer_protocol_digest"]
            ),
            proposer_call_id=_require_nonempty(
                "proposer_call_id", data["proposer_call_id"]
            ),
            scorer_call_id=_require_nonempty(
                "scorer_call_id", data["scorer_call_id"]
            ),
            dependence_cluster_id=_require_nonempty(
                "dependence_cluster_id", data["dependence_cluster_id"]
            ),
            score_record_digest=_require_sha256(
                "score_record_digest", data["score_record_digest"]
            ),
            annotation_receipt_digest=_require_sha256(
                "annotation_receipt_digest", data["annotation_receipt_digest"]
            ),
            score=_require_score("development score", data["score"]),
            affirmative_label=data["affirmative_label"],
            score_bin_index=data["score_bin_index"],
        )
        _check_expected_digest(
            "soft-family development unit", result.digest(), expected_digest
        )
        return result


@dataclass(frozen=True)
class SoftCueJudgment:
    """One closed ordinal judgment emitted by the blind scorer.

    There is no Boolean, numeric score, polarity, disposition, prose, or
    certificate in the model-visible record.  Python maps the three admitted
    strings to numbers.  ``supported`` and ``ambiguous`` must cite at least
    one verifier-owned witness; ``unsupported`` may cite none and remains an
    empirical score, not an absence certificate.
    """

    cue_id: str
    judgment: str
    witness_ids: tuple[str, ...]
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_nonempty("cue_id", self.cue_id)
        if self.judgment not in dict(_SOFT_ORDINAL_MAP):
            raise ValueError(
                "cue judgment must be supported, ambiguous, or unsupported"
            )
        if not isinstance(self.witness_ids, tuple):
            raise TypeError("cue witness_ids must be an immutable tuple")
        for witness_id in self.witness_ids:
            _require_nonempty("cue witness_id", witness_id)
        if tuple(sorted(self.witness_ids)) != self.witness_ids:
            raise ValueError("cue witness_ids must be sorted")
        if len(self.witness_ids) != len(set(self.witness_ids)):
            raise ValueError("cue witness_ids must be unique")
        if self.judgment in {"supported", "ambiguous"} and not self.witness_ids:
            raise ValueError(
                f"{self.judgment} cue judgment requires a verifier witness"
            )
        object.__setattr__(self, "_sealed_digest", self.digest())

    @property
    def ordinal_score(self) -> float:
        return dict(_SOFT_ORDINAL_MAP)[self.judgment]

    def model_data(self) -> dict[str, object]:
        """Return the entire model-emittable shape for one cue."""

        return {
            "cue_id": self.cue_id,
            "judgment": self.judgment,
            "witness_ids": list(self.witness_ids),
        }

    def to_data(self) -> dict[str, object]:
        return {"schema": _SOFT_CUE_JUDGMENT_SCHEMA, **self.model_data()}

    def digest(self) -> str:
        return _canonical_sha256(self.to_data())

    def assert_untampered(self) -> None:
        if self.digest() != self._sealed_digest:
            raise SoftPredicateIntegrityError("soft cue judgment changed after sealing")

    @classmethod
    def from_model_data(cls, value: Mapping[str, Any]) -> "SoftCueJudgment":
        data = _require_mapping(value, "model cue judgment")
        _require_fields(
            data,
            {"cue_id", "judgment", "witness_ids"},
            "model cue judgment",
        )
        witness_ids = _require_list(data["witness_ids"], "cue witness_ids")
        if any(not isinstance(item, str) for item in witness_ids):
            raise TypeError("cue witness_ids must contain strings")
        if not isinstance(data["judgment"], str):
            raise TypeError("cue judgment must be a string")
        return cls(
            cue_id=_require_nonempty("cue_id", data["cue_id"]),
            judgment=data["judgment"],
            witness_ids=tuple(witness_ids),
        )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> "SoftCueJudgment":
        data = _require_mapping(value, "soft cue judgment")
        _require_fields(
            data,
            {"schema", "cue_id", "judgment", "witness_ids"},
            "soft cue judgment",
        )
        if data["schema"] != _SOFT_CUE_JUDGMENT_SCHEMA:
            raise ValueError("unsupported soft-cue-judgment schema")
        result = cls.from_model_data(
            {
                "cue_id": data["cue_id"],
                "judgment": data["judgment"],
                "witness_ids": data["witness_ids"],
            }
        )
        _check_expected_digest("soft cue judgment", result.digest(), expected_digest)
        return result


def _fit_soft_family_intervals(
    score_bin_edges: tuple[float, ...],
    confidence_level: float,
    minimum_clusters_per_bin: int,
    units: tuple[SoftFamilyDevelopmentUnit, ...],
) -> tuple[tuple[float, float], ...]:
    """Reproduce raw simultaneous cluster-level intervals for each score bin.

    The Bonferroni-adjusted Hoeffding radius gives simultaneous coverage over
    the fixed bins.  We intentionally make no cross-bin monotonicity
    assumption: imposing one would require a separately justified statistical
    model and can otherwise tighten or reject valid nonmonotone populations.
    """

    bin_count = len(score_bin_edges) - 1
    labels_by_bin_cluster: list[dict[str, list[bool]]] = [
        {} for _ in range(bin_count)
    ]
    for unit in units:
        actual_bin = _observation_bin(float(unit.score), score_bin_edges)
        if actual_bin != unit.score_bin_index:
            raise SoftPredicateIntegrityError(
                f"development unit {unit.observation_id!r} has a forged score bin"
            )
        labels_by_bin_cluster[actual_bin].setdefault(
            unit.dependence_cluster_id, []
        ).append(unit.affirmative_label)
    sparse = [
        index
        for index, clusters in enumerate(labels_by_bin_cluster)
        if len(clusters) < minimum_clusters_per_bin
    ]
    if sparse:
        raise CalibrationError(
            "insufficient dependence clusters in family score bins: "
            + ", ".join(str(index) for index in sparse)
        )
    alpha = 1.0 - float(confidence_level)
    raw_lower: list[float] = []
    raw_upper: list[float] = []
    for clusters in labels_by_bin_cluster:
        cluster_means = tuple(
            sum(labels) / len(labels) for _, labels in sorted(clusters.items())
        )
        empirical = sum(cluster_means) / len(cluster_means)
        radius = math.sqrt(
            math.log((2.0 * bin_count) / alpha)
            / (2.0 * len(cluster_means))
        )
        raw_lower.append(max(0.0, empirical - radius))
        raw_upper.append(min(1.0, empirical + radius))
    return tuple(zip(raw_lower, raw_upper, strict=True))


@dataclass(frozen=True)
class SoftScorerFamily:
    """A fitted calibration artifact over one prospective scorer protocol."""

    protocol: SoftScorerProtocol
    protocol_digest: str
    development_units: tuple[SoftFamilyDevelopmentUnit, ...]
    calibrated_support_intervals: tuple[tuple[float, float], ...]
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, SoftScorerProtocol):
            raise TypeError("family protocol must be a SoftScorerProtocol")
        self.protocol.assert_untampered()
        _require_sha256("protocol_digest", self.protocol_digest)
        if self.protocol.digest() != self.protocol_digest:
            raise SoftPredicateIntegrityError(
                "family protocol differs from its frozen digest"
            )
        if not isinstance(self.development_units, tuple) or not self.development_units:
            raise ValueError("development_units must be a non-empty immutable tuple")
        if any(
            not isinstance(unit, SoftFamilyDevelopmentUnit)
            for unit in self.development_units
        ):
            raise TypeError("development_units contains a malformed unit")
        if tuple(
            sorted(self.development_units, key=lambda item: item.observation_id)
        ) != self.development_units:
            raise ValueError("development_units must be sorted by observation_id")
        observation_ids = tuple(unit.observation_id for unit in self.development_units)
        panel_digests = tuple(unit.panel_digest for unit in self.development_units)
        score_record_digests = tuple(
            unit.score_record_digest for unit in self.development_units
        )
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("development observation ids must be unique")
        if len(panel_digests) != len(set(panel_digests)):
            raise ValueError("development panel digests must be unique")
        if len(score_record_digests) != len(set(score_record_digests)):
            raise ValueError("development score-record digests must be unique")
        if any(
            unit.scorer_protocol_digest != self.protocol_digest
            for unit in self.development_units
        ):
            raise SoftPredicateIntegrityError(
                "development unit belongs to a different scorer protocol"
            )
        bin_count = len(self.protocol.score_bin_edges) - 1
        bin_indices = {unit.score_bin_index for unit in self.development_units}
        if bin_indices != set(range(bin_count)):
            raise ValueError(
                "development manifest must populate every fixed score bin"
            )
        for identity_name in (
            "task_id",
            "claim_digest",
            "proposer_call_id",
            "scorer_call_id",
        ):
            cluster_by_identity: dict[str, str] = {}
            for unit in self.development_units:
                identity = getattr(unit, identity_name)
                prior = cluster_by_identity.setdefault(
                    identity, unit.dependence_cluster_id
                )
                if prior != unit.dependence_cluster_id:
                    raise ValueError(
                        f"{identity_name} {identity!r} is split across "
                        "dependence clusters"
                    )
        if (
            not isinstance(self.calibrated_support_intervals, tuple)
            or len(self.calibrated_support_intervals) != bin_count
        ):
            raise ValueError(
                "one calibrated support interval is required per score bin"
            )
        intervals: list[tuple[float, float]] = []
        for raw in self.calibrated_support_intervals:
            if not isinstance(raw, tuple) or len(raw) != 2:
                raise TypeError(
                    "calibrated support intervals must be immutable pairs"
                )
            lower = _require_score("calibrated support lower", raw[0])
            upper = _require_score("calibrated support upper", raw[1])
            if lower > upper:
                raise ValueError("calibrated support interval is reversed")
            intervals.append((lower, upper))
        fitted = _fit_soft_family_intervals(
            self.protocol.score_bin_edges,
            float(self.protocol.confidence_level),
            self.protocol.minimum_clusters_per_bin,
            self.development_units,
        )
        if tuple(intervals) != fitted:
            raise SoftPredicateIntegrityError(
                "family calibrated intervals do not reproduce the frozen "
                "cluster-level fit"
            )
        object.__setattr__(self, "_sealed_digest", self.digest())

    @classmethod
    def fit(
        cls,
        protocol: SoftScorerProtocol,
        development_units: tuple[SoftFamilyDevelopmentUnit, ...],
        *,
        expected_protocol_digest: str,
    ) -> "SoftScorerFamily":
        """Fit a family after the protocol and development records exist."""

        if not isinstance(protocol, SoftScorerProtocol):
            raise TypeError("protocol must be a SoftScorerProtocol")
        protocol.assert_untampered()
        _require_sha256("expected_protocol_digest", expected_protocol_digest)
        if protocol.digest() != expected_protocol_digest:
            raise SoftPredicateIntegrityError(
                "protocol differs from the prospective commitment"
            )
        if not isinstance(development_units, tuple):
            raise TypeError("development_units must be an immutable tuple")
        fitted = _fit_soft_family_intervals(
            protocol.score_bin_edges,
            float(protocol.confidence_level),
            protocol.minimum_clusters_per_bin,
            development_units,
        )
        return cls(
            protocol=protocol,
            protocol_digest=expected_protocol_digest,
            development_units=development_units,
            calibrated_support_intervals=fitted,
        )

    @property
    def family_id(self) -> str:
        return self.protocol.family_id

    @property
    def version(self) -> str:
        return self.protocol.version

    @property
    def score_bin_edges(self) -> tuple[float, ...]:
        return self.protocol.score_bin_edges

    @property
    def affirmative_boundary(self) -> float:
        return float(self.protocol.affirmative_boundary)

    @property
    def confidence_level(self) -> float:
        return float(self.protocol.confidence_level)

    @property
    def development_manifest_digest(self) -> str:
        return _canonical_sha256(
            {
                "schema": _SOFT_FAMILY_DEVELOPMENT_MANIFEST_SCHEMA,
                "protocol_digest": self.protocol_digest,
                "units": [unit.to_data() for unit in self.development_units],
            }
        )

    @property
    def development_claim_digests(self) -> frozenset[str]:
        return frozenset(unit.claim_digest for unit in self.development_units)

    @property
    def dependence_clusters(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        observations: dict[str, list[str]] = {}
        for unit in self.development_units:
            observations.setdefault(unit.dependence_cluster_id, []).append(
                unit.observation_id
            )
        return tuple(
            (cluster_id, tuple(sorted(observation_ids)))
            for cluster_id, observation_ids in sorted(observations.items())
        )

    def calibrated_interval(self, score: float) -> tuple[float, float, int]:
        checked = _require_score("blind soft score", score)
        index = bisect.bisect_right(self.protocol.score_bin_edges, checked) - 1
        index = min(index, len(self.calibrated_support_intervals) - 1)
        lower, upper = self.calibrated_support_intervals[index]
        return lower, upper, index

    def verify_calibration(self) -> None:
        """Raise unless protocol, manifest, and intervals reproduce exactly."""

        self.protocol.assert_untampered()
        if self.protocol.digest() != self.protocol_digest:
            raise SoftPredicateIntegrityError(
                "family protocol differs from its frozen digest"
            )
        fitted = _fit_soft_family_intervals(
            self.protocol.score_bin_edges,
            float(self.protocol.confidence_level),
            self.protocol.minimum_clusters_per_bin,
            self.development_units,
        )
        if fitted != self.calibrated_support_intervals:
            raise SoftPredicateIntegrityError(
                "family calibrated intervals do not reproduce development"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": _SOFT_FAMILY_SCHEMA,
            "protocol": self.protocol.to_data(),
            "protocol_digest": self.protocol_digest,
            "calibration_artifact": {
                "algorithm_id": _SOFT_FAMILY_CALIBRATION_ID,
                "estimand_id": _ESTIMAND_ID,
                "calibrated_support_intervals": [
                    [float(lower), float(upper)]
                    for lower, upper in self.calibrated_support_intervals
                ],
            },
            "development_manifest": {
                "schema": _SOFT_FAMILY_DEVELOPMENT_MANIFEST_SCHEMA,
                "protocol_digest": self.protocol_digest,
                "units": [unit.to_data() for unit in self.development_units],
                "digest": self.development_manifest_digest,
            },
            "dependence_clusters": [
                {
                    "cluster_id": cluster_id,
                    "observation_ids": list(observation_ids),
                }
                for cluster_id, observation_ids in self.dependence_clusters
            ],
        }

    def digest(self) -> str:
        return _canonical_sha256(self.to_data())

    def assert_untampered(self) -> None:
        self.protocol.assert_untampered()
        if self.digest() != self._sealed_digest:
            raise SoftPredicateIntegrityError("soft scorer family changed after sealing")

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> "SoftScorerFamily":
        data = _require_mapping(value, "soft scorer family")
        _require_fields(
            data,
            {
                "schema",
                "protocol",
                "protocol_digest",
                "calibration_artifact",
                "development_manifest",
                "dependence_clusters",
            },
            "soft scorer family",
        )
        if data["schema"] != _SOFT_FAMILY_SCHEMA:
            raise ValueError("unsupported soft-scorer-family schema")
        protocol_digest = _require_sha256(
            "protocol_digest", data["protocol_digest"]
        )
        protocol = SoftScorerProtocol.from_data(
            _require_mapping(data["protocol"], "family protocol"),
            expected_digest=protocol_digest,
        )
        artifact = _require_mapping(
            data["calibration_artifact"], "family calibration artifact"
        )
        _require_fields(
            artifact,
            {
                "algorithm_id",
                "estimand_id",
                "calibrated_support_intervals",
            },
            "family calibration artifact",
        )
        if artifact["algorithm_id"] != _SOFT_FAMILY_CALIBRATION_ID:
            raise ValueError("unsupported soft-family calibration algorithm")
        if artifact["estimand_id"] != _ESTIMAND_ID:
            raise ValueError("unsupported soft-family calibration estimand")
        raw_intervals = _require_list(
            artifact["calibrated_support_intervals"],
            "family calibrated_support_intervals",
        )
        intervals: list[tuple[float, float]] = []
        for raw in raw_intervals:
            pair = _require_list(raw, "family calibrated support interval")
            if len(pair) != 2:
                raise ValueError(
                    "family calibrated support interval must be a pair"
                )
            intervals.append(
                (
                    _require_score("calibrated support lower", pair[0]),
                    _require_score("calibrated support upper", pair[1]),
                )
            )
        manifest = _require_mapping(
            data["development_manifest"], "family development_manifest"
        )
        _require_fields(
            manifest,
            {"schema", "protocol_digest", "units", "digest"},
            "family development_manifest",
        )
        if manifest["schema"] != _SOFT_FAMILY_DEVELOPMENT_MANIFEST_SCHEMA:
            raise ValueError("unsupported family development-manifest schema")
        if manifest["protocol_digest"] != protocol_digest:
            raise SoftPredicateIntegrityError(
                "development manifest belongs to a different protocol"
            )
        raw_units = _require_list(manifest["units"], "family development units")
        units = tuple(
            SoftFamilyDevelopmentUnit.from_data(
                _require_mapping(item, "family development unit")
            )
            for item in raw_units
        )
        result = cls(
            protocol=protocol,
            protocol_digest=protocol_digest,
            development_units=units,
            calibrated_support_intervals=tuple(intervals),
        )
        if manifest["digest"] != result.development_manifest_digest:
            raise SoftPredicateIntegrityError(
                "family development manifest digest mismatch"
            )
        raw_clusters = _require_list(
            data["dependence_clusters"], "family dependence_clusters"
        )
        parsed_clusters: list[tuple[str, tuple[str, ...]]] = []
        for raw in raw_clusters:
            entry = _require_mapping(raw, "family dependence cluster")
            _require_fields(
                entry,
                {"cluster_id", "observation_ids"},
                "family dependence cluster",
            )
            raw_ids = _require_list(
                entry["observation_ids"],
                "dependence cluster observation_ids",
            )
            if any(not isinstance(item, str) for item in raw_ids):
                raise TypeError(
                    "dependence-cluster observation ids must be strings"
                )
            parsed_clusters.append(
                (
                    _require_nonempty(
                        "dependence cluster_id", entry["cluster_id"]
                    ),
                    tuple(raw_ids),
                )
            )
        if tuple(parsed_clusters) != result.dependence_clusters:
            raise SoftPredicateIntegrityError(
                "family dependence clusters do not reproduce development units"
            )
        _check_expected_digest(
            "soft scorer family", result.digest(), expected_digest
        )
        return result


@dataclass(frozen=True)
class BlindSoftScoreRecord:
    """Verifier-admitted result of one side-free, one-panel scorer call.

    The task-local claim and cue inventory are inputs.  The record binds the
    prospectively frozen scorer protocol, never the not-yet-fitted family
    artifact.  Its scorer receipt binds the exact dynamic prompt instantiated
    from the protocol's prompt template.  The model-visible
    payload is only ``{"cue_judgments": [...]}``; all identities, receipts,
    witness ownership, scoring, and dispositions are verifier/Python fields.
    Transport and parse failures are first-class error records rather than a
    zero or an unsupported cue.  ``pre_observation_commitment_digest`` is the
    already-frozen proposal/policy parent available before any panel is
    observed; it is never the digest of the later support-gate result.
    """

    scorer_protocol_digest: str
    task_id: str
    panel_id: str
    panel_digest: str
    claim_digest: str
    proposer_call_id: str
    proposer_receipt_digest: str
    scorer_call_id: str
    scorer_receipt_digest: str
    witness_packet_digest: str
    pre_observation_commitment_digest: str
    declared_cue_ids: tuple[str, ...]
    verifier_witness_ids: tuple[str, ...]
    outcome: str
    cue_judgments: tuple[SoftCueJudgment, ...] = ()
    failure_reason: str | None = None
    _sealed_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha256("scorer_protocol_digest", self.scorer_protocol_digest)
        for name in (
            "task_id",
            "panel_id",
            "proposer_call_id",
            "scorer_call_id",
        ):
            _require_nonempty(name, getattr(self, name))
        for name in (
            "panel_digest",
            "claim_digest",
            "proposer_receipt_digest",
            "scorer_receipt_digest",
            "witness_packet_digest",
            "pre_observation_commitment_digest",
        ):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.declared_cue_ids, tuple) or not self.declared_cue_ids:
            raise ValueError("declared_cue_ids must be a non-empty immutable tuple")
        for cue_id in self.declared_cue_ids:
            _require_nonempty("declared cue_id", cue_id)
        if len(self.declared_cue_ids) != len(set(self.declared_cue_ids)):
            raise ValueError("declared cue ids must be unique")
        if not isinstance(self.verifier_witness_ids, tuple):
            raise TypeError("verifier_witness_ids must be an immutable tuple")
        for witness_id in self.verifier_witness_ids:
            _require_nonempty("verifier witness_id", witness_id)
        if tuple(sorted(self.verifier_witness_ids)) != self.verifier_witness_ids:
            raise ValueError("verifier witness ids must be sorted")
        if len(self.verifier_witness_ids) != len(set(self.verifier_witness_ids)):
            raise ValueError("verifier witness ids must be unique")
        if self.outcome not in _BLIND_SCORE_OUTCOMES:
            raise ValueError(
                "blind soft score outcome must be present, transport_error, "
                "or parser_error"
            )
        if not isinstance(self.cue_judgments, tuple):
            raise TypeError("cue_judgments must be an immutable tuple")
        if any(
            not isinstance(judgment, SoftCueJudgment)
            for judgment in self.cue_judgments
        ):
            raise TypeError("blind score contains a malformed cue judgment")
        for judgment in self.cue_judgments:
            judgment.assert_untampered()
        if self.outcome == "present":
            if self.failure_reason is not None:
                raise ValueError("present blind score cannot carry a failure reason")
            judged_ids = tuple(judgment.cue_id for judgment in self.cue_judgments)
            if judged_ids != self.declared_cue_ids:
                if len(judged_ids) != len(set(judged_ids)):
                    raise ValueError("blind scorer repeated a declared cue")
                missing = sorted(set(self.declared_cue_ids) - set(judged_ids))
                unknown = sorted(set(judged_ids) - set(self.declared_cue_ids))
                detail = []
                if missing:
                    detail.append("missing cues: " + ", ".join(missing))
                if unknown:
                    detail.append("undeclared cues: " + ", ".join(unknown))
                if not detail:
                    detail.append("cue order differs from the declared order")
                raise ValueError(
                    "blind scorer must judge every cue exactly once; "
                    + "; ".join(detail)
                )
            allowed = set(self.verifier_witness_ids)
            forged = sorted(
                {
                    witness_id
                    for judgment in self.cue_judgments
                    for witness_id in judgment.witness_ids
                    if witness_id not in allowed
                }
            )
            if forged:
                raise SoftPredicateIntegrityError(
                    "blind scorer cited non-verifier witness ids: "
                    + ", ".join(forged)
                )
        else:
            _require_nonempty("failure_reason", self.failure_reason)
            if self.cue_judgments:
                raise ValueError(
                    "transport/parser failure cannot carry admitted cue judgments"
                )
        object.__setattr__(self, "_sealed_digest", self.digest())

    @property
    def score(self) -> float | None:
        """Return Python's frozen minimum, or ``None`` for a failed call.

        In particular, ``0.0`` is not missing: it is a syntactically complete
        empirical measurement produced by an ``unsupported`` judgment.
        """

        if self.outcome != "present":
            return None
        return min(judgment.ordinal_score for judgment in self.cue_judgments)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": _BLIND_SOFT_SCORE_SCHEMA,
            "scorer_protocol_digest": self.scorer_protocol_digest,
            "task_id": self.task_id,
            "panel_id": self.panel_id,
            "panel_digest": self.panel_digest,
            "claim_digest": self.claim_digest,
            "proposer_call_id": self.proposer_call_id,
            "proposer_receipt_digest": self.proposer_receipt_digest,
            "scorer_call_id": self.scorer_call_id,
            "scorer_receipt_digest": self.scorer_receipt_digest,
            "witness_packet_digest": self.witness_packet_digest,
            "pre_observation_commitment_digest": (
                self.pre_observation_commitment_digest
            ),
            "declared_cue_ids": list(self.declared_cue_ids),
            "verifier_witness_ids": list(self.verifier_witness_ids),
            "outcome": self.outcome,
            "cue_judgments": [
                judgment.to_data() for judgment in self.cue_judgments
            ],
            "derived_score": self.score,
            "failure_reason": self.failure_reason,
        }

    def digest(self) -> str:
        return _canonical_sha256(self.to_data())

    def assert_untampered(self) -> None:
        for judgment in self.cue_judgments:
            judgment.assert_untampered()
        if self.digest() != self._sealed_digest:
            raise SoftPredicateIntegrityError("blind soft score changed after sealing")

    @classmethod
    def from_model_output(
        cls,
        value: Mapping[str, Any],
        *,
        scorer_protocol_digest: str,
        task_id: str,
        panel_id: str,
        panel_digest: str,
        claim_digest: str,
        proposer_call_id: str,
        proposer_receipt_digest: str,
        scorer_call_id: str,
        scorer_receipt_digest: str,
        witness_packet_digest: str,
        pre_observation_commitment_digest: str,
        declared_cue_ids: tuple[str, ...],
        verifier_witness_ids: tuple[str, ...],
    ) -> "BlindSoftScoreRecord":
        """Admit the exact closed model payload and compute its score in Python."""

        data = _require_mapping(value, "blind scorer model output")
        _require_fields(data, {"cue_judgments"}, "blind scorer model output")
        raw_judgments = _require_list(
            data["cue_judgments"], "blind scorer cue_judgments"
        )
        judgments = tuple(
            SoftCueJudgment.from_model_data(
                _require_mapping(item, "blind scorer cue judgment")
            )
            for item in raw_judgments
        )
        return cls(
            scorer_protocol_digest=scorer_protocol_digest,
            task_id=task_id,
            panel_id=panel_id,
            panel_digest=panel_digest,
            claim_digest=claim_digest,
            proposer_call_id=proposer_call_id,
            proposer_receipt_digest=proposer_receipt_digest,
            scorer_call_id=scorer_call_id,
            scorer_receipt_digest=scorer_receipt_digest,
            witness_packet_digest=witness_packet_digest,
            pre_observation_commitment_digest=(
                pre_observation_commitment_digest
            ),
            declared_cue_ids=declared_cue_ids,
            verifier_witness_ids=verifier_witness_ids,
            outcome="present",
            cue_judgments=judgments,
        )

    @classmethod
    def from_data(
        cls,
        value: Mapping[str, Any],
        *,
        expected_digest: str | None = None,
    ) -> "BlindSoftScoreRecord":
        data = _require_mapping(value, "blind soft score")
        _require_fields(
            data,
            {
                "schema",
                "scorer_protocol_digest",
                "task_id",
                "panel_id",
                "panel_digest",
                "claim_digest",
                "proposer_call_id",
                "proposer_receipt_digest",
                "scorer_call_id",
                "scorer_receipt_digest",
                "witness_packet_digest",
                "pre_observation_commitment_digest",
                "declared_cue_ids",
                "verifier_witness_ids",
                "outcome",
                "cue_judgments",
                "derived_score",
                "failure_reason",
            },
            "blind soft score",
        )
        if data["schema"] != _BLIND_SOFT_SCORE_SCHEMA:
            raise ValueError("unsupported blind-soft-score schema")
        raw_cues = _require_list(data["declared_cue_ids"], "declared_cue_ids")
        raw_witnesses = _require_list(
            data["verifier_witness_ids"], "verifier_witness_ids"
        )
        if any(not isinstance(item, str) for item in raw_cues):
            raise TypeError("declared_cue_ids must contain strings")
        if any(not isinstance(item, str) for item in raw_witnesses):
            raise TypeError("verifier_witness_ids must contain strings")
        raw_judgments = _require_list(data["cue_judgments"], "cue_judgments")
        judgments = tuple(
            SoftCueJudgment.from_data(
                _require_mapping(item, "blind score cue judgment")
            )
            for item in raw_judgments
        )
        failure_reason = data["failure_reason"]
        if failure_reason is not None and not isinstance(failure_reason, str):
            raise TypeError("failure_reason must be a string or null")
        if not isinstance(data["outcome"], str):
            raise TypeError("blind score outcome must be a string")
        result = cls(
            scorer_protocol_digest=_require_sha256(
                "scorer_protocol_digest", data["scorer_protocol_digest"]
            ),
            task_id=_require_nonempty("task_id", data["task_id"]),
            panel_id=_require_nonempty("panel_id", data["panel_id"]),
            panel_digest=_require_sha256("panel_digest", data["panel_digest"]),
            claim_digest=_require_sha256("claim_digest", data["claim_digest"]),
            proposer_call_id=_require_nonempty(
                "proposer_call_id", data["proposer_call_id"]
            ),
            proposer_receipt_digest=_require_sha256(
                "proposer_receipt_digest", data["proposer_receipt_digest"]
            ),
            scorer_call_id=_require_nonempty(
                "scorer_call_id", data["scorer_call_id"]
            ),
            scorer_receipt_digest=_require_sha256(
                "scorer_receipt_digest", data["scorer_receipt_digest"]
            ),
            witness_packet_digest=_require_sha256(
                "witness_packet_digest", data["witness_packet_digest"]
            ),
            pre_observation_commitment_digest=_require_sha256(
                "pre_observation_commitment_digest",
                data["pre_observation_commitment_digest"],
            ),
            declared_cue_ids=tuple(raw_cues),
            verifier_witness_ids=tuple(raw_witnesses),
            outcome=data["outcome"],
            cue_judgments=judgments,
            failure_reason=failure_reason,
        )
        archived_score = data["derived_score"]
        if archived_score is not None:
            archived_score = _require_score("archived derived_score", archived_score)
        if archived_score != result.score:
            raise SoftPredicateIntegrityError(
                "archived blind score differs from Python ordinal aggregation"
            )
        _check_expected_digest("blind soft score", result.digest(), expected_digest)
        return result


def blind_soft_score_output_schema(
    declared_cue_ids: tuple[str, ...],
    verifier_witness_ids: tuple[str, ...],
) -> dict[str, object]:
    """Return the closed JSON schema for one blind scorer call.

    JSON Schema restricts the vocabulary and record shape; the Python DTO
    additionally proves exact cue coverage/order and witness ownership.  No
    score, Boolean, prose conclusion, polarity, or disposition is available
    to the model.
    """

    if not isinstance(declared_cue_ids, tuple) or not declared_cue_ids:
        raise ValueError("declared_cue_ids must be a non-empty immutable tuple")
    for cue_id in declared_cue_ids:
        _require_nonempty("declared cue_id", cue_id)
    if len(declared_cue_ids) != len(set(declared_cue_ids)):
        raise ValueError("declared cue ids must be unique")
    if not isinstance(verifier_witness_ids, tuple):
        raise TypeError("verifier_witness_ids must be an immutable tuple")
    for witness_id in verifier_witness_ids:
        _require_nonempty("verifier witness_id", witness_id)
    if tuple(sorted(verifier_witness_ids)) != verifier_witness_ids:
        raise ValueError("verifier witness ids must be sorted")
    if len(verifier_witness_ids) != len(set(verifier_witness_ids)):
        raise ValueError("verifier witness ids must be unique")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["cue_judgments"],
        "properties": {
            "cue_judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["cue_id", "judgment", "witness_ids"],
                    "properties": {
                        "cue_id": {
                            "type": "string",
                            "enum": list(declared_cue_ids),
                        },
                        "judgment": {
                            "type": "string",
                            "enum": [item for item, _ in _SOFT_ORDINAL_MAP],
                        },
                        "witness_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(verifier_witness_ids),
                            },
                        },
                    },
                },
            }
        },
    }


def blind_soft_score_output_schema_procedure_digest() -> str:
    """Identify the provider-compatible dynamic schema construction rule."""

    return _canonical_sha256(
        {
            "schema": "gkm.bongard-blind-soft-output-schema-procedure.v2",
            "strict_dialect": "openai-responses-strict-json-schema-subset/v1",
            "root_fields": ["cue_judgments"],
            "judgment_fields": ["cue_id", "judgment", "witness_ids"],
            "dynamic_enums": ["declared_cue_ids", "verifier_witness_ids"],
            "array_cardinality_in_transport_schema": False,
            "array_uniqueness_in_transport_schema": False,
            "exact_cue_coverage_order_and_witness_ownership": (
                "python-decoder-fail-closed/v1"
            ),
            "forbidden_provider_keywords": [
                "oneOf",
                "uniqueItems",
                "minItems",
                "maxItems",
                "minimum",
                "maximum",
                "minLength",
                "maxLength",
                "const",
                "not",
            ],
        }
    )


def _soft_family_base_provenance(
    family: object, expected_family_digest: str
) -> Provenance:
    family_id = getattr(family, "family_id", "malformed-family")
    version = getattr(family, "version", "unknown")
    protocol_digest = getattr(family, "protocol_digest", "0" * 64)
    return Provenance(
        producer="bongard.soft_family_scorer",
        version="1",
        method="blind_ordinal_cues_frozen_min",
        input_digests=(expected_family_digest, str(protocol_digest)),
        artifact_digest=expected_family_digest,
        details=tuple(
            sorted(
                (
                    ("aggregation", _SOFT_AGGREGATION),
                    ("family_id", str(family_id)),
                    ("family_version", str(version)),
                    (
                        "identity_semantics",
                        _SOFT_FAMILY_IDENTITY_SEMANTICS,
                    ),
                    ("scorer_protocol_digest", str(protocol_digest)),
                )
            )
        ),
    )


def measure_blind_soft_score(
    family: SoftScorerFamily,
    record: BlindSoftScoreRecord,
    *,
    expected_family_digest: str,
    expected_record_digest: str | None = None,
) -> Evidence[float]:
    """Validate and expose the empirical score without making a truth claim."""

    try:
        _require_sha256("expected_family_digest", expected_family_digest)
    except (TypeError, ValueError):
        # Provenance itself still needs a non-empty identity; a malformed
        # caller commitment cannot be repaired or interpreted as nonmatch.
        fallback = "0" * 64
        return Evidence.error(
            _soft_family_base_provenance(family, fallback),
            "SoftPredicateIntegrityError",
            "expected family digest is malformed",
        )
    base = _soft_family_base_provenance(family, expected_family_digest)
    try:
        if not isinstance(family, SoftScorerFamily):
            raise TypeError("family must be a SoftScorerFamily")
        if not isinstance(record, BlindSoftScoreRecord):
            raise TypeError("record must be a BlindSoftScoreRecord")
        family.assert_untampered()
        record.assert_untampered()
        if family.digest() != expected_family_digest:
            raise SoftPredicateIntegrityError(
                "soft scorer family differs from the frozen policy digest"
            )
        if record.scorer_protocol_digest != family.protocol_digest:
            raise SoftPredicateIntegrityError(
                "blind score belongs to a different scorer protocol"
            )
        _check_expected_digest(
            "blind soft score", record.digest(), expected_record_digest
        )
        development = family.development_units
        leaked_identity = (
            record.task_id in {unit.task_id for unit in development}
            or record.panel_digest in {unit.panel_digest for unit in development}
            or record.proposer_call_id
            in {unit.proposer_call_id for unit in development}
            or record.scorer_call_id in {unit.scorer_call_id for unit in development}
        )
        if leaked_identity:
            raise SoftPredicateIntegrityError(
                "blind score overlaps a family-development task/panel/model call"
            )
    except (SoftPredicateError, TypeError, ValueError, AttributeError) as exc:
        return Evidence.error(base, type(exc).__name__, str(exc) or repr(exc))

    provenance = Provenance.composed(
        "bongard.soft_family_scorer",
        "1",
        "verified_blind_ordinal_min_measurement",
        (base,),
        details=tuple(
            sorted(
                (
                    ("claim_digest", record.claim_digest),
                    ("panel_digest", record.panel_digest),
                    ("record_digest", record.digest()),
                    ("scorer_call_id", record.scorer_call_id),
                    ("scorer_receipt_digest", record.scorer_receipt_digest),
                    ("task_id", record.task_id),
                    ("witness_packet_digest", record.witness_packet_digest),
                )
            )
        ),
    )
    if record.outcome != "present":
        error_type = (
            "SoftScorerTransportError"
            if record.outcome == "transport_error"
            else "SoftScorerParserError"
        )
        return Evidence.error(
            provenance,
            error_type,
            record.failure_reason or "blind scorer failed",
        )
    score = record.score
    if score is None:  # pragma: no cover - guarded by record invariants.
        return Evidence.error(
            provenance,
            "SoftScorerIntegrityError",
            "present blind scorer record has no score",
        )
    return Evidence.present(
        score,
        provenance,
        Uncertainty(score, score, causes=("closed_ordinal_min",)),
    )


def compare_blind_soft_score(
    family: SoftScorerFamily,
    record: BlindSoftScoreRecord,
    *,
    expected_family_digest: str,
    expected_record_digest: str | None = None,
) -> Evidence[bool]:
    """Apply the frozen family interval and positive boundary in Python.

    ``certified_absent`` here has one deliberately narrow meaning: the whole
    calibrated support interval lies below the preregistered positive
    boundary.  It is an operational family-scorer nonmatch certificate, not a
    VLM assertion and not proof that the semantic property is absent in the
    pixels.
    """

    measurement = measure_blind_soft_score(
        family,
        record,
        expected_family_digest=expected_family_digest,
        expected_record_digest=expected_record_digest,
    )
    if measurement.disposition is not Disposition.PRESENT:
        return Evidence(
            disposition=measurement.disposition,
            provenance=measurement.provenance,
            uncertainty=measurement.uncertainty,
            certificate=measurement.certificate,
            reason=measurement.reason,
            error_type=measurement.error_type,
        )
    score = measurement.unwrap()
    try:
        lower, upper, bin_index = family.calibrated_interval(score)
        boundary = _require_score(
            "family affirmative boundary", family.affirmative_boundary
        )
    except (SoftPredicateError, TypeError, ValueError, AttributeError) as exc:
        return Evidence.error(
            measurement.provenance, type(exc).__name__, str(exc) or repr(exc)
        )
    uncertainty = Uncertainty(
        lower,
        upper,
        confidence_level=float(family.confidence_level),
        causes=(
            "family_level_cluster_calibration",
            "fixed_score_bin",
        ),
    )
    provenance = Provenance.composed(
        "bongard.soft_family_comparator",
        "1",
        "interval_at_least_positive_boundary",
        (measurement.provenance,),
        details=tuple(
            sorted(
                (
                    ("affirmative_boundary", repr(boundary)),
                    ("family_digest", expected_family_digest),
                    ("operational_absence_only", "true"),
                    ("score_bin_index", str(bin_index)),
                )
            )
        ),
    )
    if lower >= boundary:
        return Evidence.present(True, provenance, uncertainty)
    if upper < boundary:
        return Evidence.certified_absent(
            provenance,
            (
                "operational family-calibrated nonmatch: calibrated support "
                f"upper={upper!r} is below affirmative boundary={boundary!r}; "
                "this is not semantic nonexistence and was not emitted by the model"
            ),
            uncertainty,
        )
    return Evidence.indeterminate(
        provenance,
        (
            "family-calibrated support interval straddles the affirmative "
            "boundary"
        ),
        uncertainty,
    )


# A descriptive alias for callers that begin from the task-local claim rather
# than from the intermediate score packet.
evaluate_family_soft_claim = compare_blind_soft_score


__all__ = (
    "BlindSoftScoreRecord",
    "CalibrationBand",
    "CalibrationDesign",
    "CalibrationError",
    "CalibrationObservation",
    "CalibratedPredictiveSupport",
    "DevelopmentUnit",
    "FrozenVisualScore",
    "MonotoneCalibrationArtifact",
    "ObservationRole",
    "PreregisteredCalibrationPlan",
    "RegisteredSoftPredicate",
    "SoftCueJudgment",
    "SoftFamilyDevelopmentUnit",
    "SoftPredicateClaim",
    "SoftPredicateError",
    "SoftPredicateIntegrityError",
    "SoftScorerFamily",
    "SoftScorerProtocol",
    "blind_soft_score_output_schema",
    "blind_soft_score_output_schema_procedure_digest",
    "compare_blind_soft_score",
    "evaluate_family_soft_claim",
    "fit_monotone_calibration",
    "measure_blind_soft_score",
    "register_soft_predicate",
)
