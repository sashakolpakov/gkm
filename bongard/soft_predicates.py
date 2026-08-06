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
from typing import NoReturn, Sequence

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


__all__ = (
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
    "SoftPredicateClaim",
    "SoftPredicateError",
    "SoftPredicateIntegrityError",
    "fit_monotone_calibration",
    "register_soft_predicate",
)
