"""Prospective empirical calibration for typed panel-feature observations.

The panel-feature observer deliberately emits engineering-only observations.
This module is the narrow bridge from those observations to the calibration
grant vocabulary in :mod:`bongard.panel_soft_ontology`:

* the held-out panel/spec cases and their blinded labels are committed before
  observations are joined;
* every planned case must be present before a risk assessment is fitted;
* ``MATCH`` claims estimate false-positive risk and ``NONMATCH`` claims
  estimate false-negative risk;
* ``INDETERMINATE`` and ``ERROR`` remain visible attrition and are never
  counted as correct negatives;
* an exact, one-sided Clopper--Pearson binomial upper bound is computed on a
  discrete parts-per-million grid at the *preregistered* confidence; and
* inadequate claim counts, failed bounds, incomplete manifests, and missing
  inventory calibration remain typed gaps.

The output can contain :class:`PresenceCalibrationGrant` and
:class:`AbsenceCalibrationGrant` values, but it never creates a
:class:`FeatureCalibrationAuthority`, a trust root, or a scientific
projection.  As the ontology already states, parsing a grant supplies no
trust.  A separate external issuer must authenticate the source manifests,
annotation receipts, and calibration result before wrapping a grant in an
authority and pinning it through ``verify_feature_calibration_authority``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import comb
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.panel_feature_observation import (
    EngineeringFeatureDisposition,
    PanelFeatureObservationSet,
)
from bongard.panel_soft_ontology import (
    AbsenceCalibrationGrant,
    CalibrationAssessment,
    CalibrationCapability,
    CalibrationRisk,
    EnumerationResolution,
    FeatureDomain,
    PanelFeatureSpec,
    PresenceCalibrationGrant,
    RejectionKind,
)


FEATURE_CALIBRATION_CASE_SCHEMA = (
    "gkm.bongard-panel-feature-calibration-case.v1"
)
FEATURE_CALIBRATION_PLAN_SCHEMA = (
    "gkm.bongard-panel-feature-calibration-plan.v1"
)
FEATURE_CALIBRATION_MEASUREMENT_SCHEMA = (
    "gkm.bongard-panel-feature-calibration-measurement.v1"
)
FEATURE_CALIBRATION_GAP_SCHEMA = (
    "gkm.bongard-panel-feature-calibration-gap.v1"
)
FEATURE_CALIBRATION_OUTCOME_SCHEMA = (
    "gkm.bongard-panel-feature-calibration-outcome.v1"
)
FEATURE_CALIBRATION_ABSENCE_PREREQUISITES_SCHEMA = (
    "gkm.bongard-panel-feature-absence-calibration-prerequisites.v1"
)
FEATURE_CALIBRATION_GRANT_SET_SCHEMA = (
    "gkm.bongard-panel-feature-calibration-grant-set.v1"
)
FEATURE_CALIBRATION_LABEL_COMMITMENT_SCHEMA = (
    "gkm.bongard-panel-feature-calibration-label-commitment.v1"
)
FEATURE_CALIBRATION_POPULATION_SCHEMA = (
    "gkm.bongard-panel-feature-calibration-population.v1"
)
FEATURE_CALIBRATION_ASSESSMENT_RECEIPT_SCHEMA = (
    "gkm.bongard-panel-feature-calibration-assessment-receipt.v1"
)
FEATURE_CALIBRATION_GRANT_RECEIPT_SCHEMA = (
    "gkm.bongard-panel-feature-calibration-grant-receipt.v1"
)

FEATURE_CALIBRATION_SCORING_RULE_ID = (
    "bongard.panel-feature-calibration/"
    "conditional-claim-clopper-pearson-discrete-ppm-v1"
)
FEATURE_CALIBRATION_SAMPLING_RULE_ID = (
    "bongard.panel-feature-calibration/"
    "one-preregistered-spec-per-panel-one-independent-cluster-v1"
)

PPM = 1_000_000
MAX_HELD_OUT_CASES = 100_000

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[a-z][a-z0-9_.:-]{0,127}\Z")
_ALLOWED_CALIBRATION_SPLITS = frozenset({"train", "val"})


class PanelFeatureEmpiricalCalibrationError(ValueError):
    """A prospective plan, label join, score, or grant differs."""


class FeatureCalibrationTruth(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"


class FeatureCalibrationGapKind(str, Enum):
    INCOMPLETE_HELD_OUT_MANIFEST = "incomplete_held_out_manifest"
    INSUFFICIENT_DECISIVE_CLAIMS = "insufficient_decisive_claims"
    ERROR_BOUND_EXCEEDED = "error_bound_exceeded"
    MISSING_INVENTORY_COMPLETENESS_CALIBRATION = (
        "missing_inventory_completeness_calibration"
    )


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelFeatureEmpiricalCalibrationError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PanelFeatureEmpiricalCalibrationError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _code(value: object, label: str) -> str:
    if type(value) is not str or _CODE.fullmatch(value) is None:
        raise PanelFeatureEmpiricalCalibrationError(f"invalid {label}")
    return value


def _ppm(value: object, label: str, *, nonzero: bool = False) -> int:
    if type(value) is not int or not 0 <= value <= PPM:
        raise PanelFeatureEmpiricalCalibrationError(
            f"{label} must be an exact integer in [0, {PPM}]"
        )
    if nonzero and value == 0:
        raise PanelFeatureEmpiricalCalibrationError(f"{label} must be nonzero")
    return value


def feature_calibration_label_commitment(
    *,
    case_id: str,
    panel_digest: str,
    spec_digest: str,
    annotation_protocol_digest: str,
    truth: FeatureCalibrationTruth,
    label_nonce_digest: str,
) -> str:
    """Commit an externally supplied label without placing it in the plan."""

    _code(case_id, "calibration case ID")
    for label, item in (
        ("calibration panel digest", panel_digest),
        ("calibration spec digest", spec_digest),
        ("annotation protocol digest", annotation_protocol_digest),
        ("label nonce digest", label_nonce_digest),
    ):
        _digest(item, label)
    if type(truth) is not FeatureCalibrationTruth:
        raise TypeError("calibration truth must be FeatureCalibrationTruth")
    return canonical_digest(
        {
            "schema": FEATURE_CALIBRATION_LABEL_COMMITMENT_SCHEMA,
            "case_id": case_id,
            "panel_digest": panel_digest,
            "spec_digest": spec_digest,
            "annotation_protocol_digest": annotation_protocol_digest,
            "truth": truth.value,
            "label_nonce_digest": label_nonce_digest,
        }
    )


@dataclass(frozen=True, order=True, slots=True)
class HeldOutFeatureCalibrationCase:
    """One label-blinded panel/spec sampling unit frozen before observation."""

    case_id: str
    panel_digest: str
    spec_digest: str
    split: str
    dependence_cluster_id: str
    label_commitment_digest: str

    def __post_init__(self) -> None:
        _code(self.case_id, "calibration case ID")
        _digest(self.panel_digest, "calibration panel digest")
        _digest(self.spec_digest, "calibration spec digest")
        if self.split not in _ALLOWED_CALIBRATION_SPLITS:
            raise PanelFeatureEmpiricalCalibrationError(
                "held-out calibration split must be train or val, never test"
            )
        _code(self.dependence_cluster_id, "dependence cluster ID")
        _digest(self.label_commitment_digest, "label commitment digest")

    @property
    def case_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_CALIBRATION_CASE_SCHEMA,
            "case_id": self.case_id,
            "panel_digest": self.panel_digest,
            "spec_digest": self.spec_digest,
            "split": self.split,
            "dependence_cluster_id": self.dependence_cluster_id,
            "label_commitment_digest": self.label_commitment_digest,
            "label_revealed": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "HeldOutFeatureCalibrationCase":
        raw = _fields(
            value,
            {
                "schema",
                "case_id",
                "panel_digest",
                "spec_digest",
                "split",
                "dependence_cluster_id",
                "label_commitment_digest",
                "label_revealed",
            },
            "held-out feature calibration case",
        )
        if (
            raw["schema"] != FEATURE_CALIBRATION_CASE_SCHEMA
            or raw["label_revealed"] is not False
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "held-out calibration case policy differs"
            )
        result = cls(
            raw["case_id"],
            raw["panel_digest"],
            raw["spec_digest"],
            raw["split"],
            raw["dependence_cluster_id"],
            raw["label_commitment_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelFeatureEmpiricalCalibrationError(
                "held-out calibration case is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class FeatureObservationCalibrationPlan:
    """Prospective exact manifest and statistical acceptance policy."""

    domain: FeatureDomain
    observer_contract_digest: str
    observer_measurement_protocol_digest: str
    annotation_protocol_digest: str
    corpus_manifest_digest: str
    split_manifest_digest: str
    exposure_ledger_digest: str
    holdout_selection_receipt_digest: str
    cases: tuple[HeldOutFeatureCalibrationCase, ...]
    confidence_ppm: int
    accepted_false_positive_upper_ppm: int
    accepted_false_negative_upper_ppm: int
    minimum_presence_claim_count: int
    minimum_absence_claim_count: int
    valid_from_unix: int
    valid_through_unix: int

    def __post_init__(self) -> None:
        if type(self.domain) is not FeatureDomain:
            raise TypeError("calibration plan domain must be FeatureDomain")
        for label, item in (
            ("observer contract digest", self.observer_contract_digest),
            (
                "observer measurement protocol digest",
                self.observer_measurement_protocol_digest,
            ),
            ("annotation protocol digest", self.annotation_protocol_digest),
            ("corpus manifest digest", self.corpus_manifest_digest),
            ("split manifest digest", self.split_manifest_digest),
            ("exposure ledger digest", self.exposure_ledger_digest),
            (
                "holdout selection receipt digest",
                self.holdout_selection_receipt_digest,
            ),
        ):
            _digest(item, label)
        if (
            type(self.cases) is not tuple
            or not self.cases
            or len(self.cases) > MAX_HELD_OUT_CASES
            or any(type(item) is not HeldOutFeatureCalibrationCase for item in self.cases)
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "calibration plan needs a bounded non-empty exact case tuple"
            )
        case_ids = tuple(item.case_id for item in self.cases)
        if case_ids != tuple(sorted(case_ids)) or len(case_ids) != len(set(case_ids)):
            raise PanelFeatureEmpiricalCalibrationError(
                "calibration cases must be unique and sorted by case ID"
            )
        for label, values in (
            ("panel", tuple(item.panel_digest for item in self.cases)),
            (
                "dependence cluster",
                tuple(item.dependence_cluster_id for item in self.cases),
            ),
        ):
            if len(values) != len(set(values)):
                raise PanelFeatureEmpiricalCalibrationError(
                    f"calibration plan repeats a {label} sampling unit"
                )
        if any(
            not any(
                spec.spec_digest == item.spec_digest
                for spec in self.domain.admitted_specs
            )
            for item in self.cases
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "calibration case spec lies outside the frozen domain"
            )
        _ppm(self.confidence_ppm, "calibration confidence", nonzero=True)
        _ppm(
            self.accepted_false_positive_upper_ppm,
            "accepted false-positive bound",
        )
        _ppm(
            self.accepted_false_negative_upper_ppm,
            "accepted false-negative bound",
        )
        for label, count in (
            ("minimum presence claim count", self.minimum_presence_claim_count),
            ("minimum absence claim count", self.minimum_absence_claim_count),
        ):
            if (
                type(count) is not int
                or count <= 0
                or count > len(self.cases)
            ):
                raise PanelFeatureEmpiricalCalibrationError(
                    f"{label} must lie in [1, planned case count]"
                )
        if (
            type(self.valid_from_unix) is not int
            or type(self.valid_through_unix) is not int
            or self.valid_from_unix < 0
            or self.valid_through_unix < self.valid_from_unix
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "calibration validity interval differs"
            )

    @property
    def plan_digest(self) -> str:
        return canonical_digest(self.to_data())

    def case(self, case_id: str) -> HeldOutFeatureCalibrationCase:
        _code(case_id, "calibration case ID")
        for item in self.cases:
            if item.case_id == case_id:
                return item
        raise KeyError(f"case is absent from calibration plan: {case_id}")

    def spec(self, spec_digest: str) -> PanelFeatureSpec:
        _digest(spec_digest, "calibration spec digest")
        for item in self.domain.admitted_specs:
            if item.spec_digest == spec_digest:
                return item
        raise KeyError(f"spec is absent from calibration domain: {spec_digest}")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_CALIBRATION_PLAN_SCHEMA,
            "domain": self.domain.to_data(),
            "observer_contract_digest": self.observer_contract_digest,
            "observer_measurement_protocol_digest": (
                self.observer_measurement_protocol_digest
            ),
            "annotation_protocol_digest": self.annotation_protocol_digest,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "exposure_ledger_digest": self.exposure_ledger_digest,
            "holdout_selection_receipt_digest": (
                self.holdout_selection_receipt_digest
            ),
            "cases": [item.to_data() for item in self.cases],
            "confidence_ppm": self.confidence_ppm,
            "accepted_false_positive_upper_ppm": (
                self.accepted_false_positive_upper_ppm
            ),
            "accepted_false_negative_upper_ppm": (
                self.accepted_false_negative_upper_ppm
            ),
            "minimum_presence_claim_count": self.minimum_presence_claim_count,
            "minimum_absence_claim_count": self.minimum_absence_claim_count,
            "valid_from_unix": self.valid_from_unix,
            "valid_through_unix": self.valid_through_unix,
            "sampling_rule_id": FEATURE_CALIBRATION_SAMPLING_RULE_ID,
            "scoring_rule_id": FEATURE_CALIBRATION_SCORING_RULE_ID,
            "label_state": "commitment_only",
            "test_split_allowed": False,
            "authority_issued": False,
            "scientific_status": False,
        }

    @classmethod
    def from_data(cls, value: object) -> "FeatureObservationCalibrationPlan":
        raw = _fields(
            value,
            {
                "schema",
                "domain",
                "observer_contract_digest",
                "observer_measurement_protocol_digest",
                "annotation_protocol_digest",
                "corpus_manifest_digest",
                "split_manifest_digest",
                "exposure_ledger_digest",
                "holdout_selection_receipt_digest",
                "cases",
                "confidence_ppm",
                "accepted_false_positive_upper_ppm",
                "accepted_false_negative_upper_ppm",
                "minimum_presence_claim_count",
                "minimum_absence_claim_count",
                "valid_from_unix",
                "valid_through_unix",
                "sampling_rule_id",
                "scoring_rule_id",
                "label_state",
                "test_split_allowed",
                "authority_issued",
                "scientific_status",
            },
            "feature observation calibration plan",
        )
        if (
            raw["schema"] != FEATURE_CALIBRATION_PLAN_SCHEMA
            or raw["sampling_rule_id"] != FEATURE_CALIBRATION_SAMPLING_RULE_ID
            or raw["scoring_rule_id"] != FEATURE_CALIBRATION_SCORING_RULE_ID
            or raw["label_state"] != "commitment_only"
            or raw["test_split_allowed"] is not False
            or raw["authority_issued"] is not False
            or raw["scientific_status"] is not False
            or type(raw["cases"]) is not list
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "feature observation calibration plan policy differs"
            )
        result = cls(
            FeatureDomain.from_data(raw["domain"]),
            raw["observer_contract_digest"],
            raw["observer_measurement_protocol_digest"],
            raw["annotation_protocol_digest"],
            raw["corpus_manifest_digest"],
            raw["split_manifest_digest"],
            raw["exposure_ledger_digest"],
            raw["holdout_selection_receipt_digest"],
            tuple(HeldOutFeatureCalibrationCase.from_data(item) for item in raw["cases"]),
            raw["confidence_ppm"],
            raw["accepted_false_positive_upper_ppm"],
            raw["accepted_false_negative_upper_ppm"],
            raw["minimum_presence_claim_count"],
            raw["minimum_absence_claim_count"],
            raw["valid_from_unix"],
            raw["valid_through_unix"],
        )
        if result.to_data() != dict(raw):
            raise PanelFeatureEmpiricalCalibrationError(
                "feature observation calibration plan is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class HeldOutLabeledFeatureObservation:
    """Exact observation artifact joined to a committed external label."""

    plan_digest: str
    case_id: str
    observation_set: PanelFeatureObservationSet
    truth: FeatureCalibrationTruth
    label_nonce_digest: str
    annotation_receipt_digest: str
    label_commitment_digest: str

    def __post_init__(self) -> None:
        _digest(self.plan_digest, "calibration plan digest")
        _code(self.case_id, "calibration case ID")
        if type(self.observation_set) is not PanelFeatureObservationSet:
            raise TypeError(
                "held-out calibration measurement needs PanelFeatureObservationSet"
            )
        if type(self.truth) is not FeatureCalibrationTruth:
            raise TypeError("calibration truth must be FeatureCalibrationTruth")
        _digest(self.label_nonce_digest, "label nonce digest")
        _digest(self.annotation_receipt_digest, "annotation receipt digest")
        _digest(self.label_commitment_digest, "label commitment digest")

    @classmethod
    def create(
        cls,
        plan: FeatureObservationCalibrationPlan,
        *,
        case_id: str,
        observation_set: PanelFeatureObservationSet,
        truth: FeatureCalibrationTruth,
        label_nonce_digest: str,
        annotation_receipt_digest: str,
    ) -> "HeldOutLabeledFeatureObservation":
        if type(plan) is not FeatureObservationCalibrationPlan:
            raise TypeError("label join needs FeatureObservationCalibrationPlan")
        if type(observation_set) is not PanelFeatureObservationSet:
            raise TypeError("label join needs PanelFeatureObservationSet")
        case = plan.case(case_id)
        if (
            observation_set.panel_digest != case.panel_digest
            or observation_set.observer_contract_digest
            != plan.observer_contract_digest
            or observation_set.measurement_protocol_digest
            != plan.observer_measurement_protocol_digest
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "observation artifact differs from the preregistered case custody"
            )
        commitment = feature_calibration_label_commitment(
            case_id=case.case_id,
            panel_digest=case.panel_digest,
            spec_digest=case.spec_digest,
            annotation_protocol_digest=plan.annotation_protocol_digest,
            truth=truth,
            label_nonce_digest=label_nonce_digest,
        )
        if commitment != case.label_commitment_digest:
            raise PanelFeatureEmpiricalCalibrationError(
                "revealed label does not open the preregistered commitment"
            )
        return cls(
            plan.plan_digest,
            case.case_id,
            observation_set,
            truth,
            label_nonce_digest,
            annotation_receipt_digest,
            commitment,
        )

    @property
    def measurement_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_CALIBRATION_MEASUREMENT_SCHEMA,
            "plan_digest": self.plan_digest,
            "case_id": self.case_id,
            "observation_set": self.observation_set.to_data(),
            "observation_set_digest": self.observation_set.observation_set_digest,
            "truth": self.truth.value,
            "label_nonce_digest": self.label_nonce_digest,
            "annotation_receipt_digest": self.annotation_receipt_digest,
            "label_commitment_digest": self.label_commitment_digest,
            "causal_join": (
                "preregistered-case-and-label-commitment/"
                "then-observation-artifact/then-label-opening-v1"
            ),
            "label_source": "external_annotation_receipt",
            "model_generated_label": False,
        }

    @classmethod
    def from_data(
        cls,
        value: object,
        *,
        plan: FeatureObservationCalibrationPlan,
    ) -> "HeldOutLabeledFeatureObservation":
        raw = _fields(
            value,
            {
                "schema",
                "plan_digest",
                "case_id",
                "observation_set",
                "observation_set_digest",
                "truth",
                "label_nonce_digest",
                "annotation_receipt_digest",
                "label_commitment_digest",
                "causal_join",
                "label_source",
                "model_generated_label",
            },
            "held-out labeled feature observation",
        )
        if (
            raw["schema"] != FEATURE_CALIBRATION_MEASUREMENT_SCHEMA
            or raw["plan_digest"] != plan.plan_digest
            or raw["causal_join"]
            != (
                "preregistered-case-and-label-commitment/"
                "then-observation-artifact/then-label-opening-v1"
            )
            or raw["label_source"] != "external_annotation_receipt"
            or raw["model_generated_label"] is not False
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "held-out calibration measurement policy differs"
            )
        try:
            observation_set = PanelFeatureObservationSet.from_data(
                raw["observation_set"]
            )
            truth = FeatureCalibrationTruth(raw["truth"])
        except (TypeError, ValueError) as exc:
            raise PanelFeatureEmpiricalCalibrationError(
                "held-out calibration measurement value differs"
            ) from exc
        if raw["observation_set_digest"] != observation_set.observation_set_digest:
            raise PanelFeatureEmpiricalCalibrationError(
                "held-out observation-set digest differs"
            )
        result = cls.create(
            plan,
            case_id=raw["case_id"],
            observation_set=observation_set,
            truth=truth,
            label_nonce_digest=raw["label_nonce_digest"],
            annotation_receipt_digest=raw["annotation_receipt_digest"],
        )
        if (
            raw["label_commitment_digest"] != result.label_commitment_digest
            or result.to_data() != dict(raw)
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "held-out calibration measurement is not canonical"
            )
        return result


def _binomial_cdf_at_most(
    *, errors: int, trials: int, probability_ppm: int, alpha_ppm: int
) -> bool:
    """Compare Binomial(n, p) CDF(errors) <= alpha using exact integers."""

    p = probability_ppm
    q = PPM - p
    denominator = PPM**trials
    if errors <= trials // 2:
        cdf_numerator = sum(
            comb(trials, index) * p**index * q ** (trials - index)
            for index in range(errors + 1)
        )
        return cdf_numerator * PPM <= alpha_ppm * denominator
    survival_numerator = sum(
        comb(trials, index) * p**index * q ** (trials - index)
        for index in range(errors + 1, trials + 1)
    )
    # CDF <= alpha iff survival >= 1 - alpha.
    return survival_numerator * PPM >= (PPM - alpha_ppm) * denominator


def one_sided_binomial_error_upper_ppm(
    *, errors: int, trials: int, confidence_ppm: int
) -> int:
    """Return the conservative exact upper confidence endpoint on a PPM grid.

    This is the smallest integer ``u`` such that the binomial CDF through the
    observed error count at ``p=u/1e6`` is at most ``1-confidence``.  Rounding
    therefore goes outward (up), never toward a more favorable bound.
    """

    if (
        type(errors) is not int
        or type(trials) is not int
        or trials <= 0
        or errors < 0
        or errors > trials
    ):
        raise PanelFeatureEmpiricalCalibrationError(
            "binomial errors/trials must satisfy 0 <= errors <= trials and trials > 0"
        )
    _ppm(confidence_ppm, "binomial confidence", nonzero=True)
    if errors == trials:
        return PPM
    alpha = PPM - confidence_ppm
    low = 0
    high = PPM
    while low < high:
        middle = (low + high) // 2
        if _binomial_cdf_at_most(
            errors=errors,
            trials=trials,
            probability_ppm=middle,
            alpha_ppm=alpha,
        ):
            high = middle
        else:
            low = middle + 1
    return low


@dataclass(frozen=True, order=True, slots=True)
class FeatureCalibrationGap:
    """Typed, content-addressed reason no assessment or grant was produced."""

    kind: FeatureCalibrationGapKind
    risk: CalibrationRisk | None
    required_count: int
    observed_count: int
    missing_case_ids: tuple[str, ...] = ()
    accepted_error_upper_ppm: int | None = None
    assessed_error_upper_ppm: int | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not FeatureCalibrationGapKind:
            raise TypeError("calibration gap kind differs")
        if self.risk is not None and type(self.risk) is not CalibrationRisk:
            raise TypeError("calibration gap risk differs")
        for label, value in (
            ("required calibration count", self.required_count),
            ("observed calibration count", self.observed_count),
        ):
            if type(value) is not int or value < 0:
                raise PanelFeatureEmpiricalCalibrationError(
                    f"{label} must be a non-negative exact int"
                )
        if (
            type(self.missing_case_ids) is not tuple
            or any(type(item) is not str for item in self.missing_case_ids)
            or self.missing_case_ids != tuple(sorted(self.missing_case_ids))
            or len(self.missing_case_ids) != len(set(self.missing_case_ids))
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "calibration gap missing case IDs must be unique and sorted"
            )
        if self.kind is FeatureCalibrationGapKind.INCOMPLETE_HELD_OUT_MANIFEST:
            if (
                self.risk is not None
                or not self.missing_case_ids
                or self.observed_count >= self.required_count
                or self.accepted_error_upper_ppm is not None
                or self.assessed_error_upper_ppm is not None
            ):
                raise PanelFeatureEmpiricalCalibrationError(
                    "incomplete-manifest calibration gap fields differ"
                )
        elif self.kind is FeatureCalibrationGapKind.INSUFFICIENT_DECISIVE_CLAIMS:
            if (
                self.risk not in {
                    CalibrationRisk.FALSE_POSITIVE_CLAIM,
                    CalibrationRisk.FALSE_NEGATIVE_CLAIM,
                }
                or self.missing_case_ids
                or self.observed_count >= self.required_count
                or self.accepted_error_upper_ppm is not None
                or self.assessed_error_upper_ppm is not None
            ):
                raise PanelFeatureEmpiricalCalibrationError(
                    "insufficient-claims calibration gap fields differ"
                )
        elif self.kind is FeatureCalibrationGapKind.ERROR_BOUND_EXCEEDED:
            if (
                self.risk not in {
                    CalibrationRisk.FALSE_POSITIVE_CLAIM,
                    CalibrationRisk.FALSE_NEGATIVE_CLAIM,
                }
                or self.missing_case_ids
                or self.accepted_error_upper_ppm is None
                or self.assessed_error_upper_ppm is None
                or self.assessed_error_upper_ppm <= self.accepted_error_upper_ppm
            ):
                raise PanelFeatureEmpiricalCalibrationError(
                    "error-bound calibration gap fields differ"
                )
            _ppm(self.accepted_error_upper_ppm, "gap accepted error")
            _ppm(self.assessed_error_upper_ppm, "gap assessed error")
        else:
            if (
                self.risk is not CalibrationRisk.OWNER_INVENTORY_OMISSION
                or self.required_count != 1
                or self.observed_count != 0
                or self.missing_case_ids
                or self.accepted_error_upper_ppm is not None
                or self.assessed_error_upper_ppm is not None
            ):
                raise PanelFeatureEmpiricalCalibrationError(
                    "missing-inventory calibration gap fields differ"
                )

    @property
    def gap_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_CALIBRATION_GAP_SCHEMA,
            "kind": self.kind.value,
            "risk": None if self.risk is None else self.risk.value,
            "required_count": self.required_count,
            "observed_count": self.observed_count,
            "missing_case_ids": list(self.missing_case_ids),
            "accepted_error_upper_ppm": self.accepted_error_upper_ppm,
            "assessed_error_upper_ppm": self.assessed_error_upper_ppm,
        }

    @classmethod
    def from_data(cls, value: object) -> "FeatureCalibrationGap":
        raw = _fields(
            value,
            {
                "schema",
                "kind",
                "risk",
                "required_count",
                "observed_count",
                "missing_case_ids",
                "accepted_error_upper_ppm",
                "assessed_error_upper_ppm",
            },
            "feature calibration gap",
        )
        if (
            raw["schema"] != FEATURE_CALIBRATION_GAP_SCHEMA
            or type(raw["missing_case_ids"]) is not list
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "feature calibration gap schema differs"
            )
        try:
            result = cls(
                FeatureCalibrationGapKind(raw["kind"]),
                None if raw["risk"] is None else CalibrationRisk(raw["risk"]),
                raw["required_count"],
                raw["observed_count"],
                tuple(raw["missing_case_ids"]),
                raw["accepted_error_upper_ppm"],
                raw["assessed_error_upper_ppm"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelFeatureEmpiricalCalibrationError):
                raise
            raise PanelFeatureEmpiricalCalibrationError(
                "feature calibration gap value differs"
            ) from exc
        if result.to_data() != dict(raw):
            raise PanelFeatureEmpiricalCalibrationError(
                "feature calibration gap is not canonical"
            )
        return result


def _assessment(
    plan: FeatureObservationCalibrationPlan,
    *,
    population_digest: str,
    risk: CalibrationRisk,
    claim_count: int,
    error_count: int,
) -> tuple[CalibrationAssessment | None, FeatureCalibrationGap | None]:
    if risk is CalibrationRisk.FALSE_POSITIVE_CLAIM:
        minimum = plan.minimum_presence_claim_count
        accepted = plan.accepted_false_positive_upper_ppm
    elif risk is CalibrationRisk.FALSE_NEGATIVE_CLAIM:
        minimum = plan.minimum_absence_claim_count
        accepted = plan.accepted_false_negative_upper_ppm
    else:  # pragma: no cover - internal guard.
        raise PanelFeatureEmpiricalCalibrationError(
            "claim assessment received a non-claim risk"
        )
    if claim_count < minimum:
        return None, FeatureCalibrationGap(
            FeatureCalibrationGapKind.INSUFFICIENT_DECISIVE_CLAIMS,
            risk,
            minimum,
            claim_count,
        )
    upper = one_sided_binomial_error_upper_ppm(
        errors=error_count,
        trials=claim_count,
        confidence_ppm=plan.confidence_ppm,
    )
    if upper > accepted:
        return None, FeatureCalibrationGap(
            FeatureCalibrationGapKind.ERROR_BOUND_EXCEEDED,
            risk,
            minimum,
            claim_count,
            accepted_error_upper_ppm=accepted,
            assessed_error_upper_ppm=upper,
        )
    assessment_receipt = canonical_digest(
        {
            "schema": FEATURE_CALIBRATION_ASSESSMENT_RECEIPT_SCHEMA,
            "scoring_rule_id": FEATURE_CALIBRATION_SCORING_RULE_ID,
            "plan_digest": plan.plan_digest,
            "population_digest": population_digest,
            "risk": risk.value,
            "claim_count": claim_count,
            "error_count": error_count,
            "confidence_ppm": plan.confidence_ppm,
            "accepted_error_upper_ppm": accepted,
            "assessed_error_upper_ppm": upper,
            "receipt_kind": "deterministic_empirical_derivation",
            "external_signature_supplied": False,
        }
    )
    return (
        CalibrationAssessment(
            risk,
            population_digest,
            plan.annotation_protocol_digest,
            claim_count,
            accepted,
            upper,
            plan.confidence_ppm,
            plan.valid_from_unix,
            plan.valid_through_unix,
            assessment_receipt,
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class EmpiricalFeatureCalibrationOutcome:
    """Complete score or typed insufficiency over one exact prospective plan."""

    plan_digest: str
    population_digest: str
    measurement_case_ids: tuple[str, ...]
    measurement_digests: tuple[str, ...]
    missing_case_ids: tuple[str, ...]
    match_claim_count: int
    false_positive_count: int
    nonmatch_claim_count: int
    false_negative_count: int
    indeterminate_count: int
    error_count: int
    presence_assessment: CalibrationAssessment | None
    absence_claim_assessment: CalibrationAssessment | None
    gaps: tuple[FeatureCalibrationGap, ...]

    def __post_init__(self) -> None:
        _digest(self.plan_digest, "outcome plan digest")
        _digest(self.population_digest, "calibration population digest")
        if (
            type(self.measurement_case_ids) is not tuple
            or self.measurement_case_ids != tuple(sorted(self.measurement_case_ids))
            or len(self.measurement_case_ids) != len(set(self.measurement_case_ids))
            or type(self.measurement_digests) is not tuple
            or len(self.measurement_case_ids) != len(self.measurement_digests)
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "outcome measurement bindings differ"
            )
        for item in self.measurement_digests:
            _digest(item, "outcome measurement digest")
        if (
            type(self.missing_case_ids) is not tuple
            or self.missing_case_ids != tuple(sorted(self.missing_case_ids))
            or len(self.missing_case_ids) != len(set(self.missing_case_ids))
            or set(self.missing_case_ids) & set(self.measurement_case_ids)
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "outcome missing case IDs differ"
            )
        for label, count in (
            ("match claim count", self.match_claim_count),
            ("false-positive count", self.false_positive_count),
            ("nonmatch claim count", self.nonmatch_claim_count),
            ("false-negative count", self.false_negative_count),
            ("indeterminate count", self.indeterminate_count),
            ("error count", self.error_count),
        ):
            if type(count) is not int or count < 0:
                raise PanelFeatureEmpiricalCalibrationError(
                    f"{label} must be a non-negative exact int"
                )
        if self.false_positive_count > self.match_claim_count:
            raise PanelFeatureEmpiricalCalibrationError(
                "false positives exceed presence claims"
            )
        if self.false_negative_count > self.nonmatch_claim_count:
            raise PanelFeatureEmpiricalCalibrationError(
                "false negatives exceed absence claims"
            )
        if (
            self.match_claim_count
            + self.nonmatch_claim_count
            + self.indeterminate_count
            + self.error_count
            != len(self.measurement_case_ids)
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "outcome disposition counts do not cover submitted measurements"
            )
        if self.presence_assessment is not None and (
            type(self.presence_assessment) is not CalibrationAssessment
            or self.presence_assessment.risk
            is not CalibrationRisk.FALSE_POSITIVE_CLAIM
            or self.presence_assessment.calibration_population_digest
            != self.population_digest
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "outcome presence assessment differs"
            )
        if self.absence_claim_assessment is not None and (
            type(self.absence_claim_assessment) is not CalibrationAssessment
            or self.absence_claim_assessment.risk
            is not CalibrationRisk.FALSE_NEGATIVE_CLAIM
            or self.absence_claim_assessment.calibration_population_digest
            != self.population_digest
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "outcome absence assessment differs"
            )
        if (
            type(self.gaps) is not tuple
            or any(type(item) is not FeatureCalibrationGap for item in self.gaps)
            or self.gaps != tuple(sorted(self.gaps, key=lambda item: item.gap_digest))
            or len({item.gap_digest for item in self.gaps}) != len(self.gaps)
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "outcome gaps must be unique and digest-sorted"
            )
        incomplete = any(
            item.kind is FeatureCalibrationGapKind.INCOMPLETE_HELD_OUT_MANIFEST
            for item in self.gaps
        )
        if incomplete != bool(self.missing_case_ids):
            raise PanelFeatureEmpiricalCalibrationError(
                "outcome incomplete-manifest gap differs from missing cases"
            )
        if incomplete and (
            self.presence_assessment is not None
            or self.absence_claim_assessment is not None
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "partial held-out manifests cannot produce assessments"
            )

    @property
    def outcome_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_CALIBRATION_OUTCOME_SCHEMA,
            "plan_digest": self.plan_digest,
            "population_digest": self.population_digest,
            "measurement_bindings": [
                {"case_id": case_id, "measurement_digest": digest}
                for case_id, digest in zip(
                    self.measurement_case_ids,
                    self.measurement_digests,
                    strict=True,
                )
            ],
            "missing_case_ids": list(self.missing_case_ids),
            "disposition_counts": {
                "match_claims": self.match_claim_count,
                "false_positives": self.false_positive_count,
                "nonmatch_claims": self.nonmatch_claim_count,
                "false_negatives": self.false_negative_count,
                "indeterminate": self.indeterminate_count,
                "error": self.error_count,
            },
            "presence_assessment": (
                None
                if self.presence_assessment is None
                else self.presence_assessment.to_data()
            ),
            "absence_claim_assessment": (
                None
                if self.absence_claim_assessment is None
                else self.absence_claim_assessment.to_data()
            ),
            "gaps": [item.to_data() for item in self.gaps],
            "scoring_rule_id": FEATURE_CALIBRATION_SCORING_RULE_ID,
            "indeterminate_counted_as_negative": False,
            "error_counted_as_negative": False,
            "authority_issued": False,
            "scientific_projection_authorized": False,
        }


def score_feature_observation_calibration(
    plan: FeatureObservationCalibrationPlan,
    measurements: Sequence[HeldOutLabeledFeatureObservation],
) -> EmpiricalFeatureCalibrationOutcome:
    """Score exact joined artifacts; an incomplete manifest returns a gap."""

    if type(plan) is not FeatureObservationCalibrationPlan:
        raise TypeError("calibration scoring needs FeatureObservationCalibrationPlan")
    if isinstance(measurements, (str, bytes)) or not isinstance(measurements, Sequence):
        raise TypeError("calibration measurements must be a sequence")
    if any(type(item) is not HeldOutLabeledFeatureObservation for item in measurements):
        raise TypeError(
            "calibration measurements must contain HeldOutLabeledFeatureObservation"
        )
    by_case: dict[str, HeldOutLabeledFeatureObservation] = {}
    for item in measurements:
        if item.plan_digest != plan.plan_digest:
            raise PanelFeatureEmpiricalCalibrationError(
                "calibration measurement belongs to a different plan"
            )
        try:
            case = plan.case(item.case_id)
        except KeyError as exc:
            raise PanelFeatureEmpiricalCalibrationError(
                "calibration measurement case is not preregistered"
            ) from exc
        if item.case_id in by_case:
            raise PanelFeatureEmpiricalCalibrationError(
                "calibration measurements repeat a planned case"
            )
        # Recreate the causal join.  This also verifies the observation and
        # external-label commitment custody rather than trusting dataclass shape.
        replayed = HeldOutLabeledFeatureObservation.create(
            plan,
            case_id=case.case_id,
            observation_set=item.observation_set,
            truth=item.truth,
            label_nonce_digest=item.label_nonce_digest,
            annotation_receipt_digest=item.annotation_receipt_digest,
        )
        if replayed.to_data() != item.to_data():
            raise PanelFeatureEmpiricalCalibrationError(
                "calibration measurement does not replay exactly"
            )
        by_case[item.case_id] = item
    ordered = tuple(by_case[key] for key in sorted(by_case))
    submitted_ids = tuple(item.case_id for item in ordered)
    planned_ids = tuple(item.case_id for item in plan.cases)
    missing = tuple(item for item in planned_ids if item not in by_case)
    population_digest = canonical_digest(
        {
            "schema": FEATURE_CALIBRATION_POPULATION_SCHEMA,
            "plan_digest": plan.plan_digest,
            "measurement_bindings": [
                {
                    "case_id": item.case_id,
                    "measurement_digest": item.measurement_digest,
                }
                for item in ordered
            ],
            "missing_case_ids": list(missing),
            "exact_planned_manifest_complete": not missing,
        }
    )
    match_claims = false_positives = 0
    nonmatch_claims = false_negatives = 0
    indeterminate = errors = 0
    for item in ordered:
        case = plan.case(item.case_id)
        disposition = item.observation_set.evaluate(plan.spec(case.spec_digest))
        if disposition is EngineeringFeatureDisposition.MATCH:
            match_claims += 1
            false_positives += item.truth is FeatureCalibrationTruth.ABSENT
        elif disposition is EngineeringFeatureDisposition.NONMATCH:
            nonmatch_claims += 1
            false_negatives += item.truth is FeatureCalibrationTruth.PRESENT
        elif disposition is EngineeringFeatureDisposition.INDETERMINATE:
            indeterminate += 1
        else:
            errors += 1
    gaps: list[FeatureCalibrationGap] = []
    presence_assessment: CalibrationAssessment | None = None
    absence_assessment: CalibrationAssessment | None = None
    if missing:
        gaps.append(
            FeatureCalibrationGap(
                FeatureCalibrationGapKind.INCOMPLETE_HELD_OUT_MANIFEST,
                None,
                len(plan.cases),
                len(ordered),
                missing,
            )
        )
    else:
        presence_assessment, presence_gap = _assessment(
            plan,
            population_digest=population_digest,
            risk=CalibrationRisk.FALSE_POSITIVE_CLAIM,
            claim_count=match_claims,
            error_count=false_positives,
        )
        absence_assessment, absence_gap = _assessment(
            plan,
            population_digest=population_digest,
            risk=CalibrationRisk.FALSE_NEGATIVE_CLAIM,
            claim_count=nonmatch_claims,
            error_count=false_negatives,
        )
        gaps.extend(
            item for item in (presence_gap, absence_gap) if item is not None
        )
    return EmpiricalFeatureCalibrationOutcome(
        plan.plan_digest,
        population_digest,
        submitted_ids,
        tuple(item.measurement_digest for item in ordered),
        missing,
        match_claims,
        false_positives,
        nonmatch_claims,
        false_negatives,
        indeterminate,
        errors,
        presence_assessment,
        absence_assessment,
        tuple(sorted(gaps, key=lambda item: item.gap_digest)),
    )


def cold_replay_feature_observation_calibration(
    plan: FeatureObservationCalibrationPlan,
    measurements: Sequence[HeldOutLabeledFeatureObservation],
    archived_outcome: EmpiricalFeatureCalibrationOutcome,
) -> EmpiricalFeatureCalibrationOutcome:
    """Recompute a score without vision/model calls and require exact equality."""

    if type(archived_outcome) is not EmpiricalFeatureCalibrationOutcome:
        raise TypeError("cold replay needs EmpiricalFeatureCalibrationOutcome")
    replayed = score_feature_observation_calibration(plan, measurements)
    if replayed.to_data() != archived_outcome.to_data():
        raise PanelFeatureEmpiricalCalibrationError(
            "archived empirical calibration outcome differs from cold replay"
        )
    return replayed


@dataclass(frozen=True, slots=True)
class AbsenceCalibrationPrerequisites:
    """Externally calibrated inventory/search inputs required by absence grants."""

    domain_digest: str
    observer_contract_digest: str
    owner_enumeration_protocol_digest: str
    search_protocol_digest: str
    inventory_completeness_assessment: CalibrationAssessment
    allowed_resolution: EnumerationResolution
    allowed_rejection_kinds: tuple[RejectionKind, ...]
    inventory_calibration_receipt_digest: str

    def __post_init__(self) -> None:
        for label, item in (
            ("absence prerequisite domain digest", self.domain_digest),
            (
                "absence prerequisite observer contract digest",
                self.observer_contract_digest,
            ),
            (
                "absence prerequisite enumeration protocol digest",
                self.owner_enumeration_protocol_digest,
            ),
            ("absence prerequisite search protocol digest", self.search_protocol_digest),
            (
                "absence prerequisite inventory calibration receipt",
                self.inventory_calibration_receipt_digest,
            ),
        ):
            _digest(item, label)
        if (
            type(self.inventory_completeness_assessment) is not CalibrationAssessment
            or self.inventory_completeness_assessment.risk
            is not CalibrationRisk.OWNER_INVENTORY_OMISSION
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "absence prerequisites need an external owner-inventory assessment"
            )
        if type(self.allowed_resolution) is not EnumerationResolution:
            raise TypeError("absence prerequisite resolution differs")
        if (
            type(self.allowed_rejection_kinds) is not tuple
            or not self.allowed_rejection_kinds
            or any(type(item) is not RejectionKind for item in self.allowed_rejection_kinds)
        ):
            raise TypeError("absence prerequisite rejection kinds differ")
        values = tuple(item.value for item in self.allowed_rejection_kinds)
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise PanelFeatureEmpiricalCalibrationError(
                "absence prerequisite rejection kinds must be unique and sorted"
            )

    @property
    def prerequisites_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_CALIBRATION_ABSENCE_PREREQUISITES_SCHEMA,
            "domain_digest": self.domain_digest,
            "observer_contract_digest": self.observer_contract_digest,
            "owner_enumeration_protocol_digest": (
                self.owner_enumeration_protocol_digest
            ),
            "search_protocol_digest": self.search_protocol_digest,
            "inventory_completeness_assessment": (
                self.inventory_completeness_assessment.to_data()
            ),
            "allowed_resolution": self.allowed_resolution.value,
            "allowed_rejection_kinds": [
                item.value for item in self.allowed_rejection_kinds
            ],
            "inventory_calibration_receipt_digest": (
                self.inventory_calibration_receipt_digest
            ),
            "source": "external_inventory_calibration",
        }


@dataclass(frozen=True, slots=True)
class EmpiricalFeatureCalibrationGrantSet:
    """Empirical grants plus gaps; deliberately not a calibration authority."""

    plan_digest: str
    outcome_digest: str
    presence_grant: PresenceCalibrationGrant | None
    absence_grant: AbsenceCalibrationGrant | None
    gaps: tuple[FeatureCalibrationGap, ...]

    def __post_init__(self) -> None:
        _digest(self.plan_digest, "grant-set plan digest")
        _digest(self.outcome_digest, "grant-set outcome digest")
        if self.presence_grant is not None and type(
            self.presence_grant
        ) is not PresenceCalibrationGrant:
            raise TypeError("grant-set presence grant differs")
        if self.absence_grant is not None and type(
            self.absence_grant
        ) is not AbsenceCalibrationGrant:
            raise TypeError("grant-set absence grant differs")
        if (
            type(self.gaps) is not tuple
            or any(type(item) is not FeatureCalibrationGap for item in self.gaps)
            or self.gaps != tuple(sorted(self.gaps, key=lambda item: item.gap_digest))
            or len({item.gap_digest for item in self.gaps}) != len(self.gaps)
        ):
            raise PanelFeatureEmpiricalCalibrationError(
                "grant-set gaps must be unique and digest-sorted"
            )

    @property
    def grant_set_digest(self) -> str:
        return canonical_digest(self.to_data())

    def to_data(self) -> dict[str, object]:
        return {
            "schema": FEATURE_CALIBRATION_GRANT_SET_SCHEMA,
            "plan_digest": self.plan_digest,
            "outcome_digest": self.outcome_digest,
            "presence_grant": (
                None if self.presence_grant is None else self.presence_grant.to_data()
            ),
            "absence_grant": (
                None if self.absence_grant is None else self.absence_grant.to_data()
            ),
            "gaps": [item.to_data() for item in self.gaps],
            "feature_calibration_authority_issued": False,
            "trust_root_created": False,
            "external_authority_verification_required": True,
            "scientific_projection_authorized": False,
        }


def _grant_receipt(
    *,
    plan: FeatureObservationCalibrationPlan,
    outcome: EmpiricalFeatureCalibrationOutcome,
    capability: CalibrationCapability,
    assessment_digests: tuple[str, ...],
    absence_prerequisites_digest: str | None,
) -> str:
    return canonical_digest(
        {
            "schema": FEATURE_CALIBRATION_GRANT_RECEIPT_SCHEMA,
            "plan_digest": plan.plan_digest,
            "outcome_digest": outcome.outcome_digest,
            "capability": capability.value,
            "assessment_digests": list(assessment_digests),
            "absence_prerequisites_digest": absence_prerequisites_digest,
            "receipt_kind": "deterministic_empirical_grant_derivation",
            "feature_calibration_authority_issued": False,
            "external_signature_supplied": False,
        }
    )


def derive_empirical_feature_calibration_grants(
    plan: FeatureObservationCalibrationPlan,
    outcome: EmpiricalFeatureCalibrationOutcome,
    *,
    absence_prerequisites: AbsenceCalibrationPrerequisites | None = None,
) -> EmpiricalFeatureCalibrationGrantSet:
    """Derive grant values while leaving authority issuance external."""

    if type(plan) is not FeatureObservationCalibrationPlan:
        raise TypeError("grant derivation needs FeatureObservationCalibrationPlan")
    if type(outcome) is not EmpiricalFeatureCalibrationOutcome:
        raise TypeError("grant derivation needs EmpiricalFeatureCalibrationOutcome")
    if outcome.plan_digest != plan.plan_digest:
        raise PanelFeatureEmpiricalCalibrationError(
            "grant outcome belongs to a different calibration plan"
        )
    gaps = list(outcome.gaps)
    presence_grant: PresenceCalibrationGrant | None = None
    absence_grant: AbsenceCalibrationGrant | None = None
    if outcome.presence_assessment is not None:
        receipt = _grant_receipt(
            plan=plan,
            outcome=outcome,
            capability=CalibrationCapability.PRESENCE,
            assessment_digests=(
                outcome.presence_assessment.assessment_digest,
            ),
            absence_prerequisites_digest=None,
        )
        presence_grant = PresenceCalibrationGrant(
            plan.domain,
            plan.observer_contract_digest,
            plan.observer_measurement_protocol_digest,
            outcome.presence_assessment,
            receipt,
        )
    if outcome.absence_claim_assessment is not None:
        if absence_prerequisites is None:
            gaps.append(
                FeatureCalibrationGap(
                    FeatureCalibrationGapKind.MISSING_INVENTORY_COMPLETENESS_CALIBRATION,
                    CalibrationRisk.OWNER_INVENTORY_OMISSION,
                    1,
                    0,
                )
            )
        else:
            if type(absence_prerequisites) is not AbsenceCalibrationPrerequisites:
                raise TypeError(
                    "absence prerequisites must be AbsenceCalibrationPrerequisites"
                )
            inventory_assessment = (
                absence_prerequisites.inventory_completeness_assessment
            )
            if (
                absence_prerequisites.domain_digest != plan.domain.domain_digest
                or absence_prerequisites.observer_contract_digest
                != plan.observer_contract_digest
                or inventory_assessment.confidence_ppm < plan.confidence_ppm
                or inventory_assessment.valid_from_unix > plan.valid_from_unix
                or inventory_assessment.valid_through_unix < plan.valid_through_unix
            ):
                raise PanelFeatureEmpiricalCalibrationError(
                    "external inventory calibration does not cover the planned grant"
                )
            receipt = _grant_receipt(
                plan=plan,
                outcome=outcome,
                capability=CalibrationCapability.ABSENCE,
                assessment_digests=(
                    outcome.absence_claim_assessment.assessment_digest,
                    inventory_assessment.assessment_digest,
                ),
                absence_prerequisites_digest=(
                    absence_prerequisites.prerequisites_digest
                ),
            )
            absence_grant = AbsenceCalibrationGrant(
                plan.domain,
                plan.observer_contract_digest,
                plan.observer_measurement_protocol_digest,
                absence_prerequisites.owner_enumeration_protocol_digest,
                absence_prerequisites.search_protocol_digest,
                outcome.absence_claim_assessment,
                inventory_assessment,
                absence_prerequisites.allowed_resolution,
                absence_prerequisites.allowed_rejection_kinds,
                receipt,
            )
    return EmpiricalFeatureCalibrationGrantSet(
        plan.plan_digest,
        outcome.outcome_digest,
        presence_grant,
        absence_grant,
        tuple(
            sorted(
                {item.gap_digest: item for item in gaps}.values(),
                key=lambda item: item.gap_digest,
            )
        ),
    )


__all__ = [
    "AbsenceCalibrationPrerequisites",
    "EmpiricalFeatureCalibrationGrantSet",
    "EmpiricalFeatureCalibrationOutcome",
    "FeatureCalibrationGap",
    "FeatureCalibrationGapKind",
    "FeatureCalibrationTruth",
    "FeatureObservationCalibrationPlan",
    "HeldOutFeatureCalibrationCase",
    "HeldOutLabeledFeatureObservation",
    "PanelFeatureEmpiricalCalibrationError",
    "cold_replay_feature_observation_calibration",
    "derive_empirical_feature_calibration_grants",
    "feature_calibration_label_commitment",
    "one_sided_binomial_error_upper_ppm",
    "score_feature_observation_calibration",
]
