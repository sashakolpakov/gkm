"""Exact familywise binomial bounds for frozen cluster-level decisions.

This module is deliberately small and dependency-free.  It is used only when
the score thresholds and the finite hypothesis family were committed before
the certification observations were made.  Each calibration cluster is first
reduced to one Boolean error (normally ``any(panel_errors)``); the function
below then computes a one-sided Clopper--Pearson upper bound with a Bonferroni
familywise allocation.

The calculation never uses binary floating point.  It searches the one-part-
per-million output grid and compares the binomial CDF to the allocated alpha
using exact Python integers.  The returned grid point is therefore rounded
outward: it is the smallest ppm value whose exact CDF is at most alpha.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest


PPM_SCALE = 1_000_000
FIXED_THRESHOLD_CLUSTER_BOUND_SCHEMA = (
    "gkm.bongard-fixed-threshold-cluster-bound.v1"
)
FIXED_THRESHOLD_CLUSTER_ALGORITHM_ID = (
    "bongard.soft-vision/frozen-threshold-cluster-clopper-pearson-"
    "bonferroni-exact-integer-ppm-v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ClusterBinomialError(ValueError):
    """A bound request or archived exact bound is malformed."""


def _integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        suffix = "" if maximum is None else f" and at most {maximum}"
        raise ClusterBinomialError(
            f"{label} must be an integer of at least {minimum}{suffix}"
        )
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ClusterBinomialError(f"{label} must be a lowercase sha256")
    return value


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def fixed_threshold_cluster_algorithm_digest() -> str:
    """Bind the pure-Python authority and its exact numerical convention."""

    return canonical_digest(
        {
            "algorithm_id": FIXED_THRESHOLD_CLUSTER_ALGORITHM_ID,
            "source_digest": _source_digest(),
            "cluster_reduction": "one-error-iff-any-deployed-unit-errors",
            "confidence_allocation": "bonferroni-one-sided-alpha/H",
            "bound": "clopper-pearson-binomial-upper",
            "numeric_domain": "exact-integer-binomial-cdf-on-ppm-grid",
            "rounding": "smallest-grid-value-with-cdf-at-most-alpha",
            "threshold_selection": "must-be-frozen-before-certification-data",
            "python_is_authority": True,
            "lean_required": False,
            "secondary_checker_affects_decision": False,
        }
    )


def _binomial_cdf_numerator(
    *, cluster_count: int, error_cluster_count: int, probability_ppm: int
) -> int:
    """Return ``PPM_SCALE**n * P[X <= errors]`` exactly."""

    n = cluster_count
    errors = error_cluster_count
    success = probability_ppm
    failure = PPM_SCALE - success
    return sum(
        math.comb(n, index)
        * pow(success, index)
        * pow(failure, n - index)
        for index in range(errors + 1)
    )


def _cdf_is_at_most_allocated_alpha(
    *,
    cluster_count: int,
    error_cluster_count: int,
    probability_ppm: int,
    confidence_level_ppm: int,
    hypothesis_count: int,
) -> bool:
    # cdf / M**n <= ((M-confidence)/M) / H
    numerator = _binomial_cdf_numerator(
        cluster_count=cluster_count,
        error_cluster_count=error_cluster_count,
        probability_ppm=probability_ppm,
    )
    return (
        numerator * PPM_SCALE * hypothesis_count
        <= (PPM_SCALE - confidence_level_ppm)
        * pow(PPM_SCALE, cluster_count)
    )


def familywise_clopper_pearson_upper_ppm(
    *,
    cluster_count: int,
    error_cluster_count: int,
    confidence_level_ppm: int,
    hypothesis_count: int,
) -> int:
    """Return an exact outward one-sided familywise upper risk bound.

    ``hypothesis_count`` is the complete, precommitted set of decision
    directions that can be deployed.  Calling this after selecting a smaller
    successful subset is invalid at the protocol level; callers bind that
    inventory in their calibration plan.
    """

    n = _integer(cluster_count, "cluster_count", minimum=1)
    errors = _integer(
        error_cluster_count,
        "error_cluster_count",
        minimum=0,
        maximum=n,
    )
    confidence = _integer(
        confidence_level_ppm,
        "confidence_level_ppm",
        minimum=1,
        maximum=PPM_SCALE - 1,
    )
    hypotheses = _integer(hypothesis_count, "hypothesis_count", minimum=1)
    if errors == n:
        return PPM_SCALE

    low = 0
    high = PPM_SCALE
    if not _cdf_is_at_most_allocated_alpha(
        cluster_count=n,
        error_cluster_count=errors,
        probability_ppm=high,
        confidence_level_ppm=confidence,
        hypothesis_count=hypotheses,
    ):
        raise ClusterBinomialError("binomial upper-bound bracket is invalid")

    while low + 1 < high:
        midpoint = (low + high) // 2
        if _cdf_is_at_most_allocated_alpha(
            cluster_count=n,
            error_cluster_count=errors,
            probability_ppm=midpoint,
            confidence_level_ppm=confidence,
            hypothesis_count=hypotheses,
        ):
            high = midpoint
        else:
            low = midpoint

    return high


@dataclass(frozen=True, slots=True)
class FixedThresholdClusterBound:
    """Canonical receipt for one predeclared tag/direction risk bound."""

    tag_id: str
    direction: str
    cluster_count: int
    error_cluster_count: int
    empirical_error_ppm: int
    conditional_error_upper_ppm: int
    confidence_level_ppm: int
    hypothesis_count: int
    calibration_plan_digest: str
    observation_set_digest: str
    algorithm_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.tag_id, str) or not self.tag_id:
            raise ClusterBinomialError("tag_id must be non-empty text")
        if self.direction not in {"present", "absent"}:
            raise ClusterBinomialError("direction must be present or absent")
        n = _integer(self.cluster_count, "cluster_count", minimum=1)
        errors = _integer(
            self.error_cluster_count,
            "error_cluster_count",
            maximum=n,
        )
        expected_empirical = (errors * PPM_SCALE + n - 1) // n
        if self.empirical_error_ppm != expected_empirical:
            raise ClusterBinomialError("empirical error ppm differs")
        expected_upper = familywise_clopper_pearson_upper_ppm(
            cluster_count=n,
            error_cluster_count=errors,
            confidence_level_ppm=self.confidence_level_ppm,
            hypothesis_count=self.hypothesis_count,
        )
        if self.conditional_error_upper_ppm != expected_upper:
            raise ClusterBinomialError("conditional upper bound differs")
        for name in (
            "calibration_plan_digest",
            "observation_set_digest",
            "algorithm_digest",
            "record_digest",
        ):
            _digest(getattr(self, name), name)
        if self.algorithm_digest != fixed_threshold_cluster_algorithm_digest():
            raise ClusterBinomialError("bound algorithm identity drifted")
        if self.record_digest != canonical_digest(self._preimage()):
            raise ClusterBinomialError("cluster bound digest differs")

    def _preimage(self) -> dict[str, object]:
        return {
            "schema": FIXED_THRESHOLD_CLUSTER_BOUND_SCHEMA,
            "tag_id": self.tag_id,
            "direction": self.direction,
            "cluster_count": self.cluster_count,
            "error_cluster_count": self.error_cluster_count,
            "empirical_error_ppm": self.empirical_error_ppm,
            "conditional_error_upper_ppm": self.conditional_error_upper_ppm,
            "confidence_level_ppm": self.confidence_level_ppm,
            "hypothesis_count": self.hypothesis_count,
            "calibration_plan_digest": self.calibration_plan_digest,
            "observation_set_digest": self.observation_set_digest,
            "algorithm_digest": self.algorithm_digest,
            "predicate_execution_authority": "pure-python",
            "python_is_authority": True,
            "lean_required": False,
            "secondary_checker_affects_decision": False,
        }

    @classmethod
    def create(
        cls,
        *,
        tag_id: str,
        direction: str,
        cluster_count: int,
        error_cluster_count: int,
        confidence_level_ppm: int,
        hypothesis_count: int,
        calibration_plan_digest: str,
        observation_set_digest: str,
    ) -> "FixedThresholdClusterBound":
        n = _integer(cluster_count, "cluster_count", minimum=1)
        errors = _integer(
            error_cluster_count, "error_cluster_count", maximum=n
        )
        values: dict[str, object] = {
            "tag_id": tag_id,
            "direction": direction,
            "cluster_count": n,
            "error_cluster_count": errors,
            "empirical_error_ppm": (
                errors * PPM_SCALE + n - 1
            )
            // n,
            "conditional_error_upper_ppm": (
                familywise_clopper_pearson_upper_ppm(
                    cluster_count=n,
                    error_cluster_count=errors,
                    confidence_level_ppm=confidence_level_ppm,
                    hypothesis_count=hypothesis_count,
                )
            ),
            "confidence_level_ppm": confidence_level_ppm,
            "hypothesis_count": hypothesis_count,
            "calibration_plan_digest": _digest(
                calibration_plan_digest, "calibration_plan_digest"
            ),
            "observation_set_digest": _digest(
                observation_set_digest, "observation_set_digest"
            ),
            "algorithm_digest": fixed_threshold_cluster_algorithm_digest(),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=canonical_digest(provisional._preimage()),
        )

    def to_data(self) -> dict[str, object]:
        return {**self._preimage(), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: Mapping[str, Any]
    ) -> "FixedThresholdClusterBound":
        expected = {
            "schema",
            "tag_id",
            "direction",
            "cluster_count",
            "error_cluster_count",
            "empirical_error_ppm",
            "conditional_error_upper_ppm",
            "confidence_level_ppm",
            "hypothesis_count",
            "calibration_plan_digest",
            "observation_set_digest",
            "algorithm_digest",
            "predicate_execution_authority",
            "python_is_authority",
            "lean_required",
            "secondary_checker_affects_decision",
            "record_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ClusterBinomialError("cluster bound fields differ")
        if (
            value["schema"] != FIXED_THRESHOLD_CLUSTER_BOUND_SCHEMA
            or value["predicate_execution_authority"] != "pure-python"
            or value["python_is_authority"] is not True
            or value["lean_required"] is not False
            or value["secondary_checker_affects_decision"] is not False
        ):
            raise ClusterBinomialError("cluster bound authority differs")
        result = cls(
            tag_id=value["tag_id"],
            direction=value["direction"],
            cluster_count=value["cluster_count"],
            error_cluster_count=value["error_cluster_count"],
            empirical_error_ppm=value["empirical_error_ppm"],
            conditional_error_upper_ppm=value[
                "conditional_error_upper_ppm"
            ],
            confidence_level_ppm=value["confidence_level_ppm"],
            hypothesis_count=value["hypothesis_count"],
            calibration_plan_digest=value["calibration_plan_digest"],
            observation_set_digest=value["observation_set_digest"],
            algorithm_digest=value["algorithm_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise ClusterBinomialError("cluster bound is not canonical")
        return result


__all__ = [
    "FIXED_THRESHOLD_CLUSTER_ALGORITHM_ID",
    "FIXED_THRESHOLD_CLUSTER_BOUND_SCHEMA",
    "ClusterBinomialError",
    "FixedThresholdClusterBound",
    "familywise_clopper_pearson_upper_ppm",
    "fixed_threshold_cluster_algorithm_digest",
]
