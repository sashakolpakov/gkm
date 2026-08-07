from __future__ import annotations

from dataclasses import replace

import pytest

from bongard.cluster_binomial import (
    ClusterBinomialError,
    FixedThresholdClusterBound,
    familywise_clopper_pearson_upper_ppm,
)


def test_zero_error_bounds_are_exact_outward_ppm_grid_values() -> None:
    assert familywise_clopper_pearson_upper_ppm(
        cluster_count=15,
        error_cluster_count=0,
        confidence_level_ppm=950_000,
        hypothesis_count=2,
    ) == 218_020
    assert familywise_clopper_pearson_upper_ppm(
        cluster_count=15,
        error_cluster_count=0,
        confidence_level_ppm=950_000,
        hypothesis_count=6,
    ) == 273_246


def test_bound_is_monotone_and_all_error_case_is_one() -> None:
    values = tuple(
        familywise_clopper_pearson_upper_ppm(
            cluster_count=20,
            error_cluster_count=errors,
            confidence_level_ppm=950_000,
            hypothesis_count=4,
        )
        for errors in range(21)
    )
    assert values == tuple(sorted(values))
    assert values[-1] == 1_000_000
    assert familywise_clopper_pearson_upper_ppm(
        cluster_count=20,
        error_cluster_count=0,
        confidence_level_ppm=950_000,
        hypothesis_count=8,
    ) > values[0]


def test_canonical_bound_round_trip_and_tamper_rejection() -> None:
    bound = FixedThresholdClusterBound.create(
        tag_id="component.geometry.has_obtuse_angle",
        direction="present",
        cluster_count=15,
        error_cluster_count=0,
        confidence_level_ppm=950_000,
        hypothesis_count=2,
        calibration_plan_digest="1" * 64,
        observation_set_digest="2" * 64,
    )
    assert bound.conditional_error_upper_ppm == 218_020
    assert FixedThresholdClusterBound.from_data(bound.to_data()) == bound
    with pytest.raises(ClusterBinomialError, match="conditional upper"):
        replace(bound, conditional_error_upper_ppm=bound.conditional_error_upper_ppm - 1)

    malformed = bound.to_data()
    malformed["lean_required"] = True
    with pytest.raises(ClusterBinomialError, match="authority"):
        FixedThresholdClusterBound.from_data(malformed)

    malformed = bound.to_data()
    malformed["secondary_checker_affects_decision"] = True
    with pytest.raises(ClusterBinomialError, match="authority"):
        FixedThresholdClusterBound.from_data(malformed)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"cluster_count": True},
        {"cluster_count": 0},
        {"error_cluster_count": -1},
        {"error_cluster_count": 16},
        {"confidence_level_ppm": 1_000_000},
        {"hypothesis_count": 0},
    ),
)
def test_invalid_requests_fail_closed(kwargs: dict[str, int | bool]) -> None:
    values: dict[str, int | bool] = {
        "cluster_count": 15,
        "error_cluster_count": 0,
        "confidence_level_ppm": 950_000,
        "hypothesis_count": 2,
    }
    values.update(kwargs)
    with pytest.raises(ClusterBinomialError):
        familywise_clopper_pearson_upper_ppm(**values)  # type: ignore[arg-type]
