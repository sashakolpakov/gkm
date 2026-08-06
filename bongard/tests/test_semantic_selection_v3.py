from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import random

import pytest

from bongard.artifacts import canonical_digest
from bongard.exposure import ExposureLedger, semantic_resolver_policy_digest
from bongard.historical_exposure import load_historical_exposure
from bongard.semantic_calibration_campaign import (
    CAMPAIGN_SELECTION_ALGORITHM,
    CAMPAIGN_SELECTION_ALGORITHM_V2,
    CAMPAIGN_SELECTION_ALGORITHM_V3,
    SemanticCalibrationCampaignError,
    SemanticCalibrationCapacityError,
    SemanticCalibrationTaskSelectionV3,
    _V3SelectionGroup,
    _clean_cohort_whitelist,
    _select_from_whitelist_v3,
    _v3_collision_tokens,
    _v3_exact_maximum_group_indices,
    replay_semantic_calibration_tasks_v3,
    select_semantic_calibration_tasks_v3,
    semantic_generator_cluster_id,
)
from bongard.cohorts import parse_official_task_id
from bongard.tests.test_semantic_calibration_campaign import (
    SOURCE_MANIFEST,
    _metadata_only_corpus,
)


def _official_group(task_id: str) -> _V3SelectionGroup:
    parsed = parse_official_task_id(task_id)
    return _V3SelectionGroup(
        family=parsed.family,
        cluster_id=semantic_generator_cluster_id(
            parsed.family, parsed.concepts
        ),
        collision_tokens=_v3_collision_tokens(parsed),
        members=(parsed,),
    )


def _a3_pair_vs_singleton_groups() -> tuple[_V3SelectionGroup, ...]:
    # These are the two exact A3 composite groups that arrived before and
    # blocked their four singleton competitors under the v2 greedy order.
    return tuple(
        _official_group(task_id)
        for task_id in (
            "bd_thin_asymmetric_arrow-transposed_lamp_0000",
            "bd_thin_asymmetric_arrow_0000",
            "bd_transposed_lamp_0000",
            "bd_inverse_sector_arc-open_band_half_circles_0000",
            "bd_inverse_sector_arc_0000",
            "bd_open_band_half_circles_0000",
        )
    )


def test_a3_pair_before_singletons_has_exact_capacity_four_for_every_seed() -> None:
    groups = _a3_pair_vs_singleton_groups()
    singleton_ids = {
        group.cluster_id
        for group in groups
        if len(group.collision_tokens) == 1
    }
    assert len(singleton_ids) == 4

    for seed in ("0" * 64, "1" * 64, "f" * 64, "a3-frozen-seed"):
        chosen = _v3_exact_maximum_group_indices(groups, seed=seed)
        assert len(chosen) == 4
        assert {groups[index].cluster_id for index in chosen} == singleton_ids


def test_v3_exact_capacity_and_tie_result_are_input_order_invariant() -> None:
    groups = _a3_pair_vs_singleton_groups()
    seed = "input-order-invariance"
    baseline = {
        groups[index].cluster_id
        for index in _v3_exact_maximum_group_indices(groups, seed=seed)
    }
    reversed_groups = tuple(reversed(groups))
    reversed_result = {
        reversed_groups[index].cluster_id
        for index in _v3_exact_maximum_group_indices(
            reversed_groups, seed=seed
        )
    }
    assert reversed_result == baseline


def test_v3_exact_optimizer_matches_random_small_brute_force() -> None:
    rng = random.Random(20260806)
    for case in range(80):
        token_count = rng.randint(1, 7)
        groups = tuple(
            _V3SelectionGroup(
                family="bd",
                cluster_id=f"synthetic-{case:03d}-{index:02d}",
                collision_tokens=frozenset(
                    rng.sample(
                        [f"token-{item}" for item in range(token_count)],
                        rng.randint(1, min(3, token_count)),
                    )
                ),
                members=(),
            )
            for index in range(rng.randint(1, 10))
        )
        exact = _v3_exact_maximum_group_indices(
            groups, seed=f"random-case-{case}"
        )
        assert all(
            groups[left].collision_tokens.isdisjoint(
                groups[right].collision_tokens
            )
            for left, right in combinations(exact, 2)
        )
        brute_capacity = max(
            len(indices)
            for mask in range(1 << len(groups))
            if all(
                groups[left].collision_tokens.isdisjoint(
                    groups[right].collision_tokens
                )
                for left, right in combinations(
                    tuple(
                        index
                        for index in range(len(groups))
                        if mask & (1 << index)
                    ),
                    2,
                )
            )
            for indices in (
                tuple(
                    index
                    for index in range(len(groups))
                    if mask & (1 << index)
                ),
            )
        )
        assert len(exact) == brute_capacity


def test_v3_rank_two_solver_handles_official_scale_singleton_cover() -> None:
    # The complete clean corpus has a 164-group BD conflict component.  Its
    # singleton morphology groups are a tight token-count upper bound; the
    # exact solver must exploit that certificate instead of enumerating 2^164
    # group subsets.
    singletons = tuple(
        _V3SelectionGroup(
            family="bd",
            cluster_id=f"singleton-{index:03d}",
            collision_tokens=frozenset({f"token-{index:03d}"}),
            members=(),
        )
        for index in range(102)
    )
    composites = tuple(
        _V3SelectionGroup(
            family="bd",
            cluster_id=f"composite-{index:03d}",
            collision_tokens=frozenset(
                {
                    f"token-{index:03d}",
                    f"token-{(index + 1) % 102:03d}",
                }
            ),
            members=(),
        )
        for index in range(102)
    )
    groups = singletons + composites
    chosen = _v3_exact_maximum_group_indices(
        groups, seed="official-scale-singleton-cover"
    )
    assert len(chosen) == 102
    assert {groups[index].cluster_id for index in chosen} == {
        group.cluster_id for group in singletons
    }


def test_v3_rank_two_matching_tie_is_seeded_and_input_order_invariant() -> None:
    groups = tuple(
        _V3SelectionGroup(
            family="hd",
            cluster_id=f"cycle-edge-{left}-{right}",
            collision_tokens=frozenset({left, right}),
            members=(),
        )
        for left, right in (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))
    )
    seed = "rank-two-cycle-tie"
    baseline = {
        groups[index].cluster_id
        for index in _v3_exact_maximum_group_indices(groups, seed=seed)
    }
    reversed_groups = tuple(reversed(groups))
    replay = {
        reversed_groups[index].cluster_id
        for index in _v3_exact_maximum_group_indices(
            reversed_groups, seed=seed
        )
    }
    assert len(baseline) == 2
    assert replay == baseline


def test_v3_hd_constituents_are_prefiltered_before_optimization() -> None:
    historical = load_historical_exposure()
    resolver = semantic_resolver_policy_digest(historical)
    predecessor = ExposureLedger.create("sha256:" + "6" * 64).record(
        phase="semantic-calibration",
        actor="fixture",
        purpose="adversarial prior HD constituent disclosure",
        task_ids=("hd_has_seven_straight_lines-exist_triangle_0000",),
        observed_at="2026-08-06T11:00:00Z",
    )
    whitelist = tuple(
        (
            task_id,
            "hd",
            parse_official_task_id(task_id).concepts,
            "train",
        )
        for task_id in (
            "hd_has_seven_straight_lines-exist_quadrangle_0000",
            "hd_exist_regular-exist_triangle_0000",
            "hd_exist_regular-exist_quadrangle_0000",
        )
    )
    selected, certificate = _select_from_whitelist_v3(
        whitelist,
        families=("hd",),
        candidate_count=1,
        seed="constituent-prefilter",
        exposure_ledger=predecessor,
        historical_seed_digest=historical.seed_digest,
        resolver_policy_digest=resolver,
    )
    assert tuple(item.task_id for item in selected) == (
        "hd_exist_regular-exist_quadrangle_0000",
    )
    assert certificate.eligible_task_count == 1
    assert certificate.eligible_group_count == 1
    assert certificate.maximum_capacity == 1
    assert certificate.predecessor_token_ineligible_task_ids_digest == (
        "sha256:"
        + canonical_digest(
            [
                "hd_exist_regular-exist_triangle_0000",
                "hd_has_seven_straight_lines-exist_quadrangle_0000",
            ]
        )
    )


def test_v3_fails_c_plus_one_with_the_exact_capacity_certificate() -> None:
    groups = _a3_pair_vs_singleton_groups()
    historical = load_historical_exposure()
    resolver = semantic_resolver_policy_digest(historical)
    predecessor = ExposureLedger.create("sha256:" + "7" * 64)
    whitelist = tuple(
        (
            group.members[0].task_id,
            "bd",
            group.members[0].concepts,
            "train",
        )
        for group in groups
    )
    with pytest.raises(SemanticCalibrationCapacityError) as failure:
        _select_from_whitelist_v3(
            whitelist,
            families=("bd",),
            candidate_count=5,
            seed="capacity-failure",
            exposure_ledger=predecessor,
            historical_seed_digest=historical.seed_digest,
            resolver_policy_digest=resolver,
        )
    assert failure.value.requested_count == 5
    assert failure.value.certificate.maximum_capacity == 4
    assert "exact v3 capacity is 4" in str(failure.value)
    assert failure.value.certificate.digest in str(failure.value)


def test_v3_selection_archive_roundtrip_cold_replay_and_tamper_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _metadata_only_corpus()
    predecessor = ExposureLedger.create(SOURCE_MANIFEST)
    monkeypatch.setattr(
        "bongard.corpus.BongardTask.build_manifest",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("v3 metadata selection opened panel bytes")
        ),
    )
    selection = select_semantic_calibration_tasks_v3(
        corpus,
        candidate_count=4,
        seed="v3-public-selection-seed",
        exposure_ledger=predecessor,
        expected_exposure_ledger_digest=predecessor.digest,
    )
    assert CAMPAIGN_SELECTION_ALGORITHM == CAMPAIGN_SELECTION_ALGORITHM_V2
    assert selection.to_data()["selection_algorithm"] == (
        CAMPAIGN_SELECTION_ALGORITHM_V3
    )
    assert SemanticCalibrationTaskSelectionV3.from_data(
        selection.to_data()
    ) == selection
    assert replay_semantic_calibration_tasks_v3(
        selection.to_data(), corpus=corpus, exposure_ledger=predecessor
    ) == selection

    forged = deepcopy(selection.to_data())
    forged_certificate = forged["capacity_certificate"]
    forged_certificate["conflict_graph_digest"] = "sha256:" + "0" * 64
    forged_certificate["certificate_digest"] = "sha256:" + canonical_digest(
        {
            key: value
            for key, value in forged_certificate.items()
            if key != "certificate_digest"
        }
    )
    forged["capacity_certificate_digest"] = forged_certificate[
        "certificate_digest"
    ]
    forged["selection_digest"] = "sha256:" + canonical_digest(
        {
            key: value
            for key, value in forged.items()
            if key != "selection_digest"
        }
    )
    structurally_valid_forgery = SemanticCalibrationTaskSelectionV3.from_data(
        forged
    )
    with pytest.raises(
        SemanticCalibrationCampaignError,
        match="differs from cold corpus/ledger replay",
    ):
        replay_semantic_calibration_tasks_v3(
            structurally_valid_forgery,
            corpus=corpus,
            exposure_ledger=predecessor,
        )


def test_v3_whitelist_replay_is_input_order_invariant() -> None:
    corpus = _metadata_only_corpus()
    predecessor = ExposureLedger.create(SOURCE_MANIFEST)
    (
        whitelist,
        historical_seed_digest,
        resolver_policy_digest,
        *_rest,
    ) = _clean_cohort_whitelist(corpus, ("bd", "hd"), "drill")
    first = _select_from_whitelist_v3(
        whitelist,
        families=("bd", "hd"),
        candidate_count=4,
        seed="whitelist-order",
        exposure_ledger=predecessor,
        historical_seed_digest=historical_seed_digest,
        resolver_policy_digest=resolver_policy_digest,
    )
    second = _select_from_whitelist_v3(
        tuple(reversed(whitelist)),
        families=("bd", "hd"),
        candidate_count=4,
        seed="whitelist-order",
        exposure_ledger=predecessor,
        historical_seed_digest=historical_seed_digest,
        resolver_policy_digest=resolver_policy_digest,
    )
    assert first == second
