from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from bongard.canonical import canonical_digest
from bongard import panel_action_count_synthetic_benchmark as subject
from bongard import panel_action_count_synthetic_identifiability as synthetic


EXPECTED_SOURCE_SHA256 = "876ac52c7b74afcb9cf6be657655ed50ce5438af194bf446b4b76797b8c1ea0b"
EXPECTED_RESULT_RECORD_DIGEST = (
    "sha256:4a2221b9b39a22ee0b60b2b3dd0ac5859c0b15de92e16a6c238cb6a5aaf774f3"
)


@pytest.fixture(scope="module")
def paired_result() -> dict[str, object]:
    # This is the one deliberately slow integration check.  It really extracts
    # all 112 pooled-control features and runs the ordered observer on the same
    # 432 held-out PNG rows; no model or observer is patched.
    return subject.run_paired_synthetic_benchmark()


def test_real_all_pair_carrier_disjoint_benchmark_is_paired_and_non_authorizing(
    paired_result: dict[str, object],
) -> None:
    result = paired_result
    assert subject.source_sha256() == EXPECTED_SOURCE_SHA256
    assert result["record_digest"] == EXPECTED_RESULT_RECORD_DIGEST
    assert result["schema"] == subject.SCHEMA
    assert result["schema"].endswith(".v3")
    body = dict(result)
    assert body.pop("record_digest") == "sha256:" + canonical_digest(body)
    assert result["authorization"] == {
        "benchmark_promotion": False,
        "calibration_target_query_authorized": False,
        "new_campaign_authority_created": False,
        "official_benchmark_or_generalization_claim_authorized": False,
        "official_data_inputs_authorized": False,
        "synthetic_only": True,
    }
    assert result["training"]["row_count"] == 648
    assert result["training"]["unique_png_count"] == 641
    assert result["training"]["deduplicated_fit_row_count"] == 641
    assert result["evaluation"]["row_count"] == 432
    assert result["evaluation"]["unique_png_count"] == 425
    assert result["training"]["carrier_families"] == [
        "lattice", "perimeter", "pinwheel",
    ]
    assert result["evaluation"]["carrier_families"] == ["radial", "staggered"]
    assert len(result["paired_rows"]) == 432
    assert len({row["png_sha256"] for row in result["paired_rows"]}) == 425
    assert {
        tuple(row["canonical_visible_pair"]) for row in result["paired_rows"]
    } == {pair.as_tuple() for pair in synthetic.valid_count_pairs()}
    assert all(
        set(row)
        >= {
            "canonical_visible_pair",
            "control_pair",
            "ordered_candidate_pairs",
            "png_sha256",
        }
        for row in result["paired_rows"]
    )
    collision = result["balanced_corpus_collision_audit"]
    assert collision["candidate_count"] == 1080
    assert collision["exact_canonical_conflict_count"] == 0
    assert collision["exact_collision_class_count"] == 14
    assert collision["qualifying_near_collision_count"] == 20
    assert collision["retained_near_collision_count"] == 12
    assert collision["max_retained_near_collisions"] == 12
    assert result["gates"]["balanced_corpus_has_no_exact_target_conflict"] is True
    assert result["target"]["partial_target"] is True
    assert result["target"][
        "connected_component_without_exact_singleton_normal_form"
    ] == "unresolved_and_ordered_observer_must_return_set_or_gap"
    assert result["ordered_observer"]["normal_form_role"] == (
        "post_fit_target_resolvability_gate_not_pair_selection"
    )
    assert result["ordered_observer"]["pair_selection_uses_target"] is False
    assert result["ordered_observer"][
        "unresolved_singleton_suppression_uses_target"
    ] is True
    assert result["ordered_observer"][
        "structural_set_or_gap_is_independent_observer_evidence"
    ] is False


def test_real_result_exposes_lift_without_generalization_authority(
    paired_result: dict[str, object],
) -> None:
    result = paired_result
    metrics = result["metrics"]
    ordered = metrics["ordered"]
    control = metrics["control"]
    assert ordered["denominator"] == control["denominator"] == 432
    assert sum(metrics["ordered_disposition_counts"].values()) == 432
    assert metrics["ordered_disposition_counts"] == {
        "AMBIGUOUS": 0,
        "ERROR": 0,
        "GAP": 0,
        "IDENTIFIED": 432,
    }
    assert ordered["joint_singleton_accuracy"] == 432 / 432
    assert control["joint_singleton_accuracy"] == 189 / 432
    assert ordered["joint_singleton_accuracy"] > control["joint_singleton_accuracy"]
    assert metrics["paired_joint_singleton_accuracy_delta"] == 432 / 432 - 189 / 432
    resampling = result["family_resampling_sensitivity"]
    assert resampling["resampling_unit"] == "held_out_carrier_family"
    assert resampling["carrier_cluster_count"] == 2
    assert resampling["ordered_joint_accuracy_family_resampling_p05"] == 1.0
    assert resampling["paired_delta_family_resampling_p05"] == 101 / 216
    assert resampling["paired_delta_family_resampling_p05"] > 0.0
    assert result["gates"]["paired_delta_descriptive_p05_positive"] is True
    # There is a paired synthetic lift in both observed families, but two
    # clusters do not establish an inferential confidence bound or any
    # official/generalization claim.
    assert resampling["inferential_confidence_bound_claimed"] is False
    assert result["gates"]["ordered_joint_accuracy"] is True
    assert result["gates"]["historical_panel_anchor_descriptive_p05_exceeded"] is True
    assert result["gates"]["passed"] is True
    assert result["gates"]["historically_unseen_pair_macro_accuracy"] == (
        ordered["historically_unseen_pair_macro_singleton_accuracy"]
        >= subject.GATE_THRESHOLDS[
            "historically_unseen_pair_macro_accuracy_at_least"
        ]
    )
    historical = result["historical_comparison"]
    assert historical["comparison_kind"] == (
        "cross_corpus_sanity_anchor_not_paired_baseline"
    )
    assert historical["panel_joint_anchor_head"] == (
        "historical_separate_marginal_head_oof"
    )
    assert historical["selected_direct_pair_head_comparison_claimed"] is False


def test_real_result_reports_each_held_out_family_without_pooling_it_away(
    paired_result: dict[str, object],
) -> None:
    by_family = paired_result["metrics"]["by_carrier_family"]
    assert set(by_family) == {"radial", "staggered"}
    assert all(
        by_family[family][observer]["denominator"] == 216
        for family in by_family
        for observer in ("control", "ordered")
    )
    assert by_family["radial"]["ordered"]["joint_singleton_accuracy"] == 216 / 216
    assert by_family["radial"]["control"]["joint_singleton_accuracy"] == 115 / 216
    assert by_family["staggered"]["ordered"]["joint_singleton_accuracy"] == 216 / 216
    assert by_family["staggered"]["control"]["joint_singleton_accuracy"] == 74 / 216
    assert by_family["radial"]["ordered"]["joint_singleton_accuracy"] > (
        by_family["radial"]["control"]["joint_singleton_accuracy"]
    )
    assert by_family["staggered"]["ordered"]["joint_singleton_accuracy"] > (
        by_family["staggered"]["control"]["joint_singleton_accuracy"]
    )


def test_exact_history_collisions_are_scored_against_visible_target(
    paired_result: dict[str, object],
) -> None:
    audit = paired_result["counterfactual_identifiability_audit"]
    assert audit["sample_count"] == 22
    assert audit["exact_png_class_count"] == 9
    assert audit["exact_case_count"] == 10
    assert audit["exact_case_safe_outcome_count"] == 10
    assert audit["exact_canonical_conflict_count"] == 0
    assert audit["exact_declared_history_oracle_accuracy"] == 11 / 21
    assert audit["resolved_target_case_count"] == 9
    assert audit["unresolved_target_case_count"] == 2
    assert audit["false_visible_singleton_count"] == 0
    assert audit["identical_png_prediction_inconsistency_count"] == 0
    by_counterfactual = {row["case_id"]: row for row in audit["rows"]}
    for case_id in (
        "endpoint-branch-alias-with-touching-context",
        "line-chain-arc-alias-with-touching-context",
    ):
        row = by_counterfactual[case_id]
        assert row["target_status"] == "unresolved"
        assert row["left_visible_raster_target"] is None
        assert row["right_visible_raster_target"] is None
        assert row["disposition"] in ("AMBIGUOUS", "GAP")
        if row["disposition"] == "GAP":
            assert row["candidate_pairs"] == []
            assert isinstance(row["reason"], str)
    stress = paired_result["structural_stress_audit"]
    assert stress["safe_set_or_gap_count"] == stress["case_count"] == 9
    assert stress["resolved_target_case_count"] == 1
    assert stress["unresolved_target_case_count"] == 8
    by_case = {row["case_id"]: row for row in stress["rows"]}
    assert by_case["stress-thinning-erased-crossbar"]["disposition"] == "GAP"
    assert by_case["stress-thinning-erased-crossbar"]["reason"] == (
        "foreground_residual_exceeds_single_path_stroke_envelope"
    )


def test_ambiguous_and_gap_predictions_stay_in_fixed_denominators() -> None:
    samples = tuple(
        SimpleNamespace(
            panel=SimpleNamespace(
                canonical_visible_pair=pair, carrier_family="synthetic-fixture"
            )
        )
        for pair in (item.as_tuple() for item in synthetic.valid_count_pairs())
    )
    candidates = [
        (sample.panel.canonical_visible_pair,) for sample in samples
    ]
    candidates[0] = ()
    wanted = samples[1].panel.canonical_visible_pair
    alternative = next(
        item.as_tuple()
        for item in synthetic.valid_count_pairs()
        if item.as_tuple() != wanted
    )
    candidates[1] = tuple(sorted((wanted, alternative)))
    metrics = subject._metrics(samples, candidates)
    assert metrics["denominator"] == 54
    assert metrics["joint_singleton_accuracy"] == 52 / 54
    assert metrics["candidate_set_contains_truth_accuracy"] == 53 / 54
    assert metrics["nonempty_candidate_set_rate"] == 53 / 54


def test_family_resampling_is_deterministic_and_not_row_resampling() -> None:
    rows = synthetic.build_identifiability_counterfactuals()[:2]
    samples = (
        *rows,
        *(
            SimpleNamespace(
                panel=SimpleNamespace(carrier_family="second-family")
            )
            for _row in rows
        ),
    )
    first = subject._descriptive_family_resampling(
        samples, [True] * len(samples), [False] * len(samples)
    )
    second = subject._descriptive_family_resampling(
        samples, [True] * len(samples), [False] * len(samples)
    )
    assert first == second
    assert first["carrier_cluster_count"] == 2
    assert first["resampling_unit"] == "held_out_carrier_family"
    assert first["ordered_joint_accuracy_family_resampling_p05"] == 1.0
    assert first["paired_delta_family_resampling_p05"] == 1.0


def test_result_tampering_changes_digest_and_module_has_no_live_surface(
    paired_result: dict[str, object],
) -> None:
    tampered = deepcopy(paired_result)
    tampered["authorization"]["benchmark_promotion"] = True
    body = dict(tampered)
    recorded = body.pop("record_digest")
    assert recorded != "sha256:" + canonical_digest(body)
    assert not hasattr(subject, "main")
    assert subject.source_sha256() == EXPECTED_SOURCE_SHA256
    source = open(subject.__file__, encoding="utf-8").read()  # noqa: PTH123
    assert "downloads/" not in source
    assert "official_panel_archive" not in source
