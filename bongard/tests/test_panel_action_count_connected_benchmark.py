from __future__ import annotations

from collections import Counter
from copy import deepcopy
import ast
import inspect

import pytest

from bongard.canonical import canonical_digest
from bongard import panel_action_count_connected_benchmark as subject
from bongard import panel_action_count_connected_synthetic as connected


# Updated only after the complete connected fixture, raw synthesizer, and
# benchmark result are frozen together.
EXPECTED_SOURCE_SHA256 = (
    "17cbad185e085d514e9973662329d44d81e60049959e4bdf929864ae13ed3e7c"
)
EXPECTED_RESULT_RECORD_DIGEST = (
    "sha256:0e5f711a6e686cfb9c2b1ff2cde1559a06f15542846794ce44cde57e6a368aff"
)


@pytest.fixture(scope="module")
def connected_result() -> dict[str, object]:
    return subject.run_connected_benchmark()


def test_connected_benchmark_is_exactly_paired_carrier_disjoint_and_non_authorizing(
    connected_result: dict[str, object],
) -> None:
    result = connected_result
    assert result["schema"] == subject.SCHEMA
    assert result["training"]["row_count"] == 636
    assert result["evaluation"]["row_count"] == 424
    assert result["training"]["carrier_families"] == [
        "lattice", "perimeter", "pinwheel",
    ]
    assert result["evaluation"]["carrier_families"] == ["radial", "staggered"]
    assert len(result["training"]["nuisances"]) == 2
    assert result["training"]["nuisances"] == result["evaluation"]["nuisances"]
    assert result["corpus_coverage"]["cell_count"] == 10
    assert result["corpus_coverage"][
        "complete_54_target_set_in_every_cell"
    ] is True
    assert len(result["corpus_coverage"]["target_set_universe"]) == 54
    assert len(result["paired_rows"]) == 424
    assert len({row["sample_id"] for row in result["paired_rows"]}) == 424
    assert result["authorization"] == {
        "benchmark_promotion": False,
        "calibration_target_query_authorized": False,
        "new_campaign_authority_created": False,
        "official_benchmark_or_generalization_claim_authorized": False,
        "official_data_inputs_authorized": False,
        "synthetic_only": True,
    }
    assert result["limitations"] == {
        "carrier_split_tests_unseen_catalog_induction": False,
        "held_out_family_geometry_present_in_raw_catalog": True,
        "held_out_reconstructible_after_removing_held_family_masks": False,
        "official_transfer_tested": False,
        "raw_and_target_share_fixed_primitive_catalog": True,
    }
    assert result["catalog_dependency_audit"] == {
        "audit_uses_target_oracle": False,
        "full_catalog_mask_count": 384,
        "held_family_only_masks_removed": True,
        "held_family_only_mask_count": 152,
        "held_out_exact_cover_count": 0,
        "held_out_row_count": 424,
        "layout_exact_cover_counts": {
            "single_shape": 0,
            "two_shape": 0,
        },
        "non_held_catalog_mask_count": 232,
        "strict_training_family_catalog_mask_count": 228,
        "synthetic_stress_catalog_mask_count": 4,
    }
    assert result["raw_synthesizer"] == {
        "algorithm_id": subject.SYNTHESIZER_ID,
        "candidate_construction_uses_target": False,
        "exact_catalog_reconstruction_required": True,
        "learning_used_for_candidate_construction": False,
    }
    assert result["control"]["feature_count"] == 112
    assert result["control"]["parameters"] == dict(subject.CONTROL_PARAMETERS)
    assert result["control"]["parameters"]["n_estimators"] == 32
    assert (
        result["target"]["evaluation_targets_constructed_after_raw_predictions"]
        is True
    )
    assert result["target"]["target_passed_to_raw_prediction"] is False
    assert result["target"]["generator_history_used_for_scoring"] is False


def test_connected_metrics_cover_layout_family_boundary_and_ambiguity_denominators(
    connected_result: dict[str, object],
) -> None:
    metrics = connected_result["metrics"]
    assert metrics["control"]["denominator"] == 424
    assert metrics["raw_synthesizer"]["denominator"] == 424
    assert sum(metrics["raw_synthesizer"]["disposition_counts"].values()) == 424
    assert set(metrics["by_layout"]) == {"single_shape", "two_shape"}
    assert metrics["by_layout"]["single_shape"]["raw_synthesizer"][
        "denominator"
    ] == 216
    assert metrics["by_layout"]["two_shape"]["raw_synthesizer"][
        "denominator"
    ] == 208
    assert set(metrics["by_carrier_family"]) == {"radial", "staggered"}
    assert all(
        row["raw_synthesizer"]["denominator"] == 212
        for row in metrics["by_carrier_family"].values()
    )
    assert set(metrics["by_boundary_kind"]) == {"AA", "AL", "LA", "LL"}
    assert metrics["boundary_kind_membership"][
        "empty_rows_excluded_from_kind_groups"
    ] is True
    assert metrics["boundary_kind_membership"][
        "groups_are_nonexclusive_presence_strata"
    ] is True
    assert all(
        row["raw_synthesizer"]["denominator"] > 0
        for row in metrics["by_boundary_kind"].values()
    )
    raw = metrics["raw_synthesizer"]
    assert raw["exact_reconstruction_rate"] == 1.0
    assert raw["false_singleton_on_ambiguous_target_count"] == 0
    assert connected_result["gates"]["passed"] is True
    assert all(
        value is True
        for key, value in connected_result["gates"].items()
        if key != "passed"
    )


def test_full_d4_cross_role_audit_and_matched_assignment_are_exact_partitions(
    connected_result: dict[str, object],
) -> None:
    d4 = connected_result["d4_cross_role_audit"]
    assert d4["full_eight_element_square_symmetry_orbit"] is True
    assert d4["cross_role_overlap_count"] == 0
    assert d4["overlap"] == []

    matched = connected_result["matched_pooled_feature_counterfactuals"]
    assert matched["pair_count"] == 212
    assert matched["occurrence_count"] == 424
    assert matched["layout_pair_counts"] == {
        "single_shape": 108,
        "two_shape": 104,
    }
    assert matched["every_evaluation_occurrence_used_exactly_once"] is True
    assert matched["same_target_pair_count"] == 0
    occurrences = [
        sample_id
        for row in matched["pairs"]
        for sample_id in (row["first_sample_id"], row["second_sample_id"])
    ]
    assert len(occurrences) == len(set(occurrences)) == 424
    assert set(occurrences) == {
        row["sample_id"] for row in connected_result["paired_rows"]
    }
    assert Counter(
        (row["family"], row["nuisance"], row["layout"])
        for row in matched["pairs"]
    ) == Counter(
        {
            (family, nuisance, layout): 27 if layout == "single_shape" else 26
            for family in ("radial", "staggered")
            for nuisance in {row["nuisance"] for row in matched["pairs"]}
            for layout in ("single_shape", "two_shape")
        }
    )
    assert all(
        row["first_target_pairs"] != row["second_target_pairs"]
        for row in matched["pairs"]
    )
    pair_level = matched["pair_level_both_endpoints_exact"]
    assert pair_level["denominator"] == 212
    assert pair_level["raw_minus_control_accuracy"] >= (
        subject.GATE_THRESHOLDS["matched_raw_minus_control_at_least"]
    )
    assert matched["occurrence_raw_minus_control_exact_accuracy"] == (
        connected_result["metrics"]["raw_minus_control_exact_accuracy"]
    )


def test_raw_synthesizer_predictions_do_not_call_exact_cover_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = connected.build_connected_corpus()
    evaluation = tuple(
        sample
        for sample in corpus
        if sample.panel_program.carrier_family in subject.EVALUATION_FAMILIES
    )
    representatives = []
    seen = set()
    for sample in evaluation:
        truth = sample.boundary_truth
        key = truth if isinstance(truth, str) else repr(truth)
        if key not in seen:
            representatives.append(sample)
            seen.add(key)
        if len(representatives) >= 4:
            break
    assert representatives

    def forbidden_target(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("raw synthesizer consulted target oracle")

    monkeypatch.setattr(connected, "exact_cover_target", forbidden_target)
    outputs = subject.raw_synthesizer_outputs(tuple(representatives))
    assert len(outputs) == len(representatives)
    assert all(
        output["disposition"] in ("IDENTIFIED", "AMBIGUOUS", "GAP")
        for output in outputs
    )


def test_raw_wrapper_has_no_target_input_or_target_oracle_call() -> None:
    signature = inspect.signature(subject.raw_synthesizer_outputs)
    assert tuple(signature.parameters) == ("samples",)
    tree = ast.parse(inspect.getsource(subject.raw_synthesizer_outputs))
    called_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "exact_cover_target" not in called_names


def test_connected_result_is_canonical_source_bound_and_has_no_live_surface(
    connected_result: dict[str, object],
) -> None:
    body = dict(connected_result)
    recorded = body.pop("record_digest")
    assert recorded == "sha256:" + canonical_digest(body)
    assert subject.source_sha256() == EXPECTED_SOURCE_SHA256
    assert recorded == EXPECTED_RESULT_RECORD_DIGEST
    tampered = deepcopy(connected_result)
    tampered["authorization"]["benchmark_promotion"] = True
    altered = dict(tampered)
    assert altered.pop("record_digest") != "sha256:" + canonical_digest(altered)
    assert not hasattr(subject, "main")
    source = open(subject.__file__, encoding="utf-8").read()  # noqa: PTH123
    assert "downloads/" not in source
    assert "official_panel_archive" not in source
