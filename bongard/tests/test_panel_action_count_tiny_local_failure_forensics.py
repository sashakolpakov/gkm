from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from bongard.canonical import canonical_digest, canonical_json
from bongard import panel_action_count_tiny_local_failure_forensics as audit


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / audit.OUTPUT_RELATIVE
ARTIFACT_DIGEST = (
    "sha256:53464cd512459d892ea3c6059e42ac047d852304f7ea1e2d6faf749cadced572"
)


def _artifact() -> dict:
    raw = ARTIFACT_PATH.read_bytes()
    value = json.loads(raw)
    assert raw == canonical_json(value) + b"\n"
    return value


def test_forensics_is_canonical_source_bound_and_development_only() -> None:
    artifact = _artifact()
    body = dict(artifact)
    assert body.pop("record_digest") == "sha256:" + canonical_digest(body)
    assert artifact["record_digest"] == ARTIFACT_DIGEST
    assert artifact["source_sha256"] == audit.source_sha256()
    assert artifact["bindings"]["result_record_digest"] == audit.RESULT_DIGEST
    assert artifact["bindings"]["replay_record_digest"] == audit.REPLAY_DIGEST
    assert artifact["bindings"]["checkpoint_raw_sha256"] == (
        audit.CHECKPOINT_RAW_SHA256
    )
    assert artifact["scope"] == {
        "already_exposed_development_pixel_occurrences_reread": 12_592,
        "calibration_evaluation_family_target_query_identifiers_opened": 0,
        "failed_checkpoint_diagnostic_only": True,
        "fresh_cohort_pixels_opened": 0,
        "validation_digest_groups_inferred": 1_392,
    }
    audit.verify_failure_forensics(artifact, repository_root=ROOT)


def test_full_confusions_reproduce_the_failed_checkpoint() -> None:
    artifact = _artifact()
    matrices = artifact["confusion_matrices"]
    for name, matrix in matrices.items():
        size = 3 if name.startswith("catalog_") else 10
        assert len(matrix) == size
        assert all(len(row) == size for row in matrix)
        assert sum(sum(row) for row in matrix) == 1_392
    assert matrices[
        "catalog_rows_true_columns_predicted_unresolved_nonconvex_convex"
    ] == [[977, 6, 15], [231, 27, 6], [102, 3, 25]]

    metrics = artifact["decoding_diagnostics"]["overall"]
    assert metrics["straight_dp_top1"] == 0.2650862068965517
    assert metrics["arc_dp_top1"] == 0.47270114942528735
    assert metrics["pair_dp_top1"] == 0.14152298850574713
    assert metrics["catalog_known_balanced_accuracy"] == 0.1472902097902098
    assert metrics["catalog_recall_unresolved_nonconvex_convex"] == {
        "0": 0.9789579158316634,
        "1": 0.10227272727272728,
        "2": 0.19230769230769232,
    }


def test_attention_collapse_and_dp_mismatch_are_quantified() -> None:
    artifact = _artifact()
    attention = artifact["attention_diagnostics"]
    assert attention["map_count"] == 12_528
    assert attention["pair_count"] == 50_112
    assert attention["normalized_entropy_uniform_equals_one"]["mean"] == (
        0.9873712072418721
    )
    assert attention["pairwise_cosine"]["mean"] == 0.9202867266575366
    assert attention["top8_pairwise_iou"]["mean"] == 0.29429102010519637
    assert attention["top8_union_ink_token_coverage"]["mean"] == (
        0.43754562094327837
    )
    assert attention["maximum_token_probability"]["mean"] == 0.03213325978433987
    assert attention["uniform_token_probability"] == 0.015625

    decoding = artifact["decoding_diagnostics"]
    assert decoding["overall"]["dp_slot_pair_disagreement"] == 0.555316091954023
    assert decoding["overall"]["pair_slot_argmax_top1"] > decoding["overall"][
        "pair_dp_top1"
    ]
    assert decoding["overall"]["arc_slot_argmax_top1"] > decoding["overall"][
        "arc_dp_top1"
    ]
    assert decoding["prediction_marginals"]["arc_dp"] == {
        "0": 723,
        "1": 605,
        "2": 64,
    }


def test_counts_lose_to_majority_and_imbalance_is_not_hidden() -> None:
    artifact = _artifact()
    metrics = artifact["decoding_diagnostics"]["overall"]
    baseline = artifact["majority_baselines"]
    assert metrics["straight_dp_top1"] < baseline["straight_validation_top1"]
    assert metrics["arc_dp_top1"] < baseline["arc_validation_top1"]
    assert metrics["pair_dp_top1"] < baseline["joint_pair_validation_top1"]

    imbalance = artifact["imbalance_and_gradient_diagnostics"]
    assert imbalance[
        "train_catalog_occurrence_histogram_unresolved_nonconvex_convex"
    ] == {"0": 7_982, "1": 2_038, "2": 1_180}
    assert imbalance["train_descriptor_slot_target_histogram_none_line_arc"] == {
        "0": 36_254,
        "1": 53_798,
        "2": 10_235,
    }
    assert imbalance["train_joint_count_pair_imbalance"] == {
        "largest_class_occurrences": 1_896,
        "largest_to_smallest_ratio": 118.5,
        "occupied_pair_count": 33,
        "smallest_class_occurrences": 16,
    }
    gradients = imbalance["loss_component_gradients_at_selected_checkpoint"]
    assert gradients["catalog_globally_class_balanced"][
        "parameter_gradient_l2"
    ] > gradients["catalog_unweighted"]["parameter_gradient_l2"]
    assert gradients["count_globally_class_balanced"][
        "parameter_gradient_l2"
    ] > gradients["count_unweighted"]["parameter_gradient_l2"]
    assert gradients["descriptor_geometry"]["parameter_gradient_l2"] < 0.021


def test_selection_failure_and_single_shape_hole_are_explicit() -> None:
    artifact = _artifact()
    architecture = artifact["architecture_audit"]
    assert architecture["explicit_token_coordinates_present"] is False
    assert architecture["attention_diversity_or_coverage_loss_present"] is False
    assert architecture[
        "training_and_validation_authority_shape_count_histograms"
    ] == {"train": {"1": 11_200}, "validation": {"1": 1_400}}
    assert architecture["two_shape_empirical_development_support"] is False

    epochs = artifact["epoch_selection_counterfactual"]
    assert epochs["selected_epoch"] == 3
    assert epochs["best_epoch_by_straight"] == 3
    assert epochs["best_epoch_by_mean_gate_metric"] == 5
    assert epochs["best_epoch_by_minimum_threshold_fraction"] == 5
    assert epochs["any_epoch_passed_gate"] is False
    assert epochs["archived_states_available"] == [3]

    successor = artifact["next_frozen_development_experiment"]
    assert successor["selection_status"] == "NOT_SELECTED"
    assert "skeleton graph" in successor["nonselection_reason"]
    assert successor["multi_shape_disposition"].startswith("GAP")


def test_verifier_rejects_tampering() -> None:
    artifact = _artifact()
    tampered = deepcopy(artifact)
    tampered["scope"]["fresh_cohort_pixels_opened"] = 1
    with pytest.raises(audit.TinyFailureForensicsError):
        audit.verify_failure_forensics(tampered, repository_root=ROOT)
