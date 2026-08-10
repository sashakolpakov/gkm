"""Frozen development-only skeleton-graph action observer.

This module deliberately contains no calibration, query, target, or benchmark
entry point.  It freezes the pixel-only feature bank and estimator topology
selected after inspecting the already-exposed ShapeBongard development split.
Its results are therefore adaptive engineering evidence for the finite HD
carrier catalog, never an independent evaluation or a novel-carrier claim.

The original 256-tree v1 fit failed closed at its prewrite model-size guard.
V2 is a capacity-only repair: it freezes 32 trees, the first passing candidate
in a disclosed eight-prefix adaptive study over a raw-addressed cached feature
matrix.  Runtime fitting performs no capacity search.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from io import BytesIO
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import signal
import stat
import sys
import time
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

from bongard.canonical import canonical_digest, canonical_json
from bongard import panel_action_local_duplicate_conflict_audit as conflict_module
from bongard.panel_action_local_duplicate_conflict_audit import _pose_free_target
from bongard.panel_action_local_supervision_authority import (
    compile_pose_free_panel,
    load_development_authority,
)


def _freeze_literal(value: Any) -> Any:
    """Deep-freeze one source literal before it becomes protocol authority."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_literal(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_literal(item) for item in value)
    return value


SCHEMA_PRECOMMIT: Final = "gkm.bongard-skeleton-graph-development-precommit.v2"
SCHEMA_RESULT: Final = "gkm.bongard-skeleton-graph-development-result.v2"
SCHEMA_FEATURES: Final = "gkm.bongard-skeleton-graph-development-features.v2"
SCHEMA_PREDICTIONS: Final = "gkm.bongard-skeleton-graph-development-predictions.v2"
SCHEMA_REPLAY: Final = "gkm.bongard-skeleton-graph-development-replay.v2"
MODEL_SCHEMA: Final = "gkm.bongard-skeleton-graph-development-model-pickle.v2"
MODEL_MAX_BYTES: Final = 512 * 1024 * 1024
FEATURE_ARTIFACT_MAX_BYTES: Final = 32 * 1024 * 1024
PREDICTION_ARTIFACT_MAX_BYTES: Final = 32 * 1024 * 1024
RESULT_MAX_BYTES: Final = 8 * 1024 * 1024

FIT_PRECOMMIT_SCHEMA: Final = (
    "gkm.bongard-action-count-catalog-cnn-fit-pixel-precommit.v2"
)
FIT_PRECOMMIT_RECORD_DIGEST: Final = (
    "sha256:e8c7c15fbfb723c5b2305094f035e2567c1fb9b7e80b9f13eeae32fe35d1b15a"
)
FIT_PRECOMMIT_FILE_SHA256: Final = (
    "sha256:bfc1267150bac4f823dc72fc483d64f2c32e98ace90ff63bad2556fe4d6aec97"
)
COMMITTED_LABEL_AUTHORITY_AUDIT_RECORD_DIGEST: Final = (
    "sha256:ac74773f5dfc05fcd935822eee5567d06d806a71f79b17b011557b648c2e4a25"
)
COMMITTED_LABEL_AUTHORITY_AUDIT_RELATIVE_PATH: Final = (
    "bongard/data/panel_action_local_duplicate_conflict_audit_20260810_v1.json"
)

BASE_PROTOCOL_COMMIT: Final = "f65d3f79b45fa0fa1b001bf9c0cbb6c7f3e0b302"
BASE_PROTOCOL_SOURCE_SHA256: Final = (
    "8154c3099ccbf4980004574b0bb023a13720dbbaa858c3ba1184567d918a81b0"
)
PRIOR_FAILED_PRECOMMIT_SCHEMA: Final = (
    "gkm.bongard-skeleton-graph-development-precommit.v1"
)
PRIOR_FAILED_PRECOMMIT_RECORD_DIGEST: Final = (
    "sha256:540712206d87c3d50e1663e64c6e1048103f07bc198cb926603bd30968837fe8"
)
PRIOR_FAILED_PRECOMMIT_FILE_SHA256: Final = (
    "sha256:37a18000f2b6b9feb4e2a1adce2a862496a96a2511de15da0f03fffefd769879"
)
PRIOR_FAILED_OUTPUT_ROOT: Final = (
    "/Users/sasha/gkm/downloads/ShapeBongard_V2_full/"
    "panel_action_count_skeleton_graph_dev_20260810_v1"
)
PRIOR_FAILED_OUTPUT_ROOT_IDENTITY: Final = MappingProxyType(
    {"st_dev": 16_777_234, "st_ino": 38_039_341, "st_mode": 16_877}
)
PRIOR_FAILED_INTENDED_OUTPUT_NAMES: Final = (
    "features.json", "model.pkl", "predictions.json", "replay.json", "result.json",
)

CLAIM_SCOPE: Final = "finite_catalog_known_carrier_style_pose_transfer_engineering"
PROMOTION_REQUIRES: Final = "separate_same_family_calibration_authority"
IMAGE_SIZE: Final = 64
FIXED_CLASSIFIER_SEEDS: Final = MappingProxyType(
    {"direct_pair": 260813, "catalog_three_class": 260812}
)
VALID_PAIR_CLASS_ORDER: Final = tuple(
    10 * straight + arc
    for straight in range(10)
    for arc in range(10)
    if 1 <= straight + arc <= 9
)
CATALOG_CLASS_ORDER: Final = (-1, 0, 1)
OBSERVED_TRAIN_PAIR_CLASS_ORDER: Final = (
    1, 2, 4, 6, 8, 11, 12, 20, 21, 22, 23, 30, 31, 32, 33, 34,
    40, 41, 42, 43, 44, 50, 51, 52, 60, 61, 62, 63, 70, 71, 80, 81, 90,
)

SCALE_SPECS: Final = (
    ("raw_threshold_10_of_255", 0.0, 10.0 / 255.0, False),
    ("raw_threshold_then_3x3_closing_once", 0.0, 10.0 / 255.0, True),
    ("gaussian_sigma_1p5_threshold_0p08", 1.5, 0.08, False),
    ("gaussian_sigma_3p0_threshold_0p035", 3.0, 0.035, False),
)

PER_SCALE_FEATURE_NAMES: Final = (
    "foreground_area_fraction",
    "boundary_area_fraction",
    "component_count_div_32",
    "hole_count_div_32",
    "largest_component_fraction",
    "component_area_sd_div_foreground_area",
    "bbox_height_fraction",
    "bbox_width_fraction",
    "bbox_aspect_ratio",
    "centroid_y_fraction",
    "centroid_x_fraction",
    "second_moment_anisotropy",
    "skeleton_area_fraction",
    "endpoint_cluster_count_div_32",
    "branch_cluster_count_div_32",
    "isolated_skeleton_pixel_count_div_32",
    "eight_neighbor_raster_cycle_rank_div_32",
    "skeleton_edge_count_fraction",
    "edge_orientation_horizontal_fraction",
    "edge_orientation_diagonal_up_fraction",
    "edge_orientation_vertical_fraction",
    "edge_orientation_diagonal_down_fraction",
    "degree_two_turn_cos_le_neg_0p9_fraction",
    "degree_two_turn_cos_neg_0p9_to_neg_0p25_fraction",
    "degree_two_turn_cos_neg_0p25_to_0p25_fraction",
    "degree_two_turn_cos_gt_0p25_fraction",
    "mean_skeleton_half_width_div_8",
    "max_skeleton_half_width_div_8",
)
FEATURE_NAMES: Final = tuple(
    f"{scale_name}:{feature_name}"
    for scale_name, _sigma, _threshold, _closing in SCALE_SPECS
    for feature_name in PER_SCALE_FEATURE_NAMES
)

PROTOCOL: Final = MappingProxyType(
    {
        "authorized_pixels": "already_exposed_decontaminated_development_only",
        "classifier": "sklearn.ensemble.ExtraTreesClassifier",
        "class_weight": "balanced_fit_on_training_labels_only",
        "feature_count": 112,
        "image_size": IMAGE_SIZE,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "n_estimators": 32,
        "n_jobs": 1,
        "one_vote_per_unique_png_digest_group": True,
        "promotion": False,
        "promotion_requires": PROMOTION_REQUIRES,
        "validation_status": "adaptive_architecture_selection_development",
    }
)

# This ledger is part of the protocol because the official development
# validation pixels were inspected while choosing the fourth representation
# and the direct pair head.  These are diagnostics, not preregistered gates.
ADAPTIVE_VARIANT_LEDGER: Final = (
    MappingProxyType({
        "variant": "raw_graph_28_features",
        "validation_accuracy": MappingProxyType(
            {"straight": 0.618534, "arc": 0.734914, "joint": 0.497845}
        ),
    }),
    MappingProxyType({
        "variant": "coarse_sigma_3_graph_28_features",
        "validation_accuracy": MappingProxyType(
            {"straight": 0.685345, "arc": 0.818247, "joint": 0.609914}
        ),
    }),
    MappingProxyType({
        "variant": "four_scale_graph_112_features_separate_marginals",
        "validation_accuracy": MappingProxyType(
            {"straight": 0.828305, "arc": 0.863506, "joint": 0.752874}
        ),
    }),
    MappingProxyType({
        "variant": "four_scale_graph_direct_pair_head",
        "validation_accuracy": MappingProxyType({"joint": 0.832615}),
        "status": "post_validation_adapted_diagnostic",
    }),
    MappingProxyType({
        "variant": "four_scale_graph_direct_catalog_three_class",
        "validation_accuracy": 0.9238505747126436,
        "validation_balanced_accuracy": 0.8347292487072046,
        "known_truth_balanced_accuracy_unresolved_is_wrong": 0.7576048951048951,
        "status": "post_validation_adapted_diagnostic",
    }),
    MappingProxyType({
        "variant": "conditional_catalog_given_known",
        "known_truth_balanced_accuracy": 0.9138986013986015,
        "status": "diagnostic_only_never_serialized_or_projected",
    }),
)

CARRIER_SIGNATURE_CV_DIAGNOSTIC: Final = MappingProxyType(
    {
        "cohort": "train_only",
        "fold_algorithm": (
            "uint64_be(sha256(b'gkm-carrier-cv-v1\\0'+signature_utf8)[:8]) mod 5"
        ),
        "fold_row_counts": (2349, 1880, 2178, 2199, 2537),
        "fold_signature_counts": (130, 110, 119, 124, 142),
        "signature_count": 625,
        "separate_marginal_head_oof": MappingProxyType({
            "straight_accuracy": 0.4422507403751234,
            "straight_balanced_accuracy": 0.3756278150456772,
            "straight_signature_macro_accuracy": 0.4065820411665552,
            "arc_accuracy": 0.6925424033025218,
            "arc_balanced_accuracy": 0.4328421389974387,
            "arc_signature_macro_accuracy": 0.6348011083110183,
            "joint_accuracy": 0.352149331418828,
            "joint_signature_macro_accuracy": 0.3054062978972449,
        }),
        "catalog_three_class_oof": MappingProxyType({
            "accuracy": 0.8591941128959885,
            "balanced_accuracy": 0.7124530667698593,
            "known_truth_balanced_accuracy_unresolved_is_wrong": 0.576179,
            "signature_macro_accuracy": 0.8075660558615144,
        }),
        "interpretation": "novel_carrier_counts_are_not_promoted",
        "status": "adaptive_diagnostic_not_pristine_model_selection_evidence",
    }
)

ENGINEERING_THRESHOLDS: Final = MappingProxyType(
    {
        "direct_pair_joint_accuracy": 0.78,
        "catalog_known_truth_balanced_accuracy": 0.70,
    }
)

# The 256-tree v1 protocol failed closed before its first post-precommit write:
# its exact deterministic pickle exceeded the unchanged 512 MiB replay cap.
# Capacity repair inspected only an already-derived, raw-addressed feature cache.
# It fit no candidate during v2 execution: the ordered study below is historical,
# validation-adapted development evidence and selects the first disclosed prefix
# satisfying both existing gates and the existing byte cap.  Counts 17..31 were
# not tried, so 32 is not claimed to be the globally smallest passing tree count.
CAPACITY_SELECTION_LEDGER: Final = _freeze_literal(
    {
        "schema": "gkm.bongard-skeleton-graph-capacity-selection-ledger.v1",
        "status": "adaptive_development_capacity_repair_not_fresh_gate_evidence",
        "base_protocol": {
            "commit": BASE_PROTOCOL_COMMIT,
            "source_sha256": BASE_PROTOCOL_SOURCE_SHA256,
            "n_estimators": 256,
            "model_max_bytes": MODEL_MAX_BYTES,
        },
        "failed_v1_attempt": {
            "precommit_schema": PRIOR_FAILED_PRECOMMIT_SCHEMA,
            "precommit_record_digest": PRIOR_FAILED_PRECOMMIT_RECORD_DIGEST,
            "precommit_file_sha256": PRIOR_FAILED_PRECOMMIT_FILE_SHA256,
            "output_root": PRIOR_FAILED_OUTPUT_ROOT,
            "output_root_identity": PRIOR_FAILED_OUTPUT_ROOT_IDENTITY,
            "durable_file_names": ("precommit.json",),
            "absent_intended_output_names": PRIOR_FAILED_INTENDED_OUTPUT_NAMES,
            "durable_output_count_beyond_precommit": 0,
            "result_exists": False,
            "failure_stage": "prewrite_artifact_size_guard_after_fit",
            "failure_message": (
                "model, feature, or prediction artifact exceeds replay cap"
            ),
            "model_serialized_bytes": 780_044_909,
            "model_max_bytes": MODEL_MAX_BYTES,
            "direct_pair_total_node_count": 2_086_910,
            "catalog_three_class_total_node_count": 1_083_066,
        },
        "cached_feature_matrix": {
            "path_is_not_runtime_authority": True,
            "raw_file_sha256": (
                "sha256:ad7160dfa7f7f40d1889a22eab17b9782f1aee2999ff7752e080883563cd9680"
            ),
            "raw_file_bytes": 2_991_202,
            "custody_limitation": (
                "ephemeral_uncommitted_diagnostic_cache_bound_by_raw_address"
            ),
            "arrays": {
                "X": {
                    "shape": (12_535, 112),
                    "dtype": "float32",
                    "raw_array_sha256": (
                        "sha256:bd90575c33d0368058407d93810dee0b684d7579e12ef46a1641c88c7e437cd0"
                    ),
                },
                "Y": {
                    "shape": (12_535, 3),
                    "dtype": "int64",
                    "raw_array_sha256": (
                        "sha256:c850130862e90ebf38e16811f7fecfd358c89bcce768d5ac56b2ce7ddee231c0"
                    ),
                },
                "S": {
                    "shape": (12_535,),
                    "dtype": "<U10",
                    "raw_array_sha256": (
                        "sha256:b4f68697fc2c35bd8c00754068c2822ba806c73ba08d78d2b58f6120936f31fc"
                    ),
                },
                "T": {
                    "shape": (12_535,),
                    "dtype": "<U56",
                    "raw_array_sha256": (
                        "sha256:39cc51a208d00e7b0afa8e6b4ee6a14f6430ee991465a8eb4f44b38c91711056"
                    ),
                },
            },
            "effective_group_counts": {"train": 11_143, "validation": 1_392},
            "capacity_selection_data_access": {
                "cached_npz_read_only": True,
                "png_reads": 0,
                "action_program_reads": 0,
                "fresh_cohort_reads": 0,
            },
            "generator_history_limitation": (
                "cache_generator_was_exploratory_and_not_precommitted_write_once"
            ),
        },
        "estimator_protocol": {
            "classifier": "sklearn.ensemble.ExtraTreesClassifier",
            "class_weight": "balanced_fit_on_training_labels_only",
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "n_jobs": 1,
            "random_seeds": FIXED_CLASSIFIER_SEEDS,
            "scikit_learn": "1.8.0",
            "numpy": "2.4.4",
            "source_addresses": {
                "sklearn.base": (
                    "sha256:10992a4472940aa33f499e3bfc2477c2fe82509963907127fbd71ce3f7380ec6"
                ),
                "sklearn.ensemble._forest": (
                    "sha256:c1e8d4ce4036fda18b8033f7d7892ec9b8534f0e5663cfa57e5455ca404c891c"
                ),
                "sklearn.tree._classes": (
                    "sha256:5494c9f6821d092a207a5529a0a57b47c2a249f92cafa8a52dd53b452abaaf63"
                ),
                "sklearn.tree._tree": (
                    "sha256:9e011f8940abbf61c81b12991ee31887c05b98fe3787a180c884455b0e221365"
                ),
            },
        },
        "selection_rule": (
            "first candidate in the exact disclosed tested sequence meeting both "
            "unchanged engineering gates and the unchanged model byte cap"
        ),
        "candidate_tree_count_order": (16, 32, 48, 64, 96, 128, 192, 256),
        "candidate_rows": (
            {
                "n_estimators": 16,
                "direct_pair_joint_accuracy": 0.7600574712643678,
                "decoded_straight_accuracy": 0.7837643678160919,
                "decoded_arc_accuracy": 0.8469827586206896,
                "catalog_known_truth_balanced_accuracy": 0.7118589743589744,
                "catalog_three_class_accuracy": 0.9001436781609196,
                "direct_pair_total_node_count": 130_264,
                "catalog_three_class_total_node_count": 69_320,
                "estimators_only_serialized_bytes": 48_842_341,
                "model_bundle_serialized_bytes": 48_850_411,
                "within_model_cap": True,
                "direct_pair_passed": False,
                "catalog_three_class_passed": True,
                "selected": False,
            },
            {
                "n_estimators": 32,
                "direct_pair_joint_accuracy": 0.7801724137931034,
                "decoded_straight_accuracy": 0.7995689655172413,
                "decoded_arc_accuracy": 0.8663793103448276,
                "catalog_known_truth_balanced_accuracy": 0.7385489510489511,
                "catalog_three_class_accuracy": 0.9166666666666666,
                "direct_pair_total_node_count": 261_776,
                "catalog_three_class_total_node_count": 136_500,
                "estimators_only_serialized_bytes": 97_903_781,
                "model_bundle_serialized_bytes": 97_911_851,
                "within_model_cap": True,
                "direct_pair_passed": True,
                "catalog_three_class_passed": True,
                "selected": True,
            },
            {
                "n_estimators": 48,
                "direct_pair_joint_accuracy": 0.7880747126436781,
                "decoded_straight_accuracy": 0.805316091954023,
                "decoded_arc_accuracy": 0.8706896551724138,
                "catalog_known_truth_balanced_accuracy": 0.7346445221445221,
                "catalog_three_class_accuracy": 0.9181034482758621,
                "direct_pair_total_node_count": 392_960,
                "catalog_three_class_total_node_count": 204_082,
                "estimators_only_serialized_bytes": 146_893_013,
                "model_bundle_serialized_bytes": 146_901_083,
                "within_model_cap": True,
                "direct_pair_passed": True,
                "catalog_three_class_passed": True,
                "selected": False,
            },
            {
                "n_estimators": 64,
                "direct_pair_joint_accuracy": 0.8038793103448276,
                "decoded_straight_accuracy": 0.8225574712643678,
                "decoded_arc_accuracy": 0.8793103448275862,
                "catalog_known_truth_balanced_accuracy": 0.7212703962703964,
                "catalog_three_class_accuracy": 0.915948275862069,
                "direct_pair_total_node_count": 523_402,
                "catalog_three_class_total_node_count": 272_314,
                "estimators_only_serialized_bytes": 195_696_069,
                "model_bundle_serialized_bytes": 195_704_139,
                "within_model_cap": True,
                "direct_pair_passed": True,
                "catalog_three_class_passed": True,
                "selected": False,
            },
            {
                "n_estimators": 96,
                "direct_pair_joint_accuracy": 0.819683908045977,
                "decoded_straight_accuracy": 0.8347701149425287,
                "decoded_arc_accuracy": 0.8922413793103449,
                "catalog_known_truth_balanced_accuracy": 0.738490675990676,
                "catalog_three_class_accuracy": 0.9195402298850575,
                "direct_pair_total_node_count": 784_480,
                "catalog_three_class_total_node_count": 408_474,
                "estimators_only_serialized_bytes": 293_339_061,
                "model_bundle_serialized_bytes": 293_347_131,
                "within_model_cap": True,
                "direct_pair_passed": True,
                "catalog_three_class_passed": True,
                "selected": False,
            },
            {
                "n_estimators": 128,
                "direct_pair_joint_accuracy": 0.826867816091954,
                "decoded_straight_accuracy": 0.8433908045977011,
                "decoded_arc_accuracy": 0.896551724137931,
                "catalog_known_truth_balanced_accuracy": 0.738490675990676,
                "catalog_three_class_accuracy": 0.9195402298850575,
                "direct_pair_total_node_count": 1_044_820,
                "catalog_three_class_total_node_count": 542_838,
                "estimators_only_serialized_bytes": 390_581_941,
                "model_bundle_serialized_bytes": 390_590_011,
                "within_model_cap": True,
                "direct_pair_passed": True,
                "catalog_three_class_passed": True,
                "selected": False,
            },
            {
                "n_estimators": 192,
                "direct_pair_joint_accuracy": 0.834051724137931,
                "decoded_straight_accuracy": 0.8512931034482759,
                "decoded_arc_accuracy": 0.8972701149425287,
                "catalog_known_truth_balanced_accuracy": 0.7441724941724941,
                "catalog_three_class_accuracy": 0.9216954022988506,
                "direct_pair_total_node_count": 1_563_876,
                "catalog_three_class_total_node_count": 814_946,
                "estimators_only_serialized_bytes": 584_832_469,
                "model_bundle_serialized_bytes": 584_840_539,
                "within_model_cap": False,
                "direct_pair_passed": True,
                "catalog_three_class_passed": True,
                "selected": False,
            },
            {
                "n_estimators": 256,
                "direct_pair_joint_accuracy": 0.8326149425287356,
                "decoded_straight_accuracy": 0.8484195402298851,
                "decoded_arc_accuracy": 0.8958333333333334,
                "catalog_known_truth_balanced_accuracy": 0.757604895104895,
                "catalog_three_class_accuracy": 0.9238505747126436,
                "direct_pair_total_node_count": 2_086_910,
                "catalog_three_class_total_node_count": 1_083_066,
                "estimators_only_serialized_bytes": 780_036_839,
                "model_bundle_serialized_bytes": 780_044_909,
                "within_model_cap": False,
                "direct_pair_passed": True,
                "catalog_three_class_passed": True,
                "selected": False,
            },
        ),
        "prefix_equivalence_replay": {
            "tested_tree_counts": (16, 32),
            "separate_fit_tree_states_equal_prefix": True,
            "separate_fit_validation_probabilities_byte_equal_prefix": True,
            "probability_raw_sha256": {
                "16": {
                    "direct_pair": (
                        "sha256:2417a298b57476da1cde0c7d594c5fdd2abd34c78b928712b664d0998ef310e8"
                    ),
                    "catalog_three_class": (
                        "sha256:3bc1396ac2cab48aec3f87ca002579c14cc0d55d04faec350bcb16adac89f745"
                    ),
                },
                "32": {
                    "direct_pair": (
                        "sha256:307ab73c09481c30ca8d174f4ffb84e0591680e6a9a3947815615c90239e0296"
                    ),
                    "catalog_three_class": (
                        "sha256:0cfd53b434dcb4d99d8a7f3a4aff18d1b1d269c3b234d64805893301b926bf3c"
                    ),
                },
            },
        },
        "known_selection_and_reproduction_fit_history": {
            "capacity_selection_head_fit_count": 2,
            "capacity_selection_counts": {
                "direct_pair_256": 1,
                "catalog_three_class_256": 1,
            },
            "implementation_reproduction_head_fit_count": 10,
            "implementation_reproduction_counts": {
                "direct_pair_16": 1, "catalog_three_class_16": 1,
                "direct_pair_32": 2, "catalog_three_class_32": 2,
                "direct_pair_256": 2, "catalog_three_class_256": 2,
            },
            "total_head_fit_count": 12,
            "regression_and_independent_audit_replays_excluded": True,
            "purpose": "selection_then_metric_size_and_prefix_equivalence_reproduction",
        },
        "selected_n_estimators": 32,
        "v2_runtime_capacity_search": False,
        "unchanged_engineering_thresholds": {
            "direct_pair_joint_accuracy": 0.78,
            "catalog_known_truth_balanced_accuracy": 0.70,
        },
        "unchanged_model_max_bytes": MODEL_MAX_BYTES,
    }
)


class SkeletonGraphDevelopmentError(RuntimeError):
    """The frozen development-only protocol or its input differs."""


def source_sha256() -> str:
    """Return the import-time source address, rejecting post-import drift."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_preflight() -> None:
    source_sha256()
    if len(FEATURE_NAMES) != 112 or len(set(FEATURE_NAMES)) != 112:
        raise SkeletonGraphDevelopmentError("feature vocabulary differs")
    if len(VALID_PAIR_CLASS_ORDER) != 54 or CATALOG_CLASS_ORDER != (-1, 0, 1):
        raise SkeletonGraphDevelopmentError("class universe differs")


def preprocess_png_bytes(raw: bytes) -> np.ndarray:
    """Tight-crop one exact PNG to the frozen 64x64 uint8 ink plane."""

    _authority_preflight()
    if not isinstance(raw, bytes) or len(raw) == 0 or len(raw) > 16 * 1024 * 1024:
        raise SkeletonGraphDevelopmentError("PNG byte count is outside the fixed cap")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise SkeletonGraphDevelopmentError("input must be one PNG frame")
            if (
                image.width <= 0
                or image.height <= 0
                or image.width > 2048
                or image.height > 2048
                or image.width * image.height > 4_194_304
            ):
                raise SkeletonGraphDevelopmentError("PNG dimensions exceed the fixed cap")
            image.load()
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
    except SkeletonGraphDevelopmentError:
        raise
    except Exception as exc:  # pragma: no cover - decoder/environment failure
        raise SkeletonGraphDevelopmentError(f"cannot decode PNG: {exc}") from exc
    ys, xs = np.nonzero(gray < 250)
    if len(xs) == 0:
        raise SkeletonGraphDevelopmentError("PNG has no ink")
    crop = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = crop.shape
    margin = math.ceil(0.08 * max(height, width))
    side = max(height, width) + 2 * margin
    canvas = np.full((side, side), 255, dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top : top + height, left : left + width] = crop
    resized = Image.fromarray(canvas, mode="L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    return np.ascontiguousarray(255 - np.asarray(resized, dtype=np.uint8))


def _zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Deterministic topology-preserving thinning until a fixed point."""

    current = np.ascontiguousarray(mask, dtype=bool)
    if not current.any():
        return current
    ys, xs = np.nonzero(current)
    cropped = current[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    work = np.pad(cropped, 1, constant_values=False)
    for _ in range(max(work.shape)):
        changed = False
        for phase in (0, 1):
            padded = np.pad(work, 1, constant_values=False)
            neighbours = (
                padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:],
                padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2],
                padded[1:-1, :-2], padded[:-2, :-2],
            )
            count = sum(item.astype(np.uint8) for item in neighbours)
            transitions = sum(
                ((~neighbours[index]) & neighbours[(index + 1) % 8]).astype(np.uint8)
                for index in range(8)
            )
            p2, _p3, p4, _p5, p6, _p7, p8, _p9 = neighbours
            if phase == 0:
                gate_a, gate_b = ~(p2 & p4 & p6), ~(p4 & p6 & p8)
            else:
                gate_a, gate_b = ~(p2 & p4 & p8), ~(p2 & p6 & p8)
            delete = work & (count >= 2) & (count <= 6) & (transitions == 1) & gate_a & gate_b
            if delete.any():
                work &= ~delete
                changed = True
        if not changed:
            break
    else:  # pragma: no cover - convergence guard
        raise SkeletonGraphDevelopmentError("thinning exceeded its dimension bound")
    result = np.zeros_like(current)
    result[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = work[1:-1, 1:-1]
    return np.ascontiguousarray(result)


_DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _one_scale_features(mask: np.ndarray) -> np.ndarray:
    s8 = np.ones((3, 3), dtype=np.uint8)
    s4 = np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8)
    unit_directions = np.asarray(_DIRECTIONS, dtype=np.float64)
    unit_directions /= np.linalg.norm(unit_directions, axis=1)[:, None]
    mask = np.ascontiguousarray(mask, dtype=bool)
    area = int(mask.sum())
    labels, component_count = ndimage.label(mask, structure=s8)
    sizes = np.bincount(labels.ravel())[1:]
    holes = ndimage.binary_fill_holes(mask) & ~mask
    _hole_labels, hole_count = ndimage.label(holes, structure=s4)
    boundary = mask & ~ndimage.binary_erosion(mask, structure=s8, border_value=0)
    ys, xs = np.nonzero(mask)
    if area:
        height, width = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
        centroid_y, centroid_x = float(ys.mean() / 63), float(xs.mean() / 63)
        centered = np.stack(((ys - ys.mean()) / 64, (xs - xs.mean()) / 64))
        covariance = np.cov(centered, bias=True) if area > 1 else np.zeros((2, 2))
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        anisotropy = float(eigenvalues[-1] / (eigenvalues.sum() + 1e-9))
    else:  # Gaussian and raw variants of a valid input are nonempty.
        height = width = 0
        centroid_y = centroid_x = anisotropy = 0.0
    skeleton = _zhang_suen(mask)
    vertex_count = int(skeleton.sum())
    _component_labels, skeleton_components = ndimage.label(skeleton, structure=s8)
    padded = np.pad(skeleton, 1)
    neighbours = [
        padded[1 + dy : 65 + dy, 1 + dx : 65 + dx]
        for dy, dx in _DIRECTIONS
    ]
    degree = sum(item.astype(np.uint8) for item in neighbours)
    endpoints = skeleton & (degree == 1)
    branches = skeleton & (degree >= 3)
    isolated = skeleton & (degree == 0)
    _labels, endpoint_clusters = ndimage.label(endpoints, structure=s8)
    _labels, branch_clusters = ndimage.label(branches, structure=s8)
    edges = np.asarray(
        (
            np.count_nonzero(skeleton[:, :-1] & skeleton[:, 1:]),
            np.count_nonzero(skeleton[:-1, 1:] & skeleton[1:, :-1]),
            np.count_nonzero(skeleton[:-1, :] & skeleton[1:, :]),
            np.count_nonzero(skeleton[:-1, :-1] & skeleton[1:, 1:]),
        ),
        dtype=np.float64,
    )
    edge_count = float(edges.sum())
    orientations = edges / (edge_count + 1e-9)
    turns = np.zeros(4, dtype=np.float64)
    degree_two = skeleton & (degree == 2)
    for first in range(8):
        for second in range(first + 1, 8):
            count = np.count_nonzero(degree_two & neighbours[first] & neighbours[second])
            cosine = float(np.dot(unit_directions[first], unit_directions[second]))
            bucket = 0 if cosine <= -0.9 else 1 if cosine <= -0.25 else 2 if cosine <= 0.25 else 3
            turns[bucket] += count
    turns /= float(degree_two.sum()) + 1e-9
    widths = ndimage.distance_transform_edt(mask)[skeleton]
    cycle_rank = max(0.0, edge_count - vertex_count + float(skeleton_components))
    values = (
        area / 4096, float(boundary.sum()) / 4096, component_count / 32,
        hole_count / 32, float(sizes.max()) / area if len(sizes) and area else 0,
        float(np.std(sizes)) / area if len(sizes) and area else 0,
        height / 64, width / 64, height / (width + 1e-9), centroid_y, centroid_x,
        anisotropy, vertex_count / 4096, endpoint_clusters / 32,
        branch_clusters / 32, float(isolated.sum()) / 32, cycle_rank / 32,
        edge_count / 4096, *orientations.tolist(), *turns.tolist(),
        (float(widths.mean()) if len(widths) else 0) / 8,
        (float(widths.max()) if len(widths) else 0) / 8,
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (28,) or not np.isfinite(result).all():
        raise SkeletonGraphDevelopmentError("feature vector is nonfinite or malformed")
    return result


def extract_feature_vector(png_bytes: bytes) -> np.ndarray:
    """Return the frozen 112-float vector from exact PNG bytes only."""

    _authority_preflight()
    ink = preprocess_png_bytes(png_bytes)
    raw = ink >= 10
    if not raw.any():
        raise SkeletonGraphDevelopmentError("PNG has no ink at the frozen threshold")
    strength = ink.astype(np.float32) / 255.0
    s8 = np.ones((3, 3), dtype=np.uint8)
    masks = (
        raw,
        ndimage.binary_closing(raw, structure=s8, iterations=1),
        ndimage.gaussian_filter(strength, 1.5, mode="constant") >= 0.08,
        ndimage.gaussian_filter(strength, 3.0, mode="constant") >= 0.035,
    )
    result = np.concatenate(tuple(_one_scale_features(mask) for mask in masks))
    if result.shape != (len(FEATURE_NAMES),):
        raise SkeletonGraphDevelopmentError("multiscale feature vector shape differs")
    return np.ascontiguousarray(result, dtype=np.float32)


def build_authoritative_estimators():
    """Build unfitted deterministic direct-pair and direct-catalog heads."""

    _authority_preflight()
    from sklearn.ensemble import ExtraTreesClassifier

    common = {
        "n_estimators": int(PROTOCOL["n_estimators"]),
        "min_samples_leaf": int(PROTOCOL["min_samples_leaf"]),
        "max_features": str(PROTOCOL["max_features"]),
        "class_weight": "balanced",
        "n_jobs": 1,
    }
    return {
        "direct_pair": ExtraTreesClassifier(
            **common, random_state=FIXED_CLASSIFIER_SEEDS["direct_pair"]
        ),
        "catalog_three_class": ExtraTreesClassifier(
            **common, random_state=FIXED_CLASSIFIER_SEEDS["catalog_three_class"]
        ),
    }


def _expected_estimator_params(seed: int) -> dict[str, Any]:
    return {
        "bootstrap": False, "ccp_alpha": 0.0, "class_weight": "balanced",
        "criterion": "gini", "max_depth": None, "max_features": "sqrt",
        "max_leaf_nodes": None, "max_samples": None, "min_impurity_decrease": 0.0,
        "min_samples_leaf": 2, "min_samples_split": 2,
        "min_weight_fraction_leaf": 0.0, "monotonic_cst": None,
        "n_estimators": 32, "n_jobs": 1, "oob_score": False,
        "random_state": seed, "verbose": 0, "warm_start": False,
    }


def _validate_fitted_estimators(estimators: Mapping[str, Any]) -> None:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import ExtraTreeClassifier

    if type(estimators) is not dict or set(estimators) != {"direct_pair", "catalog_three_class"}:
        raise SkeletonGraphDevelopmentError("estimator head inventory differs")
    specifications = (
        ("direct_pair", OBSERVED_TRAIN_PAIR_CLASS_ORDER, FIXED_CLASSIFIER_SEEDS["direct_pair"]),
        ("catalog_three_class", CATALOG_CLASS_ORDER, FIXED_CLASSIFIER_SEEDS["catalog_three_class"]),
    )
    for name, classes, seed in specifications:
        estimator = estimators[name]
        if (
            type(estimator) is not ExtraTreesClassifier
            or estimator.get_params(deep=False) != _expected_estimator_params(seed)
            or tuple(int(value) for value in getattr(estimator, "classes_", ())) != classes
            or getattr(estimator, "n_features_in_", None) != 112
            or type(getattr(estimator, "estimators_", None)) is not list
            or len(estimator.estimators_) != 32
            or any(
                type(tree) is not ExtraTreeClassifier
                or getattr(tree, "n_features_in_", None) != 112
                or getattr(getattr(tree, "tree_", None), "node_count", 0) <= 0
                for tree in estimator.estimators_
            )
        ):
            raise SkeletonGraphDevelopmentError(f"fitted {name} forest structure differs")


def _model_structure(estimators: Mapping[str, Any], serialized_bytes: int) -> dict[str, Any]:
    _validate_fitted_estimators(estimators)
    return {
        "serialized_bytes": serialized_bytes,
        "heads": {
            name: {
                "tree_count": len(estimator.estimators_),
                "tree_node_counts": [int(tree.tree_.node_count) for tree in estimator.estimators_],
                "total_node_count": sum(int(tree.tree_.node_count) for tree in estimator.estimators_),
            }
            for name, estimator in sorted(estimators.items())
        },
    }


def _build_model_bundle(
    estimators: Mapping[str, Any], *, precommit_record_digest: str
) -> dict[str, Any]:
    """Build the exact production pickle body around authenticated heads."""

    _validate_fitted_estimators(estimators)
    if not _ADDRESS.fullmatch(precommit_record_digest):
        raise SkeletonGraphDevelopmentError("model precommit address differs")
    return {
        "catalog_class_order": CATALOG_CLASS_ORDER,
        "config_digest": config_digest(),
        "estimators": estimators,
        "feature_names": FEATURE_NAMES,
        "observed_pair_class_order": OBSERVED_TRAIN_PAIR_CLASS_ORDER,
        "precommit_record_digest": precommit_record_digest,
        "runtime": runtime_fingerprint(),
        "schema": MODEL_SCHEMA,
        "source_sha256": source_sha256(),
        "valid_pair_class_order": VALID_PAIR_CLASS_ORDER,
    }


def fit_authoritative_estimators(features: np.ndarray, labels: np.ndarray):
    """Validate the frozen train vocabulary and fit exactly the two heads."""

    _authority_preflight()
    matrix = np.asarray(features, dtype=np.float32)
    target = np.asarray(labels)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(FEATURE_NAMES)
        or not np.isfinite(matrix).all()
        or target.shape != (len(matrix), 3)
        or not np.issubdtype(target.dtype, np.integer)
    ):
        raise SkeletonGraphDevelopmentError("training matrix or labels differ")
    straight, arc, catalog = (target[:, index].astype(np.int64) for index in range(3))
    pairs = 10 * straight + arc
    if any(int(value) not in VALID_PAIR_CLASS_ORDER for value in pairs):
        raise SkeletonGraphDevelopmentError("training contains an invalid action pair")
    if tuple(sorted(int(value) for value in np.unique(pairs))) != OBSERVED_TRAIN_PAIR_CLASS_ORDER:
        raise SkeletonGraphDevelopmentError("observed training pair vocabulary differs")
    if tuple(sorted(int(value) for value in np.unique(catalog))) != CATALOG_CLASS_ORDER:
        raise SkeletonGraphDevelopmentError("observed catalog vocabulary differs")
    estimators = build_authoritative_estimators()
    estimators["direct_pair"].fit(matrix, pairs)
    estimators["catalog_three_class"].fit(matrix, catalog)
    if tuple(int(value) for value in estimators["direct_pair"].classes_) != OBSERVED_TRAIN_PAIR_CLASS_ORDER:
        raise SkeletonGraphDevelopmentError("fitted pair class order differs")
    if tuple(int(value) for value in estimators["catalog_three_class"].classes_) != CATALOG_CLASS_ORDER:
        raise SkeletonGraphDevelopmentError("fitted catalog class order differs")
    _validate_fitted_estimators(estimators)
    return estimators


def predict_authoritative_probabilities(
    estimators: Mapping[str, Any], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return direct joint and direct three-class probabilities in bound order."""

    _authority_preflight()
    _validate_fitted_estimators(estimators)
    pair_head, catalog_head = estimators["direct_pair"], estimators["catalog_three_class"]
    if tuple(int(value) for value in pair_head.classes_) != OBSERVED_TRAIN_PAIR_CLASS_ORDER:
        raise SkeletonGraphDevelopmentError("pair estimator class order differs")
    if tuple(int(value) for value in catalog_head.classes_) != CATALOG_CLASS_ORDER:
        raise SkeletonGraphDevelopmentError("catalog estimator class order differs")
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 112 or not np.isfinite(matrix).all():
        raise SkeletonGraphDevelopmentError("prediction feature matrix differs")
    pair = np.ascontiguousarray(pair_head.predict_proba(matrix), dtype="<f8")
    catalog = np.ascontiguousarray(catalog_head.predict_proba(matrix), dtype="<f8")
    if pair.shape != (len(matrix), 33) or catalog.shape != (len(matrix), 3):
        raise SkeletonGraphDevelopmentError("probability matrix shape differs")
    if (
        not np.isfinite(pair).all()
        or not np.isfinite(catalog).all()
        or not np.allclose(pair.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        or not np.allclose(catalog.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise SkeletonGraphDevelopmentError("probability normalization differs")
    return pair, catalog


def pair_marginals(
    pair_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum the learned direct joint; this never replaces joint calibration."""

    values = np.asarray(pair_probabilities, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(OBSERVED_TRAIN_PAIR_CLASS_ORDER)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise SkeletonGraphDevelopmentError("direct-pair probability shape differs")
    straight = np.zeros((len(values), 10), dtype="<f8")
    arc = np.zeros((len(values), 10), dtype="<f8")
    for column, encoded in enumerate(OBSERVED_TRAIN_PAIR_CLASS_ORDER):
        straight_count, arc_count = decode_pair_class(encoded)
        straight[:, straight_count] += values[:, column]
        arc[:, arc_count] += values[:, column]
    return straight, arc


def catalog_candidate_projection(candidates: Sequence[int]) -> dict[str, Any]:
    """Project catalog candidates; any unresolved candidate makes the axis a GAP."""

    if any(type(value) is not int for value in candidates):
        raise SkeletonGraphDevelopmentError("catalog candidate must be an exact integer")
    values = tuple(sorted(set(candidates)))
    if not values or any(value not in CATALOG_CLASS_ORDER for value in values):
        raise SkeletonGraphDevelopmentError("catalog candidate set differs")
    if -1 in values:
        return {
            "disposition": "indeterminate",
            "reason": "catalog_unresolved_in_candidate_set",
            "candidates": [],
        }
    if len(values) > 1:
        return {
            "disposition": "indeterminate",
            "reason": "multiple_catalog_candidates",
            "candidates": list(values),
        }
    return {
        "disposition": "present",
        "reason": "singleton_catalog_candidate",
        "candidates": list(values),
    }


_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"hd/(?P<task>hd_[^/]+)/(?P<side>[01])/(?P<ordinal>[0-6])\.png\Z")


@dataclass(frozen=True)
class DevelopmentGroup:
    index: int
    cohort: str
    png_sha256: str
    png_size_bytes: int
    panel_ids: tuple[str, ...]
    representative_panel_id: str
    task_ids: tuple[str, ...]
    labels: tuple[int, int, int]
    metric_strata: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class WallDeadline:
    started: float
    maximum_seconds: float

    @classmethod
    def start(cls, maximum_seconds: float) -> "WallDeadline":
        if not 1 <= maximum_seconds <= 600:
            raise SkeletonGraphDevelopmentError("wall limit must be in [1,600] seconds")
        return cls(time.monotonic(), float(maximum_seconds))

    def check(self) -> None:
        if time.monotonic() - self.started > self.maximum_seconds:
            raise SkeletonGraphDevelopmentError("development wall deadline exceeded")


def _address(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    plain = _plain(body)
    return {**plain, "record_digest": "sha256:" + canonical_digest(plain)}


def _source_file_address(module_name: str) -> str:
    module = sys.modules.get(module_name)
    path = Path(str(getattr(module, "__file__", ""))).resolve()
    return _address(_stable_regular_bytes(path, maximum=4 * 1024 * 1024))


def runtime_fingerprint() -> dict[str, Any]:
    """Return the exact runtime/dependency surface bound by artifacts."""

    _authority_preflight()
    return {
        "byteorder": sys.byteorder,
        "machine": platform.machine(),
        "numpy": importlib.metadata.version("numpy"),
        "pillow": importlib.metadata.version("Pillow"),
        "platform": platform.system(),
        "python": platform.python_version(),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "scipy": importlib.metadata.version("scipy"),
        "thread_policy": {"extra_trees_n_jobs": 1},
    }


def dependency_source_addresses() -> dict[str, str]:
    _authority_preflight()
    return {
        "bongard.canonical": _source_file_address("bongard.canonical"),
        "bongard.panel_action_local_duplicate_conflict_audit": _source_file_address(
            "bongard.panel_action_local_duplicate_conflict_audit"
        ),
        "bongard.panel_action_local_supervision_authority": _source_file_address(
            "bongard.panel_action_local_supervision_authority"
        ),
        "bongard.runtime_source_snapshot": _source_file_address(
            "bongard.runtime_source_snapshot"
        ),
    }


def config_digest() -> str:
    _authority_preflight()
    return "sha256:" + canonical_digest(
        _plain({
            "adaptive_variant_ledger": ADAPTIVE_VARIANT_LEDGER,
            "capacity_selection_ledger": CAPACITY_SELECTION_LEDGER,
            "catalog_class_order": CATALOG_CLASS_ORDER,
            "carrier_signature_cv": CARRIER_SIGNATURE_CV_DIAGNOSTIC,
            "claim_scope": CLAIM_SCOPE,
            "engineering_thresholds": ENGINEERING_THRESHOLDS,
            "feature_names": FEATURE_NAMES,
            "fit_precommit_record_digest": FIT_PRECOMMIT_RECORD_DIGEST,
            "observed_train_pair_class_order": OBSERVED_TRAIN_PAIR_CLASS_ORDER,
            "protocol": PROTOCOL,
            "scale_specs": SCALE_SPECS,
            "source_sha256": source_sha256(),
            "valid_pair_class_order": VALID_PAIR_CLASS_ORDER,
        })
    )


def _stable_regular_bytes(path: Path, *, maximum: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise SkeletonGraphDevelopmentError(f"cannot stat artifact {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SkeletonGraphDevelopmentError(f"artifact is not a regular nonsymlink file: {path}")
    if before.st_size <= 0 or before.st_size > maximum:
        raise SkeletonGraphDevelopmentError(f"artifact byte count is outside bound: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise SkeletonGraphDevelopmentError(f"artifact exceeds byte cap: {path}")
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    fingerprint = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    if not (fingerprint(before) == fingerprint(opened) == fingerprint(after_open) == fingerprint(after)):
        raise SkeletonGraphDevelopmentError(f"artifact changed during read: {path}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise SkeletonGraphDevelopmentError(f"artifact read length differs: {path}")
    return raw


def _load_canonical_record(path: Path, *, schema: str, maximum: int) -> dict[str, Any]:
    raw = _stable_regular_bytes(path, maximum=maximum)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SkeletonGraphDevelopmentError(f"cannot decode {path}: {exc}") from exc
    if type(value) is not dict or raw != canonical_json(value) + b"\n":
        raise SkeletonGraphDevelopmentError(f"artifact is not canonical JSON plus newline: {path}")
    body = dict(value)
    digest = body.pop("record_digest", None)
    if value.get("schema") != schema or digest != "sha256:" + canonical_digest(body):
        raise SkeletonGraphDevelopmentError(f"artifact schema or record digest differs: {path}")
    return value


def _verify_committed_label_authority_audit(repository_root: Path) -> dict[str, Any]:
    path = repository_root / COMMITTED_LABEL_AUTHORITY_AUDIT_RELATIVE_PATH
    artifact = _load_canonical_record(
        path, schema=conflict_module.AUDIT_SCHEMA, maximum=32 * 1024 * 1024
    )
    if artifact.get("record_digest") != COMMITTED_LABEL_AUTHORITY_AUDIT_RECORD_DIGEST:
        raise SkeletonGraphDevelopmentError("committed label-authority audit digest differs")
    try:
        conflict_module.verify_duplicate_target_conflict_audit(
            artifact, repository_root=repository_root
        )
    except Exception as exc:
        raise SkeletonGraphDevelopmentError(
            f"committed label-authority audit replay failed: {exc}"
        ) from exc
    result = artifact.get("result")
    cohorts = artifact.get("cohorts")
    if (
        type(result) is not dict
        or result.get("all_effective_png_groups_descriptor_loss_eligible") is not True
        or result.get("pose_free_target_conflict_group_count") != 0
        or result.get("straight_arc_pair_conflict_group_count") != 0
        or result.get("catalog_convexity_conflict_group_count") != 0
        or type(cohorts) is not dict
        or cohorts.get("train", {}).get("descriptor_loss_eligibility", {}).get(
            "eligible_group_count"
        ) != 11_143
        or cohorts.get("validation", {}).get("descriptor_loss_eligibility", {}).get(
            "eligible_group_count"
        ) != 1_392
    ):
        raise SkeletonGraphDevelopmentError("committed label-authority audit did not pass")
    return {
        "record_digest": artifact["record_digest"],
        "audit_source_sha256": artifact["bindings"]["audit_source_sha256"],
        "all_effective_groups_eligible": True,
        "straight_arc_pair_conflict_group_count": 0,
        "catalog_convexity_conflict_group_count": 0,
    }


def _directory_identity(value: os.stat_result) -> dict[str, int]:
    return {"st_dev": int(value.st_dev), "st_ino": int(value.st_ino), "st_mode": int(value.st_mode)}


def _open_nonsymlink_directory(path: Path) -> int:
    absolute = path.absolute()
    if not absolute.is_absolute():  # pragma: no cover - Path.absolute guarantees this
        raise SkeletonGraphDevelopmentError("output directory is not absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open("/", flags)
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(
                part,
                flags | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _write_once(
    path: Path,
    payload: bytes,
    *,
    expected_parent_identity: Mapping[str, int],
) -> None:
    if path.name in {"", ".", ".."} or path.parent == path:
        raise SkeletonGraphDevelopmentError("output filename differs")
    parent_descriptor = _open_nonsymlink_directory(path.parent)
    if _directory_identity(os.fstat(parent_descriptor)) != dict(expected_parent_identity):
        os.close(parent_descriptor)
        raise SkeletonGraphDevelopmentError("output root identity changed after precommit")
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise SkeletonGraphDevelopmentError(f"refusing to overwrite {path}") from exc
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _write_record_once(
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_parent_identity: Mapping[str, int],
) -> None:
    _write_once(
        path,
        canonical_json(value) + b"\n",
        expected_parent_identity=expected_parent_identity,
    )


def _verify_prior_failed_capacity_attempt() -> dict[str, Any]:
    """Replay the inode-bound v1 failure state without writing to that root."""

    root = Path(PRIOR_FAILED_OUTPUT_ROOT)
    try:
        descriptor = _open_nonsymlink_directory(root)
    except OSError as exc:
        raise SkeletonGraphDevelopmentError(
            f"prior failed v1 root is unavailable or symlinked: {exc}"
        ) from exc
    expected_identity = dict(PRIOR_FAILED_OUTPUT_ROOT_IDENTITY)
    try:
        if _directory_identity(os.fstat(descriptor)) != expected_identity:
            raise SkeletonGraphDevelopmentError("prior failed v1 root identity differs")
        if sorted(os.listdir(descriptor)) != ["precommit.json"]:
            raise SkeletonGraphDevelopmentError(
                "prior failed v1 root has post-precommit artifacts"
            )
        flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        precommit_descriptor = os.open("precommit.json", flags, dir_fd=descriptor)
        try:
            before = os.fstat(precommit_descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size <= 0
                or before.st_size > 4 * 1024 * 1024
            ):
                raise SkeletonGraphDevelopmentError(
                    "prior failed v1 precommit is not a bounded regular file"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    precommit_descriptor,
                    min(1 << 20, 4 * 1024 * 1024 + 1 - total),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > 4 * 1024 * 1024:
                    raise SkeletonGraphDevelopmentError(
                        "prior failed v1 precommit exceeds its byte cap"
                    )
            after = os.fstat(precommit_descriptor)
        finally:
            os.close(precommit_descriptor)
        fingerprint = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns,
        )
        if fingerprint(before) != fingerprint(after):
            raise SkeletonGraphDevelopmentError(
                "prior failed v1 precommit changed during verification"
            )
        raw = b"".join(chunks)
        if len(raw) != before.st_size or _address(raw) != PRIOR_FAILED_PRECOMMIT_FILE_SHA256:
            raise SkeletonGraphDevelopmentError("prior failed v1 precommit address differs")
        try:
            value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise SkeletonGraphDevelopmentError(
                f"cannot decode prior failed v1 precommit: {exc}"
            ) from exc
        body = dict(value) if type(value) is dict else {}
        record_digest = body.pop("record_digest", None)
        expected_outputs = {
            "features": str(root / "features.json"),
            "model": str(root / "model.pkl"),
            "predictions": str(root / "predictions.json"),
            "replay": str(root / "replay.json"),
            "result": str(root / "result.json"),
        }
        if (
            type(value) is not dict
            or raw != canonical_json(value) + b"\n"
            or value.get("schema") != PRIOR_FAILED_PRECOMMIT_SCHEMA
            or record_digest != PRIOR_FAILED_PRECOMMIT_RECORD_DIGEST
            or record_digest != "sha256:" + canonical_digest(body)
            or value.get("source_sha256") != BASE_PROTOCOL_SOURCE_SHA256
            or value.get("output_root") != PRIOR_FAILED_OUTPUT_ROOT
            or value.get("output_root_identity") != expected_identity
            or value.get("intended_outputs") != expected_outputs
        ):
            raise SkeletonGraphDevelopmentError("prior failed v1 precommit policy differs")
        if _directory_identity(os.fstat(descriptor)) != expected_identity:
            raise SkeletonGraphDevelopmentError(
                "prior failed v1 root changed during verification"
            )
        if sorted(os.listdir(descriptor)) != ["precommit.json"]:
            raise SkeletonGraphDevelopmentError(
                "prior failed v1 root changed during verification"
            )
    finally:
        os.close(descriptor)
    return {
        "base_protocol_commit": BASE_PROTOCOL_COMMIT,
        "base_protocol_source_sha256": BASE_PROTOCOL_SOURCE_SHA256,
        "precommit_schema": PRIOR_FAILED_PRECOMMIT_SCHEMA,
        "precommit_record_digest": PRIOR_FAILED_PRECOMMIT_RECORD_DIGEST,
        "precommit_file_sha256": PRIOR_FAILED_PRECOMMIT_FILE_SHA256,
        "output_root": PRIOR_FAILED_OUTPUT_ROOT,
        "output_root_identity": expected_identity,
        "durable_file_names": ["precommit.json"],
        "absent_intended_output_names": list(PRIOR_FAILED_INTENDED_OUTPUT_NAMES),
        "durable_output_count_beyond_precommit": 0,
        "verified_read_only": True,
    }


def _task_from_panel_id(panel_id: str) -> str:
    match = _PANEL_ID.fullmatch(panel_id)
    if match is None:
        raise SkeletonGraphDevelopmentError("development panel ID syntax differs")
    task = match.group("task")
    if task.startswith("hd_convex-has_four_straight_lines_"):
        raise SkeletonGraphDevelopmentError("target family entered development custody")
    return task


def load_fit_inventory(path: Path) -> tuple[dict[str, Any], tuple[DevelopmentGroup, ...], dict[str, Any]]:
    """Load and replay the exact effective v1 development custody metadata."""

    _authority_preflight()
    raw = _stable_regular_bytes(path, maximum=32 * 1024 * 1024)
    if _address(raw) != FIT_PRECOMMIT_FILE_SHA256:
        raise SkeletonGraphDevelopmentError("fit precommit file address differs")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SkeletonGraphDevelopmentError(f"cannot decode fit precommit: {exc}") from exc
    if type(value) is not dict or raw != canonical_json(value) + b"\n":
        raise SkeletonGraphDevelopmentError("fit precommit is not canonical JSON plus newline")
    body = dict(value)
    digest = body.pop("record_digest", None)
    if (
        value.get("schema") != FIT_PRECOMMIT_SCHEMA
        or digest != FIT_PRECOMMIT_RECORD_DIGEST
        or digest != "sha256:" + canonical_digest(body)
        or value.get("validation_decontamination_gate", {}).get("passed") is not True
    ):
        raise SkeletonGraphDevelopmentError("fit precommit binding or decontamination gate differs")
    raw_groups = value.get("path_independent_digest_groups")
    observations = value.get("exact_png_observations")
    removed = value.get("validation_removed_due_exact_train_duplicate")
    if type(raw_groups) is not list or type(observations) is not list or type(removed) is not dict:
        raise SkeletonGraphDevelopmentError("fit inventory structure differs")
    observation_by_id: dict[str, Any] = {}
    for row in observations:
        if type(row) is not dict or type(row.get("panel_id")) is not str:
            raise SkeletonGraphDevelopmentError("fit observation structure differs")
        if row["panel_id"] in observation_by_id:
            raise SkeletonGraphDevelopmentError("duplicate panel ID in fit observations")
        observation_by_id[row["panel_id"]] = row
    groups: list[DevelopmentGroup] = []
    effective_ids: set[str] = set()
    cohort_digests: dict[str, set[str]] = {"train": set(), "validation": set()}
    cohort_tasks: dict[str, set[str]] = {"train": set(), "validation": set()}
    occurrence_counts = {"train": 0, "validation": 0}
    for index, row in enumerate(raw_groups):
        if type(row) is not dict or row.get("fit_cohort") not in cohort_digests:
            raise SkeletonGraphDevelopmentError("fit group cohort differs")
        cohort = str(row["fit_cohort"])
        panel_ids = row.get("panel_ids")
        labels = row.get("label_triple")
        metric_strata = row.get("metric_strata")
        digest_value = row.get("png_sha256")
        size = row.get("png_size_bytes")
        if (
            type(panel_ids) is not list
            or not panel_ids
            or row.get("multiplicity") != len(panel_ids)
            or type(labels) is not list
            or len(labels) != 3
            or any(type(item) is not int for item in labels)
            or type(metric_strata) is not list
            or not _ADDRESS.fullmatch(str(digest_value))
            or type(size) is not int
            or not 1 <= size <= 16 * 1024 * 1024
        ):
            raise SkeletonGraphDevelopmentError("fit group fields differ")
        straight, arc, catalog = labels
        if 10 * straight + arc not in VALID_PAIR_CLASS_ORDER or catalog not in CATALOG_CLASS_ORDER:
            raise SkeletonGraphDevelopmentError("fit group label is outside the frozen universe")
        task_ids: list[str] = []
        for panel_id in panel_ids:
            if type(panel_id) is not str or panel_id in effective_ids:
                raise SkeletonGraphDevelopmentError("effective panel inventory overlaps")
            effective_ids.add(panel_id)
            task = _task_from_panel_id(panel_id)
            task_ids.append(task)
            observed = observation_by_id.get(panel_id)
            if (
                type(observed) is not dict
                or observed.get("fit_cohort") != cohort
                or observed.get("png_sha256") != digest_value
                or observed.get("png_size_bytes") != size
                or observed.get("label_triple") != labels
            ):
                raise SkeletonGraphDevelopmentError("group-to-observation custody join differs")
        cohort_tasks[cohort].update(task_ids)
        if digest_value in cohort_digests[cohort]:
            raise SkeletonGraphDevelopmentError("digest appears in multiple groups")
        cohort_digests[cohort].add(str(digest_value))
        occurrence_counts[cohort] += len(panel_ids)
        groups.append(
            DevelopmentGroup(
                index=index, cohort=cohort, png_sha256=str(digest_value),
                png_size_bytes=size, panel_ids=tuple(panel_ids),
                representative_panel_id=panel_ids[0], task_ids=tuple(task_ids),
                labels=(straight, arc, catalog),
                metric_strata=tuple(_deep_freeze(item) for item in metric_strata),
            )
        )
    group_counts = {
        cohort: sum(group.cohort == cohort for group in groups)
        for cohort in ("train", "validation")
    }
    removed_ids = removed.get("panel_ids")
    if (
        len(observations) != 12_600
        or len(groups) != 12_535
        or group_counts != {"train": 11_143, "validation": 1_392}
        or occurrence_counts != {"train": 11_200, "validation": 1_392}
        or len(effective_ids) != 12_592
        or type(removed_ids) is not list
        or len(removed_ids) != 8
        or set(observation_by_id) - effective_ids != set(removed_ids)
        or effective_ids & set(removed_ids)
        or cohort_digests["train"] & cohort_digests["validation"]
        or len(cohort_tasks["train"]) != 800
        or len(cohort_tasks["validation"]) != 100
        or cohort_tasks["train"] & cohort_tasks["validation"]
    ):
        raise SkeletonGraphDevelopmentError("effective cohort count/disjointness custody differs")
    audit = {
        "effective_group_counts": group_counts,
        "effective_occurrence_counts": occurrence_counts,
        "removed_validation_panel_count": 8,
        "task_counts": {key: len(value_) for key, value_ in cohort_tasks.items()},
        "cross_cohort_png_digest_overlap": 0,
        "cross_cohort_task_overlap": 0,
    }
    return value, tuple(groups), audit


def create_development_precommit(
    *,
    repository_root: Path,
    dataset_root: Path,
    fit_precommit_path: Path,
    model_path: Path,
    feature_manifest_path: Path,
    predictions_path: Path,
    result_path: Path,
    replay_path: Path,
    output_path: Path,
    maximum_seconds: float = 600.0,
) -> dict[str, Any]:
    """Write the adaptive dev precommit without opening a PNG."""

    if not 60 <= maximum_seconds <= 600:
        raise SkeletonGraphDevelopmentError("command wall limit must be in [60,600] seconds")
    deadline = WallDeadline.start(maximum_seconds)
    fit, _groups, inventory = load_fit_inventory(fit_precommit_path)
    label_authority_audit = _verify_committed_label_authority_audit(
        repository_root.resolve(strict=True)
    )
    prior_failed_capacity_attempt = _verify_prior_failed_capacity_attempt()
    deadline.check()
    outputs = {
        "model": str(model_path.resolve()),
        "features": str(feature_manifest_path.resolve()),
        "predictions": str(predictions_path.resolve()),
        "result": str(result_path.resolve()),
        "replay": str(replay_path.resolve()),
    }
    output_root = output_path.parent.absolute()
    prior_root = Path(PRIOR_FAILED_OUTPUT_ROOT).absolute()
    if (
        output_root == prior_root
        or output_root.is_relative_to(prior_root)
        or prior_root.is_relative_to(output_root)
    ):
        raise SkeletonGraphDevelopmentError(
            "v2 capacity repair requires a fresh disjoint output root"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    output_root_descriptor = _open_nonsymlink_directory(output_root)
    output_root_identity = _directory_identity(os.fstat(output_root_descriptor))
    os.close(output_root_descriptor)
    resolved_output_root = output_root.resolve(strict=True)
    if (
        str(output_path.resolve()) in outputs.values()
        or len(set(outputs.values())) != len(outputs)
        or any(Path(value).parent != resolved_output_root for value in outputs.values())
        or any(Path(value).exists() for value in outputs.values())
    ):
        raise SkeletonGraphDevelopmentError("intended outputs overlap or already exist")
    body = {
        "adaptive_variant_ledger": ADAPTIVE_VARIANT_LEDGER,
        "authorized_input": "already_exposed_decontaminated_development_only",
        "capacity_selection_ledger": CAPACITY_SELECTION_LEDGER,
        "carrier_signature_cv_diagnostic": CARRIER_SIGNATURE_CV_DIAGNOSTIC,
        "claim_scope": CLAIM_SCOPE,
        "config_digest": config_digest(),
        "dataset_root": str(dataset_root.resolve()),
        "dependency_source_addresses": dependency_source_addresses(),
        "engineering_thresholds": ENGINEERING_THRESHOLDS,
        "fit_inventory_audit": inventory,
        "fit_precommit_path": str(fit_precommit_path.resolve()),
        "fit_precommit_record_digest": fit["record_digest"],
        "fit_precommit_file_sha256": FIT_PRECOMMIT_FILE_SHA256,
        "forbidden_inputs": [
            "removed_validation_duplicates", "calibration", "evaluation",
            "same_family_calibration", "same_family_evaluation", "target", "query",
        ],
        "intended_outputs": outputs,
        "label_authority_audit": label_authority_audit,
        "maximum_seconds": float(maximum_seconds),
        "pixels_read_by_precommit": 0,
        "prior_failed_capacity_attempt": prior_failed_capacity_attempt,
        "output_root": str(resolved_output_root),
        "output_root_identity": output_root_identity,
        "promotion": False,
        "promotion_requires": PROMOTION_REQUIRES,
        "protocol": PROTOCOL,
        "repository_root": str(repository_root.resolve()),
        "runtime": runtime_fingerprint(),
        "schema": SCHEMA_PRECOMMIT,
        "source_sha256": source_sha256(),
        "validation_status": "adaptive_architecture_selection_development",
    }
    value = _seal(body)
    _write_record_once(
        output_path,
        value,
        expected_parent_identity=output_root_identity,
    )
    if _load_canonical_record(output_path, schema=SCHEMA_PRECOMMIT, maximum=4 * 1024 * 1024) != value:
        raise SkeletonGraphDevelopmentError("fresh precommit reload differs")
    return value


def _load_development_precommit(
    path: Path, *, expected_record_digest: str
) -> dict[str, Any]:
    value = _load_canonical_record(path, schema=SCHEMA_PRECOMMIT, maximum=4 * 1024 * 1024)
    expected_forbidden = [
        "removed_validation_duplicates", "calibration", "evaluation",
        "same_family_calibration", "same_family_evaluation", "target", "query",
    ]
    if (
        value.get("record_digest") != expected_record_digest
        or not _ADDRESS.fullmatch(str(expected_record_digest))
        or value.get("source_sha256") != source_sha256()
        or value.get("config_digest") != config_digest()
        or value.get("dependency_source_addresses") != dependency_source_addresses()
        or value.get("runtime") != runtime_fingerprint()
        or value.get("fit_precommit_record_digest") != FIT_PRECOMMIT_RECORD_DIGEST
        or value.get("fit_precommit_file_sha256") != FIT_PRECOMMIT_FILE_SHA256
        or value.get("pixels_read_by_precommit") != 0
        or value.get("promotion") is not False
        or value.get("promotion_requires") != PROMOTION_REQUIRES
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("authorized_input") != "already_exposed_decontaminated_development_only"
        or value.get("forbidden_inputs") != expected_forbidden
        or value.get("protocol") != _plain(PROTOCOL)
        or value.get("adaptive_variant_ledger") != _plain(ADAPTIVE_VARIANT_LEDGER)
        or value.get("capacity_selection_ledger") != _plain(CAPACITY_SELECTION_LEDGER)
        or value.get("carrier_signature_cv_diagnostic") != _plain(CARRIER_SIGNATURE_CV_DIAGNOSTIC)
        or value.get("engineering_thresholds") != _plain(ENGINEERING_THRESHOLDS)
        or value.get("label_authority_audit")
        != _verify_committed_label_authority_audit(Path(value["repository_root"]))
        or value.get("prior_failed_capacity_attempt")
        != _verify_prior_failed_capacity_attempt()
        or value.get("fit_inventory_audit")
        != {
            "cross_cohort_png_digest_overlap": 0,
            "cross_cohort_task_overlap": 0,
            "effective_group_counts": {"train": 11_143, "validation": 1_392},
            "effective_occurrence_counts": {"train": 11_200, "validation": 1_392},
            "removed_validation_panel_count": 8,
            "task_counts": {"train": 800, "validation": 100},
        }
        or not isinstance(value.get("maximum_seconds"), (int, float))
        or not 60 <= float(value["maximum_seconds"]) <= 600
    ):
        raise SkeletonGraphDevelopmentError("development precommit policy differs")
    outputs = value.get("intended_outputs")
    if type(outputs) is not dict or set(outputs) != {"model", "features", "predictions", "result", "replay"}:
        raise SkeletonGraphDevelopmentError("development output inventory differs")
    output_paths = [Path(item) for item in outputs.values()]
    output_root = Path(str(value.get("output_root", "")))
    prior_root = Path(PRIOR_FAILED_OUTPUT_ROOT)
    identity = value.get("output_root_identity")
    if (
        not output_root.is_absolute()
        or output_root == prior_root
        or output_root.is_relative_to(prior_root)
        or prior_root.is_relative_to(output_root)
        or type(identity) is not dict
        or set(identity) != {"st_dev", "st_ino", "st_mode"}
        or any(not item.is_absolute() for item in output_paths)
        or len(set(output_paths)) != 5
        or len({item.parent for item in output_paths}) != 1
        or any(item.parent != output_root for item in output_paths)
        or path.absolute().parent != output_root
        or path.resolve() in output_paths
    ):
        raise SkeletonGraphDevelopmentError("development output paths differ")
    try:
        descriptor = _open_nonsymlink_directory(output_root)
    except OSError as exc:
        raise SkeletonGraphDevelopmentError(
            f"output root path is no longer nonsymlinked: {exc}"
        ) from exc
    try:
        if _directory_identity(os.fstat(descriptor)) != identity:
            raise SkeletonGraphDevelopmentError("output root identity changed after precommit")
    finally:
        os.close(descriptor)
    return value


def _panel_path(dataset_root: Path, panel_id: str) -> Path:
    task = _task_from_panel_id(panel_id)
    match = _PANEL_ID.fullmatch(panel_id)
    assert match is not None
    supplied = dataset_root.absolute()
    supplied_stat = supplied.lstat()
    if stat.S_ISLNK(supplied_stat.st_mode) or not stat.S_ISDIR(supplied_stat.st_mode):
        raise SkeletonGraphDevelopmentError("dataset root is not a real directory")
    root = supplied.resolve(strict=True)
    candidate = root / "hd/images" / task / match.group("side") / f"{match.group('ordinal')}.png"
    if candidate.resolve(strict=True) != candidate.absolute():
        raise SkeletonGraphDevelopmentError("panel path crosses a symlink")
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover - guarded by the syntax regex
        raise SkeletonGraphDevelopmentError("panel path escapes dataset root") from exc
    return candidate


def _array_digest(name: str, array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = canonical_json(
        {"dtype": value.dtype.str, "name": name, "shape": list(value.shape)}
    )
    return "sha256:" + hashlib.sha256(header + b"\0" + value.tobytes(order="C")).hexdigest()


def _carrier_authority(
    repository_root: Path, authority: Any, panel_id: str
) -> tuple[str, tuple[int, int]]:
    supervision = compile_pose_free_panel(authority, panel_id).to_data()
    straight = 0
    arc = 0
    for shape in supervision.get("shape_multiset", []):
        shape_multiplicity = shape.get("multiplicity")
        if type(shape_multiplicity) is not int or shape_multiplicity <= 0:
            raise SkeletonGraphDevelopmentError("authority shape multiplicity differs")
        for action in shape.get("action_multiset", []):
            multiplicity = action.get("multiplicity")
            primitive = action.get("primitive")
            if type(multiplicity) is not int or multiplicity <= 0 or primitive not in {"line", "arc"}:
                raise SkeletonGraphDevelopmentError("authority action multiset differs")
            if primitive == "line":
                straight += shape_multiplicity * multiplicity
            else:
                arc += shape_multiplicity * multiplicity
    return (
        "sha256:" + canonical_digest(_pose_free_target(supervision)),
        (straight, arc),
    )


def _materialize_feature_bank(
    *,
    repository_root: Path,
    dataset_root: Path,
    groups: Sequence[DevelopmentGroup],
    deadline: WallDeadline,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, Any], ...], dict[str, Any]]:
    """Open exactly one representative per effective digest group."""

    _authority_preflight()
    authority = load_development_authority(repository_root=repository_root)
    features: list[np.ndarray] = []
    labels: list[tuple[int, int, int]] = []
    manifest: list[dict[str, Any]] = []
    signatures: list[str] = []
    for index, group in enumerate(groups):
        if index % 32 == 0:
            deadline.check()
            _authority_preflight()
        path = _panel_path(dataset_root, group.representative_panel_id)
        raw = _stable_regular_bytes(path, maximum=16 * 1024 * 1024)
        if len(raw) != group.png_size_bytes or _address(raw) != group.png_sha256:
            raise SkeletonGraphDevelopmentError("representative PNG changed after precommit")
        vector = extract_feature_vector(raw)
        authority_by_panel = {
            panel_id: _carrier_authority(repository_root, authority, panel_id)
            for panel_id in group.panel_ids
        }
        signature_by_panel = {
            panel_id: value[0] for panel_id, value in authority_by_panel.items()
        }
        if (
            len(set(signature_by_panel.values())) != 1
            or any(value[1] != group.labels[:2] for value in authority_by_panel.values())
        ):
            raise SkeletonGraphDevelopmentError(
                "PNG group label or pose-free carrier authority differs"
            )
        signature = signature_by_panel[group.representative_panel_id]
        features.append(vector)
        labels.append(group.labels)
        signatures.append(signature)
        manifest.append(
            {
                "cohort": group.cohort,
                "feature_digest": _array_digest("feature_row", vector),
                "group_index": group.index,
                "labels": list(group.labels),
                "metric_strata": group.metric_strata,
                "multiplicity": len(group.panel_ids),
                "panel_ids": group.panel_ids,
                "panel_id": group.representative_panel_id,
                "png_sha256": group.png_sha256,
                "pose_free_carrier_signature": signature,
                "task_ids": group.task_ids,
            }
        )
    matrix = np.ascontiguousarray(np.stack(features), dtype="<f4")
    targets = np.ascontiguousarray(np.asarray(labels, dtype="<i8"))
    cohort = np.asarray([group.cohort for group in groups])
    train = cohort == "train"
    validation = cohort == "validation"
    train_rows = {matrix[index].tobytes() for index in np.flatnonzero(train)}
    feature_overlap = sum(
        matrix[index].tobytes() in train_rows for index in np.flatnonzero(validation)
    )
    train_signatures = {signatures[index] for index in np.flatnonzero(train)}
    validation_signatures = {signatures[index] for index in np.flatnonzero(validation)}
    scope = {
        "feature_vector_validation_to_train_exact_overlap": feature_overlap,
        "train_pose_free_carrier_signature_count": len(train_signatures),
        "validation_groups_with_train_seen_signature": sum(
            signatures[index] in train_signatures for index in np.flatnonzero(validation)
        ),
        "validation_pose_free_carrier_signature_count": len(validation_signatures),
        "validation_pose_free_carrier_signatures_all_train_seen": (
            validation_signatures <= train_signatures
        ),
    }
    if (
        matrix.shape != (12_535, 112)
        or targets.shape != (12_535, 3)
        or int(train.sum()) != 11_143
        or int(validation.sum()) != 1_392
        or scope
        != {
            "feature_vector_validation_to_train_exact_overlap": 0,
            "train_pose_free_carrier_signature_count": 625,
            "validation_groups_with_train_seen_signature": 1_392,
            "validation_pose_free_carrier_signature_count": 512,
            "validation_pose_free_carrier_signatures_all_train_seen": True,
        }
    ):
        raise SkeletonGraphDevelopmentError("materialized feature bank scope differs")
    deadline.check()
    return matrix, targets, tuple(manifest), scope


@contextmanager
def _hard_wall_timeout(seconds: float):
    if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
        raise SkeletonGraphDevelopmentError("hard wall timer is unavailable")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum: int, _frame: Any) -> None:
        raise SkeletonGraphDevelopmentError("hard development wall timeout expired")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.asarray(truth) == np.asarray(prediction)))


def _balanced_accuracy(
    truth: np.ndarray, prediction: np.ndarray, *, classes: Sequence[int] | None = None
) -> float:
    actual = np.asarray(truth)
    guessed = np.asarray(prediction)
    vocabulary = tuple(int(item) for item in (classes if classes is not None else np.unique(actual)))
    recalls = []
    for value in vocabulary:
        mask = actual == value
        if not mask.any():
            raise SkeletonGraphDevelopmentError("balanced-accuracy class has no support")
        recalls.append(float(np.mean(guessed[mask] == value)))
    return float(np.mean(recalls))


def _metric_summary(
    groups: Sequence[DevelopmentGroup],
    pair_prediction: np.ndarray,
    catalog_prediction: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray([group.labels for group in groups], dtype=np.int64)
    true_pair = 10 * truth[:, 0] + truth[:, 1]
    pair_prediction = np.asarray(pair_prediction, dtype=np.int64)
    catalog_prediction = np.asarray(catalog_prediction, dtype=np.int64)
    decoded = np.asarray([decode_pair_class(value) for value in pair_prediction], dtype=np.int64)
    known = truth[:, 2] != -1
    known_vocabulary = set(int(value) for value in np.unique(truth[known, 2]))
    known_balanced = (
        _balanced_accuracy(truth[known, 2], catalog_prediction[known], classes=(0, 1))
        if known_vocabulary == {0, 1}
        else None
    )
    return {
        "arc_accuracy": _accuracy(truth[:, 1], decoded[:, 1]),
        "catalog_accuracy": _accuracy(truth[:, 2], catalog_prediction),
        "catalog_balanced_accuracy": _balanced_accuracy(truth[:, 2], catalog_prediction),
        "catalog_known_truth_balanced_accuracy_unresolved_is_wrong": known_balanced,
        "direct_pair_joint_accuracy": _accuracy(true_pair, pair_prediction),
        "group_count": len(groups),
        "straight_accuracy": _accuracy(truth[:, 0], decoded[:, 0]),
    }


def _task_macro_metrics(
    groups: Sequence[DevelopmentGroup],
    pair_prediction: np.ndarray,
    catalog_prediction: np.ndarray,
) -> dict[str, float]:
    tasks = sorted({task for group in groups for task in group.task_ids})
    truth = np.asarray([group.labels for group in groups], dtype=np.int64)
    decoded = np.asarray([decode_pair_class(value) for value in pair_prediction], dtype=np.int64)
    values = {"straight": [], "arc": [], "pair": [], "catalog": []}
    for task in tasks:
        indices = np.asarray(
            [
                index
                for index, group in enumerate(groups)
                for occurrence_task in group.task_ids
                if occurrence_task == task
            ],
            dtype=np.int64,
        )
        values["straight"].append(_accuracy(truth[indices, 0], decoded[indices, 0]))
        values["arc"].append(_accuracy(truth[indices, 1], decoded[indices, 1]))
        values["pair"].append(
            _accuracy(
                10 * truth[indices, 0] + truth[indices, 1], pair_prediction[indices]
            )
        )
        values["catalog"].append(
            _accuracy(truth[indices, 2], catalog_prediction[indices])
        )
    return {
        "arc_accuracy": float(np.mean(values["arc"])),
        "catalog_accuracy": float(np.mean(values["catalog"])),
        "direct_pair_joint_accuracy": float(np.mean(values["pair"])),
        "straight_accuracy": float(np.mean(values["straight"])),
        "task_count": len(tasks),
    }


def _stratum_metrics(
    groups: Sequence[DevelopmentGroup],
    pair_prediction: np.ndarray,
    catalog_prediction: np.ndarray,
) -> dict[str, Any]:
    tags: list[set[str]] = []
    for group in groups:
        current: set[str] = set()
        for row in group.metric_strata:
            current.add(f"line_decoration:{row['line_decoration']}")
            if row.get("thin_task") is True:
                current.add("thin_task:true")
            if row.get("crossing_task") is True:
                current.add("crossing_task:true")
        tags.append(current)
    result = {}
    for tag in sorted(set().union(*tags)):
        indices = [index for index, current in enumerate(tags) if tag in current]
        result[tag] = _metric_summary(
            [groups[index] for index in indices],
            np.asarray(pair_prediction)[indices],
            np.asarray(catalog_prediction)[indices],
        )
    return result


def _recomputed_gate(metrics: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    pair_value = metrics.get("direct_pair_joint_accuracy")
    catalog_value = metrics.get(
        "catalog_known_truth_balanced_accuracy_unresolved_is_wrong"
    )
    if (
        not isinstance(pair_value, (int, float))
        or isinstance(pair_value, bool)
        or not math.isfinite(float(pair_value))
        or not isinstance(catalog_value, (int, float))
        or isinstance(catalog_value, bool)
        or not math.isfinite(float(catalog_value))
    ):
        raise SkeletonGraphDevelopmentError("gate metric is missing or nonfinite")
    pair_passed = float(pair_value) >= ENGINEERING_THRESHOLDS[
        "direct_pair_joint_accuracy"
    ]
    catalog_passed = float(catalog_value) >= ENGINEERING_THRESHOLDS[
        "catalog_known_truth_balanced_accuracy"
    ]
    gate = {
        "catalog_three_class_passed": catalog_passed,
        "direct_pair_passed": pair_passed,
        "passed": pair_passed,
        "thresholds": _plain(ENGINEERING_THRESHOLDS),
    }
    promoted = [
        name for name, passed in (
            ("direct_pair", pair_passed), ("catalog_three_class", catalog_passed)
        ) if passed
    ]
    return gate, promoted


def _build_predictions(
    groups: Sequence[DevelopmentGroup],
    pair_probabilities: np.ndarray,
    catalog_probabilities: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for index, group in enumerate(groups):
        rows.append(
            {
                "catalog_probabilities": [float(value) for value in catalog_probabilities[index]],
                "group_index": group.index,
                "pair_probabilities": [float(value) for value in pair_probabilities[index]],
                "panel_id": group.representative_panel_id,
                "panel_ids": group.panel_ids,
                "png_sha256": group.png_sha256,
                "task_ids": group.task_ids,
            }
        )
    return _seal(
        {
            "catalog_class_order": CATALOG_CLASS_ORDER,
            "catalog_probability_digest": _array_digest(
                "validation_catalog_probabilities", catalog_probabilities
            ),
            "catalog_projection_policy": (
                "candidate_set_containing_unresolved_minus_one_yields_whole_axis_gap"
            ),
            "direct_pair_calibration_policy": (
                "calibrate_over_full_54_cell_valid_universe;_raw_missing_class_probability_is_zero"
            ),
            "observed_pair_class_order": OBSERVED_TRAIN_PAIR_CLASS_ORDER,
            "pair_probability_digest": _array_digest(
                "validation_direct_pair_probabilities", pair_probabilities
            ),
            "rows": rows,
            "schema": SCHEMA_PREDICTIONS,
            "valid_pair_class_order": VALID_PAIR_CLASS_ORDER,
        }
    )


def _validate_intended_path(precommit: Mapping[str, Any], key: str) -> Path:
    outputs = precommit["intended_outputs"]
    path = Path(outputs[key])
    root = Path(precommit["output_root"])
    if path.parent != root:
        raise SkeletonGraphDevelopmentError("intended output escaped bound root")
    descriptor = _open_nonsymlink_directory(root)
    try:
        if _directory_identity(os.fstat(descriptor)) != precommit["output_root_identity"]:
            raise SkeletonGraphDevelopmentError("output root identity changed after precommit")
    finally:
        os.close(descriptor)
    if path.exists():
        raise SkeletonGraphDevelopmentError(f"refusing to overwrite intended {key}")
    return path


def train_development(
    *, precommit_path: Path, expected_precommit_record_digest: str
) -> dict[str, Any]:
    """Fit the two heads on unique train digests and seal dev predictions."""

    operation_started = time.monotonic()
    with _hard_wall_timeout(600.0):
        precommit = _load_development_precommit(
            precommit_path, expected_record_digest=expected_precommit_record_digest
        )
    maximum = float(precommit["maximum_seconds"])
    remaining = maximum - (time.monotonic() - operation_started)
    if remaining <= 20.0:
        raise SkeletonGraphDevelopmentError("preflight consumed the finalization reserve")
    deadline = WallDeadline(operation_started, maximum - 20.0)
    with _hard_wall_timeout(remaining):
        fit, groups, inventory = load_fit_inventory(Path(precommit["fit_precommit_path"]))
        if inventory != precommit["fit_inventory_audit"] or fit["record_digest"] != FIT_PRECOMMIT_RECORD_DIGEST:
            raise SkeletonGraphDevelopmentError("fit inventory differs after precommit")
        repository_root = Path(precommit["repository_root"])
        dataset_root = Path(precommit["dataset_root"])
        matrix, labels, manifest, scope = _materialize_feature_bank(
            repository_root=repository_root,
            dataset_root=dataset_root,
            groups=groups,
            deadline=deadline,
        )
        cohort = np.asarray([group.cohort for group in groups])
        train_mask, validation_mask = cohort == "train", cohort == "validation"
        estimators = fit_authoritative_estimators(matrix[train_mask], labels[train_mask])
        deadline.check()
        pair_probability, catalog_probability = predict_authoritative_probabilities(
            estimators, matrix[validation_mask]
        )
        validation_groups = [group for group in groups if group.cohort == "validation"]
        pair_classes = np.asarray(OBSERVED_TRAIN_PAIR_CLASS_ORDER, dtype=np.int64)
        catalog_classes = np.asarray(CATALOG_CLASS_ORDER, dtype=np.int64)
        pair_prediction = pair_classes[np.argmax(pair_probability, axis=1)]
        catalog_prediction = catalog_classes[np.argmax(catalog_probability, axis=1)]
        metrics = _metric_summary(validation_groups, pair_prediction, catalog_prediction)
        task_macro = _task_macro_metrics(
            validation_groups, pair_prediction, catalog_prediction
        )
        strata = _stratum_metrics(validation_groups, pair_prediction, catalog_prediction)
        development_gate, promoted_heads = _recomputed_gate(metrics)
        feature_artifact = _seal(
            {
                "feature_array_digest": _array_digest("all_effective_features", matrix),
                "feature_dtype": matrix.dtype.str,
                "feature_names": FEATURE_NAMES,
                "feature_shape": list(matrix.shape),
                "label_array_digest": _array_digest("all_effective_labels", labels),
                "rows": manifest,
                "schema": SCHEMA_FEATURES,
                "scope_audit": scope,
            }
        )
        prediction_artifact = _build_predictions(
            validation_groups, pair_probability, catalog_probability
        )
        model_bundle = _build_model_bundle(
            estimators, precommit_record_digest=expected_precommit_record_digest
        )
        model_bytes = pickle.dumps(model_bundle, protocol=5)
        feature_bytes = canonical_json(feature_artifact) + b"\n"
        prediction_bytes = canonical_json(prediction_artifact) + b"\n"
        if (
            len(model_bytes) > MODEL_MAX_BYTES
            or len(feature_bytes) > FEATURE_ARTIFACT_MAX_BYTES
            or len(prediction_bytes) > PREDICTION_ARTIFACT_MAX_BYTES
        ):
            raise SkeletonGraphDevelopmentError("model, feature, or prediction artifact exceeds replay cap")
        model_structure = _model_structure(estimators, len(model_bytes))
        result = _seal(
            {
                "adaptive_evidence": True,
                "benchmark_promotion": False,
                "capacity_selection_ledger": CAPACITY_SELECTION_LEDGER,
                "carrier_signature_cv_diagnostic": CARRIER_SIGNATURE_CV_DIAGNOSTIC,
                "claim_scope": CLAIM_SCOPE,
                "config_digest": config_digest(),
                "development_gate": development_gate,
                "feature_artifact_file_sha256": _address(feature_bytes),
                "feature_artifact_record_digest": feature_artifact["record_digest"],
                "fit_precommit_record_digest": FIT_PRECOMMIT_RECORD_DIGEST,
                "model_file_sha256": _address(model_bytes),
                "model_structure": model_structure,
                "novel_carrier_policy": "gap_until_separate_scope_grant",
                "population_scope_self_detectable_from_pixels": False,
                "external_population_grant_required": True,
                "prediction_artifact_file_sha256": _address(prediction_bytes),
                "prediction_artifact_record_digest": prediction_artifact["record_digest"],
                "promoted_heads": promoted_heads,
                "precommit_record_digest": expected_precommit_record_digest,
                "prior_failed_capacity_attempt": precommit[
                    "prior_failed_capacity_attempt"
                ],
                "promotion_requires": PROMOTION_REQUIRES,
                "runtime": runtime_fingerprint(),
                "schema": SCHEMA_RESULT,
                "scope_audit": scope,
                "source_sha256": source_sha256(),
                "task_macro_validation_metrics": task_macro,
                "unique_digest_validation_metrics": metrics,
                "validation_stratum_metrics": strata,
            }
        )
        result_bytes = canonical_json(result) + b"\n"
        if len(result_bytes) > RESULT_MAX_BYTES:
            raise SkeletonGraphDevelopmentError("result artifact exceeds replay cap")
        deadline.check()
        identity = precommit["output_root_identity"]
        _write_once(
            _validate_intended_path(precommit, "model"), model_bytes,
            expected_parent_identity=identity,
        )
        _write_once(
            _validate_intended_path(precommit, "features"), feature_bytes,
            expected_parent_identity=identity,
        )
        _write_once(
            _validate_intended_path(precommit, "predictions"), prediction_bytes,
            expected_parent_identity=identity,
        )
        _write_once(
            _validate_intended_path(precommit, "result"), result_bytes,
            expected_parent_identity=identity,
        )
        return result


def _validate_model_bundle(
    bundle: Any, *, expected_precommit_record_digest: str
) -> dict[str, Any]:
    if (
        type(bundle) is not dict
        or bundle.get("schema") != MODEL_SCHEMA
        or bundle.get("source_sha256") != source_sha256()
        or bundle.get("config_digest") != config_digest()
        or bundle.get("runtime") != runtime_fingerprint()
        or bundle.get("precommit_record_digest") != expected_precommit_record_digest
        or tuple(bundle.get("feature_names", ())) != FEATURE_NAMES
        or tuple(bundle.get("valid_pair_class_order", ())) != VALID_PAIR_CLASS_ORDER
        or tuple(bundle.get("observed_pair_class_order", ()))
        != OBSERVED_TRAIN_PAIR_CLASS_ORDER
        or tuple(bundle.get("catalog_class_order", ())) != CATALOG_CLASS_ORDER
    ):
        raise SkeletonGraphDevelopmentError("model bundle policy differs")
    _validate_fitted_estimators(bundle.get("estimators", {}))
    return bundle


def _load_result_and_model(
    *,
    precommit: Mapping[str, Any],
    expected_result_record_digest: str,
    require_passed_development_gate: bool,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    result_path = Path(precommit["intended_outputs"]["result"])
    result = _load_canonical_record(result_path, schema=SCHEMA_RESULT, maximum=RESULT_MAX_BYTES)
    if (
        result.get("record_digest") != expected_result_record_digest
        or not _ADDRESS.fullmatch(str(expected_result_record_digest))
        or result.get("precommit_record_digest") != precommit["record_digest"]
        or result.get("source_sha256") != source_sha256()
        or result.get("config_digest") != config_digest()
        or result.get("capacity_selection_ledger")
        != _plain(CAPACITY_SELECTION_LEDGER)
        or result.get("prior_failed_capacity_attempt")
        != precommit.get("prior_failed_capacity_attempt")
        or result.get("runtime") != runtime_fingerprint()
        or result.get("benchmark_promotion") is not False
        or result.get("claim_scope") != CLAIM_SCOPE
        or result.get("promotion_requires") != PROMOTION_REQUIRES
    ):
        raise SkeletonGraphDevelopmentError("development result policy differs")
    recomputed_gate, recomputed_heads = _recomputed_gate(
        result.get("unique_digest_validation_metrics", {})
    )
    if (
        result.get("development_gate") != recomputed_gate
        or result.get("promoted_heads") != recomputed_heads
    ):
        raise SkeletonGraphDevelopmentError("development gate or promoted heads differ")
    if (
        require_passed_development_gate
        and result.get("development_gate", {}).get("passed") is not True
    ):
        raise SkeletonGraphDevelopmentError("development gate did not pass")
    model_path = Path(precommit["intended_outputs"]["model"])
    model_bytes = _stable_regular_bytes(model_path, maximum=MODEL_MAX_BYTES)
    if _address(model_bytes) != result.get("model_file_sha256"):
        raise SkeletonGraphDevelopmentError("model file address differs")
    try:
        bundle = pickle.loads(model_bytes)
    except Exception as exc:
        raise SkeletonGraphDevelopmentError(f"cannot load bound model: {exc}") from exc
    bundle = _validate_model_bundle(
        bundle, expected_precommit_record_digest=precommit["record_digest"]
    )
    if result.get("model_structure") != _model_structure(
        bundle["estimators"], len(model_bytes)
    ):
        raise SkeletonGraphDevelopmentError("model structure result binding differs")
    return result, bundle, model_bytes


@dataclass(frozen=True, init=False)
class VerifiedDevelopmentModel:
    """Immutable serialized development model with head-scoped prediction."""

    _model_bytes: bytes
    model_file_sha256: str
    precommit_record_digest: str
    promoted_heads: tuple[str, ...]
    result_record_digest: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise SkeletonGraphDevelopmentError(
            "VerifiedDevelopmentModel is factory-only"
        )

    def predict(self, *, head: str, features: np.ndarray) -> np.ndarray:
        if head not in self.promoted_heads:
            raise SkeletonGraphDevelopmentError("requested head was not promoted")
        if _address(self._model_bytes) != self.model_file_sha256:
            raise SkeletonGraphDevelopmentError("verified model bytes changed")
        try:
            raw_bundle = pickle.loads(self._model_bytes)
        except Exception as exc:
            raise SkeletonGraphDevelopmentError(
                f"cannot reload authenticated model: {exc}"
            ) from exc
        bundle = _validate_model_bundle(
            raw_bundle,
            expected_precommit_record_digest=self.precommit_record_digest,
        )
        pair, catalog = predict_authoritative_probabilities(
            bundle["estimators"], features
        )
        return pair if head == "direct_pair" else catalog


def _make_verified_development_model(
    *,
    model_bytes: bytes,
    model_file_sha256: str,
    precommit_record_digest: str,
    promoted_heads: tuple[str, ...],
    result_record_digest: str,
) -> VerifiedDevelopmentModel:
    value = object.__new__(VerifiedDevelopmentModel)
    object.__setattr__(value, "_model_bytes", model_bytes)
    object.__setattr__(value, "model_file_sha256", model_file_sha256)
    object.__setattr__(value, "precommit_record_digest", precommit_record_digest)
    object.__setattr__(value, "promoted_heads", promoted_heads)
    object.__setattr__(value, "result_record_digest", result_record_digest)
    return value


def load_verified_development_model(
    *,
    precommit_path: Path,
    expected_precommit_record_digest: str,
    expected_result_record_digest: str,
    required_heads: Sequence[str] = ("direct_pair",),
) -> VerifiedDevelopmentModel:
    """Load the dev model; this is not a benchmark-promotion API."""

    precommit = _load_development_precommit(
        precommit_path, expected_record_digest=expected_precommit_record_digest
    )
    result, _bundle, model_bytes = _load_result_and_model(
        precommit=precommit,
        expected_result_record_digest=expected_result_record_digest,
        require_passed_development_gate=True,
    )
    requested = tuple(required_heads)
    if (
        not requested
        or any(type(name) is not str for name in requested)
        or not set(requested) <= set(result["promoted_heads"])
    ):
        raise SkeletonGraphDevelopmentError("required head was not promoted")
    return _make_verified_development_model(
        model_bytes=model_bytes,
        model_file_sha256=result["model_file_sha256"],
        precommit_record_digest=precommit["record_digest"],
        promoted_heads=requested,
        result_record_digest=result["record_digest"],
    )


def replay_development(
    *,
    precommit_path: Path,
    expected_precommit_record_digest: str,
    expected_result_record_digest: str,
) -> dict[str, Any]:
    """Cold exact pixel/model replay without refitting the forest."""

    operation_started = time.monotonic()
    with _hard_wall_timeout(600.0):
        precommit = _load_development_precommit(
            precommit_path, expected_record_digest=expected_precommit_record_digest
        )
        result, bundle, _model_bytes = _load_result_and_model(
            precommit=precommit,
            expected_result_record_digest=expected_result_record_digest,
            require_passed_development_gate=False,
        )
    maximum = float(precommit["maximum_seconds"])
    remaining = maximum - (time.monotonic() - operation_started)
    if remaining <= 20.0:
        raise SkeletonGraphDevelopmentError("replay preflight consumed finalization reserve")
    deadline = WallDeadline(operation_started, maximum - 20.0)
    with _hard_wall_timeout(remaining):
        _fit, groups, inventory = load_fit_inventory(Path(precommit["fit_precommit_path"]))
        if inventory != precommit["fit_inventory_audit"]:
            raise SkeletonGraphDevelopmentError("replay fit inventory differs")
        matrix, labels, manifest, scope = _materialize_feature_bank(
            repository_root=Path(precommit["repository_root"]),
            dataset_root=Path(precommit["dataset_root"]),
            groups=groups,
            deadline=deadline,
        )
        cohort = np.asarray([group.cohort for group in groups])
        validation_mask = cohort == "validation"
        validation_groups = [group for group in groups if group.cohort == "validation"]
        pair_probability, catalog_probability = predict_authoritative_probabilities(
            bundle["estimators"], matrix[validation_mask]
        )
        replayed_predictions = _build_predictions(
            validation_groups, pair_probability, catalog_probability
        )
        stored_predictions_path = Path(precommit["intended_outputs"]["predictions"])
        stored_predictions_raw = _stable_regular_bytes(
            stored_predictions_path, maximum=PREDICTION_ARTIFACT_MAX_BYTES
        )
        stored_predictions = _load_canonical_record(
            stored_predictions_path,
            schema=SCHEMA_PREDICTIONS,
            maximum=PREDICTION_ARTIFACT_MAX_BYTES,
        )
        if (
            _address(stored_predictions_raw) != result["prediction_artifact_file_sha256"]
            or stored_predictions.get("record_digest")
            != result["prediction_artifact_record_digest"]
            or stored_predictions != replayed_predictions
        ):
            raise SkeletonGraphDevelopmentError("cold probability replay differs")
        replayed_features = _seal(
            {
                "feature_array_digest": _array_digest("all_effective_features", matrix),
                "feature_dtype": matrix.dtype.str,
                "feature_names": FEATURE_NAMES,
                "feature_shape": list(matrix.shape),
                "label_array_digest": _array_digest("all_effective_labels", labels),
                "rows": manifest,
                "schema": SCHEMA_FEATURES,
                "scope_audit": scope,
            }
        )
        stored_features_path = Path(precommit["intended_outputs"]["features"])
        stored_features_raw = _stable_regular_bytes(
            stored_features_path, maximum=FEATURE_ARTIFACT_MAX_BYTES
        )
        stored_features = _load_canonical_record(
            stored_features_path,
            schema=SCHEMA_FEATURES,
            maximum=FEATURE_ARTIFACT_MAX_BYTES,
        )
        if (
            _address(stored_features_raw) != result["feature_artifact_file_sha256"]
            or stored_features.get("record_digest")
            != result["feature_artifact_record_digest"]
            or stored_features != replayed_features
        ):
            raise SkeletonGraphDevelopmentError("cold feature replay differs")
        pair_classes = np.asarray(OBSERVED_TRAIN_PAIR_CLASS_ORDER, dtype=np.int64)
        catalog_classes = np.asarray(CATALOG_CLASS_ORDER, dtype=np.int64)
        pair_prediction = pair_classes[np.argmax(pair_probability, axis=1)]
        catalog_prediction = catalog_classes[np.argmax(catalog_probability, axis=1)]
        metrics = _metric_summary(validation_groups, pair_prediction, catalog_prediction)
        if (
            metrics != result.get("unique_digest_validation_metrics")
            or _task_macro_metrics(validation_groups, pair_prediction, catalog_prediction)
            != result.get("task_macro_validation_metrics")
            or _stratum_metrics(validation_groups, pair_prediction, catalog_prediction)
            != result.get("validation_stratum_metrics")
            or scope != result.get("scope_audit")
        ):
            raise SkeletonGraphDevelopmentError("cold metric replay differs")
        replay = _seal(
            {
                "feature_replay_exact": True,
                "metrics_replay_exact": True,
                "model_file_sha256": result["model_file_sha256"],
                "model_inference_panel_count": len(validation_groups),
                "model_refit_calls": 0,
                "pixel_reextract_group_count": len(groups),
                "precommit_record_digest": precommit["record_digest"],
                "prediction_replay_exact": True,
                "probability_digests": {
                    "catalog": replayed_predictions["catalog_probability_digest"],
                    "direct_pair": replayed_predictions["pair_probability_digest"],
                },
                "result_record_digest": result["record_digest"],
                "schema": SCHEMA_REPLAY,
                "source_sha256": source_sha256(),
            }
        )
        deadline.check()
        replay_path = _validate_intended_path(precommit, "replay")
        _write_record_once(
            replay_path,
            replay,
            expected_parent_identity=precommit["output_root_identity"],
        )
        return replay


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--repository-root", type=Path, required=True)
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--fit-precommit", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--maximum-seconds", type=float, default=600.0)
    train = commands.add_parser("train")
    train.add_argument("--precommit", type=Path, required=True)
    train.add_argument("--expected-precommit-record-digest", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--precommit", type=Path, required=True)
    replay.add_argument("--expected-precommit-record-digest", required=True)
    replay.add_argument("--expected-result-record-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        output = arguments.output_dir
        value = create_development_precommit(
            repository_root=arguments.repository_root,
            dataset_root=arguments.dataset_root,
            fit_precommit_path=arguments.fit_precommit,
            model_path=output / "model.pkl",
            feature_manifest_path=output / "features.json",
            predictions_path=output / "predictions.json",
            result_path=output / "result.json",
            replay_path=output / "replay.json",
            output_path=output / "precommit.json",
            maximum_seconds=arguments.maximum_seconds,
        )
        print(canonical_json(value).decode("utf-8"))
        return 0
    if arguments.command == "train":
        value = train_development(
            precommit_path=arguments.precommit,
            expected_precommit_record_digest=arguments.expected_precommit_record_digest,
        )
        print(canonical_json(value).decode("utf-8"))
        return 0 if value["development_gate"]["passed"] else 2
    value = replay_development(
        precommit_path=arguments.precommit,
        expected_precommit_record_digest=arguments.expected_precommit_record_digest,
        expected_result_record_digest=arguments.expected_result_record_digest,
    )
    print(canonical_json(value).decode("utf-8"))
    return 0


def decode_pair_class(encoded: int) -> tuple[int, int]:
    """Decode one supported direct-pair class without inventing a joint cell."""

    if isinstance(encoded, (bool, np.bool_)) or not isinstance(encoded, (int, np.integer)):
        raise SkeletonGraphDevelopmentError("pair class is not an integer")
    encoded = int(encoded)
    if encoded not in VALID_PAIR_CLASS_ORDER:
        raise SkeletonGraphDevelopmentError("pair class is outside the valid universe")
    return divmod(encoded, 10)


if __name__ == "__main__":  # pragma: no cover - exercised by live preflight
    raise SystemExit(main())
