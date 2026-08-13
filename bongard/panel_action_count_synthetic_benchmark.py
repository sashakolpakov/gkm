"""Paired synthetic benchmark for visible line/arc count inversion.

The benchmark constructs its inputs locally, splits whole carrier families,
and presents the exact same PNG bytes to two observers:

* a synthetic-only reimplementation of the historical 112 pooled skeleton
  features with a freshly fitted fixed-32 ExtraTrees control; and
* the ordered path-graph inversion observer.

The target is a partial pure-raster component normal form, not hidden
generator history and not a claim of universally minimal raster explanation.
Balanced metric rows are deliberately restricted to disconnected components
with exact scalar targets. Candidate pairs are independently fitted, while a
post-fit shared target-resolvability gate suppresses singleton claims for
connected unresolved stress rows. Their set/GAP safety is therefore a policy
guarantee, not independent observer evidence. The result is synthetic
mechanistic evidence only: it grants no access to, and makes no claim about,
official Bongard data or any calibration/query cohort.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from collections import Counter, defaultdict
import hashlib
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np

from bongard.canonical import canonical_digest
from bongard import panel_action_count_ordered_path_inversion as ordered
from bongard import panel_action_count_synthetic_pooled_control as pooled
from bongard import panel_action_count_synthetic_identifiability as synthetic


SCHEMA: Final = "gkm.bongard-paired-synthetic-visible-count-benchmark.v3"
CONTROL_ID: Final = "four-scale-112-pooled-features-fixed-32-extra-trees-synthetic-refit/v1"
ORDERED_ID: Final = "ordered-path-graph-visible-line-arc-inversion/v2"
TARGET_ID: Final = "partial-componentwise-visible-raster-normal-form/v4"
FAMILY_RESAMPLING_SEED: Final = 260_812
FAMILY_RESAMPLING_REPLICATES: Final = 2_048
HISTORICAL_PANEL_JOINT_ANCHOR: Final = 0.352149331418828
HISTORICAL_CARRIER_MACRO_ANCHOR: Final = 0.3054062978972449

CONTROL_PARAMETERS: Final = MappingProxyType(
    {
        "bootstrap": False,
        "class_weight": "balanced",
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "n_estimators": 32,
        "n_jobs": 1,
        "random_state": 260_813,
    }
)

GATE_THRESHOLDS: Final = MappingProxyType(
    {
        "historically_unseen_pair_macro_accuracy_at_least": 0.35,
        "ordered_joint_accuracy_at_least": 0.50,
        "paired_joint_accuracy_delta_at_least": 0.10,
    }
)

_HISTORICALLY_OBSERVED_PAIR_CODES: Final = (
    1, 2, 4, 6, 8, 11, 12, 20, 21, 22, 23, 30, 31, 32, 33, 34,
    40, 41, 42, 43, 44, 50, 51, 52, 60, 61, 62, 63, 70, 71, 80, 81, 90,
)
_HISTORICALLY_OBSERVED_PAIRS: Final = frozenset(
    divmod(encoded, 10) for encoded in _HISTORICALLY_OBSERVED_PAIR_CODES
)
HISTORICALLY_UNSEEN_PAIRS: Final = tuple(
    pair.as_tuple()
    for pair in synthetic.valid_count_pairs()
    if pair.as_tuple() not in _HISTORICALLY_OBSERVED_PAIRS
)


class SyntheticBenchmarkError(RuntimeError):
    """The synthetic corpus, paired evaluation, or metric invariant differs."""


def source_sha256() -> str:
    """Verify and return the import-time source address."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _address(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _pair_tuple(pair: object) -> tuple[int, int]:
    if type(pair) is synthetic.CountPair:
        return pair.as_tuple()
    if (
        type(pair) is tuple
        and len(pair) == 2
        and all(type(value) is int for value in pair)
        and pair in {item.as_tuple() for item in synthetic.valid_count_pairs()}
    ):
        return pair
    raise SyntheticBenchmarkError("count pair leaves the exact valid universe")


def _pair_data(pair: object) -> list[int]:
    value = _pair_tuple(pair)
    return [value[0], value[1]]


def _optional_pair_data(pair: object) -> list[int] | None:
    if pair is None:
        return None
    return _pair_data(pair)


def _required_visible_pair(
    panel: synthetic.RenderedPanel,
) -> synthetic.CountPair:
    pair = panel.canonical_visible_pair
    if type(pair) is not synthetic.CountPair:
        raise SyntheticBenchmarkError(
            "balanced benchmark row has no exact raster target"
        )
    return pair


def _sample_key(sample: synthetic.SyntheticSample) -> tuple[str, str]:
    return sample.sample_id, sample.panel.png_sha256


def _sample_manifest(
    samples: Sequence[synthetic.SyntheticSample], *, role: str
) -> list[dict[str, object]]:
    return [
        {
            "canonical_visible_pair": _pair_data(
                _required_visible_pair(sample.panel)
            ),
            "carrier_family": sample.panel.carrier_family,
            "carrier_id": sample.carrier_id,
            "declared_pair": _pair_data(sample.panel.declared_pair),
            "nuisance": sample.panel.nuisance.identity,
            "png_sha256": sample.panel.png_sha256,
            "role": role,
            "sample_id": sample.sample_id,
        }
        for sample in samples
    ]


def _validate_split(split: synthetic.CorpusSplit) -> tuple[
    tuple[synthetic.SyntheticSample, ...], tuple[synthetic.SyntheticSample, ...]
]:
    if type(split) is not synthetic.CorpusSplit:
        raise SyntheticBenchmarkError("benchmark requires an exact synthetic split")
    synthetic.CorpusSplit.__post_init__(split)
    training = tuple(sorted(split.training, key=_sample_key))
    evaluation = tuple(sorted(split.evaluation, key=_sample_key))
    all_rows = training + evaluation
    sample_ids = [sample.sample_id for sample in all_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise SyntheticBenchmarkError("sample identifiers are not globally unique")
    train_families = {sample.panel.carrier_family for sample in training}
    evaluation_families = {sample.panel.carrier_family for sample in evaluation}
    train_carriers = {sample.carrier_id for sample in training}
    evaluation_carriers = {sample.carrier_id for sample in evaluation}
    train_pngs = {sample.panel.png_sha256 for sample in training}
    evaluation_pngs = {sample.panel.png_sha256 for sample in evaluation}
    if train_families & evaluation_families:
        raise SyntheticBenchmarkError("carrier family crosses paired split roles")
    if train_carriers & evaluation_carriers:
        raise SyntheticBenchmarkError("carrier identity crosses paired split roles")
    if train_pngs & evaluation_pngs:
        raise SyntheticBenchmarkError("exact PNG bytes cross paired split roles")
    valid_pairs = set(synthetic.valid_count_pairs())
    for role, rows in (("training", training), ("evaluation", evaluation)):
        found = {_required_visible_pair(sample.panel) for sample in rows}
        if found != valid_pairs:
            raise SyntheticBenchmarkError(
                f"{role} does not contain the complete 54-pair visible target"
            )
        for family in sorted({sample.panel.carrier_family for sample in rows}):
            family_rows = tuple(
                sample for sample in rows if sample.panel.carrier_family == family
            )
            family_targets = tuple(
                _required_visible_pair(sample.panel) for sample in family_rows
            )
            nuisance_count = len(
                {sample.panel.nuisance.identity for sample in family_rows}
            )
            if (
                set(family_targets) != valid_pairs
                or len(family_targets) != len(valid_pairs) * nuisance_count
                or any(
                    family_targets.count(pair) != nuisance_count
                    for pair in valid_pairs
                )
            ):
                raise SyntheticBenchmarkError(
                    f"{role} family {family} is not balanced over all 54 targets"
                )
    return training, evaluation


def _deduplicated_training(
    samples: Sequence[synthetic.SyntheticSample],
) -> tuple[synthetic.SyntheticSample, ...]:
    by_png: dict[str, list[synthetic.SyntheticSample]] = defaultdict(list)
    for sample in samples:
        by_png[sample.panel.png_sha256].append(sample)
    result: list[synthetic.SyntheticSample] = []
    for digest, rows in sorted(by_png.items()):
        targets = {_required_visible_pair(row.panel) for row in rows}
        if len(targets) != 1:
            raise SyntheticBenchmarkError(
                f"training PNG {digest} has conflicting visible targets"
            )
        result.append(min(rows, key=_sample_key))
    return tuple(result)


def _feature_matrix(
    samples: Sequence[synthetic.SyntheticSample],
) -> np.ndarray:
    if not samples:
        raise SyntheticBenchmarkError("pooled feature population is empty")
    return np.ascontiguousarray(
        np.stack(
            [pooled.extract_feature_vector(row.panel.png_bytes) for row in samples]
        ),
        dtype=np.float32,
    )


def _fit_pooled_control(
    training: Sequence[synthetic.SyntheticSample],
) -> tuple[Any, tuple[synthetic.SyntheticSample, ...], np.ndarray]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
    except ImportError as exc:  # pragma: no cover - environment failure
        raise SyntheticBenchmarkError("scikit-learn is unavailable") from exc
    unique = _deduplicated_training(training)
    features = _feature_matrix(unique)
    target = np.asarray(
        [
            10 * _required_visible_pair(row.panel).straight
            + _required_visible_pair(row.panel).arc
         for row in unique],
        dtype=np.int64,
    )
    expected_classes = tuple(
        10 * pair.straight + pair.arc for pair in synthetic.valid_count_pairs()
    )
    if tuple(sorted(int(value) for value in np.unique(target))) != expected_classes:
        raise SyntheticBenchmarkError("deduplicated control training lost a valid pair")
    estimator = ExtraTreesClassifier(**dict(CONTROL_PARAMETERS)).fit(features, target)
    if tuple(int(value) for value in estimator.classes_) != expected_classes:
        raise SyntheticBenchmarkError("pooled control class order differs")
    return estimator, unique, features


def _pooled_predictions(
    estimator: Any, evaluation: Sequence[synthetic.SyntheticSample]
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    features = _feature_matrix(evaluation)
    encoded = estimator.predict(features)
    predictions = tuple(divmod(int(value), 10) for value in encoded)
    valid_pairs = {item.as_tuple() for item in synthetic.valid_count_pairs()}
    if any(pair not in valid_pairs for pair in predictions):
        raise SyntheticBenchmarkError("pooled control emitted an invalid pair")
    return predictions, features


def _safe_ordered_prediction(png_bytes: bytes) -> dict[str, object]:
    try:
        outcome = ordered.invert_png(png_bytes)
    except ordered.OrderedPathInversionError as exc:
        return {"candidate_pairs": [], "disposition": "ERROR", "reason": str(exc)}
    candidates = tuple(outcome.candidate_pairs)
    if any(
        pair not in {item.as_tuple() for item in synthetic.valid_count_pairs()}
        for pair in candidates
    ):
        raise SyntheticBenchmarkError("ordered observer emitted an invalid pair")
    return {
        "candidate_pairs": [_pair_data(pair) for pair in candidates],
        "disposition": outcome.disposition,
        "reason": outcome.reason,
    }


def _macro_accuracy(
    truth: Sequence[tuple[int, int]],
    correct: Sequence[bool],
    *,
    included: frozenset[tuple[int, int]] | None = None,
) -> float:
    values: list[float] = []
    for pair_record in synthetic.valid_count_pairs():
        pair = pair_record.as_tuple()
        if included is not None and pair not in included:
            continue
        indices = [index for index, value in enumerate(truth) if value == pair]
        if not indices:
            raise SyntheticBenchmarkError("macro metric target stratum is absent")
        values.append(sum(bool(correct[index]) for index in indices) / len(indices))
    if not values:
        raise SyntheticBenchmarkError("macro metric has no target strata")
    return float(sum(values) / len(values))


def _carrier_macro_accuracy(
    samples: Sequence[synthetic.SyntheticSample], correct: Sequence[bool]
) -> float:
    families = sorted({sample.panel.carrier_family for sample in samples})
    if not families:
        raise SyntheticBenchmarkError("carrier macro metric has no families")
    values = []
    for family in families:
        indices = [
            index
            for index, sample in enumerate(samples)
            if sample.panel.carrier_family == family
        ]
        values.append(sum(bool(correct[index]) for index in indices) / len(indices))
    return float(sum(values) / len(values))


def _carrier_family_metrics(
    samples: Sequence[synthetic.SyntheticSample],
    candidate_sets: Sequence[tuple[tuple[int, int], ...]],
) -> dict[str, dict[str, float | int]]:
    if len(samples) != len(candidate_sets):
        raise SyntheticBenchmarkError("carrier-family metric population differs")
    result: dict[str, dict[str, float | int]] = {}
    for family in sorted({sample.panel.carrier_family for sample in samples}):
        indices = tuple(
            index
            for index, sample in enumerate(samples)
            if sample.panel.carrier_family == family
        )
        truth = tuple(
            _pair_tuple(samples[index].panel.canonical_visible_pair)
            for index in indices
        )
        candidates = tuple(candidate_sets[index] for index in indices)
        result[family] = {
            "candidate_set_contains_truth_accuracy": sum(
                wanted in values
                for wanted, values in zip(truth, candidates, strict=True)
            )
            / len(indices),
            "denominator": len(indices),
            "joint_singleton_accuracy": sum(
                values == (wanted,)
                for wanted, values in zip(truth, candidates, strict=True)
            )
            / len(indices),
            "nonempty_candidate_set_rate": sum(bool(values) for values in candidates)
            / len(indices),
        }
    return result


def _metrics(
    samples: Sequence[synthetic.SyntheticSample],
    candidate_sets: Sequence[tuple[tuple[int, int], ...]],
) -> dict[str, object]:
    if len(samples) != len(candidate_sets) or not samples:
        raise SyntheticBenchmarkError("metric prediction population differs")
    truth = tuple(
        _pair_tuple(sample.panel.canonical_visible_pair) for sample in samples
    )
    singleton = tuple(candidates[0] if len(candidates) == 1 else None for candidates in candidate_sets)
    joint = tuple(prediction == wanted for prediction, wanted in zip(singleton, truth, strict=True))
    straight = tuple(
        prediction is not None and prediction[0] == wanted[0]
        for prediction, wanted in zip(singleton, truth, strict=True)
    )
    arc = tuple(
        prediction is not None and prediction[1] == wanted[1]
        for prediction, wanted in zip(singleton, truth, strict=True)
    )
    contained = tuple(
        wanted in candidates
        for candidates, wanted in zip(candidate_sets, truth, strict=True)
    )
    straight_contained = tuple(
        wanted[0] in {pair[0] for pair in candidates}
        for candidates, wanted in zip(candidate_sets, truth, strict=True)
    )
    arc_contained = tuple(
        wanted[1] in {pair[1] for pair in candidates}
        for candidates, wanted in zip(candidate_sets, truth, strict=True)
    )
    denominator = len(samples)
    unseen = frozenset(HISTORICALLY_UNSEEN_PAIRS)
    return {
        "arc_candidate_set_accuracy": sum(arc_contained) / denominator,
        "arc_singleton_accuracy": sum(arc) / denominator,
        "candidate_set_contains_truth_accuracy": sum(contained) / denominator,
        "carrier_family_macro_joint_singleton_accuracy": _carrier_macro_accuracy(samples, joint),
        "denominator": denominator,
        "historically_unseen_pair_macro_candidate_set_accuracy": _macro_accuracy(
            truth, contained, included=unseen
        ),
        "historically_unseen_pair_macro_singleton_accuracy": _macro_accuracy(
            truth, joint, included=unseen
        ),
        "joint_pair_macro_candidate_set_accuracy": _macro_accuracy(truth, contained),
        "joint_pair_macro_singleton_accuracy": _macro_accuracy(truth, joint),
        "joint_singleton_accuracy": sum(joint) / denominator,
        "nonempty_candidate_set_rate": sum(bool(value) for value in candidate_sets) / denominator,
        "straight_candidate_set_accuracy": sum(straight_contained) / denominator,
        "straight_singleton_accuracy": sum(straight) / denominator,
    }


def _disposition_counts(outputs: Sequence[Mapping[str, object]]) -> dict[str, int]:
    allowed = ("AMBIGUOUS", "ERROR", "GAP", "IDENTIFIED")
    result = {
        disposition: sum(output.get("disposition") == disposition for output in outputs)
        for disposition in allowed
    }
    if sum(result.values()) != len(outputs):
        raise SyntheticBenchmarkError("ordered disposition leaves fixed vocabulary")
    return result


def _descriptive_family_resampling(
    samples: Sequence[synthetic.SyntheticSample],
    ordered_correct: Sequence[bool],
    control_correct: Sequence[bool],
) -> dict[str, float | int | str]:
    if not (
        len(samples) == len(ordered_correct) == len(control_correct) and samples
    ):
        raise SyntheticBenchmarkError("family-resampling population differs")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[sample.panel.carrier_family].append(index)
    cluster_ids = tuple(sorted(groups))
    if len(cluster_ids) < 2:
        raise SyntheticBenchmarkError("family resampling needs at least two carriers")
    rng = np.random.default_rng(FAMILY_RESAMPLING_SEED)
    ordered_values = np.asarray(ordered_correct, dtype=np.float64)
    control_values = np.asarray(control_correct, dtype=np.float64)
    ordered_draws = np.empty(FAMILY_RESAMPLING_REPLICATES, dtype=np.float64)
    delta_draws = np.empty(FAMILY_RESAMPLING_REPLICATES, dtype=np.float64)
    for replicate in range(FAMILY_RESAMPLING_REPLICATES):
        selected = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        indices = [
            index
            for selected_index in selected
            for index in groups[cluster_ids[int(selected_index)]]
        ]
        ordered_draws[replicate] = float(np.mean(ordered_values[indices]))
        delta_draws[replicate] = float(
            np.mean(ordered_values[indices] - control_values[indices])
        )
    return {
        "carrier_cluster_count": len(cluster_ids),
        "descriptive_resampling_quantile": 0.05,
        "inferential_confidence_bound_claimed": False,
        "ordered_joint_accuracy_family_resampling_p05": float(
            np.quantile(ordered_draws, 0.05, method="lower")
        ),
        "paired_delta_family_resampling_p05": float(
            np.quantile(delta_draws, 0.05, method="lower")
        ),
        "replicates": FAMILY_RESAMPLING_REPLICATES,
        "resampling_unit": "held_out_carrier_family",
        "seed": FAMILY_RESAMPLING_SEED,
    }


def _nearest_cross_target_pooled_audit(
    training_features: np.ndarray,
    evaluation: Sequence[synthetic.SyntheticSample],
    evaluation_features: np.ndarray,
    *,
    retained_pair_count: int = 12,
) -> dict[str, object]:
    """Expose different-count panels nearest under the pooled representation.

    This is a descriptive adversarial search over the finite synthetic
    evaluation rows. It is not an exhaustive perceptual search and its
    distances are not uncertainty estimates.
    """

    if (
        type(training_features) is not np.ndarray
        or type(evaluation_features) is not np.ndarray
        or training_features.ndim != 2
        or evaluation_features.ndim != 2
        or training_features.shape[1:] != evaluation_features.shape[1:]
        or len(evaluation) != len(evaluation_features)
        or type(retained_pair_count) is not int
        or retained_pair_count <= 0
    ):
        raise SyntheticBenchmarkError("pooled nearest-neighbour audit shape differs")
    scale = np.std(training_features.astype(np.float64), axis=0)
    scale = np.where(scale > 1e-7, scale, 1.0)
    normalized = evaluation_features.astype(np.float64) / scale
    rows: list[tuple[float, int, int]] = []
    for first in range(len(evaluation)):
        for second in range(first + 1, len(evaluation)):
            if (
                _required_visible_pair(evaluation[first].panel)
                == _required_visible_pair(evaluation[second].panel)
            ):
                continue
            distance = float(
                np.sqrt(np.mean(np.square(normalized[first] - normalized[second])))
            )
            rows.append((distance, first, second))
    if not rows:
        raise SyntheticBenchmarkError("pooled audit found no cross-target pairs")
    retained = sorted(rows)[:retained_pair_count]
    return {
        "audit_scope": "finite_evaluation_nearest_cross_target_only",
        "distance": "training_feature_sd_scaled_root_mean_square",
        "evaluation_pair_population": len(rows),
        "minimum_distance": retained[0][0],
        "rows": [
            {
                "distance": distance,
                "first_pair": _pair_data(
                    _required_visible_pair(evaluation[first].panel)
                ),
                "first_png_sha256": evaluation[first].panel.png_sha256,
                "first_sample_id": evaluation[first].sample_id,
                "second_pair": _pair_data(
                    _required_visible_pair(evaluation[second].panel)
                ),
                "second_png_sha256": evaluation[second].panel.png_sha256,
                "second_sample_id": evaluation[second].sample_id,
            }
            for distance, first, second in retained
        ],
    }


def _identifiability_counterfactual_audit() -> dict[str, object]:
    cases = synthetic.ambiguity_cases()
    audit_candidates: list[synthetic.AuditCandidate] = []
    rows: list[dict[str, object]] = []
    exact_safe = 0
    false_visible_singletons = 0
    prediction_inconsistencies = 0
    for case in cases:
        left = synthetic.render_program(case.left)
        right = synthetic.render_program(case.right)
        audit_candidates.extend(
            (
                synthetic.AuditCandidate(f"{case.case_id}:left", left),
                synthetic.AuditCandidate(f"{case.case_id}:right", right),
            )
        )
        outcomes = tuple(
            _safe_ordered_prediction(panel.png_bytes) for panel in (left, right)
        )
        identities = {
            canonical_digest(
                {
                    "candidate_pairs": outcome["candidate_pairs"],
                    "disposition": outcome["disposition"],
                    "reason": outcome["reason"],
                }
            )
            for outcome in outcomes
        }
        if case.expected_relation == "exact":
            if left.png_bytes != right.png_bytes:
                raise SyntheticBenchmarkError("declared exact case is not exact")
            prediction_inconsistencies += len(identities) != 1
        elif left.png_bytes == right.png_bytes:
            raise SyntheticBenchmarkError("declared near case is exact")
        outcome = outcomes[0]
        predicted_candidates = tuple(
            tuple(pair) for pair in outcome["candidate_pairs"]
        )
        left_target = left.canonical_visible_pair
        right_target = right.canonical_visible_pair
        if case.expected_relation == "exact" and left_target != right_target:
            raise SyntheticBenchmarkError(
                "identical PNGs have different pure-raster target states"
            )
        unresolved = left_target is None or right_target is None
        if unresolved:
            visible: set[tuple[int, int]] = set()
            safe = (
                left_target is None
                and right_target is None
                and outcome["disposition"] in ("AMBIGUOUS", "GAP")
                and (
                    outcome["disposition"] == "AMBIGUOUS"
                    or not predicted_candidates
                )
            )
            target_status = "unresolved"
        else:
            visible = {
                _pair_tuple(left_target),
                _pair_tuple(right_target),
            }
            safe = (
                outcome["disposition"] == "GAP"
                or visible.issubset(set(predicted_candidates))
            )
            target_status = "resolved"
        if case.expected_relation == "exact":
            exact_safe += safe
            false_visible_singletons += (
                outcome["disposition"] == "IDENTIFIED"
                and (
                    unresolved
                    or set(predicted_candidates) != visible
                )
            )
        rows.append(
            {
                "candidate_pairs": [list(pair) for pair in predicted_candidates],
                "case_id": case.case_id,
                "expected_relation": case.expected_relation,
                "visible_raster_target_set": [
                    _pair_data(pair) for pair in sorted(visible)
                ],
                "declared_pairs": [
                    _pair_data(pair)
                    for pair in sorted({left.declared_pair, right.declared_pair})
                ],
                "disposition": outcome["disposition"],
                "exact_case_safe": safe if case.expected_relation == "exact" else None,
                "left_visible_raster_target": _optional_pair_data(left_target),
                "png_sha256": left.png_sha256 if case.expected_relation == "exact" else None,
                "reason": outcome["reason"],
                "right_visible_raster_target": _optional_pair_data(right_target),
                "target_status": target_status,
            }
        )
    collision = synthetic.audit_collisions(
        tuple(audit_candidates), max_near_comparisons=64, near_xor_limit=64
    )
    exact_cases = sum(case.expected_relation == "exact" for case in cases)
    exact_members = sum(len(row.candidate_ids) for row in collision.exact_collisions)
    exact_oracle_correct = sum(
        max(Counter(row.declared_pairs).values())
        for row in collision.exact_collisions
    )
    return {
        "audit_scope": collision.scope,
        "exact_case_count": exact_cases,
        "exact_case_safe_outcome_count": exact_safe,
        "exact_canonical_conflict_count": collision.exact_canonical_conflict_count,
        "exact_declared_history_oracle_accuracy": (
            exact_oracle_correct / exact_members if exact_members else 1.0
        ),
        "exact_png_class_count": len(collision.exact_collisions),
        "false_visible_singleton_count": false_visible_singletons,
        "identical_png_prediction_inconsistency_count": prediction_inconsistencies,
        "near_case_count": sum(case.expected_relation == "near" for case in cases),
        "resolved_target_case_count": sum(
            row["target_status"] == "resolved" for row in rows
        ),
        "qualifying_near_collision_count": (
            collision.qualifying_near_collision_count
        ),
        "retained_near_collision_count": len(collision.near_collisions),
        "max_retained_near_collisions": collision.max_retained_near_collisions,
        "rows": rows,
        "sample_count": len(audit_candidates),
        "unresolved_target_case_count": sum(
            row["target_status"] == "unresolved" for row in rows
        ),
    }


def _structural_stress_audit() -> dict[str, object]:
    """Exercise connected boundaries and junctions absent from the easy grid."""

    point = synthetic.Point
    line = synthetic.LineAction
    arc = synthetic.ArcAction
    programs = (
        synthetic.Program(
            "stress-corner",
            (
                line("first", point(200, 300), point(512, 300)),
                line("second", point(512, 300), point(512, 760)),
            ),
        ),
        synthetic.Program(
            "stress-line-arc",
            (
                line("line", point(160, 512), point(448, 512)),
                arc(
                    "arc",
                    point(448, 512),
                    point(640, 320),
                    point(832, 512),
                ),
            ),
        ),
        synthetic.Program(
            "stress-t-junction",
            (
                line("stem", point(512, 160), point(512, 864)),
                line("branch", point(512, 512), point(800, 512)),
            ),
        ),
        synthetic.Program(
            "stress-thinning-erased-crossbar",
            (
                line("stem", point(512, 160), point(512, 864)),
                line("crossbar", point(480, 512), point(544, 512)),
            ),
        ),
        synthetic.Program(
            "stress-shallow-corner",
            (
                line("first", point(192, 512), point(512, 512)),
                line("second", point(512, 512), point(632, 556)),
            ),
        ),
        synthetic.Program(
            "stress-offset-endpoint-branch",
            (
                line("stem", point(512, 160), point(512, 864)),
                line("branch", point(555, 69), point(501, 219)),
            ),
        ),
        synthetic.Program(
            "stress-moderate-corner",
            (
                line("first", point(192, 512), point(512, 512)),
                line("second", point(512, 512), point(693, 693)),
            ),
        ),
        synthetic.Program(
            "stress-polygonal-semicircle",
            (
                line("line-0", point(192, 512), point(286, 286)),
                line("line-1", point(286, 286), point(512, 192)),
                line("line-2", point(512, 192), point(738, 286)),
                line("line-3", point(738, 286), point(832, 512)),
            ),
        ),
        synthetic.Program(
            "stress-legal-shallow-arc",
            (
                arc(
                    "arc",
                    point(432, 512),
                    point(512, 480),
                    point(592, 512),
                ),
            ),
        ),
    )
    rows: list[dict[str, object]] = []
    safe_count = 0
    for program in programs:
        panel = synthetic.render_program(program)
        output = _safe_ordered_prediction(panel.png_bytes)
        candidates = tuple(tuple(pair) for pair in output["candidate_pairs"])
        target = panel.canonical_visible_pair
        if target is None:
            truth: tuple[int, int] | None = None
            safe = output["disposition"] in ("AMBIGUOUS", "GAP") and (
                output["disposition"] == "AMBIGUOUS" or not candidates
            )
            target_status = "unresolved"
        else:
            truth = target.as_tuple()
            safe = output["disposition"] == "GAP" or truth in candidates
            target_status = "resolved"
        safe_count += safe
        rows.append(
            {
                "candidate_pairs": [list(pair) for pair in candidates],
                "case_id": program.carrier_family,
                "disposition": output["disposition"],
                "reason": output["reason"],
                "safe_set_or_gap": safe,
                "target_pair": None if truth is None else _pair_data(truth),
                "target_status": target_status,
            }
        )
    return {
        "case_count": len(rows),
        "claim_scope": "bounded_connected_boundary_and_junction_stress_only",
        "rows": rows,
        "resolved_target_case_count": sum(
            row["target_status"] == "resolved" for row in rows
        ),
        "safe_set_or_gap_count": safe_count,
        "unresolved_target_case_count": sum(
            row["target_status"] == "unresolved" for row in rows
        ),
    }


def run_paired_synthetic_benchmark(
    *,
    evaluation_families: tuple[str, ...] = ("radial", "staggered"),
    nuisances: tuple[synthetic.Nuisance, ...] | None = None,
) -> dict[str, object]:
    """Run the non-authorizing paired experiment on locally rendered PNGs."""

    source_sha256()
    if nuisances is None:
        nuisances = synthetic.default_nuisances()
    if type(evaluation_families) is not tuple or not evaluation_families:
        raise SyntheticBenchmarkError("evaluation-family inventory differs")
    corpus = synthetic.build_balanced_corpus(nuisances=nuisances)
    corpus_collision_audit = synthetic.audit_collisions(
        corpus,
        max_near_comparisons=20_000,
        max_near_results=12,
        near_xor_limit=32,
    )
    split = synthetic.carrier_disjoint_split(
        corpus, held_out_families=evaluation_families
    )
    training, evaluation = _validate_split(split)
    control, unique_training, training_features = _fit_pooled_control(training)
    control_predictions, evaluation_features = _pooled_predictions(
        control, evaluation
    )
    ordered_outputs = tuple(
        _safe_ordered_prediction(sample.panel.png_bytes) for sample in evaluation
    )
    ordered_candidates = tuple(
        tuple(tuple(pair) for pair in output["candidate_pairs"])
        for output in ordered_outputs
    )
    control_candidates = tuple((pair,) for pair in control_predictions)
    control_metrics = _metrics(evaluation, control_candidates)
    ordered_metrics = _metrics(evaluation, ordered_candidates)
    control_by_family = _carrier_family_metrics(evaluation, control_candidates)
    ordered_by_family = _carrier_family_metrics(evaluation, ordered_candidates)
    truth = tuple(
        _required_visible_pair(sample.panel).as_tuple() for sample in evaluation
    )
    ordered_correct = tuple(
        len(candidates) == 1 and candidates[0] == wanted
        for candidates, wanted in zip(ordered_candidates, truth, strict=True)
    )
    control_correct = tuple(
        prediction == wanted
        for prediction, wanted in zip(control_predictions, truth, strict=True)
    )
    family_resampling = _descriptive_family_resampling(
        evaluation, ordered_correct, control_correct
    )
    counterfactual = _identifiability_counterfactual_audit()
    structural_stress = _structural_stress_audit()
    pooled_nearest = _nearest_cross_target_pooled_audit(
        training_features, evaluation, evaluation_features
    )
    paired_delta = (
        float(ordered_metrics["joint_singleton_accuracy"])
        - float(control_metrics["joint_singleton_accuracy"])
    )
    gates = {
        "all_54_visible_pairs_in_both_roles": (
            {_required_visible_pair(sample.panel) for sample in training}
            == set(synthetic.valid_count_pairs())
            == {_required_visible_pair(sample.panel) for sample in evaluation}
        ),
        "balanced_corpus_has_no_exact_target_conflict": (
            corpus_collision_audit.exact_canonical_conflict_count == 0
        ),
        "bounded_counterfactuals_are_preserved_without_false_singletons": (
            counterfactual["exact_case_count"] > 0
            and counterfactual["exact_case_safe_outcome_count"]
            == counterfactual["exact_case_count"]
            and counterfactual["exact_canonical_conflict_count"] == 0
            and counterfactual["false_visible_singleton_count"] == 0
            and counterfactual["identical_png_prediction_inconsistency_count"] == 0
        ),
        "structural_stress_returns_safe_set_or_gap": (
            structural_stress["safe_set_or_gap_count"]
            == structural_stress["case_count"]
        ),
        "historical_panel_anchor_descriptive_p05_exceeded": (
            family_resampling["ordered_joint_accuracy_family_resampling_p05"]
            > HISTORICAL_PANEL_JOINT_ANCHOR
        ),
        "historical_carrier_macro_anchor_exceeded": (
            ordered_metrics["carrier_family_macro_joint_singleton_accuracy"]
            > HISTORICAL_CARRIER_MACRO_ANCHOR
        ),
        "historically_unseen_pair_macro_accuracy": (
            ordered_metrics["historically_unseen_pair_macro_singleton_accuracy"]
            >= GATE_THRESHOLDS[
                "historically_unseen_pair_macro_accuracy_at_least"
            ]
        ),
        "ordered_joint_accuracy": (
            ordered_metrics["joint_singleton_accuracy"]
            >= GATE_THRESHOLDS["ordered_joint_accuracy_at_least"]
        ),
        "paired_delta_descriptive_p05_positive": (
            family_resampling["paired_delta_family_resampling_p05"] > 0.0
        ),
        "paired_delta_material": (
            paired_delta
            >= GATE_THRESHOLDS["paired_joint_accuracy_delta_at_least"]
        ),
    }
    row_records = []
    for sample, control_pair, output in zip(
        evaluation, control_predictions, ordered_outputs, strict=True
    ):
        row_records.append(
            {
                "canonical_visible_pair": _pair_data(
                    _required_visible_pair(sample.panel)
                ),
                "carrier_family": sample.panel.carrier_family,
                "carrier_id": sample.carrier_id,
                "control_pair": _pair_data(control_pair),
                "declared_pair": _pair_data(sample.panel.declared_pair),
                "ordered_candidate_pairs": output["candidate_pairs"],
                "ordered_disposition": output["disposition"],
                "ordered_reason": output["reason"],
                "png_sha256": sample.panel.png_sha256,
                "sample_id": sample.sample_id,
            }
        )
    training_manifest = _sample_manifest(training, role="training")
    evaluation_manifest = _sample_manifest(evaluation, role="evaluation")
    body: dict[str, object] = {
        "authorization": {
            "benchmark_promotion": False,
            "calibration_target_query_authorized": False,
            "new_campaign_authority_created": False,
            "official_benchmark_or_generalization_claim_authorized": False,
            "official_data_inputs_authorized": False,
            "synthetic_only": True,
        },
        "bindings": {
            "control_dependency_source_addresses": pooled.dependency_source_addresses(),
            "control_feature_source_sha256": "sha256:" + pooled.source_sha256(),
            "ordered_observer_source_sha256": "sha256:" + ordered.source_sha256(),
            "synthetic_generator_source_sha256": "sha256:" + synthetic.source_sha256(),
            "benchmark_source_sha256": "sha256:" + source_sha256(),
        },
        "family_resampling_sensitivity": family_resampling,
        "claim_scope": (
            "paired_locally_generated_synthetic_carrier_disjoint_visible_"
            "primitive_count_engineering_only"
        ),
        "control": {
            "algorithm_id": CONTROL_ID,
            "feature_count": len(pooled.FEATURE_NAMES),
            "parameters": dict(CONTROL_PARAMETERS),
        },
        "counterfactual_identifiability_audit": counterfactual,
        "balanced_corpus_collision_audit": {
            "audit_scope": corpus_collision_audit.scope,
            "candidate_count": len(corpus_collision_audit.examined_candidate_ids),
            "exact_canonical_conflict_count": (
                corpus_collision_audit.exact_canonical_conflict_count
            ),
            "exact_collision_class_count": len(
                corpus_collision_audit.exact_collisions
            ),
            "qualifying_near_collision_count": (
                corpus_collision_audit.qualifying_near_collision_count
            ),
            "retained_near_collision_count": len(
                corpus_collision_audit.near_collisions
            ),
            "max_retained_near_collisions": (
                corpus_collision_audit.max_retained_near_collisions
            ),
            "near_comparisons": corpus_collision_audit.compared_different_target_pairs,
            "possible_different_target_pairs": (
                corpus_collision_audit.possible_different_target_pairs
            ),
        },
        "pooled_feature_nearest_cross_target_audit": pooled_nearest,
        "structural_stress_audit": structural_stress,
        "evaluation": {
            "carrier_families": sorted(
                {sample.panel.carrier_family for sample in evaluation}
            ),
            "manifest_digest": "sha256:" + canonical_digest(evaluation_manifest),
            "row_count": len(evaluation),
            "unique_png_count": len(
                {sample.panel.png_sha256 for sample in evaluation}
            ),
        },
        "exposure_accounting": {
            "fresh_non_synthetic_pixels_opened": 0,
            "network_calls": 0,
            "official_labels_opened": 0,
            "official_pixels_opened": 0,
            "official_programs_opened": 0,
            "synthetic_evaluation_png_occurrences_opened_by_each_observer": len(
                evaluation
            ),
            "synthetic_training_png_occurrences_opened_by_control": len(training),
        },
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "gates": {**gates, "passed": all(gates.values())},
        "historical_comparison": {
            "carrier_macro_anchor": HISTORICAL_CARRIER_MACRO_ANCHOR,
            "comparison_kind": "cross_corpus_sanity_anchor_not_paired_baseline",
            "panel_joint_anchor": HISTORICAL_PANEL_JOINT_ANCHOR,
            "panel_joint_anchor_head": "historical_separate_marginal_head_oof",
            "selected_direct_pair_head_comparison_claimed": False,
        },
        "metrics": {
            "by_carrier_family": {
                family: {
                    "control": control_by_family[family],
                    "ordered": ordered_by_family[family],
                }
                for family in sorted(ordered_by_family)
            },
            "control": control_metrics,
            "ordered": ordered_metrics,
            "ordered_disposition_counts": _disposition_counts(ordered_outputs),
            "paired_joint_singleton_accuracy_delta": paired_delta,
        },
        "ordered_observer": {
            "algorithm_id": ORDERED_ID,
            "normal_form_role": (
                "post_fit_target_resolvability_gate_not_pair_selection"
            ),
            "pair_selection_uses_target": False,
            "set_valued_minimum_fit_inventory_retained": True,
            "structural_set_or_gap_is_independent_observer_evidence": False,
            "unresolved_singleton_suppression_uses_target": True,
        },
        "paired_rows": row_records,
        "runtime": pooled.runtime_fingerprint(),
        "schema": SCHEMA,
        "target": {
            "algorithm_id": TARGET_ID,
            "balanced_rows_require_resolved_target": True,
            "connected_component_without_exact_singleton_normal_form": (
                "unresolved_and_ordered_observer_must_return_set_or_gap"
            ),
            "generator_history_recovery_claimed": False,
            "historically_unseen_pair_count": len(HISTORICALLY_UNSEEN_PAIRS),
            "partial_target": True,
            "valid_pair_count": len(synthetic.valid_count_pairs()),
        },
        "training": {
            "carrier_families": sorted(
                {sample.panel.carrier_family for sample in training}
            ),
            "manifest_digest": "sha256:" + canonical_digest(training_manifest),
            "row_count": len(training),
            "unique_png_count": len(
                {sample.panel.png_sha256 for sample in training}
            ),
            "deduplicated_fit_row_count": len(unique_training),
        },
    }
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


__all__ = (
    "FAMILY_RESAMPLING_REPLICATES",
    "CONTROL_ID",
    "HISTORICALLY_UNSEEN_PAIRS",
    "ORDERED_ID",
    "SCHEMA",
    "SyntheticBenchmarkError",
    "run_paired_synthetic_benchmark",
    "source_sha256",
)
