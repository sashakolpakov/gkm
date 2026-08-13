"""Carrier-disjoint benchmark for connected line/arc program inversion.

This module compares two observers on the exact same locally rendered PNGs:

* a freshly fitted fixed-32 ExtraTrees control over the neutral 112 pooled
  features; and
* the target-independent exact-cover synthesizer.

Targets come from the separately implemented exhaustive connected-fixture
oracle.  Training targets supervise only the pooled control; held-out targets
are constructed after raw predictions have been materialized.  In particular,
:func:`raw_synthesizer_outputs` has no target callback and remains usable when
the target oracle is replaced by a failing sentinel.

The target oracle and raw observer deliberately share the fixed primitive
catalog, including the geometry inventory of held carrier families.  Thus the
carrier split tests exact catalog inversion and a pooled learning control; it
does not test induction of an unseen rendering grammar or unseen catalog.

The experiment is synthetic engineering evidence only.  It has no corpus or
filesystem input, no command-line surface, and grants no official benchmark,
calibration, target, query, or campaign authority.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from io import BytesIO
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from bongard import panel_action_count_connected_synthetic as connected
from bongard import panel_action_count_connected_synthesizer as synthesizer
from bongard import panel_action_count_ordered_path_inversion as ordered
from bongard import panel_action_count_synthetic_pooled_control as pooled
from bongard.canonical import canonical_digest
from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)


SCHEMA: Final = "gkm.bongard-connected-paired-synthetic-benchmark.v1"
CONTROL_ID: Final = "connected-four-scale-112-pooled-fixed-32-extra-trees/v1"
SYNTHESIZER_ID: Final = "connected-exact-catalog-cover-raw-synthesizer/v1"
EVALUATION_FAMILIES: Final = ("radial", "staggered")
TRAINING_FAMILIES: Final = ("lattice", "perimeter", "pinwheel")
EXPECTED_LAYOUT_COUNTS_PER_CELL: Final = MappingProxyType(
    {"single_shape": 54, "two_shape": 52}
)
EXPECTED_BOUNDARY_KINDS: Final = ("AA", "AL", "LA", "LL")
EXPECTED_SINGLE_TARGET_SETS: Final = tuple(
    ((straight_count, total - straight_count),)
    for total in range(1, 10)
    for straight_count in range(total + 1)
)

CONTROL_PARAMETERS: Final = MappingProxyType(
    {
        "bootstrap": False,
        "class_weight": "balanced",
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "n_estimators": 32,
        "n_jobs": 1,
        "random_state": 260_814,
    }
)

GATE_THRESHOLDS: Final = MappingProxyType(
    {
        "raw_exact_target_set_accuracy_at_least": 0.50,
        "raw_minus_control_accuracy_at_least": 0.10,
        "each_layout_raw_accuracy_at_least": 0.50,
        "each_layout_raw_minus_control_at_least": 0.10,
        "matched_raw_minus_control_at_least": 0.10,
    }
)


class ConnectedBenchmarkError(RuntimeError):
    """The connected corpus, target, prediction, or metric invariant differs."""


def source_sha256() -> str:
    """Verify and return the import-time source address."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _pair_tuple(value: object) -> tuple[int, int]:
    if type(value) is tuple and len(value) == 2 and all(
        type(item) is int for item in value
    ):
        pair = value
    elif (
        type(getattr(value, "straight", None)) is int
        and type(getattr(value, "arc", None)) is int
    ):
        pair = (value.straight, value.arc)
    elif callable(getattr(value, "as_tuple", None)):
        pair = value.as_tuple()
    else:
        raise ConnectedBenchmarkError("count pair transport differs")
    if (
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(item) is not int or item < 0 for item in pair)
        or not 1 <= sum(pair) <= 9
    ):
        raise ConnectedBenchmarkError("count pair leaves the 54-cell universe")
    return pair


def _pair_data(value: object) -> list[int]:
    pair = _pair_tuple(value)
    return [pair[0], pair[1]]


def _sample_id(sample: object) -> str:
    value = getattr(sample, "sample_id", None)
    if type(value) is not str or not value:
        raise ConnectedBenchmarkError("sample identifier differs")
    return value


def _png_bytes(sample: object) -> bytes:
    value = getattr(sample, "png_bytes", None)
    if type(value) is not bytes or not value:
        raise ConnectedBenchmarkError("sample PNG transport differs")
    return value


def _png_digest(sample: object) -> str:
    raw = _png_bytes(sample)
    expected = "sha256:" + hashlib.sha256(raw).hexdigest()
    value = getattr(sample, "raster_digest", None)
    if value is None:
        value = getattr(sample, "png_sha256", None)
    if type(value) is not str or value != expected:
        raise ConnectedBenchmarkError("sample PNG address differs")
    return value


def _family(sample: object) -> str:
    program = getattr(sample, "panel_program", None)
    value = getattr(program, "carrier_family", None)
    if type(value) is not str or not value:
        raise ConnectedBenchmarkError("sample carrier family differs")
    return value


def _layout(sample: object) -> str:
    value = getattr(sample, "layout_truth", None)
    if value is None:
        value = getattr(getattr(sample, "panel_program", None), "layout", None)
    if type(value) is not str or value not in EXPECTED_LAYOUT_COUNTS_PER_CELL:
        raise ConnectedBenchmarkError("sample layout differs")
    return value


def _nuisance_id(sample: object) -> str:
    nuisance = getattr(sample, "nuisance", None)
    for name in ("identity", "nuisance_id"):
        value = getattr(nuisance, name, None)
        if type(value) is str and value:
            return value
    if type(nuisance) is str and nuisance:
        return nuisance
    raise ConnectedBenchmarkError("sample nuisance identity differs")


def _boundary_kinds(sample: object) -> tuple[str, ...]:
    raw = getattr(sample, "boundary_truth", None)
    if raw == ():
        return ()
    if type(raw) is str:
        values = (raw,)
    elif type(raw) is tuple:
        values = tuple(
            value
            if type(value) is str
            else getattr(value, "kind", None)
            or getattr(value, "boundary_kind", None)
            for value in raw
        )
    else:
        value = getattr(raw, "kind", None) or getattr(raw, "boundary_kind", None)
        values = (value,)
    normalized = tuple(sorted(set(values)))
    if any(
        type(value) is not str or value not in EXPECTED_BOUNDARY_KINDS
        for value in normalized
    ):
        raise ConnectedBenchmarkError("connected boundary truth differs")
    return normalized


def _target_for_sample(sample: object) -> object:
    target = connected.exact_cover_target(_png_bytes(sample))
    if getattr(target, "png_digest", None) != _png_digest(sample):
        raise ConnectedBenchmarkError("exact-cover target is not PNG-bound")
    return target


def _target_pairs(target: object) -> tuple[tuple[int, int], ...]:
    values = getattr(target, "count_pairs", None)
    if type(values) is not tuple or not values:
        raise ConnectedBenchmarkError("exact-cover target pair set differs")
    result = tuple(sorted({_pair_tuple(value) for value in values}))
    if len(result) != len(values):
        raise ConnectedBenchmarkError("exact-cover target repeats count pairs")
    return result


def _foreground_pixels(png_bytes: bytes) -> tuple[int, ...]:
    try:
        with Image.open(BytesIO(png_bytes)) as image:
            if (
                image.format != "PNG"
                or getattr(image, "n_frames", 1) != 1
                or image.size != (64, 64)
            ):
                raise ConnectedBenchmarkError("target integrity PNG differs")
            image.load()
            mask = np.asarray(image.convert("L"), dtype=np.uint8) < 128
    except ConnectedBenchmarkError:
        raise
    except Exception as exc:
        raise ConnectedBenchmarkError("target integrity PNG cannot be decoded") from exc
    foreground = tuple(int(pixel) for pixel in np.flatnonzero(mask))
    if not foreground:
        raise ConnectedBenchmarkError("target integrity PNG has no foreground")
    return foreground


def _target_integrity(target: object, png_bytes: bytes) -> bool:
    minimum = getattr(target, "minimum_primitive_count", None)
    hypotheses = getattr(target, "hypotheses", None)
    foreground = _foreground_pixels(png_bytes)
    if (
        type(minimum) is not int
        or not 1 <= minimum <= 9
        or type(hypotheses) is not tuple
        or not hypotheses
    ):
        return False
    pairs: set[tuple[int, int]] = set()
    for hypothesis in hypotheses:
        primitive_ids = getattr(hypothesis, "primitive_ids", None)
        covered = getattr(hypothesis, "covered_pixels", None)
        pair_value = getattr(hypothesis, "count_pair", None)
        if (
            type(primitive_ids) is not tuple
            or len(primitive_ids) != minimum
            or len(primitive_ids) != len(set(primitive_ids))
            or type(covered) is not tuple
            or covered != tuple(sorted(set(covered)))
            or covered != foreground
            or sum(_pair_tuple(pair_value)) != minimum
        ):
            return False
        pairs.add(_pair_tuple(pair_value))
    return tuple(sorted(pairs)) == _target_pairs(target)


def _build_and_split() -> tuple[tuple[object, ...], tuple[object, ...]]:
    corpus = connected.build_connected_corpus()
    if type(corpus) is not tuple or len(corpus) != 1_060:
        raise ConnectedBenchmarkError(
            "connected corpus must contain exactly 1,060 rows"
        )
    ids = tuple(_sample_id(sample) for sample in corpus)
    if len(ids) != len(set(ids)):
        raise ConnectedBenchmarkError("connected sample identifiers repeat")
    training = tuple(
        sorted(
            (sample for sample in corpus if _family(sample) not in EVALUATION_FAMILIES),
            key=lambda sample: (_sample_id(sample), _png_digest(sample)),
        )
    )
    evaluation = tuple(
        sorted(
            (sample for sample in corpus if _family(sample) in EVALUATION_FAMILIES),
            key=lambda sample: (_sample_id(sample), _png_digest(sample)),
        )
    )
    if len(training) != 636 or len(evaluation) != 424:
        raise ConnectedBenchmarkError("connected carrier split must be 636/424")
    train_families = {_family(sample) for sample in training}
    evaluation_families = {_family(sample) for sample in evaluation}
    if (
        len(train_families) != 3
        or evaluation_families != set(EVALUATION_FAMILIES)
        or train_families & evaluation_families
    ):
        raise ConnectedBenchmarkError("connected carrier families cross split roles")
    if {_png_digest(sample) for sample in training} & {
        _png_digest(sample) for sample in evaluation
    }:
        raise ConnectedBenchmarkError("exact connected PNG crosses split roles")
    for role, rows, family_count in (
        ("training", training, 3),
        ("evaluation", evaluation, 2),
    ):
        cells: dict[tuple[str, str, str], int] = Counter(
            (_family(sample), _nuisance_id(sample), _layout(sample))
            for sample in rows
        )
        nuisances = {_nuisance_id(sample) for sample in rows}
        if len(nuisances) != 2 or len(cells) != family_count * 2 * 2:
            raise ConnectedBenchmarkError(
                f"{role} carrier/nuisance/layout grid differs"
            )
        if any(
            count != EXPECTED_LAYOUT_COUNTS_PER_CELL[layout]
            for (_family_id, _nuisance, layout), count in cells.items()
        ):
            raise ConnectedBenchmarkError(f"{role} layout cell count differs")
    return training, evaluation


def _d4_orbit_digest(sample: object) -> str:
    callback = getattr(connected, "d4_raster_orbit_digest", None)
    if not callable(callback):
        raise ConnectedBenchmarkError("connected D4 orbit address is unavailable")
    value = callback(_png_bytes(sample))
    if type(value) is not str or not value.startswith("sha256:") or len(value) != 71:
        raise ConnectedBenchmarkError("connected D4 orbit address differs")
    return value


def _single_shape_target_coverage(
    samples: Sequence[object],
    targets: Mapping[str, tuple[tuple[int, int], ...]],
) -> dict[str, object]:
    cells: dict[tuple[str, str], list[tuple[tuple[int, int], ...]]] = defaultdict(
        list
    )
    for sample in samples:
        if _layout(sample) == "single_shape":
            cells[(_family(sample), _nuisance_id(sample))].append(
                targets[_sample_id(sample)]
            )
    expected = set(EXPECTED_SINGLE_TARGET_SETS)
    complete = all(
        len(values) == 54 and len(set(values)) == 54 and set(values) == expected
        for values in cells.values()
    )
    if len(cells) != 10 or not complete:
        raise ConnectedBenchmarkError(
            "single-shape cells do not contain the complete 54-target universe"
        )
    return {
        "cell_count": len(cells),
        "complete_54_target_set_in_every_cell": True,
        "target_set_universe": [
            [_pair_data(pair) for pair in target]
            for target in EXPECTED_SINGLE_TARGET_SETS
        ],
    }


def _d4_cross_role_audit(
    training: Sequence[object], evaluation: Sequence[object]
) -> dict[str, object]:
    train = tuple(sorted({_d4_orbit_digest(sample) for sample in training}))
    held_out = tuple(sorted({_d4_orbit_digest(sample) for sample in evaluation}))
    overlap = tuple(sorted(set(train) & set(held_out)))
    return {
        "cross_role_overlap_count": len(overlap),
        "full_eight_element_square_symmetry_orbit": True,
        "held_out_orbit_count": len(held_out),
        "held_out_orbit_set_digest": "sha256:" + canonical_digest(list(held_out)),
        "overlap": list(overlap),
        "training_orbit_count": len(train),
        "training_orbit_set_digest": "sha256:" + canonical_digest(list(train)),
    }


def _features(samples: Sequence[object]) -> np.ndarray:
    if not samples:
        raise ConnectedBenchmarkError("feature population is empty")
    result = np.ascontiguousarray(
        np.stack(
            [
                pooled.extract_issued_feature_vector(
                    _png_bytes(sample),
                    require_issued=connected.require_issued_connected_png,
                )
                for sample in samples
            ]
        ),
        dtype=np.float32,
    )
    if result.shape != (len(samples), 112) or not np.isfinite(result).all():
        raise ConnectedBenchmarkError("connected pooled feature matrix differs")
    return result


def _fit_control(
    training: Sequence[object], targets: Mapping[str, tuple[tuple[int, int], ...]]
) -> tuple[
    Any,
    tuple[object, ...],
    np.ndarray,
    tuple[tuple[tuple[int, int], ...], ...],
]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
    except ImportError as exc:  # pragma: no cover - environment failure
        raise ConnectedBenchmarkError("scikit-learn is unavailable") from exc
    by_png: dict[str, list[object]] = defaultdict(list)
    for sample in training:
        by_png[_png_digest(sample)].append(sample)
    unique: list[object] = []
    label_sets: list[tuple[tuple[int, int], ...]] = []
    for digest, rows in sorted(by_png.items()):
        pair_sets = {targets[_sample_id(sample)] for sample in rows}
        if len(pair_sets) != 1:
            raise ConnectedBenchmarkError(
                f"training PNG {digest} has conflicting exact-cover targets"
            )
        pair_set = next(iter(pair_sets))
        unique.append(min(rows, key=_sample_id))
        label_sets.append(pair_set)
    class_targets = tuple(sorted(set(label_sets)))
    if not class_targets:
        raise ConnectedBenchmarkError("pooled-control class universe is empty")
    class_index = {target: index for index, target in enumerate(class_targets)}
    labels = [class_index[target] for target in label_sets]
    features = _features(unique)
    estimator = ExtraTreesClassifier(**dict(CONTROL_PARAMETERS)).fit(
        features, np.asarray(labels, dtype=np.int64)
    )
    return estimator, tuple(unique), features, class_targets


def _control_outputs(
    estimator: Any,
    features: np.ndarray,
    class_targets: Sequence[tuple[tuple[int, int], ...]],
) -> tuple[dict[str, object], ...]:
    encoded = estimator.predict(features)
    rows: list[dict[str, object]] = []
    for value in encoded:
        index = int(value)
        if not 0 <= index < len(class_targets):
            raise ConnectedBenchmarkError("pooled-control class prediction differs")
        candidates = tuple(class_targets[index])
        rows.append(
            {
                "candidate_pairs": candidates,
                "disposition": (
                    "IDENTIFIED" if len(candidates) == 1 else "AMBIGUOUS"
                ),
                "exact_reconstruction": False,
                "hypothesis_count": len(candidates),
                "minimum_primitive_count": None,
                "reason": None,
            }
        )
    return tuple(rows)


def raw_synthesizer_outputs(samples: Sequence[object]) -> tuple[dict[str, object], ...]:
    """Construct raw predictions without reading or accepting any target value."""

    if type(samples) not in (tuple, list) or any(
        type(_png_bytes(sample)) is not bytes for sample in samples
    ):
        raise ConnectedBenchmarkError("raw synthesizer population differs")
    rows: list[dict[str, object]] = []
    for sample in samples:
        png_bytes = _png_bytes(sample)
        outcome = synthesizer.fit_png_hypotheses(png_bytes)
        candidates = tuple(outcome.candidate_pairs)
        if candidates != tuple(sorted(set(candidates))) or any(
            _pair_tuple(pair) != pair for pair in candidates
        ):
            raise ConnectedBenchmarkError("raw synthesizer candidate set differs")
        foreground = _foreground_pixels(png_bytes)
        independently_exact = bool(outcome.hypotheses) and all(
            hypothesis.reconstructed_ink_pixels == foreground
            and hypothesis.xor_pixel_count == 0
            and hypothesis.intersection_over_union == 1.0
            for hypothesis in outcome.hypotheses
        )
        if independently_exact != outcome.exact_reconstruction:
            raise ConnectedBenchmarkError(
                "raw synthesizer exact reconstruction evidence differs"
            )
        rows.append(
            {
                "candidate_pairs": candidates,
                "disposition": outcome.disposition,
                "exact_reconstruction": independently_exact,
                "hypothesis_count": len(outcome.hypotheses),
                "minimum_primitive_count": outcome.minimum_primitive_count,
                "reason": outcome.reason,
            }
        )
    return tuple(rows)


def _non_held_catalog_dependency_audit(
    samples: Sequence[object],
) -> dict[str, object]:
    """Measure exact cover after withholding every held-family-only mask.

    This deliberately reuses the raw synthesizer's target-free mask search,
    not the exact-cover target.  A same-geometry alias remains available when
    at least one equivalent primitive ID belongs to a training family.
    """

    all_masks = synthesizer._catalog_masks()  # noqa: SLF001
    non_held_masks = tuple(
        mask
        for mask in all_masks
        if any(
            primitive_id.split(".", maxsplit=1)[0] not in EVALUATION_FAMILIES
            for primitive_id in mask.equivalent_ids
        )
    )
    if not non_held_masks:
        raise ConnectedBenchmarkError("non-held catalog audit is empty")
    strict_training_mask_count = sum(
        any(
            primitive_id.split(".", maxsplit=1)[0] in TRAINING_FAMILIES
            for primitive_id in mask.equivalent_ids
        )
        for mask in all_masks
    )
    stress_mask_count = len(non_held_masks) - strict_training_mask_count
    exact_by_layout: Counter[str] = Counter()
    for sample in samples:
        mask = synthesizer._decode_exact_mask(_png_bytes(sample))  # noqa: SLF001
        foreground = tuple(int(pixel) for pixel in np.flatnonzero(mask))
        target_bits = synthesizer._pixels_to_bits(foreground)  # noqa: SLF001
        eligible = tuple(
            candidate
            for candidate in non_held_masks
            if candidate.bits & ~target_bits == 0
        )
        covers = synthesizer._minimum_exact_covers(  # noqa: SLF001
            target_bits, eligible
        )
        exact_by_layout[_layout(sample)] += bool(covers)
    exact_count = sum(exact_by_layout.values())
    return {
        "audit_uses_target_oracle": False,
        "full_catalog_mask_count": len(all_masks),
        "held_family_only_masks_removed": True,
        "held_family_only_mask_count": len(all_masks) - len(non_held_masks),
        "held_out_exact_cover_count": exact_count,
        "held_out_row_count": len(samples),
        "layout_exact_cover_counts": {
            layout: exact_by_layout[layout]
            for layout in EXPECTED_LAYOUT_COUNTS_PER_CELL
        },
        "non_held_catalog_mask_count": len(non_held_masks),
        "strict_training_family_catalog_mask_count": strict_training_mask_count,
        "synthetic_stress_catalog_mask_count": stress_mask_count,
    }


def _metric_record(
    targets: Sequence[tuple[tuple[int, int], ...]],
    outputs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(targets) != len(outputs) or not targets:
        raise ConnectedBenchmarkError("metric population differs")
    exact = []
    contains = []
    singleton = []
    false_singleton = []
    exact_reconstruction = []
    dispositions: Counter[str] = Counter()
    for target, output in zip(targets, outputs, strict=True):
        candidates = tuple(output["candidate_pairs"])
        dispositions[str(output["disposition"])] += 1
        exact.append(candidates == target)
        contains.append(bool(set(candidates) & set(target)))
        singleton.append(len(target) == 1 and candidates == target)
        false_singleton.append(len(target) > 1 and len(candidates) == 1)
        exact_reconstruction.append(bool(output["exact_reconstruction"]))
    denominator = len(targets)
    return {
        "ambiguous_target_count": sum(len(target) > 1 for target in targets),
        "candidate_set_intersects_target_accuracy": sum(contains) / denominator,
        "denominator": denominator,
        "disposition_counts": {
            value: dispositions.get(value, 0)
            for value in ("AMBIGUOUS", "ERROR", "GAP", "IDENTIFIED")
        },
        "exact_count_pair_set_accuracy": sum(exact) / denominator,
        "exact_reconstruction_rate": sum(exact_reconstruction) / denominator,
        "false_singleton_on_ambiguous_target_count": sum(false_singleton),
        "resolved_joint_singleton_accuracy_in_fixed_denominator": (
            sum(singleton) / denominator
        ),
    }


def _indexed_metrics(
    samples: Sequence[object],
    targets: Sequence[tuple[tuple[int, int], ...]],
    outputs: Sequence[Mapping[str, object]],
    indices: Sequence[int],
) -> dict[str, object]:
    del samples  # The index binding is checked by the caller's strict zips.
    return _metric_record(
        tuple(targets[index] for index in indices),
        tuple(outputs[index] for index in indices),
    )


def _group_metrics(
    samples: Sequence[object],
    targets: Sequence[tuple[tuple[int, int], ...]],
    control_outputs: Sequence[Mapping[str, object]],
    raw_outputs: Sequence[Mapping[str, object]],
    *,
    key_values: Mapping[str, tuple[int, ...]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, indices in sorted(key_values.items()):
        control = _indexed_metrics(samples, targets, control_outputs, indices)
        raw = _indexed_metrics(samples, targets, raw_outputs, indices)
        result[key] = {
            "control": control,
            "raw_synthesizer": raw,
            "raw_minus_control_exact_accuracy": (
                float(raw["exact_count_pair_set_accuracy"])
                - float(control["exact_count_pair_set_accuracy"])
            ),
        }
    return result


def _matched_counterfactual_assignment(
    samples: Sequence[object],
    features: np.ndarray,
    targets: Sequence[tuple[tuple[int, int], ...]],
    control_outputs: Sequence[Mapping[str, object]],
    raw_outputs: Sequence[Mapping[str, object]],
    training_features: np.ndarray,
) -> dict[str, object]:
    if features.shape != (424, 112) or training_features.ndim != 2:
        raise ConnectedBenchmarkError("matched feature population differs")
    scale = np.std(training_features.astype(np.float64), axis=0)
    scale = np.where(scale > 1e-7, scale, 1.0)
    normalized = features.astype(np.float64) / scale
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[(_family(sample), _nuisance_id(sample), _layout(sample))].append(index)
    if len(groups) != 8:
        raise ConnectedBenchmarkError("matched assignment stratum count differs")
    pairs: list[dict[str, object]] = []
    used: list[int] = []
    for stratum, indices in sorted(groups.items()):
        ordered_indices = sorted(
            indices,
            key=lambda index: (targets[index], _sample_id(samples[index])),
        )
        left = ordered_indices[::2]
        right = ordered_indices[1::2]
        expected = 27 if stratum[2] == "single_shape" else 26
        if len(left) != expected or len(right) != expected:
            raise ConnectedBenchmarkError("matched alternating half size differs")
        cost = np.sqrt(
            np.mean(
                np.square(
                    normalized[np.asarray(left), None, :]
                    - normalized[np.asarray(right), :][None, :, :]
                ),
                axis=2,
            )
        )
        for row, left_index in enumerate(left):
            for column, right_index in enumerate(right):
                if targets[left_index] == targets[right_index]:
                    cost[row, column] = np.inf
        if (
            not np.isfinite(cost).any(axis=1).all()
            or not np.isfinite(cost).any(axis=0).all()
        ):
            raise ConnectedBenchmarkError(
                "matched assignment has no cross-target cover"
            )
        row_indices, columns = linear_sum_assignment(cost)
        for row, column in zip(row_indices, columns, strict=True):
            first = left[int(row)]
            second = right[int(column)]
            distance = float(cost[int(row), int(column)])
            if not np.isfinite(distance) or targets[first] == targets[second]:
                raise ConnectedBenchmarkError("matched assignment retained same target")
            used.extend((first, second))
            first_control_candidates = tuple(
                control_outputs[first]["candidate_pairs"]
            )
            second_control_candidates = tuple(
                control_outputs[second]["candidate_pairs"]
            )
            first_raw_candidates = tuple(raw_outputs[first]["candidate_pairs"])
            second_raw_candidates = tuple(raw_outputs[second]["candidate_pairs"])
            pairs.append(
                {
                    "distance": distance,
                    "family": stratum[0],
                    "first_control_candidate_pairs": [
                        _pair_data(pair) for pair in first_control_candidates
                    ],
                    "first_control_exact": first_control_candidates == targets[first],
                    "first_raw_candidate_pairs": [
                        _pair_data(pair) for pair in first_raw_candidates
                    ],
                    "first_raw_exact": first_raw_candidates == targets[first],
                    "first_sample_id": _sample_id(samples[first]),
                    "first_target_pairs": [_pair_data(pair) for pair in targets[first]],
                    "layout": stratum[2],
                    "nuisance": stratum[1],
                    "second_control_candidate_pairs": [
                        _pair_data(pair) for pair in second_control_candidates
                    ],
                    "second_control_exact": (
                        second_control_candidates == targets[second]
                    ),
                    "second_raw_candidate_pairs": [
                        _pair_data(pair) for pair in second_raw_candidates
                    ],
                    "second_raw_exact": second_raw_candidates == targets[second],
                    "second_sample_id": _sample_id(samples[second]),
                    "second_target_pairs": [
                        _pair_data(pair) for pair in targets[second]
                    ],
                }
            )
    pairs.sort(
        key=lambda row: (
            row["family"], row["nuisance"], row["layout"],
            row["first_sample_id"], row["second_sample_id"],
        )
    )
    if len(pairs) != 212 or sorted(used) != list(range(424)):
        raise ConnectedBenchmarkError(
            "matched assignment is not an exact occurrence partition"
        )
    if Counter(row["layout"] for row in pairs) != Counter(
        {"single_shape": 108, "two_shape": 104}
    ):
        raise ConnectedBenchmarkError("matched assignment layout count differs")
    occurrence_indices = tuple(sorted(used))
    control = _indexed_metrics(
        samples, targets, control_outputs, occurrence_indices
    )
    raw = _indexed_metrics(samples, targets, raw_outputs, occurrence_indices)
    control_both = sum(
        bool(row["first_control_exact"] and row["second_control_exact"])
        for row in pairs
    ) / len(pairs)
    raw_both = sum(
        bool(row["first_raw_exact"] and row["second_raw_exact"])
        for row in pairs
    ) / len(pairs)
    return {
        "assignment_algorithm": (
            "stratum_local_training_sd_scaled_rms_hungarian_alternating_target_halves"
        ),
        "control": control,
        "every_evaluation_occurrence_used_exactly_once": True,
        "layout_pair_counts": {"single_shape": 108, "two_shape": 104},
        "occurrence_count": 424,
        "pair_count": 212,
        "pair_level_both_endpoints_exact": {
            "control_accuracy": control_both,
            "denominator": len(pairs),
            "raw_minus_control_accuracy": raw_both - control_both,
            "raw_synthesizer_accuracy": raw_both,
        },
        "pair_manifest_digest": "sha256:" + canonical_digest(pairs),
        "pairs": pairs,
        "occurrence_raw_minus_control_exact_accuracy": (
            float(raw["exact_count_pair_set_accuracy"])
            - float(control["exact_count_pair_set_accuracy"])
        ),
        "raw_synthesizer": raw,
        "same_target_pair_count": 0,
    }


def run_connected_benchmark() -> dict[str, object]:
    """Run the non-authorizing connected synthetic comparison."""

    source_sha256()
    training, evaluation = _build_and_split()
    targets_by_id: dict[str, tuple[tuple[int, int], ...]] = {}
    target_integrity: dict[str, bool] = {}
    for sample in training:
        target = _target_for_sample(sample)
        targets_by_id[_sample_id(sample)] = _target_pairs(target)
        target_integrity[_sample_id(sample)] = _target_integrity(
            target, _png_bytes(sample)
        )

    estimator, unique_training, training_features, control_class_targets = _fit_control(
        training, targets_by_id
    )
    evaluation_features = _features(evaluation)
    control_outputs = _control_outputs(
        estimator, evaluation_features, control_class_targets
    )
    # Materialize the raw held-out observations before constructing any
    # held-out target.  This makes the no-target prediction boundary temporal
    # as well as an API and monkeypatch invariant.
    raw_outputs = raw_synthesizer_outputs(evaluation)
    for sample in evaluation:
        target = _target_for_sample(sample)
        targets_by_id[_sample_id(sample)] = _target_pairs(target)
        target_integrity[_sample_id(sample)] = _target_integrity(
            target, _png_bytes(sample)
        )
    if len(target_integrity) != 1_060 or not all(target_integrity.values()):
        raise ConnectedBenchmarkError("exact-cover target integrity differs")
    single_shape_coverage = _single_shape_target_coverage(
        training + evaluation, targets_by_id
    )
    evaluation_targets = tuple(
        targets_by_id[_sample_id(sample)] for sample in evaluation
    )
    control_metrics = _metric_record(evaluation_targets, control_outputs)
    raw_metrics = _metric_record(evaluation_targets, raw_outputs)

    layouts = {
        layout: tuple(
            index for index, sample in enumerate(evaluation)
            if _layout(sample) == layout
        )
        for layout in EXPECTED_LAYOUT_COUNTS_PER_CELL
    }
    families = {
        family: tuple(
            index for index, sample in enumerate(evaluation)
            if _family(sample) == family
        )
        for family in EVALUATION_FAMILIES
    }
    boundary_kinds = {
        kind: tuple(
            index for index, sample in enumerate(evaluation)
            if kind in _boundary_kinds(sample)
        )
        for kind in EXPECTED_BOUNDARY_KINDS
    }
    if any(not indices for indices in boundary_kinds.values()):
        raise ConnectedBenchmarkError("a connected boundary kind is absent")
    by_layout = _group_metrics(
        evaluation, evaluation_targets, control_outputs, raw_outputs,
        key_values=layouts,
    )
    by_family = _group_metrics(
        evaluation, evaluation_targets, control_outputs, raw_outputs,
        key_values=families,
    )
    by_boundary = _group_metrics(
        evaluation, evaluation_targets, control_outputs, raw_outputs,
        key_values=boundary_kinds,
    )
    matched = _matched_counterfactual_assignment(
        evaluation,
        evaluation_features,
        evaluation_targets,
        control_outputs,
        raw_outputs,
        training_features,
    )
    d4_audit = _d4_cross_role_audit(training, evaluation)
    catalog_dependency = _non_held_catalog_dependency_audit(evaluation)

    overall_delta = (
        float(raw_metrics["exact_count_pair_set_accuracy"])
        - float(control_metrics["exact_count_pair_set_accuracy"])
    )
    gates = {
        "all_exact_cover_targets_integral": all(target_integrity.values()),
        "all_54_single_shape_targets_in_every_cell": bool(
            single_shape_coverage["complete_54_target_set_in_every_cell"]
        ),
        "d4_orbits_disjoint_across_roles": d4_audit["cross_role_overlap_count"] == 0,
        "each_layout_raw_accuracy": all(
            float(row["raw_synthesizer"]["exact_count_pair_set_accuracy"])
            >= GATE_THRESHOLDS["each_layout_raw_accuracy_at_least"]
            for row in by_layout.values()
        ),
        "each_layout_raw_lift": all(
            float(row["raw_minus_control_exact_accuracy"])
            >= GATE_THRESHOLDS["each_layout_raw_minus_control_at_least"]
            for row in by_layout.values()
        ),
        "matched_raw_lift": (
            float(
                matched["pair_level_both_endpoints_exact"][
                    "raw_minus_control_accuracy"
                ]
            )
            >= GATE_THRESHOLDS["matched_raw_minus_control_at_least"]
        ),
        "overall_raw_accuracy": (
            float(raw_metrics["exact_count_pair_set_accuracy"])
            >= GATE_THRESHOLDS["raw_exact_target_set_accuracy_at_least"]
        ),
        "overall_raw_lift": (
            overall_delta
            >= GATE_THRESHOLDS["raw_minus_control_accuracy_at_least"]
        ),
        "raw_exact_reconstruction": raw_metrics["exact_reconstruction_rate"] == 1.0,
        "raw_zero_false_singletons_on_ambiguous_targets": (
            raw_metrics["false_singleton_on_ambiguous_target_count"] == 0
        ),
    }

    paired_rows = [
        {
            "boundary_kinds": list(_boundary_kinds(sample)),
            "carrier_family": _family(sample),
            "control_candidate_pairs": [
                _pair_data(pair) for pair in control["candidate_pairs"]
            ],
            "layout": _layout(sample),
            "nuisance": _nuisance_id(sample),
            "png_sha256": _png_digest(sample),
            "raw_candidate_pairs": [
                _pair_data(pair) for pair in raw["candidate_pairs"]
            ],
            "raw_disposition": raw["disposition"],
            "raw_exact_reconstruction": raw["exact_reconstruction"],
            "sample_id": _sample_id(sample),
            "target_pairs": [_pair_data(pair) for pair in target],
        }
        for sample, target, control, raw in zip(
            evaluation,
            evaluation_targets,
            control_outputs,
            raw_outputs,
            strict=True,
        )
    ]
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
            "benchmark_source_sha256": "sha256:" + source_sha256(),
            "connected_fixture_source_sha256": "sha256:" + connected.source_sha256(),
            "ordered_graph_source_sha256": "sha256:" + ordered.source_sha256(),
            "pooled_control_dependency_source_addresses": (
                pooled.dependency_source_addresses()
            ),
            "pooled_control_source_sha256": "sha256:" + pooled.source_sha256(),
            "raw_synthesizer_source_sha256": "sha256:" + synthesizer.source_sha256(),
        },
        "claim_scope": (
            "locally_generated_connected_catalog_exact_cover_engineering_only"
        ),
        "catalog_dependency_audit": catalog_dependency,
        "control": {
            "algorithm_id": CONTROL_ID,
            "deduplicated_fit_row_count": len(unique_training),
            "feature_count": len(pooled.FEATURE_NAMES),
            "target_set_class_count": len(control_class_targets),
            "target_set_classes": [
                [_pair_data(pair) for pair in target]
                for target in control_class_targets
            ],
            "parameters": dict(CONTROL_PARAMETERS),
        },
        "corpus_coverage": single_shape_coverage,
        "d4_cross_role_audit": d4_audit,
        "evaluation": {
            "carrier_families": sorted({_family(sample) for sample in evaluation}),
            "nuisances": sorted({_nuisance_id(sample) for sample in evaluation}),
            "row_count": len(evaluation),
            "unique_png_count": len({_png_digest(sample) for sample in evaluation}),
        },
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "gates": {**gates, "passed": all(gates.values())},
        "limitations": {
            "carrier_split_tests_unseen_catalog_induction": False,
            "held_out_family_geometry_present_in_raw_catalog": True,
            "held_out_reconstructible_after_removing_held_family_masks": (
                catalog_dependency["held_out_exact_cover_count"] > 0
            ),
            "official_transfer_tested": False,
            "raw_and_target_share_fixed_primitive_catalog": True,
        },
        "matched_pooled_feature_counterfactuals": matched,
        "metrics": {
            "boundary_kind_membership": {
                "empty_boundary_truth_row_count": sum(
                    not _boundary_kinds(sample) for sample in evaluation
                ),
                "empty_rows_excluded_from_kind_groups": True,
                "groups_are_nonexclusive_presence_strata": True,
            },
            "by_boundary_kind": by_boundary,
            "by_carrier_family": by_family,
            "by_layout": by_layout,
            "control": control_metrics,
            "raw_minus_control_exact_accuracy": overall_delta,
            "raw_synthesizer": raw_metrics,
        },
        "paired_rows": paired_rows,
        "raw_synthesizer": {
            "algorithm_id": SYNTHESIZER_ID,
            "candidate_construction_uses_target": False,
            "exact_catalog_reconstruction_required": True,
            "learning_used_for_candidate_construction": False,
        },
        "runtime": pooled.runtime_fingerprint(),
        "schema": SCHEMA,
        "target": {
            "algorithm_id": "connected-exhaustive-exact-cover-target/v1",
            "ambiguous_count_pair_targets_retained": True,
            "evaluation_targets_constructed_after_raw_predictions": True,
            "generator_history_used_for_scoring": False,
            "target_passed_to_raw_prediction": False,
        },
        "training": {
            "carrier_families": sorted({_family(sample) for sample in training}),
            "nuisances": sorted({_nuisance_id(sample) for sample in training}),
            "row_count": len(training),
            "unique_png_count": len({_png_digest(sample) for sample in training}),
        },
    }
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


__all__ = (
    "CONTROL_ID",
    "EVALUATION_FAMILIES",
    "GATE_THRESHOLDS",
    "SCHEMA",
    "SYNTHESIZER_ID",
    "ConnectedBenchmarkError",
    "raw_synthesizer_outputs",
    "run_connected_benchmark",
    "source_sha256",
)
