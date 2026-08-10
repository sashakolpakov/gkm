"""Frozen development-only forensics for the failed tiny local observer.

This audit is intentionally downstream of the archived failed fit.  It may
re-read only the already-exposed, decontaminated development pixels and their
already-exposed action programs.  It has no path or API for calibration,
evaluation, same-family, target, support, or query cohorts.

The output separates four questions which the original aggregate result did
not answer:

* whether the nine query attentions actually separated image regions;
* whether exact joint-DP decoding helped relative to direct slot argmax;
* whether count-pair and catalog imbalance dominated the objectives; and
* whether checkpoint selection, rather than the representation, caused the
  failed development gate.

The module diagnoses the archived checkpoint.  It does not modify the v1
trainer/core and does not authorize production inference from a failed fit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bongard.canonical import canonical_digest, canonical_json
from bongard import panel_action_count_tiny_local_dev_command as core
from bongard import panel_action_count_tiny_local_train_command as trainer
from bongard import panel_action_local_supervision_authority as authority_module


SCHEMA = "gkm.bongard-tiny-local-action-failure-forensics.v1"
RESULT_DIGEST = (
    "sha256:48e0e6404ba0f070712abad25e0219ceea7482d396782a21d07b7e898734b824"
)
REPLAY_DIGEST = (
    "sha256:1b0edd12ec8467e96baae672f041dea4b96c6eab1e30f6df09c7ff86929d5710"
)
TRAINING_PRECOMMIT_DIGEST = (
    "sha256:f23f7217b23614d74cf25a972546160fbb9635a94808de3ffa594e188e56160d"
)
CHECKPOINT_RAW_SHA256 = (
    "sha256:6f8934122e25b271a8539388e2e47413905c5536e2b0c0c6af3f46cf6bd3c8d5"
)
CHECKPOINT_STATE_SHA256 = (
    "sha256:e6a7bdd9b577218d599e32661b2a1248c154ea9060226cfc79f9b9fbc136a943"
)
PREDICTION_ROWS_DIGEST = (
    "sha256:011c9c1a8922c013adad2e54fa6bc9673eb8d8a04d7f47a2e342bbc0f61e1565"
)
RUN_RELATIVE = Path(
    "downloads/ShapeBongard_V2_full/panel_action_count_tiny_local_20260810_v1"
)
DATASET_RELATIVE = Path("downloads/ShapeBongard_V2_full/ShapeBongard_V2")
FIT_PRECOMMIT_RELATIVE = Path(
    "downloads/ShapeBongard_V2_full/panel_action_count_cnn_fit_20260810_v3/"
    "fit_pixel_precommit.json"
)
OUTPUT_RELATIVE = Path(
    "bongard/data/panel_action_count_tiny_local_failure_forensics_20260810_v1.json"
)


class TinyFailureForensicsError(RuntimeError):
    """A frozen input, scope, checkpoint, or forensic invariant differs."""


def source_sha256() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def _load_record(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TinyFailureForensicsError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise TinyFailureForensicsError(f"{label} is not canonical JSON plus newline")
    body = dict(value)
    found = body.pop("record_digest", None)
    if found != "sha256:" + canonical_digest(body):
        raise TinyFailureForensicsError(f"{label} record digest differs")
    return value


def _assert_digest(value: Mapping[str, Any], expected: str, *, label: str) -> None:
    if value.get("record_digest") != expected:
        raise TinyFailureForensicsError(f"{label} is not the frozen failed-fit input")


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise TinyFailureForensicsError("forensic statistic population differs")
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _confusion(
    true_values: Sequence[int], predicted_values: Sequence[int], *, size: int
) -> list[list[int]]:
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for truth, prediction in zip(true_values, predicted_values):
        if truth not in range(size) or prediction not in range(size):
            raise TinyFailureForensicsError("confusion value leaves frozen domain")
        matrix[truth][prediction] += 1
    return matrix


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise TinyFailureForensicsError("metric stratum is empty")
    count = len(rows)
    recalls: dict[str, float | None] = {}
    for catalog_class in range(3):
        selected = [row for row in rows if row["true_catalog"] == catalog_class]
        recalls[str(catalog_class)] = (
            sum(row["predicted_catalog"] == catalog_class for row in selected)
            / len(selected)
            if selected
            else None
        )
    known_balanced = (
        (float(recalls["1"]) + float(recalls["2"])) / 2.0
        if recalls["1"] is not None and recalls["2"] is not None
        else None
    )
    return {
        "arc_dp_top1": sum(row["predicted_arc"] == row["true_arc"] for row in rows)
        / count,
        "arc_slot_argmax_top1": sum(
            row["slot_arc"] == row["true_arc"] for row in rows
        )
        / count,
        "catalog_all_class_top1": sum(
            row["predicted_catalog"] == row["true_catalog"] for row in rows
        )
        / count,
        "catalog_known_balanced_accuracy": known_balanced,
        "catalog_recall_unresolved_nonconvex_convex": recalls,
        "panel_count": count,
        "pair_dp_top1": sum(
            (row["predicted_straight"], row["predicted_arc"])
            == (row["true_straight"], row["true_arc"])
            for row in rows
        )
        / count,
        "pair_slot_argmax_top1": sum(
            (row["slot_straight"], row["slot_arc"])
            == (row["true_straight"], row["true_arc"])
            for row in rows
        )
        / count,
        "straight_dp_top1": sum(
            row["predicted_straight"] == row["true_straight"] for row in rows
        )
        / count,
        "straight_slot_argmax_top1": sum(
            row["slot_straight"] == row["true_straight"] for row in rows
        )
        / count,
    }


def _compact_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value = _metrics(rows)
    return {
        key: value[key]
        for key in (
            "panel_count",
            "straight_dp_top1",
            "arc_dp_top1",
            "pair_dp_top1",
            "catalog_known_balanced_accuracy",
        )
    }


def _histogram(values: Sequence[int]) -> dict[str, int]:
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _pair_histogram(rows: Sequence[Mapping[str, Any]], fields: tuple[str, str]) -> dict[str, int]:
    counts = Counter((int(row[fields[0]]), int(row[fields[1]])) for row in rows)
    return {f"{straight},{arc}": count for (straight, arc), count in sorted(counts.items())}


def _panel_program_metadata(authority, panel_id: str) -> tuple[int, tuple[str, ...]]:
    parts = panel_id.split("/")
    if len(parts) != 4 or parts[0] != "hd" or parts[2] not in {"0", "1"}:
        raise TinyFailureForensicsError("development panel ID differs")
    program = authority.program_for(parts[1])
    image = program[0 if parts[2] == "1" else 1][int(parts[3][:-4])]
    if not isinstance(image, list) or not image:
        raise TinyFailureForensicsError("development action program differs")
    styles: set[str] = set()
    for shape in image:
        for action in shape:
            if not isinstance(action, str):
                raise TinyFailureForensicsError("development action token differs")
            fields = action.split("_")
            if len(fields) < 3 or fields[1] not in authority_module.KNOWN_STYLES:
                raise TinyFailureForensicsError("development style token differs")
            styles.add(fields[1])
    return len(image), tuple(sorted(styles))


def _loss_gradient_diagnostics(model, validation) -> dict[str, Any]:
    """Compare frozen unweighted losses to fixed global inverse-frequency losses."""

    torch, _, functional = core._torch_runtime()
    panel_count = len(validation)
    batch_size = int(core.PROTOCOL["batch_size"])
    count_frequencies = Counter((group.straight, group.arc) for group in validation)
    catalog_frequencies = Counter(group.catalog for group in validation)
    count_class_count = len(count_frequencies)
    catalog_class_count = len(catalog_frequencies)
    result: dict[str, Any] = {}
    objectives = (
        "count_unweighted",
        "count_globally_class_balanced",
        "catalog_unweighted",
        "catalog_globally_class_balanced",
        "descriptor_classification",
        "descriptor_geometry",
    )
    selected_names = {
        "queries",
        "stem.0.weight",
        "slot_class.weight",
        "slot_class.bias",
        "catalog.weight",
        "catalog.bias",
    }
    for objective in objectives:
        model.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for start in range(0, panel_count, batch_size):
            indices = list(range(start, min(panel_count, start + batch_size)))
            pixels, selected = trainer._batch(
                validation, indices, epoch=0, augment=False
            )
            output = model(pixels)
            if objective.startswith("count"):
                joint = core.joint_count_probabilities(
                    output["slot_logits"].softmax(dim=-1)
                )
                losses = -torch.stack(
                    [
                        joint[index, group.straight, group.arc]
                        .clamp_min(1e-12)
                        .log()
                        for index, group in enumerate(selected)
                    ]
                )
                if objective.endswith("balanced"):
                    weights = losses.new_tensor(
                        [
                            panel_count
                            / (
                                count_class_count
                                * count_frequencies[(group.straight, group.arc)]
                            )
                            for group in selected
                        ]
                    )
                    loss = (losses * weights).mean()
                else:
                    loss = losses.mean()
            elif objective.startswith("catalog"):
                targets = torch.tensor(
                    [group.catalog for group in selected], dtype=torch.long
                )
                losses = functional.cross_entropy(
                    output["catalog_logits"], targets, reduction="none"
                )
                if objective.endswith("balanced"):
                    weights = losses.new_tensor(
                        [
                            panel_count
                            / (catalog_class_count * catalog_frequencies[group.catalog])
                            for group in selected
                        ]
                    )
                    loss = (losses * weights).mean()
                else:
                    loss = losses.mean()
            else:
                components = trainer.group_normalized_loss(output, selected)
                component_name = {
                    "descriptor_classification": (
                        "descriptor_classification_group_normalized"
                    ),
                    "descriptor_geometry": "descriptor_geometry_group_normalized",
                }[objective]
                loss = components[component_name]
            batch_fraction = len(selected) / panel_count
            (loss * batch_fraction).backward()
            mean_loss += float(loss.detach()) * batch_fraction
        total_squared = 0.0
        selected_gradients: dict[str, float] = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            norm = float(parameter.grad.detach().norm())
            total_squared += norm * norm
            if name in selected_names:
                selected_gradients[name] = norm
        result[objective] = {
            "mean_loss": mean_loss,
            "parameter_gradient_l2": math.sqrt(total_squared),
            "selected_parameter_gradient_l2": selected_gradients,
        }
    model.zero_grad(set_to_none=True)
    return result


def _epoch_counterfactual(result: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = {
        "arc_top1": 0.80,
        "known_catalog_binary_balanced_accuracy": 0.65,
        "straight_top1": 0.65,
    }
    rows = []
    for archived in result["history"]:
        metrics = {name: float(archived[name]) for name in thresholds}
        rows.append(
            {
                "epoch": archived["epoch"],
                "gate_metrics": metrics,
                "mean_gate_metric": sum(metrics.values()) / len(metrics),
                "minimum_threshold_fraction": min(
                    metrics[name] / threshold for name, threshold in thresholds.items()
                ),
                "passes_all": all(
                    metrics[name] >= threshold for name, threshold in thresholds.items()
                ),
            }
        )
    return {
        "archived_states_available": [result["selected_epoch"]],
        "any_epoch_passed_gate": any(row["passes_all"] for row in rows),
        "best_epoch_by_arc": max(rows, key=lambda row: row["gate_metrics"]["arc_top1"])[
            "epoch"
        ],
        "best_epoch_by_catalog_known_balanced_accuracy": max(
            rows,
            key=lambda row: row["gate_metrics"][
                "known_catalog_binary_balanced_accuracy"
            ],
        )["epoch"],
        "best_epoch_by_mean_gate_metric": max(
            rows, key=lambda row: row["mean_gate_metric"]
        )["epoch"],
        "best_epoch_by_minimum_threshold_fraction": max(
            rows, key=lambda row: row["minimum_threshold_fraction"]
        )["epoch"],
        "best_epoch_by_straight": max(
            rows, key=lambda row: row["gate_metrics"]["straight_top1"]
        )["epoch"],
        "frozen_v1_selection_order": [
            "straight_top1",
            "known_catalog_binary_balanced_accuracy",
            "descriptor_primitive_multiset_exact",
            "descriptor_matched_primitive_accuracy",
            "descriptor_geometry_interval_hit",
            "arc_top1",
            "earlier_epoch",
        ],
        "rows": rows,
        "selected_epoch": result["selected_epoch"],
    }


def build_failure_forensics(*, repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    run = root / RUN_RELATIVE
    dataset = root / DATASET_RELATIVE
    result = _load_record(run / "result.json", label="tiny failed result")
    replay = _load_record(run / "replay.json", label="tiny failed replay")
    precommit = _load_record(
        run / "training_precommit.json", label="tiny training precommit"
    )
    authorization = _load_record(run / "authorization.json", label="tiny authorization")
    fit_precommit = _load_record(
        root / FIT_PRECOMMIT_RELATIVE, label="development fit precommit"
    )
    _assert_digest(result, RESULT_DIGEST, label="tiny failed result")
    _assert_digest(replay, REPLAY_DIGEST, label="tiny failed replay")
    _assert_digest(
        precommit, TRAINING_PRECOMMIT_DIGEST, label="tiny training precommit"
    )
    if (
        result.get("checkpoint_raw_sha256") != CHECKPOINT_RAW_SHA256
        or result.get("checkpoint_state_dict_sha256") != CHECKPOINT_STATE_SHA256
        or result.get("validation_prediction_rows_digest") != PREDICTION_ROWS_DIGEST
        or replay.get("predictions_exact") is not True
        or replay.get("metrics_exact") is not True
        or Path(authorization.get("dataset_root", "")).resolve() != dataset
        or Path(authorization.get("fit_precommit_path", "")).resolve()
        != root / FIT_PRECOMMIT_RELATIVE
    ):
        raise TinyFailureForensicsError("failed result/replay custody differs")

    deadline = core.WallDeadline(seconds=600.0)
    groups = trainer.materialize_groups(
        repository_root=root,
        dataset_root=dataset,
        fit_precommit=fit_precommit,
        conflict_audit=precommit["descriptor_target_conflict_audit"],
        deadline=deadline,
    )
    training = tuple(group for group in groups if group.cohort == "train")
    validation = tuple(group for group in groups if group.cohort == "validation")
    if (
        sum(group.multiplicity for group in training) != 11_200
        or sum(group.multiplicity for group in validation) != 1_392
        or any(group.multiplicity != 1 for group in validation)
    ):
        raise TinyFailureForensicsError("decontaminated development population differs")
    model, checkpoint, checkpoint_raw = core.load_verified_checkpoint(
        run / "model.pt",
        expected_training_precommit_record_digest=precommit["record_digest"],
        training_result=result,
        expected_training_result_record_digest=result["record_digest"],
        require_passed_development_gate=False,
    )
    if checkpoint_raw != CHECKPOINT_RAW_SHA256:
        raise TinyFailureForensicsError("failed checkpoint bytes differ")
    model.eval()
    torch, _, functional = core._torch_runtime()

    inventory = {
        row["png_sha256"]: row
        for row in fit_precommit["path_independent_digest_groups"]
        if row["fit_cohort"] == "validation"
    }
    authority = authority_module.load_development_authority(repository_root=root)
    rows: list[dict[str, Any]] = []
    attention_cosines: list[float] = []
    attention_top8_ious: list[float] = []
    attention_entropies: list[float] = []
    attention_maxima: list[float] = []
    attention_ink_mass: list[float] = []
    attention_top8_union_size: list[float] = []
    attention_top8_union_ink_coverage: list[float] = []
    mean_slot_probabilities = np.zeros(3, dtype=np.float64)
    slot_argmax_class_counts = np.zeros(3, dtype=np.int64)
    batch_size = int(core.PROTOCOL["batch_size"])
    for start in range(0, len(validation), batch_size):
        selected = validation[start : start + batch_size]
        pixels, _ = trainer._batch(
            selected, list(range(len(selected))), epoch=0, augment=False
        )
        with torch.no_grad():
            output = model(pixels)
            slot_probabilities = output["slot_logits"].softmax(dim=-1)
            joint = core.joint_count_probabilities(slot_probabilities)
            catalog_probabilities = output["catalog_logits"].softmax(dim=-1)
            ink_tokens = functional.max_pool2d(
                pixels[:, 1:2], kernel_size=8, stride=8
            ).flatten(1) > 0
        for index, group in enumerate(selected):
            flat_index = int(joint[index].argmax().item())
            predicted_straight, predicted_arc = divmod(flat_index, 10)
            classes = slot_probabilities[index].argmax(dim=-1).cpu().numpy()
            slot_straight = int((classes == 1).sum())
            slot_arc = int((classes == 2).sum())
            predicted_catalog = int(catalog_probabilities[index].argmax().item())
            attention = (
                output["attention"][index]
                .detach()
                .cpu()
                .numpy()
                .reshape(core.MAX_ACTION_SLOTS, 64)
                .astype(np.float64)
            )
            norms = np.linalg.norm(attention, axis=1)
            cosine = (attention @ attention.T) / (norms[:, None] * norms[None, :])
            top_sets = []
            for query in range(core.MAX_ACTION_SLOTS):
                top_sets.append(set(np.argpartition(attention[query], -8)[-8:].tolist()))
                for other in range(query + 1, core.MAX_ACTION_SLOTS):
                    attention_cosines.append(float(cosine[query, other]))
            for query in range(core.MAX_ACTION_SLOTS):
                for other in range(query + 1, core.MAX_ACTION_SLOTS):
                    union = top_sets[query] | top_sets[other]
                    attention_top8_ious.append(
                        len(top_sets[query] & top_sets[other]) / len(union)
                    )
            entropy = -(
                attention * np.log(np.maximum(attention, 1e-30))
            ).sum(axis=1) / math.log(64)
            attention_entropies.extend(float(value) for value in entropy)
            attention_maxima.extend(float(value) for value in attention.max(axis=1))
            ink = ink_tokens[index].cpu().numpy().astype(bool)
            attention_ink_mass.extend(
                float(value) for value in attention[:, ink].sum(axis=1)
            )
            top_union = set().union(*top_sets)
            ink_indices = set(np.flatnonzero(ink).tolist())
            attention_top8_union_size.append(float(len(top_union)))
            attention_top8_union_ink_coverage.append(
                len(top_union & ink_indices) / len(ink_indices)
            )
            mean_slot_probabilities += slot_probabilities[index].sum(dim=0).cpu().numpy()
            slot_argmax_class_counts += np.bincount(classes, minlength=3)

            group_row = inventory.get(group.png_sha256)
            if not isinstance(group_row, Mapping):
                raise TinyFailureForensicsError("validation digest inventory differs")
            panel_ids = group_row.get("panel_ids")
            if not isinstance(panel_ids, list) or len(panel_ids) != 1:
                raise TinyFailureForensicsError("validation occurrence identity differs")
            panel_id = panel_ids[0]
            shape_count, styles = _panel_program_metadata(authority, panel_id)
            rows.append(
                {
                    "action_count": group.straight + group.arc,
                    "all_normal": styles == ("normal",),
                    "panel_id": panel_id,
                    "predicted_arc": predicted_arc,
                    "predicted_catalog": predicted_catalog,
                    "predicted_straight": predicted_straight,
                    "catalog_probabilities": tuple(
                        float(value)
                        for value in catalog_probabilities[index].cpu().tolist()
                    ),
                    "shape_count": shape_count,
                    "slot_arc": slot_arc,
                    "slot_straight": slot_straight,
                    "styles": styles,
                    "true_arc": group.arc,
                    "true_catalog": group.catalog,
                    "true_straight": group.straight,
                    "truth_joint_probability": float(
                        joint[index, group.straight, group.arc].item()
                    ),
                }
            )
        deadline.check()

    if len(rows) != 1_392:
        raise TinyFailureForensicsError("validation inference population differs")
    overall = _metrics(rows)
    overall.update(
        {
            "dp_slot_pair_disagreement": sum(
                (row["predicted_straight"], row["predicted_arc"])
                != (row["slot_straight"], row["slot_arc"])
                for row in rows
            )
            / len(rows),
            "mean_truth_joint_probability": float(
                np.mean([row["truth_joint_probability"] for row in rows])
            ),
        }
    )

    train_pair_counts = Counter(
        (group.straight, group.arc) for group in training for _ in range(group.multiplicity)
    )
    # Above expansion is bounded at exactly 11,200 values and makes the
    # occurrence-weighted semantics explicit.
    train_straight_counts = Counter()
    train_arc_counts = Counter()
    train_catalog_counts = Counter()
    train_descriptor_slot_counts = Counter()
    for group in training:
        train_straight_counts[group.straight] += group.multiplicity
        train_arc_counts[group.arc] += group.multiplicity
        train_catalog_counts[group.catalog] += group.multiplicity
        train_descriptor_slot_counts[0] += 9 - group.straight - group.arc
        train_descriptor_slot_counts[1] += group.straight
        train_descriptor_slot_counts[2] += group.arc
    mode_straight = max(train_straight_counts, key=train_straight_counts.get)
    mode_arc = max(train_arc_counts, key=train_arc_counts.get)
    mode_pair = max(train_pair_counts, key=train_pair_counts.get)
    mode_catalog = max(train_catalog_counts, key=train_catalog_counts.get)
    majority_baselines = {
        "arc_train_mode": mode_arc,
        "arc_validation_top1": sum(row["true_arc"] == mode_arc for row in rows)
        / len(rows),
        "catalog_all_class_validation_top1": sum(
            row["true_catalog"] == mode_catalog for row in rows
        )
        / len(rows),
        "catalog_known_balanced_accuracy": 0.0,
        "catalog_train_mode": mode_catalog,
        "joint_pair_train_mode": list(mode_pair),
        "joint_pair_validation_top1": sum(
            (row["true_straight"], row["true_arc"]) == mode_pair for row in rows
        )
        / len(rows),
        "straight_train_mode": mode_straight,
        "straight_validation_top1": sum(
            row["true_straight"] == mode_straight for row in rows
        )
        / len(rows),
    }

    by_action_count: dict[str, Any] = {}
    for action_count in sorted({row["action_count"] for row in rows}):
        by_action_count[str(action_count)] = _compact_metrics(
            [row for row in rows if row["action_count"] == action_count]
        )
    decoration = {
        "all_normal": _compact_metrics([row for row in rows if row["all_normal"]]),
        "any_decorated": _compact_metrics(
            [row for row in rows if not row["all_normal"]]
        ),
    }
    style_presence = {
        style: _compact_metrics([row for row in rows if style in row["styles"]])
        for style in sorted(authority_module.KNOWN_STYLES)
    }
    shape_strata = {
        str(shape_count): _compact_metrics(
            [row for row in rows if row["shape_count"] == shape_count]
        )
        for shape_count in sorted({row["shape_count"] for row in rows})
    }
    authority_shape_counts = {}
    for cohort, panel_ids in authority.cohort_panel_ids:
        counts = Counter(
            _panel_program_metadata(authority, panel_id)[0] for panel_id in panel_ids
        )
        authority_shape_counts[cohort] = {
            str(key): count for key, count in sorted(counts.items())
        }

    catalog_logit_gradient_contributions = {}
    # d(CE)/d(logit) = softmax - one_hot.  Re-run only the compact catalog
    # head outputs already bound above, so class population effects are visible.
    for catalog_class in range(3):
        contribution = np.zeros(3, dtype=np.float64)
        count = 0
        for row in rows:
            if row["true_catalog"] != catalog_class:
                continue
            value = np.asarray(row["catalog_probabilities"], dtype=np.float64)
            value[catalog_class] -= 1.0
            contribution += value / len(validation)
            count += 1
        catalog_logit_gradient_contributions[str(catalog_class)] = {
            "dataset_mean_gradient_contribution": contribution.tolist(),
            "l2": float(np.linalg.norm(contribution)),
            "panel_count": count,
        }

    gradients = _loss_gradient_diagnostics(model, validation)
    deadline.check()
    count_min = min(train_pair_counts.values())
    count_max = max(train_pair_counts.values())
    body = {
        "architecture_audit": {
            "attention_diversity_or_coverage_loss_present": False,
            "explicit_token_coordinates_present": False,
            "feature_token_grid": [8, 8],
            "maximum_action_slots": 9,
            "parameter_count": core.parameter_count(model),
            "training_and_validation_authority_shape_count_histograms": authority_shape_counts,
            "two_shape_empirical_development_support": False,
        },
        "attention_diagnostics": {
            "interpretation": (
                "high entropy plus high pairwise cosine certifies diffuse overlapping "
                "queries, not separated action instances"
            ),
            "map_count": len(rows) * core.MAX_ACTION_SLOTS,
            "maximum_token_probability": _quantiles(attention_maxima),
            "normalized_entropy_uniform_equals_one": _quantiles(attention_entropies),
            "pair_count": len(attention_cosines),
            "pairwise_cosine": _quantiles(attention_cosines),
            "top8_pairwise_iou": _quantiles(attention_top8_ious),
            "top8_union_ink_token_coverage": _quantiles(
                attention_top8_union_ink_coverage
            ),
            "top8_union_token_count": _quantiles(attention_top8_union_size),
            "attention_mass_on_maxpooled_ink_tokens": _quantiles(attention_ink_mass),
            "uniform_token_probability": 1.0 / 64.0,
        },
        "bindings": {
            "authorization_record_digest": authorization["record_digest"],
            "checkpoint_raw_sha256": CHECKPOINT_RAW_SHA256,
            "checkpoint_state_dict_sha256": CHECKPOINT_STATE_SHA256,
            "config_digest": result["config_digest"],
            "core_source_sha256": checkpoint["source_sha256"],
            "development_fit_precommit_record_digest": fit_precommit["record_digest"],
            "replay_record_digest": REPLAY_DIGEST,
            "result_record_digest": RESULT_DIGEST,
            "supervision_authority_record_digest": authority.record_digest,
            "trainer_source_sha256": result["source_sha256"],
            "training_precommit_record_digest": TRAINING_PRECOMMIT_DIGEST,
            "validation_prediction_rows_digest": PREDICTION_ROWS_DIGEST,
        },
        "confusion_matrices": {
            "arc_dp_rows_true_columns_predicted_0_through_9": _confusion(
                [row["true_arc"] for row in rows],
                [row["predicted_arc"] for row in rows],
                size=10,
            ),
            "arc_slot_argmax_rows_true_columns_predicted_0_through_9": _confusion(
                [row["true_arc"] for row in rows],
                [row["slot_arc"] for row in rows],
                size=10,
            ),
            "catalog_rows_true_columns_predicted_unresolved_nonconvex_convex": _confusion(
                [row["true_catalog"] for row in rows],
                [row["predicted_catalog"] for row in rows],
                size=3,
            ),
            "straight_dp_rows_true_columns_predicted_0_through_9": _confusion(
                [row["true_straight"] for row in rows],
                [row["predicted_straight"] for row in rows],
                size=10,
            ),
            "straight_slot_argmax_rows_true_columns_predicted_0_through_9": _confusion(
                [row["true_straight"] for row in rows],
                [row["slot_straight"] for row in rows],
                size=10,
            ),
        },
        "decoding_diagnostics": {
            "conclusion": (
                "exact joint DP is internally correct but is not a rescue; direct "
                "slot argmax has higher arc and joint-pair accuracy"
            ),
            "overall": overall,
            "prediction_marginals": {
                "arc_dp": _histogram([row["predicted_arc"] for row in rows]),
                "arc_slot_argmax": _histogram([row["slot_arc"] for row in rows]),
                "catalog": _histogram([row["predicted_catalog"] for row in rows]),
                "joint_dp": _pair_histogram(
                    rows, ("predicted_straight", "predicted_arc")
                ),
                "joint_slot_argmax": _pair_histogram(
                    rows, ("slot_straight", "slot_arc")
                ),
                "mean_slot_probabilities_none_line_arc": (
                    mean_slot_probabilities / (len(rows) * core.MAX_ACTION_SLOTS)
                ).tolist(),
                "slot_argmax_class_counts_none_line_arc": (
                    slot_argmax_class_counts.tolist()
                ),
                "straight_dp": _histogram(
                    [row["predicted_straight"] for row in rows]
                ),
                "straight_slot_argmax": _histogram(
                    [row["slot_straight"] for row in rows]
                ),
            },
        },
        "epoch_selection_counterfactual": _epoch_counterfactual(result),
        "imbalance_and_gradient_diagnostics": {
            "catalog_dataset_mean_logit_gradient_contribution_by_true_class": (
                catalog_logit_gradient_contributions
            ),
            "conclusion": (
                "unweighted objectives preserve the dominant unresolved catalog "
                "class and common count pairs; geometry is numerically inert"
            ),
            "loss_component_gradients_at_selected_checkpoint": gradients,
            "train_catalog_occurrence_histogram_unresolved_nonconvex_convex": {
                str(key): count for key, count in sorted(train_catalog_counts.items())
            },
            "train_descriptor_slot_target_histogram_none_line_arc": {
                str(key): count
                for key, count in sorted(train_descriptor_slot_counts.items())
            },
            "train_joint_count_pair_imbalance": {
                "largest_class_occurrences": count_max,
                "largest_to_smallest_ratio": count_max / count_min,
                "occupied_pair_count": len(train_pair_counts),
                "smallest_class_occurrences": count_min,
            },
            "train_joint_count_pair_occurrence_histogram": {
                f"{straight},{arc}": count
                for (straight, arc), count in sorted(train_pair_counts.items())
            },
        },
        "majority_baselines": majority_baselines,
        "next_frozen_development_experiment": {
            "authorization": (
                "one preregistered rerun on the identical already-exposed "
                "development split; no later cohort may be opened"
            ),
            "nonselection_reason": (
                "multiscale skeleton features produced stronger frozen-development "
                "count evidence; the immediate successor is the skeleton graph"
            ),
            "selection_status": "NOT_SELECTED",
            "changes": {
                "attention_auxiliary": {
                    "active_weight": "one_minus_slot_none_probability",
                    "entropy_weight": 0.02,
                    "ink_coverage_weight": 0.10,
                    "pairwise_cosine_overlap_weight": 0.10,
                    "rule": (
                        "active-weighted attention entropy plus pairwise cosine "
                        "overlap; mean active attention L1-matched to normalized "
                        "max-pooled input-ink support"
                    ),
                },
                "catalog_loss": (
                    "fixed global inverse-training-frequency weights N/(3*n_class)"
                ),
                "checkpoint_selection": (
                    "maximize minimum frozen-gate threshold fraction; then mean "
                    "threshold fraction; then earliest epoch"
                ),
                "descriptor_geometry": (
                    "remove from optimization; retain interval-hit as diagnostic_only"
                ),
                "feature_grid": [16, 16],
                "joint_count_loss": (
                    "fixed global inverse-training-frequency weights "
                    "N/(K*n_pair) over K occupied pairs"
                ),
                "position": (
                    "append D4-regenerated normalized x/y coordinates to every "
                    "feature token before key/value projection"
                ),
            },
            "frozen_limits": {
                "development_gate_unchanged": True,
                "maximum_parameter_count": 20_000,
                "maximum_wall_runtime_seconds": 600.0,
                "one_fit_only": True,
            },
            "multi_shape_disposition": (
                "GAP until a separately authorized development cohort contains "
                "multi-shape panels; current train and validation contain none"
            ),
            "not_changed": [
                "exact joint DP implementation",
                "nine-slot capacity",
                "pose-free action authority",
                "development gate thresholds",
            ],
        },
        "schema": SCHEMA,
        "scope": {
            "already_exposed_development_pixel_occurrences_reread": 12_592,
            "calibration_evaluation_family_target_query_identifiers_opened": 0,
            "failed_checkpoint_diagnostic_only": True,
            "fresh_cohort_pixels_opened": 0,
            "validation_digest_groups_inferred": 1_392,
        },
        "source_sha256": source_sha256(),
        "stratified_metrics": {
            "action_count": by_action_count,
            "decoration": decoration,
            "shape_count": shape_strata,
            "style_presence": style_presence,
        },
    }
    return _seal(body)


def verify_failure_forensics(
    artifact: Mapping[str, Any], *, repository_root: Path
) -> None:
    if not isinstance(artifact, Mapping):
        raise TinyFailureForensicsError("forensics artifact is not an object")
    body = dict(artifact)
    found = body.pop("record_digest", None)
    if (
        artifact.get("schema") != SCHEMA
        or found != "sha256:" + canonical_digest(body)
        or artifact.get("source_sha256") != source_sha256()
        or artifact.get("bindings", {}).get("result_record_digest") != RESULT_DIGEST
        or artifact.get("bindings", {}).get("replay_record_digest") != REPLAY_DIGEST
        or artifact.get("bindings", {}).get("checkpoint_raw_sha256")
        != CHECKPOINT_RAW_SHA256
        or artifact.get("scope", {}).get(
            "calibration_evaluation_family_target_query_identifiers_opened"
        )
        != 0
        or artifact.get("scope", {}).get("fresh_cohort_pixels_opened") != 0
    ):
        raise TinyFailureForensicsError("forensics artifact binding/scope differs")
    expected_path = repository_root.resolve() / OUTPUT_RELATIVE
    if expected_path.exists() and expected_path.read_bytes() != canonical_json(artifact) + b"\n":
        raise TinyFailureForensicsError("committed forensics bytes differ")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.repository_root.resolve()
    expected_output = root / OUTPUT_RELATIVE
    output = expected_output if arguments.output is None else arguments.output.resolve()
    if output != expected_output:
        raise TinyFailureForensicsError("forensics output must use the frozen path")
    artifact = build_failure_forensics(repository_root=root)
    core._write_once(output, artifact)
    verify_failure_forensics(artifact, repository_root=root)
    print(artifact["record_digest"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_RAW_SHA256",
    "OUTPUT_RELATIVE",
    "REPLAY_DIGEST",
    "RESULT_DIGEST",
    "SCHEMA",
    "TinyFailureForensicsError",
    "build_failure_forensics",
    "main",
    "source_sha256",
    "verify_failure_forensics",
]
