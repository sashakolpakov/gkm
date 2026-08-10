"""Development-only spatial successor for the failed action-count CNN.

This command can reread only the already-exposed, decontaminated FIT cohort.
It has no calibration, evaluation, family, query, or target entry point.  The
successor wraps the audited v2 trainer machinery, but freezes a new pixel
representation and architecture in a durable precommit before rereading any
development PNG.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
from contextlib import contextmanager
from io import BytesIO
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image

from bongard.canonical import canonical_digest, canonical_json
from bongard import panel_action_count_cnn_train_command as base


SCHEMA_PRECOMMIT = "gkm.bongard-action-count-spatial-development-precommit.v1"
SCHEMA_RESULT = "gkm.bongard-action-count-spatial-development-result.v1"
SCHEMA_REPLAY = "gkm.bongard-action-count-spatial-development-replay.v1"
ARCHITECTURE_ID = "coarse-carrier-spatial-residual-three-head/v1"
FAILED_FIT_DIGEST = (
    "sha256:f8b79047228a91fd3fdd47a262299b0cd683daa727981e568450371be4e4dff2"
)
FAILED_REPLAY_DIGEST = (
    "sha256:69802bf42f429aeeca31f62863b576e8620d646db61afa1410073578cb0008dc"
)
FIT_PRECOMMIT_DIGEST = (
    "sha256:e8c7c15fbfb723c5b2305094f035e2567c1fb9b7e80b9f13eeae32fe35d1b15a"
)
FIT_AUTHORIZATION_DIGEST = (
    "sha256:4fd347caba29c41ce1c433319b92efdde7d9857adfa1067cdf83fffec41224ee"
)
BASE_SOURCE_SHA256 = "2706faf07052e580331346ea209c60bc59987366be53f6a729570f0d2cbc9e6a"

PROTOCOL = MappingProxyType(
    {
        "adaptive_development_after_failed_gate": True,
        "batch_size": 64,
        "catalog_branch_inputs": ["raw_ink", "binary_ink", "coarse_carrier"],
        "class_weight": "effective_number_beta_0.999_nonzero_mean_normalized",
        "coarse_carrier": (
            "binary_threshold_10_of_255_then_21x21_mean_then_per-panel_"
            "unit-mass_normalization_times_128"
        ),
        "count_branch_inputs": ["coarse_carrier_only"],
        "cpu_threads": 1,
        "d4_augmentation": True,
        "epochs": 24,
        "image_size": 128,
        "learning_rate": 0.001,
        "model": (
            "separate_raw_and_coarse_residual_encoders_to_8x8;_avg_and_max_"
            "spatial_pyramid_1x1_2x2_4x4;_coarse_mass_4x4;_count_heads_never_"
            "receive_raw_or_binary_features"
        ),
        "count_head": (
            "nine_ordered_threshold_logits_projected_to_ten_class_log-"
            "likelihoods_then_effective-number-weighted_cross-entropy"
        ),
        "optimizer": "AdamW",
        "random_seed": 260810,
        "validation_is_selection_biased_development_not_fresh_evidence": True,
        "weight_decay": 0.0001,
    }
)

STRATUM_THRESHOLDS = MappingProxyType(
    {
        "crossing_task": 0.40,
        "line_decoration:decorated_only": 0.45,
        "line_decoration:mixed_normal_and_decorated": 0.45,
        "line_decoration:normal_only": 0.50,
        "straight_true_count_4": 0.50,
        "thin_task": 0.40,
    }
)


class SpatialDevelopmentError(RuntimeError):
    """The frozen development-only successor or its custody differs."""


def source_sha256() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def successor_config_digest() -> str:
    return "sha256:" + canonical_digest(
        {
            "architecture_id": ARCHITECTURE_ID,
            "protocol": json.loads(canonical_json(dict(PROTOCOL))),
            "source_sha256": source_sha256(),
            "stratum_thresholds": dict(STRATUM_THRESHOLDS),
        }
    )


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpatialDevelopmentError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise SpatialDevelopmentError(f"{label} is not canonical JSON plus newline")
    body = dict(value)
    digest = body.pop("record_digest", None)
    if digest != "sha256:" + canonical_digest(body):
        raise SpatialDevelopmentError(f"{label} digest differs")
    return value


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def _verify_predecessor(
    *, fit_authorization: Path, fit_precommit: Path, failed_fit: Path, failed_replay: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    authorization = _load(fit_authorization, "fit authorization")
    precommit = _load(fit_precommit, "fit precommit")
    result = _load(failed_fit, "failed fit")
    replay = _load(failed_replay, "failed fit replay")
    if authorization.get("record_digest") != FIT_AUTHORIZATION_DIGEST:
        raise SpatialDevelopmentError("fit authorization differs")
    if precommit.get("record_digest") != FIT_PRECOMMIT_DIGEST:
        raise SpatialDevelopmentError("fit precommit differs")
    if result.get("record_digest") != FAILED_FIT_DIGEST:
        raise SpatialDevelopmentError("failed fit result differs")
    if replay.get("record_digest") != FAILED_REPLAY_DIGEST:
        raise SpatialDevelopmentError("failed fit replay differs")
    audit = precommit.get("duplicate_digest_audit")
    if (
        authorization.get("trainer_source_sha256") != BASE_SOURCE_SHA256
        or precommit.get("trainer_source_sha256") != BASE_SOURCE_SHA256
        or not isinstance(audit, dict)
        or audit.get("effective_training_panel_count") != 11_200
        or audit.get("effective_validation_panel_count") != 1_392
        or precommit.get("effective_fit_panel_count") != 12_592
        or precommit.get("fit_panel_count") != 12_600
        or audit.get("validation_removed_due_exact_train_duplicate", {}).get(
            "panel_count"
        )
        != 8
        or precommit.get("validation_decontamination_gate", {}).get("passed")
        is not True
    ):
        raise SpatialDevelopmentError("decontaminated development cohort differs")
    groups = precommit.get("path_independent_digest_groups")
    if not isinstance(groups, list):
        raise SpatialDevelopmentError("fit digest groups are missing")
    train = sum(
        row.get("multiplicity", -1)
        for row in groups
        if isinstance(row, dict) and row.get("fit_cohort") == "train"
    )
    validation = sum(
        row.get("multiplicity", -1)
        for row in groups
        if isinstance(row, dict) and row.get("fit_cohort") == "validation"
    )
    if (train, validation) != (11_200, 1_392):
        raise SpatialDevelopmentError("development group multiplicities differ")
    if result.get("validation_gate", {}).get("passed") is not False:
        raise SpatialDevelopmentError("predecessor is not the frozen failed fit")
    if replay.get("predictions_exact") is not True or replay.get("metrics_exact") is not True:
        raise SpatialDevelopmentError("predecessor replay is not exact")
    return authorization, precommit, result, replay


def create_successor_precommit(
    *,
    fit_authorization: Path,
    fit_precommit: Path,
    failed_fit: Path,
    failed_replay: Path,
    intended_checkpoint: Path,
    intended_inner_result: Path,
    intended_result: Path,
    output: Path,
) -> dict[str, Any]:
    authorization, precommit, result, replay = _verify_predecessor(
        fit_authorization=fit_authorization,
        fit_precommit=fit_precommit,
        failed_fit=failed_fit,
        failed_replay=failed_replay,
    )
    nuisance = decoration_invariance_diagnostic()
    if nuisance["passed"] is not True:
        raise SpatialDevelopmentError("synthetic decoration-invariance launch guard failed")
    body = {
        "architecture_id": ARCHITECTURE_ID,
        "authorized_input": "already_exposed_decontaminated_development_only",
        "base_trainer_source_sha256": BASE_SOURCE_SHA256,
        "config_digest": successor_config_digest(),
        "development_occurrence_counts": {"train": 11_200, "validation": 1_392},
        "decoration_invariance_diagnostic": nuisance,
        "fit_authorization_record_digest": authorization["record_digest"],
        "fit_precommit_record_digest": precommit["record_digest"],
        "failed_fit_record_digest": result["record_digest"],
        "failed_replay_record_digest": replay["record_digest"],
        "forbidden_cohorts": [
            "old_v2_calibration",
            "old_v2_evaluation",
            "fresh_v3_calibration",
            "fresh_v3_evaluation",
            "same_family_calibration",
            "target",
            "query",
        ],
        "intended_outputs": {
            "checkpoint": str(intended_checkpoint.resolve()),
            "inner_result": str(intended_inner_result.resolve()),
            "result": str(intended_result.resolve()),
        },
        "pixels_read_by_precommit": 0,
        "protocol": json.loads(canonical_json(dict(PROTOCOL))),
        "schema": SCHEMA_PRECOMMIT,
        "source_sha256": source_sha256(),
        "stratum_thresholds": dict(STRATUM_THRESHOLDS),
    }
    value = _seal(body)
    base._write_fsynced(output, value)
    if _load(output, "successor precommit") != value:
        raise SpatialDevelopmentError("successor precommit fresh-load differs")
    return value


def preprocess_png_bytes(raw: bytes, *, image_size: int = 128) -> np.ndarray:
    """Return tight-cropped uint8 ink; semantic channels are derived later."""

    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise SpatialDevelopmentError("development input must be one PNG frame")
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
    except SpatialDevelopmentError:
        raise
    except Exception as exc:
        raise SpatialDevelopmentError(f"cannot decode development PNG: {exc}") from exc
    ys, xs = np.nonzero(gray < 250)
    if not len(xs):
        raise SpatialDevelopmentError("development PNG has no ink")
    crop = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = crop.shape
    margin = math.ceil(0.08 * max(height, width))
    side = max(height, width) + 2 * margin
    canvas = np.full((side, side), 255, dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top : top + height, left : left + width] = crop
    resized = Image.fromarray(canvas, mode="L").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    return np.ascontiguousarray(255 - np.asarray(resized, dtype=np.uint8))


def semantic_channels(ink):
    """Derive raw, binary, and large-scale mass-normalized carrier views."""

    torch, _, functional = base._torch_runtime()
    if ink.ndim != 4 or ink.shape[1] != 1:
        raise SpatialDevelopmentError("ink tensor shape differs")
    raw = ink.to(torch.float32) / 255.0
    binary = (raw >= (10.0 / 255.0)).to(torch.float32)
    smooth = functional.avg_pool2d(binary, 21, stride=1, padding=10)
    coarse = 128.0 * smooth / smooth.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    return torch.cat((raw, binary, coarse), dim=1)


def decoration_invariance_diagnostic() -> dict[str, Any]:
    """Run the frozen synthetic nuisance bank used only as a launch guard."""

    torch, _, _ = base._torch_runtime()
    plain = torch.zeros((1, 1, 128, 128), dtype=torch.uint8)
    plain[:, :, 63:65, 14:114] = 255
    squares = plain.clone()
    for center in range(19, 114, 10):
        squares[:, :, 59:69, center - 2 : center + 3] = 255
    dots = torch.zeros_like(plain)
    for center in range(15, 114, 4):
        dots[:, :, 61:67, center : center + 2] = 255
    zigzag = torch.zeros_like(plain)
    for column in range(14, 114):
        row = 63 + ((column - 14) % 8 - 4) // 2
        zigzag[:, :, row : row + 3, column] = 255
    reference = semantic_channels(plain)
    rows = []
    for name, variant in (("square_markers", squares), ("dots", dots), ("zigzag", zigzag)):
        candidate = semantic_channels(variant)
        raw_a = reference[:, 1] / reference[:, 1].sum()
        raw_b = candidate[:, 1] / candidate[:, 1].sum()
        coarse_a = reference[:, 2] / reference[:, 2].sum()
        coarse_b = candidate[:, 2] / candidate[:, 2].sum()
        raw_distance = float((raw_a - raw_b).abs().sum().item())
        coarse_distance = float((coarse_a - coarse_b).abs().sum().item())
        rows.append(
            {
                "coarse_normalized_l1": coarse_distance,
                "coarse_to_raw_ratio": coarse_distance / raw_distance,
                "raw_binary_normalized_l1": raw_distance,
                "variant": name,
            }
        )
    maximum = max(row["coarse_to_raw_ratio"] for row in rows)
    return {
        "claim_scope": "synthetic_straight-carrier_nuisance_bank_only",
        "maximum_allowed_coarse_to_raw_ratio": 0.20,
        "maximum_observed_coarse_to_raw_ratio": maximum,
        "passed": maximum <= 0.20,
        "rows": rows,
    }


def _batch_tensor(groups, indices, *, epoch: int, seed: int, augment: bool):
    torch, _, _ = base._torch_runtime()
    arrays, labels = [], [[], [], []]
    for index in indices:
        group = groups[index]
        array = group.ink
        if augment:
            key = base.content_epoch_key(seed, epoch, group.digest)
            array = base.d4_transform(array, int.from_bytes(key, "big") % 8)
        arrays.append(array)
        for head in range(3):
            labels[head].append(group.labels[head])
    ink = torch.from_numpy(np.stack(arrays)[:, None])
    targets = tuple(torch.tensor(value, dtype=torch.long) for value in labels)
    return semantic_channels(ink), targets


def _effective_class_weights(groups, head: int, classes: int):
    torch, _, _ = base._torch_runtime()
    counts = [0] * classes
    for group in groups:
        counts[group.labels[head]] += group.multiplicity
    return torch.tensor(_weight_values(counts), dtype=torch.float32), counts


def _weight_values(counts: Sequence[int]) -> list[float]:
    beta = 0.999
    raw = [(1.0 - beta) / (1.0 - beta**count) if count else 0.0 for count in counts]
    mean = sum(value for value in raw if value) / sum(value != 0 for value in raw)
    return [value / mean if value else 0.0 for value in raw]


def build_model(*, seed: int = 260810):
    torch = base._configure_torch(seed)
    _, nn, functional = base._torch_runtime()

    class Block(nn.Module):
        def __init__(self, incoming: int, outgoing: int, stride: int) -> None:
            super().__init__()
            self.a = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
            self.ab = nn.BatchNorm2d(outgoing)
            self.b = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
            self.bb = nn.BatchNorm2d(outgoing)
            self.skip = (
                nn.Identity()
                if incoming == outgoing and stride == 1
                else nn.Sequential(
                    nn.Conv2d(incoming, outgoing, 1, stride, bias=False),
                    nn.BatchNorm2d(outgoing),
                )
            )

        def forward(self, value):
            residual = self.skip(value)
            value = functional.relu(self.ab(self.a(value)), inplace=False)
            return functional.relu(self.bb(self.b(value)) + residual, inplace=False)

    class Encoder(nn.Module):
        def __init__(self, incoming: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(incoming, 32, 5, 2, 2, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                Block(32, 32, 1),
                Block(32, 64, 2),
                Block(64, 96, 2),
                Block(96, 128, 2),
                nn.Conv2d(128, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
            )

        def forward(self, value):
            return self.layers(value)

    def pyramid(value):
        pieces = []
        for size in (1, 2, 4):
            pieces.append(functional.adaptive_avg_pool2d(value, size).flatten(1))
            pieces.append(functional.adaptive_max_pool2d(value, size).flatten(1))
        return torch.cat(pieces, dim=1)

    def ordinal_class_log_likelihoods(threshold_logits):
        if threshold_logits.ndim != 2 or threshold_logits.shape[1] != 9:
            raise SpatialDevelopmentError("ordinal threshold shape differs")
        positive = functional.logsigmoid(threshold_logits)
        negative = functional.logsigmoid(-threshold_logits)
        return torch.stack(
            [
                positive[:, :count].sum(dim=1)
                + negative[:, count:].sum(dim=1)
                for count in range(10)
            ],
            dim=1,
        )

    class SpatialModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.coarse_encoder = Encoder(1)
            self.raw_encoder = Encoder(2)
            feature_count = 64 * 2 * (1 + 4 + 16)
            self.count_hidden = nn.Sequential(
                nn.Linear(feature_count + 16, 384), nn.ReLU(inplace=False)
            )
            self.catalog_hidden = nn.Sequential(
                nn.Linear(feature_count * 2, 384), nn.ReLU(inplace=False)
            )
            self.straight = nn.Linear(384, 9)
            self.arc = nn.Linear(384, 9)
            self.catalog = nn.Linear(384, 3)
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        def forward(self, value):
            raw_binary, coarse = value[:, :2], value[:, 2:3]
            coarse_features = pyramid(self.coarse_encoder(coarse))
            mass = functional.adaptive_avg_pool2d(coarse, 4).flatten(1)
            count = self.count_hidden(torch.cat((coarse_features, mass), dim=1))
            catalog = self.catalog_hidden(
                torch.cat((coarse_features, pyramid(self.raw_encoder(raw_binary))), dim=1)
            )
            return (
                ordinal_class_log_likelihoods(self.straight(count)),
                ordinal_class_log_likelihoods(self.arc(count)),
                self.catalog(catalog),
            )

    return SpatialModel()


@contextmanager
def _installed_successor() -> Iterator[None]:
    saved = {
        "ARCHITECTURE_ID": base.ARCHITECTURE_ID,
        "EXECUTION_PROTOCOL": base.EXECUTION_PROTOCOL,
        "preprocess_png_bytes": base.preprocess_png_bytes,
        "_batch_tensor": base._batch_tensor,
        "_class_weights": base._class_weights,
        "build_model": base.build_model,
        "_verify_execution_protocol": base._verify_execution_protocol,
    }
    base.ARCHITECTURE_ID = ARCHITECTURE_ID
    base.EXECUTION_PROTOCOL = MappingProxyType(
        {
            "batch_size": 64,
            "cpu_threads": 1,
            "epochs": 24,
            "image_size": 128,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "optimizer_betas": (0.9, 0.999),
            "optimizer_eps": 1e-8,
            "random_seed": 260810,
            "weight_decay": 0.0001,
        }
    )
    base.preprocess_png_bytes = preprocess_png_bytes
    base._batch_tensor = _batch_tensor
    base._class_weights = _effective_class_weights
    base.build_model = build_model

    def verify_frozen_parent_protocol(protocol_plan):
        successor_protocol = base.EXECUTION_PROTOCOL
        base.EXECUTION_PROTOCOL = saved["EXECUTION_PROTOCOL"]
        try:
            saved["_verify_execution_protocol"](protocol_plan)
        finally:
            base.EXECUTION_PROTOCOL = successor_protocol

    # The v2 artifact remains authority for the retained cohort and optimizer
    # family, while this source-bound outer precommit explicitly supersedes
    # only image size, epoch count, representation, and architecture.
    base._verify_execution_protocol = verify_frozen_parent_protocol
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def _verify_successor_precommit(
    path: Path,
    *,
    fit_authorization: Path,
    fit_precommit: Path,
    failed_fit: Path,
    failed_replay: Path,
    checkpoint: Path,
    inner_result: Path,
    result: Path,
) -> dict[str, Any]:
    expected = create_successor_precommit(
        fit_authorization=fit_authorization,
        fit_precommit=fit_precommit,
        failed_fit=failed_fit,
        failed_replay=failed_replay,
        intended_checkpoint=checkpoint,
        intended_inner_result=inner_result,
        intended_result=result,
        output=path,
    )
    if expected.get("source_sha256") != source_sha256():
        raise SpatialDevelopmentError("successor source changed after precommit")
    return expected


def _successor_gate(inner: Mapping[str, Any]) -> dict[str, Any]:
    metrics = inner["validation_metrics"]
    strata = metrics["straight_required_strata"]
    checks = {
        "base_validation_gate": inner["validation_gate"]["passed"] is True,
        **{
            name: strata[name]["straight_top1"] >= threshold
            for name, threshold in STRATUM_THRESHOLDS.items()
        },
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "stratum_thresholds": dict(STRATUM_THRESHOLDS),
        "on_failure": "development_GAP;_fresh_CAL_eval_family_target_and_query_remain_sealed",
    }


def _result_body(
    inner: Mapping[str, Any], precommit: Mapping[str, Any]
) -> dict[str, Any]:
    counts = inner["training_class_counts"]
    weights = {
        name: _weight_values(counts[name])
        for name in ("straight", "arc", "catalog_convexity")
    }
    return {
        "architecture_id": ARCHITECTURE_ID,
        "authority": "adaptive_already_exposed_development_only",
        "authoritative_checkpoint_envelope": {
            "checkpoint_raw_sha256": inner["checkpoint_raw_sha256"],
            "checkpoint_state_dict_sha256": inner["checkpoint_state_dict_sha256"],
            "successor_config_digest": successor_config_digest(),
            "successor_source_sha256": source_sha256(),
        },
        "checkpoint_internal_config_digest_is_parent_protocol_only": True,
        "class_counts": counts,
        "class_weights": weights,
        "config_digest": successor_config_digest(),
        "forbidden_cohorts_opened": 0,
        "inner_result_record_digest": inner["record_digest"],
        "predecessor_failed_fit_record_digest": FAILED_FIT_DIGEST,
        "protocol": json.loads(canonical_json(dict(PROTOCOL))),
        "runtime": inner["runtime"],
        "schema": SCHEMA_RESULT,
        "selection_key": (
            "straight_and_known_catalog_joint_exact;_straight_top1;_catalog_"
            "binary_balanced_accuracy;_arc_top1;_negative_cross_entropy;_"
            "negative_epoch"
        ),
        "source_sha256": source_sha256(),
        "successor_gate": _successor_gate(inner),
        "successor_precommit_record_digest": precommit["record_digest"],
        "validation_metrics": inner["validation_metrics"],
        "validation_is_selection_biased_development": True,
    }


def run_successor_training(
    *,
    v3_plan: Path,
    v3_development_panels: Path,
    v2_training_plan: Path,
    v2_development_labels: Path,
    dataset_root: Path,
    fit_authorization: Path,
    fit_precommit: Path,
    failed_fit: Path,
    failed_replay: Path,
    successor_precommit: Path,
    checkpoint: Path,
    inner_result: Path,
    output: Path,
) -> dict[str, Any]:
    precommit = _verify_successor_precommit(
        successor_precommit,
        fit_authorization=fit_authorization,
        fit_precommit=fit_precommit,
        failed_fit=failed_fit,
        failed_replay=failed_replay,
        checkpoint=checkpoint,
        inner_result=inner_result,
        result=output,
    )
    with _installed_successor():
        inner = base.run_fit_training(
            v3_plan_path=v3_plan,
            v3_development_path=v3_development_panels,
            v2_protocol_plan_path=v2_training_plan,
            v2_development_labels_path=v2_development_labels,
            dataset_root=dataset_root,
            authorization_path=fit_authorization,
            precommit_path=fit_precommit,
            checkpoint_path=checkpoint,
            record_path=inner_result,
        )
    body = _result_body(inner, precommit)
    value = _seal(body)
    base._write_fsynced(output, value)
    if _load(output, "successor result fresh load") != value:
        raise SpatialDevelopmentError("successor result fresh-load differs")
    return value


def replay_successor_training(
    *,
    v3_plan: Path,
    v3_development_panels: Path,
    v2_training_plan: Path,
    v2_development_labels: Path,
    dataset_root: Path,
    fit_authorization: Path,
    fit_precommit: Path,
    failed_fit: Path,
    failed_replay: Path,
    successor_precommit: Path,
    checkpoint: Path,
    inner_result: Path,
    result: Path,
    inner_replay: Path,
    output: Path,
) -> dict[str, Any]:
    precommit = _verify_successor_precommit(
        successor_precommit,
        fit_authorization=fit_authorization,
        fit_precommit=fit_precommit,
        failed_fit=failed_fit,
        failed_replay=failed_replay,
        checkpoint=checkpoint,
        inner_result=inner_result,
        result=result,
    )
    archived = _load(result, "successor result")
    inner_archived = _load(inner_result, "successor inner result")
    expected_archived = _seal(_result_body(inner_archived, precommit))
    if archived != expected_archived:
        raise SpatialDevelopmentError("successor result deterministic reconstruction differs")
    with _installed_successor():
        replay = base.replay_fit_training(
            v3_plan_path=v3_plan,
            v3_development_path=v3_development_panels,
            v2_protocol_plan_path=v2_training_plan,
            v2_development_labels_path=v2_development_labels,
            dataset_root=dataset_root,
            authorization_path=fit_authorization,
            precommit_path=fit_precommit,
            checkpoint_path=checkpoint,
            record_path=inner_result,
            replay_output_path=inner_replay,
        )
    body = {
        "archived_result_record_digest": archived["record_digest"],
        "forbidden_cohorts_opened": 0,
        "inner_replay_record_digest": replay["record_digest"],
        "outer_result_reconstructed_exactly": True,
        "metrics_exact": replay["metrics_exact"],
        "predictions_exact": replay["predictions_exact"],
        "schema": SCHEMA_REPLAY,
        "source_sha256": source_sha256(),
        "successor_precommit_record_digest": precommit["record_digest"],
    }
    value = _seal(body)
    base._write_fsynced(output, value)
    if _load(output, "successor replay fresh load") != value:
        raise SpatialDevelopmentError("successor replay fresh-load differs")
    return value


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fit-authorization", type=Path, required=True)
    parser.add_argument("--fit-precommit", type=Path, required=True)
    parser.add_argument("--failed-fit", type=Path, required=True)
    parser.add_argument("--failed-replay", type=Path, required=True)


def _execution(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    parser.add_argument("--successor-precommit", type=Path, required=True)
    parser.add_argument("--v3-plan", type=Path, required=True)
    parser.add_argument("--v3-development-panels", type=Path, required=True)
    parser.add_argument("--v2-training-plan", type=Path, required=True)
    parser.add_argument("--v2-development-labels", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inner-result", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    precommit = commands.add_parser("precommit-successor")
    _common(precommit)
    precommit.add_argument("--intended-checkpoint", type=Path, required=True)
    precommit.add_argument("--intended-inner-result", type=Path, required=True)
    precommit.add_argument("--intended-result", type=Path, required=True)
    precommit.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train-successor")
    _execution(train)
    train.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay-successor")
    _execution(replay)
    replay.add_argument("--result", type=Path, required=True)
    replay.add_argument("--inner-replay", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    common = {
        "fit_authorization": args.fit_authorization,
        "fit_precommit": args.fit_precommit,
        "failed_fit": args.failed_fit,
        "failed_replay": args.failed_replay,
    }
    if args.command == "precommit-successor":
        result = create_successor_precommit(
            **common,
            intended_checkpoint=args.intended_checkpoint,
            intended_inner_result=args.intended_inner_result,
            intended_result=args.intended_result,
            output=args.output,
        )
    else:
        execution = {
            **common,
            "successor_precommit": args.successor_precommit,
            "v3_plan": args.v3_plan,
            "v3_development_panels": args.v3_development_panels,
            "v2_training_plan": args.v2_training_plan,
            "v2_development_labels": args.v2_development_labels,
            "dataset_root": args.dataset_root,
            "checkpoint": args.checkpoint,
            "inner_result": args.inner_result,
        }
        if args.command == "train-successor":
            result = run_successor_training(**execution, output=args.output)
        else:
            result = replay_successor_training(
                **execution,
                result=args.result,
                inner_replay=args.inner_replay,
                output=args.output,
            )
    print(json.dumps({"record_digest": result["record_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
