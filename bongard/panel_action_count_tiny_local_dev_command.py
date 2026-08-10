"""Tiny, pose-free development observer for action counts and catalog class.

The official ShapeBongard action programs contain action descriptors, but not
the sampled pose, scale, or renderer RNG state used for the released PNGs.
Consequently they do *not* authorize pixel-aligned masks.  This successor
learns an unordered set of at most nine action descriptors instead: nine
query slots predict ``none``/``line``/``arc`` plus style-stripped local
geometry.  A detached Hungarian assignment makes the descriptor loss
permutation invariant.  An exact differentiable categorical DP maps the slot
probabilities to a finite joint distribution over straight and arc counts.

This module is deliberately development-only.  It contains no dataset path,
CAL/evaluation/family/query/target loader, and it never interprets a typed GAP
as a negative or a zero target.  Its precommit binds predecessor failures,
the separate supervision authority, exact capacity coverage, intended
outputs, and hard CPU limits before any PNG reader may be called.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from bongard.canonical import canonical_digest, canonical_json


ARCHITECTURE_ID = "tiny-pose-free-nine-action-query-set-catalog/v1"
PRECOMMIT_SCHEMA = "gkm.bongard-tiny-local-action-development-precommit.v1"
RAW_OBSERVATION_SCHEMA = "gkm.bongard-tiny-local-action-raw-observation.v1"
CALIBRATION_SCHEMA = "gkm.bongard-tiny-local-action-joint-calibration.v1"
CALIBRATED_OBSERVATION_SCHEMA = (
    "gkm.bongard-tiny-local-action-calibrated-observation.v1"
)
REPLAY_SCHEMA = "gkm.bongard-tiny-local-action-model-free-replay.v1"

FAILED_BASELINE_DIGEST = (
    "sha256:f8b79047228a91fd3fdd47a262299b0cd683daa727981e568450371be4e4dff2"
)
RETIRED_SPATIAL_OUTCOME_DIGEST = (
    "sha256:92f4905b5c002aab7cc7288f60037c538fa761bd4d3b56cb4154ffddf5bcf9d7"
)

SLOT_CLASSES = ("none", "line", "arc")
CATALOG_CLASSES = ("catalog_unresolved", "nonconvex", "convex")
MAX_ACTION_SLOTS = 9
COUNT_DOMAIN = tuple(range(MAX_ACTION_SLOTS + 1))
CATALOG_DOMAIN = tuple(range(len(CATALOG_CLASSES)))

PROTOCOL = MappingProxyType(
    {
        "batch_size": 128,
        "cpu_threads": 1,
        "epochs": 6,
        "image_size": 64,
        "maximum_action_slots": MAX_ACTION_SLOTS,
        "maximum_optimizer_steps": 528,
        "maximum_parameter_count": 20_000,
        "maximum_projected_runtime_seconds": 420.0,
        "maximum_wall_runtime_seconds": 600.0,
        "optimizer": "AdamW",
        "learning_rate": 0.002,
        "weight_decay": 0.0001,
        "random_seed": 260810,
        "set_assignment": "detached_scipy_linear_sum_assignment",
        "count_projection": "exact_nine-slot_categorical_dynamic_program",
        "two_shape_policy": (
            "retain_shape_index_in_target_provenance_then_flatten_for_matching_"
            "only_when_total_actions_at_most_nine_else_capacity_GAP"
        ),
        "pixel_registration_required": False,
        "pixel_aligned_dense_targets_allowed": False,
        "forbidden_cohorts": (
            "fresh_v3_calibration",
            "fresh_v3_evaluation",
            "same_family_calibration",
            "target",
            "query",
        ),
    }
)


class TinyLocalObserverError(RuntimeError):
    """The tiny observer, its custody, or a typed input differs."""


class RuntimeBudgetExceeded(TinyLocalObserverError):
    """The frozen CPU work or wall-clock budget was exceeded."""


def source_sha256() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def successor_config_digest() -> str:
    return "sha256:" + canonical_digest(
        {
            "architecture_id": ARCHITECTURE_ID,
            "protocol": json.loads(canonical_json(dict(PROTOCOL))),
            "source_sha256": source_sha256(),
        }
    )


def _address(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def _load_record(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TinyLocalObserverError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise TinyLocalObserverError(f"{label} is not canonical JSON plus newline")
    body = dict(value)
    found = body.pop("record_digest", None)
    if found != "sha256:" + canonical_digest(body):
        raise TinyLocalObserverError(f"{label} record digest differs")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise TinyLocalObserverError(f"refusing to overwrite {path}")
        return
    temporary = path.with_name(path.name + ".tmp-tiny-local-observer")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise TinyLocalObserverError(f"cannot durably write {path}: {exc}") from exc


def _torch_runtime():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - environment failure
        raise TinyLocalObserverError("PyTorch is unavailable") from exc
    return torch, nn, functional


def _configure_torch(seed: int):
    torch, _, _ = _torch_runtime()
    torch.set_num_threads(int(PROTOCOL["cpu_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise TinyLocalObserverError("torch interop threads cannot be fixed")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    return torch


def preprocess_png_bytes(raw: bytes, *, image_size: int = 64) -> np.ndarray:
    """Tight-crop one PNG to uint8 ink without reading any label source."""

    if image_size != int(PROTOCOL["image_size"]):
        raise TinyLocalObserverError("image size differs from frozen protocol")
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise TinyLocalObserverError("input must be one PNG frame")
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
    except TinyLocalObserverError:
        raise
    except Exception as exc:
        raise TinyLocalObserverError(f"cannot decode PNG: {exc}") from exc
    ys, xs = np.nonzero(gray < 250)
    if len(xs) == 0:
        raise TinyLocalObserverError("PNG has no ink")
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


def input_channels(ink):
    """Return raw and thresholded ink; no program-derived pixel channel exists."""

    torch, _, _ = _torch_runtime()
    if ink.ndim != 4 or ink.shape[1:] != (1, 64, 64):
        raise TinyLocalObserverError("ink tensor must have shape Bx1x64x64")
    raw = ink.to(torch.float32) / 255.0
    binary = (raw >= (10.0 / 255.0)).to(torch.float32)
    return torch.cat((raw, binary), dim=1)


def build_model(*, seed: int = 260810):
    """Build the frozen tiny query-set model and three-class catalog head."""

    torch = _configure_torch(seed)
    _, nn, functional = _torch_runtime()

    class DepthwiseBlock(nn.Module):
        def __init__(self, incoming: int, outgoing: int, stride: int) -> None:
            super().__init__()
            self.depthwise = nn.Conv2d(
                incoming, incoming, 3, stride, 1, groups=incoming, bias=False
            )
            self.depthwise_norm = nn.BatchNorm2d(incoming)
            self.pointwise = nn.Conv2d(incoming, outgoing, 1, bias=False)
            self.pointwise_norm = nn.BatchNorm2d(outgoing)

        def forward(self, value):
            value = functional.silu(
                self.depthwise_norm(self.depthwise(value)), inplace=False
            )
            return functional.silu(
                self.pointwise_norm(self.pointwise(value)), inplace=False
            )

    class TinyActionSetModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(2, 12, 5, 2, 2, bias=False),
                nn.BatchNorm2d(12),
                nn.SiLU(inplace=False),
                DepthwiseBlock(12, 20, 2),
                DepthwiseBlock(20, 24, 2),
                DepthwiseBlock(24, 24, 1),
            )
            self.queries = nn.Parameter(torch.empty(MAX_ACTION_SLOTS, 24))
            self.key = nn.Linear(24, 24, bias=False)
            self.value = nn.Linear(24, 24, bias=False)
            self.slot_hidden = nn.Sequential(
                nn.Linear(48, 32), nn.SiLU(inplace=False)
            )
            self.slot_class = nn.Linear(32, len(SLOT_CLASSES))
            # line_length, arc_radius, arc_sweep_magnitude.  Junction turns are
            # an independent unordered multiset in the authority and are not
            # falsely assigned to action slots.
            self.slot_geometry = nn.Linear(32, 3)
            self.catalog = nn.Linear(48, len(CATALOG_CLASSES))
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            nn.init.normal_(self.queries, mean=0.0, std=0.02)

        def forward(self, value):
            if value.ndim != 4 or value.shape[1:] != (2, 64, 64):
                raise TinyLocalObserverError("model input must have shape Bx2x64x64")
            feature = self.stem(value)  # B x 24 x 8 x 8
            tokens = feature.flatten(2).transpose(1, 2)  # B x 64 x 24
            keys = self.key(tokens)
            values = self.value(tokens)
            scores = torch.einsum("qc,bnc->bqn", self.queries, keys) / math.sqrt(24.0)
            attention = scores.softmax(dim=-1)
            contexts = torch.einsum("bqn,bnc->bqc", attention, values)
            queries = self.queries.unsqueeze(0).expand(value.shape[0], -1, -1)
            hidden = self.slot_hidden(torch.cat((contexts, queries), dim=-1))
            raw_geometry = self.slot_geometry(hidden)
            geometry = raw_geometry.sigmoid()
            pooled = torch.cat(
                (
                    functional.adaptive_avg_pool2d(feature, 1).flatten(1),
                    functional.adaptive_max_pool2d(feature, 1).flatten(1),
                ),
                dim=1,
            )
            return {
                "attention": attention.reshape(-1, MAX_ACTION_SLOTS, 8, 8),
                "catalog_logits": self.catalog(pooled),
                "geometry": geometry,
                "slot_logits": self.slot_class(hidden),
            }

    model = TinyActionSetModel()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count > int(PROTOCOL["maximum_parameter_count"]):
        raise TinyLocalObserverError("model exceeds frozen parameter limit")
    return model


def parameter_count(model=None) -> int:
    value = build_model() if model is None else model
    return sum(parameter.numel() for parameter in value.parameters())


def state_dict_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"gkm.tiny-local-action-state-dict.v1\0")
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = canonical_json(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)}
        )
        payload = tensor.numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def load_verified_checkpoint(
    path: Path,
    *,
    expected_training_precommit_record_digest: str | None,
    expected_training_result_record_digest: str | None,
):
    """Load a checkpoint only with explicit precommit/result authority.

    No training command/result exists in this infrastructure-only revision, so
    callers cannot accidentally promote a merely source-compatible state to
    benchmark evidence by omitting those external bindings.
    """

    torch, _, _ = _torch_runtime()
    try:
        raw = path.read_bytes()
        payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TinyLocalObserverError(f"cannot load tiny checkpoint: {exc}") from exc
    required = {
        "architecture_id",
        "config_digest",
        "selected_epoch",
        "source_sha256",
        "state_dict",
        "state_dict_sha256",
        "training_precommit_record_digest",
        "training_result_record_digest",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise TinyLocalObserverError("tiny checkpoint envelope fields differ")
    if (
        expected_training_precommit_record_digest is None
        or expected_training_result_record_digest is None
    ):
        raise TinyLocalObserverError("benchmark checkpoint authority is absent")
    if (
        payload["architecture_id"] != ARCHITECTURE_ID
        or payload["source_sha256"] != source_sha256()
        or payload["config_digest"] != successor_config_digest()
        or payload["training_precommit_record_digest"]
        != expected_training_precommit_record_digest
        or payload["training_result_record_digest"]
        != expected_training_result_record_digest
        or payload["state_dict_sha256"] != state_dict_digest(payload["state_dict"])
        or isinstance(payload["selected_epoch"], bool)
        or not isinstance(payload["selected_epoch"], int)
        or payload["selected_epoch"] not in range(int(PROTOCOL["epochs"]))
    ):
        raise TinyLocalObserverError("tiny checkpoint envelope binding differs")
    model = build_model(seed=int(PROTOCOL["random_seed"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    if state_dict_digest(model.state_dict()) != payload["state_dict_sha256"]:
        raise TinyLocalObserverError("fresh model state differs from checkpoint")
    return model, payload, _address(raw)


def joint_count_probabilities(slot_probabilities):
    """Exact DP for P(number of lines, number of arcs) across nine slots."""

    torch, _, _ = _torch_runtime()
    if (
        slot_probabilities.ndim != 3
        or slot_probabilities.shape[1:] != (MAX_ACTION_SLOTS, len(SLOT_CLASSES))
    ):
        raise TinyLocalObserverError("slot probabilities have wrong shape")
    if not torch.isfinite(slot_probabilities).all():
        raise TinyLocalObserverError("slot probabilities are not finite")
    if torch.any(slot_probabilities < 0):
        raise TinyLocalObserverError("slot probabilities are negative")
    totals = slot_probabilities.sum(dim=-1)
    if not torch.allclose(totals, torch.ones_like(totals), atol=1e-6, rtol=0.0):
        raise TinyLocalObserverError("slot probabilities do not sum to one")
    batch = slot_probabilities.shape[0]
    distribution = slot_probabilities.new_zeros((batch, 10, 10))
    distribution[:, 0, 0] = 1.0
    for slot in range(MAX_ACTION_SLOTS):
        probability = slot_probabilities[:, slot]
        updated = distribution * probability[:, 0, None, None]
        line_shift = distribution.new_zeros(distribution.shape)
        line_shift[:, 1:, :] = distribution[:, :-1, :]
        updated = updated + line_shift * probability[:, 1, None, None]
        arc_shift = distribution.new_zeros(distribution.shape)
        arc_shift[:, :, 1:] = distribution[:, :, :-1]
        updated = updated + arc_shift * probability[:, 2, None, None]
        distribution = updated
    return distribution


def _target_number(value: object, *, label: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TinyLocalObserverError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise TinyLocalObserverError(f"{label} leaves [{lower},{upper}]")
    return number


def _normalized_interval(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"center", "lower", "upper"}:
        raise TinyLocalObserverError(f"{label} is not an exact interval triple")
    center = _target_number(value["center"], label=f"{label} center", lower=0.0, upper=1.0)
    lower = _target_number(value["lower"], label=f"{label} lower", lower=0.0, upper=1.0)
    upper = _target_number(value["upper"], label=f"{label} upper", lower=0.0, upper=1.0)
    if not lower <= center <= upper:
        raise TinyLocalObserverError(f"{label} center leaves its interval")
    return {"center": center, "lower": lower, "upper": upper}


def normalize_pose_free_action(action: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one authority-compiled descriptor without inventing coordinates."""

    if not isinstance(action, Mapping):
        raise TinyLocalObserverError("action target is not an object")
    kind = action.get("kind")
    if kind not in {"line", "arc"}:
        raise TinyLocalObserverError("action target kind is not line/arc")
    membership = action.get("shape_membership_local_index")
    if membership is not None and (
        isinstance(membership, bool) or not isinstance(membership, int) or membership < 0
    ):
        raise TinyLocalObserverError("action target shape membership is invalid")
    result: dict[str, Any] = {
        "kind": kind,
    }
    if membership is not None:
        result["shape_membership_local_index"] = membership
    if kind == "line":
        result["line_length"] = _normalized_interval(
            action.get("line_length"), label="line length"
        )
    else:
        result["arc_radius"] = _normalized_interval(
            action.get("arc_radius"), label="arc radius"
        )
        result["arc_sweep_magnitude"] = _normalized_interval(
            action.get("arc_sweep_magnitude"), label="arc sweep magnitude"
        )
    return result


def flatten_pose_free_shapes(
    shapes: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], ...]:
    """Retain shape provenance, then flatten only within the nine-slot capacity."""

    if not isinstance(shapes, Sequence) or isinstance(shapes, (str, bytes)) or not shapes:
        raise TinyLocalObserverError("pose-free shapes must be a nonempty sequence")
    flattened: list[dict[str, Any]] = []
    for shape_index, shape in enumerate(shapes):
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or not shape:
            raise TinyLocalObserverError("pose-free shape must contain actions")
        for action in shape:
            with_membership = {**dict(action), "shape_membership_local_index": shape_index}
            normalized = normalize_pose_free_action(with_membership)
            flattened.append(normalized)
    if len(flattened) > MAX_ACTION_SLOTS:
        raise TinyLocalObserverError("capacity_GAP: total actions exceed nine slots")
    return tuple(flattened)


def _distance_outside_interval(prediction, interval: Mapping[str, float]):
    """Zero inside a certified rounding interval, nearest-bound distance outside."""

    torch, _, functional = _torch_runtime()
    lower = prediction.new_tensor(interval["lower"])
    upper = prediction.new_tensor(interval["upper"])
    return functional.relu(lower - prediction) + functional.relu(prediction - upper)


def _hungarian_matches(slot_logits, geometry, targets: Sequence[Mapping[str, Any]]):
    """Return deterministic detached minimum-cost slot/target pairs."""

    torch, _, functional = _torch_runtime()
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise TinyLocalObserverError("SciPy assignment runtime is unavailable") from exc
    if slot_logits.shape != (MAX_ACTION_SLOTS, 3) or geometry.shape != (
        MAX_ACTION_SLOTS,
        3,
    ):
        raise TinyLocalObserverError("slot tensors have wrong shape")
    if len(targets) > MAX_ACTION_SLOTS:
        raise TinyLocalObserverError("capacity_GAP: target set exceeds nine slots")
    if not targets:
        return tuple()
    log_probabilities = functional.log_softmax(slot_logits.detach(), dim=-1)
    detached_geometry = geometry.detach()
    rows = []
    for target in targets:
        value = normalize_pose_free_action(target)
        class_index = 1 if value["kind"] == "line" else 2
        cost = -log_probabilities[:, class_index]
        if class_index == 1:
            cost = cost + _distance_outside_interval(
                detached_geometry[:, 0], value["line_length"]
            )
        else:
            cost = cost + _distance_outside_interval(
                detached_geometry[:, 1], value["arc_radius"]
            )
            cost = cost + _distance_outside_interval(
                detached_geometry[:, 2], value["arc_sweep_magnitude"]
            )
        rows.append(cost.cpu().numpy())
    cost_matrix = np.stack(rows, axis=1)  # slots x targets
    slot_indices, target_indices = linear_sum_assignment(cost_matrix)
    pairs = tuple(
        sorted(
            zip(slot_indices.tolist(), target_indices.tolist()),
            key=lambda item: item[0],
        )
    )
    if len(pairs) != len(targets):
        raise TinyLocalObserverError("Hungarian assignment is incomplete")
    return pairs


def set_prediction_loss(
    output: Mapping[str, Any],
    action_targets: Sequence[Sequence[Mapping[str, Any]]],
    catalog_targets,
):
    """Permutation-invariant descriptor + exact count + catalog loss."""

    torch, _, functional = _torch_runtime()
    logits = output.get("slot_logits")
    geometry = output.get("geometry")
    catalog_logits = output.get("catalog_logits")
    if logits is None or geometry is None or catalog_logits is None:
        raise TinyLocalObserverError("model output fields differ")
    batch = logits.shape[0]
    if len(action_targets) != batch or tuple(catalog_targets.shape) != (batch,):
        raise TinyLocalObserverError("target batch cardinality differs")
    classification_targets = torch.zeros(
        (batch, MAX_ACTION_SLOTS), dtype=torch.long, device=logits.device
    )
    geometry_loss = logits.new_zeros(())
    truth_pairs: list[tuple[int, int]] = []
    matched_actions = 0
    for batch_index, target_set in enumerate(action_targets):
        normalized = tuple(normalize_pose_free_action(item) for item in target_set)
        matches = _hungarian_matches(
            logits[batch_index], geometry[batch_index], normalized
        )
        straight = sum(item["kind"] == "line" for item in normalized)
        arc = sum(item["kind"] == "arc" for item in normalized)
        truth_pairs.append((straight, arc))
        for slot_index, target_index in matches:
            target = normalized[target_index]
            class_index = 1 if target["kind"] == "line" else 2
            classification_targets[batch_index, slot_index] = class_index
            prediction = geometry[batch_index, slot_index]
            if class_index == 1:
                geometry_loss = geometry_loss + functional.smooth_l1_loss(
                    _distance_outside_interval(prediction[0], target["line_length"]),
                    prediction.new_zeros(()),
                    reduction="sum",
                )
            else:
                geometry_loss = geometry_loss + functional.smooth_l1_loss(
                    _distance_outside_interval(prediction[1], target["arc_radius"]),
                    prediction.new_zeros(()),
                    reduction="sum",
                )
                geometry_loss = geometry_loss + functional.smooth_l1_loss(
                    _distance_outside_interval(
                        prediction[2], target["arc_sweep_magnitude"]
                    ),
                    prediction.new_zeros(()),
                    reduction="sum",
                )
            matched_actions += 1
    classification_loss = functional.cross_entropy(
        logits.reshape(-1, 3), classification_targets.reshape(-1)
    )
    if matched_actions:
        geometry_loss = geometry_loss / matched_actions
    probabilities = logits.softmax(dim=-1)
    joint = joint_count_probabilities(probabilities)
    count_loss = -torch.stack(
        [
            joint[index, straight, arc].clamp_min(1e-12).log()
            for index, (straight, arc) in enumerate(truth_pairs)
        ]
    ).mean()
    catalog_loss = functional.cross_entropy(catalog_logits, catalog_targets)
    total = classification_loss + geometry_loss + count_loss + catalog_loss
    return {
        "catalog": catalog_loss,
        "classification": classification_loss,
        "count": count_loss,
        "geometry": geometry_loss,
        "total": total,
    }


def _canonical_float(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise TinyLocalObserverError("observation contains non-finite probability")
    return number


def predict_raw_evidence(
    model,
    *,
    panel_ids: Sequence[str],
    panel_png_bytes: Sequence[bytes],
    checkpoint_state_dict_sha256: str,
    config_digest: str,
) -> tuple[dict[str, Any], ...]:
    """Decode exact PNG bytes and archive complete evidence before label access."""

    torch, _, _ = _torch_runtime()
    if len(panel_ids) != len(panel_png_bytes) or not panel_ids:
        raise TinyLocalObserverError("prediction identity cardinality differs")
    if len(set(panel_ids)) != len(panel_ids):
        raise TinyLocalObserverError("prediction panel IDs are not unique")
    if state_dict_digest(model.state_dict()) != checkpoint_state_dict_sha256:
        raise TinyLocalObserverError("prediction model state differs from checkpoint binding")
    arrays = [preprocess_png_bytes(raw) for raw in panel_png_bytes]
    ink = torch.from_numpy(np.stack(arrays)[:, None])
    pixels = input_channels(ink)
    model.eval()
    with torch.no_grad():
        output = model(pixels)
        slot_probabilities = output["slot_logits"].softmax(dim=-1)
        count_probabilities = joint_count_probabilities(slot_probabilities)
        catalog_probabilities = output["catalog_logits"].softmax(dim=-1)
    result = []
    for index, (panel_id, png_raw) in enumerate(zip(panel_ids, panel_png_bytes)):
        if not isinstance(panel_id, str) or not panel_id:
            raise TinyLocalObserverError("panel ID is invalid")
        if not isinstance(png_raw, bytes) or not png_raw:
            raise TinyLocalObserverError("PNG payload is invalid")
        png_digest = _address(png_raw)
        slots = []
        for slot_index in range(MAX_ACTION_SLOTS):
            slots.append(
                {
                    "attention_8x8": [
                        [_canonical_float(value) for value in row]
                        for row in output["attention"][index, slot_index].cpu().tolist()
                    ],
                    "class_probabilities_none_line_arc": [
                        _canonical_float(value)
                        for value in slot_probabilities[index, slot_index].cpu().tolist()
                    ],
                    "geometry_line_length_arc_radius_arc_sweep_magnitude": [
                        _canonical_float(value)
                        for value in output["geometry"][index, slot_index].cpu().tolist()
                    ],
                    "slot_index": slot_index,
                }
            )
        body = {
            "architecture_id": ARCHITECTURE_ID,
            "catalog_class_order": list(CATALOG_CLASSES),
            "catalog_probabilities": [
                _canonical_float(value)
                for value in catalog_probabilities[index].cpu().tolist()
            ],
            "checkpoint_state_dict_sha256": checkpoint_state_dict_sha256,
            "config_digest": config_digest,
            "joint_count_probabilities_straight_rows_arc_columns": [
                [_canonical_float(value) for value in row]
                for row in count_probabilities[index].cpu().tolist()
            ],
            "panel_id": panel_id,
            "pixel_registration_claimed": False,
            "png_sha256": png_digest,
            "png_size_bytes": len(png_raw),
            "schema": RAW_OBSERVATION_SCHEMA,
            "slot_class_order": list(SLOT_CLASSES),
            "slots": slots,
            "source_sha256": source_sha256(),
        }
        result.append(_seal(body))
    return tuple(result)


def verify_raw_evidence_from_checkpoint(
    *,
    checkpoint_path: Path,
    panel_ids: Sequence[str],
    panel_png_bytes: Sequence[bytes],
    archived: Sequence[Mapping[str, Any]],
    expected_training_precommit_record_digest: str,
    expected_training_result_record_digest: str,
) -> dict[str, Any]:
    """Recompute an exact inference batch from checkpoint and PNG bytes."""

    model, checkpoint, checkpoint_raw_sha256 = load_verified_checkpoint(
        checkpoint_path,
        expected_training_precommit_record_digest=(
            expected_training_precommit_record_digest
        ),
        expected_training_result_record_digest=expected_training_result_record_digest,
    )
    reconstructed = predict_raw_evidence(
        model,
        panel_ids=panel_ids,
        panel_png_bytes=panel_png_bytes,
        checkpoint_state_dict_sha256=checkpoint["state_dict_sha256"],
        config_digest=checkpoint["config_digest"],
    )
    if tuple(dict(item) for item in archived) != reconstructed:
        raise TinyLocalObserverError("raw evidence exact inference replay differs")
    return {
        "checkpoint_raw_sha256": checkpoint_raw_sha256,
        "checkpoint_state_dict_sha256": checkpoint["state_dict_sha256"],
        "exact_raw_observation_count": len(reconstructed),
        "pixel_bytes_redecoded": sum(len(raw) for raw in panel_png_bytes),
        "raw_evidence_exact": True,
    }


def _validate_probability_vector(value: object, size: int, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise TinyLocalObserverError(f"{label} cardinality differs")
    numbers = [_target_number(item, label=label, lower=0.0, upper=1.0) for item in value]
    if not math.isclose(sum(numbers), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise TinyLocalObserverError(f"{label} does not sum to one")
    return numbers


def _joint_triples(raw: Mapping[str, Any]) -> list[tuple[tuple[int, int, int], float]]:
    """Parse a raw observation into the 165 fixed count/catalog hypotheses."""

    counts = raw.get("joint_count_probabilities_straight_rows_arc_columns")
    if not isinstance(counts, list) or len(counts) != 10:
        raise TinyLocalObserverError("joint count matrix has wrong rows")
    count_values: list[list[float]] = []
    for straight, row in enumerate(counts):
        if not isinstance(row, list) or len(row) != 10:
            raise TinyLocalObserverError("joint count matrix has wrong columns")
        parsed = [
            _target_number(
                item,
                label=f"joint count ({straight},{arc})",
                lower=0.0,
                upper=1.0,
            )
            for arc, item in enumerate(row)
        ]
        for arc, probability in enumerate(parsed):
            if straight + arc > MAX_ACTION_SLOTS and probability != 0.0:
                raise TinyLocalObserverError("impossible count cell has nonzero mass")
        count_values.append(parsed)
    if not math.isclose(
        sum(sum(row) for row in count_values), 1.0, rel_tol=0.0, abs_tol=1e-5
    ):
        raise TinyLocalObserverError("whole joint count matrix does not sum to one")
    catalog = _validate_probability_vector(
        raw.get("catalog_probabilities"), 3, label="catalog probabilities"
    )
    triples = [
        ((straight, arc, catalog_index), count_values[straight][arc] * catalog_value)
        for straight in COUNT_DOMAIN
        for arc in COUNT_DOMAIN
        if straight + arc <= MAX_ACTION_SLOTS
        for catalog_index, catalog_value in enumerate(catalog)
    ]
    if not math.isclose(
        sum(probability for _triple, probability in triples),
        1.0,
        rel_tol=0.0,
        abs_tol=2e-5,
    ):
        raise TinyLocalObserverError("joint count/catalog hypotheses do not sum to one")
    return triples


def _rank_mass_before(
    hypotheses: Sequence[tuple[tuple[int, int, int], float]],
) -> dict[tuple[int, int, int], float]:
    """APS-style deterministic rank score; the most likely label scores zero."""

    ordered = sorted(hypotheses, key=lambda item: (-item[1], item[0]))
    cumulative = 0.0
    scores: dict[tuple[int, int, int], float] = {}
    for triple, probability in ordered:
        scores[triple] = cumulative
        cumulative += probability
    return scores


def _record_body(value: Mapping[str, Any], *, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != schema:
        raise TinyLocalObserverError(f"{label} schema differs")
    body = dict(value)
    found = body.pop("record_digest", None)
    if found != "sha256:" + canonical_digest(body):
        raise TinyLocalObserverError(f"{label} digest differs")
    return body


def fit_joint_calibrator(
    raw_observations: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    calibration_manifest_record_digest: str,
) -> dict[str, Any]:
    """Fit one joint split-conformal threshold after the external label barrier."""

    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise TinyLocalObserverError("alpha must lie strictly inside (0,1)")
    if not raw_observations or len(raw_observations) != len(truth_rows):
        raise TinyLocalObserverError("calibration observations/truth cardinality differs")
    checkpoint = None
    config = None
    scores: list[float] = []
    raw_digests: list[str] = []
    for index, (raw, truth) in enumerate(zip(raw_observations, truth_rows)):
        _record_body(raw, schema=RAW_OBSERVATION_SCHEMA, label=f"raw observation {index}")
        if not isinstance(truth, Mapping) or truth.get("panel_id") != raw.get("panel_id"):
            raise TinyLocalObserverError("calibration truth order differs")
        straight = truth.get("straight_action_count")
        arc = truth.get("arc_action_count")
        catalog_index = truth.get("catalog_class_index")
        if (
            isinstance(straight, bool)
            or not isinstance(straight, int)
            or isinstance(arc, bool)
            or not isinstance(arc, int)
            or isinstance(catalog_index, bool)
            or not isinstance(catalog_index, int)
            or straight not in COUNT_DOMAIN
            or arc not in COUNT_DOMAIN
            or straight + arc > MAX_ACTION_SLOTS
            or catalog_index not in CATALOG_DOMAIN
        ):
            raise TinyLocalObserverError("calibration truth leaves closed domains")
        if checkpoint is None:
            checkpoint = raw.get("checkpoint_state_dict_sha256")
            config = raw.get("config_digest")
        elif (
            raw.get("checkpoint_state_dict_sha256") != checkpoint
            or raw.get("config_digest") != config
        ):
            raise TinyLocalObserverError("calibration model binding differs")
        hypothesis_scores = _rank_mass_before(_joint_triples(raw))
        scores.append(hypothesis_scores[(straight, arc, catalog_index)])
        raw_digests.append(str(raw.get("record_digest")))
    sample_count = len(scores)
    order = math.ceil((sample_count + 1) * (1.0 - float(alpha)))
    threshold = 1.0 if order > sample_count else sorted(scores)[order - 1]
    body = {
        "alpha": float(alpha),
        "architecture_id": ARCHITECTURE_ID,
        "calibration_manifest_record_digest": calibration_manifest_record_digest,
        "calibration_raw_observation_record_digests": raw_digests,
        "checkpoint_state_dict_sha256": checkpoint,
        "config_digest": config,
        "finite_sample_order_statistic": order,
        "joint_candidate_domain": (
            "straight_0..9_x_arc_0..9_with_sum_at_most_9_x_catalog_0..2"
        ),
        "method": "split_conformal_rank_probability_mass_before_label",
        "sample_count": sample_count,
        "schema": CALIBRATION_SCHEMA,
        "source_sha256": source_sha256(),
        "threshold_q": threshold,
    }
    return _seal(body)


def apply_joint_calibrator(
    raw: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    """Project one archived raw observation to a finite calibrated set."""

    _record_body(raw, schema=RAW_OBSERVATION_SCHEMA, label="raw observation")
    _record_body(calibration, schema=CALIBRATION_SCHEMA, label="calibration")
    if (
        raw.get("checkpoint_state_dict_sha256")
        != calibration.get("checkpoint_state_dict_sha256")
        or raw.get("config_digest") != calibration.get("config_digest")
        or calibration.get("source_sha256") != source_sha256()
    ):
        raise TinyLocalObserverError("raw observation/calibration binding differs")
    threshold = _target_number(
        calibration.get("threshold_q"), label="calibration q", lower=0.0, upper=1.0
    )
    scores = _rank_mass_before(_joint_triples(raw))
    candidates = tuple(sorted(triple for triple, score in scores.items() if score <= threshold))
    if not candidates:
        raise TinyLocalObserverError("calibration produced an empty candidate set")
    straight = sorted({triple[0] for triple in candidates})
    arc = sorted({triple[1] for triple in candidates})
    catalog = sorted({triple[2] for triple in candidates})
    body = {
        "arc_action_count_candidates": arc,
        "calibration_record_digest": calibration["record_digest"],
        "catalog_class_candidates": catalog,
        "catalog_class_names": [CATALOG_CLASSES[index] for index in catalog],
        "disposition": "calibrated_set",
        "joint_straight_arc_catalog_candidates": [list(item) for item in candidates],
        "panel_id": raw["panel_id"],
        "png_sha256": raw["png_sha256"],
        "raw_observation_record_digest": raw["record_digest"],
        "schema": CALIBRATED_OBSERVATION_SCHEMA,
        "source_sha256": source_sha256(),
        "straight_action_count_candidates": straight,
    }
    return _seal(body)


def cold_replay_observation(
    raw: Mapping[str, Any],
    calibration: Mapping[str, Any],
    archived: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay candidate construction without pixels, torch, or a model call."""

    _record_body(
        archived, schema=CALIBRATED_OBSERVATION_SCHEMA, label="calibrated observation"
    )
    reconstructed = apply_joint_calibrator(raw, calibration)
    if reconstructed != dict(archived):
        raise TinyLocalObserverError("model-free calibrated reconstruction differs")
    body = {
        "archived_calibrated_record_digest": archived["record_digest"],
        "calibration_record_digest": calibration["record_digest"],
        "candidate_set_exact": True,
        "model_calls": 0,
        "pixel_reads": 0,
        "raw_observation_record_digest": raw["record_digest"],
        "schema": REPLAY_SCHEMA,
        "source_sha256": source_sha256(),
    }
    return _seal(body)


def runtime_work_bound(
    *,
    training_occurrences: int,
    validation_occurrences: int,
    measured_seconds_per_frozen_batch: float,
) -> dict[str, Any]:
    """Refuse projected work above seven minutes before any development PNG read."""

    for value, label in (
        (training_occurrences, "training occurrences"),
        (validation_occurrences, "validation occurrences"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TinyLocalObserverError(f"{label} must be a positive integer")
    batch_size = int(PROTOCOL["batch_size"])
    epochs = int(PROTOCOL["epochs"])
    training_batches = math.ceil(training_occurrences / batch_size) * epochs
    validation_batches = math.ceil(validation_occurrences / batch_size) * (epochs + 2)
    optimizer_steps = training_batches
    if optimizer_steps > int(PROTOCOL["maximum_optimizer_steps"]):
        raise RuntimeBudgetExceeded("static optimizer-step limit exceeded")
    seconds = _target_number(
        measured_seconds_per_frozen_batch,
        label="measured seconds per frozen batch",
        lower=0.0,
        upper=float(PROTOCOL["maximum_wall_runtime_seconds"]),
    )
    # Includes validation/replay batches and a 3x margin for I/O and matching.
    projected = (training_batches + validation_batches) * seconds * 3.0
    if projected > float(PROTOCOL["maximum_projected_runtime_seconds"]):
        raise RuntimeBudgetExceeded("synthetic preflight projects over seven minutes")
    return {
        "batch_size": batch_size,
        "epochs": epochs,
        "maximum_optimizer_steps": int(PROTOCOL["maximum_optimizer_steps"]),
        "maximum_projected_runtime_seconds": float(
            PROTOCOL["maximum_projected_runtime_seconds"]
        ),
        "maximum_wall_runtime_seconds": float(PROTOCOL["maximum_wall_runtime_seconds"]),
        "measured_seconds_per_frozen_batch": seconds,
        "optimizer_steps": optimizer_steps,
        "parameter_count": parameter_count(),
        "passed": True,
        "projected_runtime_seconds_with_3x_margin": projected,
        "training_batches": training_batches,
        "validation_and_replay_batches": validation_batches,
    }


def synthetic_runtime_probe(*, repetitions: int = 3) -> dict[str, Any]:
    """Measure exact frozen-batch forward/matching/backward work on synthetic data."""

    torch = _configure_torch(int(PROTOCOL["random_seed"]))
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions not in range(1, 6):
        raise TinyLocalObserverError("runtime-probe repetitions must be 1..5")
    model = build_model(seed=int(PROTOCOL["random_seed"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(PROTOCOL["learning_rate"]),
        weight_decay=float(PROTOCOL["weight_decay"]),
    )
    batch = int(PROTOCOL["batch_size"])
    generator = torch.Generator().manual_seed(91)
    pixels = torch.rand((batch, 2, 64, 64), generator=generator)
    def interval(center: float, radius: float = 0.01) -> dict[str, float]:
        return {
            "center": center,
            "lower": max(0.0, center - radius),
            "upper": min(1.0, center + radius),
        }

    target = (
        {
            "kind": "line",
            "line_length": interval(0.5),
            "shape_membership_local_index": 0,
        },
        {
            "arc_radius": interval(0.4),
            "arc_sweep_magnitude": interval(0.5),
            "kind": "arc",
            "shape_membership_local_index": 0,
        },
    )
    targets = [target] * batch
    catalog = torch.tensor([index % 3 for index in range(batch)], dtype=torch.long)
    durations = []
    for iteration in range(repetitions + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = set_prediction_loss(model(pixels), targets, catalog)
        losses["total"].backward()
        optimizer.step()
        elapsed = time.monotonic() - started
        if iteration > 0:
            durations.append(elapsed)
    return {
        "frozen_batch_size": batch,
        "maximum_seconds_per_batch_for_launch": (
            float(PROTOCOL["maximum_projected_runtime_seconds"])
            / (3.0 * (528 + math.ceil(1_392 / batch) * 8))
        ),
        "median_seconds_per_frozen_batch": sorted(durations)[len(durations) // 2],
        "parameter_count": parameter_count(model),
        "repetitions": repetitions,
        "synthetic_only": True,
    }


class WallDeadline:
    """Hard batch-boundary deadline; callers must check before and after every batch."""

    def __init__(self, *, seconds: float | None = None) -> None:
        limit = float(PROTOCOL["maximum_wall_runtime_seconds"] if seconds is None else seconds)
        if not 0.0 < limit <= float(PROTOCOL["maximum_wall_runtime_seconds"]):
            raise TinyLocalObserverError("wall deadline leaves frozen bound")
        self._limit = limit
        self._started = time.monotonic()

    def check(self) -> None:
        if time.monotonic() - self._started > self._limit:
            raise RuntimeBudgetExceeded("hard ten-minute wall deadline exceeded")


def _authority_interval(
    *,
    center_integer: object,
    interval: object,
    center_denominator: int,
    interval_denominator: int,
    expected_unit: str,
    label: str,
) -> dict[str, float]:
    if isinstance(center_integer, bool) or not isinstance(center_integer, int):
        raise TinyLocalObserverError(f"{label} source center is not an integer")
    if not isinstance(interval, Mapping) or set(interval) != {"lower", "upper", "unit"}:
        raise TinyLocalObserverError(f"{label} authority interval fields differ")
    if interval.get("unit") != expected_unit:
        raise TinyLocalObserverError(f"{label} authority interval unit differs")
    lower_integer, upper_integer = interval.get("lower"), interval.get("upper")
    if (
        isinstance(lower_integer, bool)
        or not isinstance(lower_integer, int)
        or isinstance(upper_integer, bool)
        or not isinstance(upper_integer, int)
    ):
        raise TinyLocalObserverError(f"{label} authority bounds are not integers")
    value = {
        "center": center_integer / center_denominator,
        "lower": lower_integer / interval_denominator,
        "upper": upper_integer / interval_denominator,
    }
    return _normalized_interval(value, label=label)


def authority_panel_targets(supervision: Any) -> tuple[dict[str, Any], ...]:
    """Convert certified authority multisets; GAP never becomes none or count zero."""

    data = supervision.to_data() if hasattr(supervision, "to_data") else supervision
    if not isinstance(data, Mapping) or data.get("schema") != (
        "gkm.bongard-pose-free-local-action-supervision.v1"
    ):
        raise TinyLocalObserverError("local supervision schema differs")
    if data.get("disposition") != "CERTIFIED":
        gap = data.get("gap")
        code = gap.get("code") if isinstance(gap, Mapping) else "unknown"
        raise TinyLocalObserverError(f"authority_GAP:{code}")
    for name in ("pixel_registration", "pixel_instance_assignment"):
        value = data.get(name)
        if not isinstance(value, Mapping) or value.get("disposition") != "GAP":
            raise TinyLocalObserverError(f"{name} must remain an explicit GAP")
    shape_rows = data.get("shape_multiset")
    if not isinstance(shape_rows, list) or not shape_rows:
        raise TinyLocalObserverError("authority shape multiset is empty")
    shapes: list[list[dict[str, Any]]] = []
    for shape_row in shape_rows:
        if not isinstance(shape_row, Mapping):
            raise TinyLocalObserverError("authority shape row is invalid")
        shape_multiplicity = shape_row.get("multiplicity")
        action_rows = shape_row.get("action_multiset")
        if (
            isinstance(shape_multiplicity, bool)
            or not isinstance(shape_multiplicity, int)
            or shape_multiplicity <= 0
            or not isinstance(action_rows, list)
            or not action_rows
        ):
            raise TinyLocalObserverError("authority shape multiplicity differs")
        expanded_shape: list[dict[str, Any]] = []
        for action in action_rows:
            if not isinstance(action, Mapping):
                raise TinyLocalObserverError("authority action row is invalid")
            multiplicity = action.get("multiplicity")
            if (
                isinstance(multiplicity, bool)
                or not isinstance(multiplicity, int)
                or multiplicity <= 0
            ):
                raise TinyLocalObserverError("authority action multiplicity differs")
            primitive = action.get("primitive")
            if primitive == "line":
                target = {
                    "kind": "line",
                    "line_length": _authority_interval(
                        center_integer=action.get("length_source_normalized_milli"),
                        interval=action.get("length_normalized_micro_interval"),
                        center_denominator=1_000,
                        interval_denominator=1_000_000,
                        expected_unit="normalized_micro",
                        label="line length",
                    ),
                }
            elif primitive == "arc":
                target = {
                    "arc_radius": _authority_interval(
                        center_integer=action.get("radius_source_normalized_milli"),
                        interval=action.get("radius_normalized_micro_interval"),
                        center_denominator=1_000,
                        interval_denominator=1_000_000,
                        expected_unit="normalized_micro",
                        label="arc radius",
                    ),
                    "arc_sweep_magnitude": _authority_interval(
                        center_integer=action.get(
                            "sweep_magnitude_source_degrees_milli"
                        ),
                        interval=action.get(
                            "sweep_magnitude_degrees_milli_interval"
                        ),
                        center_denominator=360_000,
                        interval_denominator=360_000,
                        expected_unit="degree_milli",
                        label="arc sweep magnitude",
                    ),
                    "kind": "arc",
                }
            else:
                raise TinyLocalObserverError("authority primitive is unsupported")
            expanded_shape.extend(dict(target) for _ in range(multiplicity))
        if len(expanded_shape) != shape_row.get("action_count"):
            raise TinyLocalObserverError("authority action count/multiset differs")
        shapes.extend([expanded_shape] * shape_multiplicity)
    flattened = flatten_pose_free_shapes(shapes)
    carrier = data.get("carrier_instance_count")
    if (
        not isinstance(carrier, Mapping)
        or carrier.get("disposition") != "CERTIFIED"
        or carrier.get("value") != len(flattened)
    ):
        raise TinyLocalObserverError("authority carrier count differs")
    return flattened


def audit_supervision_coverage(supervisions: Iterable[Any]) -> dict[str, Any]:
    """Count every GAP/capacity/interval row before any training run."""

    cohort_counts = {"train": 0, "validation": 0}
    eligible = {"train": 0, "validation": 0}
    action_histogram = {"train": {}, "validation": {}}
    gap_codes: dict[str, int] = {}
    interval_supervised = 0
    authority_digests: set[str] = set()
    panel_ids: set[str] = set()
    for supervision in supervisions:
        data = supervision.to_data() if hasattr(supervision, "to_data") else supervision
        if not isinstance(data, Mapping):
            raise TinyLocalObserverError("supervision coverage row is invalid")
        panel_id, cohort = data.get("panel_id"), data.get("cohort")
        if not isinstance(panel_id, str) or panel_id in panel_ids or cohort not in cohort_counts:
            raise TinyLocalObserverError("supervision panel identity differs")
        panel_ids.add(panel_id)
        cohort_counts[cohort] += 1
        digest = data.get("authority_record_digest")
        if not isinstance(digest, str):
            raise TinyLocalObserverError("supervision authority digest is missing")
        authority_digests.add(digest)
        if data.get("disposition") != "CERTIFIED":
            gap = data.get("gap")
            code = gap.get("code") if isinstance(gap, Mapping) else "malformed_GAP"
            gap_codes[str(code)] = gap_codes.get(str(code), 0) + 1
            # Crucial: do not call authority_panel_targets, and hence do not
            # synthesize `none` slots or a zero count for this row.
            continue
        try:
            targets = authority_panel_targets(data)
        except TinyLocalObserverError as exc:
            code = "capacity_GAP" if "capacity_GAP" in str(exc) else "consumer_GAP"
            gap_codes[code] = gap_codes.get(code, 0) + 1
            continue
        eligible[cohort] += 1
        interval_supervised += 1
        count = len(targets)
        histogram = action_histogram[cohort]
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    if len(authority_digests) != 1:
        raise TinyLocalObserverError("coverage mixes supervision authorities")
    return {
        "action_count_histogram": action_histogram,
        "authority_record_digest": next(iter(authority_digests)),
        "capacity_gap_panel_count": gap_codes.get("capacity_GAP", 0),
        "certified_interval_supervised_panel_count": interval_supervised,
        "cohort_panel_counts": cohort_counts,
        "eligible_panel_counts": eligible,
        "gap_code_counts": dict(sorted(gap_codes.items())),
        "gap_rows_coerced_to_none_or_zero": 0,
        "panel_count": len(panel_ids),
        "pixel_aligned_targets_created": 0,
        "scalar_midpoints_substituted_for_intervals": 0,
    }


def audit_descriptor_target_conflicts(
    fit_precommit: Mapping[str, Any], supervisions: Iterable[Any]
) -> dict[str, Any]:
    """GAP descriptor loss for identical PNGs with conflicting action multisets."""

    by_panel: dict[str, str | None] = {}
    for supervision in supervisions:
        data = supervision.to_data() if hasattr(supervision, "to_data") else supervision
        if not isinstance(data, Mapping) or not isinstance(data.get("panel_id"), str):
            raise TinyLocalObserverError("descriptor-conflict supervision row is invalid")
        panel_id = data["panel_id"]
        if panel_id in by_panel:
            raise TinyLocalObserverError("descriptor-conflict panel ID repeats")
        if data.get("disposition") != "CERTIFIED":
            by_panel[panel_id] = None
            continue
        targets = authority_panel_targets(data)
        # Membership indices are local expansion bookkeeping, not authority.
        digest_target = [
            {key: value for key, value in target.items() if key != "shape_membership_local_index"}
            for target in targets
        ]
        by_panel[panel_id] = "sha256:" + canonical_digest(digest_target)
    groups = fit_precommit.get("path_independent_digest_groups")
    if not isinstance(groups, list):
        raise TinyLocalObserverError("fit precommit digest groups are missing")
    conflict_groups = 0
    conflict_occurrences = 0
    authority_gap_occurrences = 0
    descriptor_eligible = 0
    total = 0
    conflict_png_digests: list[str] = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("panel_ids"), list):
            raise TinyLocalObserverError("fit digest group is invalid")
        panel_ids = group["panel_ids"]
        if group.get("multiplicity") != len(panel_ids):
            raise TinyLocalObserverError("fit digest group multiplicity differs")
        values = []
        for panel_id in panel_ids:
            if panel_id not in by_panel:
                raise TinyLocalObserverError("fit panel lacks local supervision")
            values.append(by_panel[panel_id])
        total += len(values)
        if any(value is None for value in values):
            authority_gap_occurrences += len(values)
        elif len(set(values)) != 1:
            conflict_groups += 1
            conflict_occurrences += len(values)
            conflict_png_digests.append(str(group.get("png_sha256")))
        else:
            descriptor_eligible += len(values)
    if total != 12_592:
        raise TinyLocalObserverError("effective descriptor audit occurrence count differs")
    return {
        "action_descriptor_loss_policy": (
            "apply_only_when_all_same-PNG_occurrences_have_one_exact_target_digest;_"
            "otherwise_descriptor_GAP_while_count_and_catalog_losses_remain"
        ),
        "authority_gap_occurrences": authority_gap_occurrences,
        "count_and_catalog_supervision_occurrences": total,
        "descriptor_conflict_group_count": conflict_groups,
        "descriptor_conflict_occurrences": conflict_occurrences,
        "descriptor_eligible_occurrences": descriptor_eligible,
        "descriptor_gap_is_never_none_or_zero": True,
        "effective_occurrence_count": total,
        "sorted_conflict_png_sha256s": sorted(conflict_png_digests),
    }


def create_successor_precommit(
    *,
    failed_baseline_path: Path,
    retired_spatial_outcome_path: Path,
    fit_precommit_path: Path,
    supervision_authority_record: Mapping[str, Any],
    supervision_coverage: Mapping[str, Any],
    descriptor_conflict_audit: Mapping[str, Any],
    runtime_probe: Mapping[str, Any],
    intended_checkpoint: Path,
    intended_result: Path,
    output: Path,
) -> dict[str, Any]:
    """Seal the exact tiny successor and work bounds without opening a PNG."""

    baseline = _load_record(failed_baseline_path, label="failed baseline")
    spatial = _load_record(retired_spatial_outcome_path, label="retired spatial outcome")
    fit_precommit = _load_record(fit_precommit_path, label="fit precommit")
    if baseline.get("record_digest") != FAILED_BASELINE_DIGEST:
        raise TinyLocalObserverError("failed baseline binding differs")
    if spatial.get("record_digest") != RETIRED_SPATIAL_OUTCOME_DIGEST:
        raise TinyLocalObserverError("retired spatial outcome binding differs")
    if fit_precommit.get("record_digest") != (
        "sha256:e8c7c15fbfb723c5b2305094f035e2567c1fb9b7e80b9f13eeae32fe35d1b15a"
    ):
        raise TinyLocalObserverError("fit precommit binding differs")
    authority_body = _record_body(
        supervision_authority_record,
        schema="gkm.bongard-pose-free-local-action-authority.v1",
        label="local supervision authority",
    )
    expected_full_counts = {"train": 11_200, "validation": 1_400}
    if (
        supervision_coverage.get("authority_record_digest")
        != supervision_authority_record.get("record_digest")
        or supervision_coverage.get("cohort_panel_counts") != expected_full_counts
        or supervision_coverage.get("eligible_panel_counts") != expected_full_counts
        or supervision_coverage.get("panel_count") != 12_600
        or supervision_coverage.get("capacity_gap_panel_count") != 0
        or supervision_coverage.get("gap_code_counts") != {}
        or supervision_coverage.get("gap_rows_coerced_to_none_or_zero") != 0
        or supervision_coverage.get("pixel_aligned_targets_created") != 0
        or supervision_coverage.get("scalar_midpoints_substituted_for_intervals") != 0
    ):
        raise TinyLocalObserverError("supervision coverage is not launch-complete")
    if (
        fit_precommit.get("duplicate_digest_audit", {}).get(
            "effective_training_panel_count"
        )
        != 11_200
        or fit_precommit.get("effective_validation_panel_count") != 1_392
        or fit_precommit.get("validation_removed_due_exact_train_duplicate", {}).get(
            "panel_count"
        )
        != 8
    ):
        raise TinyLocalObserverError("decontaminated fit custody differs")
    if (
        descriptor_conflict_audit.get("effective_occurrence_count") != 12_592
        or descriptor_conflict_audit.get("count_and_catalog_supervision_occurrences")
        != 12_592
        or descriptor_conflict_audit.get("descriptor_gap_is_never_none_or_zero")
        is not True
        or descriptor_conflict_audit.get("descriptor_eligible_occurrences", -1)
        + descriptor_conflict_audit.get("descriptor_conflict_occurrences", -1)
        + descriptor_conflict_audit.get("authority_gap_occurrences", -1)
        != 12_592
    ):
        raise TinyLocalObserverError("descriptor target-conflict audit differs")
    if (
        runtime_probe.get("synthetic_only") is not True
        or runtime_probe.get("frozen_batch_size") != int(PROTOCOL["batch_size"])
        or runtime_probe.get("parameter_count") != parameter_count()
    ):
        raise TinyLocalObserverError("synthetic runtime probe differs")
    work = runtime_work_bound(
        training_occurrences=11_200,
        validation_occurrences=1_392,
        measured_seconds_per_frozen_batch=float(
            runtime_probe["median_seconds_per_frozen_batch"]
        ),
    )
    body = {
        "architecture_id": ARCHITECTURE_ID,
        "authorized_input": "already_exposed_development_only",
        "config_digest": successor_config_digest(),
        "decontaminated_occurrence_counts": {"train": 11_200, "validation": 1_392},
        "failed_baseline_record_digest": baseline["record_digest"],
        "fit_precommit_record_digest": fit_precommit["record_digest"],
        "descriptor_target_conflict_audit": json.loads(
            canonical_json(dict(descriptor_conflict_audit))
        ),
        "forbidden_cohorts": list(PROTOCOL["forbidden_cohorts"]),
        "intended_outputs": {
            "checkpoint": str(intended_checkpoint.resolve()),
            "result": str(intended_result.resolve()),
        },
        "pixels_read_by_precommit": 0,
        "protocol": json.loads(canonical_json(dict(PROTOCOL))),
        "retired_spatial_outcome_record_digest": spatial["record_digest"],
        "runtime_probe": dict(runtime_probe),
        "runtime_work_bound": work,
        "schema": PRECOMMIT_SCHEMA,
        "source_sha256": source_sha256(),
        "supervision_authority_body_digest": "sha256:" + canonical_digest(authority_body),
        "supervision_authority_record_digest": supervision_authority_record[
            "record_digest"
        ],
        "supervision_coverage": json.loads(canonical_json(dict(supervision_coverage))),
        "training_entrypoint_status": "infrastructure_only_not_yet_live_runnable",
    }
    value = _seal(body)
    _write_once(output, value)
    if _load_record(output, label="tiny successor precommit") != value:
        raise TinyLocalObserverError("tiny successor precommit fresh load differs")
    return value
