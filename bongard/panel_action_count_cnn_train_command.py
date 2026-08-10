"""Deterministic, CPU-only train/validation command for the typed CNN.

This executable intentionally has no calibration or evaluation command.  The
v3 plan and identifier-only development manifest are the execution cohort
authority.  The frozen v2 plan supplies only the retained training protocol,
and the frozen v2 development labels supply supervised targets for the exact
v3-retained train and validation IDs.  No old or fresh calibration/evaluation
cohort is reachable here.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
from io import BytesIO
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from bongard.canonical import canonical_digest, canonical_json


V3_PLAN_SCHEMA = "gkm.bongard-action-count-catalog-cnn-preregistration.v3"
EXPECTED_V3_PLAN_RECORD_DIGEST = (
    "sha256:bb4524a0958cd21f2d4d49bc6a9caa964ccb96c67fbf7c6192185f7b2f363dcb"
)
EXPECTED_V3_PLAN_SOURCE_SHA256 = (
    "sha256:71c68771b356658843c3d848cdeea0ba7f2d96fffacd1816ef72934214b055d0"
)
EXPECTED_V3_PLAN_CLAIM = (
    "metadata-only-v3-cohort-repair;_no_selected-pixel-or-fresh-target-access"
)
V3_DEVELOPMENT_SCHEMA = (
    "gkm.bongard-action-count-cnn-development-panel-ids.v3"
)
EXPECTED_V3_DEVELOPMENT_RECORD_DIGEST = (
    "sha256:ee02e48ea3e07dd4804ad24e5c1c9228addc4a0fe658efe821993451bc749fde"
)
EXPECTED_V3_DEVELOPMENT_SOURCE_SHA256 = (
    "sha256:9f0c8957bd1be7885022c0bf12d8104c531eea36b1680b902406c1b5e39923db"
)
EXPECTED_V2_PLAN_RECORD_DIGEST = (
    "sha256:0de57e610763a7fb77adbcaeb2be21b20864a02eb5af0656b76c291ef5b0a3a8"
)
EXPECTED_V2_PLAN_SOURCE_SHA256 = (
    "sha256:b38fd75badd5090dc03fe8cecb34053d4045bc9a47e3a4b9ad2b7e433aa0ca5b"
)
EXPECTED_V2_DEVELOPMENT_RECORD_DIGEST = (
    "sha256:c72d09eaa2bee02572694dacdb48ec80d2e23615c1c54f4c6616136b235b3d52"
)
EXPECTED_V2_DEVELOPMENT_SOURCE_SHA256 = (
    "sha256:913c826d6be2aca47610b771c57859053d985640dca9a8ce9ea01c663a701333"
)
EXPECTED_V2_PLAN_CLAIM = (
    "oracle-supervised-exact-unused-official-TRAIN-representation-engineering-"
    "not-bongard-benchmark"
)
V2_DEVELOPMENT_SCHEMA = "gkm.bongard-action-count-catalog-cnn-development-labels.v2"
AUTHORIZATION_SCHEMA = (
    "gkm.bongard-action-count-catalog-cnn-fit-exposure-authorization.v1"
)
PRECOMMIT_SCHEMA = "gkm.bongard-action-count-catalog-cnn-fit-pixel-precommit.v1"
TRAINING_SCHEMA = "gkm.bongard-action-count-catalog-cnn-fit-result.v1"
REPLAY_SCHEMA = "gkm.bongard-action-count-catalog-cnn-fit-replay.v1"
ARCHITECTURE_ID = "shared-cnn-16-32-64-96-three-head/v1"
CATALOG_VALUES = (-1, 0, 1)
CATALOG_TO_INDEX = {-1: 0, 0: 1, 1: 2}
_SHA256_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
FIT_CORRECTION = MappingProxyType({
    "fresh_v3_calibration_or_evaluation_authorized_by_this_command": False,
    "old_v2_calibration_and_evaluation_are_design_tainted": True,
    "old_v2_calibration_or_evaluation_authorized_by_this_command": False,
    "v2_development_labels_are_supervised_target_authority_only": True,
    "v2_plan_is_training_protocol_authority_only": True,
    "v3_plan_and_development_ids_are_execution_cohort_authority": True,
})
EXECUTION_PROTOCOL = MappingProxyType(
    {
        "batch_size": 64,
        "cpu_threads": 1,
        "epochs": 16,
        "image_size": 96,
        "learning_rate": 0.001,
        "optimizer": "AdamW",
        "optimizer_betas": (0.9, 0.999),
        "optimizer_eps": 1e-08,
        "random_seed": 260810,
        "weight_decay": 0.0001,
    }
)


class ActionCountCNNFitError(RuntimeError):
    """Fit custody, deterministic runtime, data, or replay differs."""


def _address(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _execution_protocol_data() -> dict[str, Any]:
    return json.loads(canonical_json(dict(EXECUTION_PROTOCOL)))


def _load_record(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActionCountCNNFitError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ActionCountCNNFitError(f"{label} must be a JSON object")
    if raw != canonical_json(value) + b"\n":
        raise ActionCountCNNFitError(f"{label} is not canonical JSON plus newline")
    digest = value.get("record_digest")
    body = dict(value)
    body.pop("record_digest", None)
    if digest != "sha256:" + canonical_digest(body):
        raise ActionCountCNNFitError(f"{label} record digest differs")
    return value, raw


def _write_fsynced(path: Path, value: Mapping[str, Any]) -> bytes:
    payload = canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise ActionCountCNNFitError(f"cannot verify existing {path}: {exc}") from exc
        if existing == payload:
            return payload
        raise ActionCountCNNFitError(f"refusing to overwrite nonidentical artifact {path}")
    temporary = path.with_name(path.name + ".tmp-action-count-cnn-fit")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
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
        raise ActionCountCNNFitError(f"cannot durably write {path}: {exc}") from exc
    reloaded = path.read_bytes()
    if reloaded != payload:
        raise ActionCountCNNFitError("durable artifact fresh-load differs")
    return payload


def _seal_body(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def _runtime_identity() -> dict[str, Any]:
    try:
        versions = {
            name: importlib.metadata.version(distribution)
            for name, distribution in (
                ("numpy", "numpy"),
                ("pillow", "Pillow"),
                ("torch", "torch"),
            )
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise ActionCountCNNFitError(f"training dependency is unavailable: {exc}") from exc
    try:
        import numpy
        import PIL
        torch = _configure_torch(int(EXECUTION_PROTOCOL["random_seed"]))

        module_sources = {
            name: _address(Path(module.__file__).resolve().read_bytes())
            for name, module in (("numpy", numpy), ("pillow", PIL), ("torch", torch))
        }
        torch_build = _address(torch.__config__.show().encode("utf-8"))
    except (OSError, AttributeError) as exc:
        raise ActionCountCNNFitError(f"cannot bind dependency runtime: {exc}") from exc
    return {
        **versions,
        "dependency_entry_source_sha256": module_sources,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_build_config_sha256": torch_build,
        "torch_git_version": torch.version.git_version,
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


@dataclass(frozen=True)
class FitAuthorities:
    v3_plan: dict[str, Any]
    v3_plan_raw: bytes
    v3_development: dict[str, Any]
    v3_development_raw: bytes
    v2_protocol_plan: dict[str, Any]
    v2_protocol_plan_raw: bytes
    v2_development_labels: dict[str, Any]
    v2_development_labels_raw: bytes


def _verify_execution_protocol(protocol_plan: Mapping[str, Any]) -> None:
    training = protocol_plan.get("training_protocol")
    if not isinstance(training, dict):
        raise ActionCountCNNFitError("v2 training protocol is missing")
    scalar_fields = {
        "batch_size": EXECUTION_PROTOCOL["batch_size"],
        "cpu_threads": EXECUTION_PROTOCOL["cpu_threads"],
        "epochs": EXECUTION_PROTOCOL["epochs"],
        "image_size": EXECUTION_PROTOCOL["image_size"],
        "learning_rate": EXECUTION_PROTOCOL["learning_rate"],
        "optimizer": EXECUTION_PROTOCOL["optimizer"],
        "random_seed": EXECUTION_PROTOCOL["random_seed"],
        "weight_decay": EXECUTION_PROTOCOL["weight_decay"],
    }
    if any(training.get(field) != value for field, value in scalar_fields.items()):
        raise ActionCountCNNFitError("executable protocol differs from v2 plan")
    if training.get("optimizer_parameters") != {
        "betas": list(EXECUTION_PROTOCOL["optimizer_betas"]),
        "eps": EXECUTION_PROTOCOL["optimizer_eps"],
        "weight_decay": EXECUTION_PROTOCOL["weight_decay"],
    }:
        raise ActionCountCNNFitError("optimizer parameters differ from v2 plan")
    if (
        training.get("heads")
        != {"arc": 10, "catalog_convexity": 3, "straight": 10}
        or training.get("catalog_head_class_order")
        != ["catalog_unresolved", "nonconvex", "convex"]
        or training.get("pretrained_or_network_weights") is not False
        or training.get("torch_deterministic_algorithms") is not True
    ):
        raise ActionCountCNNFitError("model/head protocol differs from v2 plan")


def _expected_panel_ids(task_ids: Sequence[str]) -> list[str]:
    return [
        f"hd/{task_id}/{folder}/{panel_index}.png"
        for task_id in task_ids
        for folder in (1, 0)
        for panel_index in range(7)
    ]


def _verify_fit_authorities(
    *,
    v3_plan_path: Path,
    v3_development_path: Path,
    v2_protocol_plan_path: Path,
    v2_development_labels_path: Path,
) -> FitAuthorities:
    v3_plan, v3_plan_raw = _load_record(v3_plan_path, label="v3 plan")
    v3_development, v3_development_raw = _load_record(
        v3_development_path, label="v3 development panel IDs"
    )
    v2_plan, v2_plan_raw = _load_record(
        v2_protocol_plan_path, label="v2 training-protocol plan"
    )
    v2_labels, v2_labels_raw = _load_record(
        v2_development_labels_path, label="v2 retained development labels"
    )
    if (
        v3_plan.get("schema") != V3_PLAN_SCHEMA
        or v3_plan.get("claim") != EXPECTED_V3_PLAN_CLAIM
        or v3_plan.get("record_digest") != EXPECTED_V3_PLAN_RECORD_DIGEST
        or _address(v3_plan_raw) != EXPECTED_V3_PLAN_SOURCE_SHA256
    ):
        raise ActionCountCNNFitError("fit execution plan is not the frozen v3 plan")
    if (
        v3_development.get("schema") != V3_DEVELOPMENT_SCHEMA
        or v3_development.get("record_digest")
        != EXPECTED_V3_DEVELOPMENT_RECORD_DIGEST
        or _address(v3_development_raw) != EXPECTED_V3_DEVELOPMENT_SOURCE_SHA256
    ):
        raise ActionCountCNNFitError("development IDs are not the frozen v3 manifest")
    v3_binding = v3_plan.get("identifier_manifest_bindings", {}).get(
        "development_panel_ids"
    )
    if not isinstance(v3_binding, dict) or v3_binding != {
        "path": "bongard/data/panel_action_count_cnn_development_panels_20260810_v3.json",
        "record_digest": EXPECTED_V3_DEVELOPMENT_RECORD_DIGEST,
        "source_sha256": EXPECTED_V3_DEVELOPMENT_SOURCE_SHA256,
    }:
        raise ActionCountCNNFitError("v3 development-ID binding differs")
    if (
        v2_plan.get("claim") != EXPECTED_V2_PLAN_CLAIM
        or v2_plan.get("record_digest") != EXPECTED_V2_PLAN_RECORD_DIGEST
        or _address(v2_plan_raw) != EXPECTED_V2_PLAN_SOURCE_SHA256
    ):
        raise ActionCountCNNFitError("training protocol is not the frozen v2 plan")
    v2_source_binding = v3_plan.get("metadata_source_bindings", {})
    if (
        v2_source_binding.get("v2_plan_record_digest")
        != EXPECTED_V2_PLAN_RECORD_DIGEST
        or v2_source_binding.get("v2_plan_source_sha256")
        != EXPECTED_V2_PLAN_SOURCE_SHA256
    ):
        raise ActionCountCNNFitError("v3 does not bind the supplied v2 protocol")
    v2_label_binding = v2_plan.get("manifest_bindings", {}).get(
        "development_labels"
    )
    if (
        v2_labels.get("schema") != V2_DEVELOPMENT_SCHEMA
        or v2_labels.get("record_digest")
        != EXPECTED_V2_DEVELOPMENT_RECORD_DIGEST
        or _address(v2_labels_raw) != EXPECTED_V2_DEVELOPMENT_SOURCE_SHA256
        or not isinstance(v2_label_binding, dict)
        or v2_label_binding.get("record_digest")
        != EXPECTED_V2_DEVELOPMENT_RECORD_DIGEST
        or v2_label_binding.get("source_sha256")
        != EXPECTED_V2_DEVELOPMENT_SOURCE_SHA256
    ):
        raise ActionCountCNNFitError("retained v2 development-label binding differs")
    if set(v3_development.get("cohorts", {})) != {"train", "validation"}:
        raise ActionCountCNNFitError("v3 development manifest exposes another cohort")
    if set(v2_labels.get("cohorts", {})) != {"train", "validation"}:
        raise ActionCountCNNFitError("v2 label manifest exposes another cohort")
    if set(v3_plan.get("cohorts", {})) != {
        "train",
        "validation",
        "calibration",
        "evaluation",
    }:
        raise ActionCountCNNFitError("v3 plan cohort structure differs")

    all_panel_ids: list[str] = []
    for cohort, task_count, panel_count in (
        ("train", 800, 11_200),
        ("validation", 100, 1_400),
    ):
        v3_cohort = v3_plan["cohorts"][cohort]
        id_cohort = v3_development["cohorts"][cohort]
        label_rows = v2_labels["cohorts"][cohort].get("rows")
        task_ids = id_cohort.get("task_ids") if isinstance(id_cohort, dict) else None
        panel_ids = id_cohort.get("panel_ids") if isinstance(id_cohort, dict) else None
        if (
            not isinstance(task_ids, list)
            or not isinstance(panel_ids, list)
            or any(not isinstance(value, str) for value in task_ids + panel_ids)
            or len(task_ids) != task_count
            or len(task_ids) != len(set(task_ids))
            or len(panel_ids) != panel_count
            or len(panel_ids) != len(set(panel_ids))
            or not isinstance(label_rows, list)
            or len(label_rows) != panel_count
        ):
            raise ActionCountCNNFitError(f"{cohort} retained inventory is invalid")
        expected_panels = _expected_panel_ids(task_ids)
        v2_panel_ids = [
            row.get("panel_id") if isinstance(row, dict) else None for row in label_rows
        ]
        task_digest = "sha256:" + canonical_digest(task_ids)
        panel_digest = "sha256:" + canonical_digest(panel_ids)
        if (
            panel_ids != expected_panels
            or v2_panel_ids != panel_ids
            or v3_cohort.get("task_ids") != task_ids
            or v3_cohort.get("task_count") != task_count
            or v3_cohort.get("panel_count") != panel_count
            or v3_cohort.get("task_ids_digest") != task_digest
            or v2_plan["cohorts"][cohort].get("task_ids") != task_ids
            or v2_plan["cohorts"][cohort].get("task_ids_digest") != task_digest
            or v2_plan["cohorts"][cohort].get("panel_ids_digest") != panel_digest
            or v2_plan["cohorts"][cohort].get("panel_count") != panel_count
            or v2_plan["cohorts"][cohort].get("action_and_catalog_label_rows_digest")
            != "sha256:" + canonical_digest(label_rows)
        ):
            raise ActionCountCNNFitError(
                f"{cohort} IDs/order/digests differ across v3 and retained v2"
            )
        all_panel_ids.extend(panel_ids)
    if len(all_panel_ids) != 12_600 or len(set(all_panel_ids)) != 12_600:
        raise ActionCountCNNFitError("retained train/validation panels overlap")
    if (
        v3_plan.get("supersession", {}).get("retained_exactly")
        != ["v2_train_800", "v2_validation_100"]
        or v3_plan.get("old_v2_design_taint", {}).get("reuse_allowed") is not False
    ):
        raise ActionCountCNNFitError("v3 supersession/old-v2 exclusion differs")
    _verify_execution_protocol(v2_plan)
    return FitAuthorities(
        v3_plan=v3_plan,
        v3_plan_raw=v3_plan_raw,
        v3_development=v3_development,
        v3_development_raw=v3_development_raw,
        v2_protocol_plan=v2_plan,
        v2_protocol_plan_raw=v2_plan_raw,
        v2_development_labels=v2_labels,
        v2_development_labels_raw=v2_labels_raw,
    )


def _label_triple(row: Mapping[str, Any]) -> tuple[int, int, int]:
    values = (
        row.get("straight_action_count"),
        row.get("arc_action_count"),
        row.get("catalog_convexity_target"),
    )
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        or not 0 <= values[0] <= 9
        or not 0 <= values[1] <= 9
        or values[2] not in CATALOG_TO_INDEX
    ):
        raise ActionCountCNNFitError("development label triple is invalid")
    return values


def _panel_path(dataset_root: Path, panel_id: str) -> Path:
    if not isinstance(panel_id, str):
        raise ActionCountCNNFitError("panel ID must be text")
    parts = PurePosixPath(panel_id).parts
    if (
        len(parts) != 4
        or parts[0] != "hd"
        or parts[2] not in {"0", "1"}
        or not parts[3].endswith(".png")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ActionCountCNNFitError(f"invalid HD panel ID: {panel_id!r}")
    candidate = dataset_root.joinpath(parts[0], "images", *parts[1:])
    try:
        resolved = candidate.resolve(strict=True)
        root = dataset_root.resolve(strict=True)
    except OSError as exc:
        raise ActionCountCNNFitError(f"cannot resolve fit panel: {exc}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ActionCountCNNFitError("fit panel escapes dataset root or is not a file")
    return resolved


def _development_rows(development: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cohort in ("train", "validation"):
        rows = development["cohorts"][cohort].get("rows")
        if not isinstance(rows, list):
            raise ActionCountCNNFitError(f"{cohort} rows are invalid")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("panel_id"), str):
                raise ActionCountCNNFitError(f"{cohort} row is invalid")
            panel_id = row["panel_id"]
            if panel_id in seen:
                raise ActionCountCNNFitError("fit panel ID repeats")
            seen.add(panel_id)
            _label_triple(row)
            result.append({**row, "fit_cohort": cohort})
    if len(result) != 12_600:
        raise ActionCountCNNFitError("fit row count differs")
    return result


def _resolved_dataset_root(dataset_root: Path) -> Path:
    try:
        root = dataset_root.resolve(strict=True)
    except OSError as exc:
        raise ActionCountCNNFitError(f"cannot resolve dataset root: {exc}") from exc
    if not root.is_dir():
        raise ActionCountCNNFitError("dataset root is not a directory")
    return root


def _authorization_body(
    *,
    authorities: FitAuthorities,
    dataset_root: Path,
    intended_precommit_path: Path,
) -> dict[str, Any]:
    rows = _development_rows(authorities.v2_development_labels)
    panel_ids = [row["panel_id"] for row in rows]
    root = _resolved_dataset_root(dataset_root)
    source_sha = verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )
    return {
        "authorized_fit_cohorts": ["train", "validation"],
        "authorized_panel_count": len(panel_ids),
        "authorized_panel_ids": panel_ids,
        "authorized_panel_ids_digest": "sha256:" + canonical_digest(panel_ids),
        "conservative_crash_exposure_policy": {
            "all_authorized_panels_count_as_exposed_once_this_record_is_durable": True,
            "crash_or_interruption_before_completed_pixel_precommit": (
                "count_all_12600_authorized_train_validation_PNGs_as_exposed"
            ),
        },
        "correction": dict(FIT_CORRECTION),
        "dataset_root": str(root),
        "intended_pixel_precommit_path": str(intended_precommit_path.resolve()),
        "runtime": _runtime_identity(),
        "schema": AUTHORIZATION_SCHEMA,
        "trainer_source_sha256": source_sha,
        "unopened_and_unauthorized": [
            "old_v2_calibration_panel_PNGs",
            "old_v2_evaluation_panel_PNGs",
            "fresh_v3_calibration_panel_PNGs",
            "fresh_v3_evaluation_panel_PNGs",
        ],
        "v2_development_labels_record_digest": authorities.v2_development_labels[
            "record_digest"
        ],
        "v2_development_labels_source_sha256": _address(
            authorities.v2_development_labels_raw
        ),
        "v2_protocol_plan_record_digest": authorities.v2_protocol_plan[
            "record_digest"
        ],
        "v2_protocol_plan_source_sha256": _address(
            authorities.v2_protocol_plan_raw
        ),
        "v3_development_ids_record_digest": authorities.v3_development[
            "record_digest"
        ],
        "v3_development_ids_source_sha256": _address(
            authorities.v3_development_raw
        ),
        "v3_plan_record_digest": authorities.v3_plan["record_digest"],
        "v3_plan_source_sha256": _address(authorities.v3_plan_raw),
    }


def create_fit_exposure_authorization(
    *,
    v3_plan_path: Path,
    v3_development_path: Path,
    v2_protocol_plan_path: Path,
    v2_development_labels_path: Path,
    dataset_root: Path,
    intended_precommit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    authorities = _verify_fit_authorities(
        v3_plan_path=v3_plan_path,
        v3_development_path=v3_development_path,
        v2_protocol_plan_path=v2_protocol_plan_path,
        v2_development_labels_path=v2_development_labels_path,
    )
    if output_path.resolve() == intended_precommit_path.resolve():
        raise ActionCountCNNFitError(
            "authorization and intended pixel precommit paths must differ"
        )
    result = _seal_body(
        _authorization_body(
            authorities=authorities,
            dataset_root=dataset_root,
            intended_precommit_path=intended_precommit_path,
        )
    )
    _write_fsynced(output_path, result)
    reloaded, _ = _load_record(output_path, label="fit exposure authorization")
    if reloaded != result:
        raise ActionCountCNNFitError("fit exposure authorization fresh-load differs")
    return result


def _verify_fit_exposure_authorization(
    *,
    authorization_path: Path,
    authorities: FitAuthorities,
    dataset_root: Path,
    intended_precommit_path: Path,
) -> tuple[dict[str, Any], bytes]:
    authorization, authorization_raw = _load_record(
        authorization_path, label="fit exposure authorization"
    )
    expected = _seal_body(
        _authorization_body(
            authorities=authorities,
            dataset_root=dataset_root,
            intended_precommit_path=intended_precommit_path,
        )
    )
    if authorization != expected:
        raise ActionCountCNNFitError("fit exposure authorization differs")
    return authorization, authorization_raw


def _audit_digest_groups(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        if not isinstance(row, Mapping) or set(row) != {
            "fit_cohort",
            "label_triple",
            "metric_strata",
            "panel_id",
            "png_sha256",
            "png_size_bytes",
        }:
            raise ActionCountCNNFitError("fit PNG observation fields differ")
        label_triple = row["label_triple"]
        metric_strata = row["metric_strata"]
        if (
            row["fit_cohort"] not in {"train", "validation"}
            or not isinstance(row["panel_id"], str)
            or not isinstance(row["png_sha256"], str)
            or _SHA256_ADDRESS.fullmatch(row["png_sha256"]) is None
            or isinstance(row["png_size_bytes"], bool)
            or not isinstance(row["png_size_bytes"], int)
            or row["png_size_bytes"] <= 0
            or not isinstance(label_triple, list)
            or len(label_triple) != 3
            or _label_triple(
                {
                    "straight_action_count": label_triple[0],
                    "arc_action_count": label_triple[1],
                    "catalog_convexity_target": label_triple[2],
                }
            )
            != tuple(label_triple)
            or not isinstance(metric_strata, dict)
            or set(metric_strata) != {
                "crossing_task",
                "line_decoration",
                "thin_task",
            }
            or not isinstance(metric_strata["crossing_task"], bool)
            or not isinstance(metric_strata["thin_task"], bool)
            or not isinstance(metric_strata["line_decoration"], str)
            or not metric_strata["line_decoration"]
        ):
            raise ActionCountCNNFitError("fit PNG observation value is invalid")
        grouped[row["png_sha256"]].append(row)
    groups: list[dict[str, Any]] = []
    duplicate_panel_count = duplicate_group_count = 0
    for digest in sorted(grouped):
        members = grouped[digest]
        triples = {tuple(member["label_triple"]) for member in members}
        cohorts = {str(member["fit_cohort"]) for member in members}
        sizes = {int(member["png_size_bytes"]) for member in members}
        if len(triples) != 1:
            raise ActionCountCNNFitError(
                "byte-identical PNGs carry different label triples"
            )
        if len(cohorts) != 1:
            raise ActionCountCNNFitError(
                "byte-identical PNG leaks across train and validation"
            )
        if len(sizes) != 1:
            raise ActionCountCNNFitError("one PNG digest has different sizes")
        if len(members) > 1:
            duplicate_group_count += 1
            duplicate_panel_count += len(members)
        groups.append(
            {
                "fit_cohort": next(iter(cohorts)),
                "label_triple": list(next(iter(triples))),
                "multiplicity": len(members),
                "metric_strata": [member["metric_strata"] for member in members],
                "panel_ids": sorted(str(member["panel_id"]) for member in members),
                "png_sha256": digest,
                "png_size_bytes": next(iter(sizes)),
            }
        )
    return groups, {
        "cross_cohort_duplicate_group_count": 0,
        "different_label_duplicate_group_count": 0,
        "duplicate_group_count": duplicate_group_count,
        "panels_in_duplicate_groups": duplicate_panel_count,
        "path_independent_training_policy": (
            "sort_unique_digest_groups_by_epoch_hash_then_expand_each_group_by_"
            "multiplicity;_members_are_identical_pixels_and_labels"
        ),
        "unique_png_digest_count": len(groups),
    }


def create_fit_precommit(
    *,
    v3_plan_path: Path,
    v3_development_path: Path,
    v2_protocol_plan_path: Path,
    v2_development_labels_path: Path,
    dataset_root: Path,
    authorization_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    authorities = _verify_fit_authorities(
        v3_plan_path=v3_plan_path,
        v3_development_path=v3_development_path,
        v2_protocol_plan_path=v2_protocol_plan_path,
        v2_development_labels_path=v2_development_labels_path,
    )
    authorization, authorization_raw = _verify_fit_exposure_authorization(
        authorization_path=authorization_path,
        authorities=authorities,
        dataset_root=dataset_root,
        intended_precommit_path=output_path,
    )
    # No PNG path is resolved or read above this line.  Once the durable
    # authorization exists, its policy conservatively counts this entire
    # 12,600-panel inventory as exposed even if this process crashes.
    rows = _development_rows(authorities.v2_development_labels)
    observations: list[dict[str, Any]] = []
    for row in rows:
        panel_path = _panel_path(dataset_root, row["panel_id"])
        try:
            raw = panel_path.read_bytes()
        except OSError as exc:
            raise ActionCountCNNFitError(f"cannot read fit PNG: {exc}") from exc
        observations.append(
            {
                "fit_cohort": row["fit_cohort"],
                "label_triple": list(_label_triple(row)),
                "panel_id": row["panel_id"],
                "png_sha256": _address(raw),
                "png_size_bytes": len(raw),
                "metric_strata": {
                    "crossing_task": bool(row["crossing_task_stratum"]),
                    "line_decoration": row["line_decoration_stratum"],
                    "thin_task": bool(row["thin_task_stratum"]),
                },
            }
        )
    groups, duplicate_audit = _audit_digest_groups(observations)
    source_sha = verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )
    body = {
        "authorization_record_digest": authorization["record_digest"],
        "authorization_source_sha256": _address(authorization_raw),
        "correction": dict(FIT_CORRECTION),
        "duplicate_digest_audit": duplicate_audit,
        "exact_png_observations": observations,
        "fit_panel_count": len(observations),
        "path_independent_digest_groups": groups,
        "runtime": _runtime_identity(),
        "schema": PRECOMMIT_SCHEMA,
        "trainer_source_sha256": source_sha,
        "unopened_by_this_precommit": [
            "old_v2_calibration_panel_PNGs",
            "old_v2_evaluation_panel_PNGs",
            "fresh_v3_calibration_panel_PNGs",
            "fresh_v3_evaluation_panel_PNGs",
        ],
        "v2_development_labels_record_digest": authorities.v2_development_labels[
            "record_digest"
        ],
        "v2_development_labels_source_sha256": _address(
            authorities.v2_development_labels_raw
        ),
        "v2_protocol_plan_record_digest": authorities.v2_protocol_plan[
            "record_digest"
        ],
        "v2_protocol_plan_source_sha256": _address(
            authorities.v2_protocol_plan_raw
        ),
        "v3_development_ids_record_digest": authorities.v3_development[
            "record_digest"
        ],
        "v3_development_ids_source_sha256": _address(
            authorities.v3_development_raw
        ),
        "v3_plan_record_digest": authorities.v3_plan["record_digest"],
        "v3_plan_source_sha256": _address(authorities.v3_plan_raw),
    }
    result = _seal_body(body)
    _write_fsynced(output_path, result)
    reloaded, _ = _load_record(output_path, label="fit precommit fresh load")
    if reloaded != result:
        raise ActionCountCNNFitError("fit precommit fresh-load replay differs")
    return result


def _verify_precommit(
    *,
    precommit_path: Path,
    authorization_path: Path,
    authorities: FitAuthorities,
    dataset_root: Path,
) -> dict[str, Any]:
    authorization, authorization_raw = _verify_fit_exposure_authorization(
        authorization_path=authorization_path,
        authorities=authorities,
        dataset_root=dataset_root,
        intended_precommit_path=precommit_path,
    )
    precommit, _ = _load_record(precommit_path, label="fit precommit")
    if precommit.get("schema") != PRECOMMIT_SCHEMA:
        raise ActionCountCNNFitError("fit precommit schema differs")
    expected = {
        "authorization_record_digest": authorization["record_digest"],
        "authorization_source_sha256": _address(authorization_raw),
        "v3_plan_record_digest": authorities.v3_plan["record_digest"],
        "v3_plan_source_sha256": _address(authorities.v3_plan_raw),
        "v3_development_ids_record_digest": authorities.v3_development[
            "record_digest"
        ],
        "v3_development_ids_source_sha256": _address(
            authorities.v3_development_raw
        ),
        "v2_protocol_plan_record_digest": authorities.v2_protocol_plan[
            "record_digest"
        ],
        "v2_protocol_plan_source_sha256": _address(
            authorities.v2_protocol_plan_raw
        ),
        "v2_development_labels_record_digest": authorities.v2_development_labels[
            "record_digest"
        ],
        "v2_development_labels_source_sha256": _address(
            authorities.v2_development_labels_raw
        ),
        "trainer_source_sha256": verify_loaded_source(
            __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
        ),
        "correction": dict(FIT_CORRECTION),
    }
    for field, value in expected.items():
        if precommit.get(field) != value:
            raise ActionCountCNNFitError(f"fit precommit {field} differs")
    if precommit.get("runtime") != _runtime_identity():
        raise ActionCountCNNFitError("fit runtime differs from precommit")
    if precommit.get("fit_panel_count") != 12_600:
        raise ActionCountCNNFitError("fit precommit panel count differs")
    observations = precommit.get("exact_png_observations")
    groups = precommit.get("path_independent_digest_groups")
    if not isinstance(observations, list) or not isinstance(groups, list):
        raise ActionCountCNNFitError("fit precommit observations are invalid")
    if len(observations) != 12_600 or precommit.get("fit_panel_count") != len(
        observations
    ):
        raise ActionCountCNNFitError("fit precommit observation count differs")
    expected_rows = _development_rows(authorities.v2_development_labels)
    expected = {
        row["panel_id"]: {
            "fit_cohort": row["fit_cohort"],
            "label_triple": list(_label_triple(row)),
            "metric_strata": {
                "crossing_task": bool(row["crossing_task_stratum"]),
                "line_decoration": row["line_decoration_stratum"],
                "thin_task": bool(row["thin_task_stratum"]),
            },
        }
        for row in expected_rows
    }
    found_ids = [row.get("panel_id") for row in observations if isinstance(row, dict)]
    if len(found_ids) != len(set(found_ids)) or set(found_ids) != set(expected):
        raise ActionCountCNNFitError("fit precommit panel inventory differs")
    for row in observations:
        panel_id = row["panel_id"]
        for field in ("fit_cohort", "label_triple", "metric_strata"):
            if row.get(field) != expected[panel_id][field]:
                raise ActionCountCNNFitError(
                    f"fit precommit {field} differs from development labels"
                )
    rebuilt_groups, rebuilt_audit = _audit_digest_groups(observations)
    if rebuilt_groups != groups or rebuilt_audit != precommit.get(
        "duplicate_digest_audit"
    ):
        raise ActionCountCNNFitError("fit precommit duplicate audit differs")
    if (
        sum(group["multiplicity"] for group in rebuilt_groups) != len(observations)
        or sum(len(group["panel_ids"]) for group in rebuilt_groups)
        != len(observations)
        or any(
            group["multiplicity"] != len(group["panel_ids"])
            or group["multiplicity"] != len(group["metric_strata"])
            for group in rebuilt_groups
        )
    ):
        raise ActionCountCNNFitError("fit precommit group multiplicity differs")
    return precommit


def preprocess_png_bytes(raw: bytes, *, image_size: int = 96) -> np.ndarray:
    """Apply the exact preregistered grayscale/ink crop as uint8 ink."""

    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            if image.format != "PNG":
                raise ActionCountCNNFitError("fit image is not PNG")
            if getattr(image, "n_frames", 1) != 1:
                raise ActionCountCNNFitError("fit PNG must contain exactly one frame")
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
    except ActionCountCNNFitError:
        raise
    except Exception as exc:
        raise ActionCountCNNFitError(f"cannot decode fit PNG: {exc}") from exc
    if gray.ndim != 2 or gray.size == 0:
        raise ActionCountCNNFitError("decoded grayscale image is invalid")
    ys, xs = np.nonzero(gray < 250)
    if len(xs) == 0:
        raise ActionCountCNNFitError("fit image has no ink")
    crop = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = crop.shape
    margin = math.ceil(0.08 * max(height, width))
    side = max(height, width) + 2 * margin
    canvas = np.full((side, side), 255, dtype=np.uint8)
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top : top + height, left : left + width] = crop
    resized = Image.fromarray(canvas, mode="L").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    gray_resized = np.asarray(resized, dtype=np.uint8)
    return np.ascontiguousarray(255 - gray_resized)


def content_epoch_key(seed: int, epoch: int, png_sha256: str) -> bytes:
    """Return the sole path-independent augmentation/shuffle key."""

    if not isinstance(seed, int) or not isinstance(epoch, int) or epoch < 0:
        raise ActionCountCNNFitError("invalid content epoch key inputs")
    if not isinstance(png_sha256, str) or not png_sha256.startswith("sha256:"):
        raise ActionCountCNNFitError("invalid PNG content address")
    return hashlib.sha256(
        f"{seed}\0{epoch}\0{png_sha256}".encode("utf-8")
    ).digest()


def d4_transform(array: np.ndarray, index: int) -> np.ndarray:
    if array.ndim != 2 or index not in range(8):
        raise ActionCountCNNFitError("invalid D4 transform input")
    result = array if index < 4 else np.fliplr(array)
    result = np.rot90(result, k=index % 4)
    return np.ascontiguousarray(result)


def _torch_runtime():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ActionCountCNNFitError("PyTorch is unavailable") from exc
    return torch, nn, functional


def _configure_torch(seed: int) -> Any:
    torch, _, _ = _torch_runtime()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise ActionCountCNNFitError("torch interop threads cannot be fixed to one")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    return torch


def build_model(*, seed: int = 260810):
    torch = _configure_torch(seed)
    _, nn, _ = _torch_runtime()

    class SharedCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = []
            incoming = 1
            for outgoing in (16, 32, 64, 96):
                blocks.extend(
                    (
                        nn.Conv2d(
                            incoming,
                            outgoing,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(
                            outgoing,
                            eps=1e-5,
                            momentum=0.1,
                            affine=True,
                            track_running_stats=True,
                        ),
                        nn.ReLU(inplace=False),
                    )
                )
                incoming = outgoing
            self.encoder = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.straight = nn.Linear(96, 10, bias=True)
            self.arc = nn.Linear(96, 10, bias=True)
            self.catalog = nn.Linear(96, 3, bias=True)
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
                    nn.init.zeros_(module.bias)

        def forward(self, value):
            encoded = self.pool(self.encoder(value)).flatten(1)
            return self.straight(encoded), self.arc(encoded), self.catalog(encoded)

    return SharedCNN()


def state_dict_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"gkm.action-count-cnn-state-dict.v1\0")
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        metadata = {
            "dtype": str(tensor.dtype),
            "name": name,
            "shape": list(tensor.shape),
        }
        encoded = canonical_json(metadata)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class MaterializedGroup:
    cohort: str
    digest: str
    ink: np.ndarray
    labels: tuple[int, int, int]
    multiplicity: int
    metric_strata: tuple[Mapping[str, Any], ...] = ()


def _materialize_groups(
    *, precommit: Mapping[str, Any], dataset_root: Path
) -> list[MaterializedGroup]:
    observations = precommit.get("exact_png_observations")
    groups = precommit.get("path_independent_digest_groups")
    if not isinstance(observations, list) or not isinstance(groups, list):
        raise ActionCountCNNFitError("fit precommit manifests are invalid")
    observed_by_panel = {row["panel_id"]: row for row in observations}
    result: list[MaterializedGroup] = []
    for group in groups:
        panel_ids = group["panel_ids"]
        if (
            not isinstance(panel_ids, list)
            or not panel_ids
            or group.get("multiplicity") != len(panel_ids)
            or not isinstance(group.get("metric_strata"), list)
            or len(group["metric_strata"]) != len(panel_ids)
        ):
            raise ActionCountCNNFitError("digest group has no panels")
        representative_raw: bytes | None = None
        for panel_id in panel_ids:
            raw = _panel_path(dataset_root, panel_id).read_bytes()
            expected = observed_by_panel.get(panel_id)
            if (
                expected is None
                or expected.get("png_sha256") != group["png_sha256"]
                or expected.get("png_size_bytes") != group["png_size_bytes"]
                or _address(raw) != group["png_sha256"]
                or len(raw) != group["png_size_bytes"]
            ):
                raise ActionCountCNNFitError("fit PNG changed after precommit")
            if representative_raw is None:
                representative_raw = raw
        assert representative_raw is not None
        labels = tuple(group["label_triple"])
        result.append(
            MaterializedGroup(
                cohort=group["fit_cohort"],
                digest=group["png_sha256"],
                ink=preprocess_png_bytes(representative_raw),
                labels=(labels[0], labels[1], CATALOG_TO_INDEX[labels[2]]),
                multiplicity=group["multiplicity"],
                metric_strata=tuple(group["metric_strata"]),
            )
        )
    return result


def _class_weights(groups: Sequence[MaterializedGroup], head: int, classes: int):
    torch, _, _ = _torch_runtime()
    counts = [0] * classes
    for group in groups:
        counts[group.labels[head]] += group.multiplicity
    nonzero = [1 / math.sqrt(count) for count in counts if count]
    mean = sum(nonzero) / len(nonzero)
    weights = [(1 / math.sqrt(count)) / mean if count else 0.0 for count in counts]
    return torch.tensor(weights, dtype=torch.float32), counts


def _expanded_order(groups: Sequence[MaterializedGroup], *, seed: int, epoch: int) -> list[int]:
    indices = sorted(
        range(len(groups)),
        key=lambda index: (content_epoch_key(seed, epoch, groups[index].digest), groups[index].digest),
    )
    return [index for index in indices for _ in range(groups[index].multiplicity)]


def _batch_tensor(groups, indices, *, epoch: int, seed: int, augment: bool):
    torch, _, _ = _torch_runtime()
    arrays = []
    labels = [[], [], []]
    for index in indices:
        group = groups[index]
        array = group.ink
        if augment:
            key = content_epoch_key(seed, epoch, group.digest)
            array = d4_transform(array, int.from_bytes(key, "big") % 8)
        arrays.append(array)
        for head in range(3):
            labels[head].append(group.labels[head])
    pixels = torch.from_numpy(np.stack(arrays)[:, None]).to(torch.float32) / 255.0
    targets = tuple(torch.tensor(value, dtype=torch.long) for value in labels)
    return pixels, targets


def _predict_groups(
    model,
    groups: Sequence[MaterializedGroup],
    *,
    class_weights: Sequence[Any] | None = None,
) -> tuple[list[list[int]], float]:
    torch, _, functional = _torch_runtime()
    model.eval()
    predictions = [[], [], []]
    loss_sum = 0.0
    sample_count = 0
    weights = tuple(class_weights) if class_weights is not None else tuple(
        _class_weights(groups, head, classes)[0]
        for head, classes in enumerate((10, 10, 3))
    )
    if len(weights) != 3:
        raise ActionCountCNNFitError("three class-weight tensors are required")
    with torch.no_grad():
        order = [index for index, group in enumerate(groups) for _ in range(group.multiplicity)]
        for start in range(0, len(order), 64):
            pixels, targets = _batch_tensor(
                groups, order[start : start + 64], epoch=0, seed=260810, augment=False
            )
            logits = model(pixels)
            loss = sum(
                functional.cross_entropy(output, target, weight=weight, reduction="mean")
                for output, target, weight in zip(logits, targets, weights)
            )
            count = len(order[start : start + 64])
            loss_sum += float(loss.item()) * count
            sample_count += count
            for head, output in enumerate(logits):
                predictions[head].extend(output.argmax(1).tolist())
    return predictions, loss_sum / sample_count


def _confusion(true: Sequence[int], predicted: Sequence[int], classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    for expected, found in zip(true, predicted):
        matrix[expected][found] += 1
    return matrix


def _accuracy_at(indices: Sequence[int], true: Sequence[int], predicted: Sequence[int]) -> float:
    if not indices:
        raise ActionCountCNNFitError("required metric stratum is empty")
    return sum(predicted[index] == true[index] for index in indices) / len(indices)


def _validation_metrics(groups: Sequence[MaterializedGroup], predictions, loss: float) -> dict[str, Any]:
    truths = [[], [], []]
    strata: list[Mapping[str, Any]] = []
    for group in groups:
        group_strata = group.metric_strata or tuple({} for _ in range(group.multiplicity))
        if len(group_strata) != group.multiplicity:
            raise ActionCountCNNFitError("metric strata multiplicity differs")
        for metric_stratum in group_strata:
            for head in range(3):
                truths[head].append(group.labels[head])
            strata.append(metric_stratum)
    total = len(truths[0])
    accuracies = [
        sum(a == b for a, b in zip(truths[head], predictions[head])) / total
        for head in range(3)
    ]
    known = [index for index, value in enumerate(truths[2]) if value in {1, 2}]
    recalls = []
    for value in (1, 2):
        positions = [index for index in known if truths[2][index] == value]
        recalls.append(
            sum(predictions[2][index] == value for index in positions) / len(positions)
        )
    joint = sum(
        predictions[0][index] == truths[0][index]
        and predictions[2][index] == truths[2][index]
        for index in known
    ) / len(known)
    decoration_values = sorted(
        {str(item["line_decoration"]) for item in strata if "line_decoration" in item}
    )
    straight_strata: dict[str, dict[str, Any]] = {}
    selections = {
        "straight_true_count_4": [
            index for index, value in enumerate(truths[0]) if value == 4
        ],
        "thin_task": [
            index for index, value in enumerate(strata) if value.get("thin_task") is True
        ],
        "crossing_task": [
            index
            for index, value in enumerate(strata)
            if value.get("crossing_task") is True
        ],
    }
    for decoration in decoration_values:
        selections[f"line_decoration:{decoration}"] = [
            index
            for index, value in enumerate(strata)
            if value.get("line_decoration") == decoration
        ]
    for name, indices in selections.items():
        if indices:
            straight_strata[name] = {
                "correct": sum(
                    predictions[0][index] == truths[0][index] for index in indices
                ),
                "panel_count": len(indices),
                "straight_top1": _accuracy_at(indices, truths[0], predictions[0]),
            }
    return {
        "arc_top1": accuracies[1],
        "catalog_all-class_top1": accuracies[2],
        "confusions_true_rows_predicted_columns": {
            "arc_10x10": _confusion(truths[1], predictions[1], 10),
            "catalog_3x3_unresolved_nonconvex_convex": _confusion(
                truths[2], predictions[2], 3
            ),
            "straight_10x10": _confusion(truths[0], predictions[0], 10),
        },
        "known_catalog_binary_balanced_accuracy": sum(recalls) / 2,
        "known_catalog_panel_count": len(known),
        "straight_and_known_catalog_joint_exact": joint,
        "straight_required_strata": straight_strata,
        "straight_top1": accuracies[0],
        "total_cross_entropy": loss,
    }


def _selection_key(metrics: Mapping[str, Any], epoch: int) -> tuple[float, ...]:
    return (
        metrics["straight_and_known_catalog_joint_exact"],
        metrics["straight_top1"],
        metrics["known_catalog_binary_balanced_accuracy"],
        metrics["arc_top1"],
        -metrics["total_cross_entropy"],
        -epoch,
    )


def train_core(groups: Sequence[MaterializedGroup], *, epochs: int = 16, seed: int = 260810):
    torch = _configure_torch(seed)
    _, _, functional = _torch_runtime()
    training = [group for group in groups if group.cohort == "train"]
    validation = [group for group in groups if group.cohort == "validation"]
    if not training or not validation:
        raise ActionCountCNNFitError("train/validation groups are required")
    model = build_model(seed=seed)
    weights_and_counts = tuple(
        _class_weights(training, head, classes) for head, classes in enumerate((10, 10, 3))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001
    )
    history = []
    best_key = None
    best_state = None
    best_epoch = None
    best_predictions = None
    for epoch in range(epochs):
        model.train()
        order = _expanded_order(training, seed=seed, epoch=epoch)
        train_loss_sum = 0.0
        for start in range(0, len(order), 64):
            pixels, targets = _batch_tensor(
                training,
                order[start : start + 64],
                epoch=epoch,
                seed=seed,
                augment=True,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixels)
            loss = sum(
                functional.cross_entropy(output, target, weight=weight, reduction="mean")
                for output, target, (weight, _) in zip(logits, targets, weights_and_counts)
            )
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(order[start : start + 64])
        predictions, validation_loss = _predict_groups(
            model,
            validation,
            class_weights=[item[0] for item in weights_and_counts],
        )
        metrics = _validation_metrics(validation, predictions, validation_loss)
        metrics["epoch"] = epoch
        metrics["training_cross_entropy"] = train_loss_sum / len(order)
        history.append(metrics)
        key = _selection_key(metrics, epoch)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_predictions = predictions
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    assert best_state is not None and best_epoch is not None and best_predictions is not None
    model.load_state_dict(best_state, strict=True)
    replay_predictions, replay_loss = _predict_groups(
        model,
        validation,
        class_weights=[item[0] for item in weights_and_counts],
    )
    if replay_predictions != best_predictions:
        raise ActionCountCNNFitError("selected in-memory state replay differs")
    training_predictions, training_loss = _predict_groups(
        model,
        training,
        class_weights=[item[0] for item in weights_and_counts],
    )
    return {
        "best_epoch": best_epoch,
        "best_metrics": _validation_metrics(validation, replay_predictions, replay_loss),
        "class_counts": {
            name: counts
            for name, (_, counts) in zip(
                ("straight", "arc", "catalog_convexity"), weights_and_counts
            )
        },
        "history": history,
        "model": model,
        "predictions": replay_predictions,
        "state": best_state,
        "selected_checkpoint_training_metrics": _validation_metrics(
            training, training_predictions, training_loss
        ),
        "training_class_weights": [item[0] for item in weights_and_counts],
        "validation_groups": validation,
    }


def _prediction_rows(groups: Sequence[MaterializedGroup], predictions) -> list[dict[str, Any]]:
    rows = []
    cursor = 0
    for group in groups:
        for occurrence in range(group.multiplicity):
            rows.append(
                {
                    "catalog_convexity_predicted": CATALOG_VALUES[predictions[2][cursor]],
                    "digest_occurrence": occurrence,
                    "png_sha256": group.digest,
                    "predicted_arc_action_count": predictions[1][cursor],
                    "predicted_straight_action_count": predictions[0][cursor],
                }
            )
            cursor += 1
    return rows


def _validation_gate(plan: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = plan["metrics_and_checkpoint_selection"][
        "validation_gate_before_any_calibration_pixel"
    ]
    checks = {
        "arc_top1": metrics["arc_top1"] >= thresholds["arc_top1_at_least"],
        "known_catalog_binary_balanced_accuracy": (
            metrics["known_catalog_binary_balanced_accuracy"]
            >= thresholds["known_catalog_binary_balanced_accuracy_at_least"]
        ),
        "straight_top1": metrics["straight_top1"]
        >= thresholds["straight_top1_at_least"],
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "on_failure": thresholds["on_failure"],
        "thresholds": thresholds,
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> bytes:
    torch, _, _ = _torch_runtime()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing, raw = _load_checkpoint(path)
        scalar_fields = (
            "architecture_id",
            "catalog_class_values",
            "config_digest",
            "selected_epoch",
        )
        if any(existing[field] != payload[field] for field in scalar_fields) or (
            state_dict_digest(existing["state_dict"])
            != state_dict_digest(payload["state_dict"])
        ):
            raise ActionCountCNNFitError(
                "refusing to overwrite nonidentical checkpoint"
            )
        return raw
    temporary = path.with_name(path.name + ".tmp-action-count-cnn-fit")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        raw = path.read_bytes()
    except OSError as exc:
        raise ActionCountCNNFitError(f"cannot save checkpoint: {exc}") from exc
    return raw


def _load_checkpoint(path: Path, *, expected_raw_sha256: str | None = None):
    torch, _, _ = _torch_runtime()
    raw = path.read_bytes()
    if expected_raw_sha256 is not None and _address(raw) != expected_raw_sha256:
        raise ActionCountCNNFitError("checkpoint raw digest differs")
    try:
        payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ActionCountCNNFitError(f"cannot load checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "architecture_id",
        "catalog_class_values",
        "config_digest",
        "selected_epoch",
        "state_dict",
    }:
        raise ActionCountCNNFitError("checkpoint fields differ")
    return payload, raw


def run_fit_training(
    *,
    v3_plan_path: Path,
    v3_development_path: Path,
    v2_protocol_plan_path: Path,
    v2_development_labels_path: Path,
    dataset_root: Path,
    authorization_path: Path,
    precommit_path: Path,
    checkpoint_path: Path,
    record_path: Path,
) -> dict[str, Any]:
    authorities = _verify_fit_authorities(
        v3_plan_path=v3_plan_path,
        v3_development_path=v3_development_path,
        v2_protocol_plan_path=v2_protocol_plan_path,
        v2_development_labels_path=v2_development_labels_path,
    )
    precommit = _verify_precommit(
        precommit_path=precommit_path,
        authorization_path=authorization_path,
        authorities=authorities,
        dataset_root=dataset_root,
    )
    groups = _materialize_groups(precommit=precommit, dataset_root=dataset_root)
    result = train_core(
        groups,
        epochs=int(EXECUTION_PROTOCOL["epochs"]),
        seed=int(EXECUTION_PROTOCOL["random_seed"]),
    )
    state_digest = state_dict_digest(result["state"])
    config_digest = "sha256:" + canonical_digest(
        authorities.v2_protocol_plan["training_protocol"]
    )
    checkpoint = {
        "architecture_id": ARCHITECTURE_ID,
        "catalog_class_values": list(CATALOG_VALUES),
        "config_digest": config_digest,
        "selected_epoch": result["best_epoch"],
        "state_dict": result["state"],
    }
    checkpoint_raw = _save_checkpoint(checkpoint_path, checkpoint)
    loaded, _ = _load_checkpoint(
        checkpoint_path, expected_raw_sha256=_address(checkpoint_raw)
    )
    if (
        loaded["architecture_id"] != ARCHITECTURE_ID
        or loaded["config_digest"] != config_digest
        or loaded["selected_epoch"] != result["best_epoch"]
        or loaded["catalog_class_values"] != list(CATALOG_VALUES)
    ):
        raise ActionCountCNNFitError("fresh-loaded checkpoint configuration differs")
    if state_dict_digest(loaded["state_dict"]) != state_digest:
        raise ActionCountCNNFitError("fresh-loaded checkpoint state differs")
    fresh_model = build_model(seed=int(EXECUTION_PROTOCOL["random_seed"]))
    fresh_model.load_state_dict(loaded["state_dict"], strict=True)
    fresh_predictions, fresh_loss = _predict_groups(
        fresh_model,
        result["validation_groups"],
        class_weights=result["training_class_weights"],
    )
    fresh_metrics = _validation_metrics(
        result["validation_groups"], fresh_predictions, fresh_loss
    )
    prediction_rows = _prediction_rows(result["validation_groups"], fresh_predictions)
    if fresh_metrics != result["best_metrics"]:
        raise ActionCountCNNFitError("fresh-loaded validation metrics differ")
    body = {
        "architecture_id": ARCHITECTURE_ID,
        "checkpoint_raw_sha256": _address(checkpoint_raw),
        "checkpoint_state_dict_sha256": state_digest,
        "config_digest": config_digest,
        "correction": dict(FIT_CORRECTION),
        "duplicate_digest_audit": precommit["duplicate_digest_audit"],
        "fit_precommit_record_digest": precommit["record_digest"],
        "fresh_load_replay": {
            "checkpoint_reloaded": True,
            "metrics_exact": True,
            "predictions_exact": True,
            "state_dict_digest_exact": True,
        },
        "history": result["history"],
        "execution_protocol": _execution_protocol_data(),
        "v2_protocol_plan_record_digest": authorities.v2_protocol_plan[
            "record_digest"
        ],
        "v3_plan_record_digest": authorities.v3_plan["record_digest"],
        "runtime": precommit["runtime"],
        "schema": TRAINING_SCHEMA,
        "selected_epoch": result["best_epoch"],
        "training_class_counts": result["class_counts"],
        "training_metrics_at_selected_checkpoint": result[
            "selected_checkpoint_training_metrics"
        ],
        "validation_gate": _validation_gate(
            authorities.v2_protocol_plan, fresh_metrics
        ),
        "validation_metrics": fresh_metrics,
        "validation_prediction_rows": prediction_rows,
        "validation_predictions_digest": "sha256:" + canonical_digest(prediction_rows),
    }
    final = _seal_body(body)
    _write_fsynced(record_path, final)
    reloaded, _ = _load_record(record_path, label="fit result fresh load")
    if reloaded != final:
        raise ActionCountCNNFitError("fit result fresh-load differs")
    return final


def replay_fit_training(
    *,
    v3_plan_path: Path,
    v3_development_path: Path,
    v2_protocol_plan_path: Path,
    v2_development_labels_path: Path,
    dataset_root: Path,
    authorization_path: Path,
    precommit_path: Path,
    checkpoint_path: Path,
    record_path: Path,
    replay_output_path: Path,
) -> dict[str, Any]:
    authorities = _verify_fit_authorities(
        v3_plan_path=v3_plan_path,
        v3_development_path=v3_development_path,
        v2_protocol_plan_path=v2_protocol_plan_path,
        v2_development_labels_path=v2_development_labels_path,
    )
    precommit = _verify_precommit(
        precommit_path=precommit_path,
        authorization_path=authorization_path,
        authorities=authorities,
        dataset_root=dataset_root,
    )
    archived, _ = _load_record(record_path, label="archived fit result")
    if archived.get("schema") != TRAINING_SCHEMA:
        raise ActionCountCNNFitError("archived fit result schema differs")
    expected_execution = _execution_protocol_data()
    if (
        archived.get("correction") != dict(FIT_CORRECTION)
        or archived.get("v3_plan_record_digest")
        != authorities.v3_plan["record_digest"]
        or archived.get("v2_protocol_plan_record_digest")
        != authorities.v2_protocol_plan["record_digest"]
        or archived.get("fit_precommit_record_digest") != precommit["record_digest"]
        or archived.get("runtime") != precommit["runtime"]
        or archived.get("execution_protocol") != expected_execution
        or archived.get("config_digest")
        != "sha256:"
        + canonical_digest(authorities.v2_protocol_plan["training_protocol"])
    ):
        raise ActionCountCNNFitError("archived fit authority bindings differ")
    checkpoint, _ = _load_checkpoint(
        checkpoint_path, expected_raw_sha256=archived["checkpoint_raw_sha256"]
    )
    if state_dict_digest(checkpoint["state_dict"]) != archived[
        "checkpoint_state_dict_sha256"
    ]:
        raise ActionCountCNNFitError("cold replay state digest differs")
    if checkpoint["selected_epoch"] != archived["selected_epoch"]:
        raise ActionCountCNNFitError("cold replay selected epoch differs")
    if checkpoint["config_digest"] != archived["config_digest"]:
        raise ActionCountCNNFitError("cold replay checkpoint config differs")
    if checkpoint["architecture_id"] != archived["architecture_id"]:
        raise ActionCountCNNFitError("cold replay architecture differs")
    if checkpoint["catalog_class_values"] != list(CATALOG_VALUES):
        raise ActionCountCNNFitError("cold replay catalog class order differs")
    groups = _materialize_groups(precommit=precommit, dataset_root=dataset_root)
    training = [group for group in groups if group.cohort == "train"]
    validation = [group for group in groups if group.cohort == "validation"]
    model = build_model(seed=260810)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    weights = [
        _class_weights(training, head, classes)[0]
        for head, classes in enumerate((10, 10, 3))
    ]
    predictions, loss = _predict_groups(
        model, validation, class_weights=weights
    )
    metrics = _validation_metrics(validation, predictions, loss)
    rows = _prediction_rows(validation, predictions)
    if metrics != archived["validation_metrics"]:
        raise ActionCountCNNFitError("cold replay metrics differ")
    if "sha256:" + canonical_digest(rows) != archived["validation_predictions_digest"]:
        raise ActionCountCNNFitError("cold replay predictions differ")
    body = {
        "archived_fit_record_digest": archived["record_digest"],
        "checkpoint_state_dict_sha256": archived["checkpoint_state_dict_sha256"],
        "correction": dict(FIT_CORRECTION),
        "fit_precommit_record_digest": precommit["record_digest"],
        "metrics_exact": True,
        "predictions_exact": True,
        "schema": REPLAY_SCHEMA,
    }
    result = _seal_body(body)
    _write_fsynced(replay_output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    authorize = commands.add_parser("authorize-fit-exposure")
    precommit = commands.add_parser("precommit-fit")
    train = commands.add_parser("train-fit")
    replay = commands.add_parser("replay-fit")
    for subparser in (authorize, precommit, train, replay):
        subparser.add_argument("--v3-plan", type=Path, required=True)
        subparser.add_argument("--v3-development-panels", type=Path, required=True)
        subparser.add_argument("--v2-training-plan", type=Path, required=True)
        subparser.add_argument("--v2-development-labels", type=Path, required=True)
        subparser.add_argument("--dataset-root", type=Path, required=True)
    authorize.add_argument("--intended-precommit", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)
    precommit.add_argument("--authorization", type=Path, required=True)
    precommit.add_argument("--output", type=Path, required=True)
    for subparser in (train, replay):
        subparser.add_argument("--authorization", type=Path, required=True)
        subparser.add_argument("--precommit", type=Path, required=True)
        subparser.add_argument("--checkpoint", type=Path, required=True)
        subparser.add_argument("--record", type=Path, required=True)
    replay.add_argument("--replay-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "v3_plan_path": args.v3_plan.resolve(),
        "v3_development_path": args.v3_development_panels.resolve(),
        "v2_protocol_plan_path": args.v2_training_plan.resolve(),
        "v2_development_labels_path": args.v2_development_labels.resolve(),
        "dataset_root": args.dataset_root.resolve(),
    }
    if args.command == "authorize-fit-exposure":
        result = create_fit_exposure_authorization(
            **common,
            intended_precommit_path=args.intended_precommit.resolve(),
            output_path=args.output.resolve(),
        )
    elif args.command == "precommit-fit":
        result = create_fit_precommit(
            **common,
            authorization_path=args.authorization.resolve(),
            output_path=args.output.resolve(),
        )
    elif args.command == "train-fit":
        result = run_fit_training(
            **common,
            authorization_path=args.authorization.resolve(),
            precommit_path=args.precommit.resolve(),
            checkpoint_path=args.checkpoint.resolve(),
            record_path=args.record.resolve(),
        )
    elif args.command == "replay-fit":
        result = replay_fit_training(
            **common,
            authorization_path=args.authorization.resolve(),
            precommit_path=args.precommit.resolve(),
            checkpoint_path=args.checkpoint.resolve(),
            record_path=args.record.resolve(),
            replay_output_path=args.replay_output.resolve(),
        )
    else:  # pragma: no cover
        raise ActionCountCNNFitError("unknown fit command")
    print(json.dumps({"record_digest": result["record_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
