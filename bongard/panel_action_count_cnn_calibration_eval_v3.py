"""Closed V3 calibration/evaluation runner for the typed panel CNN.

The only public live commands are whole-stage commands.  ``calibrate`` writes
an exposure authorization before touching a calibration PNG, freezes complete
pre-label predictions, derives delayed labels, and freezes the conformal grant.
``evaluate`` requires that grant before it can authorize an evaluation read.
``replay`` reads no PNG and performs no training or inference.

Importing this module reads only its own source through the runtime source seal.
Fresh labels and action programs remain behind the V3 post-prediction barrier.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from collections import Counter
from dataclasses import dataclass
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.panel_action_count_cnn_postprediction_labels_v3 import (
    CatalogTarget,
    LabelAuthorityBindings,
    LabelSources,
    PREDICTION_SCHEMA,
    PredictionBarrier,
    contract_digest as postprediction_contract_digest,
    derive_labels_after_durable_predictions,
    source_digest as postprediction_source_digest,
)


V3_AUTHORITY_COMMIT = "5df5064c"
V3_PLAN_SCHEMA = "gkm.bongard-action-count-catalog-cnn-preregistration.v3"
V3_PLAN_RECORD_DIGEST = (
    "sha256:bb4524a0958cd21f2d4d49bc6a9caa964ccb96c67fbf7c6192185f7b2f363dcb"
)
V3_PLAN_SOURCE_SHA256 = (
    "sha256:71c68771b356658843c3d848cdeea0ba7f2d96fffacd1816ef72934214b055d0"
)
V3_CALIBRATION_MANIFEST_SCHEMA = (
    "gkm.bongard-action-count-cnn-calibration-panel-ids.v3"
)
V3_CALIBRATION_MANIFEST_RECORD_DIGEST = (
    "sha256:17088e6b72544a12829b255b4ada9f3b50e03423595c295185dbcfb02f9f515f"
)
V3_CALIBRATION_MANIFEST_SOURCE_SHA256 = (
    "sha256:d2f891e7fb5236dea5a2609d95c862bae103b3fe0f85724dea7b9b07a1caab9d"
)
V3_EVALUATION_MANIFEST_SCHEMA = (
    "gkm.bongard-action-count-cnn-evaluation-panel-ids.v3"
)
V3_EVALUATION_MANIFEST_RECORD_DIGEST = (
    "sha256:6e0e17a91b48547a83706968d58fbc1ef8c61bbe3f082d8986d9b6bff33678cd"
)
V3_EVALUATION_MANIFEST_SOURCE_SHA256 = (
    "sha256:61472b41332231abc813939a94ec60ad40c1c287a9d138ade6b3642b377f8516"
)
V3_POSTPREDICTION_SOURCE_SHA256 = (
    "sha256:f2b9adc70b3e16794531358e8e80613bb50546752739c2b2ccd8953019850354"
)

# This is the one binding updated after the fit prelaunch audit freezes.  The
# schemas and artifact field contract are already stable; the runner also
# requires the fit precommit and result to agree with this exact source pin.
FINAL_FIT_BINDING: Mapping[str, str] = {
    "architecture_id": "shared-cnn-16-32-64-96-three-head/v1",
    "fit_authorization_schema": (
        "gkm.bongard-action-count-catalog-cnn-fit-exposure-authorization.v2"
    ),
    "fit_precommit_schema": (
        "gkm.bongard-action-count-catalog-cnn-fit-pixel-precommit.v2"
    ),
    "fit_result_schema": "gkm.bongard-action-count-catalog-cnn-fit-result.v2",
    "fit_authorization_record_digest": (
        "sha256:4fd347caba29c41ce1c433319b92efdde7d9857adfa1067cdf83fffec41224ee"
    ),
    "fit_precommit_record_digest": (
        "sha256:e8c7c15fbfb723c5b2305094f035e2567c1fb9b7e80b9f13eeae32fe35d1b15a"
    ),
    "fit_result_record_digest": (
        "sha256:f8b79047228a91fd3fdd47a262299b0cd683daa727981e568450371be4e4dff2"
    ),
    "trainer_source_sha256": (
        "sha256:2706faf07052e580331346ea209c60bc59987366be53f6a729570f0d2cbc9e6a"
    ),
}

AUTHORIZATION_SCHEMA = "gkm.bongard-action-count-cnn-stage-authorization.v3"
PIXEL_PRECOMMIT_SCHEMA = "gkm.bongard-action-count-cnn-stage-pixel-precommit.v3"
LABEL_RECORD_SCHEMA = "gkm.bongard-action-count-cnn-delayed-labels.v3"
CALIBRATION_GRANT_SCHEMA = "gkm.bongard-action-count-cnn-calibration-grant.v3"
EVALUATION_RESULT_SCHEMA = "gkm.bongard-action-count-cnn-evaluation-result.v3"
REPLAY_SCHEMA = "gkm.bongard-action-count-cnn-calibration-eval-replay.v3"

HEADS: tuple[tuple[str, int], ...] = (
    ("straight", 10),
    ("arc", 10),
    ("catalog", 3),
)
CATALOG_CLASS_ORDER = ("catalog_unresolved", "nonconvex", "convex")
CATALOG_TARGET_TO_INDEX = {-1: 0, 0: 1, 1: 2}
PANELS_PER_TASK = 14
CALIBRATION_TASK_COUNT = 100
CALIBRATION_ORDER_INDEX = 95
ALPHA = 0.05
EVALUATION_GATES: Mapping[str, float] = {
    "arc_top1": 0.85,
    "empirical_joint_whole_task_set_coverage": 0.90,
    "known_catalog_binary_balanced_accuracy": 0.70,
    "known_catalog_typed_decisive_rate": 0.30,
    "mean_straight_joint_q_set_size": 4.0,
    "straight_and_known_catalog_joint_exact": 0.55,
    "straight_joint_q_singleton_rate": 0.25,
    "straight_top1": 0.70,
    "true_straight_count_4_joint_q_singleton_rate": 0.25,
}
PREREG_EVALUATION_THRESHOLDS: Mapping[str, float | str] = {
    "arc_top1_at_least": 0.85,
    "empirical_joint_whole-task_set_coverage_at_least": 0.90,
    "known_catalog_binary_balanced_accuracy_at_least": 0.70,
    "known_catalog_typed_decisive_rate_at_least": 0.30,
    "mean_straight_joint-q_set_size_at_most": 4.0,
    "straight_and_known_catalog_joint_exact_at_least": 0.55,
    "straight_joint-q_singleton_rate_at_least": 0.25,
    "straight_top1_at_least": 0.70,
    "true-straight-count-4_joint-q_singleton_rate_at_least": 0.25,
}


class ActionCountCNNV3RunnerError(RuntimeError):
    """A chronology, custody, inference, calibration, or replay edge differs."""


@dataclass(frozen=True)
class V3Authority:
    stage: str
    plan: Mapping[str, Any]
    plan_raw: bytes
    manifest: Mapping[str, Any]
    manifest_raw: bytes
    task_ids: tuple[str, ...]
    panel_ids: tuple[str, ...]


@dataclass(frozen=True)
class FitAuthority:
    authorization: Mapping[str, Any]
    authorization_raw: bytes
    precommit: Mapping[str, Any]
    precommit_raw: bytes
    result: Mapping[str, Any]
    result_raw: bytes
    checkpoint_raw_sha256: str
    checkpoint_state_dict_sha256: str
    config_digest: str
    trainer_source_sha256: str


@dataclass(frozen=True)
class InferenceRow:
    panel_id: str
    straight_logits: tuple[float, ...]
    straight_probabilities: tuple[float, ...]
    arc_logits: tuple[float, ...]
    arc_probabilities: tuple[float, ...]
    catalog_logits: tuple[float, ...]
    catalog_probabilities: tuple[float, ...]


InferenceFunction = Callable[
    [str, Sequence[str], Sequence[bytes], Path, FitAuthority], Sequence[InferenceRow]
]
LabelSourceLoader = Callable[[PredictionBarrier], LabelSources]
MetricStrataLoader = Callable[[PredictionBarrier], Mapping[str, str]]


def _address(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_address(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
    ):
        raise ActionCountCNNV3RunnerError(f"{label} is not a SHA-256 address")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ActionCountCNNV3RunnerError(f"{label} is not hexadecimal") from exc
    return value


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "record_digest": "sha256:" + canonical_digest(body)}


def _load_record(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActionCountCNNV3RunnerError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise ActionCountCNNV3RunnerError(f"{label} is not canonical JSON plus newline")
    body = dict(value)
    found = body.pop("record_digest", None)
    if found != "sha256:" + canonical_digest(body):
        raise ActionCountCNNV3RunnerError(f"{label} record digest differs")
    return value, raw


def _write_once(path: Path, value: Mapping[str, Any]) -> bytes:
    payload = canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing == payload:
            return existing
        raise ActionCountCNNV3RunnerError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".tmp-v3-calibration-eval")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ActionCountCNNV3RunnerError(f"cannot durably write {path}: {exc}") from exc
    if path.read_bytes() != payload:
        raise ActionCountCNNV3RunnerError("durable artifact reload differs")
    return payload


def _runtime_identity() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for key, distribution in (("numpy", "numpy"), ("pillow", "Pillow"), ("torch", "torch")):
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    return {
        **versions,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def runner_source_digest() -> str:
    return "sha256:" + verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def _task_id(panel_id: str) -> str:
    parts = PurePosixPath(panel_id).parts
    if (
        len(parts) != 4
        or parts[0] != "hd"
        or parts[2] not in {"0", "1"}
        or not parts[3].endswith(".png")
        or parts[3][:-4] not in {str(index) for index in range(7)}
    ):
        raise ActionCountCNNV3RunnerError(f"invalid HD panel ID: {panel_id!r}")
    return parts[1]


def _load_v3_authority(
    *, stage: str, plan_path: Path, manifest_path: Path
) -> V3Authority:
    if stage not in {"calibration", "evaluation"}:
        raise ActionCountCNNV3RunnerError("stage leaves calibration/evaluation")
    plan, plan_raw = _load_record(plan_path, label="V3 plan")
    if (
        plan.get("schema") != V3_PLAN_SCHEMA
        or plan.get("record_digest") != V3_PLAN_RECORD_DIGEST
        or _address(plan_raw) != V3_PLAN_SOURCE_SHA256
    ):
        raise ActionCountCNNV3RunnerError("V3 plan is not exact 5df5064c authority")
    expected = {
        "calibration": (
            V3_CALIBRATION_MANIFEST_SCHEMA,
            V3_CALIBRATION_MANIFEST_RECORD_DIGEST,
            V3_CALIBRATION_MANIFEST_SOURCE_SHA256,
        ),
        "evaluation": (
            V3_EVALUATION_MANIFEST_SCHEMA,
            V3_EVALUATION_MANIFEST_RECORD_DIGEST,
            V3_EVALUATION_MANIFEST_SOURCE_SHA256,
        ),
    }[stage]
    manifest, manifest_raw = _load_record(manifest_path, label=f"V3 {stage} manifest")
    if (
        manifest.get("schema") != expected[0]
        or manifest.get("record_digest") != expected[1]
        or _address(manifest_raw) != expected[2]
    ):
        raise ActionCountCNNV3RunnerError(f"{stage} is not the exact V3 manifest")
    bound = plan["identifier_manifest_bindings"][f"{stage}_panel_ids"]
    if bound.get("record_digest") != expected[1] or bound.get("source_sha256") != expected[2]:
        raise ActionCountCNNV3RunnerError(f"V3 plan does not bind {stage} manifest")
    cohort = manifest.get("cohorts", {}).get(stage)
    if not isinstance(cohort, dict):
        raise ActionCountCNNV3RunnerError(f"{stage} manifest cohort is missing")
    task_ids = cohort.get("task_ids")
    panel_ids = cohort.get("panel_ids")
    if (
        not isinstance(task_ids, list)
        or not isinstance(panel_ids, list)
        or len(task_ids) != 100
        or len(panel_ids) != 1_400
        or len(set(task_ids)) != len(task_ids)
        or len(set(panel_ids)) != len(panel_ids)
        or any(not isinstance(value, str) for value in (*task_ids, *panel_ids))
    ):
        raise ActionCountCNNV3RunnerError(f"{stage} manifest cardinality differs")
    expected_order = [task_id for task_id in task_ids for _ in range(PANELS_PER_TASK)]
    if [_task_id(panel_id) for panel_id in panel_ids] != expected_order:
        raise ActionCountCNNV3RunnerError(f"{stage} panel order is not task-major 14-panel")
    plan_cohort = plan["cohorts"][stage]
    if (
        plan_cohort.get("task_ids") != task_ids
        or plan_cohort.get("task_count") != 100
        or plan_cohort.get("panel_count") != 1_400
        or plan_cohort.get("panel_ids_digest")
        != "sha256:" + canonical_digest(panel_ids)
    ):
        raise ActionCountCNNV3RunnerError(f"{stage} plan/manifest inventory differs")
    return V3Authority(
        stage=stage,
        plan=plan,
        plan_raw=plan_raw,
        manifest=manifest,
        manifest_raw=manifest_raw,
        task_ids=tuple(task_ids),
        panel_ids=tuple(panel_ids),
    )


def _fit_source_pin() -> str:
    value = FINAL_FIT_BINDING["trainer_source_sha256"]
    if value == "PENDING_FINAL_AUDIT_SOURCE_SHA256":
        raise ActionCountCNNV3RunnerError(
            "final audited trainer source pin has not been installed"
        )
    return _require_address(value, "final trainer source")


def _verify_fit_authority(
    *,
    authority: V3Authority,
    fit_authorization_path: Path,
    fit_precommit_path: Path,
    fit_result_path: Path,
    checkpoint_path: Path,
) -> FitAuthority:
    authorization, authorization_raw = _load_record(
        fit_authorization_path, label="fit exposure authorization"
    )
    precommit, precommit_raw = _load_record(fit_precommit_path, label="fit precommit")
    result, result_raw = _load_record(fit_result_path, label="fit result")
    trainer_source = _fit_source_pin()
    if authorization.get("schema") != FINAL_FIT_BINDING["fit_authorization_schema"]:
        raise ActionCountCNNV3RunnerError("fit authorization schema differs")
    if precommit.get("schema") != FINAL_FIT_BINDING["fit_precommit_schema"]:
        raise ActionCountCNNV3RunnerError("fit precommit schema differs")
    if result.get("schema") != FINAL_FIT_BINDING["fit_result_schema"]:
        raise ActionCountCNNV3RunnerError("fit result schema differs")
    if (
        authorization.get("record_digest")
        != FINAL_FIT_BINDING["fit_authorization_record_digest"]
        or precommit.get("record_digest")
        != FINAL_FIT_BINDING["fit_precommit_record_digest"]
        or result.get("record_digest") != FINAL_FIT_BINDING["fit_result_record_digest"]
    ):
        raise ActionCountCNNV3RunnerError("fit artifacts are not the final V3 run")
    if (
        authorization.get("v3_plan_record_digest") != authority.plan["record_digest"]
        or precommit.get("authorization_record_digest") != authorization["record_digest"]
        or precommit.get("authorization_source_sha256") != _address(authorization_raw)
        or
        precommit.get("v3_plan_record_digest") != authority.plan["record_digest"]
        or result.get("v3_plan_record_digest") != authority.plan["record_digest"]
        or result.get("fit_precommit_record_digest") != precommit["record_digest"]
    ):
        raise ActionCountCNNV3RunnerError("fit artifacts do not bind exact V3 plan/precommit")
    if (
        precommit.get("trainer_source_sha256") != trainer_source
        or result.get("trainer_source_sha256", trainer_source) != trainer_source
    ):
        raise ActionCountCNNV3RunnerError("fit artifacts do not bind final trainer source")
    gate = result.get("validation_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("passed") is not True
        or not isinstance(gate.get("checks"), dict)
        or set(gate["checks"]) != {
            "arc_top1",
            "known_catalog_binary_balanced_accuracy",
            "straight_top1",
        }
        or not all(value is True for value in gate["checks"].values())
    ):
        raise ActionCountCNNV3RunnerError(
            "fit validation gate did not pass exactly; fresh pixels remain unauthorized"
        )
    adaptive = result.get("adaptive_post_exposure_development_correction")
    if (
        not isinstance(adaptive, dict)
        or set(adaptive)
        != {
            "effective_training_panel_count",
            "effective_validation_class_counts",
            "effective_validation_panel_count",
            "validation_decontamination_gate",
            "validation_removed_due_exact_train_duplicate",
        }
        or adaptive.get("validation_decontamination_gate", {}).get("passed") is not True
        or adaptive.get("effective_training_panel_count") != 11_200
        or not isinstance(adaptive.get("effective_validation_panel_count"), int)
        or adaptive["effective_validation_panel_count"] != 1_392
        or len(adaptive.get("validation_removed_due_exact_train_duplicate", [])) != 8
    ):
        raise ActionCountCNNV3RunnerError("fit validation decontamination gate differs")
    if (
        precommit.get("validation_decontamination_gate")
        != adaptive["validation_decontamination_gate"]
        or precommit.get("effective_validation_panel_count")
        != adaptive["effective_validation_panel_count"]
        or precommit.get("validation_removed_due_exact_train_duplicate")
        != adaptive["validation_removed_due_exact_train_duplicate"]
        or precommit.get("effective_validation_class_counts")
        != adaptive["effective_validation_class_counts"]
    ):
        raise ActionCountCNNV3RunnerError("fit precommit/result decontamination differs")
    if result.get("architecture_id") != FINAL_FIT_BINDING["architecture_id"]:
        raise ActionCountCNNV3RunnerError("fit architecture differs")
    checkpoint_raw = checkpoint_path.read_bytes()
    checkpoint_raw_sha = _address(checkpoint_raw)
    if result.get("checkpoint_raw_sha256") != checkpoint_raw_sha:
        raise ActionCountCNNV3RunnerError("checkpoint raw bytes differ from fit result")
    for field in ("checkpoint_state_dict_sha256", "config_digest"):
        _require_address(result.get(field), f"fit {field}")
    # Loading the checkpoint is deliberately delegated to the final audited
    # trainer implementation, then checked against the fit result.
    from bongard import panel_action_count_cnn_train_command as trainer

    current_trainer_source = "sha256:" + trainer.verify_loaded_source(
        trainer.__name__, expected_source_sha256=trainer._LOADED_SOURCE_SHA256
    )
    if current_trainer_source != trainer_source:
        raise ActionCountCNNV3RunnerError("loaded trainer source differs from final pin")
    checkpoint, _ = trainer._load_checkpoint(
        checkpoint_path, expected_raw_sha256=checkpoint_raw_sha
    )
    if trainer.state_dict_digest(checkpoint["state_dict"]) != result["checkpoint_state_dict_sha256"]:
        raise ActionCountCNNV3RunnerError("checkpoint state digest differs")
    if checkpoint.get("config_digest") != result["config_digest"]:
        raise ActionCountCNNV3RunnerError("checkpoint config differs")
    return FitAuthority(
        authorization=authorization,
        authorization_raw=authorization_raw,
        precommit=precommit,
        precommit_raw=precommit_raw,
        result=result,
        result_raw=result_raw,
        checkpoint_raw_sha256=checkpoint_raw_sha,
        checkpoint_state_dict_sha256=result["checkpoint_state_dict_sha256"],
        config_digest=result["config_digest"],
        trainer_source_sha256=trainer_source,
    )


def _custody(authority: V3Authority, fit: FitAuthority) -> dict[str, Any]:
    return {
        "checkpoint_raw_sha256": fit.checkpoint_raw_sha256,
        "checkpoint_state_dict_sha256": fit.checkpoint_state_dict_sha256,
        "config_digest": fit.config_digest,
        "fit_authorization_record_digest": fit.authorization["record_digest"],
        "fit_authorization_source_sha256": _address(fit.authorization_raw),
        "fit_precommit_record_digest": fit.precommit["record_digest"],
        "fit_precommit_source_sha256": _address(fit.precommit_raw),
        "fit_result_record_digest": fit.result["record_digest"],
        "fit_result_source_sha256": _address(fit.result_raw),
        "panel_manifest_record_digest": authority.manifest["record_digest"],
        "panel_manifest_source_sha256": _address(authority.manifest_raw),
        "plan_record_digest": authority.plan["record_digest"],
        "plan_source_sha256": _address(authority.plan_raw),
        "postprediction_contract_digest": postprediction_contract_digest(),
        "postprediction_source_sha256": postprediction_source_digest(),
        "runner_source_sha256": runner_source_digest(),
        "trainer_source_sha256": fit.trainer_source_sha256,
        "v3_authority_commit": V3_AUTHORITY_COMMIT,
    }


def _verify_postprediction_binding(authority: V3Authority) -> LabelAuthorityBindings:
    binding = authority.plan.get("postprediction_target_authority")
    if not isinstance(binding, dict):
        raise ActionCountCNNV3RunnerError("postprediction authority is missing")
    if (
        binding.get("source_sha256") != V3_POSTPREDICTION_SOURCE_SHA256
        or postprediction_source_digest() != V3_POSTPREDICTION_SOURCE_SHA256
    ):
        raise ActionCountCNNV3RunnerError("postprediction source differs from V3 plan")
    frozen = binding.get("frozen_label_source_bindings")
    if not isinstance(frozen, dict):
        raise ActionCountCNNV3RunnerError("label source bindings are missing")
    return LabelAuthorityBindings(
        hd_action_program_raw_sha256=frozen["hd_action_program_raw_sha256"],
        catalog_algorithm_digest=frozen["catalog_algorithm_digest"],
        catalog_audit_record_digest=frozen["catalog_audit_record_digest"],
        catalog_authority_source_sha256=frozen["catalog_authority_source_sha256"],
    )


def _authorize_stage(
    *,
    authority: V3Authority,
    fit: FitAuthority,
    output_path: Path,
    calibration_grant: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if authority.stage == "calibration" and calibration_grant is not None:
        raise ActionCountCNNV3RunnerError("calibration authorization cannot consume a grant")
    if authority.stage == "evaluation" and calibration_grant is None:
        raise ActionCountCNNV3RunnerError("evaluation cannot be authorized before grant freeze")
    body: dict[str, Any] = {
        "allowed_reads": {
            "panel_count": len(authority.panel_ids),
            "panel_ids_digest": "sha256:" + canonical_digest(list(authority.panel_ids)),
            "stage": authority.stage,
        },
        "calibration_grant_record_digest": (
            None if calibration_grant is None else calibration_grant["record_digest"]
        ),
        "chronology": (
            "fit-validation-pass_then-write-authorization-before-any-stage-pixel-read"
            if authority.stage == "calibration"
            else "frozen-calibration-grant_then-write-authorization-before-any-stage-pixel-read"
        ),
        "custody": _custody(authority, fit),
        "forbidden_reads": [
            "v2_calibration_panel_PNGs",
            "v2_evaluation_panel_PNGs",
            "official_validation_or_test_PNGs",
            "other_V3_stage_PNGs",
        ],
        "partial_exposure_policy": (
            "authorization_survives_any_later_failure_and_records_the_entire_"
            "possibly_partially_exposed_stage_inventory"
        ),
        "runtime": _runtime_identity(),
        "schema": AUTHORIZATION_SCHEMA,
        "stage": authority.stage,
    }
    final = _seal(body)
    _write_once(output_path, final)
    reloaded, _ = _load_record(output_path, label=f"{authority.stage} authorization")
    if reloaded != final:
        raise ActionCountCNNV3RunnerError("authorization fresh reload differs")
    return final


def _panel_path(dataset_root: Path, panel_id: str) -> Path:
    _task_id(panel_id)
    parts = PurePosixPath(panel_id).parts
    candidate = dataset_root.joinpath(parts[0], "images", *parts[1:])
    try:
        root = dataset_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ActionCountCNNV3RunnerError(f"cannot resolve stage panel: {exc}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ActionCountCNNV3RunnerError("stage panel escapes dataset root")
    return resolved


def filesystem_panel_reader(dataset_root: Path) -> Callable[[str], bytes]:
    root = dataset_root.resolve()

    def read(panel_id: str) -> bytes:
        try:
            return _panel_path(root, panel_id).read_bytes()
        except OSError as exc:
            raise ActionCountCNNV3RunnerError(f"cannot read stage PNG: {exc}") from exc

    return read


def _create_pixel_precommit(
    *,
    authority: V3Authority,
    fit: FitAuthority,
    authorization: Mapping[str, Any],
    panel_reader: Callable[[str], bytes],
    output_path: Path,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for panel_id in authority.panel_ids:
        raw = panel_reader(panel_id)
        if not isinstance(raw, bytes) or not raw:
            raise ActionCountCNNV3RunnerError("panel reader did not return nonempty bytes")
        observations.append(
            {
                "panel_id": panel_id,
                "png_sha256": _address(raw),
                "png_size_bytes": len(raw),
            }
        )
    body = {
        "authorization_record_digest": authorization["record_digest"],
        "custody": _custody(authority, fit),
        "exact_png_observations": observations,
        "panel_count": len(observations),
        "panel_order_digest": "sha256:" + canonical_digest(list(authority.panel_ids)),
        "schema": PIXEL_PRECOMMIT_SCHEMA,
        "stage": authority.stage,
    }
    final = _seal(body)
    _write_once(output_path, final)
    reloaded, _ = _load_record(output_path, label=f"{authority.stage} pixel precommit")
    if reloaded != final:
        raise ActionCountCNNV3RunnerError("pixel precommit fresh reload differs")
    return final


def _reload_precommitted_pixels(
    *,
    authority: V3Authority,
    precommit: Mapping[str, Any],
    panel_reader: Callable[[str], bytes],
) -> tuple[bytes, ...]:
    rows = precommit.get("exact_png_observations")
    if not isinstance(rows, list) or len(rows) != len(authority.panel_ids):
        raise ActionCountCNNV3RunnerError("pixel precommit inventory differs")
    result: list[bytes] = []
    for panel_id, observation in zip(authority.panel_ids, rows):
        if not isinstance(observation, dict) or observation.get("panel_id") != panel_id:
            raise ActionCountCNNV3RunnerError("pixel precommit order differs")
        raw = panel_reader(panel_id)
        if (
            not isinstance(raw, bytes)
            or _address(raw) != observation.get("png_sha256")
            or len(raw) != observation.get("png_size_bytes")
        ):
            raise ActionCountCNNV3RunnerError("stage PNG changed after precommit")
        result.append(raw)
    return tuple(result)


def _finite_vector(values: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    if len(values) != size:
        raise ActionCountCNNV3RunnerError(f"{label} class count differs")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ActionCountCNNV3RunnerError(f"{label} contains nonfinite value")
    return result


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    largest = max(logits)
    exponentials = [math.exp(value - largest) for value in logits]
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


def _prediction_row_data(row: InferenceRow, expected_panel_id: str) -> dict[str, Any]:
    if not isinstance(row, InferenceRow) or row.panel_id != expected_panel_id:
        raise ActionCountCNNV3RunnerError("inference row order differs")
    result: dict[str, Any] = {"panel_id": expected_panel_id}
    for name, size in HEADS:
        logits = _finite_vector(getattr(row, f"{name}_logits"), size, f"{name} logits")
        probabilities = _finite_vector(
            getattr(row, f"{name}_probabilities"), size, f"{name} probabilities"
        )
        expected = _softmax(logits)
        if any(
            not 0.0 <= value <= 1.0 or abs(value - target) > 1e-7
            for value, target in zip(probabilities, expected)
        ):
            raise ActionCountCNNV3RunnerError(f"{name} probabilities differ from logits")
        result[f"{name}_logits"] = list(logits)
        result[f"{name}_probabilities"] = list(probabilities)
    return result


def default_inference(
    stage: str,
    panel_ids: Sequence[str],
    raws: Sequence[bytes],
    checkpoint_path: Path,
    fit: FitAuthority,
) -> Sequence[InferenceRow]:
    """CPU-only inference using the final audited trainer; never trains."""

    if stage not in {"calibration", "evaluation"} or len(panel_ids) != len(raws):
        raise ActionCountCNNV3RunnerError("default inference inputs differ")
    from bongard import panel_action_count_cnn_train_command as trainer

    torch, _, _ = trainer._torch_runtime()
    checkpoint, _ = trainer._load_checkpoint(
        checkpoint_path, expected_raw_sha256=fit.checkpoint_raw_sha256
    )
    if trainer.state_dict_digest(checkpoint["state_dict"]) != fit.checkpoint_state_dict_sha256:
        raise ActionCountCNNV3RunnerError("inference checkpoint state differs")
    model = trainer.build_model(seed=260810)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    output: list[InferenceRow] = []
    with torch.no_grad():
        for start in range(0, len(raws), 64):
            arrays = [trainer.preprocess_png_bytes(raw) for raw in raws[start : start + 64]]
            import numpy as np

            pixels = torch.from_numpy(np.stack(arrays)[:, None]).to(torch.float32) / 255.0
            tensors = model(pixels)
            probabilities = tuple(torch.softmax(tensor, dim=1) for tensor in tensors)
            for offset, panel_id in enumerate(panel_ids[start : start + 64]):
                values: list[tuple[float, ...]] = []
                probs: list[tuple[float, ...]] = []
                for tensor, probability in zip(tensors, probabilities):
                    values.append(tuple(float(item) for item in tensor[offset].tolist()))
                    probs.append(tuple(float(item) for item in probability[offset].tolist()))
                output.append(
                    InferenceRow(
                        panel_id=panel_id,
                        straight_logits=values[0],
                        straight_probabilities=probs[0],
                        arc_logits=values[1],
                        arc_probabilities=probs[1],
                        catalog_logits=values[2],
                        catalog_probabilities=probs[2],
                    )
                )
    return output


def _create_prediction_record(
    *,
    authority: V3Authority,
    fit: FitAuthority,
    authorization: Mapping[str, Any],
    pixel_precommit: Mapping[str, Any],
    inference_rows: Sequence[InferenceRow],
    calibration_grant: Mapping[str, Any] | None,
    output_path: Path,
) -> dict[str, Any]:
    if len(inference_rows) != len(authority.panel_ids):
        raise ActionCountCNNV3RunnerError("inference row count differs")
    rows = [
        _prediction_row_data(row, panel_id)
        for row, panel_id in zip(inference_rows, authority.panel_ids)
    ]
    if authority.stage == "evaluation":
        if calibration_grant is None:
            raise ActionCountCNNV3RunnerError("evaluation predictions require grant")
        q = float(calibration_grant["deployment_joint_q"])
        for row in rows:
            for name, size in HEADS:
                probabilities = row[f"{name}_probabilities"]
                row[f"{name}_class_set"] = [
                    index for index in range(size) if 1.0 - probabilities[index] <= q
                ]
        q_digest = calibration_grant["record_digest"]
    else:
        if calibration_grant is not None:
            raise ActionCountCNNV3RunnerError("calibration predictions cannot consume q")
        q = None
        q_digest = None
    body = {
        "arc_class_order": list(range(10)),
        "authorization_record_digest": authorization["record_digest"],
        "catalog_class_order": list(CATALOG_CLASS_ORDER),
        "checkpoint_state_dict_sha256": fit.checkpoint_state_dict_sha256,
        "config_digest": fit.config_digest,
        "custody": _custody(authority, fit),
        "joint_q": q,
        "joint_q_record_digest": q_digest,
        "panel_ids": list(authority.panel_ids),
        "panel_manifest_record_digest": authority.manifest["record_digest"],
        "pixel_precommit_record_digest": pixel_precommit["record_digest"],
        "plan_record_digest": authority.plan["record_digest"],
        "rows": rows,
        "schema": PREDICTION_SCHEMA,
        "stage": authority.stage,
        "straight_class_order": list(range(10)),
    }
    final = _seal(body)
    _write_once(output_path, final)
    reloaded, _ = _load_record(output_path, label=f"{authority.stage} predictions")
    if reloaded != final:
        raise ActionCountCNNV3RunnerError("prediction fresh reload differs")
    return final


def _derive_and_freeze_labels(
    *,
    authority: V3Authority,
    fit: FitAuthority,
    prediction_path: Path,
    label_source_loader: LabelSourceLoader,
    metric_strata_loader: MetricStrataLoader | None,
    output_path: Path,
) -> dict[str, Any]:
    bindings = _verify_postprediction_binding(authority)
    rows = derive_labels_after_durable_predictions(
        prediction_path=prediction_path,
        expected_stage=authority.stage,
        expected_panel_ids=authority.panel_ids,
        expected_plan_record_digest=authority.plan["record_digest"],
        expected_panel_manifest_record_digest=authority.manifest["record_digest"],
        expected_checkpoint_state_dict_sha256=fit.checkpoint_state_dict_sha256,
        expected_config_digest=fit.config_digest,
        expected_label_authority_bindings=bindings,
        source_loader=label_source_loader,
    )
    prediction, prediction_raw = _load_record(
        prediction_path, label=f"{authority.stage} durable predictions"
    )
    if [row.get("panel_id") for row in rows] != list(authority.panel_ids):
        raise ActionCountCNNV3RunnerError("delayed label order differs")
    frozen_rows = [dict(row) for row in rows]
    if metric_strata_loader is not None:
        barrier = PredictionBarrier(
            stage=authority.stage,
            panel_ids=authority.panel_ids,
            prediction_record_digest=prediction["record_digest"],
            prediction_source_sha256=_address(prediction_raw),
            checkpoint_state_dict_sha256=fit.checkpoint_state_dict_sha256,
            config_digest=fit.config_digest,
        )
        line_profiles = metric_strata_loader(barrier)
        if (
            not isinstance(line_profiles, Mapping)
            or set(line_profiles) != set(authority.panel_ids)
            or any(
                value
                not in {
                    "no_straight_actions",
                    "normal_only",
                    "decorated_only",
                    "mixed_normal_and_decorated",
                }
                for value in line_profiles.values()
            )
        ):
            raise ActionCountCNNV3RunnerError("delayed line-decoration strata differ")
        for row in frozen_rows:
            task_id = _task_id(row["panel_id"])
            row["metric_strata"] = {
                "crossing_task": "has_line_crossing" in task_id,
                "line_decoration": line_profiles[row["panel_id"]],
                "thin_task": "thin_shape" in task_id,
            }
    body = {
        "checkpoint_state_dict_sha256": fit.checkpoint_state_dict_sha256,
        "config_digest": fit.config_digest,
        "custody": _custody(authority, fit),
        "label_authority_bindings": {
            "catalog_algorithm_digest": bindings.catalog_algorithm_digest,
            "catalog_audit_record_digest": bindings.catalog_audit_record_digest,
            "catalog_authority_source_sha256": bindings.catalog_authority_source_sha256,
            "hd_action_program_raw_sha256": bindings.hd_action_program_raw_sha256,
        },
        "panel_manifest_record_digest": authority.manifest["record_digest"],
        "prediction_record_digest": prediction["record_digest"],
        "prediction_source_sha256": _address(prediction_raw),
        "rows": frozen_rows,
        "schema": LABEL_RECORD_SCHEMA,
        "stage": authority.stage,
    }
    final = _seal(body)
    _write_once(output_path, final)
    reloaded, _ = _load_record(output_path, label=f"{authority.stage} labels")
    if reloaded != final:
        raise ActionCountCNNV3RunnerError("label record fresh reload differs")
    return final


def _paired_rows(
    prediction: Mapping[str, Any], labels: Mapping[str, Any], *, stage: str
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    if prediction.get("stage") != stage or labels.get("stage") != stage:
        raise ActionCountCNNV3RunnerError("prediction/label stage differs")
    if labels.get("prediction_record_digest") != prediction.get("record_digest"):
        raise ActionCountCNNV3RunnerError("labels do not bind predictions")
    prediction_rows = prediction.get("rows")
    label_rows = labels.get("rows")
    if (
        not isinstance(prediction_rows, list)
        or not isinstance(label_rows, list)
        or len(prediction_rows) != len(label_rows)
    ):
        raise ActionCountCNNV3RunnerError("prediction/label row counts differ")
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for predicted, labelled in zip(prediction_rows, label_rows):
        if (
            not isinstance(predicted, dict)
            or not isinstance(labelled, dict)
            or predicted.get("panel_id") != labelled.get("panel_id")
        ):
            raise ActionCountCNNV3RunnerError("prediction/label panel order differs")
        straight = labelled.get("straight_action_count")
        arc = labelled.get("arc_action_count")
        catalog = labelled.get("catalog_convexity_target")
        if (
            isinstance(straight, bool)
            or not isinstance(straight, int)
            or straight not in range(10)
            or isinstance(arc, bool)
            or not isinstance(arc, int)
            or arc not in range(10)
            or catalog not in CATALOG_TARGET_TO_INDEX
        ):
            raise ActionCountCNNV3RunnerError("delayed label leaves closed classes")
        pairs.append((predicted, labelled))
    return tuple(pairs)


def _true_indices(label: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(label["straight_action_count"]),
        int(label["arc_action_count"]),
        CATALOG_TARGET_TO_INDEX[int(label["catalog_convexity_target"])],
    )


def calibration_scores(
    *,
    task_ids: Sequence[str],
    prediction_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute task-max joint and per-head scores in exact task order."""

    if (
        len(prediction_rows) != len(label_rows)
        or len(prediction_rows) != len(task_ids) * PANELS_PER_TASK
    ):
        raise ActionCountCNNV3RunnerError("calibration score cardinality differs")
    task_records: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(task_ids):
        start = task_index * PANELS_PER_TASK
        per_head = {name: 0.0 for name, _ in HEADS}
        for offset in range(PANELS_PER_TASK):
            predicted = prediction_rows[start + offset]
            labelled = label_rows[start + offset]
            if (
                predicted.get("panel_id") != labelled.get("panel_id")
                or _task_id(str(predicted.get("panel_id"))) != task_id
            ):
                raise ActionCountCNNV3RunnerError("calibration task grouping differs")
            for (name, _), true_class in zip(HEADS, _true_indices(labelled)):
                probabilities = predicted.get(f"{name}_probabilities")
                if not isinstance(probabilities, list):
                    raise ActionCountCNNV3RunnerError("calibration probabilities missing")
                score = 1.0 - float(probabilities[true_class])
                per_head[name] = max(per_head[name], score)
        task_records.append(
            {
                "head_scores": per_head,
                "joint_score": max(per_head.values()),
                "task_id": task_id,
            }
        )
    sorted_joint = sorted(record["joint_score"] for record in task_records)
    sorted_heads = {
        name: sorted(record["head_scores"][name] for record in task_records)
        for name, _ in HEADS
    }
    return {
        "sorted_head_task_scores": sorted_heads,
        "sorted_joint_task_scores": sorted_joint,
        "task_scores_in_manifest_order": task_records,
    }


def _calibration_grant_body(
    *,
    authority: V3Authority,
    fit: FitAuthority,
    prediction: Mapping[str, Any],
    prediction_raw: bytes,
    labels: Mapping[str, Any],
    labels_raw: bytes,
) -> dict[str, Any]:
    pairs = _paired_rows(prediction, labels, stage="calibration")
    if len(authority.task_ids) != CALIBRATION_TASK_COUNT:
        raise ActionCountCNNV3RunnerError("calibration task count differs from q96 plan")
    scores = calibration_scores(
        task_ids=authority.task_ids,
        prediction_rows=[pair[0] for pair in pairs],
        label_rows=[pair[1] for pair in pairs],
    )
    if math.ceil((CALIBRATION_TASK_COUNT + 1) * (1.0 - ALPHA)) != 96:
        raise ActionCountCNNV3RunnerError("frozen conformal rank arithmetic differs")
    sorted_joint = scores["sorted_joint_task_scores"]
    sorted_heads = scores["sorted_head_task_scores"]
    return {
        "alpha": ALPHA,
        "calibration_label_record_digest": labels["record_digest"],
        "calibration_label_source_sha256": _address(labels_raw),
        "calibration_prediction_record_digest": prediction["record_digest"],
        "calibration_prediction_source_sha256": _address(prediction_raw),
        "calibration_task_count": CALIBRATION_TASK_COUNT,
        "canonical_deployment_q": "joint_q_only",
        "checkpoint_state_dict_sha256": fit.checkpoint_state_dict_sha256,
        "config_digest": fit.config_digest,
        "conformal_claim": (
            "split-conformal marginal whole-task 95-percent target under exchangeability; "
            "not deterministic visual truth and not guaranteed under target-family shift"
        ),
        "custody": _custody(authority, fit),
        "deployment_joint_q": sorted_joint[CALIBRATION_ORDER_INDEX],
        "diagnostic_head_q": {
            name: sorted_heads[name][CALIBRATION_ORDER_INDEX] for name, _ in HEADS
        },
        "individual_head_q_values_are_diagnostics_only": True,
        "no_interpolation": True,
        "order_statistic_one_indexed": 96,
        "q_rule": "sorted_scores[95]",
        "schema": CALIBRATION_GRANT_SCHEMA,
        **scores,
    }


def _freeze_calibration_grant(
    *,
    authority: V3Authority,
    fit: FitAuthority,
    prediction_path: Path,
    label_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    prediction, prediction_raw = _load_record(
        prediction_path, label="calibration predictions"
    )
    labels, labels_raw = _load_record(label_path, label="calibration labels")
    body = _calibration_grant_body(
        authority=authority,
        fit=fit,
        prediction=prediction,
        prediction_raw=prediction_raw,
        labels=labels,
        labels_raw=labels_raw,
    )
    final = _seal(body)
    _write_once(output_path, final)
    reloaded, _ = _load_record(output_path, label="calibration grant")
    if reloaded != final:
        raise ActionCountCNNV3RunnerError("calibration grant fresh reload differs")
    return final


def _verify_calibration_grant(
    *,
    authority: V3Authority,
    fit: FitAuthority,
    prediction_path: Path,
    label_path: Path,
    grant_path: Path,
) -> dict[str, Any]:
    prediction, prediction_raw = _load_record(
        prediction_path, label="archived calibration predictions"
    )
    labels, labels_raw = _load_record(label_path, label="archived calibration labels")
    grant, _ = _load_record(grant_path, label="archived calibration grant")
    expected = _seal(
        _calibration_grant_body(
            authority=authority,
            fit=fit,
            prediction=prediction,
            prediction_raw=prediction_raw,
            labels=labels,
            labels_raw=labels_raw,
        )
    )
    if grant != expected:
        raise ActionCountCNNV3RunnerError("calibration grant differs from cold recomputation")
    return grant


def _confusion(truth: Sequence[int], predicted: Sequence[int], size: int) -> list[list[int]]:
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    if len(truth) != len(predicted):
        raise ActionCountCNNV3RunnerError("confusion vectors differ")
    for expected, found in zip(truth, predicted):
        matrix[expected][found] += 1
    return matrix


def _mean(values: Sequence[float], label: str) -> float:
    if not values:
        raise ActionCountCNNV3RunnerError(f"{label} denominator is empty")
    return sum(values) / len(values)


def _selection_metrics(
    indices: Sequence[int],
    *,
    straight_truth: Sequence[int],
    straight_top1: Sequence[int],
    straight_sets: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if not indices:
        return {"panel_count": 0, "status": "empty-stratum"}
    return {
        "mean_straight_set_size": _mean(
            [float(len(straight_sets[index])) for index in indices], "stratum set size"
        ),
        "panel_count": len(indices),
        "straight_set_coverage": _mean(
            [float(straight_truth[index] in straight_sets[index]) for index in indices],
            "stratum coverage",
        ),
        "straight_singleton_rate": _mean(
            [float(len(straight_sets[index]) == 1) for index in indices],
            "stratum singleton",
        ),
        "straight_top1": _mean(
            [float(straight_top1[index] == straight_truth[index]) for index in indices],
            "stratum top1",
        ),
    }


def evaluation_metrics(
    *,
    task_ids: Sequence[str],
    prediction_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute every preregistered metric, confusion, and required stratum."""

    if (
        len(prediction_rows) != len(label_rows)
        or len(prediction_rows) != len(task_ids) * PANELS_PER_TASK
    ):
        raise ActionCountCNNV3RunnerError("evaluation metric cardinality differs")
    truths = [[], [], []]
    top1 = [[], [], []]
    sets = [[], [], []]
    strata: list[Mapping[str, Any]] = []
    for predicted, labelled in zip(prediction_rows, label_rows):
        if predicted.get("panel_id") != labelled.get("panel_id"):
            raise ActionCountCNNV3RunnerError("evaluation row order differs")
        indices = _true_indices(labelled)
        metric_strata = labelled.get("metric_strata")
        if not isinstance(metric_strata, dict):
            raise ActionCountCNNV3RunnerError("evaluation metric strata were not delayed")
        strata.append(metric_strata)
        for head_index, ((name, size), true_class) in enumerate(zip(HEADS, indices)):
            probabilities = predicted.get(f"{name}_probabilities")
            class_set = predicted.get(f"{name}_class_set")
            if (
                not isinstance(probabilities, list)
                or len(probabilities) != size
                or not isinstance(class_set, list)
            ):
                raise ActionCountCNNV3RunnerError("evaluation prediction row differs")
            truths[head_index].append(true_class)
            top1[head_index].append(max(range(size), key=probabilities.__getitem__))
            sets[head_index].append(tuple(class_set))
    panel_count = len(label_rows)
    known = [index for index, value in enumerate(truths[2]) if value in {1, 2}]
    recalls: list[float] = []
    for value in (1, 2):
        selected = [index for index in known if truths[2][index] == value]
        recalls.append(
            _mean([float(top1[2][index] == value) for index in selected], "catalog recall")
        )
    task_coverage: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(task_ids):
        selected = range(task_index * PANELS_PER_TASK, (task_index + 1) * PANELS_PER_TASK)
        covered = all(
            truths[head][index] in sets[head][index]
            for index in selected
            for head in range(3)
        )
        task_coverage.append({"covered": covered, "task_id": task_id})
    metric_values = {
        "arc_top1": _mean(
            [float(a == b) for a, b in zip(truths[1], top1[1])], "arc top1"
        ),
        "empirical_joint_whole_task_set_coverage": _mean(
            [float(row["covered"]) for row in task_coverage], "task coverage"
        ),
        "known_catalog_binary_balanced_accuracy": sum(recalls) / 2.0,
        "known_catalog_typed_decisive_rate": _mean(
            [float(sets[2][index] in {(1,), (2,)}) for index in known],
            "known catalog decisive",
        ),
        "mean_straight_joint_q_set_size": _mean(
            [float(len(value)) for value in sets[0]], "straight set size"
        ),
        "straight_and_known_catalog_joint_exact": _mean(
            [
                float(
                    top1[0][index] == truths[0][index]
                    and top1[2][index] == truths[2][index]
                )
                for index in known
            ],
            "straight/catalog joint exact",
        ),
        "straight_joint_q_singleton_rate": _mean(
            [float(len(value) == 1) for value in sets[0]], "straight singleton"
        ),
        "straight_top1": _mean(
            [float(a == b) for a, b in zip(truths[0], top1[0])], "straight top1"
        ),
        "true_straight_count_4_joint_q_singleton_rate": _mean(
            [float(len(sets[0][index]) == 1) for index in range(panel_count) if truths[0][index] == 4],
            "count4 singleton",
        ),
    }
    selections: dict[str, list[int]] = {
        "straight_true_count_4": [index for index, value in enumerate(truths[0]) if value == 4],
        "thin_shape_task_name": [index for index, value in enumerate(strata) if value.get("thin_task") is True],
        "has_line_crossing_task_name": [index for index, value in enumerate(strata) if value.get("crossing_task") is True],
    }
    for profile in (
        "no_straight_actions",
        "normal_only",
        "decorated_only",
        "mixed_normal_and_decorated",
    ):
        selections[f"line_decoration:{profile}"] = [
            index for index, value in enumerate(strata) if value.get("line_decoration") == profile
        ]
    required_strata = {
        name: _selection_metrics(
            indices,
            straight_truth=truths[0],
            straight_top1=top1[0],
            straight_sets=sets[0],
        )
        for name, indices in selections.items()
    }
    unresolved = [index for index, value in enumerate(truths[2]) if value == 0]
    required_strata["known_catalog_binary_rows"] = {
        "panel_count": len(known),
        "balanced_accuracy": metric_values["known_catalog_binary_balanced_accuracy"],
        "typed_decisive_rate": metric_values["known_catalog_typed_decisive_rate"],
    }
    required_strata["catalog_unresolved_rows"] = {
        "empirical_coverage": _mean(
            [float(0 in sets[2][index]) for index in unresolved], "unresolved coverage"
        ),
        "mean_set_width": _mean(
            [float(len(sets[2][index])) for index in unresolved], "unresolved width"
        ),
        "panel_count": len(unresolved),
        "typed_gap_rate": _mean(
            [float(0 in sets[2][index]) for index in unresolved], "unresolved gap"
        ),
    }
    required_strata["overall_catalog_typed_gap_rate"] = {
        "panel_count": panel_count,
        "typed_gap_rate": _mean(
            [float(0 in value) for value in sets[2]], "catalog gap"
        ),
    }
    required_strata[
        "catalog_unresolved_rows_empirical_coverage_mean-set-width_and_typed-GAP-rate"
    ] = required_strata["catalog_unresolved_rows"]
    required_strata["overall_catalog_typed-GAP-rate"] = required_strata[
        "overall_catalog_typed_gap_rate"
    ]
    preregistered_metric_values = {
        "arc_top1": metric_values["arc_top1"],
        "empirical_joint_whole-task_set_coverage": metric_values[
            "empirical_joint_whole_task_set_coverage"
        ],
        "known_catalog_binary_balanced_accuracy": metric_values[
            "known_catalog_binary_balanced_accuracy"
        ],
        "known_catalog_typed_decisive_rate": metric_values[
            "known_catalog_typed_decisive_rate"
        ],
        "mean_straight_joint-q_set_size": metric_values[
            "mean_straight_joint_q_set_size"
        ],
        "straight_and_known_catalog_joint_exact": metric_values[
            "straight_and_known_catalog_joint_exact"
        ],
        "straight_joint-q_singleton_rate": metric_values[
            "straight_joint_q_singleton_rate"
        ],
        "straight_top1": metric_values["straight_top1"],
        "true-straight-count-4_joint-q_singleton_rate": metric_values[
            "true_straight_count_4_joint_q_singleton_rate"
        ],
    }
    return {
        "confusions_true_rows_predicted_columns": {
            "arc_10x10": _confusion(truths[1], top1[1], 10),
            "catalog_3x3_unresolved_nonconvex_convex": _confusion(truths[2], top1[2], 3),
            "straight_10x10": _confusion(truths[0], top1[0], 10),
        },
        "metric_values": metric_values,
        "preregistered_metric_values": preregistered_metric_values,
        "required_strata": required_strata,
        "task_coverage_in_manifest_order": task_coverage,
    }


def _evaluation_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    values = metrics["metric_values"]
    checks = {
        "arc_top1_at_least": values["arc_top1"] >= EVALUATION_GATES["arc_top1"],
        "empirical_joint_whole-task_set_coverage_at_least": (
            values["empirical_joint_whole_task_set_coverage"]
            >= EVALUATION_GATES["empirical_joint_whole_task_set_coverage"]
        ),
        "known_catalog_binary_balanced_accuracy_at_least": (
            values["known_catalog_binary_balanced_accuracy"]
            >= EVALUATION_GATES["known_catalog_binary_balanced_accuracy"]
        ),
        "known_catalog_typed_decisive_rate_at_least": (
            values["known_catalog_typed_decisive_rate"]
            >= EVALUATION_GATES["known_catalog_typed_decisive_rate"]
        ),
        "mean_straight_joint-q_set_size_at_most": (
            values["mean_straight_joint_q_set_size"]
            <= EVALUATION_GATES["mean_straight_joint_q_set_size"]
        ),
        "straight_and_known_catalog_joint_exact_at_least": (
            values["straight_and_known_catalog_joint_exact"]
            >= EVALUATION_GATES["straight_and_known_catalog_joint_exact"]
        ),
        "straight_joint-q_singleton_rate_at_least": (
            values["straight_joint_q_singleton_rate"]
            >= EVALUATION_GATES["straight_joint_q_singleton_rate"]
        ),
        "straight_top1_at_least": values["straight_top1"] >= EVALUATION_GATES["straight_top1"],
        "true-straight-count-4_joint-q_singleton_rate_at_least": (
            values["true_straight_count_4_joint_q_singleton_rate"]
            >= EVALUATION_GATES["true_straight_count_4_joint_q_singleton_rate"]
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "thresholds": dict(PREREG_EVALUATION_THRESHOLDS),
    }


def _evaluation_result_body(
    *,
    authority: V3Authority,
    fit: FitAuthority,
    grant: Mapping[str, Any],
    prediction: Mapping[str, Any],
    prediction_raw: bytes,
    labels: Mapping[str, Any],
    labels_raw: bytes,
) -> dict[str, Any]:
    pairs = _paired_rows(prediction, labels, stage="evaluation")
    if prediction.get("joint_q_record_digest") != grant.get("record_digest"):
        raise ActionCountCNNV3RunnerError("evaluation predictions do not bind grant")
    metrics = evaluation_metrics(
        task_ids=authority.task_ids,
        prediction_rows=[pair[0] for pair in pairs],
        label_rows=[pair[1] for pair in pairs],
    )
    gate = _evaluation_gate(metrics)
    typed_outcome = (
        {
            "disposition": "PRESENT",
            "value": "typed_observer_release_gate_passed",
        }
        if gate["passed"]
        else {
            "disposition": "INDETERMINATE",
            "reason": "typed_observer_release_gate_failed",
            "failed_checks": sorted(name for name, passed in gate["checks"].items() if not passed),
        }
    )
    return {
        "calibration_grant_record_digest": grant["record_digest"],
        "checkpoint_state_dict_sha256": fit.checkpoint_state_dict_sha256,
        "config_digest": fit.config_digest,
        "custody": _custody(authority, fit),
        "evaluation_gate": gate,
        "evaluation_label_record_digest": labels["record_digest"],
        "evaluation_label_source_sha256": _address(labels_raw),
        "evaluation_prediction_record_digest": prediction["record_digest"],
        "evaluation_prediction_source_sha256": _address(prediction_raw),
        **metrics,
        "schema": EVALUATION_RESULT_SCHEMA,
        "typed_outcome": typed_outcome,
    }


def _freeze_evaluation_result(
    *,
    authority: V3Authority,
    fit: FitAuthority,
    grant: Mapping[str, Any],
    prediction_path: Path,
    label_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    prediction, prediction_raw = _load_record(prediction_path, label="evaluation predictions")
    labels, labels_raw = _load_record(label_path, label="evaluation labels")
    body = _evaluation_result_body(
        authority=authority,
        fit=fit,
        grant=grant,
        prediction=prediction,
        prediction_raw=prediction_raw,
        labels=labels,
        labels_raw=labels_raw,
    )
    final = _seal(body)
    _write_once(output_path, final)
    reloaded, _ = _load_record(output_path, label="evaluation result")
    if reloaded != final:
        raise ActionCountCNNV3RunnerError("evaluation result fresh reload differs")
    return final


def run_calibration(
    *,
    plan_path: Path,
    calibration_manifest_path: Path,
    fit_authorization_path: Path,
    fit_precommit_path: Path,
    fit_result_path: Path,
    checkpoint_path: Path,
    authorization_output_path: Path,
    pixel_precommit_output_path: Path,
    prediction_output_path: Path,
    label_output_path: Path,
    grant_output_path: Path,
    panel_reader: Callable[[str], bytes],
    inference: InferenceFunction,
    label_source_loader: LabelSourceLoader,
) -> dict[str, Any]:
    """Run the inseparable authorization→precommit→prediction→label→grant stage."""

    authority = _load_v3_authority(
        stage="calibration", plan_path=plan_path, manifest_path=calibration_manifest_path
    )
    fit = _verify_fit_authority(
        authority=authority,
        fit_authorization_path=fit_authorization_path,
        fit_precommit_path=fit_precommit_path,
        fit_result_path=fit_result_path,
        checkpoint_path=checkpoint_path,
    )
    authorization = _authorize_stage(
        authority=authority,
        fit=fit,
        output_path=authorization_output_path,
        calibration_grant=None,
    )
    # The first stage PNG access is below this durable authorization barrier.
    pixel_precommit = _create_pixel_precommit(
        authority=authority,
        fit=fit,
        authorization=authorization,
        panel_reader=panel_reader,
        output_path=pixel_precommit_output_path,
    )
    raws = _reload_precommitted_pixels(
        authority=authority, precommit=pixel_precommit, panel_reader=panel_reader
    )
    inference_rows = inference(
        "calibration", authority.panel_ids, raws, checkpoint_path, fit
    )
    prediction = _create_prediction_record(
        authority=authority,
        fit=fit,
        authorization=authorization,
        pixel_precommit=pixel_precommit,
        inference_rows=inference_rows,
        calibration_grant=None,
        output_path=prediction_output_path,
    )
    labels = _derive_and_freeze_labels(
        authority=authority,
        fit=fit,
        prediction_path=prediction_output_path,
        label_source_loader=label_source_loader,
        metric_strata_loader=None,
        output_path=label_output_path,
    )
    grant = _freeze_calibration_grant(
        authority=authority,
        fit=fit,
        prediction_path=prediction_output_path,
        label_path=label_output_path,
        output_path=grant_output_path,
    )
    return {
        "authorization": authorization,
        "pixel_precommit": pixel_precommit,
        "prediction": prediction,
        "labels": labels,
        "grant": grant,
    }


def run_evaluation(
    *,
    plan_path: Path,
    calibration_manifest_path: Path,
    evaluation_manifest_path: Path,
    fit_authorization_path: Path,
    fit_precommit_path: Path,
    fit_result_path: Path,
    checkpoint_path: Path,
    calibration_authorization_path: Path,
    calibration_pixel_precommit_path: Path,
    calibration_prediction_path: Path,
    calibration_label_path: Path,
    calibration_grant_path: Path,
    authorization_output_path: Path,
    pixel_precommit_output_path: Path,
    prediction_output_path: Path,
    label_output_path: Path,
    result_output_path: Path,
    panel_reader: Callable[[str], bytes],
    inference: InferenceFunction,
    label_source_loader: LabelSourceLoader,
    metric_strata_loader: MetricStrataLoader,
) -> dict[str, Any]:
    """Run evaluation only after exact cold verification of the frozen grant."""

    calibration_authority = _load_v3_authority(
        stage="calibration", plan_path=plan_path, manifest_path=calibration_manifest_path
    )
    calibration_fit = _verify_fit_authority(
        authority=calibration_authority,
        fit_authorization_path=fit_authorization_path,
        fit_precommit_path=fit_precommit_path,
        fit_result_path=fit_result_path,
        checkpoint_path=checkpoint_path,
    )
    _validate_stage_archive(
        authority=calibration_authority,
        fit=calibration_fit,
        authorization_path=calibration_authorization_path,
        pixel_precommit_path=calibration_pixel_precommit_path,
        prediction_path=calibration_prediction_path,
        label_path=calibration_label_path,
        calibration_grant=None,
    )
    grant = _verify_calibration_grant(
        authority=calibration_authority,
        fit=calibration_fit,
        prediction_path=calibration_prediction_path,
        label_path=calibration_label_path,
        grant_path=calibration_grant_path,
    )
    authority = _load_v3_authority(
        stage="evaluation", plan_path=plan_path, manifest_path=evaluation_manifest_path
    )
    fit = _verify_fit_authority(
        authority=authority,
        fit_authorization_path=fit_authorization_path,
        fit_precommit_path=fit_precommit_path,
        fit_result_path=fit_result_path,
        checkpoint_path=checkpoint_path,
    )
    authorization = _authorize_stage(
        authority=authority,
        fit=fit,
        output_path=authorization_output_path,
        calibration_grant=grant,
    )
    # The first evaluation PNG access is below both grant and authorization.
    pixel_precommit = _create_pixel_precommit(
        authority=authority,
        fit=fit,
        authorization=authorization,
        panel_reader=panel_reader,
        output_path=pixel_precommit_output_path,
    )
    raws = _reload_precommitted_pixels(
        authority=authority, precommit=pixel_precommit, panel_reader=panel_reader
    )
    inference_rows = inference(
        "evaluation", authority.panel_ids, raws, checkpoint_path, fit
    )
    prediction = _create_prediction_record(
        authority=authority,
        fit=fit,
        authorization=authorization,
        pixel_precommit=pixel_precommit,
        inference_rows=inference_rows,
        calibration_grant=grant,
        output_path=prediction_output_path,
    )
    labels = _derive_and_freeze_labels(
        authority=authority,
        fit=fit,
        prediction_path=prediction_output_path,
        label_source_loader=label_source_loader,
        metric_strata_loader=metric_strata_loader,
        output_path=label_output_path,
    )
    result = _freeze_evaluation_result(
        authority=authority,
        fit=fit,
        grant=grant,
        prediction_path=prediction_output_path,
        label_path=label_output_path,
        output_path=result_output_path,
    )
    return {
        "authorization": authorization,
        "pixel_precommit": pixel_precommit,
        "prediction": prediction,
        "labels": labels,
        "result": result,
    }


def _validate_stage_archive(
    *,
    authority: V3Authority,
    fit: FitAuthority,
    authorization_path: Path,
    pixel_precommit_path: Path,
    prediction_path: Path,
    label_path: Path,
    calibration_grant: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    authorization, _ = _load_record(
        authorization_path, label=f"archived {authority.stage} authorization"
    )
    precommit, _ = _load_record(
        pixel_precommit_path, label=f"archived {authority.stage} pixel precommit"
    )
    prediction, prediction_raw = _load_record(
        prediction_path, label=f"archived {authority.stage} predictions"
    )
    labels, _ = _load_record(label_path, label=f"archived {authority.stage} labels")
    expected_custody = _custody(authority, fit)
    expected_grant_digest = (
        None if calibration_grant is None else calibration_grant["record_digest"]
    )
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("stage") != authority.stage
        or authorization.get("custody") != expected_custody
        or authorization.get("calibration_grant_record_digest") != expected_grant_digest
        or authorization.get("allowed_reads", {}).get("panel_ids_digest")
        != "sha256:" + canonical_digest(list(authority.panel_ids))
    ):
        raise ActionCountCNNV3RunnerError("archived stage authorization differs")
    observations = precommit.get("exact_png_observations")
    if (
        precommit.get("schema") != PIXEL_PRECOMMIT_SCHEMA
        or precommit.get("stage") != authority.stage
        or precommit.get("authorization_record_digest") != authorization["record_digest"]
        or precommit.get("custody") != expected_custody
        or not isinstance(observations, list)
        or [row.get("panel_id") for row in observations if isinstance(row, dict)]
        != list(authority.panel_ids)
    ):
        raise ActionCountCNNV3RunnerError("archived stage pixel precommit differs")
    for row in observations:
        _require_address(row.get("png_sha256"), "archived PNG")
        if not isinstance(row.get("png_size_bytes"), int) or row["png_size_bytes"] <= 0:
            raise ActionCountCNNV3RunnerError("archived PNG size differs")
    if (
        prediction.get("schema") != PREDICTION_SCHEMA
        or prediction.get("stage") != authority.stage
        or prediction.get("custody") != expected_custody
        or prediction.get("authorization_record_digest") != authorization["record_digest"]
        or prediction.get("pixel_precommit_record_digest") != precommit["record_digest"]
        or prediction.get("panel_ids") != list(authority.panel_ids)
        or prediction.get("checkpoint_state_dict_sha256")
        != fit.checkpoint_state_dict_sha256
        or prediction.get("config_digest") != fit.config_digest
    ):
        raise ActionCountCNNV3RunnerError("archived stage prediction custody differs")
    rows = prediction.get("rows")
    if not isinstance(rows, list) or len(rows) != len(authority.panel_ids):
        raise ActionCountCNNV3RunnerError("archived prediction row count differs")
    q = None if calibration_grant is None else float(calibration_grant["deployment_joint_q"])
    for panel_id, row in zip(authority.panel_ids, rows):
        if not isinstance(row, dict):
            raise ActionCountCNNV3RunnerError("archived prediction row is invalid")
        inference = InferenceRow(
            panel_id=str(row.get("panel_id")),
            straight_logits=tuple(row.get("straight_logits", ())),
            straight_probabilities=tuple(row.get("straight_probabilities", ())),
            arc_logits=tuple(row.get("arc_logits", ())),
            arc_probabilities=tuple(row.get("arc_probabilities", ())),
            catalog_logits=tuple(row.get("catalog_logits", ())),
            catalog_probabilities=tuple(row.get("catalog_probabilities", ())),
        )
        rebuilt = _prediction_row_data(inference, panel_id)
        for key, value in rebuilt.items():
            if row.get(key) != value:
                raise ActionCountCNNV3RunnerError("archived logits/probabilities differ")
        for name, size in HEADS:
            class_set = row.get(f"{name}_class_set")
            if q is None:
                if class_set is not None:
                    raise ActionCountCNNV3RunnerError("calibration archive contains class set")
            else:
                expected = [
                    index
                    for index in range(size)
                    if 1.0 - row[f"{name}_probabilities"][index] <= q
                ]
                if class_set != expected:
                    raise ActionCountCNNV3RunnerError("archived class set differs from joint q")
    if (
        labels.get("schema") != LABEL_RECORD_SCHEMA
        or labels.get("stage") != authority.stage
        or labels.get("custody") != expected_custody
        or labels.get("prediction_record_digest") != prediction["record_digest"]
        or labels.get("prediction_source_sha256") != _address(prediction_raw)
    ):
        raise ActionCountCNNV3RunnerError("archived delayed-label custody differs")
    _paired_rows(prediction, labels, stage=authority.stage)
    return authorization, precommit, prediction, labels


def cold_replay(
    *,
    plan_path: Path,
    calibration_manifest_path: Path,
    evaluation_manifest_path: Path,
    fit_authorization_path: Path,
    fit_precommit_path: Path,
    fit_result_path: Path,
    checkpoint_path: Path,
    calibration_authorization_path: Path,
    calibration_pixel_precommit_path: Path,
    calibration_prediction_path: Path,
    calibration_label_path: Path,
    calibration_grant_path: Path,
    evaluation_authorization_path: Path,
    evaluation_pixel_precommit_path: Path,
    evaluation_prediction_path: Path,
    evaluation_label_path: Path,
    evaluation_result_path: Path,
    replay_output_path: Path,
) -> dict[str, Any]:
    """Cold replay archived math/custody with zero PNG, inference, or training calls."""

    calibration_authority = _load_v3_authority(
        stage="calibration", plan_path=plan_path, manifest_path=calibration_manifest_path
    )
    calibration_fit = _verify_fit_authority(
        authority=calibration_authority,
        fit_authorization_path=fit_authorization_path,
        fit_precommit_path=fit_precommit_path,
        fit_result_path=fit_result_path,
        checkpoint_path=checkpoint_path,
    )
    _validate_stage_archive(
        authority=calibration_authority,
        fit=calibration_fit,
        authorization_path=calibration_authorization_path,
        pixel_precommit_path=calibration_pixel_precommit_path,
        prediction_path=calibration_prediction_path,
        label_path=calibration_label_path,
        calibration_grant=None,
    )
    grant = _verify_calibration_grant(
        authority=calibration_authority,
        fit=calibration_fit,
        prediction_path=calibration_prediction_path,
        label_path=calibration_label_path,
        grant_path=calibration_grant_path,
    )
    evaluation_authority = _load_v3_authority(
        stage="evaluation", plan_path=plan_path, manifest_path=evaluation_manifest_path
    )
    evaluation_fit = _verify_fit_authority(
        authority=evaluation_authority,
        fit_authorization_path=fit_authorization_path,
        fit_precommit_path=fit_precommit_path,
        fit_result_path=fit_result_path,
        checkpoint_path=checkpoint_path,
    )
    _, _, prediction, labels = _validate_stage_archive(
        authority=evaluation_authority,
        fit=evaluation_fit,
        authorization_path=evaluation_authorization_path,
        pixel_precommit_path=evaluation_pixel_precommit_path,
        prediction_path=evaluation_prediction_path,
        label_path=evaluation_label_path,
        calibration_grant=grant,
    )
    result, _ = _load_record(evaluation_result_path, label="archived evaluation result")
    prediction_raw = evaluation_prediction_path.read_bytes()
    labels_raw = evaluation_label_path.read_bytes()
    expected_result = _seal(
        _evaluation_result_body(
            authority=evaluation_authority,
            fit=evaluation_fit,
            grant=grant,
            prediction=prediction,
            prediction_raw=prediction_raw,
            labels=labels,
            labels_raw=labels_raw,
        )
    )
    if result != expected_result:
        raise ActionCountCNNV3RunnerError("evaluation result differs from cold replay")
    body = {
        "calibration_grant_record_digest": grant["record_digest"],
        "evaluation_result_record_digest": result["record_digest"],
        "inference_calls": 0,
        "label_source_calls": 0,
        "model_training_calls": 0,
        "png_reads": 0,
        "recomputed": [
            "artifact_custody",
            "logit_probability_consistency",
            "joint_q_sorted_scores_index_95",
            "evaluation_class_sets",
            "metrics_confusions_strata_gates_and_typed_outcome",
        ],
        "runner_source_sha256": runner_source_digest(),
        "schema": REPLAY_SCHEMA,
    }
    final = _seal(body)
    _write_once(replay_output_path, final)
    reloaded, _ = _load_record(replay_output_path, label="cold replay result")
    if reloaded != final:
        raise ActionCountCNNV3RunnerError("cold replay fresh reload differs")
    return final


def _actions_from_programs(programs: Mapping[str, Any], panel_id: str) -> Sequence[str]:
    task_id = _task_id(panel_id)
    parts = PurePosixPath(panel_id).parts
    folder = int(parts[2])
    panel_index = int(parts[3][:-4])
    task = programs.get(task_id)
    if not isinstance(task, list) or len(task) != 2:
        raise ActionCountCNNV3RunnerError("delayed action-program task differs")
    side = task[1 - folder]
    if not isinstance(side, list) or len(side) != 7:
        raise ActionCountCNNV3RunnerError("delayed action-program side differs")
    panel = side[panel_index]
    if (
        not isinstance(panel, list)
        or len(panel) != 1
        or not isinstance(panel[0], list)
        or any(not isinstance(value, str) for value in panel[0])
    ):
        raise ActionCountCNNV3RunnerError("delayed action-program panel differs")
    return panel[0]


def _line_profile(actions: Sequence[str]) -> str:
    styles = [action.split("_", 2)[1] for action in actions if action.startswith("line_")]
    if not styles:
        return "no_straight_actions"
    normal = sum(style == "normal" for style in styles)
    if normal == len(styles):
        return "normal_only"
    if normal == 0:
        return "decorated_only"
    return "mixed_normal_and_decorated"


def filesystem_delayed_loaders(
    *,
    catalog_audit_path: Path,
    shape_rows_path: Path,
    attribute_rows_path: Path,
    hd_programs_path: Path,
    bd_programs_path: Path,
) -> tuple[LabelSourceLoader, MetricStrataLoader]:
    """Create closures that perform no file access until a prediction barrier."""

    cache: dict[str, Any] = {}

    def load_sources(barrier: PredictionBarrier) -> LabelSources:
        import csv

        from bongard.panel_convexity_catalog_audit import (
            build_catalog_binding,
            catalog_label_for_actions,
            convexity_catalog_algorithm_digest,
            convexity_catalog_source_digest,
        )

        audit, _ = _load_record(catalog_audit_path, label="catalog audit")
        if audit.get("record_digest") != (
            "sha256:333a6cf4cbdf135484cbacf42ddb20aaf9fad482f70430583f36e60f3198f971"
        ):
            raise ActionCountCNNV3RunnerError("catalog audit is not frozen authority")
        expected_bindings = audit.get("source_bindings")
        if not isinstance(expected_bindings, dict):
            raise ActionCountCNNV3RunnerError("catalog source bindings are missing")
        paths = {
            "attribute_rows": attribute_rows_path,
            "bd_action_programs": bd_programs_path,
            "hd_action_programs": hd_programs_path,
            "shape_rows": shape_rows_path,
        }
        raws: dict[str, bytes] = {}
        for name, path in paths.items():
            raw = path.read_bytes()
            if expected_bindings.get(name, {}).get("sha256") != _address(raw):
                raise ActionCountCNNV3RunnerError(f"delayed {name} source differs")
            raws[name] = raw
        try:
            shape_rows = list(
                csv.DictReader(
                    raws["shape_rows"].decode("utf-8").splitlines(), delimiter="\t"
                )
            )
            attribute_rows = list(
                csv.DictReader(
                    raws["attribute_rows"].decode("utf-8").splitlines(), delimiter="\t"
                )
            )
            hd_programs = json.loads(raws["hd_action_programs"])
            bd_programs = json.loads(raws["bd_action_programs"])
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ActionCountCNNV3RunnerError(f"delayed label source is invalid: {exc}") from exc
        if not isinstance(hd_programs, dict) or not isinstance(bd_programs, dict):
            raise ActionCountCNNV3RunnerError("delayed program sources are not objects")
        binding = build_catalog_binding(
            shape_rows=shape_rows,
            attribute_rows=attribute_rows,
            hd_programs=hd_programs,
            bd_programs=bd_programs,
        )
        plan_bindings = LabelAuthorityBindings(
            hd_action_program_raw_sha256=_address(raws["hd_action_programs"]),
            catalog_algorithm_digest=convexity_catalog_algorithm_digest(),
            catalog_audit_record_digest=audit["record_digest"],
            catalog_authority_source_sha256=convexity_catalog_source_digest(),
        )
        cache.clear()
        cache.update(
            {
                "barrier": barrier,
                "hd_programs": hd_programs,
            }
        )

        def lookup(actions: Sequence[str]) -> CatalogTarget:
            found = catalog_label_for_actions(actions, binding)
            return CatalogTarget(
                raw_target=int(found.raw_label),
                supervised_class=found.supervised_class,
                match_kind=found.match_kind,
            )

        return LabelSources(
            hd_action_program_raw=raws["hd_action_programs"],
            catalog_lookup=lookup,
            authority_bindings=plan_bindings,
        )

    def load_strata(barrier: PredictionBarrier) -> Mapping[str, str]:
        if cache.get("barrier") != barrier or not isinstance(cache.get("hd_programs"), dict):
            raise ActionCountCNNV3RunnerError(
                "metric strata requested before matching delayed labels"
            )
        programs = cache["hd_programs"]
        return {
            panel_id: _line_profile(_actions_from_programs(programs, panel_id))
            for panel_id in barrier.panel_ids
        }

    return load_sources, load_strata


def _stage_paths(directory: Path, stage: str) -> dict[str, Path]:
    if stage not in {"calibration", "evaluation"}:
        raise ActionCountCNNV3RunnerError("stage path kind differs")
    return {
        "authorization": directory / f"{stage}_authorization.json",
        "pixel_precommit": directory / f"{stage}_pixel_precommit.json",
        "prediction": directory / f"{stage}_predictions.json",
        "labels": directory / f"{stage}_delayed_labels.json",
        "terminal": directory
        / ("calibration_grant.json" if stage == "calibration" else "evaluation_result.json"),
    }


def _add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fit-authorization", type=Path, required=True)
    parser.add_argument("--fit-precommit", type=Path, required=True)
    parser.add_argument("--fit-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)


def _add_delayed_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog-audit", type=Path, required=True)
    parser.add_argument("--shape-rows", type=Path, required=True)
    parser.add_argument("--attribute-rows", type=Path, required=True)
    parser.add_argument("--hd-programs", type=Path, required=True)
    parser.add_argument("--bd-programs", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Closed V3 calibration/evaluation chronology; no V2 CAL/eval inputs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser("calibrate")
    _add_fit_arguments(calibrate)
    _add_delayed_source_arguments(calibrate)
    calibrate.add_argument("--calibration-manifest", type=Path, required=True)
    calibrate.add_argument("--dataset-root", type=Path, required=True)
    calibrate.add_argument("--output-dir", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    _add_fit_arguments(evaluate)
    _add_delayed_source_arguments(evaluate)
    evaluate.add_argument("--calibration-manifest", type=Path, required=True)
    evaluate.add_argument("--evaluation-manifest", type=Path, required=True)
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--calibration-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)

    replay = subparsers.add_parser("replay")
    _add_fit_arguments(replay)
    replay.add_argument("--calibration-manifest", type=Path, required=True)
    replay.add_argument("--evaluation-manifest", type=Path, required=True)
    replay.add_argument("--calibration-dir", type=Path, required=True)
    replay.add_argument("--evaluation-dir", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fit = {
        "plan_path": args.plan.resolve(),
        "fit_authorization_path": args.fit_authorization.resolve(),
        "fit_precommit_path": args.fit_precommit.resolve(),
        "fit_result_path": args.fit_result.resolve(),
        "checkpoint_path": args.checkpoint.resolve(),
    }
    if args.command == "calibrate":
        paths = _stage_paths(args.output_dir.resolve(), "calibration")
        label_loader, _ = filesystem_delayed_loaders(
            catalog_audit_path=args.catalog_audit.resolve(),
            shape_rows_path=args.shape_rows.resolve(),
            attribute_rows_path=args.attribute_rows.resolve(),
            hd_programs_path=args.hd_programs.resolve(),
            bd_programs_path=args.bd_programs.resolve(),
        )
        run_calibration(
            **fit,
            calibration_manifest_path=args.calibration_manifest.resolve(),
            authorization_output_path=paths["authorization"],
            pixel_precommit_output_path=paths["pixel_precommit"],
            prediction_output_path=paths["prediction"],
            label_output_path=paths["labels"],
            grant_output_path=paths["terminal"],
            panel_reader=filesystem_panel_reader(args.dataset_root.resolve()),
            inference=default_inference,
            label_source_loader=label_loader,
        )
    elif args.command == "evaluate":
        calibration_paths = _stage_paths(args.calibration_dir.resolve(), "calibration")
        paths = _stage_paths(args.output_dir.resolve(), "evaluation")
        label_loader, strata_loader = filesystem_delayed_loaders(
            catalog_audit_path=args.catalog_audit.resolve(),
            shape_rows_path=args.shape_rows.resolve(),
            attribute_rows_path=args.attribute_rows.resolve(),
            hd_programs_path=args.hd_programs.resolve(),
            bd_programs_path=args.bd_programs.resolve(),
        )
        run_evaluation(
            **fit,
            calibration_manifest_path=args.calibration_manifest.resolve(),
            evaluation_manifest_path=args.evaluation_manifest.resolve(),
            calibration_prediction_path=calibration_paths["prediction"],
            calibration_authorization_path=calibration_paths["authorization"],
            calibration_pixel_precommit_path=calibration_paths["pixel_precommit"],
            calibration_label_path=calibration_paths["labels"],
            calibration_grant_path=calibration_paths["terminal"],
            authorization_output_path=paths["authorization"],
            pixel_precommit_output_path=paths["pixel_precommit"],
            prediction_output_path=paths["prediction"],
            label_output_path=paths["labels"],
            result_output_path=paths["terminal"],
            panel_reader=filesystem_panel_reader(args.dataset_root.resolve()),
            inference=default_inference,
            label_source_loader=label_loader,
            metric_strata_loader=strata_loader,
        )
    else:
        calibration_paths = _stage_paths(args.calibration_dir.resolve(), "calibration")
        evaluation_paths = _stage_paths(args.evaluation_dir.resolve(), "evaluation")
        cold_replay(
            **fit,
            calibration_manifest_path=args.calibration_manifest.resolve(),
            evaluation_manifest_path=args.evaluation_manifest.resolve(),
            calibration_authorization_path=calibration_paths["authorization"],
            calibration_pixel_precommit_path=calibration_paths["pixel_precommit"],
            calibration_prediction_path=calibration_paths["prediction"],
            calibration_label_path=calibration_paths["labels"],
            calibration_grant_path=calibration_paths["terminal"],
            evaluation_authorization_path=evaluation_paths["authorization"],
            evaluation_pixel_precommit_path=evaluation_paths["pixel_precommit"],
            evaluation_prediction_path=evaluation_paths["prediction"],
            evaluation_label_path=evaluation_paths["labels"],
            evaluation_result_path=evaluation_paths["terminal"],
            replay_output_path=args.output.resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ActionCountCNNV3RunnerError",
    "InferenceRow",
    "calibration_scores",
    "cold_replay",
    "default_inference",
    "evaluation_metrics",
    "filesystem_delayed_loaders",
    "filesystem_panel_reader",
    "main",
    "run_calibration",
    "run_evaluation",
    "runner_source_digest",
)
