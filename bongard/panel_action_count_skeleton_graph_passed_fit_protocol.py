"""Exact promotion boundary for the skeleton-graph development observer.

This module is deliberately narrower than a calibration or benchmark grant.
It authenticates the complete development chain (precommit, model, feature
bank, predictions, result, and cold replay) and answers whether *both* frozen
development heads earned eligibility for a later calibration precommit.

A structurally valid run that misses either named development gate becomes a
typed GAP.  Neither outcome authorizes calibration pixels, target pixels,
support/query inference, or benchmark release by itself.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, TypeAlias

from bongard.canonical import canonical_digest, canonical_json
from bongard import panel_action_count_skeleton_graph_dev_command as development


PROTOCOL_SCHEMA = "gkm.bongard-skeleton-graph-passed-fit-protocol.v1"
GAP_SCHEMA = "gkm.bongard-skeleton-graph-passed-fit-gap.v1"
REQUIRED_HEADS = ("direct_pair", "catalog_three_class")
PINNED_DEVELOPMENT_COMMIT = "dffd14a232bd213653c1d3b5eaffb08bb716cdd9"
PINNED_DEVELOPMENT_SOURCE_SHA256 = (
    "7399d4e0a3b05f14adff890b11a4674ece8904cc3e60cb9ea0b857fcd107a523"
)
PINNED_DEVELOPMENT_CONFIG_DIGEST = (
    "sha256:7dff25c405ddd05419a6c20b7c53559b9f6524735c09d271ca3dbd74c477b665"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_RECORD_BYTES = 64 * 1024 * 1024

PASSED_FIT_ALGORITHM_DIGEST = "sha256:" + canonical_digest(
    {
        "catalog_class_order": list(development.CATALOG_CLASS_ORDER),
        "dependency_source_addresses": development.dependency_source_addresses(),
        "development_commit": PINNED_DEVELOPMENT_COMMIT,
        "development_config_digest": PINNED_DEVELOPMENT_CONFIG_DIGEST,
        "development_source_sha256": PINNED_DEVELOPMENT_SOURCE_SHA256,
        "development_schemas": {
            "features": development.SCHEMA_FEATURES,
            "precommit": development.SCHEMA_PRECOMMIT,
            "predictions": development.SCHEMA_PREDICTIONS,
            "replay": development.SCHEMA_REPLAY,
            "result": development.SCHEMA_RESULT,
        },
        "exact_replay_requirements": {
            "feature_replay_exact": True,
            "metrics_replay_exact": True,
            "model_refit_calls": 0,
            "prediction_replay_exact": True,
        },
        "observed_pair_class_order": list(
            development.OBSERVED_TRAIN_PAIR_CLASS_ORDER
        ),
        "estimator_parameters": {
            name: development._expected_estimator_params(seed)
            for name, seed in development.FIXED_CLASSIFIER_SEEDS.items()
        },
        "feature_names_digest": "sha256:"
        + canonical_digest(list(development.FEATURE_NAMES)),
        "feature_width": len(development.FEATURE_NAMES),
        "frozen_gate_thresholds": development._plain(
            development.ENGINEERING_THRESHOLDS
        ),
        "policy": "both_named_heads_and_full_six_file_chain",
        "protocol": development._plain(development.PROTOCOL),
        "required_heads": list(REQUIRED_HEADS),
        "runtime": development.runtime_fingerprint(),
        "valid_pair_class_order": list(development.VALID_PAIR_CLASS_ORDER),
        "version": 1,
    }
)


class SkeletonGraphPassedFitProtocolError(RuntimeError):
    """The development chain or its frozen promotion policy differs."""


def source_sha256() -> str:
    """Return the immutable loaded source digest."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise SkeletonGraphPassedFitProtocolError(
            f"{label} must be a sha256: address"
        )
    return value


def _stable_bytes(path: Path, *, label: str, maximum: int) -> bytes:
    supplied = Path(path)
    absolute = supplied.absolute()
    try:
        resolved = supplied.resolve(strict=True)
        before = supplied.lstat()
    except OSError as exc:
        raise SkeletonGraphPassedFitProtocolError(
            f"cannot stat {label}: {exc}"
        ) from exc
    if (
        resolved != absolute
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > maximum
    ):
        raise SkeletonGraphPassedFitProtocolError(
            f"{label} must be a bounded regular nonsymlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(supplied, flags)
    try:
        opened = os.fstat(descriptor)
        remaining = maximum + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after_read = os.fstat(descriptor)
    except OSError as exc:
        raise SkeletonGraphPassedFitProtocolError(
            f"cannot read {label}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)
    try:
        after = supplied.lstat()
    except OSError as exc:
        raise SkeletonGraphPassedFitProtocolError(
            f"cannot restat {label}: {exc}"
        ) from exc
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if not fingerprint(before) == fingerprint(opened) == fingerprint(after_read) == fingerprint(after):
        raise SkeletonGraphPassedFitProtocolError(f"{label} changed while reading")
    raw = b"".join(chunks)
    if len(raw) != before.st_size or len(raw) > maximum:
        raise SkeletonGraphPassedFitProtocolError(f"{label} read size differs")
    return raw


def _record(
    path: Path, *, schema: str, label: str, maximum: int = _MAX_RECORD_BYTES
) -> tuple[dict[str, Any], str]:
    raw = _stable_bytes(path, label=label, maximum=maximum)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SkeletonGraphPassedFitProtocolError(
            f"cannot decode {label}: {exc}"
        ) from exc
    if type(value) is not dict or raw != canonical_json(value) + b"\n":
        raise SkeletonGraphPassedFitProtocolError(
            f"{label} is not canonical JSON plus newline"
        )
    body = dict(value)
    found = body.pop("record_digest", None)
    if value.get("schema") != schema or found != "sha256:" + canonical_digest(body):
        raise SkeletonGraphPassedFitProtocolError(
            f"{label} schema or record digest differs"
        )
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()


def _strict_fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise SkeletonGraphPassedFitProtocolError(f"{label} fields differ")
    return value


def _same_path(found: object, supplied: Path, label: str) -> None:
    if type(found) is not str:
        raise SkeletonGraphPassedFitProtocolError(f"{label} intended path differs")
    try:
        found_path = Path(found).resolve(strict=True)
        supplied_path = Path(supplied).resolve(strict=True)
    except OSError as exc:
        raise SkeletonGraphPassedFitProtocolError(
            f"cannot resolve {label} path: {exc}"
        ) from exc
    if found_path != supplied_path:
        raise SkeletonGraphPassedFitProtocolError(f"{label} intended path differs")


def _common_content(value: object) -> dict[str, object]:
    return {
        "passed_fit_algorithm_digest": value.passed_fit_algorithm_digest,
        "passed_fit_authority_source_sha256": value.passed_fit_authority_source_sha256,
        "development_source_sha256": value.development_source_sha256,
        "development_config_digest": value.development_config_digest,
        "claim_scope": value.claim_scope,
        "required_heads": list(value.required_heads),
        "development_precommit_record_digest": value.development_precommit_record_digest,
        "development_precommit_file_sha256": value.development_precommit_file_sha256,
        "development_result_record_digest": value.development_result_record_digest,
        "development_result_file_sha256": value.development_result_file_sha256,
        "development_replay_record_digest": value.development_replay_record_digest,
        "development_replay_file_sha256": value.development_replay_file_sha256,
        "model_file_sha256": value.model_file_sha256,
        "model_size_bytes": value.model_size_bytes,
        "model_structure_digest": value.model_structure_digest,
        "feature_artifact_record_digest": value.feature_artifact_record_digest,
        "feature_artifact_file_sha256": value.feature_artifact_file_sha256,
        "prediction_artifact_record_digest": value.prediction_artifact_record_digest,
        "prediction_artifact_file_sha256": value.prediction_artifact_file_sha256,
        "feature_array_digest": value.feature_array_digest,
        "label_array_digest": value.label_array_digest,
        "direct_pair_probability_digest": value.direct_pair_probability_digest,
        "catalog_probability_digest": value.catalog_probability_digest,
        "observed_pair_class_order": list(value.observed_pair_class_order),
        "valid_pair_class_order": list(value.valid_pair_class_order),
        "catalog_class_order": list(value.catalog_class_order),
        "development_gate_digest": value.development_gate_digest,
        "validation_metrics_digest": value.validation_metrics_digest,
        "replay_feature_exact": value.replay_feature_exact,
        "replay_prediction_exact": value.replay_prediction_exact,
        "replay_metrics_exact": value.replay_metrics_exact,
        "replay_model_refit_calls": value.replay_model_refit_calls,
        "replay_model_inference_panel_count": value.replay_model_inference_panel_count,
        "replay_pixel_reextract_group_count": value.replay_pixel_reextract_group_count,
        "prior_failed_capacity_attempt_digest": value.prior_failed_capacity_attempt_digest,
        "calibration_pixel_authorized": False,
        "support_query_inference_authorized": False,
        "target_pixel_authorized": False,
        "benchmark_sealable": False,
    }


_COMMON_FIELDS = {
    "passed_fit_algorithm_digest",
    "passed_fit_authority_source_sha256",
    "development_source_sha256",
    "development_config_digest",
    "claim_scope",
    "required_heads",
    "development_precommit_record_digest",
    "development_precommit_file_sha256",
    "development_result_record_digest",
    "development_result_file_sha256",
    "development_replay_record_digest",
    "development_replay_file_sha256",
    "model_file_sha256",
    "model_size_bytes",
    "model_structure_digest",
    "feature_artifact_record_digest",
    "feature_artifact_file_sha256",
    "prediction_artifact_record_digest",
    "prediction_artifact_file_sha256",
    "feature_array_digest",
    "label_array_digest",
    "direct_pair_probability_digest",
    "catalog_probability_digest",
    "observed_pair_class_order",
    "valid_pair_class_order",
    "catalog_class_order",
    "development_gate_digest",
    "validation_metrics_digest",
    "replay_feature_exact",
    "replay_prediction_exact",
    "replay_metrics_exact",
    "replay_model_refit_calls",
    "replay_model_inference_panel_count",
    "replay_pixel_reextract_group_count",
    "prior_failed_capacity_attempt_digest",
    "calibration_pixel_authorized",
    "support_query_inference_authorized",
    "target_pixel_authorized",
    "benchmark_sealable",
}


def _validate_identity(value: object) -> None:
    for name in (
        "passed_fit_algorithm_digest",
        "development_config_digest",
        "development_precommit_record_digest",
        "development_precommit_file_sha256",
        "development_result_record_digest",
        "development_result_file_sha256",
        "development_replay_record_digest",
        "development_replay_file_sha256",
        "model_file_sha256",
        "model_structure_digest",
        "feature_artifact_record_digest",
        "feature_artifact_file_sha256",
        "prediction_artifact_record_digest",
        "prediction_artifact_file_sha256",
        "feature_array_digest",
        "label_array_digest",
        "direct_pair_probability_digest",
        "catalog_probability_digest",
        "development_gate_digest",
        "validation_metrics_digest",
        "prior_failed_capacity_attempt_digest",
    ):
        _address(getattr(value, name), name)
    if (
        type(value.passed_fit_authority_source_sha256) is not str
        or value.passed_fit_authority_source_sha256
        != "sha256:" + source_sha256()
        or type(value.development_source_sha256) is not str
        or value.development_source_sha256 != PINNED_DEVELOPMENT_SOURCE_SHA256
        or development.source_sha256() != PINNED_DEVELOPMENT_SOURCE_SHA256
        or value.development_config_digest != PINNED_DEVELOPMENT_CONFIG_DIGEST
        or development.config_digest() != PINNED_DEVELOPMENT_CONFIG_DIGEST
        or value.passed_fit_algorithm_digest != PASSED_FIT_ALGORITHM_DIGEST
        or value.claim_scope != development.CLAIM_SCOPE
        or value.required_heads != REQUIRED_HEADS
        or value.observed_pair_class_order
        != tuple(development.OBSERVED_TRAIN_PAIR_CLASS_ORDER)
        or value.valid_pair_class_order != tuple(development.VALID_PAIR_CLASS_ORDER)
        or value.catalog_class_order != tuple(development.CATALOG_CLASS_ORDER)
        or type(value.model_size_bytes) is not int
        or not 0 < value.model_size_bytes <= development.MODEL_MAX_BYTES
        or type(value.replay_model_refit_calls) is not int
        or type(value.replay_model_inference_panel_count) is not int
        or type(value.replay_pixel_reextract_group_count) is not int
        or value.replay_model_refit_calls != 0
        or value.replay_model_inference_panel_count <= 0
        or value.replay_pixel_reextract_group_count <= 0
        or value.replay_feature_exact is not True
        or value.replay_prediction_exact is not True
        or value.replay_metrics_exact is not True
    ):
        raise SkeletonGraphPassedFitProtocolError(
            "passed-fit identity or replay contract differs"
        )


@dataclass(frozen=True, slots=True, init=False)
class _OutcomeBase:
    passed_fit_algorithm_digest: str
    passed_fit_authority_source_sha256: str
    development_source_sha256: str
    development_config_digest: str
    claim_scope: str
    required_heads: tuple[str, ...]
    development_precommit_record_digest: str
    development_precommit_file_sha256: str
    development_result_record_digest: str
    development_result_file_sha256: str
    development_replay_record_digest: str
    development_replay_file_sha256: str
    model_file_sha256: str
    model_size_bytes: int
    model_structure_digest: str
    feature_artifact_record_digest: str
    feature_artifact_file_sha256: str
    prediction_artifact_record_digest: str
    prediction_artifact_file_sha256: str
    feature_array_digest: str
    label_array_digest: str
    direct_pair_probability_digest: str
    catalog_probability_digest: str
    observed_pair_class_order: tuple[int, ...]
    valid_pair_class_order: tuple[int, ...]
    catalog_class_order: tuple[int, ...]
    development_gate_digest: str
    validation_metrics_digest: str
    replay_feature_exact: bool
    replay_prediction_exact: bool
    replay_metrics_exact: bool
    replay_model_refit_calls: int
    replay_model_inference_panel_count: int
    replay_pixel_reextract_group_count: int
    prior_failed_capacity_attempt_digest: str
    record_digest: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SkeletonGraphPassedFitProtocolError("passed-fit outcomes are factory-only")


@dataclass(frozen=True, slots=True, init=False)
class SkeletonGraphPassedFitGap(_OutcomeBase):
    failed_checks: tuple[str, ...]

    def content_data(self) -> dict[str, object]:
        return {
            "schema": GAP_SCHEMA,
            **_common_content(self),
            "disposition": "gap",
            "development_fit_passed": False,
            "failed_checks": list(self.failed_checks),
            "gap_reason_code": "both_named_development_heads_not_promoted",
            "eligible_for_calibration_execution_precommit": False,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "SkeletonGraphPassedFitGap":
        raw = _strict_fields(
            value,
            _COMMON_FIELDS
            | {
                "schema",
                "disposition",
                "development_fit_passed",
                "failed_checks",
                "gap_reason_code",
                "eligible_for_calibration_execution_precommit",
                "record_digest",
            },
            "passed-fit GAP",
        )
        if (
            raw["schema"] != GAP_SCHEMA
            or raw["disposition"] != "gap"
            or raw["development_fit_passed"] is not False
            or raw["gap_reason_code"]
            != "both_named_development_heads_not_promoted"
            or raw["eligible_for_calibration_execution_precommit"] is not False
            or type(raw["failed_checks"]) is not list
        ):
            raise SkeletonGraphPassedFitProtocolError("passed-fit GAP policy differs")
        return _outcome_from_data(cls, raw)


@dataclass(frozen=True, slots=True, init=False)
class SkeletonGraphPassedFitProtocol(_OutcomeBase):
    @property
    def protocol_address(self) -> str:
        return self.record_digest

    def content_data(self) -> dict[str, object]:
        return {
            "schema": PROTOCOL_SCHEMA,
            **_common_content(self),
            "development_fit_passed": True,
            "both_named_development_heads_passed": True,
            "promoted_heads_exact": True,
            "full_six_file_chain_verified": True,
            "eligible_for_calibration_execution_precommit": True,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "SkeletonGraphPassedFitProtocol":
        raw = _strict_fields(
            value,
            _COMMON_FIELDS
            | {
                "schema",
                "development_fit_passed",
                "both_named_development_heads_passed",
                "promoted_heads_exact",
                "full_six_file_chain_verified",
                "eligible_for_calibration_execution_precommit",
                "record_digest",
            },
            "passed-fit protocol",
        )
        if (
            raw["schema"] != PROTOCOL_SCHEMA
            or raw["development_fit_passed"] is not True
            or raw["both_named_development_heads_passed"] is not True
            or raw["promoted_heads_exact"] is not True
            or raw["full_six_file_chain_verified"] is not True
            or raw["eligible_for_calibration_execution_precommit"] is not True
        ):
            raise SkeletonGraphPassedFitProtocolError(
                "passed-fit protocol policy differs"
            )
        return _outcome_from_data(cls, raw)


SkeletonGraphPassedFitOutcome: TypeAlias = (
    SkeletonGraphPassedFitProtocol | SkeletonGraphPassedFitGap
)


def _outcome_from_data(cls: type[Any], raw: Mapping[str, Any]) -> Any:
    common: dict[str, object] = {
        name: raw[name]
        for name in (
            "passed_fit_algorithm_digest",
            "passed_fit_authority_source_sha256",
            "development_source_sha256",
            "development_config_digest",
            "claim_scope",
            "development_precommit_record_digest",
            "development_precommit_file_sha256",
            "development_result_record_digest",
            "development_result_file_sha256",
            "development_replay_record_digest",
            "development_replay_file_sha256",
            "model_file_sha256",
            "model_size_bytes",
            "model_structure_digest",
            "feature_artifact_record_digest",
            "feature_artifact_file_sha256",
            "prediction_artifact_record_digest",
            "prediction_artifact_file_sha256",
            "feature_array_digest",
            "label_array_digest",
            "direct_pair_probability_digest",
            "catalog_probability_digest",
            "development_gate_digest",
            "validation_metrics_digest",
            "replay_feature_exact",
            "replay_prediction_exact",
            "replay_metrics_exact",
            "replay_model_refit_calls",
            "replay_model_inference_panel_count",
            "replay_pixel_reextract_group_count",
            "prior_failed_capacity_attempt_digest",
        )
    }
    for key in ("required_heads", "observed_pair_class_order", "valid_pair_class_order", "catalog_class_order"):
        if type(raw[key]) is not list:
            raise SkeletonGraphPassedFitProtocolError(f"{key} must be a list")
        common[key] = tuple(raw[key])
    if cls is SkeletonGraphPassedFitGap:
        common["failed_checks"] = tuple(raw["failed_checks"])
    value = object.__new__(cls)
    for name, item in common.items():
        object.__setattr__(value, name, item)
    object.__setattr__(value, "record_digest", raw["record_digest"])
    _validate_identity(value)
    if cls is SkeletonGraphPassedFitGap:
        expected_failed = tuple(
            name for name in REQUIRED_HEADS if name in value.failed_checks
        )
        if not expected_failed or value.failed_checks != expected_failed:
            raise SkeletonGraphPassedFitProtocolError("failed check order differs")
    expected_digest = "sha256:" + canonical_digest(value.content_data())
    if value.record_digest != expected_digest or value.to_data() != dict(raw):
        raise SkeletonGraphPassedFitProtocolError(
            "passed-fit outcome digest or canonical data differs"
        )
    return value


def _build_outcome(cls: type[Any], values: Mapping[str, object]) -> Any:
    provisional = object.__new__(cls)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    digest = "sha256:" + canonical_digest(provisional.content_data())
    data = {**provisional.content_data(), "record_digest": digest}
    return cls.from_data(data)


def _file_address(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def resolve_skeleton_graph_passed_fit_protocol(
    *,
    development_precommit_path: Path,
    development_result_path: Path,
    development_replay_path: Path,
    model_path: Path,
    feature_artifact_path: Path,
    prediction_artifact_path: Path,
) -> SkeletonGraphPassedFitOutcome:
    """Authenticate the full development chain and return PASS or typed GAP."""

    source_sha256()
    precommit, precommit_file = _record(
        development_precommit_path,
        schema=development.SCHEMA_PRECOMMIT,
        label="development precommit",
        maximum=development.RESULT_MAX_BYTES,
    )
    intended = precommit.get("intended_outputs")
    if not isinstance(intended, Mapping):
        raise SkeletonGraphPassedFitProtocolError("intended outputs differ")
    for key, path in (
        ("result", development_result_path),
        ("replay", development_replay_path),
        ("model", model_path),
        ("features", feature_artifact_path),
        ("predictions", prediction_artifact_path),
    ):
        _same_path(intended.get(key), path, key)
    try:
        verified_precommit = development._load_development_precommit(
            Path(development_precommit_path),
            expected_record_digest=precommit["record_digest"],
        )
    except Exception as exc:
        raise SkeletonGraphPassedFitProtocolError(
            "development precommit verification failed"
        ) from exc
    if verified_precommit != precommit:
        raise SkeletonGraphPassedFitProtocolError("development precommit differs")

    result, result_file = _record(
        development_result_path,
        schema=development.SCHEMA_RESULT,
        label="development result",
        maximum=development.RESULT_MAX_BYTES,
    )
    try:
        verified_result, _bundle, verified_model_bytes = development._load_result_and_model(
            precommit=verified_precommit,
            expected_result_record_digest=result["record_digest"],
            require_passed_development_gate=False,
        )
    except Exception as exc:
        raise SkeletonGraphPassedFitProtocolError(
            "development result/model verification failed"
        ) from exc
    if verified_result != result:
        raise SkeletonGraphPassedFitProtocolError("development result differs")
    model_bytes = _stable_bytes(
        model_path, label="development model", maximum=development.MODEL_MAX_BYTES
    )
    if model_bytes != verified_model_bytes or _file_address(model_bytes) != result.get(
        "model_file_sha256"
    ):
        raise SkeletonGraphPassedFitProtocolError("development model differs")

    features, feature_file = _record(
        feature_artifact_path,
        schema=development.SCHEMA_FEATURES,
        label="feature artifact",
        maximum=development.FEATURE_ARTIFACT_MAX_BYTES,
    )
    predictions, prediction_file = _record(
        prediction_artifact_path,
        schema=development.SCHEMA_PREDICTIONS,
        label="prediction artifact",
        maximum=development.PREDICTION_ARTIFACT_MAX_BYTES,
    )
    replay, replay_file = _record(
        development_replay_path,
        schema=development.SCHEMA_REPLAY,
        label="development replay",
        maximum=development.RESULT_MAX_BYTES,
    )
    if (
        feature_file != result.get("feature_artifact_file_sha256")
        or features["record_digest"]
        != result.get("feature_artifact_record_digest")
        or prediction_file != result.get("prediction_artifact_file_sha256")
        or predictions["record_digest"]
        != result.get("prediction_artifact_record_digest")
        or tuple(predictions.get("observed_pair_class_order", ()))
        != tuple(development.OBSERVED_TRAIN_PAIR_CLASS_ORDER)
        or tuple(predictions.get("valid_pair_class_order", ()))
        != tuple(development.VALID_PAIR_CLASS_ORDER)
        or tuple(predictions.get("catalog_class_order", ()))
        != tuple(development.CATALOG_CLASS_ORDER)
        or replay.get("precommit_record_digest") != precommit["record_digest"]
        or replay.get("result_record_digest") != result["record_digest"]
        or replay.get("model_file_sha256") != result["model_file_sha256"]
        or replay.get("source_sha256") != development.source_sha256()
        or replay.get("feature_replay_exact") is not True
        or replay.get("prediction_replay_exact") is not True
        or replay.get("metrics_replay_exact") is not True
        or replay.get("model_refit_calls") != 0
        or replay.get("model_inference_panel_count") != 1392
        or replay.get("pixel_reextract_group_count") != 12535
        or replay.get("probability_digests")
        != {
            "direct_pair": predictions.get("pair_probability_digest"),
            "catalog": predictions.get("catalog_probability_digest"),
        }
    ):
        raise SkeletonGraphPassedFitProtocolError(
            "feature, prediction, result, or replay chain differs"
        )

    gate = result.get("development_gate")
    if not isinstance(gate, Mapping):
        raise SkeletonGraphPassedFitProtocolError("development gate differs")
    failed = tuple(
        name
        for name, key in (
            ("direct_pair", "direct_pair_passed"),
            ("catalog_three_class", "catalog_three_class_passed"),
        )
        if gate.get(key) is not True
    )
    passed = not failed and tuple(result.get("promoted_heads", ())) == REQUIRED_HEADS
    if not passed and not failed:
        failed = tuple(
            name for name in REQUIRED_HEADS if name not in result.get("promoted_heads", ())
        )
    if passed:
        try:
            development.load_verified_development_model(
                precommit_path=Path(development_precommit_path),
                expected_precommit_record_digest=precommit["record_digest"],
                expected_result_record_digest=result["record_digest"],
                required_heads=REQUIRED_HEADS,
            )
        except Exception as exc:
            raise SkeletonGraphPassedFitProtocolError(
                "both-head authenticated model load failed"
            ) from exc

    values: dict[str, object] = {
        "passed_fit_algorithm_digest": PASSED_FIT_ALGORITHM_DIGEST,
        "passed_fit_authority_source_sha256": "sha256:" + source_sha256(),
        "development_source_sha256": PINNED_DEVELOPMENT_SOURCE_SHA256,
        "development_config_digest": PINNED_DEVELOPMENT_CONFIG_DIGEST,
        "claim_scope": development.CLAIM_SCOPE,
        "required_heads": REQUIRED_HEADS,
        "development_precommit_record_digest": precommit["record_digest"],
        "development_precommit_file_sha256": precommit_file,
        "development_result_record_digest": result["record_digest"],
        "development_result_file_sha256": result_file,
        "development_replay_record_digest": replay["record_digest"],
        "development_replay_file_sha256": replay_file,
        "model_file_sha256": result["model_file_sha256"],
        "model_size_bytes": len(model_bytes),
        "model_structure_digest": "sha256:" + canonical_digest(result["model_structure"]),
        "feature_artifact_record_digest": features["record_digest"],
        "feature_artifact_file_sha256": feature_file,
        "prediction_artifact_record_digest": predictions["record_digest"],
        "prediction_artifact_file_sha256": prediction_file,
        "feature_array_digest": features["feature_array_digest"],
        "label_array_digest": features["label_array_digest"],
        "direct_pair_probability_digest": predictions["pair_probability_digest"],
        "catalog_probability_digest": predictions["catalog_probability_digest"],
        "observed_pair_class_order": tuple(predictions["observed_pair_class_order"]),
        "valid_pair_class_order": tuple(predictions["valid_pair_class_order"]),
        "catalog_class_order": tuple(predictions["catalog_class_order"]),
        "development_gate_digest": "sha256:" + canonical_digest(gate),
        "validation_metrics_digest": "sha256:"
        + canonical_digest(result["unique_digest_validation_metrics"]),
        "replay_feature_exact": replay["feature_replay_exact"],
        "replay_prediction_exact": replay["prediction_replay_exact"],
        "replay_metrics_exact": replay["metrics_replay_exact"],
        "replay_model_refit_calls": replay["model_refit_calls"],
        "replay_model_inference_panel_count": replay["model_inference_panel_count"],
        "replay_pixel_reextract_group_count": replay["pixel_reextract_group_count"],
        "prior_failed_capacity_attempt_digest": "sha256:"
        + canonical_digest(result["prior_failed_capacity_attempt"]),
    }
    if passed:
        return _build_outcome(SkeletonGraphPassedFitProtocol, values)
    values["failed_checks"] = failed
    return _build_outcome(SkeletonGraphPassedFitGap, values)


def verify_skeleton_graph_passed_fit_protocol(
    outcome: SkeletonGraphPassedFitOutcome,
    *,
    development_precommit_path: Path,
    development_result_path: Path,
    development_replay_path: Path,
    model_path: Path,
    feature_artifact_path: Path,
    prediction_artifact_path: Path,
    expected_record_digest: str,
) -> SkeletonGraphPassedFitOutcome:
    """Freshly rebuild and compare one exact passed-fit outcome."""

    if type(outcome) not in (SkeletonGraphPassedFitProtocol, SkeletonGraphPassedFitGap):
        raise SkeletonGraphPassedFitProtocolError("passed-fit outcome type differs")
    expected = _address(expected_record_digest, "expected passed-fit record")
    rebuilt = resolve_skeleton_graph_passed_fit_protocol(
        development_precommit_path=development_precommit_path,
        development_result_path=development_result_path,
        development_replay_path=development_replay_path,
        model_path=model_path,
        feature_artifact_path=feature_artifact_path,
        prediction_artifact_path=prediction_artifact_path,
    )
    if type(rebuilt) is not type(outcome) or rebuilt.to_data() != outcome.to_data():
        raise SkeletonGraphPassedFitProtocolError("passed-fit replay differs")
    if outcome.record_digest != expected:
        raise SkeletonGraphPassedFitProtocolError("passed-fit expected digest differs")
    return outcome


__all__ = (
    "GAP_SCHEMA",
    "PASSED_FIT_ALGORITHM_DIGEST",
    "PROTOCOL_SCHEMA",
    "REQUIRED_HEADS",
    "SkeletonGraphPassedFitGap",
    "SkeletonGraphPassedFitOutcome",
    "SkeletonGraphPassedFitProtocol",
    "SkeletonGraphPassedFitProtocolError",
    "resolve_skeleton_graph_passed_fit_protocol",
    "source_sha256",
    "verify_skeleton_graph_passed_fit_protocol",
)
