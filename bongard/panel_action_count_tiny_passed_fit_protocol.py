"""Exact promotion boundary for the tiny observer's development fit.

This module does not authorize calibration, support, target, or query pixels.
It only answers whether an exact training precommit/result/checkpoint chain
earned the right to proceed to a later calibration gate.  A well-formed run
whose frozen development thresholds failed becomes a typed GAP.  It can be
replayed as audit evidence but can never instantiate ``TinyPassedFitProtocol``.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, TypeAlias

from bongard.canonical import canonical_digest, canonical_json
from bongard import panel_action_count_tiny_local_dev_command as core
from bongard import panel_action_count_tiny_local_train_command as trainer


PROTOCOL_SCHEMA = "gkm.bongard-tiny-local-action-passed-fit-protocol.v1"
GAP_SCHEMA = "gkm.bongard-tiny-local-action-passed-fit-gap.v1"
GATE_KEYS = (
    "arc_top1",
    "known_catalog_binary_balanced_accuracy",
    "straight_top1",
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


class TinyPassedFitProtocolError(RuntimeError):
    """The fit files, frozen gate, or checkpoint commitment differs."""


def source_sha256() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise TinyPassedFitProtocolError(f"{label} must be a sha256: address")
    return value


def _finite(value: object, label: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TinyPassedFitProtocolError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise TinyPassedFitProtocolError(f"{label} leaves [{lower},{upper}]")
    return result


def _stable_bytes(path: Path, *, label: str, maximum: int) -> bytes:
    supplied = Path(path)
    absolute = supplied.absolute()
    try:
        resolved = supplied.resolve(strict=True)
        before = supplied.lstat()
    except OSError as exc:
        raise TinyPassedFitProtocolError(f"cannot stat {label}: {exc}") from exc
    if resolved != absolute or stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TinyPassedFitProtocolError(f"{label} must be a regular nonsymlink file")
    if before.st_size <= 0 or before.st_size > maximum:
        raise TinyPassedFitProtocolError(f"{label} size leaves the frozen bound")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(supplied, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after_read = os.fstat(descriptor)
    except OSError as exc:
        raise TinyPassedFitProtocolError(f"cannot read {label}: {exc}") from exc
    finally:
        os.close(descriptor)
    try:
        after = supplied.lstat()
    except OSError as exc:
        raise TinyPassedFitProtocolError(f"cannot restat {label}: {exc}") from exc
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if not fingerprint(before) == fingerprint(opened) == fingerprint(after_read) == fingerprint(after):
        raise TinyPassedFitProtocolError(f"{label} changed while reading")
    raw = b"".join(chunks)
    if len(raw) != before.st_size or len(raw) > maximum:
        raise TinyPassedFitProtocolError(f"{label} read size differs")
    return raw


def _record(path: Path, *, schema: str, label: str) -> tuple[dict[str, Any], str]:
    raw = _stable_bytes(path, label=label, maximum=_MAX_JSON_BYTES)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TinyPassedFitProtocolError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise TinyPassedFitProtocolError(f"{label} is not canonical JSON plus newline")
    body = dict(value)
    found = body.pop("record_digest", None)
    if value.get("schema") != schema or found != "sha256:" + canonical_digest(body):
        raise TinyPassedFitProtocolError(f"{label} schema or record digest differs")
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise TinyPassedFitProtocolError(f"{label} fields differ")
    return value


def _expected_precommit_fields() -> set[str]:
    return {
        "architecture_id",
        "authorization_record_digest",
        "config_digest",
        "core_precommit_record_digest",
        "core_source_sha256",
        "decontaminated_occurrence_counts",
        "descriptor_conflict_audit_record_digest",
        "descriptor_target_conflict_audit",
        "fit_precommit_record_digest",
        "forbidden_cohorts",
        "intended_outputs",
        "pixels_read_by_precommit",
        "protocol",
        "schema",
        "source_sha256",
        "supervision_authority_record_digest",
        "record_digest",
    }


def _validate_precommit(
    value: Mapping[str, Any], *, result_path: Path, checkpoint_path: Path
) -> None:
    if set(value) != _expected_precommit_fields():
        raise TinyPassedFitProtocolError("training precommit fields differ")
    intended = value.get("intended_outputs")
    conflict = value.get("descriptor_target_conflict_audit")
    if (
        value.get("architecture_id") != core.ARCHITECTURE_ID
        or value.get("config_digest") != core.successor_config_digest()
        or value.get("core_source_sha256") != core.source_sha256()
        or value.get("source_sha256") != trainer.source_sha256()
        or value.get("fit_precommit_record_digest") != trainer.FIT_PRECOMMIT_DIGEST
        or value.get("descriptor_conflict_audit_record_digest")
        != trainer.COMMITTED_CONFLICT_AUDIT_DIGEST
        or value.get("decontaminated_occurrence_counts")
        != {"train": 11_200, "validation": 1_392}
        or value.get("forbidden_cohorts") != list(core.PROTOCOL["forbidden_cohorts"])
        or type(value.get("pixels_read_by_precommit")) is not int
        or value["pixels_read_by_precommit"] != 0
        or value.get("protocol")
        != json.loads(canonical_json(dict(core.PROTOCOL)))
        or not isinstance(intended, Mapping)
        or set(intended)
        != {"checkpoint", "core_precommit", "precommit", "replay", "result"}
        or intended.get("checkpoint") != str(checkpoint_path.resolve())
        or intended.get("result") != str(result_path.resolve())
        or not isinstance(conflict, Mapping)
        or conflict.get("committed_audit_record_digest")
        != trainer.COMMITTED_CONFLICT_AUDIT_DIGEST
        or conflict.get("all_effective_png_groups_descriptor_loss_eligible") is not True
        or type(conflict.get("authority_gap_occurrences")) is not int
        or conflict["authority_gap_occurrences"] != 0
        or type(conflict.get("descriptor_conflict_occurrences")) is not int
        or conflict["descriptor_conflict_occurrences"] != 0
        or type(conflict.get("descriptor_eligible_occurrences")) is not int
        or conflict["descriptor_eligible_occurrences"] != 12_592
        or type(conflict.get("effective_occurrence_count")) is not int
        or conflict["effective_occurrence_count"] != 12_592
        or conflict.get("descriptor_gap_is_never_none_or_zero") is not True
    ):
        raise TinyPassedFitProtocolError("training precommit policy or lineage differs")
    for key in (
        "authorization_record_digest",
        "core_precommit_record_digest",
        "supervision_authority_record_digest",
    ):
        _address(value.get(key), f"training precommit {key}")


_METRIC_FIELDS = {
    "arc_top1",
    "catalog_all_class_top1",
    "known_catalog_binary_balanced_accuracy",
    "panel_occurrences",
    "descriptor_deployment_authority",
    "descriptor_eligible_digest_groups",
    "descriptor_geometry_interval_hit",
    "descriptor_geometry_interval_hit_denominator",
    "descriptor_geometry_interval_hit_numerator",
    "descriptor_matched_primitive_accuracy",
    "descriptor_matched_primitive_denominator",
    "descriptor_matched_primitive_numerator",
    "descriptor_primitive_multiset_exact",
    "descriptor_primitive_multiset_exact_denominator",
    "descriptor_primitive_multiset_exact_numerator",
    "straight_top1",
}


def _validate_metrics(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _METRIC_FIELDS:
        raise TinyPassedFitProtocolError(f"{label} fields differ")
    for key in (
        "arc_top1",
        "catalog_all_class_top1",
        "known_catalog_binary_balanced_accuracy",
        "descriptor_geometry_interval_hit",
        "descriptor_matched_primitive_accuracy",
        "descriptor_primitive_multiset_exact",
        "straight_top1",
    ):
        _finite(value[key], f"{label} {key}", lower=0.0, upper=1.0)
    if value.get("descriptor_deployment_authority") is not False:
        raise TinyPassedFitProtocolError(f"{label} descriptor authority differs")
    integer_expectations = {
        "panel_occurrences": 1_392,
        "descriptor_eligible_digest_groups": 1_392,
    }
    for key, expected in integer_expectations.items():
        if type(value.get(key)) is not int or value[key] != expected:
            raise TinyPassedFitProtocolError(f"{label} {key} differs")
    for prefix in (
        "descriptor_geometry_interval_hit",
        "descriptor_matched_primitive",
        "descriptor_primitive_multiset_exact",
    ):
        numerator = value[f"{prefix}_numerator"]
        denominator = value[f"{prefix}_denominator"]
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or denominator <= 0
            or not 0 <= numerator <= denominator
        ):
            raise TinyPassedFitProtocolError(f"{label} {prefix} counts differ")
    ratios = (
        ("descriptor_geometry_interval_hit", "descriptor_geometry_interval_hit"),
        ("descriptor_matched_primitive_accuracy", "descriptor_matched_primitive"),
        ("descriptor_primitive_multiset_exact", "descriptor_primitive_multiset_exact"),
    )
    for metric, prefix in ratios:
        if value[metric] != value[f"{prefix}_numerator"] / value[f"{prefix}_denominator"]:
            raise TinyPassedFitProtocolError(f"{label} {metric} arithmetic differs")
    return value


def _expected_result_fields() -> set[str]:
    return {
        "architecture_id",
        "authorization_record_digest",
        "checkpoint_raw_sha256",
        "checkpoint_state_dict_sha256",
        "config_digest",
        "decontaminated_occurrence_counts",
        "descriptor_target_conflict_audit",
        "forbidden_cohorts_opened",
        "history",
        "pixel_occurrences_reread",
        "runtime_budget",
        "runtime_seconds",
        "schema",
        "selected_epoch",
        "source_sha256",
        "training_precommit_record_digest",
        "validation_gate",
        "validation_metrics",
        "validation_prediction_rows_digest",
        "record_digest",
    }


def _validate_result(
    value: Mapping[str, Any], *, precommit: Mapping[str, Any]
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    if set(value) != _expected_result_fields():
        raise TinyPassedFitProtocolError("training result fields differ")
    selected = value.get("selected_epoch")
    history = value.get("history")
    budget = value.get("runtime_budget")
    metrics = _validate_metrics(value.get("validation_metrics"), label="validation metrics")
    if (
        value.get("architecture_id") != core.ARCHITECTURE_ID
        or value.get("config_digest") != core.successor_config_digest()
        or value.get("source_sha256") != trainer.source_sha256()
        or value.get("training_precommit_record_digest") != precommit["record_digest"]
        or value.get("authorization_record_digest")
        != precommit["authorization_record_digest"]
        or value.get("decontaminated_occurrence_counts")
        != {"train": 11_200, "validation": 1_392}
        or value.get("descriptor_target_conflict_audit")
        != precommit["descriptor_target_conflict_audit"]
        or type(value.get("forbidden_cohorts_opened")) is not int
        or value["forbidden_cohorts_opened"] != 0
        or type(value.get("pixel_occurrences_reread")) is not int
        or value["pixel_occurrences_reread"] != 12_592
        or type(selected) is not int
        or selected not in range(int(core.PROTOCOL["epochs"]))
        or not isinstance(history, list)
        or len(history) != int(core.PROTOCOL["epochs"])
        or not isinstance(budget, Mapping)
        or budget
        != {
            "cooperative_batch_boundary_deadline": True,
            "finalization_reserve_seconds": trainer.FINALIZATION_RESERVE_SECONDS,
            "limit_seconds": float(core.PROTOCOL["maximum_wall_runtime_seconds"]),
            "passed_before_result_seal": True,
        }
    ):
        raise TinyPassedFitProtocolError("training result policy or lineage differs")
    _finite(
        value.get("runtime_seconds"),
        "training runtime",
        lower=0.0,
        upper=float(core.PROTOCOL["maximum_wall_runtime_seconds"]),
    )
    _address(value.get("checkpoint_raw_sha256"), "result checkpoint bytes")
    _address(value.get("checkpoint_state_dict_sha256"), "result checkpoint state")
    _address(value.get("validation_prediction_rows_digest"), "validation predictions")
    for index, row in enumerate(history):
        if not isinstance(row, Mapping) or set(row) != _METRIC_FIELDS | {
            "epoch",
            "training_group_mean_loss",
        }:
            raise TinyPassedFitProtocolError("training history fields differ")
        if row.get("epoch") != index:
            raise TinyPassedFitProtocolError("training history epoch order differs")
        _validate_metrics(
            {key: row[key] for key in _METRIC_FIELDS}, label=f"history epoch {index}"
        )
        _finite(
            row.get("training_group_mean_loss"),
            f"history epoch {index} loss",
            lower=0.0,
            upper=float("inf"),
        )
    selected_metrics = {key: history[selected][key] for key in _METRIC_FIELDS}
    if canonical_json(selected_metrics) != canonical_json(dict(metrics)):
        raise TinyPassedFitProtocolError("selected checkpoint metrics differ")
    rank_key = lambda row: (
        row["straight_top1"],
        row["known_catalog_binary_balanced_accuracy"],
        row["descriptor_primitive_multiset_exact"],
        row["descriptor_matched_primitive_accuracy"],
        row["descriptor_geometry_interval_hit"],
        row["arc_top1"],
        -row["epoch"],
    )
    if selected != max(range(len(history)), key=lambda index: rank_key(history[index])):
        raise TinyPassedFitProtocolError("selected checkpoint rank differs")
    expected_gate = trainer._validation_gate(metrics)
    if value.get("validation_gate") != expected_gate:
        raise TinyPassedFitProtocolError("development validation gate differs")
    failed = tuple(key for key in GATE_KEYS if expected_gate["checks"][key] is not True)
    if expected_gate["passed"] is not (not failed):
        raise TinyPassedFitProtocolError("development gate conjunction differs")
    return metrics, failed


def _common_content(value: object) -> dict[str, object]:
    return {
        "architecture_id": value.architecture_id,
        "config_digest": value.config_digest,
        "core_source_sha256": value.core_source_sha256,
        "trainer_source_sha256": value.trainer_source_sha256,
        "training_precommit_record_digest": value.training_precommit_record_digest,
        "training_precommit_file_sha256": value.training_precommit_file_sha256,
        "training_result_record_digest": value.training_result_record_digest,
        "training_result_file_sha256": value.training_result_file_sha256,
        "checkpoint_raw_sha256": value.checkpoint_raw_sha256,
        "checkpoint_state_dict_sha256": value.checkpoint_state_dict_sha256,
        "selected_epoch": value.selected_epoch,
        "validation_gate_digest": value.validation_gate_digest,
        "validation_metrics_digest": value.validation_metrics_digest,
        "passed_fit_authority_source_sha256": source_sha256(),
        "calibration_authorized": False,
        "support_query_inference_authorized": False,
        "benchmark_sealable": False,
    }


@dataclass(frozen=True, slots=True)
class TinyPassedFitGap:
    architecture_id: str
    config_digest: str
    core_source_sha256: str
    trainer_source_sha256: str
    training_precommit_record_digest: str
    training_precommit_file_sha256: str
    training_result_record_digest: str
    training_result_file_sha256: str
    checkpoint_raw_sha256: str
    checkpoint_state_dict_sha256: str
    selected_epoch: int
    validation_gate_digest: str
    validation_metrics_digest: str
    failed_checks: tuple[str, ...]
    record_digest: str

    def __post_init__(self) -> None:
        _validate_outcome_identity(self)
        if not self.failed_checks or any(item not in GATE_KEYS for item in self.failed_checks):
            raise TinyPassedFitProtocolError("passed-fit GAP checks differ")
        if self.failed_checks != tuple(key for key in GATE_KEYS if key in self.failed_checks):
            raise TinyPassedFitProtocolError("passed-fit GAP check order differs")
        if self.record_digest != "sha256:" + canonical_digest(self.content_data()):
            raise TinyPassedFitProtocolError("passed-fit GAP digest differs")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": GAP_SCHEMA,
            **_common_content(self),
            "disposition": "gap",
            "gap_reason_code": "development_validation_gate_failed",
            "failed_checks": list(self.failed_checks),
            "development_fit_passed": False,
            "failed_run_replayable_as_audit_evidence": True,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "TinyPassedFitGap":
        raw = _fields(value, _gap_fields(), "passed-fit GAP")
        if (
            raw["schema"] != GAP_SCHEMA
            or raw["disposition"] != "gap"
            or raw["gap_reason_code"] != "development_validation_gate_failed"
            or raw["development_fit_passed"] is not False
            or raw["failed_run_replayable_as_audit_evidence"] is not True
            or type(raw["failed_checks"]) is not list
        ):
            raise TinyPassedFitProtocolError("passed-fit GAP policy differs")
        result = cls(
            **_common_from_data(raw),
            failed_checks=tuple(raw["failed_checks"]),
            record_digest=raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise TinyPassedFitProtocolError("passed-fit GAP is not canonical")
        return result

    def verify(
        self,
        *,
        training_precommit_path: Path,
        training_result_path: Path,
        checkpoint_path: Path,
        expected_gap_address: str,
    ) -> "TinyPassedFitGap":
        expected = _address(expected_gap_address, "expected passed-fit GAP")
        rebuilt = _from_files(
            training_precommit_path=training_precommit_path,
            training_result_path=training_result_path,
            checkpoint_path=checkpoint_path,
        )
        if type(rebuilt) is not TinyPassedFitGap or rebuilt != self or self.record_digest != expected:
            raise TinyPassedFitProtocolError("passed-fit GAP verification differs")
        return self


@dataclass(frozen=True, slots=True)
class TinyPassedFitProtocol:
    architecture_id: str
    config_digest: str
    core_source_sha256: str
    trainer_source_sha256: str
    training_precommit_record_digest: str
    training_precommit_file_sha256: str
    training_result_record_digest: str
    training_result_file_sha256: str
    checkpoint_raw_sha256: str
    checkpoint_state_dict_sha256: str
    selected_epoch: int
    validation_gate_digest: str
    validation_metrics_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        _validate_outcome_identity(self)
        if self.record_digest != "sha256:" + canonical_digest(self.content_data()):
            raise TinyPassedFitProtocolError("passed-fit protocol digest differs")

    @property
    def protocol_address(self) -> str:
        return self.record_digest

    def content_data(self) -> dict[str, object]:
        return {
            "schema": PROTOCOL_SCHEMA,
            **_common_content(self),
            "development_fit_passed": True,
            "all_frozen_validation_checks_passed": True,
            "checkpoint_fresh_loaded_with_pass_requirement": True,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "TinyPassedFitProtocol":
        raw = _fields(value, _protocol_fields(), "passed-fit protocol")
        if (
            raw["schema"] != PROTOCOL_SCHEMA
            or raw["development_fit_passed"] is not True
            or raw["all_frozen_validation_checks_passed"] is not True
            or raw["checkpoint_fresh_loaded_with_pass_requirement"] is not True
        ):
            raise TinyPassedFitProtocolError("passed-fit protocol policy differs")
        result = cls(
            **_common_from_data(raw),
            record_digest=raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise TinyPassedFitProtocolError("passed-fit protocol is not canonical")
        return result

    @classmethod
    def from_files(
        cls,
        *,
        training_precommit_path: Path,
        training_result_path: Path,
        checkpoint_path: Path,
    ) -> "TinyPassedFitOutcome":
        if cls is not TinyPassedFitProtocol:
            raise TypeError("passed-fit protocol subclasses are not accepted")
        return _from_files(
            training_precommit_path=training_precommit_path,
            training_result_path=training_result_path,
            checkpoint_path=checkpoint_path,
        )

    def verify(
        self,
        *,
        training_precommit_path: Path,
        training_result_path: Path,
        checkpoint_path: Path,
        expected_protocol_address: str,
    ) -> "TinyPassedFitProtocol":
        expected = _address(expected_protocol_address, "expected passed-fit protocol")
        rebuilt = self.from_files(
            training_precommit_path=training_precommit_path,
            training_result_path=training_result_path,
            checkpoint_path=checkpoint_path,
        )
        if type(rebuilt) is not TinyPassedFitProtocol or rebuilt != self or self.record_digest != expected:
            raise TinyPassedFitProtocolError("passed-fit protocol verification differs")
        return self


TinyPassedFitOutcome: TypeAlias = TinyPassedFitProtocol | TinyPassedFitGap


_COMMON_FIELDS = {
    "architecture_id",
    "config_digest",
    "core_source_sha256",
    "trainer_source_sha256",
    "training_precommit_record_digest",
    "training_precommit_file_sha256",
    "training_result_record_digest",
    "training_result_file_sha256",
    "checkpoint_raw_sha256",
    "checkpoint_state_dict_sha256",
    "selected_epoch",
    "validation_gate_digest",
    "validation_metrics_digest",
    "passed_fit_authority_source_sha256",
    "calibration_authorized",
    "support_query_inference_authorized",
    "benchmark_sealable",
}


def _protocol_fields() -> set[str]:
    return _COMMON_FIELDS | {
        "schema",
        "development_fit_passed",
        "all_frozen_validation_checks_passed",
        "checkpoint_fresh_loaded_with_pass_requirement",
        "record_digest",
    }


def _gap_fields() -> set[str]:
    return _COMMON_FIELDS | {
        "schema",
        "disposition",
        "gap_reason_code",
        "failed_checks",
        "development_fit_passed",
        "failed_run_replayable_as_audit_evidence",
        "record_digest",
    }


def _common_from_data(raw: Mapping[str, Any]) -> dict[str, Any]:
    if (
        raw["passed_fit_authority_source_sha256"] != source_sha256()
        or raw["calibration_authorized"] is not False
        or raw["support_query_inference_authorized"] is not False
        or raw["benchmark_sealable"] is not False
    ):
        raise TinyPassedFitProtocolError("passed-fit common policy differs")
    return {
        key: raw[key]
        for key in (
            "architecture_id",
            "config_digest",
            "core_source_sha256",
            "trainer_source_sha256",
            "training_precommit_record_digest",
            "training_precommit_file_sha256",
            "training_result_record_digest",
            "training_result_file_sha256",
            "checkpoint_raw_sha256",
            "checkpoint_state_dict_sha256",
            "selected_epoch",
            "validation_gate_digest",
            "validation_metrics_digest",
        )
    }


def _validate_outcome_identity(value: object) -> None:
    if value.architecture_id != core.ARCHITECTURE_ID or value.config_digest != core.successor_config_digest():
        raise TinyPassedFitProtocolError("passed-fit architecture or config differs")
    if value.core_source_sha256 != core.source_sha256() or value.trainer_source_sha256 != trainer.source_sha256():
        raise TinyPassedFitProtocolError("passed-fit source binding differs")
    for label, item in (
        ("training precommit record", value.training_precommit_record_digest),
        ("training precommit file", value.training_precommit_file_sha256),
        ("training result record", value.training_result_record_digest),
        ("training result file", value.training_result_file_sha256),
        ("checkpoint bytes", value.checkpoint_raw_sha256),
        ("checkpoint state", value.checkpoint_state_dict_sha256),
        ("validation gate", value.validation_gate_digest),
        ("validation metrics", value.validation_metrics_digest),
    ):
        _address(item, label)
    if type(value.selected_epoch) is not int or value.selected_epoch not in range(int(core.PROTOCOL["epochs"])):
        raise TinyPassedFitProtocolError("passed-fit epoch differs")


def _outcome_values(
    *,
    precommit: Mapping[str, Any],
    precommit_file_sha256: str,
    result: Mapping[str, Any],
    result_file_sha256: str,
    checkpoint_raw_sha256: str,
) -> dict[str, object]:
    return {
        "architecture_id": core.ARCHITECTURE_ID,
        "config_digest": core.successor_config_digest(),
        "core_source_sha256": core.source_sha256(),
        "trainer_source_sha256": trainer.source_sha256(),
        "training_precommit_record_digest": precommit["record_digest"],
        "training_precommit_file_sha256": precommit_file_sha256,
        "training_result_record_digest": result["record_digest"],
        "training_result_file_sha256": result_file_sha256,
        "checkpoint_raw_sha256": checkpoint_raw_sha256,
        "checkpoint_state_dict_sha256": result["checkpoint_state_dict_sha256"],
        "selected_epoch": result["selected_epoch"],
        "validation_gate_digest": "sha256:" + canonical_digest(result["validation_gate"]),
        "validation_metrics_digest": "sha256:" + canonical_digest(result["validation_metrics"]),
    }


def _from_files(
    *,
    training_precommit_path: Path,
    training_result_path: Path,
    checkpoint_path: Path,
) -> TinyPassedFitOutcome:
    precommit, precommit_file = _record(
        training_precommit_path, schema=trainer.PRECOMMIT_SCHEMA, label="training precommit"
    )
    result, result_file = _record(
        training_result_path, schema=trainer.RESULT_SCHEMA, label="training result"
    )
    _validate_precommit(precommit, result_path=training_result_path, checkpoint_path=checkpoint_path)
    _metrics, failed = _validate_result(result, precommit=precommit)
    checkpoint_before = _stable_bytes(
        checkpoint_path, label="tiny checkpoint", maximum=_MAX_CHECKPOINT_BYTES
    )
    try:
        _model, envelope, checkpoint_raw_sha256 = core.load_verified_checkpoint(
            checkpoint_path,
            expected_training_precommit_record_digest=precommit["record_digest"],
            training_result=result,
            expected_training_result_record_digest=result["record_digest"],
            require_passed_development_gate=not failed,
        )
    except Exception as exc:
        raise TinyPassedFitProtocolError("tiny checkpoint verification failed") from exc
    checkpoint_after = _stable_bytes(
        checkpoint_path, label="tiny checkpoint", maximum=_MAX_CHECKPOINT_BYTES
    )
    if (
        checkpoint_before != checkpoint_after
        or checkpoint_raw_sha256 != "sha256:" + hashlib.sha256(checkpoint_before).hexdigest()
        or envelope["state_dict_sha256"] != result["checkpoint_state_dict_sha256"]
    ):
        raise TinyPassedFitProtocolError("tiny checkpoint changed or differs")
    values = _outcome_values(
        precommit=precommit,
        precommit_file_sha256=precommit_file,
        result=result,
        result_file_sha256=result_file,
        checkpoint_raw_sha256=checkpoint_raw_sha256,
    )
    if failed:
        values["failed_checks"] = failed
        provisional = object.__new__(TinyPassedFitGap)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return TinyPassedFitGap(
            **values,
            record_digest="sha256:" + canonical_digest(provisional.content_data()),
        )
    provisional = object.__new__(TinyPassedFitProtocol)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return TinyPassedFitProtocol(
        **values,
        record_digest="sha256:" + canonical_digest(provisional.content_data()),
    )


__all__ = (
    "GAP_SCHEMA",
    "PROTOCOL_SCHEMA",
    "TinyPassedFitGap",
    "TinyPassedFitOutcome",
    "TinyPassedFitProtocol",
    "TinyPassedFitProtocolError",
    "source_sha256",
)
