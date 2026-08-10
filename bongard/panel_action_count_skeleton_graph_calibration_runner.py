"""Staged split-conformal calibration for the fixed-32 skeleton observer.

The only pixel-facing stage produces role-free raw inference through the
separate inference-custody authority.  Truth is unavailable until that raw
batch, its independent recomputation receipt, and the anonymous occurrence
join have been written, fsynced, freshly reloaded, and reverified.  Cold replay
uses only the archived probabilities and delayed labels.

There is intentionally no CLI and no corpus, label, action-program, target,
support, or query loader in this module.  Callers supply the calibration-only
pixel reader, inference function, and delayed label-reader factory.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, TypeAlias

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger
from bongard import panel_action_count_skeleton_graph_calibration_prereg as prereg
from bongard import panel_action_count_skeleton_graph_inference_custody as inference_custody
from bongard import panel_action_count_skeleton_graph_passed_fit_protocol as passed_fit_authority
from bongard.panel_action_count_skeleton_graph_inference_custody import (
    SkeletonGraphInferenceRecomputeReceipt,
    SkeletonGraphRawInferenceBatch,
    cold_replay_raw_inference,
    create_raw_inference_batch,
    fresh_verify_raw_inference_batch,
)
from bongard.panel_action_count_skeleton_graph_passed_fit_protocol import (
    SkeletonGraphPassedFitGap,
    SkeletonGraphPassedFitOutcome,
    SkeletonGraphPassedFitProtocol,
    verify_skeleton_graph_passed_fit_protocol,
)


AUTHORIZATION_SCHEMA = (
    "gkm.bongard-skeleton-graph-calibration-exposure-authorization.v2"
)
EXECUTION_AUTHORIZATION_SCHEMA = (
    "gkm.bongard-skeleton-graph-calibration-execution-authorization.v2"
)
PRECOMMIT_SCHEMA = "gkm.bongard-skeleton-graph-calibration-execution-precommit.v2"
OUTPUT_ROOT_CLAIM_SCHEMA = (
    "gkm.bongard-skeleton-graph-calibration-output-root-claim.v1"
)
TERMINAL_STATE_SCHEMA = (
    "gkm.bongard-skeleton-graph-calibration-terminal-state.v1"
)
ATTEMPT_SCHEMA = "gkm.bongard-skeleton-graph-calibration-attempt.v1"
CAMPAIGN_ATTEMPT_AUTHORITY_SCHEMA = (
    "gkm.bongard-skeleton-graph-calibration-campaign-attempt-intent.v2"
)
PREDICTION_SCHEMA = "gkm.bongard-skeleton-graph-calibration-raw-predictions.v1"
LABEL_ROW_SCHEMA = "gkm.bongard-skeleton-graph-calibration-delayed-label-row.v1"
LABEL_BATCH_SCHEMA = "gkm.bongard-skeleton-graph-calibration-delayed-labels.v1"
GENERIC_GRANT_SCHEMA = "gkm.bongard-skeleton-graph-generic-population-grant.v1"
SAME_FAMILY_GRANT_SCHEMA = (
    "gkm.bongard-skeleton-graph-same-family-population-grant.v1"
)
GAP_SCHEMA = "gkm.bongard-skeleton-graph-calibration-gap.v1"
REPLAY_SCHEMA = "gkm.bongard-skeleton-graph-calibration-cold-replay.v1"

_INTEGRITY_FAILURE_STAGES = frozenset(
    {
        "execution_authorization",
        "precommit",
        "prediction_attempt",
        "pixel_or_inference_callback",
        "prediction_write",
        "prediction_fresh_reload",
        "label_attempt",
        "delayed_label_callback",
        "delayed_label_write",
        "population_evaluation",
        "outcome_write",
        "recovered_interrupted_execution",
    }
)

PINNED_PREREGISTRATION_RECORD_DIGEST = (
    "sha256:7ebecfaf1a745a1d07d5c0805ba0a36f48ebd8871662be3432f82fcf55a09724"
)
PINNED_PREREGISTRATION_FILE_SHA256 = (
    "sha256:0431ef93c44b2186a8f30d5f080d719ed88e48bd80aad997f8e9fd19929b0038"
)
PINNED_PREREGISTRATION_SOURCE_SHA256 = (
    "sha256:9413f2f00a32fa38adcbab0d745a398881a20437f930f9d202ffff74e35b67a6"
)
PINNED_PREREGISTRATION_COMMIT = "59c4a8c5b986f920677cb74b3ad384380b15b768"
PINNED_PASSED_FIT_COMMIT = "78aef7cb932ceb3dbb9006dadb71c6c1f1fa1d00"
PINNED_PASSED_FIT_SOURCE_SHA256 = (
    "c7cd9bd5abfdcbc8f846b45be3478c679d1ddd03e2380de4e9e0e95217eccc65"
)
PINNED_PASSED_FIT_ALGORITHM_DIGEST = (
    "sha256:eacc49c3304cbd3b8de4a6bb6208e25fe7d3878ed6fae49ed52fa9bc73b9151d"
)
PINNED_PASSED_FIT_PROTOCOL_RECORD_DIGEST = (
    "sha256:765b77632ad35012996be71e6effb2f56a1dbfc50080acffc6239bba84ceb15a"
)
PINNED_INFERENCE_COMMIT = "a3c27f9dc9f79258de267dd420f8b7dac805f9bd"
PINNED_INFERENCE_SOURCE_SHA256 = (
    "48c7777a9eb9d50ef1cceb509879f371e66166b4d081f39113821ad9d118dfed"
)
PINNED_INFERENCE_ALGORITHM_DIGEST = (
    "sha256:f18dd69a3bbebb55ff600121819ba4fab6f41d694a69e21b88bc0925fcb8fbb4"
)

MAX_INPUT_OCCURRENCES = 4_096
PANELS_PER_TASK = 14
SAME_FAMILY_TARGET_TASK_ID = "hd_convex-has_four_straight_lines_0000"

_SHA_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"hd/([^/]+)/([01])/([0-6])\.png\Z")
_TOKEN = re.compile(r"anon_[0-9a-f]{64}\Z")


class SkeletonGraphCalibrationRunnerError(RuntimeError):
    """A causal stage, content address, row join, or frozen policy differs."""


class SkeletonGraphCalibrationScope(str, Enum):
    GENERIC_V3 = "generic_v3"
    SAME_FAMILY = "same_family"


_EXPOSURE_AUTHORIZATION_ISSUANCE_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class SkeletonGraphCalibrationExposureAuthorization:
    """Metadata-only capability backed by one durable exposure-ledger append.

    The public value is insufficient on its own: every pixel-facing entry point
    freshly verifies the fixed intent, predecessor, exact one-event successor,
    and content-addressed authorization bytes before invoking a callback.
    """

    scope: SkeletonGraphCalibrationScope
    population_scope: str
    task_ids: tuple[str, ...]
    panel_ids: tuple[str, ...]
    task_ids_digest: str
    panel_ids_digest: str
    intended_output_directory: str
    intended_output_parent_path: str
    intended_output_parent_st_dev: int
    intended_output_parent_st_ino: int
    intended_output_parent_st_mode: int
    preregistration_record_digest: str
    preregistration_file_sha256: str
    preregistration_source_sha256: str
    passed_fit_authority_source_sha256: str
    passed_fit_algorithm_digest: str
    passed_fit_record_digest: str
    exposure_predecessor_ledger_digest: str
    exposure_predecessor_file_sha256: str
    exposure_predecessor_filename: str
    exposure_event_digest: str
    exposure_event_observed_at: str
    exposure_successor_ledger_digest: str
    exposure_successor_file_sha256: str
    exposure_successor_filename: str
    campaign_intent_record_digest: str
    campaign_intent_file_sha256: str
    campaign_intent_filename: str
    calibration_pixels_authorized: bool
    target_pixels_authorized: bool
    query_pixels_authorized: bool
    support_pixels_authorized: bool
    official_test_pixels_authorized: bool
    action_labels_or_programs_authorized: bool
    authenticated_calibration_execution: bool
    production_adapter_authorized: bool
    runner_source_sha256: str
    runner_algorithm_digest: str
    record_digest: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SkeletonGraphCalibrationRunnerError(
            "exposure authorizations are issued only by the durable metadata stage"
        )

    @property
    def file_sha256(self) -> str:
        return _file_address(canonical_json(self.to_data()) + b"\n")

    @property
    def filename(self) -> str:
        return self.record_digest.removeprefix("sha256:") + ".calibration-authorization.json"

    def content_data(self) -> dict[str, object]:
        return {
            "schema": AUTHORIZATION_SCHEMA,
            "scope": self.scope.value,
            "population_scope": self.population_scope,
            "task_ids": list(self.task_ids),
            "panel_ids": list(self.panel_ids),
            "task_ids_digest": self.task_ids_digest,
            "panel_ids_digest": self.panel_ids_digest,
            "intended_output_directory": self.intended_output_directory,
            "intended_output_parent_path": self.intended_output_parent_path,
            "intended_output_parent_st_dev": self.intended_output_parent_st_dev,
            "intended_output_parent_st_ino": self.intended_output_parent_st_ino,
            "intended_output_parent_st_mode": self.intended_output_parent_st_mode,
            "preregistration_record_digest": self.preregistration_record_digest,
            "preregistration_file_sha256": self.preregistration_file_sha256,
            "preregistration_source_sha256": self.preregistration_source_sha256,
            "passed_fit_authority_source_sha256": (
                self.passed_fit_authority_source_sha256
            ),
            "passed_fit_algorithm_digest": self.passed_fit_algorithm_digest,
            "passed_fit_record_digest": self.passed_fit_record_digest,
            "exposure_predecessor_ledger_digest": (
                self.exposure_predecessor_ledger_digest
            ),
            "exposure_predecessor_file_sha256": (
                self.exposure_predecessor_file_sha256
            ),
            "exposure_predecessor_filename": self.exposure_predecessor_filename,
            "exposure_event_digest": self.exposure_event_digest,
            "exposure_event_observed_at": self.exposure_event_observed_at,
            "exposure_successor_ledger_digest": self.exposure_successor_ledger_digest,
            "exposure_successor_file_sha256": self.exposure_successor_file_sha256,
            "exposure_successor_filename": self.exposure_successor_filename,
            "campaign_intent_record_digest": self.campaign_intent_record_digest,
            "campaign_intent_file_sha256": self.campaign_intent_file_sha256,
            "campaign_intent_filename": self.campaign_intent_filename,
            "calibration_pixels_authorized": self.calibration_pixels_authorized,
            "target_pixels_authorized": self.target_pixels_authorized,
            "query_pixels_authorized": self.query_pixels_authorized,
            "support_pixels_authorized": self.support_pixels_authorized,
            "official_test_pixels_authorized": self.official_test_pixels_authorized,
            "action_labels_or_programs_authorized": (
                self.action_labels_or_programs_authorized
            ),
            "authenticated_calibration_execution": (
                self.authenticated_calibration_execution
            ),
            "production_adapter_authorized": self.production_adapter_authorized,
            "runner_source_sha256": self.runner_source_sha256,
            "runner_algorithm_digest": self.runner_algorithm_digest,
            "metadata_only_authorization": True,
            "real_exposure_successor_fsync_reloaded_before_issuance": True,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def _issue(
        cls, *, issuance_token: object, values: Mapping[str, object]
    ) -> "SkeletonGraphCalibrationExposureAuthorization":
        if issuance_token is not _EXPOSURE_AUTHORIZATION_ISSUANCE_TOKEN:
            raise SkeletonGraphCalibrationRunnerError(
                "exposure authorization issuance token differs"
            )
        result = object.__new__(cls)
        for name in cls.__dataclass_fields__:
            if name == "record_digest":
                continue
            if name not in values:
                raise SkeletonGraphCalibrationRunnerError(
                    f"exposure authorization value {name} is missing"
                )
            object.__setattr__(result, name, values[name])
        object.__setattr__(
            result,
            "record_digest",
            "sha256:" + canonical_digest(result.content_data()),
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if type(self.scope) is not SkeletonGraphCalibrationScope:
            raise SkeletonGraphCalibrationRunnerError("authorization scope differs")
        if (
            type(self.population_scope) is not str
            or not self.population_scope
            or type(self.task_ids) is not tuple
            or not self.task_ids
            or any(type(item) is not str or not item for item in self.task_ids)
            or len(set(self.task_ids)) != len(self.task_ids)
            or type(self.panel_ids) is not tuple
            or not self.panel_ids
            or any(type(item) is not str or _PANEL_ID.fullmatch(item) is None for item in self.panel_ids)
            or len(set(self.panel_ids)) != len(self.panel_ids)
        ):
            raise SkeletonGraphCalibrationRunnerError("authorization cohort differs")
        expected_panels = tuple(
            f"hd/{task_id}/{side}/{ordinal}.png"
            for task_id in self.task_ids
            for side in (1, 0)
            for ordinal in range(7)
        )
        if (
            self.panel_ids != expected_panels
            or self.task_ids_digest != "sha256:" + canonical_digest(self.task_ids)
            or self.panel_ids_digest != "sha256:" + canonical_digest(self.panel_ids)
        ):
            raise SkeletonGraphCalibrationRunnerError("authorization cohort digest differs")
        for value, label in (
            (self.intended_output_parent_st_dev, "output parent device"),
            (self.intended_output_parent_st_ino, "output parent inode"),
            (self.intended_output_parent_st_mode, "output parent mode"),
        ):
            _exact_int(value, label)
        if (
            type(self.intended_output_directory) is not str
            or type(self.intended_output_parent_path) is not str
            or not self.intended_output_directory
            or not self.intended_output_parent_path
            or type(self.exposure_event_observed_at) is not str
            or not self.exposure_event_observed_at
            or any(
                type(value) is not str or Path(value).name != value
                for value in (
                    self.exposure_predecessor_filename,
                    self.exposure_successor_filename,
                    self.campaign_intent_filename,
                )
            )
        ):
            raise SkeletonGraphCalibrationRunnerError("authorization path/time differs")
        for value, label in (
            (self.preregistration_record_digest, "preregistration record"),
            (self.preregistration_file_sha256, "preregistration file"),
            (self.preregistration_source_sha256, "preregistration source"),
            (self.passed_fit_authority_source_sha256, "passed-fit source"),
            (self.passed_fit_algorithm_digest, "passed-fit algorithm"),
            (self.passed_fit_record_digest, "passed-fit record"),
            (self.exposure_predecessor_ledger_digest, "predecessor ledger"),
            (self.exposure_predecessor_file_sha256, "predecessor file"),
            (self.exposure_event_digest, "exposure event"),
            (self.exposure_successor_ledger_digest, "successor ledger"),
            (self.exposure_successor_file_sha256, "successor file"),
            (self.campaign_intent_record_digest, "campaign intent record"),
            (self.campaign_intent_file_sha256, "campaign intent file"),
            (self.runner_source_sha256, "runner source"),
            (self.runner_algorithm_digest, "runner algorithm"),
        ):
            _address(value, label)
        if (
            self.preregistration_record_digest != PINNED_PREREGISTRATION_RECORD_DIGEST
            or self.preregistration_file_sha256 != PINNED_PREREGISTRATION_FILE_SHA256
            or self.preregistration_source_sha256 != PINNED_PREREGISTRATION_SOURCE_SHA256
            or self.passed_fit_authority_source_sha256
            != "sha256:" + PINNED_PASSED_FIT_SOURCE_SHA256
            or self.passed_fit_algorithm_digest != PINNED_PASSED_FIT_ALGORITHM_DIGEST
            or self.passed_fit_record_digest != PINNED_PASSED_FIT_PROTOCOL_RECORD_DIGEST
            or self.exposure_successor_filename
            != self.exposure_successor_ledger_digest.removeprefix("sha256:")
            + ".exposure.json"
            or self.calibration_pixels_authorized is not True
            or self.target_pixels_authorized is not False
            or self.query_pixels_authorized is not False
            or self.support_pixels_authorized is not False
            or self.official_test_pixels_authorized is not False
            or self.action_labels_or_programs_authorized is not False
            or self.authenticated_calibration_execution is not False
            or self.production_adapter_authorized is not False
            or self.runner_source_sha256 != "sha256:" + source_sha256()
            or self.runner_algorithm_digest != algorithm_digest()
            or self.record_digest != "sha256:" + canonical_digest(self.content_data())
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "authorization authority or seal differs"
            )


def source_sha256() -> str:
    """Return the import-time source seal while on-disk bytes still match."""

    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _typed_equal(left: object, right: object) -> bool:
    """Compare canonical JSON bytes so bool/int/float substitutions never alias."""

    return canonical_json(_plain(left)) == canonical_json(_plain(right))


def _address(value: object, label: str) -> str:
    if type(value) is not str or _SHA_ADDRESS.fullmatch(value) is None:
        raise SkeletonGraphCalibrationRunnerError(
            f"{label} must be a lowercase sha256: address"
        )
    return value


def _exact_int(value: object, label: str, *, lower: int = 0) -> int:
    if type(value) is not int or value < lower:
        raise SkeletonGraphCalibrationRunnerError(f"{label} differs")
    return value


def _probability(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise SkeletonGraphCalibrationRunnerError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SkeletonGraphCalibrationRunnerError(f"{label} leaves [0,1]")
    return result


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise SkeletonGraphCalibrationRunnerError(f"{label} fields differ")
    return value


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = _plain(body)
    return {**value, "record_digest": "sha256:" + canonical_digest(value)}


def _file_address(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _authority_preflight() -> None:
    source_sha256()
    prereg_source = _stable_bytes(
        Path(prereg.__file__), label="calibration preregistration source", maximum=1 << 20
    )
    if (
        _file_address(prereg_source) != PINNED_PREREGISTRATION_SOURCE_SHA256
        or passed_fit_authority.source_sha256() != PINNED_PASSED_FIT_SOURCE_SHA256
        or passed_fit_authority.PASSED_FIT_ALGORITHM_DIGEST
        != PINNED_PASSED_FIT_ALGORITHM_DIGEST
        or inference_custody.source_sha256() != PINNED_INFERENCE_SOURCE_SHA256
        or inference_custody.algorithm_digest() != PINNED_INFERENCE_ALGORITHM_DIGEST
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "preregistration, passed-fit, or inference authority differs"
        )


def algorithm_digest() -> str:
    """Address the complete staged runner and all frozen dependencies."""

    _authority_preflight()
    return "sha256:" + canonical_digest(
        {
            "schema": "gkm.bongard-skeleton-graph-calibration-algorithm.v1",
            "runner_source_sha256": source_sha256(),
            "preregistration_record_digest": PINNED_PREREGISTRATION_RECORD_DIGEST,
            "preregistration_file_sha256": PINNED_PREREGISTRATION_FILE_SHA256,
            "preregistration_source_sha256": PINNED_PREREGISTRATION_SOURCE_SHA256,
            "preregistration_commit": PINNED_PREREGISTRATION_COMMIT,
            "passed_fit_commit": PINNED_PASSED_FIT_COMMIT,
            "passed_fit_source_sha256": PINNED_PASSED_FIT_SOURCE_SHA256,
            "passed_fit_algorithm_digest": PINNED_PASSED_FIT_ALGORITHM_DIGEST,
            "passed_fit_protocol_record_digest": (
                PINNED_PASSED_FIT_PROTOCOL_RECORD_DIGEST
            ),
            "inference_commit": PINNED_INFERENCE_COMMIT,
            "inference_source_sha256": PINNED_INFERENCE_SOURCE_SHA256,
            "inference_algorithm_digest": PINNED_INFERENCE_ALGORITHM_DIGEST,
            "task_score": "maximum_over_14_panels_and_pair_plus_catalog_heads",
            "pair_universe": list(prereg.VALID_PAIR_CODES),
            "observed_pair_order": list(prereg.OBSERVED_PAIR_CODES),
            "catalog_order": list(prereg.CATALOG_CLASS_ORDER),
            "barrier": "write_fsync_directory_fsync_fresh_reload_before_labels",
            "same_family_efficiency_gate": "exact_preregistration_v1",
            "reroll": "write_once_attempt_markers_fail_closed",
        }
    )


@dataclass(frozen=True, slots=True)
class SkeletonGraphPassedFitPaths:
    development_precommit_path: Path
    development_result_path: Path
    development_replay_path: Path
    model_path: Path
    feature_artifact_path: Path
    prediction_artifact_path: Path

    def keyword_arguments(self) -> dict[str, Path]:
        return {
            "development_precommit_path": Path(self.development_precommit_path),
            "development_result_path": Path(self.development_result_path),
            "development_replay_path": Path(self.development_replay_path),
            "model_path": Path(self.model_path),
            "feature_artifact_path": Path(self.feature_artifact_path),
            "prediction_artifact_path": Path(self.prediction_artifact_path),
        }


@dataclass(frozen=True, slots=True)
class SkeletonGraphCalibrationPanelIdentity:
    panel_id: str
    png_sha256: str
    png_size: int

    def __post_init__(self) -> None:
        if type(self.panel_id) is not str or _PANEL_ID.fullmatch(self.panel_id) is None:
            raise SkeletonGraphCalibrationRunnerError("calibration panel id differs")
        _address(self.png_sha256, "calibration PNG")
        _exact_int(self.png_size, "calibration PNG size", lower=1)

    @property
    def task_id(self) -> str:
        match = _PANEL_ID.fullmatch(self.panel_id)
        assert match is not None
        return match.group(1)

    @property
    def side(self) -> int:
        match = _PANEL_ID.fullmatch(self.panel_id)
        assert match is not None
        return int(match.group(2))

    @property
    def ordinal(self) -> int:
        match = _PANEL_ID.fullmatch(self.panel_id)
        assert match is not None
        return int(match.group(3))

    def to_data(self) -> dict[str, object]:
        return {
            "panel_id": self.panel_id,
            "png_sha256": self.png_sha256,
            "png_size": self.png_size,
        }

    @classmethod
    def from_data(cls, value: object) -> "SkeletonGraphCalibrationPanelIdentity":
        raw = _fields(value, {"panel_id", "png_sha256", "png_size"}, "panel identity")
        result = cls(raw["panel_id"], raw["png_sha256"], raw["png_size"])
        if not _typed_equal(result.to_data(), dict(raw)):
            raise SkeletonGraphCalibrationRunnerError("panel identity is not canonical")
        return result


def _probability_vector(
    values: Sequence[float], expected: int, label: str
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != expected:
        raise SkeletonGraphCalibrationRunnerError(f"{label} probability shape differs")
    result = tuple(_probability(item, label) for item in values)
    if not math.isclose(sum(result), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SkeletonGraphCalibrationRunnerError(f"{label} probabilities do not sum to one")
    return result


@dataclass(frozen=True, slots=True)
class SkeletonGraphDelayedLabelRow:
    anonymous_panel_token: str
    panel_id: str
    task_id: str
    side: int
    ordinal: int
    true_straight_action_count: int
    true_arc_action_count: int
    true_catalog_class: int
    error_code: None = None

    def __post_init__(self) -> None:
        if (
            type(self.anonymous_panel_token) is not str
            or _TOKEN.fullmatch(self.anonymous_panel_token) is None
        ):
            raise SkeletonGraphCalibrationRunnerError("anonymous label token differs")
        identity = SkeletonGraphCalibrationPanelIdentity(
            self.panel_id, "sha256:" + "0" * 64, 1
        )
        if (
            type(self.side) is not int
            or type(self.ordinal) is not int
            or self.task_id != identity.task_id
            or self.side != identity.side
            or self.ordinal != identity.ordinal
        ):
            raise SkeletonGraphCalibrationRunnerError("delayed label role join differs")
        for item, label in (
            (self.true_straight_action_count, "true straight count"),
            (self.true_arc_action_count, "true arc count"),
        ):
            if type(item) is not int or not 0 <= item <= 9:
                raise SkeletonGraphCalibrationRunnerError(f"{label} differs")
        if not 1 <= self.true_straight_action_count + self.true_arc_action_count <= 9:
            raise SkeletonGraphCalibrationRunnerError("true pair leaves the 54-class universe")
        if (
            type(self.true_catalog_class) is not int
            or self.true_catalog_class not in prereg.CATALOG_CLASS_ORDER
        ):
            raise SkeletonGraphCalibrationRunnerError("true catalog class differs")
        if self.error_code is not None:
            raise SkeletonGraphCalibrationRunnerError("delayed label row carries an error")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": LABEL_ROW_SCHEMA,
            "anonymous_panel_token": self.anonymous_panel_token,
            "panel_id": self.panel_id,
            "task_id": self.task_id,
            "side": self.side,
            "ordinal": self.ordinal,
            "true_straight_action_count": self.true_straight_action_count,
            "true_arc_action_count": self.true_arc_action_count,
            "true_catalog_class": self.true_catalog_class,
            "error_code": self.error_code,
        }

    @classmethod
    def from_data(cls, value: object) -> "SkeletonGraphDelayedLabelRow":
        raw = _fields(
            value,
            {
                "schema", "anonymous_panel_token", "panel_id", "task_id", "side",
                "ordinal", "true_straight_action_count", "true_arc_action_count",
                "true_catalog_class", "error_code",
            },
            "delayed label row",
        )
        if raw["schema"] != LABEL_ROW_SCHEMA:
            raise SkeletonGraphCalibrationRunnerError("delayed label row schema differs")
        result = cls(
            anonymous_panel_token=raw["anonymous_panel_token"],
            panel_id=raw["panel_id"],
            task_id=raw["task_id"],
            side=raw["side"],
            ordinal=raw["ordinal"],
            true_straight_action_count=raw["true_straight_action_count"],
            true_arc_action_count=raw["true_arc_action_count"],
            true_catalog_class=raw["true_catalog_class"],
            error_code=raw["error_code"],
        )
        if not _typed_equal(result.to_data(), dict(raw)):
            raise SkeletonGraphCalibrationRunnerError("delayed label row is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class SkeletonGraphDelayedLabelBatch:
    delayed_label_request_record_digest: str
    label_attempt_record_digest: str
    prediction_record_digest: str
    prediction_file_sha256: str
    action_program_authority_record_digest: str
    action_program_authority_file_sha256: str
    catalog_authority_source_sha256: str
    catalog_algorithm_digest: str
    label_extraction_algorithm_digest: str
    rows: tuple[SkeletonGraphDelayedLabelRow, ...]
    record_digest: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.delayed_label_request_record_digest, "delayed label request record"),
            (self.label_attempt_record_digest, "label attempt record"),
            (self.prediction_record_digest, "prediction record"),
            (self.prediction_file_sha256, "prediction file"),
            (self.action_program_authority_record_digest, "action-program record"),
            (self.action_program_authority_file_sha256, "action-program file"),
            (self.catalog_authority_source_sha256, "catalog authority source"),
            (self.catalog_algorithm_digest, "catalog algorithm"),
            (self.label_extraction_algorithm_digest, "label extraction algorithm"),
        ):
            _address(value, label)
        if (
            type(self.rows) is not tuple
            or not self.rows
            or any(type(row) is not SkeletonGraphDelayedLabelRow for row in self.rows)
            or len({row.anonymous_panel_token for row in self.rows}) != len(self.rows)
        ):
            raise SkeletonGraphCalibrationRunnerError("delayed label rows differ")
        if self.record_digest != "sha256:" + canonical_digest(self.content_data()):
            raise SkeletonGraphCalibrationRunnerError("delayed label batch digest differs")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": LABEL_BATCH_SCHEMA,
            "delayed_label_request_record_digest": (
                self.delayed_label_request_record_digest
            ),
            "label_attempt_record_digest": self.label_attempt_record_digest,
            "prediction_record_digest": self.prediction_record_digest,
            "prediction_file_sha256": self.prediction_file_sha256,
            "action_program_authority_record_digest": (
                self.action_program_authority_record_digest
            ),
            "action_program_authority_file_sha256": (
                self.action_program_authority_file_sha256
            ),
            "catalog_authority_source_sha256": self.catalog_authority_source_sha256,
            "catalog_algorithm_digest": self.catalog_algorithm_digest,
            "label_extraction_algorithm_digest": self.label_extraction_algorithm_digest,
            "rows": [row.to_data() for row in self.rows],
            "label_authority_opened_only_after_prediction_fresh_reload": True,
            "sealed_one_shot_request_consumed_before_label_derivation": True,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def create(
        cls,
        *,
        delayed_label_request_record_digest: str,
        label_attempt_record_digest: str,
        prediction_record_digest: str,
        prediction_file_sha256: str,
        action_program_authority_record_digest: str,
        action_program_authority_file_sha256: str,
        catalog_authority_source_sha256: str,
        catalog_algorithm_digest: str,
        label_extraction_algorithm_digest: str,
        rows: Sequence[SkeletonGraphDelayedLabelRow],
    ) -> "SkeletonGraphDelayedLabelBatch":
        values = {
            "delayed_label_request_record_digest": delayed_label_request_record_digest,
            "label_attempt_record_digest": label_attempt_record_digest,
            "prediction_record_digest": prediction_record_digest,
            "prediction_file_sha256": prediction_file_sha256,
            "action_program_authority_record_digest": action_program_authority_record_digest,
            "action_program_authority_file_sha256": action_program_authority_file_sha256,
            "catalog_authority_source_sha256": catalog_authority_source_sha256,
            "catalog_algorithm_digest": catalog_algorithm_digest,
            "label_extraction_algorithm_digest": label_extraction_algorithm_digest,
            "rows": tuple(rows),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        digest = "sha256:" + canonical_digest(provisional.content_data())
        return cls(**values, record_digest=digest)

    @classmethod
    def from_data(cls, value: object) -> "SkeletonGraphDelayedLabelBatch":
        expected = {
            "schema", "delayed_label_request_record_digest",
            "label_attempt_record_digest", "prediction_record_digest",
            "prediction_file_sha256",
            "action_program_authority_record_digest",
            "action_program_authority_file_sha256", "catalog_authority_source_sha256",
            "catalog_algorithm_digest", "label_extraction_algorithm_digest", "rows",
            "label_authority_opened_only_after_prediction_fresh_reload",
            "sealed_one_shot_request_consumed_before_label_derivation", "record_digest",
        }
        raw = _fields(value, expected, "delayed label batch")
        if (
            raw["schema"] != LABEL_BATCH_SCHEMA
            or raw["label_authority_opened_only_after_prediction_fresh_reload"] is not True
            or raw["sealed_one_shot_request_consumed_before_label_derivation"] is not True
            or type(raw["rows"]) is not list
        ):
            raise SkeletonGraphCalibrationRunnerError("delayed label policy differs")
        result = cls(
            delayed_label_request_record_digest=raw[
                "delayed_label_request_record_digest"
            ],
            label_attempt_record_digest=raw["label_attempt_record_digest"],
            prediction_record_digest=raw["prediction_record_digest"],
            prediction_file_sha256=raw["prediction_file_sha256"],
            action_program_authority_record_digest=raw[
                "action_program_authority_record_digest"
            ],
            action_program_authority_file_sha256=raw[
                "action_program_authority_file_sha256"
            ],
            catalog_authority_source_sha256=raw["catalog_authority_source_sha256"],
            catalog_algorithm_digest=raw["catalog_algorithm_digest"],
            label_extraction_algorithm_digest=raw["label_extraction_algorithm_digest"],
            rows=tuple(SkeletonGraphDelayedLabelRow.from_data(row) for row in raw["rows"]),
            record_digest=raw["record_digest"],
        )
        if not _typed_equal(result.to_data(), dict(raw)):
            raise SkeletonGraphCalibrationRunnerError("delayed label batch is not canonical")
        return result


_GRANT_ISSUANCE_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class SkeletonGraphPopulationGrant:
    """Exact conformal population grant; never an authority to open pixels."""

    scope: SkeletonGraphCalibrationScope
    population_scope: str
    alpha: float
    calibration_task_count: int
    calibration_panel_count: int
    order_statistic_k: int
    q: float
    calibration_task_ids: tuple[str, ...]
    task_scores: tuple[float, ...]
    observed_pair_class_order: tuple[int, ...]
    valid_pair_class_order: tuple[int, ...]
    catalog_class_order: tuple[int, ...]
    target_population_authorized: bool
    authorized_target_task_id: str | None
    external_population_membership_required: bool
    target_release_authorization_required: bool
    target_pixel_authorized: bool
    authenticated_calibration_execution: bool
    production_adapter_authorized: bool
    efficiency_gate_passed: bool | None
    efficiency_gate: Mapping[str, Any]
    preregistration_record_digest: str
    passed_fit_protocol_record_digest: str
    prediction_record_digest: str
    label_record_digest: str
    raw_batch_file_sha256: str
    raw_batch_record_digest: str
    recompute_receipt_file_sha256: str
    recompute_receipt_record_digest: str
    occurrence_join_digest: str
    runner_source_sha256: str
    runner_algorithm_digest: str
    record_digest: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SkeletonGraphCalibrationRunnerError(
            "population grants are issued only by a verified calibration archive"
        )

    def __post_init__(self) -> None:
        if type(self.scope) is not SkeletonGraphCalibrationScope:
            raise SkeletonGraphCalibrationRunnerError("grant scope differs")
        if type(self.population_scope) is not str or not self.population_scope:
            raise SkeletonGraphCalibrationRunnerError("population scope differs")
        expected = (
            (0.05, 100, 1_400, 96)
            if self.scope is SkeletonGraphCalibrationScope.GENERIC_V3
            else (0.10, 16, 224, 16)
        )
        if (
            type(self.alpha) is not float
            or type(self.q) is not float
            or type(self.calibration_task_count) is not int
            or type(self.calibration_panel_count) is not int
            or type(self.order_statistic_k) is not int
            or self.alpha != expected[0]
            or self.calibration_task_count != expected[1]
            or self.calibration_panel_count != expected[2]
            or self.order_statistic_k != expected[3]
            or type(self.calibration_task_ids) is not tuple
            or len(self.calibration_task_ids) != self.calibration_task_count
            or len(set(self.calibration_task_ids)) != self.calibration_task_count
            or type(self.task_scores) is not tuple
            or len(self.task_scores) != self.calibration_task_count
        ):
            raise SkeletonGraphCalibrationRunnerError("grant campaign shape differs")
        expected_population = (
            "fresh_generic_HD_TRAIN_known-carrier_style-pose_population"
            if self.scope is SkeletonGraphCalibrationScope.GENERIC_V3
            else "convex-four-lines_TRAIN_repetitions_0002_through_0017"
        )
        expected_task_digest = (
            "sha256:01ae13699706ff67f524241fe257224a6f9136b80d2f8b857e2b98ff758f82c9"
            if self.scope is SkeletonGraphCalibrationScope.GENERIC_V3
            else "sha256:22961eddab8c5ce0289751f600d336ad321a60dc08a5bfe0d97d4bd0958b3a91"
        )
        if (
            self.population_scope != expected_population
            or "sha256:" + canonical_digest(self.calibration_task_ids)
            != expected_task_digest
            or (
                self.scope is SkeletonGraphCalibrationScope.SAME_FAMILY
                and self.calibration_task_ids != tuple(prereg.SAME_FAMILY_TASK_IDS)
            )
        ):
            raise SkeletonGraphCalibrationRunnerError("grant population identity differs")
        _probability(self.q, "conformal q")
        if any(type(item) is not float for item in self.task_scores):
            raise SkeletonGraphCalibrationRunnerError(
                "grant task-score wire types differ"
            )
        for item in self.task_scores:
            _probability(item, "whole-task score")
        if (
            any(type(item) is not int for item in self.observed_pair_class_order)
            or any(type(item) is not int for item in self.valid_pair_class_order)
            or any(type(item) is not int for item in self.catalog_class_order)
            or
            self.observed_pair_class_order != tuple(prereg.OBSERVED_PAIR_CODES)
            or self.valid_pair_class_order != tuple(prereg.VALID_PAIR_CODES)
            or self.catalog_class_order != tuple(prereg.CATALOG_CLASS_ORDER)
        ):
            raise SkeletonGraphCalibrationRunnerError("grant class universe differs")
        expected_q = sorted(self.task_scores)[self.order_statistic_k - 1]
        if self.q != expected_q:
            raise SkeletonGraphCalibrationRunnerError("grant conformal quantile differs")
        if self.scope is SkeletonGraphCalibrationScope.GENERIC_V3:
            if (
                self.target_population_authorized is not False
                or self.authorized_target_task_id is not None
                or self.efficiency_gate_passed is not None
            ):
                raise SkeletonGraphCalibrationRunnerError(
                    "generic grant cannot carry target authority"
                )
        elif (
            self.target_population_authorized is not True
            or self.authorized_target_task_id != SAME_FAMILY_TARGET_TASK_ID
            or self.efficiency_gate_passed is not True
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "same-family grant requires every fixed efficiency gate"
            )
        if self.scope is SkeletonGraphCalibrationScope.SAME_FAMILY:
            admitted = self.efficiency_gate.get("formula_admitted_task_count")
            errors = self.efficiency_gate.get("formula_or_cell_error_count")
            if (
                type(admitted) is not int
                or type(errors) is not int
                or self.efficiency_gate.get("all_fixed_checks_passed") is not True
                or admitted < 14
                or errors != 0
            ):
                raise SkeletonGraphCalibrationRunnerError(
                    "same-family fixed efficiency evidence differs"
                )
        if (
            self.external_population_membership_required is not True
            or self.target_release_authorization_required is not True
            or self.target_pixel_authorized is not False
            or self.authenticated_calibration_execution is not False
            or self.production_adapter_authorized is not False
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "grant external-membership/release/pixel seal differs"
            )
        if not isinstance(self.efficiency_gate, Mapping):
            raise SkeletonGraphCalibrationRunnerError("grant efficiency record differs")
        for value, label in (
            (self.preregistration_record_digest, "preregistration record"),
            (self.passed_fit_protocol_record_digest, "passed-fit protocol"),
            (self.prediction_record_digest, "prediction record"),
            (self.label_record_digest, "label record"),
            (self.raw_batch_file_sha256, "raw batch file"),
            (self.raw_batch_record_digest, "raw batch record"),
            (self.recompute_receipt_file_sha256, "recompute receipt file"),
            (self.recompute_receipt_record_digest, "recompute receipt record"),
            (self.occurrence_join_digest, "occurrence join"),
            (self.runner_source_sha256, "runner source"),
            (self.runner_algorithm_digest, "runner algorithm"),
        ):
            _address(value, label)
        if (
            self.preregistration_record_digest
            != PINNED_PREREGISTRATION_RECORD_DIGEST
            or self.passed_fit_protocol_record_digest
            != PINNED_PASSED_FIT_PROTOCOL_RECORD_DIGEST
            or self.runner_source_sha256 != "sha256:" + source_sha256()
            or self.runner_algorithm_digest != algorithm_digest()
            or self.record_digest != "sha256:" + canonical_digest(self.content_data())
        ):
            raise SkeletonGraphCalibrationRunnerError("grant authority or digest differs")

    @property
    def schema(self) -> str:
        return (
            GENERIC_GRANT_SCHEMA
            if self.scope is SkeletonGraphCalibrationScope.GENERIC_V3
            else SAME_FAMILY_GRANT_SCHEMA
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "scope": self.scope.value,
            "population_scope": self.population_scope,
            "alpha": self.alpha,
            "calibration_task_count": self.calibration_task_count,
            "calibration_panel_count": self.calibration_panel_count,
            "order_statistic_k": self.order_statistic_k,
            "q": self.q,
            "calibration_task_ids": list(self.calibration_task_ids),
            "task_scores": list(self.task_scores),
            "observed_pair_class_order": list(self.observed_pair_class_order),
            "valid_pair_class_order": list(self.valid_pair_class_order),
            "catalog_class_order": list(self.catalog_class_order),
            "target_population_authorized": self.target_population_authorized,
            "authorized_target_task_id": self.authorized_target_task_id,
            "external_population_membership_required": (
                self.external_population_membership_required
            ),
            "target_release_authorization_required": (
                self.target_release_authorization_required
            ),
            "target_pixel_authorized": self.target_pixel_authorized,
            "authenticated_calibration_execution": (
                self.authenticated_calibration_execution
            ),
            "production_adapter_authorized": self.production_adapter_authorized,
            "efficiency_gate_passed": self.efficiency_gate_passed,
            "efficiency_gate": _plain(self.efficiency_gate),
            "preregistration_record_digest": self.preregistration_record_digest,
            "passed_fit_protocol_record_digest": (
                self.passed_fit_protocol_record_digest
            ),
            "prediction_record_digest": self.prediction_record_digest,
            "label_record_digest": self.label_record_digest,
            "raw_batch_file_sha256": self.raw_batch_file_sha256,
            "raw_batch_record_digest": self.raw_batch_record_digest,
            "recompute_receipt_file_sha256": self.recompute_receipt_file_sha256,
            "recompute_receipt_record_digest": self.recompute_receipt_record_digest,
            "occurrence_join_digest": self.occurrence_join_digest,
            "runner_source_sha256": self.runner_source_sha256,
            "runner_algorithm_digest": self.runner_algorithm_digest,
            "full_54_pair_conformal_universe": True,
            "missing_observed_pair_probability": 0.0,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def _issue_after_verified_archive(
        cls,
        *,
        scope: SkeletonGraphCalibrationScope,
        population_scope: str,
        q: float,
        calibration_task_ids: Sequence[str],
        task_scores: Sequence[float],
        efficiency_gate: Mapping[str, Any],
        prediction_record_digest: str,
        label_record_digest: str,
        frozen_inference: "SkeletonGraphFrozenInferenceAddresses",
        occurrence_join_digest: str,
        issuance_token: object,
    ) -> "SkeletonGraphPopulationGrant":
        if issuance_token is not _GRANT_ISSUANCE_TOKEN:
            raise SkeletonGraphCalibrationRunnerError(
                "population grant issuance requires the verified archive path"
            )
        generic = scope is SkeletonGraphCalibrationScope.GENERIC_V3
        values: dict[str, Any] = {
            "scope": scope,
            "population_scope": population_scope,
            "alpha": 0.05 if generic else 0.10,
            "calibration_task_count": 100 if generic else 16,
            "calibration_panel_count": 1_400 if generic else 224,
            "order_statistic_k": 96 if generic else 16,
            "q": q,
            "calibration_task_ids": tuple(calibration_task_ids),
            "task_scores": tuple(float(item) for item in task_scores),
            "observed_pair_class_order": tuple(prereg.OBSERVED_PAIR_CODES),
            "valid_pair_class_order": tuple(prereg.VALID_PAIR_CODES),
            "catalog_class_order": tuple(prereg.CATALOG_CLASS_ORDER),
            "target_population_authorized": not generic,
            "authorized_target_task_id": None if generic else SAME_FAMILY_TARGET_TASK_ID,
            "external_population_membership_required": True,
            "target_release_authorization_required": True,
            "target_pixel_authorized": False,
            "authenticated_calibration_execution": False,
            "production_adapter_authorized": False,
            "efficiency_gate_passed": None if generic else True,
            "efficiency_gate": MappingProxyType(dict(_plain(efficiency_gate))),
            "preregistration_record_digest": PINNED_PREREGISTRATION_RECORD_DIGEST,
            "passed_fit_protocol_record_digest": (
                PINNED_PASSED_FIT_PROTOCOL_RECORD_DIGEST
            ),
            "prediction_record_digest": prediction_record_digest,
            "label_record_digest": label_record_digest,
            "raw_batch_file_sha256": frozen_inference.raw_batch_file_sha256,
            "raw_batch_record_digest": frozen_inference.raw_batch_record_digest,
            "recompute_receipt_file_sha256": (
                frozen_inference.recompute_receipt_file_sha256
            ),
            "recompute_receipt_record_digest": (
                frozen_inference.recompute_receipt_record_digest
            ),
            "occurrence_join_digest": occurrence_join_digest,
            "runner_source_sha256": "sha256:" + source_sha256(),
            "runner_algorithm_digest": algorithm_digest(),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        object.__setattr__(
            provisional,
            "record_digest",
            "sha256:" + canonical_digest(provisional.content_data()),
        )
        provisional.__post_init__()
        return provisional

    @classmethod
    def from_data(cls, value: object) -> "SkeletonGraphPopulationGrant":
        expected = {
            "schema", "scope", "population_scope", "alpha",
            "calibration_task_count", "calibration_panel_count", "order_statistic_k",
            "q", "calibration_task_ids", "task_scores", "observed_pair_class_order",
            "valid_pair_class_order", "catalog_class_order",
            "target_population_authorized", "authorized_target_task_id",
            "external_population_membership_required",
            "target_release_authorization_required", "target_pixel_authorized",
            "authenticated_calibration_execution", "production_adapter_authorized",
            "efficiency_gate_passed", "efficiency_gate",
            "preregistration_record_digest", "passed_fit_protocol_record_digest",
            "prediction_record_digest", "label_record_digest", "runner_source_sha256",
            "raw_batch_file_sha256", "raw_batch_record_digest",
            "recompute_receipt_file_sha256", "recompute_receipt_record_digest",
            "occurrence_join_digest",
            "runner_algorithm_digest", "full_54_pair_conformal_universe",
            "missing_observed_pair_probability", "record_digest",
        }
        raw = _fields(value, expected, "population grant")
        try:
            scope = SkeletonGraphCalibrationScope(raw["scope"])
        except (TypeError, ValueError) as exc:
            raise SkeletonGraphCalibrationRunnerError("grant scope differs") from exc
        expected_schema = (
            GENERIC_GRANT_SCHEMA
            if scope is SkeletonGraphCalibrationScope.GENERIC_V3
            else SAME_FAMILY_GRANT_SCHEMA
        )
        if (
            raw["schema"] != expected_schema
            or raw["full_54_pair_conformal_universe"] is not True
            or type(raw["missing_observed_pair_probability"]) is not float
            or raw["missing_observed_pair_probability"] != 0.0
            or type(raw["calibration_task_ids"]) is not list
            or type(raw["task_scores"]) is not list
            or type(raw["observed_pair_class_order"]) is not list
            or type(raw["valid_pair_class_order"]) is not list
            or type(raw["catalog_class_order"]) is not list
            or not isinstance(raw["efficiency_gate"], Mapping)
        ):
            raise SkeletonGraphCalibrationRunnerError("grant wire policy differs")
        result = object.__new__(cls)
        values = {
            "scope": scope,
            "population_scope": raw["population_scope"],
            "alpha": raw["alpha"],
            "calibration_task_count": raw["calibration_task_count"],
            "calibration_panel_count": raw["calibration_panel_count"],
            "order_statistic_k": raw["order_statistic_k"],
            "q": raw["q"],
            "calibration_task_ids": tuple(raw["calibration_task_ids"]),
            "task_scores": tuple(raw["task_scores"]),
            "observed_pair_class_order": tuple(raw["observed_pair_class_order"]),
            "valid_pair_class_order": tuple(raw["valid_pair_class_order"]),
            "catalog_class_order": tuple(raw["catalog_class_order"]),
            "target_population_authorized": raw["target_population_authorized"],
            "authorized_target_task_id": raw["authorized_target_task_id"],
            "external_population_membership_required": raw[
                "external_population_membership_required"
            ],
            "target_release_authorization_required": raw[
                "target_release_authorization_required"
            ],
            "target_pixel_authorized": raw["target_pixel_authorized"],
            "authenticated_calibration_execution": raw[
                "authenticated_calibration_execution"
            ],
            "production_adapter_authorized": raw["production_adapter_authorized"],
            "efficiency_gate_passed": raw["efficiency_gate_passed"],
            "efficiency_gate": MappingProxyType(dict(raw["efficiency_gate"])),
            "preregistration_record_digest": raw["preregistration_record_digest"],
            "passed_fit_protocol_record_digest": raw[
                "passed_fit_protocol_record_digest"
            ],
            "prediction_record_digest": raw["prediction_record_digest"],
            "label_record_digest": raw["label_record_digest"],
            "raw_batch_file_sha256": raw["raw_batch_file_sha256"],
            "raw_batch_record_digest": raw["raw_batch_record_digest"],
            "recompute_receipt_file_sha256": raw[
                "recompute_receipt_file_sha256"
            ],
            "recompute_receipt_record_digest": raw[
                "recompute_receipt_record_digest"
            ],
            "occurrence_join_digest": raw["occurrence_join_digest"],
            "runner_source_sha256": raw["runner_source_sha256"],
            "runner_algorithm_digest": raw["runner_algorithm_digest"],
            "record_digest": raw["record_digest"],
        }
        for name, item in values.items():
            object.__setattr__(result, name, item)
        result.__post_init__()
        if not _typed_equal(result.to_data(), dict(raw)):
            raise SkeletonGraphCalibrationRunnerError("population grant is not canonical")
        return result

    def direct_pair_class_set(self, probabilities_33: Sequence[float]) -> tuple[int, ...]:
        probabilities = _probability_vector(probabilities_33, 33, "direct pair")
        observed = dict(zip(self.observed_pair_class_order, probabilities, strict=True))
        return tuple(
            code
            for code in self.valid_pair_class_order
            if 1.0 - observed.get(code, 0.0) <= self.q
        )

    def catalog_class_set(self, probabilities_3: Sequence[float]) -> tuple[int, ...]:
        probabilities = _probability_vector(probabilities_3, 3, "catalog")
        return tuple(
            code
            for code, probability in zip(
                self.catalog_class_order, probabilities, strict=True
            )
            if 1.0 - probability <= self.q
        )

    def authorizes_task(self, task_id: str) -> bool:
        """Return exact population-scope membership, never pixel permission."""

        if type(task_id) is not str:
            return False
        return task_id in self.calibration_task_ids or (
            self.scope is SkeletonGraphCalibrationScope.SAME_FAMILY
            and self.target_population_authorized
            and task_id == self.authorized_target_task_id
        )

    def authorizes_target_scope(self, task_id: str) -> bool:
        return (
            self.scope is SkeletonGraphCalibrationScope.SAME_FAMILY
            and self.target_population_authorized
            and task_id == self.authorized_target_task_id
        )


def verify_skeleton_graph_population_grant(
    grant: SkeletonGraphPopulationGrant,
    *,
    replay_receipt: "SkeletonGraphCalibrationReplayReceipt",
) -> SkeletonGraphPopulationGrant:
    if type(grant) is not SkeletonGraphPopulationGrant:
        raise TypeError("population grant must have exact type")
    if type(replay_receipt) is not SkeletonGraphCalibrationReplayReceipt:
        raise TypeError("population grant needs an exact cold-replay receipt")
    restored = SkeletonGraphPopulationGrant.from_data(grant.to_data())
    if (
        not _typed_equal(restored.to_data(), grant.to_data())
        or not replay_receipt.verifies(grant)
    ):
        raise SkeletonGraphCalibrationRunnerError("population grant replay differs")
    return grant


@dataclass(frozen=True, slots=True)
class SkeletonGraphCalibrationGap:
    scope: SkeletonGraphCalibrationScope
    stage: str
    reason_codes: tuple[str, ...]
    target_population_authorized: bool
    target_pixel_authorized: bool
    preregistration_record_digest: str
    passed_fit_record_digest: str
    prediction_record_digest: str | None
    label_record_digest: str | None
    integrity_custody: Mapping[str, Any] | None
    runner_source_sha256: str
    runner_algorithm_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.scope) is not SkeletonGraphCalibrationScope
            or type(self.stage) is not str
            or not self.stage
            or type(self.reason_codes) is not tuple
            or not self.reason_codes
            or len(set(self.reason_codes)) != len(self.reason_codes)
            or any(type(item) is not str or not item for item in self.reason_codes)
            or self.target_population_authorized is not False
            or self.target_pixel_authorized is not False
        ):
            raise SkeletonGraphCalibrationRunnerError("calibration GAP policy differs")
        for value, label in (
            (self.preregistration_record_digest, "GAP preregistration"),
            (self.passed_fit_record_digest, "GAP passed fit"),
            (self.runner_source_sha256, "GAP runner source"),
            (self.runner_algorithm_digest, "GAP runner algorithm"),
        ):
            _address(value, label)
        for value, label in (
            (self.prediction_record_digest, "GAP prediction"),
            (self.label_record_digest, "GAP label"),
        ):
            if value is not None:
                _address(value, label)
        integrity_stage = self.stage.removeprefix("integrity_")
        if self.stage.startswith("integrity_"):
            if (
                integrity_stage not in _INTEGRITY_FAILURE_STAGES
                or self.reason_codes
                != (
                    "execution_integrity_failure",
                    integrity_stage + "_failed",
                )
                or not isinstance(self.integrity_custody, Mapping)
                or set(self.integrity_custody)
                != {
                    "failure_stage",
                    "exposure_authorization",
                    "exposure_authorization_file_sha256",
                    "output_root_claim_record_digest",
                    "output_root_claim_file_sha256",
                    "inventory",
                }
                or self.integrity_custody.get("failure_stage") != integrity_stage
                or not isinstance(
                    self.integrity_custody.get("exposure_authorization"), Mapping
                )
                or not isinstance(self.integrity_custody.get("inventory"), Mapping)
            ):
                raise SkeletonGraphCalibrationRunnerError(
                    "integrity GAP custody differs"
                )
            _address(
                self.integrity_custody.get("exposure_authorization_file_sha256"),
                "integrity GAP exposure authorization file",
            )
            _address(
                self.integrity_custody.get("output_root_claim_record_digest"),
                "integrity GAP output claim record",
            )
            _address(
                self.integrity_custody.get("output_root_claim_file_sha256"),
                "integrity GAP output claim file",
            )
        elif self.integrity_custody is not None:
            raise SkeletonGraphCalibrationRunnerError(
                "non-integrity GAP carries integrity custody"
            )
        if (
            self.preregistration_record_digest
            != PINNED_PREREGISTRATION_RECORD_DIGEST
            or self.runner_source_sha256 != "sha256:" + source_sha256()
            or self.runner_algorithm_digest != algorithm_digest()
            or self.record_digest != "sha256:" + canonical_digest(self.content_data())
        ):
            raise SkeletonGraphCalibrationRunnerError("calibration GAP digest differs")

    def content_data(self) -> dict[str, object]:
        return {
            "schema": GAP_SCHEMA,
            "scope": self.scope.value,
            "stage": self.stage,
            "disposition": "gap",
            "reason_codes": list(self.reason_codes),
            "target_population_authorized": False,
            "target_pixel_authorized": False,
            "preregistration_record_digest": self.preregistration_record_digest,
            "passed_fit_record_digest": self.passed_fit_record_digest,
            "prediction_record_digest": self.prediction_record_digest,
            "label_record_digest": self.label_record_digest,
            "integrity_custody": (
                None
                if self.integrity_custody is None
                else _plain(self.integrity_custody)
            ),
            "runner_source_sha256": self.runner_source_sha256,
            "runner_algorithm_digest": self.runner_algorithm_digest,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def create(
        cls,
        *,
        scope: SkeletonGraphCalibrationScope,
        stage: str,
        reason_codes: Sequence[str],
        passed_fit_record_digest: str,
        prediction_record_digest: str | None = None,
        label_record_digest: str | None = None,
        integrity_custody: Mapping[str, Any] | None = None,
    ) -> "SkeletonGraphCalibrationGap":
        values = {
            "scope": scope,
            "stage": stage,
            "reason_codes": tuple(reason_codes),
            "target_population_authorized": False,
            "target_pixel_authorized": False,
            "preregistration_record_digest": PINNED_PREREGISTRATION_RECORD_DIGEST,
            "passed_fit_record_digest": passed_fit_record_digest,
            "prediction_record_digest": prediction_record_digest,
            "label_record_digest": label_record_digest,
            "integrity_custody": (
                None
                if integrity_custody is None
                else MappingProxyType(_plain(integrity_custody))
            ),
            "runner_source_sha256": "sha256:" + source_sha256(),
            "runner_algorithm_digest": algorithm_digest(),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        digest = "sha256:" + canonical_digest(provisional.content_data())
        return cls(**values, record_digest=digest)

    @classmethod
    def from_data(cls, value: object) -> "SkeletonGraphCalibrationGap":
        expected = {
            "schema", "scope", "stage", "disposition", "reason_codes",
            "target_population_authorized", "target_pixel_authorized",
            "preregistration_record_digest", "passed_fit_record_digest",
            "prediction_record_digest", "label_record_digest", "runner_source_sha256",
            "integrity_custody", "runner_algorithm_digest", "record_digest",
        }
        raw = _fields(value, expected, "calibration GAP")
        if (
            raw["schema"] != GAP_SCHEMA
            or raw["disposition"] != "gap"
            or type(raw["reason_codes"]) is not list
        ):
            raise SkeletonGraphCalibrationRunnerError("calibration GAP wire differs")
        result = cls(
            scope=SkeletonGraphCalibrationScope(raw["scope"]),
            stage=raw["stage"],
            reason_codes=tuple(raw["reason_codes"]),
            target_population_authorized=raw["target_population_authorized"],
            target_pixel_authorized=raw["target_pixel_authorized"],
            preregistration_record_digest=raw["preregistration_record_digest"],
            passed_fit_record_digest=raw["passed_fit_record_digest"],
            prediction_record_digest=raw["prediction_record_digest"],
            label_record_digest=raw["label_record_digest"],
            integrity_custody=(
                None
                if raw["integrity_custody"] is None
                else MappingProxyType(
                    dict(
                        _fields(
                            raw["integrity_custody"],
                            {
                                "failure_stage",
                                "exposure_authorization",
                                "exposure_authorization_file_sha256",
                                "output_root_claim_record_digest",
                                "output_root_claim_file_sha256",
                                "inventory",
                            },
                            "integrity GAP custody",
                        )
                    )
                )
            ),
            runner_source_sha256=raw["runner_source_sha256"],
            runner_algorithm_digest=raw["runner_algorithm_digest"],
            record_digest=raw["record_digest"],
        )
        if not _typed_equal(result.to_data(), dict(raw)):
            raise SkeletonGraphCalibrationRunnerError("calibration GAP is not canonical")
        return result


SkeletonGraphCalibrationOutcome: TypeAlias = (
    SkeletonGraphPopulationGrant | SkeletonGraphCalibrationGap
)


_REPLAY_RECEIPT_ISSUANCE_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class SkeletonGraphCalibrationReplayReceipt:
    """In-memory capability issued only by exact zero-call archive replay."""

    grant_record_digest: str
    scope: SkeletonGraphCalibrationScope
    q: float
    efficiency_gate_digest: str
    preregistration_record_digest: str
    passed_fit_protocol_record_digest: str
    prediction_record_digest: str
    label_record_digest: str
    raw_batch_file_sha256: str
    raw_batch_record_digest: str
    recompute_receipt_file_sha256: str
    recompute_receipt_record_digest: str
    occurrence_join_digest: str
    archive_custody: Mapping[str, str]
    pixel_reads: int
    feature_extraction_calls: int
    model_prediction_api_calls: int
    estimator_predict_proba_calls: int
    label_authority_reads: int
    exact_replay: bool
    authenticated_calibration_execution: bool
    production_adapter_authorized: bool
    runner_source_sha256: str
    runner_algorithm_digest: str
    record_digest: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SkeletonGraphCalibrationRunnerError(
            "replay receipts are issued only by cold_replay_calibration"
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": REPLAY_SCHEMA,
            "grant_record_digest": self.grant_record_digest,
            "scope": self.scope.value,
            "q": self.q,
            "efficiency_gate_digest": self.efficiency_gate_digest,
            "preregistration_record_digest": self.preregistration_record_digest,
            "passed_fit_protocol_record_digest": (
                self.passed_fit_protocol_record_digest
            ),
            "prediction_record_digest": self.prediction_record_digest,
            "label_record_digest": self.label_record_digest,
            "raw_batch_file_sha256": self.raw_batch_file_sha256,
            "raw_batch_record_digest": self.raw_batch_record_digest,
            "recompute_receipt_file_sha256": self.recompute_receipt_file_sha256,
            "recompute_receipt_record_digest": self.recompute_receipt_record_digest,
            "occurrence_join_digest": self.occurrence_join_digest,
            "archive_custody": _plain(self.archive_custody),
            "pixel_reads": self.pixel_reads,
            "feature_extraction_calls": self.feature_extraction_calls,
            "model_prediction_api_calls": self.model_prediction_api_calls,
            "estimator_predict_proba_calls": self.estimator_predict_proba_calls,
            "label_authority_reads": self.label_authority_reads,
            "exact_replay": self.exact_replay,
            "authenticated_calibration_execution": (
                self.authenticated_calibration_execution
            ),
            "production_adapter_authorized": self.production_adapter_authorized,
            "runner_source_sha256": self.runner_source_sha256,
            "runner_algorithm_digest": self.runner_algorithm_digest,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def _issue(
        cls,
        *,
        grant: SkeletonGraphPopulationGrant,
        archive_custody: Mapping[str, str],
        issuance_token: object,
    ) -> "SkeletonGraphCalibrationReplayReceipt":
        if issuance_token is not _REPLAY_RECEIPT_ISSUANCE_TOKEN:
            raise SkeletonGraphCalibrationRunnerError(
                "replay receipt issuance requires exact cold replay"
            )
        values: dict[str, Any] = {
            "grant_record_digest": grant.record_digest,
            "scope": grant.scope,
            "q": grant.q,
            "efficiency_gate_digest": "sha256:"
            + canonical_digest(_plain(grant.efficiency_gate)),
            "preregistration_record_digest": grant.preregistration_record_digest,
            "passed_fit_protocol_record_digest": (
                grant.passed_fit_protocol_record_digest
            ),
            "prediction_record_digest": grant.prediction_record_digest,
            "label_record_digest": grant.label_record_digest,
            "raw_batch_file_sha256": grant.raw_batch_file_sha256,
            "raw_batch_record_digest": grant.raw_batch_record_digest,
            "recompute_receipt_file_sha256": grant.recompute_receipt_file_sha256,
            "recompute_receipt_record_digest": grant.recompute_receipt_record_digest,
            "occurrence_join_digest": grant.occurrence_join_digest,
            "archive_custody": MappingProxyType(dict(archive_custody)),
            "pixel_reads": 0,
            "feature_extraction_calls": 0,
            "model_prediction_api_calls": 0,
            "estimator_predict_proba_calls": 0,
            "label_authority_reads": 0,
            "exact_replay": True,
            "authenticated_calibration_execution": False,
            "production_adapter_authorized": False,
            "runner_source_sha256": "sha256:" + source_sha256(),
            "runner_algorithm_digest": algorithm_digest(),
        }
        result = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(result, name, item)
        object.__setattr__(
            result, "record_digest", "sha256:" + canonical_digest(result.content_data())
        )
        return result

    def verifies(self, grant: SkeletonGraphPopulationGrant) -> bool:
        return (
            type(grant) is SkeletonGraphPopulationGrant
            and self.grant_record_digest == grant.record_digest
            and self.scope is grant.scope
            and self.q == grant.q
            and self.efficiency_gate_digest
            == "sha256:" + canonical_digest(_plain(grant.efficiency_gate))
            and self.preregistration_record_digest
            == grant.preregistration_record_digest
            and self.passed_fit_protocol_record_digest
            == grant.passed_fit_protocol_record_digest
            and self.prediction_record_digest == grant.prediction_record_digest
            and self.label_record_digest == grant.label_record_digest
            and self.raw_batch_file_sha256 == grant.raw_batch_file_sha256
            and self.raw_batch_record_digest == grant.raw_batch_record_digest
            and self.recompute_receipt_file_sha256
            == grant.recompute_receipt_file_sha256
            and self.recompute_receipt_record_digest
            == grant.recompute_receipt_record_digest
            and self.occurrence_join_digest == grant.occurrence_join_digest
            and isinstance(self.archive_custody, Mapping)
            and set(self.archive_custody)
            == {
                "campaign_intent_file_sha256",
                "campaign_intent_record_digest",
                "exposure_predecessor_file_sha256",
                "exposure_predecessor_ledger_digest",
                "exposure_event_digest",
                "exposure_successor_file_sha256",
                "exposure_successor_ledger_digest",
                "exposure_authorization_file_sha256",
                "exposure_authorization_record_digest",
                "output_root_claim_file_sha256",
                "output_root_claim_record_digest",
                "execution_authorization_file_sha256",
                "execution_authorization_record_digest",
                "precommit_file_sha256",
                "precommit_record_digest",
                "prediction_attempt_file_sha256",
                "prediction_attempt_record_digest",
                "prediction_file_sha256",
                "prediction_record_digest",
                "label_attempt_file_sha256",
                "label_attempt_record_digest",
                "label_file_sha256",
                "label_record_digest",
                "outcome_file_sha256",
                "outcome_record_digest",
                "terminal_state_file_sha256",
                "terminal_state_record_digest",
            }
            and all(
                type(value) is str and _SHA_ADDRESS.fullmatch(value) is not None
                for value in self.archive_custody.values()
            )
            and self.archive_custody["prediction_record_digest"]
            == grant.prediction_record_digest
            and self.archive_custody["label_record_digest"]
            == grant.label_record_digest
            and self.archive_custody["outcome_record_digest"] == grant.record_digest
            and all(
                type(value) is int
                for value in (
                    self.pixel_reads,
                    self.feature_extraction_calls,
                    self.model_prediction_api_calls,
                    self.estimator_predict_proba_calls,
                    self.label_authority_reads,
                )
            )
            and self.pixel_reads == self.feature_extraction_calls
            == self.model_prediction_api_calls
            == self.estimator_predict_proba_calls
            == self.label_authority_reads
            == 0
            and self.exact_replay is True
            and self.authenticated_calibration_execution is False
            and self.production_adapter_authorized is False
            and self.runner_source_sha256 == "sha256:" + source_sha256()
            and self.runner_algorithm_digest == algorithm_digest()
            and self.record_digest == "sha256:" + canonical_digest(self.content_data())
        )


@dataclass(frozen=True, slots=True)
class SkeletonGraphFrozenInferenceAddresses:
    """Addresses frozen after prediction fsync/reload and before any labels."""

    raw_batch_file_sha256: str
    raw_batch_record_digest: str
    recompute_receipt_file_sha256: str
    recompute_receipt_record_digest: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.raw_batch_file_sha256, "frozen raw batch file"),
            (self.raw_batch_record_digest, "frozen raw batch record"),
            (self.recompute_receipt_file_sha256, "frozen recompute receipt file"),
            (self.recompute_receipt_record_digest, "frozen recompute receipt record"),
        ):
            _address(value, label)

    def to_data(self) -> dict[str, str]:
        return {
            "raw_batch_file_sha256": self.raw_batch_file_sha256,
            "raw_batch_record_digest": self.raw_batch_record_digest,
            "recompute_receipt_file_sha256": self.recompute_receipt_file_sha256,
            "recompute_receipt_record_digest": self.recompute_receipt_record_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "SkeletonGraphFrozenInferenceAddresses":
        raw = _fields(
            value,
            {
                "raw_batch_file_sha256", "raw_batch_record_digest",
                "recompute_receipt_file_sha256", "recompute_receipt_record_digest",
            },
            "frozen inference addresses",
        )
        result = cls(**dict(raw))
        if not _typed_equal(result.to_data(), dict(raw)):
            raise SkeletonGraphCalibrationRunnerError(
                "frozen inference addresses are not canonical"
            )
        return result


_DELAYED_LABEL_REQUEST_ISSUANCE_TOKEN = object()


@dataclass(slots=True)
class _DelayedLabelRequestLease:
    issuer: object
    consumed: bool
    lock: threading.Lock


@dataclass(frozen=True, slots=True, init=False)
class SkeletonGraphDelayedLabelRequest:
    scope: SkeletonGraphCalibrationScope
    exposure_authorization_record_digest: str
    exposure_authorization_file_sha256: str
    execution_authorization_record_digest: str
    execution_authorization_file_sha256: str
    precommit_record_digest: str
    precommit_file_sha256: str
    prediction_attempt_record_digest: str
    prediction_attempt_file_sha256: str
    prediction_record_digest: str
    prediction_file_sha256: str
    occurrence_join_digest: str
    bindings: tuple[tuple[str, SkeletonGraphCalibrationPanelIdentity], ...]
    binding_digest: str
    label_attempt_record_digest: str
    label_attempt_file_sha256: str
    output_root_path: str
    output_root_st_dev: int
    output_root_st_ino: int
    output_root_st_mode: int
    prediction_file_fsync_completed: bool
    prediction_directory_fsync_completed: bool
    prediction_fresh_reload_verified: bool
    one_shot_label_stage: bool
    runner_source_sha256: str
    runner_algorithm_digest: str
    record_digest: str
    _lease: _DelayedLabelRequestLease

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SkeletonGraphCalibrationRunnerError(
            "delayed label requests are issued only after prediction fresh reload"
        )

    def content_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-skeleton-graph-delayed-label-request.v2",
            "scope": self.scope.value,
            "exposure_authorization_record_digest": (
                self.exposure_authorization_record_digest
            ),
            "exposure_authorization_file_sha256": (
                self.exposure_authorization_file_sha256
            ),
            "execution_authorization_record_digest": (
                self.execution_authorization_record_digest
            ),
            "execution_authorization_file_sha256": (
                self.execution_authorization_file_sha256
            ),
            "precommit_record_digest": self.precommit_record_digest,
            "precommit_file_sha256": self.precommit_file_sha256,
            "prediction_attempt_record_digest": self.prediction_attempt_record_digest,
            "prediction_attempt_file_sha256": self.prediction_attempt_file_sha256,
            "prediction_record_digest": self.prediction_record_digest,
            "prediction_file_sha256": self.prediction_file_sha256,
            "occurrence_join_digest": self.occurrence_join_digest,
            "bindings": [
                {
                    "anonymous_panel_token": token,
                    "panel_identity": identity.to_data(),
                }
                for token, identity in self.bindings
            ],
            "binding_digest": self.binding_digest,
            "label_attempt_record_digest": self.label_attempt_record_digest,
            "label_attempt_file_sha256": self.label_attempt_file_sha256,
            "output_root_path": self.output_root_path,
            "output_root_st_dev": self.output_root_st_dev,
            "output_root_st_ino": self.output_root_st_ino,
            "output_root_st_mode": self.output_root_st_mode,
            "prediction_file_fsync_completed": self.prediction_file_fsync_completed,
            "prediction_directory_fsync_completed": (
                self.prediction_directory_fsync_completed
            ),
            "prediction_fresh_reload_verified": self.prediction_fresh_reload_verified,
            "one_shot_label_stage": self.one_shot_label_stage,
            "runner_source_sha256": self.runner_source_sha256,
            "runner_algorithm_digest": self.runner_algorithm_digest,
        }

    def to_data(self) -> dict[str, object]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def _issue(
        cls,
        *,
        issuance_token: object,
        scope: SkeletonGraphCalibrationScope,
        exposure_authorization_record_digest: str,
        exposure_authorization_file_sha256: str,
        execution_authorization_record_digest: str,
        execution_authorization_file_sha256: str,
        precommit_record_digest: str,
        precommit_file_sha256: str,
        prediction_attempt_record_digest: str,
        prediction_attempt_file_sha256: str,
        prediction_record_digest: str,
        prediction_file_sha256: str,
        occurrence_join_digest: str,
        bindings: tuple[tuple[str, SkeletonGraphCalibrationPanelIdentity], ...],
        label_attempt_record_digest: str,
        label_attempt_file_sha256: str,
        output_identity: Mapping[str, object],
    ) -> "SkeletonGraphDelayedLabelRequest":
        if issuance_token is not _DELAYED_LABEL_REQUEST_ISSUANCE_TOKEN:
            raise SkeletonGraphCalibrationRunnerError(
                "delayed label request issuance token differs"
            )
        binding_data = tuple(
            (token, identity.to_data()) for token, identity in bindings
        )
        values: dict[str, object] = {
            "scope": scope,
            "exposure_authorization_record_digest": (
                exposure_authorization_record_digest
            ),
            "exposure_authorization_file_sha256": (
                exposure_authorization_file_sha256
            ),
            "execution_authorization_record_digest": (
                execution_authorization_record_digest
            ),
            "execution_authorization_file_sha256": (
                execution_authorization_file_sha256
            ),
            "precommit_record_digest": precommit_record_digest,
            "precommit_file_sha256": precommit_file_sha256,
            "prediction_attempt_record_digest": prediction_attempt_record_digest,
            "prediction_attempt_file_sha256": prediction_attempt_file_sha256,
            "prediction_record_digest": prediction_record_digest,
            "prediction_file_sha256": prediction_file_sha256,
            "occurrence_join_digest": occurrence_join_digest,
            "bindings": bindings,
            "binding_digest": "sha256:" + canonical_digest(binding_data),
            "label_attempt_record_digest": label_attempt_record_digest,
            "label_attempt_file_sha256": label_attempt_file_sha256,
            "output_root_path": output_identity["absolute_path"],
            "output_root_st_dev": output_identity["st_dev"],
            "output_root_st_ino": output_identity["st_ino"],
            "output_root_st_mode": output_identity["st_mode"],
            "prediction_file_fsync_completed": True,
            "prediction_directory_fsync_completed": True,
            "prediction_fresh_reload_verified": True,
            "one_shot_label_stage": True,
            "runner_source_sha256": "sha256:" + source_sha256(),
            "runner_algorithm_digest": algorithm_digest(),
        }
        result = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(result, name, item)
        object.__setattr__(
            result,
            "record_digest",
            "sha256:" + canonical_digest(result.content_data()),
        )
        object.__setattr__(
            result,
            "_lease",
            _DelayedLabelRequestLease(
                issuer=_DELAYED_LABEL_REQUEST_ISSUANCE_TOKEN,
                consumed=False,
                lock=threading.Lock(),
            ),
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if type(self.scope) is not SkeletonGraphCalibrationScope:
            raise SkeletonGraphCalibrationRunnerError("delayed label scope differs")
        for value, label in (
            (self.exposure_authorization_record_digest, "exposure authorization record"),
            (self.exposure_authorization_file_sha256, "exposure authorization file"),
            (self.execution_authorization_record_digest, "execution authorization record"),
            (self.execution_authorization_file_sha256, "execution authorization file"),
            (self.precommit_record_digest, "label request precommit record"),
            (self.precommit_file_sha256, "label request precommit file"),
            (self.prediction_attempt_record_digest, "prediction attempt record"),
            (self.prediction_attempt_file_sha256, "prediction attempt file"),
            (self.prediction_record_digest, "label request prediction record"),
            (self.prediction_file_sha256, "label request prediction file"),
            (self.occurrence_join_digest, "label request occurrence join"),
            (self.binding_digest, "label request binding"),
            (self.label_attempt_record_digest, "label attempt record"),
            (self.label_attempt_file_sha256, "label attempt file"),
            (self.runner_source_sha256, "label request runner source"),
            (self.runner_algorithm_digest, "label request runner algorithm"),
        ):
            _address(value, label)
        if (
            type(self.bindings) is not tuple
            or not self.bindings
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or _TOKEN.fullmatch(item[0]) is None
                or type(item[1]) is not SkeletonGraphCalibrationPanelIdentity
                for item in self.bindings
            )
            or len({item[0] for item in self.bindings}) != len(self.bindings)
            or self.binding_digest
            != "sha256:"
            + canonical_digest(
                tuple((token, identity.to_data()) for token, identity in self.bindings)
            )
            or type(self.output_root_path) is not str
            or not Path(self.output_root_path).is_absolute()
            or type(self.output_root_st_dev) is not int
            or type(self.output_root_st_ino) is not int
            or type(self.output_root_st_mode) is not int
            or self.prediction_file_fsync_completed is not True
            or self.prediction_directory_fsync_completed is not True
            or self.prediction_fresh_reload_verified is not True
            or self.one_shot_label_stage is not True
            or self.runner_source_sha256 != "sha256:" + source_sha256()
            or self.runner_algorithm_digest != algorithm_digest()
            or self.record_digest != "sha256:" + canonical_digest(self.content_data())
            or type(self._lease) is not _DelayedLabelRequestLease
            or self._lease.issuer is not _DELAYED_LABEL_REQUEST_ISSUANCE_TOKEN
        ):
            raise SkeletonGraphCalibrationRunnerError("delayed label request differs")


def verify_and_consume_delayed_label_request(
    request: SkeletonGraphDelayedLabelRequest,
) -> SkeletonGraphDelayedLabelRequest:
    """Freshly replay the post-prediction barrier and consume its lease once."""

    if type(request) is not SkeletonGraphDelayedLabelRequest:
        raise TypeError("delayed label request must have exact type")
    request._validate()
    with request._lease.lock:
        if request._lease.consumed:
            raise SkeletonGraphCalibrationRunnerError(
                "delayed label request was already consumed"
            )
    directory = _existing_output_directory(Path(request.output_root_path))
    try:
        if directory.identity_data() != {
            "absolute_path": request.output_root_path,
            "st_dev": request.output_root_st_dev,
            "st_ino": request.output_root_st_ino,
            "st_mode": request.output_root_st_mode,
        }:
            raise SkeletonGraphCalibrationRunnerError(
                "delayed label output custody differs"
            )
        execution, execution_raw = _read_output_record(
            directory,
            "authorization.json",
            schema=EXECUTION_AUTHORIZATION_SCHEMA,
            label="delayed label execution authorization",
            expected_file_sha256=request.execution_authorization_file_sha256,
            expected_record_digest=request.execution_authorization_record_digest,
        )
        precommit, precommit_raw = _read_output_record(
            directory,
            "precommit.json",
            schema=PRECOMMIT_SCHEMA,
            label="delayed label precommit",
            expected_file_sha256=request.precommit_file_sha256,
            expected_record_digest=request.precommit_record_digest,
        )
        prediction_attempt, prediction_attempt_raw = _read_output_record(
            directory,
            "prediction_attempt.json",
            schema=ATTEMPT_SCHEMA,
            label="delayed label prediction attempt",
            expected_file_sha256=request.prediction_attempt_file_sha256,
            expected_record_digest=request.prediction_attempt_record_digest,
        )
        prediction, prediction_raw = _read_output_record(
            directory,
            "raw_predictions.json",
            schema=PREDICTION_SCHEMA,
            label="delayed label prediction artifact",
            expected_file_sha256=request.prediction_file_sha256,
            expected_record_digest=request.prediction_record_digest,
        )
        label_attempt, label_attempt_raw = _read_output_record(
            directory,
            "label_attempt.json",
            schema=ATTEMPT_SCHEMA,
            label="delayed label attempt",
            expected_file_sha256=request.label_attempt_file_sha256,
            expected_record_digest=request.label_attempt_record_digest,
        )
        embedded_exposure = _authorization_from_data(
            execution.get("exposure_authorization")
        )
        registration = _load_preregistration(
            Path(__file__).absolute().parent
            / "data"
            / "panel_action_count_skeleton_graph_calibration_preregistration_20260810_v1.json"
        )
        _verify_archived_exposure_authorization(
            embedded_exposure, registration=registration
        )
        output_root_claim, output_root_claim_raw = _read_exact_output_root_claim(
            directory, embedded_exposure
        )
        identities = tuple(identity for _, identity in request.bindings)
        tokens = tuple(token for token, _ in request.bindings)
        _, _, _, _ = _validate_prediction_data(
            prediction, identities=identities, tokens=tokens
        )
        if (
            _file_address(execution_raw) != request.execution_authorization_file_sha256
            or _file_address(precommit_raw) != request.precommit_file_sha256
            or _file_address(prediction_attempt_raw)
            != request.prediction_attempt_file_sha256
            or _file_address(prediction_raw) != request.prediction_file_sha256
            or _file_address(label_attempt_raw) != request.label_attempt_file_sha256
            or execution.get("scope") != request.scope.value
            or embedded_exposure.record_digest
            != request.exposure_authorization_record_digest
            or embedded_exposure.file_sha256
            != request.exposure_authorization_file_sha256
            or execution.get("exposure_authorization_file_sha256")
            != request.exposure_authorization_file_sha256
            or execution.get("output_root_claim_record_digest")
            != output_root_claim["record_digest"]
            or execution.get("output_root_claim_file_sha256")
            != _file_address(output_root_claim_raw)
            or precommit.get("exposure_authorization_record_digest")
            != request.exposure_authorization_record_digest
            or precommit.get("exposure_authorization_file_sha256")
            != request.exposure_authorization_file_sha256
            or precommit.get("output_root_claim_record_digest")
            != output_root_claim["record_digest"]
            or precommit.get("output_root_claim_file_sha256")
            != _file_address(output_root_claim_raw)
            or precommit.get("scope") != request.scope.value
            or precommit.get("execution_authorization_record_digest")
            != request.execution_authorization_record_digest
            or precommit.get("execution_authorization_file_sha256")
            != request.execution_authorization_file_sha256
            or prediction_attempt.get("stage") != "prediction"
            or prediction_attempt.get("precommit_record_digest")
            != request.precommit_record_digest
            or prediction_attempt.get("precommit_file_sha256")
            != request.precommit_file_sha256
            or _exact_int(
                prediction_attempt.get("attempt_number"),
                "prediction attempt number",
                lower=1,
            )
            != 1
            or prediction_attempt.get("reroll_authorized") is not False
            or prediction.get("precommit_record_digest")
            != request.precommit_record_digest
            or prediction.get("precommit_file_sha256") != request.precommit_file_sha256
            or prediction.get("occurrence_join_digest")
            != request.occurrence_join_digest
            or prediction.get("file_fsync_completed") is not True
            or prediction.get("directory_fsync_completed") is not True
            or label_attempt.get("stage") != "delayed_labels"
            or label_attempt.get("precommit_record_digest")
            != request.precommit_record_digest
            or label_attempt.get("precommit_file_sha256") != request.precommit_file_sha256
            or label_attempt.get("prediction_record_digest")
            != request.prediction_record_digest
            or label_attempt.get("prediction_file_sha256")
            != request.prediction_file_sha256
            or label_attempt.get("prediction_fresh_reload_verified") is not True
            or _exact_int(
                label_attempt.get("attempt_number"), "label attempt number", lower=1
            )
            != 1
            or label_attempt.get("reroll_authorized") is not False
            or _output_entry_exists(directory, "terminal_state.json")
            or _output_entry_exists(directory, "population_grant.json")
            or _output_entry_exists(directory, "calibration_gap.json")
            or _output_entry_exists(directory, "delayed_labels.json")
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "delayed label request archive join differs"
            )
        with request._lease.lock:
            if request._lease.consumed:
                raise SkeletonGraphCalibrationRunnerError(
                    "delayed label request was already consumed"
                )
            request._lease.consumed = True
        return request
    finally:
        directory.close()


SkeletonGraphInferenceRunner: TypeAlias = Callable[
    [tuple[bytes, ...], SkeletonGraphPassedFitProtocol],
    tuple[SkeletonGraphRawInferenceBatch, SkeletonGraphInferenceRecomputeReceipt],
]
SkeletonGraphCalibrationPixelReader: TypeAlias = Callable[[str], bytes]
SkeletonGraphDelayedLabelReader: TypeAlias = Callable[
    [SkeletonGraphDelayedLabelRequest], SkeletonGraphDelayedLabelBatch
]
SkeletonGraphDelayedLabelReaderFactory: TypeAlias = Callable[
    [], SkeletonGraphDelayedLabelReader
]


def make_verified_inference_runner(
    paths: SkeletonGraphPassedFitPaths,
) -> SkeletonGraphInferenceRunner:
    """Capture the six artifact paths without opening a pixel or model now."""

    if type(paths) is not SkeletonGraphPassedFitPaths:
        raise TypeError("inference runner needs exact passed-fit paths")
    keywords = paths.keyword_arguments()

    def infer(
        png_payloads: tuple[bytes, ...], passed_fit: SkeletonGraphPassedFitProtocol
    ) -> tuple[SkeletonGraphRawInferenceBatch, SkeletonGraphInferenceRecomputeReceipt]:
        batch = create_raw_inference_batch(
            passed_fit=passed_fit, png_payloads=png_payloads, **keywords
        )
        receipt = fresh_verify_raw_inference_batch(
            batch,
            passed_fit=passed_fit,
            png_payloads=png_payloads,
            **keywords,
        )
        return batch, receipt

    return infer


def _stable_bytes(path: Path, *, label: str, maximum: int = 64 << 20) -> bytes:
    supplied = Path(path)
    try:
        before = supplied.lstat()
        absolute = supplied.absolute()
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise SkeletonGraphCalibrationRunnerError(f"cannot stat {label}: {exc}") from exc
    if (
        resolved != absolute
        or not supplied.is_file()
        or os.path.islink(supplied)
        or before.st_size <= 0
        or before.st_size > maximum
    ):
        raise SkeletonGraphCalibrationRunnerError(
            f"{label} must be a bounded regular nonsymlink file"
        )
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
        raise SkeletonGraphCalibrationRunnerError(f"cannot read {label}: {exc}") from exc
    finally:
        os.close(descriptor)
    try:
        after = supplied.lstat()
    except OSError as exc:
        raise SkeletonGraphCalibrationRunnerError(f"cannot restat {label}: {exc}") from exc
    fingerprint = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    if not (
        fingerprint(before)
        == fingerprint(opened)
        == fingerprint(after_read)
        == fingerprint(after)
    ):
        raise SkeletonGraphCalibrationRunnerError(f"{label} changed while reading")
    raw = b"".join(chunks)
    if len(raw) != before.st_size or len(raw) > maximum:
        raise SkeletonGraphCalibrationRunnerError(f"{label} size differs")
    return raw


def _record_from_bytes(
    raw: bytes,
    *,
    schema: str,
    label: str,
    expected_file_sha256: str | None = None,
    expected_record_digest: str | None = None,
) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SkeletonGraphCalibrationRunnerError(f"cannot decode {label}: {exc}") from exc
    if type(value) is not dict or raw != canonical_json(value) + b"\n":
        raise SkeletonGraphCalibrationRunnerError(f"{label} is not canonical JSON plus newline")
    body = dict(value)
    found = body.pop("record_digest", None)
    if value.get("schema") != schema or found != "sha256:" + canonical_digest(body):
        raise SkeletonGraphCalibrationRunnerError(f"{label} schema or digest differs")
    if expected_file_sha256 is not None and _file_address(raw) != expected_file_sha256:
        raise SkeletonGraphCalibrationRunnerError(f"{label} file address differs")
    if expected_record_digest is not None and found != expected_record_digest:
        raise SkeletonGraphCalibrationRunnerError(f"{label} record address differs")
    return value


def _read_record(
    path: Path,
    *,
    schema: str,
    label: str,
    expected_file_sha256: str | None = None,
    expected_record_digest: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    raw = _stable_bytes(path, label=label)
    return (
        _record_from_bytes(
            raw,
            schema=schema,
            label=label,
            expected_file_sha256=expected_file_sha256,
            expected_record_digest=expected_record_digest,
        ),
        raw,
    )


@dataclass(slots=True)
class _OutputDirectoryCustody:
    """One retained directory inode plus its still-bound pathname."""

    path: Path
    device: int
    inode: int
    mode: int
    descriptor: int

    def _open(self) -> int:
        try:
            retained = os.fstat(self.descriptor)
            current_descriptor, current = _open_directory_fd_no_symlink(self.path)
        except OSError as exc:
            raise SkeletonGraphCalibrationRunnerError(
                f"output root was renamed, replaced, or redirected: {exc}"
            ) from exc
        else:
            os.close(current_descriptor)
        expected = (self.device, self.inode, self.mode)
        if (
            (retained.st_dev, retained.st_ino, retained.st_mode) != expected
            or (current.st_dev, current.st_ino, current.st_mode) != expected
            or not stat.S_ISDIR(retained.st_mode)
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "output root was renamed, replaced, or redirected"
            )
        return os.dup(self.descriptor)

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def identity_data(self) -> dict[str, int | str]:
        return {
            "absolute_path": str(self.path),
            "st_dev": self.device,
            "st_ino": self.inode,
            "st_mode": self.mode,
        }


def _open_directory_fd_no_symlink(path: Path) -> tuple[int, os.stat_result]:
    """Walk an absolute directory from `/` using only no-follow openat calls."""

    supplied = Path(path)
    if not supplied.is_absolute() or ".." in supplied.parts:
        raise OSError("directory path must be absolute and normalized")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open("/", flags)
    try:
        for component in supplied.parts[1:]:
            if component in ("", ".", ".."):
                raise OSError("directory component differs")
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        found = os.fstat(descriptor)
        if not stat.S_ISDIR(found.st_mode):
            raise OSError("terminal path is not a directory")
        return descriptor, found
    except BaseException:
        os.close(descriptor)
        raise


def _open_existing_directory(path: Path, *, label: str) -> _OutputDirectoryCustody:
    supplied = Path(path).absolute()
    try:
        descriptor, opened = _open_directory_fd_no_symlink(supplied)
        check_descriptor, checked = _open_directory_fd_no_symlink(supplied)
    except OSError as exc:
        if "descriptor" in locals():
            os.close(descriptor)
        raise SkeletonGraphCalibrationRunnerError(f"{label} unavailable: {exc}") from exc
    finally:
        if "check_descriptor" in locals():
            os.close(check_descriptor)
    identity = (opened.st_dev, opened.st_ino, opened.st_mode)
    if (
        identity != (checked.st_dev, checked.st_ino, checked.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
    ):
        os.close(descriptor)
        raise SkeletonGraphCalibrationRunnerError(f"{label} identity differs")
    return _OutputDirectoryCustody(
        supplied, opened.st_dev, opened.st_ino, opened.st_mode, descriptor
    )


def _validate_intended_output_directory(
    path: Path,
) -> tuple[Path, _OutputDirectoryCustody]:
    output = Path(path).absolute()
    if output.name in ("", ".", ".."):
        raise SkeletonGraphCalibrationRunnerError("intended output name differs")
    parent = _open_existing_directory(output.parent, label="output parent")
    descriptor = parent._open()
    try:
        try:
            os.stat(output.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise SkeletonGraphCalibrationRunnerError(
                "intended output directory is not fresh"
            )
    finally:
        os.close(descriptor)
    return output, parent


def _acquire_recoverable_output_directory(
    path: Path,
    *,
    expected_parent_identity: Mapping[str, object],
) -> tuple[_OutputDirectoryCustody, bool]:
    """Create or reopen the one authorized child using its retained parent."""

    output = Path(path).absolute()
    if output.name in ("", ".", ".."):
        raise SkeletonGraphCalibrationRunnerError("authorized output name differs")
    parent = _open_existing_directory(output.parent, label="authorized output parent")
    child_descriptor: int | None = None
    created = False
    try:
        if parent.identity_data() != dict(expected_parent_identity):
            raise SkeletonGraphCalibrationRunnerError(
                "output parent differs from exposure authorization"
            )
        parent_descriptor = parent._open()
        try:
            try:
                found = os.stat(
                    output.name, dir_fd=parent_descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                os.mkdir(output.name, mode=0o700, dir_fd=parent_descriptor)
                created = True
                found = os.stat(
                    output.name, dir_fd=parent_descriptor, follow_symlinks=False
                )
            if (
                not stat.S_ISDIR(found.st_mode)
                or stat.S_IMODE(found.st_mode) != 0o700
            ):
                raise SkeletonGraphCalibrationRunnerError(
                    "authorized output child type or mode differs"
                )
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            child_descriptor = os.open(
                output.name, flags, dir_fd=parent_descriptor
            )
            child = os.fstat(child_descriptor)
            child_from_parent = os.stat(
                output.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            os.fsync(child_descriptor)
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        check_descriptor, current = _open_directory_fd_no_symlink(output)
        os.close(check_descriptor)
        identity = (child.st_dev, child.st_ino, child.st_mode)
        if (
            identity
            != (
                child_from_parent.st_dev,
                child_from_parent.st_ino,
                child_from_parent.st_mode,
            )
            or identity != (current.st_dev, current.st_ino, current.st_mode)
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "authorized output child changed during acquisition"
            )
        custody = _OutputDirectoryCustody(
            output, child.st_dev, child.st_ino, child.st_mode, child_descriptor
        )
        child_descriptor = None
        return custody, created
    except OSError as exc:
        raise SkeletonGraphCalibrationRunnerError(
            f"cannot acquire authorized output child: {exc}"
        ) from exc
    finally:
        if child_descriptor is not None:
            os.close(child_descriptor)
        parent.close()


def _existing_output_directory(path: Path) -> _OutputDirectoryCustody:
    return _open_existing_directory(Path(path), label="archive root")


def _read_dirfd_bytes(
    descriptor: int, name: str, *, label: str, maximum: int = 64 << 20
) -> bytes:
    if Path(name).name != name or name in ("", ".", ".."):
        raise SkeletonGraphCalibrationRunnerError("output artifact name differs")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(name, flags, dir_fd=descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > maximum:
            raise SkeletonGraphCalibrationRunnerError(
                f"{label} must be a bounded regular nonsymlink file"
            )
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(file_descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(file_descriptor)
    except OSError as exc:
        raise SkeletonGraphCalibrationRunnerError(f"cannot read {label}: {exc}") from exc
    finally:
        if "file_descriptor" in locals():
            os.close(file_descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns)
    ):
        raise SkeletonGraphCalibrationRunnerError(f"{label} changed while reading")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise SkeletonGraphCalibrationRunnerError(f"{label} size differs")
    return raw


def _confirm_durable_exact_file(
    descriptor: int,
    name: str,
    raw: bytes,
    *,
    label: str,
) -> bytes:
    """Require exact leaf fsync, directory fsync, and an independent reload."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(name, flags, dir_fd=descriptor)
    try:
        found = os.fstat(file_descriptor)
        if not stat.S_ISREG(found.st_mode) or found.st_size != len(raw):
            raise SkeletonGraphCalibrationRunnerError(
                f"durable {label} leaf differs"
            )
        file_error: OSError | None = None
        for _attempt in range(3):
            try:
                os.fsync(file_descriptor)
            except OSError as exc:
                file_error = exc
            else:
                file_error = None
                break
        if file_error is not None:
            raise file_error
    finally:
        os.close(file_descriptor)

    directory_error: OSError | None = None
    for _attempt in range(3):
        try:
            os.fsync(descriptor)
        except OSError as exc:
            directory_error = exc
        else:
            directory_error = None
            break
    if directory_error is not None:
        raise directory_error

    reload_error: Exception | None = None
    for _attempt in range(3):
        try:
            loaded = _read_dirfd_bytes(descriptor, name, label=label)
        except Exception as exc:
            reload_error = exc
        else:
            if loaded != raw:
                raise SkeletonGraphCalibrationRunnerError(
                    f"fresh reload of durable {label} differs"
                )
            return loaded
    assert reload_error is not None
    raise reload_error


def _atomic_write_once_bytes(
    custody: _OutputDirectoryCustody,
    name: str,
    raw: bytes,
    *,
    label: str,
    allow_identical_existing: bool,
) -> bytes:
    """Publish a complete private inode once; publication is monotonic."""

    if Path(name).name != name or name in ("", ".", "..") or not raw:
        raise SkeletonGraphCalibrationRunnerError(f"{label} name/payload differs")
    descriptor = custody._open()
    temporary = (
        f".{name}.pending.{os.getpid()}.{threading.get_ident()}."
        f"{os.urandom(16).hex()}"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        try:
            existing = _read_dirfd_bytes(descriptor, name, label=label)
        except SkeletonGraphCalibrationRunnerError as exc:
            try:
                os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                existing = None
            else:
                raise exc
        if existing is not None:
            if not allow_identical_existing or existing != raw:
                raise SkeletonGraphCalibrationRunnerError(
                    f"write-once {label} already exists or differs"
                )
            return _confirm_durable_exact_file(
                descriptor, name, raw, label=label
            )
        file_descriptor = os.open(temporary, flags, 0o600, dir_fd=descriptor)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(file_descriptor, view)
                if written <= 0:
                    raise OSError("artifact write made no progress")
                view = view[written:]
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            winner = _read_dirfd_bytes(descriptor, name, label=label)
            if not allow_identical_existing or winner != raw:
                raise SkeletonGraphCalibrationRunnerError(
                    f"concurrent {label} winner differs"
                )
            return _confirm_durable_exact_file(
                descriptor, name, raw, label=label
            )
        return _confirm_durable_exact_file(
            descriptor, name, raw, label=label
        )
    except Exception as exc:
        if isinstance(exc, SkeletonGraphCalibrationRunnerError):
            raise
        raise SkeletonGraphCalibrationRunnerError(
            f"atomic durable write of {label} failed: {exc}"
        ) from exc
    finally:
        try:
            os.unlink(temporary, dir_fd=descriptor)
        except OSError:
            pass
        os.close(descriptor)


def _write_output_record(
    custody: _OutputDirectoryCustody,
    name: str,
    body: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    value = _seal(body)
    raw = canonical_json(value) + b"\n"
    loaded_raw = _atomic_write_once_bytes(
        custody,
        name,
        raw,
        label=name,
        allow_identical_existing=False,
    )
    loaded = _record_from_bytes(
        loaded_raw,
        schema=value["schema"],
        label=name,
        expected_file_sha256=_file_address(raw),
        expected_record_digest=value["record_digest"],
    )
    if loaded != value:
        raise SkeletonGraphCalibrationRunnerError(f"fresh reload of {name} differs")
    return loaded, _file_address(raw)


def _read_output_record(
    custody: _OutputDirectoryCustody,
    name: str,
    *,
    schema: str,
    label: str,
    expected_file_sha256: str | None = None,
    expected_record_digest: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    descriptor = custody._open()
    try:
        raw = _read_dirfd_bytes(descriptor, name, label=label)
    finally:
        os.close(descriptor)
    check = custody._open()
    os.close(check)
    return (
        _record_from_bytes(
            raw,
            schema=schema,
            label=label,
            expected_file_sha256=expected_file_sha256,
            expected_record_digest=expected_record_digest,
        ),
        raw,
    )


def _output_entry_exists(custody: _OutputDirectoryCustody, name: str) -> bool:
    descriptor = custody._open()
    try:
        try:
            found = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(found.st_mode):
            raise SkeletonGraphCalibrationRunnerError(
                f"output entry {name} is not a regular file"
            )
        return True
    finally:
        os.close(descriptor)


def _output_root_claim_body(
    *,
    authorization: SkeletonGraphCalibrationExposureAuthorization,
    output_identity: Mapping[str, object],
) -> dict[str, Any]:
    return {
        "schema": OUTPUT_ROOT_CLAIM_SCHEMA,
        "exposure_authorization": authorization.to_data(),
        "exposure_authorization_file_sha256": authorization.file_sha256,
        "campaign_intent_record_digest": authorization.campaign_intent_record_digest,
        "campaign_intent_file_sha256": authorization.campaign_intent_file_sha256,
        "intended_output_directory": authorization.intended_output_directory,
        "output_root_identity": dict(output_identity),
        "created_or_reopened_before_any_calibration_pixel": True,
        "calibration_pixel_reads_so_far": 0,
        "model_calls_so_far": 0,
        "label_authority_reads_so_far": 0,
        "authenticated_calibration_execution": False,
        "production_adapter_authorized": False,
    }


def _persist_or_verify_output_root_claim(
    output: _OutputDirectoryCustody,
    authorization: SkeletonGraphCalibrationExposureAuthorization,
) -> tuple[dict[str, Any], str]:
    expected = _seal(
        _output_root_claim_body(
            authorization=authorization,
            output_identity=output.identity_data(),
        )
    )
    expected_raw = canonical_json(expected) + b"\n"
    loaded_raw = _atomic_write_once_bytes(
        output,
        "output_root_claim.json",
        expected_raw,
        label="output root claim",
        allow_identical_existing=True,
    )
    loaded = _record_from_bytes(
        loaded_raw,
        schema=OUTPUT_ROOT_CLAIM_SCHEMA,
        label="output root claim",
        expected_file_sha256=_file_address(expected_raw),
        expected_record_digest=expected["record_digest"],
    )
    if loaded != expected or loaded_raw != expected_raw:
        raise SkeletonGraphCalibrationRunnerError("output root claim differs")
    return loaded, _file_address(loaded_raw)


def _read_exact_output_root_claim(
    output: _OutputDirectoryCustody,
    authorization: SkeletonGraphCalibrationExposureAuthorization,
) -> tuple[dict[str, Any], bytes]:
    expected = _seal(
        _output_root_claim_body(
            authorization=authorization,
            output_identity=output.identity_data(),
        )
    )
    expected_raw = canonical_json(expected) + b"\n"
    loaded, loaded_raw = _read_output_record(
        output,
        "output_root_claim.json",
        schema=OUTPUT_ROOT_CLAIM_SCHEMA,
        label="output root claim",
        expected_file_sha256=_file_address(expected_raw),
        expected_record_digest=expected["record_digest"],
    )
    if loaded != expected or loaded_raw != expected_raw:
        raise SkeletonGraphCalibrationRunnerError("output root claim replay differs")
    return loaded, loaded_raw


_INVENTORY_SCHEMA_BY_NAME: Mapping[str, str] = MappingProxyType(
    {
        "output_root_claim.json": OUTPUT_ROOT_CLAIM_SCHEMA,
        "authorization.json": EXECUTION_AUTHORIZATION_SCHEMA,
        "precommit.json": PRECOMMIT_SCHEMA,
        "prediction_attempt.json": ATTEMPT_SCHEMA,
        "raw_predictions.json": PREDICTION_SCHEMA,
        "label_attempt.json": ATTEMPT_SCHEMA,
        "delayed_labels.json": LABEL_BATCH_SCHEMA,
    }
)
_PRIVATE_PENDING_NAME = re.compile(
    r"\..+\.pending\.[0-9]+\.[0-9]+\.[0-9a-f]{32}\Z"
)


def _canonical_inventory_record(raw: bytes) -> Mapping[str, Any] | None:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, Mapping)
        or type(value.get("schema")) is not str
        or type(value.get("record_digest")) is not str
        or canonical_json(value) + b"\n" != raw
    ):
        return None
    body = dict(value)
    record_digest = body.pop("record_digest")
    if record_digest != "sha256:" + canonical_digest(body):
        return None
    return value


def _execution_inventory(
    output: _OutputDirectoryCustody,
) -> dict[str, dict[str, object]]:
    """Address every nonterminal child; private incomplete inodes are inert."""

    descriptor = output._open()
    try:
        names = sorted(os.listdir(descriptor))
        result: dict[str, dict[str, object]] = {}
        for name in names:
            if _PRIVATE_PENDING_NAME.fullmatch(name) is not None:
                continue
            if name in {
                "terminal_state.json",
                "population_grant.json",
                "calibration_gap.json",
                "cold_replay.json",
            }:
                continue
            found = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(found.st_mode):
                raise SkeletonGraphCalibrationRunnerError(
                    f"execution inventory entry {name} is not regular"
                )
            raw = _read_dirfd_bytes(
                descriptor,
                name,
                label=f"execution inventory {name}",
            )
            record = _canonical_inventory_record(raw)
            schema = None if record is None else record["schema"]
            record_digest = None if record is None else record["record_digest"]
            expected_schema = _INVENTORY_SCHEMA_BY_NAME.get(name)
            if expected_schema is not None and schema != expected_schema:
                schema = None
                record_digest = None
            result[name] = {
                "schema": schema,
                "file_sha256": _file_address(raw),
                "record_digest": record_digest,
                "size_bytes": len(raw),
            }
        return result
    finally:
        os.close(descriptor)


def _validate_inventory_wire(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping) or any(type(name) is not str for name in value):
        raise SkeletonGraphCalibrationRunnerError("integrity inventory differs")
    result: dict[str, dict[str, object]] = {}
    for name, entry_value in value.items():
        if Path(name).name != name or _PRIVATE_PENDING_NAME.fullmatch(name):
            raise SkeletonGraphCalibrationRunnerError("integrity inventory name differs")
        entry = _fields(
            entry_value,
            {"schema", "file_sha256", "record_digest", "size_bytes"},
            "integrity inventory entry",
        )
        if entry["schema"] is not None and type(entry["schema"]) is not str:
            raise SkeletonGraphCalibrationRunnerError("inventory schema differs")
        _address(entry["file_sha256"], "inventory file")
        if entry["record_digest"] is not None:
            _address(entry["record_digest"], "inventory record")
        _exact_int(entry["size_bytes"], "inventory size", lower=1)
        result[name] = dict(entry)
    return result


def _outcome_from_data(value: object) -> SkeletonGraphCalibrationOutcome:
    if not isinstance(value, Mapping):
        raise SkeletonGraphCalibrationRunnerError("terminal outcome wire differs")
    schema = value.get("schema")
    if schema == GAP_SCHEMA:
        return SkeletonGraphCalibrationGap.from_data(value)
    if schema in (GENERIC_GRANT_SCHEMA, SAME_FAMILY_GRANT_SCHEMA):
        return SkeletonGraphPopulationGrant.from_data(value)
    raise SkeletonGraphCalibrationRunnerError("terminal outcome schema differs")


def _terminal_state_body(
    *,
    authorization: SkeletonGraphCalibrationExposureAuthorization,
    output_root_claim: Mapping[str, Any],
    output_root_claim_file_sha256: str,
    outcome: SkeletonGraphCalibrationOutcome,
) -> dict[str, Any]:
    outcome_name = (
        "population_grant.json"
        if type(outcome) is SkeletonGraphPopulationGrant
        else "calibration_gap.json"
    )
    outcome_raw = canonical_json(outcome.to_data()) + b"\n"
    return {
        "schema": TERMINAL_STATE_SCHEMA,
        "exposure_authorization_record_digest": authorization.record_digest,
        "exposure_authorization_file_sha256": authorization.file_sha256,
        "output_root_claim_record_digest": output_root_claim["record_digest"],
        "output_root_claim_file_sha256": output_root_claim_file_sha256,
        "outcome_name": outcome_name,
        "outcome_schema": outcome.to_data()["schema"],
        "outcome_record_digest": outcome.record_digest,
        "outcome_file_sha256": _file_address(outcome_raw),
        "outcome": outcome.to_data(),
        "single_terminal_winner": True,
        "authenticated_calibration_execution": False,
        "production_adapter_authorized": False,
    }


def _recover_terminal_outcome(
    output: _OutputDirectoryCustody,
    *,
    authorization: SkeletonGraphCalibrationExposureAuthorization,
    output_root_claim: Mapping[str, Any],
    output_root_claim_file_sha256: str,
) -> SkeletonGraphCalibrationOutcome:
    state, state_raw = _read_output_record(
        output,
        "terminal_state.json",
        schema=TERMINAL_STATE_SCHEMA,
        label="terminal state",
    )
    expected_fields = {
        "schema",
        "exposure_authorization_record_digest",
        "exposure_authorization_file_sha256",
        "output_root_claim_record_digest",
        "output_root_claim_file_sha256",
        "outcome_name",
        "outcome_schema",
        "outcome_record_digest",
        "outcome_file_sha256",
        "outcome",
        "single_terminal_winner",
        "authenticated_calibration_execution",
        "production_adapter_authorized",
        "record_digest",
    }
    _fields(state, expected_fields, "terminal state")
    outcome = _outcome_from_data(state["outcome"])
    expected = _seal(
        _terminal_state_body(
            authorization=authorization,
            output_root_claim=output_root_claim,
            output_root_claim_file_sha256=output_root_claim_file_sha256,
            outcome=outcome,
        )
    )
    if (
        not _typed_equal(state, expected)
        or state_raw != canonical_json(expected) + b"\n"
    ):
        raise SkeletonGraphCalibrationRunnerError("terminal state full join differs")
    outcome_name = state["outcome_name"]
    opposite = (
        "calibration_gap.json"
        if outcome_name == "population_grant.json"
        else "population_grant.json"
    )
    if _output_entry_exists(output, opposite):
        raise SkeletonGraphCalibrationRunnerError(
            "grant and GAP terminal outcomes coexist"
        )
    outcome_raw = canonical_json(outcome.to_data()) + b"\n"
    loaded_raw = _atomic_write_once_bytes(
        output,
        outcome_name,
        outcome_raw,
        label="terminal outcome",
        allow_identical_existing=True,
    )
    if loaded_raw != outcome_raw:
        raise SkeletonGraphCalibrationRunnerError("terminal outcome reload differs")
    return outcome


def _persist_terminal_outcome(
    output: _OutputDirectoryCustody,
    *,
    authorization: SkeletonGraphCalibrationExposureAuthorization,
    output_root_claim: Mapping[str, Any],
    output_root_claim_file_sha256: str,
    outcome: SkeletonGraphCalibrationOutcome,
) -> SkeletonGraphCalibrationOutcome:
    if _output_entry_exists(output, "terminal_state.json"):
        return _recover_terminal_outcome(
            output,
            authorization=authorization,
            output_root_claim=output_root_claim,
            output_root_claim_file_sha256=output_root_claim_file_sha256,
        )
    state = _seal(
        _terminal_state_body(
            authorization=authorization,
            output_root_claim=output_root_claim,
            output_root_claim_file_sha256=output_root_claim_file_sha256,
            outcome=outcome,
        )
    )
    state_raw = canonical_json(state) + b"\n"
    try:
        _atomic_write_once_bytes(
            output,
            "terminal_state.json",
            state_raw,
            label="terminal state",
            allow_identical_existing=True,
        )
    except SkeletonGraphCalibrationRunnerError:
        if not _output_entry_exists(output, "terminal_state.json"):
            raise
    return _recover_terminal_outcome(
        output,
        authorization=authorization,
        output_root_claim=output_root_claim,
        output_root_claim_file_sha256=output_root_claim_file_sha256,
    )


def _persist_or_verify_replay(
    custody: _OutputDirectoryCustody, body: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    expected = _seal(body)
    expected_raw = canonical_json(expected) + b"\n"
    raw = _atomic_write_once_bytes(
        custody,
        "cold_replay.json",
        expected_raw,
        label="cold replay",
        allow_identical_existing=True,
    )
    loaded = _record_from_bytes(
        raw,
        schema=REPLAY_SCHEMA,
        label="cold replay",
        expected_file_sha256=_file_address(expected_raw),
        expected_record_digest=expected["record_digest"],
    )
    if loaded != expected or raw != expected_raw:
        raise SkeletonGraphCalibrationRunnerError("existing cold replay differs")
    return loaded, _file_address(raw)


def _load_preregistration(path: Path) -> dict[str, Any]:
    _authority_preflight()
    value, _ = _read_record(
        path,
        schema=prereg.SCHEMA,
        label="skeleton calibration preregistration",
        expected_file_sha256=PINNED_PREREGISTRATION_FILE_SHA256,
        expected_record_digest=PINNED_PREREGISTRATION_RECORD_DIGEST,
    )
    if (
        value.get("preregistration_authority", {}).get("source_sha256")
        != PINNED_PREREGISTRATION_SOURCE_SHA256
        or value.get("metadata_only_preregistration") is not True
    ):
        raise SkeletonGraphCalibrationRunnerError("preregistration authority differs")
    return value


def _verify_passed_fit_outcome(
    outcome: SkeletonGraphPassedFitOutcome,
    paths: SkeletonGraphPassedFitPaths,
) -> SkeletonGraphPassedFitOutcome:
    if type(paths) is not SkeletonGraphPassedFitPaths:
        raise TypeError("passed-fit verification needs exact path inventory")
    if type(outcome) not in (SkeletonGraphPassedFitProtocol, SkeletonGraphPassedFitGap):
        raise TypeError("passed-fit outcome type differs")
    try:
        verified = verify_skeleton_graph_passed_fit_protocol(
            outcome,
            **paths.keyword_arguments(),
            expected_record_digest=outcome.record_digest,
        )
    except Exception as exc:
        raise SkeletonGraphCalibrationRunnerError(
            "passed-fit full-chain verification failed"
        ) from exc
    if type(verified) is SkeletonGraphPassedFitProtocol and (
        verified.record_digest != PINNED_PASSED_FIT_PROTOCOL_RECORD_DIGEST
        or verified.passed_fit_authority_source_sha256
        != "sha256:" + PINNED_PASSED_FIT_SOURCE_SHA256
        or verified.passed_fit_algorithm_digest != PINNED_PASSED_FIT_ALGORITHM_DIGEST
    ):
        raise SkeletonGraphCalibrationRunnerError("passed-fit protocol address differs")
    return verified


def _scope_campaign(
    registration: Mapping[str, Any], scope: SkeletonGraphCalibrationScope
) -> Mapping[str, Any]:
    key = (
        "generic_v3_calibration"
        if scope is SkeletonGraphCalibrationScope.GENERIC_V3
        else "same_family_calibration"
    )
    campaign = registration.get(key)
    if not isinstance(campaign, Mapping):
        raise SkeletonGraphCalibrationRunnerError("preregistered campaign differs")
    return campaign


@dataclass(slots=True)
class _CampaignAuthority:
    parent: _OutputDirectoryCustody
    predecessor: ExposureLedger
    predecessor_raw: bytes
    predecessor_filename: str
    intent_filename: str

    def close(self) -> None:
        self.parent.close()


def _unique_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SkeletonGraphCalibrationRunnerError(f"cannot decode {label}") from exc
    if type(value) is not dict:
        raise SkeletonGraphCalibrationRunnerError(f"{label} is not a JSON object")
    return value


def _decode_exposure_ledger(raw: bytes, *, label: str) -> ExposureLedger:
    value = _unique_json_object(raw, label=label)
    events = value.get("events")
    if (
        type(events) is not list
        or any(
            type(event) is not dict or type(event.get("sequence")) is not int
            for event in events
        )
    ):
        raise SkeletonGraphCalibrationRunnerError(
            f"{label} contains a non-exact integer event sequence"
        )
    try:
        result = ExposureLedger.from_dict(value)
    except Exception as exc:
        raise SkeletonGraphCalibrationRunnerError(f"{label} is not a valid ledger") from exc
    if result.to_json().encode("utf-8") != raw:
        raise SkeletonGraphCalibrationRunnerError(f"{label} bytes are not canonical")
    return result


def _campaign_attempt_authority(
    registration: Mapping[str, Any], scope: SkeletonGraphCalibrationScope
) -> _CampaignAuthority:
    """Retain the exact predecessor parent and load its canonical ledger."""

    if type(scope) is not SkeletonGraphCalibrationScope:
        raise TypeError("campaign authority scope differs")
    repository_root = Path(prereg.__file__).absolute().parents[1]
    relative = registration.get("exposure_predecessor", {}).get("ledger_path")
    if type(relative) is not str or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise SkeletonGraphCalibrationRunnerError("exposure predecessor path differs")
    ledger = repository_root / relative
    parent = _open_existing_directory(
        ledger.parent, label="exposure predecessor parent"
    )
    descriptor = parent._open()
    try:
        raw = _read_dirfd_bytes(
            descriptor,
            ledger.name,
            label="exposure predecessor",
            maximum=16 << 20,
        )
    finally:
        os.close(descriptor)
    predecessor = _decode_exposure_ledger(raw, label="exposure predecessor")
    expected = registration["exposure_predecessor"]
    if (
        _file_address(raw) != expected.get("ledger_source_sha256")
        or predecessor.digest != expected.get("ledger_digest")
        or ledger.name
        != predecessor.digest.removeprefix("sha256:") + ".exposure.json"
    ):
        parent.close()
        raise SkeletonGraphCalibrationRunnerError("exposure predecessor differs")
    return _CampaignAuthority(
        parent=parent,
        predecessor=predecessor,
        predecessor_raw=raw,
        predecessor_filename=ledger.name,
        intent_filename="panel_action_count_skeleton_graph_campaign_attempt_v2.json",
    )


def _derive_cohort_ids(
    registration: Mapping[str, Any], scope: SkeletonGraphCalibrationScope
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    campaign = _scope_campaign(registration, scope)
    identity = campaign.get("identity_binding")
    if not isinstance(identity, Mapping):
        raise SkeletonGraphCalibrationRunnerError("campaign identity binding differs")
    if scope is SkeletonGraphCalibrationScope.SAME_FAMILY:
        raw_tasks = identity.get("task_ids")
        if type(raw_tasks) is not list:
            raise SkeletonGraphCalibrationRunnerError("same-family task inventory differs")
        task_ids = tuple(raw_tasks)
    else:
        relative = identity.get("manifest_path")
        if type(relative) is not str or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise SkeletonGraphCalibrationRunnerError("generic manifest path differs")
        repository_root = Path(prereg.__file__).absolute().parents[1]
        raw = _stable_bytes(
            repository_root / relative,
            label="generic calibration identity manifest",
            maximum=1 << 20,
        )
        value = _record_from_bytes(
            raw,
            schema="gkm.bongard-action-count-cnn-calibration-panel-ids.v3",
            label="generic calibration identity manifest",
            expected_file_sha256=identity.get("manifest_source_sha256"),
            expected_record_digest=identity.get("manifest_record_digest"),
        )
        cohort = value.get("cohorts", {}).get("calibration")
        if not isinstance(cohort, Mapping) or type(cohort.get("task_ids")) is not list:
            raise SkeletonGraphCalibrationRunnerError("generic cohort inventory differs")
        task_ids = tuple(cohort["task_ids"])
    panel_ids = tuple(
        f"hd/{task_id}/{side}/{ordinal}.png"
        for task_id in task_ids
        for side in (1, 0)
        for ordinal in range(7)
    )
    if (
        len(task_ids) != _exact_int(
            campaign.get("calibration_task_count"), "campaign task count", lower=1
        )
        or len(panel_ids) != _exact_int(
            campaign.get("calibration_panel_count"), "campaign panel count", lower=1
        )
        or len(set(task_ids)) != len(task_ids)
        or any(type(item) is not str or not item for item in task_ids)
        or "sha256:" + canonical_digest(task_ids) != identity.get("task_ids_digest")
        or (
            scope is SkeletonGraphCalibrationScope.GENERIC_V3
            and "sha256:" + canonical_digest(panel_ids)
            != identity.get("panel_ids_digest")
        )
        or (
            scope is SkeletonGraphCalibrationScope.SAME_FAMILY
            and task_ids != tuple(prereg.SAME_FAMILY_TASK_IDS)
        )
    ):
        raise SkeletonGraphCalibrationRunnerError("derived campaign cohort differs")
    return task_ids, panel_ids


_EXPOSURE_PHASE = "fixed-skeleton-graph-calibration"
_EXPOSURE_ACTOR = "panel-action-count-skeleton-graph-calibration-runner"
_EXPOSURE_PURPOSE = "consume-exact-preregistered-calibration-cohort-once"


def _new_observed_at() -> str:
    """Internal clock edge used only when the fixed intent does not exist."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _sealed_task_ids(registration: Mapping[str, Any]) -> tuple[str, ...]:
    identity = registration.get("same_family_calibration", {}).get("identity_binding")
    if not isinstance(identity, Mapping):
        raise SkeletonGraphCalibrationRunnerError("sealed identity binding differs")
    values: list[str] = []
    for key in (
        "target_sealed_task_ids",
        "diagnostic_tainted_task_ids",
        "official_validation_sealed_task_ids",
    ):
        found = identity.get(key)
        if type(found) is not list or any(type(item) is not str for item in found):
            raise SkeletonGraphCalibrationRunnerError("sealed task inventory differs")
        values.extend(found)
    return tuple(values)


def _make_exposure_successor(
    *,
    predecessor: ExposureLedger,
    registration: Mapping[str, Any],
    task_ids: tuple[str, ...],
    panel_ids: tuple[str, ...],
    observed_at: str,
) -> ExposureLedger:
    try:
        successor = predecessor.record(
            phase=_EXPOSURE_PHASE,
            actor=_EXPOSURE_ACTOR,
            purpose=_EXPOSURE_PURPOSE,
            task_ids=task_ids,
            panel_ids=panel_ids,
            source="preregistration:" + PINNED_PREREGISTRATION_RECORD_DIGEST,
            observed_at=observed_at,
            known_task_ids=task_ids,
            known_panel_ids=panel_ids,
            sealed_task_ids=_sealed_task_ids(registration),
            require_unseen=True,
        )
    except Exception as exc:
        raise SkeletonGraphCalibrationRunnerError(
            "cannot append exact unseen calibration exposure event"
        ) from exc
    _verify_exposure_successor(
        predecessor=predecessor,
        successor=successor,
        task_ids=task_ids,
        panel_ids=panel_ids,
        observed_at=observed_at,
    )
    return successor


def _verify_exposure_successor(
    *,
    predecessor: ExposureLedger,
    successor: ExposureLedger,
    task_ids: tuple[str, ...],
    panel_ids: tuple[str, ...],
    observed_at: str,
) -> None:
    if (
        successor.corpus_digest != predecessor.corpus_digest
        or len(successor.events) != len(predecessor.events) + 1
        or successor.events[:-1] != predecessor.events
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "exposure successor is not exactly one predecessor append"
        )
    event = successor.events[-1]
    if (
        event.sequence != len(predecessor.events)
        or event.previous_digest
        != (predecessor.events[-1].digest if predecessor.events else None)
        or event.observed_at != observed_at
        or event.phase != _EXPOSURE_PHASE
        or event.actor != _EXPOSURE_ACTOR
        or event.purpose != _EXPOSURE_PURPOSE
        or event.task_ids != tuple(sorted(task_ids))
        or event.panel_ids != tuple(sorted(panel_ids))
        or event.source != "preregistration:" + PINNED_PREREGISTRATION_RECORD_DIGEST
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "exposure successor event differs from exact campaign"
        )


def _orphan_campaign_successor_names(
    authority: _CampaignAuthority,
    *,
    task_ids: tuple[str, ...],
    panel_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Find only direct children carrying this campaign's exact fixed event."""

    descriptor = authority.parent._open()
    matches: list[str] = []
    try:
        for name in os.listdir(descriptor):
            if name == authority.predecessor_filename or not name.endswith(
                ".exposure.json"
            ):
                continue
            try:
                raw = _read_dirfd_bytes(
                    descriptor,
                    name,
                    label="possible orphan campaign successor",
                    maximum=16 << 20,
                )
                candidate = _decode_exposure_ledger(
                    raw, label="possible orphan campaign successor"
                )
                _verify_exposure_successor(
                    predecessor=authority.predecessor,
                    successor=candidate,
                    task_ids=task_ids,
                    panel_ids=panel_ids,
                    observed_at=candidate.events[-1].observed_at,
                )
            except Exception:
                continue
            matches.append(name)
    finally:
        os.close(descriptor)
    return tuple(sorted(matches))


def _passed_fit_resolution(
    verified_fit: SkeletonGraphPassedFitProtocol,
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    return prereg.resolve_passed_fit_slot(
        registration["passed_fit_authority_slot"],
        outcome_schema=passed_fit_authority.PROTOCOL_SCHEMA,
        addresses={
            "passed_fit_authority_source_sha256": (
                verified_fit.passed_fit_authority_source_sha256
            ),
            "passed_fit_algorithm_digest": verified_fit.passed_fit_algorithm_digest,
            "passed_fit_record_digest": verified_fit.record_digest,
        },
    )


def _intent_body(
    *,
    scope: SkeletonGraphCalibrationScope,
    registration: Mapping[str, Any],
    passed_fit_resolution: Mapping[str, Any],
    authority: _CampaignAuthority,
    task_ids: tuple[str, ...],
    panel_ids: tuple[str, ...],
    output: Path,
    output_parent_identity: Mapping[str, object],
    successor: ExposureLedger,
) -> dict[str, Any]:
    event = successor.events[-1]
    successor_raw = successor.to_json().encode("utf-8")
    return {
        "schema": CAMPAIGN_ATTEMPT_AUTHORITY_SCHEMA,
        "intent_state": "prospective_exact_one_event_exposure_successor",
        "scope": scope.value,
        "population_scope": _scope_campaign(registration, scope)["population_scope"],
        "preregistration_record_digest": PINNED_PREREGISTRATION_RECORD_DIGEST,
        "preregistration_file_sha256": PINNED_PREREGISTRATION_FILE_SHA256,
        "preregistration_source_sha256": PINNED_PREREGISTRATION_SOURCE_SHA256,
        "passed_fit_resolution": dict(passed_fit_resolution),
        "task_ids": list(task_ids),
        "panel_ids": list(panel_ids),
        "task_ids_digest": "sha256:" + canonical_digest(task_ids),
        "panel_ids_digest": "sha256:" + canonical_digest(panel_ids),
        "intended_output_directory": str(output),
        "intended_output_parent_identity": dict(output_parent_identity),
        "exposure_predecessor_ledger_digest": authority.predecessor.digest,
        "exposure_predecessor_file_sha256": _file_address(authority.predecessor_raw),
        "exposure_predecessor_filename": authority.predecessor_filename,
        "prospective_exposure_event": event.to_dict(),
        "prospective_exposure_event_digest": event.digest,
        "prospective_exposure_successor_event_count": len(successor.events),
        "prospective_exposure_successor_ledger_digest": successor.digest,
        "prospective_exposure_successor_file_sha256": _file_address(successor_raw),
        "prospective_exposure_successor_filename": (
            successor.digest.removeprefix("sha256:") + ".exposure.json"
        ),
        "observed_at_generated_inside_fixed_intent": True,
        "attempt_number": 1,
        "reroll_or_alternate_output_root_authorized": False,
        "calibration_pixels_opened_before_intent": False,
        "runner_source_sha256": "sha256:" + source_sha256(),
        "runner_algorithm_digest": algorithm_digest(),
    }


def _reconstruct_intent_successor(
    intent: Mapping[str, Any],
    *,
    scope: SkeletonGraphCalibrationScope,
    registration: Mapping[str, Any],
    passed_fit_resolution: Mapping[str, Any],
    authority: _CampaignAuthority,
    task_ids: tuple[str, ...],
    panel_ids: tuple[str, ...],
    output: Path,
    output_parent_identity: Mapping[str, object],
) -> ExposureLedger:
    event = intent.get("prospective_exposure_event")
    if not isinstance(event, Mapping) or type(event.get("observed_at")) is not str:
        raise SkeletonGraphCalibrationRunnerError("campaign intent event differs")
    observed_at = event["observed_at"]
    successor = _make_exposure_successor(
        predecessor=authority.predecessor,
        registration=registration,
        task_ids=task_ids,
        panel_ids=panel_ids,
        observed_at=observed_at,
    )
    expected = _intent_body(
        scope=scope,
        registration=registration,
        passed_fit_resolution=passed_fit_resolution,
        authority=authority,
        task_ids=task_ids,
        panel_ids=panel_ids,
        output=output,
        output_parent_identity=output_parent_identity,
        successor=successor,
    )
    body = dict(intent)
    digest = body.pop("record_digest", None)
    if (
        not _typed_equal(body, expected)
        or digest != "sha256:" + canonical_digest(expected)
        or _exact_int(intent.get("attempt_number"), "campaign attempt number", lower=1)
        != 1
        or _exact_int(
            intent.get("prospective_exposure_successor_event_count"),
            "successor event count",
            lower=1,
        )
        != len(authority.predecessor.events) + 1
        or not _typed_equal(event, successor.events[-1].to_dict())
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "campaign intent differs from exact prospective successor"
        )
    return successor


def _load_or_create_intent(
    *,
    scope: SkeletonGraphCalibrationScope,
    registration: Mapping[str, Any],
    verified_fit: SkeletonGraphPassedFitProtocol,
    authority: _CampaignAuthority,
    task_ids: tuple[str, ...],
    panel_ids: tuple[str, ...],
    output: Path,
    output_parent_identity: Mapping[str, object],
) -> tuple[dict[str, Any], bytes, ExposureLedger]:
    passed_resolution = _passed_fit_resolution(verified_fit, registration)
    descriptor = authority.parent._open()
    try:
        try:
            found = os.stat(
                authority.intent_filename,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            final_exists = False
            raw = b""
        else:
            final_exists = True
            if not stat.S_ISREG(found.st_mode):
                raise SkeletonGraphCalibrationRunnerError(
                    "campaign intent entry type differs"
                )
            raw = _read_dirfd_bytes(
                descriptor, authority.intent_filename, label="campaign intent"
            )
        authorization_orphan = (
            not final_exists
            and any(
                name.endswith(".calibration-authorization.json")
                for name in os.listdir(descriptor)
            )
        )
    finally:
        os.close(descriptor)
    if not final_exists and (
        authorization_orphan
        or _orphan_campaign_successor_names(
            authority, task_ids=task_ids, panel_ids=panel_ids
        )
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "campaign intent is missing beside an issued authorization or successor"
        )
    if raw:
        intent = _record_from_bytes(
            raw,
            schema=CAMPAIGN_ATTEMPT_AUTHORITY_SCHEMA,
            label="campaign intent",
        )
        successor = _reconstruct_intent_successor(
            intent,
            scope=scope,
            registration=registration,
            passed_fit_resolution=passed_resolution,
            authority=authority,
            task_ids=task_ids,
            panel_ids=panel_ids,
            output=output,
            output_parent_identity=output_parent_identity,
        )
        return intent, raw, successor

    successor = _make_exposure_successor(
        predecessor=authority.predecessor,
        registration=registration,
        task_ids=task_ids,
        panel_ids=panel_ids,
        observed_at=_new_observed_at(),
    )
    intent = _seal(
        _intent_body(
            scope=scope,
            registration=registration,
            passed_fit_resolution=passed_resolution,
            authority=authority,
            task_ids=task_ids,
            panel_ids=panel_ids,
            output=output,
            output_parent_identity=output_parent_identity,
            successor=successor,
        )
    )
    raw = canonical_json(intent) + b"\n"
    loaded = _atomic_write_once_bytes(
        authority.parent,
        authority.intent_filename,
        raw,
        label="campaign intent",
        allow_identical_existing=False,
    )
    if loaded != raw:
        raise SkeletonGraphCalibrationRunnerError("campaign intent reload differs")
    return intent, raw, successor


def _persist_exposure_successor(
    authority: _CampaignAuthority, successor: ExposureLedger
) -> tuple[ExposureLedger, bytes, str]:
    name = successor.digest.removeprefix("sha256:") + ".exposure.json"
    raw = successor.to_json().encode("utf-8")
    loaded = _atomic_write_once_bytes(
        authority.parent,
        name,
        raw,
        label="exposure successor ledger",
        allow_identical_existing=True,
    )
    restored = _decode_exposure_ledger(loaded, label="exposure successor ledger")
    if (
        not _typed_equal(restored.to_dict(), successor.to_dict())
        or restored.digest != successor.digest
    ):
        raise SkeletonGraphCalibrationRunnerError("persisted exposure successor differs")
    return restored, loaded, name


def _authorization_values(
    *,
    scope: SkeletonGraphCalibrationScope,
    registration: Mapping[str, Any],
    passed_fit_authority_source_sha256: str,
    passed_fit_algorithm_digest: str,
    passed_fit_record_digest: str,
    authority: _CampaignAuthority,
    task_ids: tuple[str, ...],
    panel_ids: tuple[str, ...],
    output: Path,
    output_parent_identity: Mapping[str, object],
    intent: Mapping[str, Any],
    intent_raw: bytes,
    successor: ExposureLedger,
    successor_raw: bytes,
    successor_filename: str,
) -> dict[str, object]:
    event = successor.events[-1]
    return {
        "scope": scope,
        "population_scope": _scope_campaign(registration, scope)["population_scope"],
        "task_ids": task_ids,
        "panel_ids": panel_ids,
        "task_ids_digest": "sha256:" + canonical_digest(task_ids),
        "panel_ids_digest": "sha256:" + canonical_digest(panel_ids),
        "intended_output_directory": str(output),
        "intended_output_parent_path": output_parent_identity["absolute_path"],
        "intended_output_parent_st_dev": output_parent_identity["st_dev"],
        "intended_output_parent_st_ino": output_parent_identity["st_ino"],
        "intended_output_parent_st_mode": output_parent_identity["st_mode"],
        "preregistration_record_digest": PINNED_PREREGISTRATION_RECORD_DIGEST,
        "preregistration_file_sha256": PINNED_PREREGISTRATION_FILE_SHA256,
        "preregistration_source_sha256": PINNED_PREREGISTRATION_SOURCE_SHA256,
        "passed_fit_authority_source_sha256": (
            passed_fit_authority_source_sha256
        ),
        "passed_fit_algorithm_digest": passed_fit_algorithm_digest,
        "passed_fit_record_digest": passed_fit_record_digest,
        "exposure_predecessor_ledger_digest": authority.predecessor.digest,
        "exposure_predecessor_file_sha256": _file_address(authority.predecessor_raw),
        "exposure_predecessor_filename": authority.predecessor_filename,
        "exposure_event_digest": event.digest,
        "exposure_event_observed_at": event.observed_at,
        "exposure_successor_ledger_digest": successor.digest,
        "exposure_successor_file_sha256": _file_address(successor_raw),
        "exposure_successor_filename": successor_filename,
        "campaign_intent_record_digest": intent["record_digest"],
        "campaign_intent_file_sha256": _file_address(intent_raw),
        "campaign_intent_filename": authority.intent_filename,
        "calibration_pixels_authorized": True,
        "target_pixels_authorized": False,
        "query_pixels_authorized": False,
        "support_pixels_authorized": False,
        "official_test_pixels_authorized": False,
        "action_labels_or_programs_authorized": False,
        "authenticated_calibration_execution": False,
        "production_adapter_authorized": False,
        "runner_source_sha256": "sha256:" + source_sha256(),
        "runner_algorithm_digest": algorithm_digest(),
    }


def _persist_exposure_authorization(
    authority: _CampaignAuthority,
    authorization: SkeletonGraphCalibrationExposureAuthorization,
) -> bytes:
    raw = canonical_json(authorization.to_data()) + b"\n"
    loaded = _atomic_write_once_bytes(
        authority.parent,
        authorization.filename,
        raw,
        label="calibration exposure authorization",
        allow_identical_existing=True,
    )
    value = _record_from_bytes(
        loaded,
        schema=AUTHORIZATION_SCHEMA,
        label="calibration exposure authorization",
        expected_file_sha256=authorization.file_sha256,
        expected_record_digest=authorization.record_digest,
    )
    if not _typed_equal(value, authorization.to_data()):
        raise SkeletonGraphCalibrationRunnerError(
            "persisted calibration exposure authorization differs"
        )
    return loaded


def _authorization_from_data(
    value: object,
) -> SkeletonGraphCalibrationExposureAuthorization:
    expected = {
        "schema",
        "scope",
        "population_scope",
        "task_ids",
        "panel_ids",
        "task_ids_digest",
        "panel_ids_digest",
        "intended_output_directory",
        "intended_output_parent_path",
        "intended_output_parent_st_dev",
        "intended_output_parent_st_ino",
        "intended_output_parent_st_mode",
        "preregistration_record_digest",
        "preregistration_file_sha256",
        "preregistration_source_sha256",
        "passed_fit_authority_source_sha256",
        "passed_fit_algorithm_digest",
        "passed_fit_record_digest",
        "exposure_predecessor_ledger_digest",
        "exposure_predecessor_file_sha256",
        "exposure_predecessor_filename",
        "exposure_event_digest",
        "exposure_event_observed_at",
        "exposure_successor_ledger_digest",
        "exposure_successor_file_sha256",
        "exposure_successor_filename",
        "campaign_intent_record_digest",
        "campaign_intent_file_sha256",
        "campaign_intent_filename",
        "calibration_pixels_authorized",
        "target_pixels_authorized",
        "query_pixels_authorized",
        "support_pixels_authorized",
        "official_test_pixels_authorized",
        "action_labels_or_programs_authorized",
        "authenticated_calibration_execution",
        "production_adapter_authorized",
        "runner_source_sha256",
        "runner_algorithm_digest",
        "metadata_only_authorization",
        "real_exposure_successor_fsync_reloaded_before_issuance",
        "record_digest",
    }
    raw = _fields(value, expected, "calibration exposure authorization")
    if (
        raw["schema"] != AUTHORIZATION_SCHEMA
        or type(raw["task_ids"]) is not list
        or type(raw["panel_ids"]) is not list
        or raw["metadata_only_authorization"] is not True
        or raw["real_exposure_successor_fsync_reloaded_before_issuance"] is not True
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "calibration exposure authorization wire differs"
        )
    try:
        scope = SkeletonGraphCalibrationScope(raw["scope"])
    except (TypeError, ValueError) as exc:
        raise SkeletonGraphCalibrationRunnerError("authorization scope differs") from exc
    values = {
        name: raw[name]
        for name in SkeletonGraphCalibrationExposureAuthorization.__dataclass_fields__
        if name != "record_digest"
    }
    values["scope"] = scope
    values["task_ids"] = tuple(raw["task_ids"])
    values["panel_ids"] = tuple(raw["panel_ids"])
    restored = SkeletonGraphCalibrationExposureAuthorization._issue(
        issuance_token=_EXPOSURE_AUTHORIZATION_ISSUANCE_TOKEN,
        values=values,
    )
    if not _typed_equal(restored.to_data(), dict(raw)):
        raise SkeletonGraphCalibrationRunnerError(
            "calibration exposure authorization is not canonical"
        )
    return restored


def authorize_calibration_exposure(
    *,
    scope: SkeletonGraphCalibrationScope,
    preregistration_path: Path,
    passed_fit: SkeletonGraphPassedFitOutcome,
    passed_fit_paths: SkeletonGraphPassedFitPaths,
    output_directory: Path,
) -> SkeletonGraphCalibrationExposureAuthorization | SkeletonGraphCalibrationGap:
    """Claim one cohort and persist its real exposure successor without pixels."""

    if type(scope) is not SkeletonGraphCalibrationScope:
        raise TypeError("calibration scope must have exact enum type")
    registration = _load_preregistration(Path(preregistration_path))
    verified = _verify_passed_fit_outcome(passed_fit, passed_fit_paths)
    if type(verified) is SkeletonGraphPassedFitGap:
        return SkeletonGraphCalibrationGap.create(
            scope=scope,
            stage="passed_fit_precommit",
            reason_codes=("passed_fit_gap",),
            passed_fit_record_digest=verified.record_digest,
        )
    if type(verified) is not SkeletonGraphPassedFitProtocol:
        raise SkeletonGraphCalibrationRunnerError("passed-fit outcome differs")
    task_ids, panel_ids = _derive_cohort_ids(registration, scope)
    output, output_parent = _validate_intended_output_directory(
        Path(output_directory)
    )
    try:
        output_parent_identity = output_parent.identity_data()
    finally:
        output_parent.close()
    authority = _campaign_attempt_authority(registration, scope)
    try:
        intent, intent_raw, successor = _load_or_create_intent(
            scope=scope,
            registration=registration,
            verified_fit=verified,
            authority=authority,
            task_ids=task_ids,
            panel_ids=panel_ids,
            output=output,
            output_parent_identity=output_parent_identity,
        )
        successor, successor_raw, successor_name = _persist_exposure_successor(
            authority, successor
        )
        authorization = SkeletonGraphCalibrationExposureAuthorization._issue(
            issuance_token=_EXPOSURE_AUTHORIZATION_ISSUANCE_TOKEN,
            values=_authorization_values(
                scope=scope,
                registration=registration,
                passed_fit_authority_source_sha256=(
                    verified.passed_fit_authority_source_sha256
                ),
                passed_fit_algorithm_digest=verified.passed_fit_algorithm_digest,
                passed_fit_record_digest=verified.record_digest,
                authority=authority,
                task_ids=task_ids,
                panel_ids=panel_ids,
                output=output,
                output_parent_identity=output_parent_identity,
                intent=intent,
                intent_raw=intent_raw,
                successor=successor,
                successor_raw=successor_raw,
                successor_filename=successor_name,
            ),
        )
        _persist_exposure_authorization(authority, authorization)
    finally:
        authority.close()
    return verify_calibration_exposure_authorization(
        authorization,
        preregistration_path=preregistration_path,
        passed_fit=passed_fit,
        passed_fit_paths=passed_fit_paths,
        expected_record_digest=authorization.record_digest,
        expected_file_sha256=authorization.file_sha256,
    )


def _verify_authorized_output_parent(
    authorization: SkeletonGraphCalibrationExposureAuthorization,
) -> dict[str, object]:
    parent = _open_existing_directory(
        Path(authorization.intended_output_parent_path),
        label="authorized output parent",
    )
    try:
        found = parent.identity_data()
    finally:
        parent.close()
    expected = {
        "absolute_path": authorization.intended_output_parent_path,
        "st_dev": authorization.intended_output_parent_st_dev,
        "st_ino": authorization.intended_output_parent_st_ino,
        "st_mode": authorization.intended_output_parent_st_mode,
    }
    if found != expected:
        raise SkeletonGraphCalibrationRunnerError(
            "authorized output parent identity differs"
        )
    return expected


def verify_calibration_exposure_authorization(
    authorization: SkeletonGraphCalibrationExposureAuthorization,
    *,
    preregistration_path: Path,
    passed_fit: SkeletonGraphPassedFitOutcome,
    passed_fit_paths: SkeletonGraphPassedFitPaths,
    expected_record_digest: str | None = None,
    expected_file_sha256: str | None = None,
) -> SkeletonGraphCalibrationExposureAuthorization:
    """Freshly verify intent, exact ledger child, and authorization file."""

    if type(authorization) is not SkeletonGraphCalibrationExposureAuthorization:
        raise TypeError("exposure authorization must have exact type")
    authorization._validate()
    if (
        expected_record_digest is not None
        and authorization.record_digest
        != _address(expected_record_digest, "expected authorization record")
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "expected authorization record differs"
        )
    if (
        expected_file_sha256 is not None
        and authorization.file_sha256
        != _address(expected_file_sha256, "expected authorization file")
    ):
        raise SkeletonGraphCalibrationRunnerError("expected authorization file differs")
    registration = _load_preregistration(Path(preregistration_path))
    verified = _verify_passed_fit_outcome(passed_fit, passed_fit_paths)
    if type(verified) is not SkeletonGraphPassedFitProtocol:
        raise SkeletonGraphCalibrationRunnerError(
            "exposure authorization requires exact passed-fit protocol"
        )
    scope = authorization.scope
    task_ids, panel_ids = _derive_cohort_ids(registration, scope)
    output = Path(authorization.intended_output_directory)
    if not output.is_absolute() or output.parent != Path(
        authorization.intended_output_parent_path
    ):
        raise SkeletonGraphCalibrationRunnerError("authorized output path differs")
    output_parent_identity = _verify_authorized_output_parent(authorization)
    authority = _campaign_attempt_authority(registration, scope)
    try:
        intent, intent_raw = _read_output_record(
            authority.parent,
            authority.intent_filename,
            schema=CAMPAIGN_ATTEMPT_AUTHORITY_SCHEMA,
            label="campaign attempt intent",
            expected_file_sha256=authorization.campaign_intent_file_sha256,
            expected_record_digest=authorization.campaign_intent_record_digest,
        )
        successor = _reconstruct_intent_successor(
            intent,
            scope=scope,
            registration=registration,
            passed_fit_resolution=_passed_fit_resolution(verified, registration),
            authority=authority,
            task_ids=task_ids,
            panel_ids=panel_ids,
            output=output,
            output_parent_identity=output_parent_identity,
        )
        descriptor = authority.parent._open()
        try:
            successor_raw = _read_dirfd_bytes(
                descriptor,
                authorization.exposure_successor_filename,
                label="authorized exposure successor",
                maximum=16 << 20,
            )
            authorization_raw = _read_dirfd_bytes(
                descriptor,
                authorization.filename,
                label="persisted exposure authorization",
            )
        finally:
            os.close(descriptor)
        restored_successor = _decode_exposure_ledger(
            successor_raw, label="authorized exposure successor"
        )
        _verify_exposure_successor(
            predecessor=authority.predecessor,
            successor=restored_successor,
            task_ids=task_ids,
            panel_ids=panel_ids,
            observed_at=successor.events[-1].observed_at,
        )
        if (
            not _typed_equal(restored_successor.to_dict(), successor.to_dict())
            or restored_successor.digest != authorization.exposure_successor_ledger_digest
            or _file_address(successor_raw)
            != authorization.exposure_successor_file_sha256
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "authorized exposure successor differs"
            )
        expected_authorization = SkeletonGraphCalibrationExposureAuthorization._issue(
            issuance_token=_EXPOSURE_AUTHORIZATION_ISSUANCE_TOKEN,
            values=_authorization_values(
                scope=scope,
                registration=registration,
                passed_fit_authority_source_sha256=(
                    verified.passed_fit_authority_source_sha256
                ),
                passed_fit_algorithm_digest=verified.passed_fit_algorithm_digest,
                passed_fit_record_digest=verified.record_digest,
                authority=authority,
                task_ids=task_ids,
                panel_ids=panel_ids,
                output=output,
                output_parent_identity=output_parent_identity,
                intent=intent,
                intent_raw=intent_raw,
                successor=restored_successor,
                successor_raw=successor_raw,
                successor_filename=authorization.exposure_successor_filename,
            ),
        )
        persisted = _record_from_bytes(
            authorization_raw,
            schema=AUTHORIZATION_SCHEMA,
            label="persisted exposure authorization",
            expected_file_sha256=authorization.file_sha256,
            expected_record_digest=authorization.record_digest,
        )
        if (
            not _typed_equal(
                expected_authorization.to_data(), authorization.to_data()
            )
            or not _typed_equal(persisted, authorization.to_data())
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "exposure authorization fresh verification differs"
            )
        return authorization
    finally:
        authority.close()


def _pinned_passed_fit_resolution(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    return prereg.resolve_passed_fit_slot(
        registration["passed_fit_authority_slot"],
        outcome_schema=passed_fit_authority.PROTOCOL_SCHEMA,
        addresses={
            "passed_fit_authority_source_sha256": (
                "sha256:" + PINNED_PASSED_FIT_SOURCE_SHA256
            ),
            "passed_fit_algorithm_digest": PINNED_PASSED_FIT_ALGORITHM_DIGEST,
            "passed_fit_record_digest": PINNED_PASSED_FIT_PROTOCOL_RECORD_DIGEST,
        },
    )


def _verify_archived_exposure_authorization(
    authorization: SkeletonGraphCalibrationExposureAuthorization,
    *,
    registration: Mapping[str, Any],
) -> SkeletonGraphCalibrationExposureAuthorization:
    """Replay the persisted metadata chain without opening model artifacts."""

    authorization._validate()
    task_ids, panel_ids = _derive_cohort_ids(registration, authorization.scope)
    output = Path(authorization.intended_output_directory)
    output_parent_identity = _verify_authorized_output_parent(authorization)
    authority = _campaign_attempt_authority(registration, authorization.scope)
    try:
        intent, intent_raw = _read_output_record(
            authority.parent,
            authority.intent_filename,
            schema=CAMPAIGN_ATTEMPT_AUTHORITY_SCHEMA,
            label="archived campaign intent",
            expected_file_sha256=authorization.campaign_intent_file_sha256,
            expected_record_digest=authorization.campaign_intent_record_digest,
        )
        successor = _reconstruct_intent_successor(
            intent,
            scope=authorization.scope,
            registration=registration,
            passed_fit_resolution=_pinned_passed_fit_resolution(registration),
            authority=authority,
            task_ids=task_ids,
            panel_ids=panel_ids,
            output=output,
            output_parent_identity=output_parent_identity,
        )
        descriptor = authority.parent._open()
        try:
            successor_raw = _read_dirfd_bytes(
                descriptor,
                authorization.exposure_successor_filename,
                label="archived exposure successor",
                maximum=16 << 20,
            )
            authorization_raw = _read_dirfd_bytes(
                descriptor,
                authorization.filename,
                label="archived exposure authorization",
            )
        finally:
            os.close(descriptor)
        restored = _decode_exposure_ledger(
            successor_raw, label="archived exposure successor"
        )
        _verify_exposure_successor(
            predecessor=authority.predecessor,
            successor=restored,
            task_ids=task_ids,
            panel_ids=panel_ids,
            observed_at=successor.events[-1].observed_at,
        )
        expected = SkeletonGraphCalibrationExposureAuthorization._issue(
            issuance_token=_EXPOSURE_AUTHORIZATION_ISSUANCE_TOKEN,
            values=_authorization_values(
                scope=authorization.scope,
                registration=registration,
                passed_fit_authority_source_sha256=(
                    "sha256:" + PINNED_PASSED_FIT_SOURCE_SHA256
                ),
                passed_fit_algorithm_digest=PINNED_PASSED_FIT_ALGORITHM_DIGEST,
                passed_fit_record_digest=PINNED_PASSED_FIT_PROTOCOL_RECORD_DIGEST,
                authority=authority,
                task_ids=task_ids,
                panel_ids=panel_ids,
                output=output,
                output_parent_identity=output_parent_identity,
                intent=intent,
                intent_raw=intent_raw,
                successor=restored,
                successor_raw=successor_raw,
                successor_filename=authorization.exposure_successor_filename,
            ),
        )
        persisted = _record_from_bytes(
            authorization_raw,
            schema=AUTHORIZATION_SCHEMA,
            label="archived exposure authorization",
            expected_file_sha256=authorization.file_sha256,
            expected_record_digest=authorization.record_digest,
        )
        if (
            not _typed_equal(restored.to_dict(), successor.to_dict())
            or not _typed_equal(expected.to_data(), authorization.to_data())
            or not _typed_equal(persisted, authorization.to_data())
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "archived exposure authorization chain differs"
            )
        return authorization
    finally:
        authority.close()


def _validate_panel_identities(
    registration: Mapping[str, Any],
    scope: SkeletonGraphCalibrationScope,
    values: Sequence[SkeletonGraphCalibrationPanelIdentity],
) -> tuple[tuple[SkeletonGraphCalibrationPanelIdentity, ...], tuple[str, ...]]:
    if isinstance(values, (str, bytes)):
        raise TypeError("panel identities must be a sequence")
    identities = tuple(values)
    campaign = _scope_campaign(registration, scope)
    count = campaign.get("calibration_panel_count")
    if (
        len(identities) != count
        or len(identities) > MAX_INPUT_OCCURRENCES
        or any(type(item) is not SkeletonGraphCalibrationPanelIdentity for item in identities)
        or len({item.panel_id for item in identities}) != len(identities)
    ):
        raise SkeletonGraphCalibrationRunnerError("calibration panel inventory differs")
    panel_ids = tuple(item.panel_id for item in identities)
    task_ids = tuple(dict.fromkeys(item.task_id for item in identities))
    expected_panels = tuple(
        f"hd/{task_id}/{side}/{ordinal}.png"
        for task_id in task_ids
        for side in (1, 0)
        for ordinal in range(7)
    )
    if (
        panel_ids != expected_panels
        or len(task_ids) != campaign.get("calibration_task_count")
    ):
        raise SkeletonGraphCalibrationRunnerError("calibration task/panel order differs")
    identity = campaign.get("identity_binding")
    if not isinstance(identity, Mapping):
        raise SkeletonGraphCalibrationRunnerError("campaign identity binding differs")
    if (
        (
            scope is SkeletonGraphCalibrationScope.GENERIC_V3
            and "sha256:" + canonical_digest(panel_ids)
            != identity.get("panel_ids_digest")
        )
        or "sha256:" + canonical_digest(task_ids) != identity.get("task_ids_digest")
    ):
        raise SkeletonGraphCalibrationRunnerError("campaign identity digest differs")
    if scope is SkeletonGraphCalibrationScope.SAME_FAMILY and (
        task_ids != tuple(prereg.SAME_FAMILY_TASK_IDS)
        or any(
            sealed in task_ids
            for sealed in (
                SAME_FAMILY_TARGET_TASK_ID,
                "hd_convex-has_four_straight_lines_0001",
                "hd_convex-has_four_straight_lines_0018",
                "hd_convex-has_four_straight_lines_0019",
            )
        )
    ):
        raise SkeletonGraphCalibrationRunnerError("same-family seals differ")
    return identities, task_ids


def _anonymous_tokens(
    scope: SkeletonGraphCalibrationScope,
    identities: Sequence[SkeletonGraphCalibrationPanelIdentity],
) -> tuple[str, ...]:
    return tuple(
        "anon_"
        + canonical_digest(
            {
                "schema": "gkm.bongard-skeleton-graph-anonymous-panel-token.v1",
                "preregistration_record_digest": PINNED_PREREGISTRATION_RECORD_DIGEST,
                "scope": scope.value,
                "occurrence_index": index,
                "png_sha256": identity.png_sha256,
                "png_size": identity.png_size,
            }
        )
        for index, identity in enumerate(identities)
    )


def _read_calibration_payloads(
    identities: Sequence[SkeletonGraphCalibrationPanelIdentity],
    reader: SkeletonGraphCalibrationPixelReader,
) -> tuple[bytes, ...]:
    if not callable(reader):
        raise TypeError("calibration pixel reader must be callable")
    result: list[bytes] = []
    for identity in identities:
        payload = reader(identity.panel_id)
        if type(payload) is not bytes or not payload:
            raise SkeletonGraphCalibrationRunnerError("calibration pixel reader failed")
        if len(payload) != identity.png_size or _file_address(payload) != identity.png_sha256:
            raise SkeletonGraphCalibrationRunnerError("calibration PNG identity differs")
        result.append(payload)
    return tuple(result)


def _frozen_inference_addresses(
    batch: SkeletonGraphRawInferenceBatch,
    receipt: SkeletonGraphInferenceRecomputeReceipt,
) -> SkeletonGraphFrozenInferenceAddresses:
    return SkeletonGraphFrozenInferenceAddresses(
        raw_batch_file_sha256=_file_address(batch.to_bytes()),
        raw_batch_record_digest=batch.record_digest,
        recompute_receipt_file_sha256=_file_address(receipt.to_bytes()),
        recompute_receipt_record_digest=receipt.record_digest,
    )


def _validate_inference_pair(
    value: object,
    *,
    passed_fit: SkeletonGraphPassedFitProtocol,
    identities: Sequence[SkeletonGraphCalibrationPanelIdentity],
    tokens: Sequence[str],
) -> tuple[
    SkeletonGraphRawInferenceBatch,
    SkeletonGraphInferenceRecomputeReceipt,
    SkeletonGraphFrozenInferenceAddresses,
    list[dict[str, Any]],
]:
    if type(value) is not tuple or len(value) != 2:
        raise SkeletonGraphCalibrationRunnerError("inference callback result differs")
    batch, receipt = value
    if (
        type(batch) is not SkeletonGraphRawInferenceBatch
        or type(receipt) is not SkeletonGraphInferenceRecomputeReceipt
    ):
        raise SkeletonGraphCalibrationRunnerError("inference callback types differ")
    batch = SkeletonGraphRawInferenceBatch.from_data(batch.to_data())
    receipt = SkeletonGraphInferenceRecomputeReceipt.from_data(receipt.to_data())
    frozen = _frozen_inference_addresses(batch, receipt)
    cold_replay_raw_inference(
        raw_batch_bytes=batch.to_bytes(),
        recompute_receipt_bytes=receipt.to_bytes(),
        expected_raw_batch_file_sha256=frozen.raw_batch_file_sha256,
        expected_raw_batch_record_digest=frozen.raw_batch_record_digest,
        expected_recompute_receipt_file_sha256=frozen.recompute_receipt_file_sha256,
        expected_recompute_receipt_record_digest=frozen.recompute_receipt_record_digest,
    )
    expected_counts = Counter(item.png_sha256 for item in identities)
    expected_sizes = {item.png_sha256: item.png_size for item in identities}
    rows = {row.png_sha256: row for row in batch.rows}
    if (
        batch.input_occurrence_count != len(identities)
        or batch.input_occurrence_count > MAX_INPUT_OCCURRENCES
        or set(rows) != set(expected_counts)
        or batch.passed_fit_protocol_record_digest != passed_fit.record_digest
        or batch.passed_fit_authority_source_sha256
        != passed_fit.passed_fit_authority_source_sha256
        or batch.passed_fit_algorithm_digest != passed_fit.passed_fit_algorithm_digest
    ):
        raise SkeletonGraphCalibrationRunnerError("raw inference scope/authority differs")
    for digest, row in rows.items():
        if (
            row.occurrence_count != expected_counts[digest]
            or row.png_size_bytes != expected_sizes[digest]
        ):
            raise SkeletonGraphCalibrationRunnerError("deduplicated occurrence join differs")
    occurrences: list[dict[str, Any]] = []
    for token, identity in zip(tokens, identities, strict=True):
        row = rows[identity.png_sha256]
        occurrences.append(
            {
                "anonymous_panel_token": token,
                "png_sha256": identity.png_sha256,
                "png_size": identity.png_size,
                "feature_vector_sha256": row.feature_digest,
                "direct_pair_probabilities_33": list(row.direct_pair_probabilities),
                "catalog_probabilities_3": list(row.catalog_probabilities),
                "raw_inference_row_record_digest": row.record_digest,
            }
        )
    return batch, receipt, frozen, occurrences


def _validate_prediction_data(
    value: Mapping[str, Any],
    *,
    identities: Sequence[SkeletonGraphCalibrationPanelIdentity],
    tokens: Sequence[str],
) -> tuple[
    SkeletonGraphRawInferenceBatch,
    SkeletonGraphInferenceRecomputeReceipt,
    SkeletonGraphFrozenInferenceAddresses,
    tuple[Mapping[str, Any], ...],
]:
    expected = {
        "schema", "scope", "precommit_record_digest", "precommit_file_sha256",
        "raw_inference_batch", "raw_inference_batch_file_sha256",
        "raw_inference_batch_record_digest", "recompute_receipt",
        "recompute_receipt_file_sha256", "recompute_receipt_record_digest",
        "occurrences", "occurrence_count", "occurrence_join_digest",
        "raw_custody_contains_anonymous_tokens", "task_or_role_entered_inference",
        "file_fsync_completed", "directory_fsync_completed", "record_digest",
    }
    raw = _fields(value, expected, "raw prediction artifact")
    if (
        raw["schema"] != PREDICTION_SCHEMA
        or type(raw["occurrence_count"]) is not int
        or raw["occurrence_count"] != len(identities)
        or raw["occurrence_count"] > MAX_INPUT_OCCURRENCES
        or type(raw["occurrences"]) is not list
        or raw["raw_custody_contains_anonymous_tokens"] is not False
        or raw["task_or_role_entered_inference"] is not False
        or raw["file_fsync_completed"] is not True
        or raw["directory_fsync_completed"] is not True
    ):
        raise SkeletonGraphCalibrationRunnerError("raw prediction policy differs")
    batch = SkeletonGraphRawInferenceBatch.from_data(raw["raw_inference_batch"])
    receipt = SkeletonGraphInferenceRecomputeReceipt.from_data(raw["recompute_receipt"])
    frozen = SkeletonGraphFrozenInferenceAddresses(
        raw_batch_file_sha256=raw["raw_inference_batch_file_sha256"],
        raw_batch_record_digest=raw["raw_inference_batch_record_digest"],
        recompute_receipt_file_sha256=raw["recompute_receipt_file_sha256"],
        recompute_receipt_record_digest=raw["recompute_receipt_record_digest"],
    )
    if frozen != _frozen_inference_addresses(batch, receipt):
        raise SkeletonGraphCalibrationRunnerError("frozen inference addresses differ")
    cold_replay_raw_inference(
        raw_batch_bytes=batch.to_bytes(),
        recompute_receipt_bytes=receipt.to_bytes(),
        expected_raw_batch_file_sha256=frozen.raw_batch_file_sha256,
        expected_raw_batch_record_digest=frozen.raw_batch_record_digest,
        expected_recompute_receipt_file_sha256=frozen.recompute_receipt_file_sha256,
        expected_recompute_receipt_record_digest=frozen.recompute_receipt_record_digest,
    )
    by_digest = {row.png_sha256: row for row in batch.rows}
    occurrences = tuple(raw["occurrences"])
    if len(occurrences) != len(identities):
        raise SkeletonGraphCalibrationRunnerError("prediction occurrence count differs")
    for occurrence, token, identity in zip(occurrences, tokens, identities, strict=True):
        row = by_digest.get(identity.png_sha256)
        expected_occurrence = None if row is None else {
            "anonymous_panel_token": token,
            "png_sha256": identity.png_sha256,
            "png_size": identity.png_size,
            "feature_vector_sha256": row.feature_digest,
            "direct_pair_probabilities_33": list(row.direct_pair_probabilities),
            "catalog_probabilities_3": list(row.catalog_probabilities),
            "raw_inference_row_record_digest": row.record_digest,
        }
        if occurrence != expected_occurrence:
            raise SkeletonGraphCalibrationRunnerError("outer occurrence translation differs")
    join_digest = "sha256:" + canonical_digest(occurrences)
    if raw["occurrence_join_digest"] != join_digest:
        raise SkeletonGraphCalibrationRunnerError("occurrence join digest differs")
    return batch, receipt, frozen, occurrences


def _validate_label_batch(
    batch: SkeletonGraphDelayedLabelBatch,
    *,
    request: SkeletonGraphDelayedLabelRequest,
) -> SkeletonGraphDelayedLabelBatch:
    if type(batch) is not SkeletonGraphDelayedLabelBatch:
        raise SkeletonGraphCalibrationRunnerError("label reader result type differs")
    restored = SkeletonGraphDelayedLabelBatch.from_data(batch.to_data())
    if (
        restored.delayed_label_request_record_digest != request.record_digest
        or restored.label_attempt_record_digest != request.label_attempt_record_digest
        or restored.prediction_record_digest != request.prediction_record_digest
        or restored.prediction_file_sha256 != request.prediction_file_sha256
        or len(restored.rows) != len(request.bindings)
    ):
        raise SkeletonGraphCalibrationRunnerError("delayed label batch join differs")
    for row, (token, identity) in zip(restored.rows, request.bindings, strict=True):
        if (
            row.anonymous_panel_token != token
            or row.panel_id != identity.panel_id
            or row.task_id != identity.task_id
            or row.side != identity.side
            or row.ordinal != identity.ordinal
        ):
            raise SkeletonGraphCalibrationRunnerError("delayed label row inventory differs")
    return restored


def _class_sets(
    occurrence: Mapping[str, Any], q: float
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    pair_probabilities = _probability_vector(
        occurrence["direct_pair_probabilities_33"], 33, "direct pair"
    )
    catalog_probabilities = _probability_vector(
        occurrence["catalog_probabilities_3"], 3, "catalog"
    )
    observed = dict(zip(prereg.OBSERVED_PAIR_CODES, pair_probabilities, strict=True))
    pair_set = tuple(
        code for code in prereg.VALID_PAIR_CODES
        if 1.0 - observed.get(code, 0.0) <= q
    )
    catalog_set = tuple(
        code for code, probability in zip(
            prereg.CATALOG_CLASS_ORDER, catalog_probabilities, strict=True
        )
        if 1.0 - probability <= q
    )
    return pair_set, catalog_set


def _atom_disposition(candidates: tuple[int, ...], target: int, *, catalog: bool) -> str:
    if not candidates:
        return "error"
    if catalog and -1 in candidates:
        return "indeterminate"
    if candidates == (target,):
        return "present"
    if target not in candidates:
        return "certified_absent"
    return "indeterminate"


def _conjunction_disposition(first: str, second: str) -> str:
    if "error" in (first, second):
        return "error"
    if "certified_absent" in (first, second):
        return "certified_absent"
    if first == second == "present":
        return "present"
    return "indeterminate"


def _evaluate_archive(
    *,
    scope: SkeletonGraphCalibrationScope,
    task_ids: Sequence[str],
    occurrences: Sequence[Mapping[str, Any]],
    labels: SkeletonGraphDelayedLabelBatch,
) -> tuple[float, tuple[float, ...], dict[str, Any], tuple[str, ...]]:
    occurrence_by_token = {
        item["anonymous_panel_token"]: item for item in occurrences
    }
    scores: dict[str, list[float]] = defaultdict(list)
    for label in labels.rows:
        occurrence = occurrence_by_token.get(label.anonymous_panel_token)
        if occurrence is None:
            raise SkeletonGraphCalibrationRunnerError("label lacks raw occurrence")
        pair_probabilities = _probability_vector(
            occurrence["direct_pair_probabilities_33"], 33, "direct pair"
        )
        catalog_probabilities = _probability_vector(
            occurrence["catalog_probabilities_3"], 3, "catalog"
        )
        true_pair = 10 * label.true_straight_action_count + label.true_arc_action_count
        if true_pair not in prereg.VALID_PAIR_CODES:
            raise SkeletonGraphCalibrationRunnerError("invalid true pair")
        pair_by_code = dict(zip(prereg.OBSERVED_PAIR_CODES, pair_probabilities, strict=True))
        catalog_by_code = dict(
            zip(prereg.CATALOG_CLASS_ORDER, catalog_probabilities, strict=True)
        )
        scores[label.task_id].append(1.0 - pair_by_code.get(true_pair, 0.0))
        scores[label.task_id].append(1.0 - catalog_by_code[label.true_catalog_class])
    if set(scores) != set(task_ids) or any(len(scores[item]) != 28 for item in task_ids):
        raise SkeletonGraphCalibrationRunnerError("whole-task score inventory differs")
    task_scores = tuple(max(scores[item]) for item in task_ids)
    k = 96 if scope is SkeletonGraphCalibrationScope.GENERIC_V3 else 16
    q = sorted(task_scores)[k - 1]
    projections: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
        item["anonymous_panel_token"]: _class_sets(item, q) for item in occurrences
    }
    projection_error_count = sum(
        not pair_set or not catalog_set
        for pair_set, catalog_set in projections.values()
    )
    if scope is SkeletonGraphCalibrationScope.GENERIC_V3:
        gate = {
            "applicable": False,
            "all_fixed_checks_passed": None,
            "projection_error_count": projection_error_count,
        }
        reasons = ("empty_conformal_class_set",) if projection_error_count else ()
        return q, task_scores, gate, reasons

    straight_sizes: list[int] = []
    true_four_count = 0
    true_four_singleton_count = 0
    catalog_decisive_count = 0
    formula_rows: dict[str, dict[int, list[str]]] = {
        task_id: {1: [], 0: []} for task_id in task_ids
    }
    formula_or_cell_error_count = projection_error_count
    for label in labels.rows:
        pair_set, catalog_set = projections[label.anonymous_panel_token]
        straight_set = tuple(sorted({code // 10 for code in pair_set}))
        straight_sizes.append(len(straight_set))
        if label.true_straight_action_count == 4:
            true_four_count += 1
            true_four_singleton_count += straight_set == (4,)
        catalog_decisive_count += catalog_set in ((0,), (1,))
        if label.ordinal == 4:
            continue
        straight_state = _atom_disposition(straight_set, 4, catalog=False)
        catalog_state = _atom_disposition(catalog_set, 1, catalog=True)
        formula_state = _conjunction_disposition(straight_state, catalog_state)
        formula_rows[label.task_id][label.side].append(formula_state)
        formula_or_cell_error_count += formula_state == "error"
    mean_straight_size = sum(straight_sizes) / len(straight_sizes)
    singleton_fraction = (
        true_four_singleton_count / true_four_count if true_four_count else 0.0
    )
    catalog_decisive_fraction = catalog_decisive_count / len(labels.rows)
    task_audits: list[dict[str, Any]] = []
    admitted_count = 0
    for task_id in task_ids:
        primary = Counter(formula_rows[task_id][1])
        contrast = Counter(formula_rows[task_id][0])
        if len(formula_rows[task_id][1]) != 6 or len(formula_rows[task_id][0]) != 6:
            raise SkeletonGraphCalibrationRunnerError("fixed support role inventory differs")
        admitted = (
            primary["present"] >= 5
            and primary["certified_absent"] == 0
            and primary["indeterminate"] <= 1
            and primary["error"] == 0
            and contrast["certified_absent"] >= 5
            and contrast["present"] == 0
            and contrast["indeterminate"] <= 1
            and contrast["error"] == 0
        )
        admitted_count += admitted
        task_audits.append(
            {
                "task_id": task_id,
                "primary": {name: primary[name] for name in (
                    "present", "certified_absent", "indeterminate", "error"
                )},
                "contrast": {name: contrast[name] for name in (
                    "present", "certified_absent", "indeterminate", "error"
                )},
                "admitted": admitted,
            }
        )
    checks = {
        "straight_mean_class_set_size_at_most_4": mean_straight_size <= 4.0,
        "straight_count_4_singleton_fraction_at_least_point_25": (
            true_four_count > 0 and singleton_fraction >= 0.25
        ),
        "catalog_typed_decisive_fraction_at_least_point_30": (
            catalog_decisive_fraction >= 0.30
        ),
        "formula_admitted_task_count_at_least_14": admitted_count >= 14,
        "formula_or_cell_error_count_equals_0": formula_or_cell_error_count == 0,
    }
    passed = all(checks.values())
    gate = {
        "applicable": True,
        "policy": {
            "fixed_formula": [
                ["straight_action_count", 4],
                ["catalog_convexity", "catalog_convex"],
            ],
            "support_ordinals": [0, 1, 2, 3, 5, 6],
            "primary_side": 1,
            "contrast_side": 0,
            "conjunction_precedence": [
                "error", "certified_absent", "all_present", "indeterminate"
            ],
            "global_q_only": True,
        },
        "straight_mean_class_set_size": mean_straight_size,
        "true_straight_4_panel_count": true_four_count,
        "true_straight_4_singleton_count": true_four_singleton_count,
        "straight_count_4_singleton_fraction": singleton_fraction,
        "catalog_typed_decisive_panel_count": catalog_decisive_count,
        "catalog_typed_decisive_panel_fraction": catalog_decisive_fraction,
        "formula_admitted_task_count": admitted_count,
        "formula_admitted_task_denominator": len(task_ids),
        "formula_or_cell_error_count": formula_or_cell_error_count,
        "task_audits": task_audits,
        "fixed_checks": checks,
        "all_fixed_checks_passed": passed,
    }
    reasons = tuple(name for name, passed_check in checks.items() if not passed_check)
    return q, task_scores, gate, reasons


def _persist_terminal_integrity_gap(
    output: _OutputDirectoryCustody,
    *,
    authorization: SkeletonGraphCalibrationExposureAuthorization,
    stage: str,
    output_root_claim: Mapping[str, Any],
    output_root_claim_file_sha256: str,
) -> SkeletonGraphCalibrationOutcome:
    if stage not in _INTEGRITY_FAILURE_STAGES:
        raise SkeletonGraphCalibrationRunnerError(
            "terminal integrity stage is not allowlisted"
        )
    if _output_entry_exists(output, "terminal_state.json"):
        return _recover_terminal_outcome(
            output,
            authorization=authorization,
            output_root_claim=output_root_claim,
            output_root_claim_file_sha256=output_root_claim_file_sha256,
        )
    grant_exists = _output_entry_exists(output, "population_grant.json")
    gap_exists = _output_entry_exists(output, "calibration_gap.json")
    if grant_exists and gap_exists:
        raise SkeletonGraphCalibrationRunnerError(
            "grant and GAP terminal outcomes coexist"
        )
    if grant_exists or gap_exists:
        name = "population_grant.json" if grant_exists else "calibration_gap.json"
        schema = (
            (
                GENERIC_GRANT_SCHEMA
                if authorization.scope is SkeletonGraphCalibrationScope.GENERIC_V3
                else SAME_FAMILY_GRANT_SCHEMA
            )
            if grant_exists
            else GAP_SCHEMA
        )
        data, _raw = _read_output_record(
            output,
            name,
            schema=schema,
            label="unclaimed terminal outcome",
        )
        existing = _outcome_from_data(data)
        return _persist_terminal_outcome(
            output,
            authorization=authorization,
            output_root_claim=output_root_claim,
            output_root_claim_file_sha256=output_root_claim_file_sha256,
            outcome=existing,
        )

    inventory = _execution_inventory(output)
    prediction_entry = inventory.get("raw_predictions.json")
    label_entry = inventory.get("delayed_labels.json")
    prediction_record_digest = (
        prediction_entry["record_digest"]
        if prediction_entry is not None
        and prediction_entry["schema"] == PREDICTION_SCHEMA
        else None
    )
    label_record_digest = (
        label_entry["record_digest"]
        if label_entry is not None
        and label_entry["schema"] == LABEL_BATCH_SCHEMA
        else None
    )
    integrity_custody = {
        "failure_stage": stage,
        "exposure_authorization": authorization.to_data(),
        "exposure_authorization_file_sha256": authorization.file_sha256,
        "output_root_claim_record_digest": output_root_claim["record_digest"],
        "output_root_claim_file_sha256": output_root_claim_file_sha256,
        "inventory": inventory,
    }
    gap = SkeletonGraphCalibrationGap.create(
        scope=authorization.scope,
        stage="integrity_" + stage,
        reason_codes=("execution_integrity_failure", stage + "_failed"),
        passed_fit_record_digest=authorization.passed_fit_record_digest,
        prediction_record_digest=prediction_record_digest,
        label_record_digest=label_record_digest,
        integrity_custody=integrity_custody,
    )
    restored = _persist_terminal_outcome(
        output,
        authorization=authorization,
        output_root_claim=output_root_claim,
        output_root_claim_file_sha256=output_root_claim_file_sha256,
        outcome=gap,
    )
    if (
        type(restored) is SkeletonGraphCalibrationGap
        and not _typed_equal(restored.to_data(), gap.to_data())
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "terminal calibration integrity GAP differs"
        )
    return restored


def _execution_authorization_body(
    *,
    authorization: SkeletonGraphCalibrationExposureAuthorization,
    resolved_slot: Mapping[str, Any],
    identities: tuple[SkeletonGraphCalibrationPanelIdentity, ...],
    tokens: tuple[str, ...],
    output_identity: Mapping[str, object],
    output_root_claim_record_digest: str,
    output_root_claim_file_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": EXECUTION_AUTHORIZATION_SCHEMA,
        "exposure_authorization": authorization.to_data(),
        "exposure_authorization_file_sha256": authorization.file_sha256,
        "scope": authorization.scope.value,
        "population_scope": authorization.population_scope,
        "passed_fit_resolution": dict(resolved_slot),
        "panel_identities": [item.to_data() for item in identities],
        "panel_content_identity_digest": "sha256:"
        + canonical_digest(tuple(item.to_data() for item in identities)),
        "anonymous_bindings": [
            {"anonymous_panel_token": token, "panel_id": identity.panel_id}
            for token, identity in zip(tokens, identities, strict=True)
        ],
        "calibration_pixels_authorized": True,
        "target_pixels_authorized": False,
        "query_pixels_authorized": False,
        "support_pixels_authorized": False,
        "official_test_pixels_authorized": False,
        "action_labels_or_programs_authorized": False,
        "model_visible_identity_or_role": False,
        "authenticated_calibration_execution": False,
        "production_adapter_authorized": False,
        "output_root_identity": dict(output_identity),
        "output_root_claim_record_digest": output_root_claim_record_digest,
        "output_root_claim_file_sha256": output_root_claim_file_sha256,
    }


def _execution_precommit_body(
    *,
    authorization: SkeletonGraphCalibrationExposureAuthorization,
    resolved_slot: Mapping[str, Any],
    execution_authorization_record_digest: str,
    execution_authorization_file_sha256: str,
    identities: tuple[SkeletonGraphCalibrationPanelIdentity, ...],
    output_identity: Mapping[str, object],
    output_root_claim_record_digest: str,
    output_root_claim_file_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": PRECOMMIT_SCHEMA,
        "scope": authorization.scope.value,
        "execution_authorization_record_digest": (
            execution_authorization_record_digest
        ),
        "execution_authorization_file_sha256": execution_authorization_file_sha256,
        "exposure_authorization_record_digest": authorization.record_digest,
        "exposure_authorization_file_sha256": authorization.file_sha256,
        "campaign_intent_record_digest": authorization.campaign_intent_record_digest,
        "campaign_intent_file_sha256": authorization.campaign_intent_file_sha256,
        "exposure_predecessor_ledger_digest": (
            authorization.exposure_predecessor_ledger_digest
        ),
        "exposure_predecessor_file_sha256": (
            authorization.exposure_predecessor_file_sha256
        ),
        "exposure_event_digest": authorization.exposure_event_digest,
        "exposure_successor_ledger_digest": (
            authorization.exposure_successor_ledger_digest
        ),
        "exposure_successor_file_sha256": (
            authorization.exposure_successor_file_sha256
        ),
        "passed_fit_resolution": dict(resolved_slot),
        "panel_content_identity_digest": "sha256:"
        + canonical_digest(tuple(item.to_data() for item in identities)),
        "inference_source_sha256": "sha256:" + inference_custody.source_sha256(),
        "inference_algorithm_digest": inference_custody.algorithm_digest(),
        "intended_outputs": {
            "output_root_claim": "output_root_claim.json",
            "prediction_attempt": "prediction_attempt.json",
            "raw_predictions": "raw_predictions.json",
            "label_attempt": "label_attempt.json",
            "delayed_labels": "delayed_labels.json",
            "population_grant": "population_grant.json",
            "calibration_gap": "calibration_gap.json",
            "cold_replay": "cold_replay.json",
            "terminal_state": "terminal_state.json",
        },
        "prediction_addresses_unknown_before_inference": True,
        "prediction_addresses_must_freeze_after_fsync_reload_before_labels": True,
        "no_tuning_reroll_checkpoint_or_threshold_replacement": True,
        "calibration_pixel_reads_so_far": 0,
        "model_calls_so_far": 0,
        "label_or_action_program_reads_so_far": 0,
        "target_pixel_authorized": False,
        "authenticated_calibration_execution": False,
        "production_adapter_authorized": False,
        "output_root_identity": dict(output_identity),
        "output_root_claim_record_digest": output_root_claim_record_digest,
        "output_root_claim_file_sha256": output_root_claim_file_sha256,
    }


def run_calibration(
    *,
    exposure_authorization: SkeletonGraphCalibrationExposureAuthorization,
    preregistration_path: Path,
    passed_fit: SkeletonGraphPassedFitOutcome,
    passed_fit_paths: SkeletonGraphPassedFitPaths,
    panel_identities: Sequence[SkeletonGraphCalibrationPanelIdentity],
    output_directory: Path,
    calibration_pixel_reader: SkeletonGraphCalibrationPixelReader,
    inference_runner: SkeletonGraphInferenceRunner,
    delayed_label_reader_factory: SkeletonGraphDelayedLabelReaderFactory,
) -> SkeletonGraphCalibrationOutcome:
    """Execute exactly the cohort consumed by the prior metadata authorization."""

    if not callable(calibration_pixel_reader) or not callable(inference_runner):
        raise TypeError("pixel reader and inference runner must be callable")
    if not callable(delayed_label_reader_factory):
        raise TypeError("delayed label reader factory must be callable")
    authorization = verify_calibration_exposure_authorization(
        exposure_authorization,
        preregistration_path=preregistration_path,
        passed_fit=passed_fit,
        passed_fit_paths=passed_fit_paths,
        expected_record_digest=exposure_authorization.record_digest,
        expected_file_sha256=exposure_authorization.file_sha256,
    )
    verified_fit = _verify_passed_fit_outcome(passed_fit, passed_fit_paths)
    if type(verified_fit) is not SkeletonGraphPassedFitProtocol:
        raise SkeletonGraphCalibrationRunnerError(
            "authorized calibration no longer has a passed-fit protocol"
        )
    registration = _load_preregistration(Path(preregistration_path))
    identities, task_ids = _validate_panel_identities(
        registration, authorization.scope, panel_identities
    )
    panel_ids = tuple(item.panel_id for item in identities)
    if task_ids != authorization.task_ids or panel_ids != authorization.panel_ids:
        raise SkeletonGraphCalibrationRunnerError(
            "panel identities differ from consumed exposure cohort"
        )
    output_path = Path(output_directory).absolute()
    if str(output_path) != authorization.intended_output_directory:
        raise SkeletonGraphCalibrationRunnerError(
            "run output differs from fixed campaign intent"
        )
    tokens = _anonymous_tokens(authorization.scope, identities)
    parent_identity = {
        "absolute_path": authorization.intended_output_parent_path,
        "st_dev": authorization.intended_output_parent_st_dev,
        "st_ino": authorization.intended_output_parent_st_ino,
        "st_mode": authorization.intended_output_parent_st_mode,
    }
    output: _OutputDirectoryCustody | None = None
    output_root_claim: dict[str, Any] | None = None
    output_root_claim_file: str | None = None
    stage = "execution_authorization"
    try:
        output, _created = _acquire_recoverable_output_directory(
            output_path, expected_parent_identity=parent_identity
        )
        output_root_claim, output_root_claim_file = (
            _persist_or_verify_output_root_claim(output, authorization)
        )
        _verify_archived_exposure_authorization(
            authorization, registration=registration
        )
        if _output_entry_exists(output, "terminal_state.json"):
            return _recover_terminal_outcome(
                output,
                authorization=authorization,
                output_root_claim=output_root_claim,
                output_root_claim_file_sha256=output_root_claim_file,
            )
        initial_inventory = _execution_inventory(output)
        if set(initial_inventory) != {"output_root_claim.json"}:
            return _persist_terminal_integrity_gap(
                output,
                authorization=authorization,
                stage="recovered_interrupted_execution",
                output_root_claim=output_root_claim,
                output_root_claim_file_sha256=output_root_claim_file,
            )

        resolved_slot = _passed_fit_resolution(verified_fit, registration)
        execution_authorization, execution_authorization_file = _write_output_record(
            output,
            "authorization.json",
            _execution_authorization_body(
                authorization=authorization,
                resolved_slot=resolved_slot,
                identities=identities,
                tokens=tokens,
                output_identity=output.identity_data(),
                output_root_claim_record_digest=output_root_claim["record_digest"],
                output_root_claim_file_sha256=output_root_claim_file,
            ),
        )
        stage = "precommit"
        precommit, precommit_file = _write_output_record(
            output,
            "precommit.json",
            _execution_precommit_body(
                authorization=authorization,
                resolved_slot=resolved_slot,
                execution_authorization_record_digest=(
                    execution_authorization["record_digest"]
                ),
                execution_authorization_file_sha256=execution_authorization_file,
                identities=identities,
                output_identity=output.identity_data(),
                output_root_claim_record_digest=output_root_claim["record_digest"],
                output_root_claim_file_sha256=output_root_claim_file,
            ),
        )
        stage = "prediction_attempt"
        prediction_attempt, prediction_attempt_file = _write_output_record(
            output,
            "prediction_attempt.json",
            {
                "schema": ATTEMPT_SCHEMA,
                "stage": "prediction",
                "precommit_record_digest": precommit["record_digest"],
                "precommit_file_sha256": precommit_file,
                "exposure_authorization_record_digest": authorization.record_digest,
                "attempt_number": 1,
                "reroll_authorized": False,
            },
        )

        stage = "pixel_or_inference_callback"
        payloads = _read_calibration_payloads(identities, calibration_pixel_reader)
        inference_value = inference_runner(payloads, verified_fit)
        batch, receipt, frozen, occurrences = _validate_inference_pair(
            inference_value,
            passed_fit=verified_fit,
            identities=identities,
            tokens=tokens,
        )
        occurrence_join_digest = "sha256:" + canonical_digest(occurrences)
        stage = "prediction_write"
        prediction, prediction_file = _write_output_record(
            output,
            "raw_predictions.json",
            {
                "schema": PREDICTION_SCHEMA,
                "scope": authorization.scope.value,
                "precommit_record_digest": precommit["record_digest"],
                "precommit_file_sha256": precommit_file,
                "raw_inference_batch": batch.to_data(),
                "raw_inference_batch_file_sha256": frozen.raw_batch_file_sha256,
                "raw_inference_batch_record_digest": frozen.raw_batch_record_digest,
                "recompute_receipt": receipt.to_data(),
                "recompute_receipt_file_sha256": (
                    frozen.recompute_receipt_file_sha256
                ),
                "recompute_receipt_record_digest": (
                    frozen.recompute_receipt_record_digest
                ),
                "occurrences": occurrences,
                "occurrence_count": len(occurrences),
                "occurrence_join_digest": occurrence_join_digest,
                "raw_custody_contains_anonymous_tokens": False,
                "task_or_role_entered_inference": False,
                "file_fsync_completed": True,
                "directory_fsync_completed": True,
            },
        )
        stage = "prediction_fresh_reload"
        prediction, _ = _read_output_record(
            output,
            "raw_predictions.json",
            schema=PREDICTION_SCHEMA,
            label="fresh raw prediction artifact",
            expected_file_sha256=prediction_file,
            expected_record_digest=prediction["record_digest"],
        )
        _, _, frozen, loaded_occurrences = _validate_prediction_data(
            prediction, identities=identities, tokens=tokens
        )

        stage = "label_attempt"
        label_attempt, label_attempt_file = _write_output_record(
            output,
            "label_attempt.json",
            {
                "schema": ATTEMPT_SCHEMA,
                "stage": "delayed_labels",
                "precommit_record_digest": precommit["record_digest"],
                "precommit_file_sha256": precommit_file,
                "prediction_record_digest": prediction["record_digest"],
                "prediction_file_sha256": prediction_file,
                "prediction_fresh_reload_verified": True,
                "attempt_number": 1,
                "reroll_authorized": False,
            },
        )
        request = SkeletonGraphDelayedLabelRequest._issue(
            issuance_token=_DELAYED_LABEL_REQUEST_ISSUANCE_TOKEN,
            scope=authorization.scope,
            exposure_authorization_record_digest=authorization.record_digest,
            exposure_authorization_file_sha256=authorization.file_sha256,
            execution_authorization_record_digest=execution_authorization[
                "record_digest"
            ],
            execution_authorization_file_sha256=execution_authorization_file,
            precommit_record_digest=precommit["record_digest"],
            precommit_file_sha256=precommit_file,
            prediction_attempt_record_digest=prediction_attempt["record_digest"],
            prediction_attempt_file_sha256=prediction_attempt_file,
            prediction_record_digest=prediction["record_digest"],
            prediction_file_sha256=prediction_file,
            occurrence_join_digest=prediction["occurrence_join_digest"],
            bindings=tuple(zip(tokens, identities, strict=True)),
            label_attempt_record_digest=label_attempt["record_digest"],
            label_attempt_file_sha256=label_attempt_file,
            output_identity=output.identity_data(),
        )
        stage = "delayed_label_callback"
        label_reader = delayed_label_reader_factory()
        if not callable(label_reader):
            raise SkeletonGraphCalibrationRunnerError(
                "label factory did not return a reader"
            )
        label_value = label_reader(request)
        if request._lease.consumed is not True:
            raise SkeletonGraphCalibrationRunnerError(
                "delayed label reader did not consume the exact sealed request"
            )
        label_batch = _validate_label_batch(label_value, request=request)
        stage = "delayed_label_write"
        labels_data, labels_file = _write_output_record(
            output, "delayed_labels.json", label_batch.content_data()
        )
        labels_data, _ = _read_output_record(
            output,
            "delayed_labels.json",
            schema=LABEL_BATCH_SCHEMA,
            label="fresh delayed label artifact",
            expected_file_sha256=labels_file,
            expected_record_digest=labels_data["record_digest"],
        )
        label_batch = SkeletonGraphDelayedLabelBatch.from_data(labels_data)
        _validate_label_batch(label_batch, request=request)
        stage = "population_evaluation"
        q, task_scores, gate, failure_reasons = _evaluate_archive(
            scope=authorization.scope,
            task_ids=task_ids,
            occurrences=loaded_occurrences,
            labels=label_batch,
        )
        _verify_archived_exposure_authorization(
            authorization, registration=registration
        )
        if failure_reasons:
            outcome: SkeletonGraphCalibrationOutcome = SkeletonGraphCalibrationGap.create(
                scope=authorization.scope,
                stage="population_efficiency",
                reason_codes=failure_reasons,
                passed_fit_record_digest=verified_fit.record_digest,
                prediction_record_digest=prediction["record_digest"],
                label_record_digest=label_batch.record_digest,
            )
        else:
            outcome = SkeletonGraphPopulationGrant._issue_after_verified_archive(
                scope=authorization.scope,
                population_scope=authorization.population_scope,
                q=q,
                calibration_task_ids=task_ids,
                task_scores=task_scores,
                efficiency_gate=gate,
                prediction_record_digest=prediction["record_digest"],
                label_record_digest=label_batch.record_digest,
                frozen_inference=frozen,
                occurrence_join_digest=occurrence_join_digest,
                issuance_token=_GRANT_ISSUANCE_TOKEN,
            )
        _verify_archived_exposure_authorization(
            authorization, registration=registration
        )
        stage = "outcome_write"
        restored_outcome = _persist_terminal_outcome(
            output,
            authorization=authorization,
            output_root_claim=output_root_claim,
            output_root_claim_file_sha256=output_root_claim_file,
            outcome=outcome,
        )
        return restored_outcome
    except Exception as exc:
        if (
            output is None
            or output_root_claim is None
            or output_root_claim_file is None
        ):
            if isinstance(exc, SkeletonGraphCalibrationRunnerError):
                raise
            raise SkeletonGraphCalibrationRunnerError(
                f"calibration output transaction failed at {stage}"
            ) from exc
        try:
            terminal = _persist_terminal_integrity_gap(
                output,
                authorization=authorization,
                stage=stage,
                output_root_claim=output_root_claim,
                output_root_claim_file_sha256=output_root_claim_file,
            )
        except Exception as terminal_exc:
            raise SkeletonGraphCalibrationRunnerError(
                "calibration failed and terminal integrity GAP could not persist"
            ) from terminal_exc
        if type(terminal) is SkeletonGraphPopulationGrant:
            _verify_archived_exposure_authorization(
                authorization, registration=registration
            )
            return terminal
        if isinstance(exc, SkeletonGraphCalibrationRunnerError):
            raise
        raise SkeletonGraphCalibrationRunnerError(
            f"calibration execution failed at {stage}"
        ) from exc
    finally:
        if output is not None:
            output.close()


def _read_exact_attempt(
    directory: _OutputDirectoryCustody,
    name: str,
    body: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    expected = _seal(body)
    expected_raw = canonical_json(expected) + b"\n"
    loaded, raw = _read_output_record(
        directory,
        name,
        schema=ATTEMPT_SCHEMA,
        label=name,
        expected_file_sha256=_file_address(expected_raw),
        expected_record_digest=expected["record_digest"],
    )
    if loaded != expected or raw != expected_raw:
        raise SkeletonGraphCalibrationRunnerError(f"{name} exact join differs")
    return loaded, raw


def _base_archive_join(
    *,
    directory: _OutputDirectoryCustody,
    registration: Mapping[str, Any],
) -> tuple[
    SkeletonGraphCalibrationExposureAuthorization,
    tuple[SkeletonGraphCalibrationPanelIdentity, ...],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
]:
    execution, execution_raw = _read_output_record(
        directory,
        "authorization.json",
        schema=EXECUTION_AUTHORIZATION_SCHEMA,
        label="execution authorization",
    )
    authorization = _authorization_from_data(execution.get("exposure_authorization"))
    _verify_archived_exposure_authorization(
        authorization, registration=registration
    )
    if authorization.intended_output_directory != str(directory.path):
        raise SkeletonGraphCalibrationRunnerError(
            "archive root differs from exposure authorization"
        )
    output_root_claim, output_root_claim_raw = _read_exact_output_root_claim(
        directory, authorization
    )
    identities_raw = execution.get("panel_identities")
    if type(identities_raw) is not list:
        raise SkeletonGraphCalibrationRunnerError(
            "execution panel identity inventory differs"
        )
    identities, task_ids = _validate_panel_identities(
        registration,
        authorization.scope,
        tuple(
            SkeletonGraphCalibrationPanelIdentity.from_data(item)
            for item in identities_raw
        ),
    )
    if (
        task_ids != authorization.task_ids
        or tuple(item.panel_id for item in identities) != authorization.panel_ids
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "execution cohort differs from exposure authorization"
        )
    tokens = _anonymous_tokens(authorization.scope, identities)
    expected_execution = _seal(
        _execution_authorization_body(
            authorization=authorization,
            resolved_slot=_pinned_passed_fit_resolution(registration),
            identities=identities,
            tokens=tokens,
            output_identity=directory.identity_data(),
            output_root_claim_record_digest=output_root_claim["record_digest"],
            output_root_claim_file_sha256=_file_address(output_root_claim_raw),
        )
    )
    if (
        not _typed_equal(execution, expected_execution)
        or execution_raw != canonical_json(expected_execution) + b"\n"
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "execution authorization full replay differs"
        )
    precommit, precommit_raw = _read_output_record(
        directory,
        "precommit.json",
        schema=PRECOMMIT_SCHEMA,
        label="execution precommit",
    )
    expected_precommit = _seal(
        _execution_precommit_body(
            authorization=authorization,
            resolved_slot=_pinned_passed_fit_resolution(registration),
            execution_authorization_record_digest=execution["record_digest"],
            execution_authorization_file_sha256=_file_address(execution_raw),
            identities=identities,
            output_identity=directory.identity_data(),
            output_root_claim_record_digest=output_root_claim["record_digest"],
            output_root_claim_file_sha256=_file_address(output_root_claim_raw),
        )
    )
    if (
        not _typed_equal(precommit, expected_precommit)
        or precommit_raw != canonical_json(expected_precommit) + b"\n"
    ):
        raise SkeletonGraphCalibrationRunnerError("execution precommit full replay differs")
    return (
        authorization,
        identities,
        task_ids,
        tokens,
        execution,
        execution_raw,
        precommit,
        precommit_raw,
        output_root_claim,
        output_root_claim_raw,
    )


def _terminal_gap_replay(
    *,
    directory: _OutputDirectoryCustody,
    registration: Mapping[str, Any],
    gap: SkeletonGraphCalibrationGap,
    gap_raw: bytes,
) -> SkeletonGraphCalibrationGap:
    if gap.integrity_custody is None:
        raise SkeletonGraphCalibrationRunnerError("terminal GAP custody is missing")
    custody = gap.integrity_custody
    authorization = _authorization_from_data(custody["exposure_authorization"])
    if (
        authorization.file_sha256
        != custody["exposure_authorization_file_sha256"]
        or authorization.scope is not gap.scope
        or authorization.passed_fit_record_digest != gap.passed_fit_record_digest
        or authorization.intended_output_directory != str(directory.path)
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "terminal GAP authorization join differs"
        )
    _verify_archived_exposure_authorization(
        authorization, registration=registration
    )
    output_root_claim, output_root_claim_raw = _read_exact_output_root_claim(
        directory, authorization
    )
    if (
        output_root_claim["record_digest"]
        != custody["output_root_claim_record_digest"]
        or _file_address(output_root_claim_raw)
        != custody["output_root_claim_file_sha256"]
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "terminal GAP output-root custody differs"
        )
    expected_inventory = _validate_inventory_wire(custody["inventory"])
    actual_inventory = _execution_inventory(directory)
    if actual_inventory != expected_inventory:
        raise SkeletonGraphCalibrationRunnerError(
            "terminal GAP exact inventory differs"
        )
    order = (
        "output_root_claim.json",
        "authorization.json",
        "precommit.json",
        "prediction_attempt.json",
        "raw_predictions.json",
        "label_attempt.json",
        "delayed_labels.json",
    )
    if any(
        entry["schema"] != _INVENTORY_SCHEMA_BY_NAME[name]
        for name, entry in actual_inventory.items()
        if name in _INVENTORY_SCHEMA_BY_NAME
    ) or any(name not in order for name in actual_inventory):
        raise SkeletonGraphCalibrationRunnerError(
            "terminal GAP inventory schema differs"
        )
    present = tuple(name for name in order if name in actual_inventory)
    if present != order[: len(present)]:
        raise SkeletonGraphCalibrationRunnerError(
            "terminal GAP inventory is not a causal prefix"
        )
    stage = custody["failure_stage"]
    bounds = {
        "execution_authorization": (1, 2),
        "precommit": (2, 3),
        "prediction_attempt": (3, 4),
        "pixel_or_inference_callback": (4, 4),
        "prediction_write": (4, 5),
        "prediction_fresh_reload": (5, 5),
        "label_attempt": (5, 6),
        "delayed_label_callback": (6, 6),
        "delayed_label_write": (6, 7),
        "population_evaluation": (7, 7),
        "outcome_write": (7, 7),
        "recovered_interrupted_execution": (2, 7),
    }
    minimum, maximum = bounds[stage]
    if not minimum <= len(present) <= maximum:
        raise SkeletonGraphCalibrationRunnerError(
            "terminal GAP stage/inventory chronology differs"
        )

    identities: tuple[SkeletonGraphCalibrationPanelIdentity, ...] = ()
    tokens: tuple[str, ...] = ()
    execution: Mapping[str, Any] | None = None
    execution_raw: bytes | None = None
    precommit: Mapping[str, Any] | None = None
    precommit_raw: bytes | None = None
    if len(present) >= 3:
        (
            joined_authorization,
            identities,
            _task_ids,
            tokens,
            execution,
            execution_raw,
            precommit,
            precommit_raw,
            joined_root_claim,
            joined_root_claim_raw,
        ) = _base_archive_join(directory=directory, registration=registration)
        if (
            not _typed_equal(
                joined_authorization.to_data(), authorization.to_data()
            )
            or joined_root_claim != output_root_claim
            or joined_root_claim_raw != output_root_claim_raw
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "terminal GAP base archive join differs"
            )
    elif len(present) == 2:
        execution, execution_raw = _read_output_record(
            directory,
            "authorization.json",
            schema=EXECUTION_AUTHORIZATION_SCHEMA,
            label="terminal execution authorization",
        )
        embedded = _authorization_from_data(execution.get("exposure_authorization"))
        identities_raw = execution.get("panel_identities")
        if type(identities_raw) is not list or not _typed_equal(
            embedded.to_data(), authorization.to_data()
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "terminal execution authorization differs"
            )
        identities, task_ids = _validate_panel_identities(
            registration,
            authorization.scope,
            tuple(
                SkeletonGraphCalibrationPanelIdentity.from_data(item)
                for item in identities_raw
            ),
        )
        tokens = _anonymous_tokens(authorization.scope, identities)
        expected_execution = _seal(
            _execution_authorization_body(
                authorization=authorization,
                resolved_slot=_pinned_passed_fit_resolution(registration),
                identities=identities,
                tokens=tokens,
                output_identity=directory.identity_data(),
                output_root_claim_record_digest=output_root_claim["record_digest"],
                output_root_claim_file_sha256=_file_address(output_root_claim_raw),
            )
        )
        if (
            task_ids != authorization.task_ids
            or not _typed_equal(execution, expected_execution)
            or execution_raw != canonical_json(expected_execution) + b"\n"
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "terminal execution authorization full replay differs"
            )

    prediction_attempt: dict[str, Any] | None = None
    prediction_attempt_raw: bytes | None = None
    if len(present) >= 4:
        assert precommit is not None and precommit_raw is not None
        prediction_attempt, prediction_attempt_raw = _read_exact_attempt(
            directory,
            "prediction_attempt.json",
            {
                "schema": ATTEMPT_SCHEMA,
                "stage": "prediction",
                "precommit_record_digest": precommit["record_digest"],
                "precommit_file_sha256": _file_address(precommit_raw),
                "exposure_authorization_record_digest": authorization.record_digest,
                "attempt_number": 1,
                "reroll_authorized": False,
            },
        )
    prediction: dict[str, Any] | None = None
    prediction_raw: bytes | None = None
    if len(present) >= 5:
        assert precommit is not None and precommit_raw is not None
        prediction, prediction_raw = _read_output_record(
            directory,
            "raw_predictions.json",
            schema=PREDICTION_SCHEMA,
            label="terminal archived predictions",
        )
        _validate_prediction_data(prediction, identities=identities, tokens=tokens)
        if (
            prediction.get("precommit_record_digest") != precommit["record_digest"]
            or prediction.get("precommit_file_sha256") != _file_address(precommit_raw)
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "terminal prediction/precommit join differs"
            )
    label_attempt: dict[str, Any] | None = None
    label_attempt_raw: bytes | None = None
    if len(present) >= 6:
        assert prediction is not None and prediction_raw is not None
        assert precommit is not None and precommit_raw is not None
        label_attempt, label_attempt_raw = _read_exact_attempt(
            directory,
            "label_attempt.json",
            {
                "schema": ATTEMPT_SCHEMA,
                "stage": "delayed_labels",
                "precommit_record_digest": precommit["record_digest"],
                "precommit_file_sha256": _file_address(precommit_raw),
                "prediction_record_digest": prediction["record_digest"],
                "prediction_file_sha256": _file_address(prediction_raw),
                "prediction_fresh_reload_verified": True,
                "attempt_number": 1,
                "reroll_authorized": False,
            },
        )
    labels: SkeletonGraphDelayedLabelBatch | None = None
    labels_raw: bytes | None = None
    if len(present) >= 7:
        assert execution is not None and execution_raw is not None
        assert precommit is not None and precommit_raw is not None
        assert prediction_attempt is not None and prediction_attempt_raw is not None
        assert prediction is not None and prediction_raw is not None
        assert label_attempt is not None and label_attempt_raw is not None
        labels_data, labels_raw = _read_output_record(
            directory,
            "delayed_labels.json",
            schema=LABEL_BATCH_SCHEMA,
            label="terminal archived labels",
        )
        request = SkeletonGraphDelayedLabelRequest._issue(
            issuance_token=_DELAYED_LABEL_REQUEST_ISSUANCE_TOKEN,
            scope=authorization.scope,
            exposure_authorization_record_digest=authorization.record_digest,
            exposure_authorization_file_sha256=authorization.file_sha256,
            execution_authorization_record_digest=execution["record_digest"],
            execution_authorization_file_sha256=_file_address(execution_raw),
            precommit_record_digest=precommit["record_digest"],
            precommit_file_sha256=_file_address(precommit_raw),
            prediction_attempt_record_digest=prediction_attempt["record_digest"],
            prediction_attempt_file_sha256=_file_address(prediction_attempt_raw),
            prediction_record_digest=prediction["record_digest"],
            prediction_file_sha256=_file_address(prediction_raw),
            occurrence_join_digest=prediction["occurrence_join_digest"],
            bindings=tuple(zip(tokens, identities, strict=True)),
            label_attempt_record_digest=label_attempt["record_digest"],
            label_attempt_file_sha256=_file_address(label_attempt_raw),
            output_identity=directory.identity_data(),
        )
        labels = _validate_label_batch(
            SkeletonGraphDelayedLabelBatch.from_data(labels_data), request=request
        )

    prediction_digest = None if prediction is None else prediction["record_digest"]
    label_digest = None if labels is None else labels.record_digest
    rebuilt_custody = {
        "failure_stage": stage,
        "exposure_authorization": authorization.to_data(),
        "exposure_authorization_file_sha256": authorization.file_sha256,
        "output_root_claim_record_digest": output_root_claim["record_digest"],
        "output_root_claim_file_sha256": _file_address(output_root_claim_raw),
        "inventory": actual_inventory,
    }
    rebuilt_gap = SkeletonGraphCalibrationGap.create(
        scope=authorization.scope,
        stage="integrity_" + stage,
        reason_codes=("execution_integrity_failure", stage + "_failed"),
        passed_fit_record_digest=authorization.passed_fit_record_digest,
        prediction_record_digest=prediction_digest,
        label_record_digest=label_digest,
        integrity_custody=rebuilt_custody,
    )
    if not _typed_equal(rebuilt_gap.to_data(), gap.to_data()):
        raise SkeletonGraphCalibrationRunnerError(
            "terminal calibration GAP exact rebuild differs"
        )
    terminal_outcome = _recover_terminal_outcome(
        directory,
        authorization=authorization,
        output_root_claim=output_root_claim,
        output_root_claim_file_sha256=_file_address(output_root_claim_raw),
    )
    if (
        type(terminal_outcome) is not SkeletonGraphCalibrationGap
        or not _typed_equal(terminal_outcome.to_data(), gap.to_data())
        or _output_entry_exists(directory, "population_grant.json")
    ):
        raise SkeletonGraphCalibrationRunnerError(
            "terminal GAP is not the single outcome winner"
        )
    terminal_state, terminal_state_raw = _read_output_record(
        directory,
        "terminal_state.json",
        schema=TERMINAL_STATE_SCHEMA,
        label="terminal state",
    )
    replay_custody: dict[str, Any] = {
        "campaign_intent_file_sha256": authorization.campaign_intent_file_sha256,
        "campaign_intent_record_digest": authorization.campaign_intent_record_digest,
        "exposure_predecessor_file_sha256": authorization.exposure_predecessor_file_sha256,
        "exposure_predecessor_ledger_digest": authorization.exposure_predecessor_ledger_digest,
        "exposure_event_digest": authorization.exposure_event_digest,
        "exposure_successor_file_sha256": authorization.exposure_successor_file_sha256,
        "exposure_successor_ledger_digest": authorization.exposure_successor_ledger_digest,
        "exposure_authorization_file_sha256": authorization.file_sha256,
        "exposure_authorization_record_digest": authorization.record_digest,
        "output_root_claim_file_sha256": _file_address(output_root_claim_raw),
        "output_root_claim_record_digest": output_root_claim["record_digest"],
        "execution_inventory": actual_inventory,
        "terminal_state_file_sha256": _file_address(terminal_state_raw),
        "terminal_state_record_digest": terminal_state["record_digest"],
        "outcome_file_sha256": _file_address(gap_raw),
        "outcome_record_digest": gap.record_digest,
    }
    _verify_archived_exposure_authorization(
        authorization, registration=registration
    )
    _persist_or_verify_replay(
        directory,
        {
            "schema": REPLAY_SCHEMA,
            "outcome_schema": GAP_SCHEMA,
            "outcome_record_digest": gap.record_digest,
            "archive_custody": replay_custody,
            "terminal_integrity_gap": True,
            "exact_replay": True,
            "pixel_reads": 0,
            "feature_extraction_calls": 0,
            "model_prediction_api_calls": 0,
            "estimator_predict_proba_calls": 0,
            "label_authority_reads": 0,
            "authenticated_calibration_execution": False,
            "production_adapter_authorized": False,
            "runner_source_sha256": "sha256:" + source_sha256(),
            "runner_algorithm_digest": algorithm_digest(),
        },
    )
    return gap


def _recover_claimed_terminal_before_inventory(
    directory: _OutputDirectoryCustody,
    *,
    registration: Mapping[str, Any],
) -> SkeletonGraphCalibrationOutcome:
    """Materialize the exact terminal claim before inspecting outcome names."""

    if not _output_entry_exists(directory, "terminal_state.json"):
        raise SkeletonGraphCalibrationRunnerError(
            "archive terminal state is missing"
        )
    claim, _claim_raw = _read_output_record(
        directory,
        "output_root_claim.json",
        schema=OUTPUT_ROOT_CLAIM_SCHEMA,
        label="archive output root claim",
    )
    authorization = _authorization_from_data(claim.get("exposure_authorization"))
    if claim.get("exposure_authorization_file_sha256") != authorization.file_sha256:
        raise SkeletonGraphCalibrationRunnerError(
            "archive output claim authorization differs"
        )
    _verify_archived_exposure_authorization(
        authorization, registration=registration
    )
    exact_claim, exact_claim_raw = _read_exact_output_root_claim(
        directory, authorization
    )
    if not _typed_equal(claim, exact_claim):
        raise SkeletonGraphCalibrationRunnerError(
            "archive output root claim full replay differs"
        )
    outcome = _recover_terminal_outcome(
        directory,
        authorization=authorization,
        output_root_claim=exact_claim,
        output_root_claim_file_sha256=_file_address(exact_claim_raw),
    )
    _verify_archived_exposure_authorization(
        authorization, registration=registration
    )
    return outcome


def cold_replay_calibration(
    *,
    run_directory: Path,
    preregistration_path: Path,
) -> SkeletonGraphCalibrationReplayReceipt | SkeletonGraphCalibrationGap:
    """Replay every custody join with zero pixel/model/label callback calls."""

    registration = _load_preregistration(Path(preregistration_path))
    directory = _existing_output_directory(Path(run_directory))
    try:
        claimed_terminal_outcome = _recover_claimed_terminal_before_inventory(
            directory, registration=registration
        )
        grant_exists = _output_entry_exists(directory, "population_grant.json")
        gap_exists = _output_entry_exists(directory, "calibration_gap.json")
        if grant_exists == gap_exists:
            raise SkeletonGraphCalibrationRunnerError(
                "archive outcome inventory differs"
            )
        archived_gap: SkeletonGraphCalibrationGap | None = None
        gap_raw: bytes | None = None
        if gap_exists:
            gap_data, gap_raw = _read_output_record(
                directory,
                "calibration_gap.json",
                schema=GAP_SCHEMA,
                label="archived calibration GAP",
            )
            archived_gap = SkeletonGraphCalibrationGap.from_data(gap_data)
            if archived_gap.stage.startswith("integrity_"):
                return _terminal_gap_replay(
                    directory=directory,
                    registration=registration,
                    gap=archived_gap,
                    gap_raw=gap_raw,
                )
        (
            authorization,
            identities,
            task_ids,
            tokens,
            execution,
            execution_raw,
            precommit,
            precommit_raw,
            output_root_claim,
            output_root_claim_raw,
        ) = _base_archive_join(directory=directory, registration=registration)
        terminal_outcome = _recover_terminal_outcome(
            directory,
            authorization=authorization,
            output_root_claim=output_root_claim,
            output_root_claim_file_sha256=_file_address(output_root_claim_raw),
        )
        if not _typed_equal(
            terminal_outcome.to_data(), claimed_terminal_outcome.to_data()
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "freshly recovered terminal outcome differs"
            )
        if (
            gap_exists
            and (
                type(terminal_outcome) is not SkeletonGraphCalibrationGap
                or archived_gap is None
                or not _typed_equal(
                    terminal_outcome.to_data(), archived_gap.to_data()
                )
            )
        ) or (grant_exists and type(terminal_outcome) is not SkeletonGraphPopulationGrant):
            raise SkeletonGraphCalibrationRunnerError(
                "archive terminal state/outcome join differs"
            )
        terminal_state, terminal_state_raw = _read_output_record(
            directory,
            "terminal_state.json",
            schema=TERMINAL_STATE_SCHEMA,
            label="archive terminal state",
        )

        prediction_attempt, prediction_attempt_raw = _read_exact_attempt(
            directory,
            "prediction_attempt.json",
            {
                "schema": ATTEMPT_SCHEMA,
                "stage": "prediction",
                "precommit_record_digest": precommit["record_digest"],
                "precommit_file_sha256": _file_address(precommit_raw),
                "exposure_authorization_record_digest": authorization.record_digest,
                "attempt_number": 1,
                "reroll_authorized": False,
            },
        )
        prediction, prediction_raw = _read_output_record(
            directory,
            "raw_predictions.json",
            schema=PREDICTION_SCHEMA,
            label="archived raw predictions",
        )
        if (
            prediction.get("precommit_record_digest") != precommit["record_digest"]
            or prediction.get("precommit_file_sha256") != _file_address(precommit_raw)
        ):
            raise SkeletonGraphCalibrationRunnerError(
                "prediction/precommit join differs"
            )
        _, _, frozen, occurrences = _validate_prediction_data(
            prediction, identities=identities, tokens=tokens
        )
        prediction_file = _file_address(prediction_raw)
        label_attempt, label_attempt_raw = _read_exact_attempt(
            directory,
            "label_attempt.json",
            {
                "schema": ATTEMPT_SCHEMA,
                "stage": "delayed_labels",
                "precommit_record_digest": precommit["record_digest"],
                "precommit_file_sha256": _file_address(precommit_raw),
                "prediction_record_digest": prediction["record_digest"],
                "prediction_file_sha256": prediction_file,
                "prediction_fresh_reload_verified": True,
                "attempt_number": 1,
                "reroll_authorized": False,
            },
        )
        request = SkeletonGraphDelayedLabelRequest._issue(
            issuance_token=_DELAYED_LABEL_REQUEST_ISSUANCE_TOKEN,
            scope=authorization.scope,
            exposure_authorization_record_digest=authorization.record_digest,
            exposure_authorization_file_sha256=authorization.file_sha256,
            execution_authorization_record_digest=execution["record_digest"],
            execution_authorization_file_sha256=_file_address(execution_raw),
            precommit_record_digest=precommit["record_digest"],
            precommit_file_sha256=_file_address(precommit_raw),
            prediction_attempt_record_digest=prediction_attempt["record_digest"],
            prediction_attempt_file_sha256=_file_address(prediction_attempt_raw),
            prediction_record_digest=prediction["record_digest"],
            prediction_file_sha256=prediction_file,
            occurrence_join_digest=prediction["occurrence_join_digest"],
            bindings=tuple(zip(tokens, identities, strict=True)),
            label_attempt_record_digest=label_attempt["record_digest"],
            label_attempt_file_sha256=_file_address(label_attempt_raw),
            output_identity=directory.identity_data(),
        )
        labels_data, labels_raw = _read_output_record(
            directory,
            "delayed_labels.json",
            schema=LABEL_BATCH_SCHEMA,
            label="archived delayed labels",
        )
        labels = _validate_label_batch(
            SkeletonGraphDelayedLabelBatch.from_data(labels_data), request=request
        )
        q, task_scores, gate, failure_reasons = _evaluate_archive(
            scope=authorization.scope,
            task_ids=task_ids,
            occurrences=occurrences,
            labels=labels,
        )
        _verify_archived_exposure_authorization(
            authorization, registration=registration
        )
        outcome_raw: bytes
        if failure_reasons:
            if not gap_exists:
                raise SkeletonGraphCalibrationRunnerError(
                    "replayed GAP outcome is missing"
                )
            gap_data, outcome_raw = _read_output_record(
                directory,
                "calibration_gap.json",
                schema=GAP_SCHEMA,
                label="archived calibration GAP",
            )
            archived_gap = SkeletonGraphCalibrationGap.from_data(gap_data)
            rebuilt_gap = SkeletonGraphCalibrationGap.create(
                scope=authorization.scope,
                stage="population_efficiency",
                reason_codes=failure_reasons,
                passed_fit_record_digest=authorization.passed_fit_record_digest,
                prediction_record_digest=prediction["record_digest"],
                label_record_digest=labels.record_digest,
            )
            if not _typed_equal(rebuilt_gap.to_data(), archived_gap.to_data()):
                raise SkeletonGraphCalibrationRunnerError(
                    "calibration GAP cold replay differs"
                )
            custody = {
                "campaign_intent_file_sha256": authorization.campaign_intent_file_sha256,
                "campaign_intent_record_digest": authorization.campaign_intent_record_digest,
                "exposure_predecessor_file_sha256": authorization.exposure_predecessor_file_sha256,
                "exposure_predecessor_ledger_digest": authorization.exposure_predecessor_ledger_digest,
                "exposure_event_digest": authorization.exposure_event_digest,
                "exposure_successor_file_sha256": authorization.exposure_successor_file_sha256,
                "exposure_successor_ledger_digest": authorization.exposure_successor_ledger_digest,
                "exposure_authorization_file_sha256": authorization.file_sha256,
                "exposure_authorization_record_digest": authorization.record_digest,
                "output_root_claim_file_sha256": _file_address(output_root_claim_raw),
                "output_root_claim_record_digest": output_root_claim["record_digest"],
                "execution_authorization_file_sha256": _file_address(execution_raw),
                "execution_authorization_record_digest": execution["record_digest"],
                "precommit_file_sha256": _file_address(precommit_raw),
                "precommit_record_digest": precommit["record_digest"],
                "prediction_attempt_file_sha256": _file_address(prediction_attempt_raw),
                "prediction_attempt_record_digest": prediction_attempt["record_digest"],
                "prediction_file_sha256": prediction_file,
                "prediction_record_digest": prediction["record_digest"],
                "label_attempt_file_sha256": _file_address(label_attempt_raw),
                "label_attempt_record_digest": label_attempt["record_digest"],
                "label_file_sha256": _file_address(labels_raw),
                "label_record_digest": labels.record_digest,
                "outcome_file_sha256": _file_address(outcome_raw),
                "outcome_record_digest": archived_gap.record_digest,
                "terminal_state_file_sha256": _file_address(terminal_state_raw),
                "terminal_state_record_digest": terminal_state["record_digest"],
            }
            _persist_or_verify_replay(
                directory,
                {
                    "schema": REPLAY_SCHEMA,
                    "outcome_schema": GAP_SCHEMA,
                    "outcome_record_digest": archived_gap.record_digest,
                    "archive_custody": custody,
                    "terminal_integrity_gap": False,
                    "exact_replay": True,
                    "pixel_reads": 0,
                    "feature_extraction_calls": 0,
                    "model_prediction_api_calls": 0,
                    "estimator_predict_proba_calls": 0,
                    "label_authority_reads": 0,
                    "authenticated_calibration_execution": False,
                    "production_adapter_authorized": False,
                    "runner_source_sha256": "sha256:" + source_sha256(),
                    "runner_algorithm_digest": algorithm_digest(),
                },
            )
            return archived_gap

        grant_data, outcome_raw = _read_output_record(
            directory,
            "population_grant.json",
            schema=(
                GENERIC_GRANT_SCHEMA
                if authorization.scope is SkeletonGraphCalibrationScope.GENERIC_V3
                else SAME_FAMILY_GRANT_SCHEMA
            ),
            label="archived population grant",
        )
        archived_grant = SkeletonGraphPopulationGrant.from_data(grant_data)
        rebuilt_grant = SkeletonGraphPopulationGrant._issue_after_verified_archive(
            scope=authorization.scope,
            population_scope=authorization.population_scope,
            q=q,
            calibration_task_ids=task_ids,
            task_scores=task_scores,
            efficiency_gate=gate,
            prediction_record_digest=prediction["record_digest"],
            label_record_digest=labels.record_digest,
            frozen_inference=frozen,
            occurrence_join_digest=prediction["occurrence_join_digest"],
            issuance_token=_GRANT_ISSUANCE_TOKEN,
        )
        if not _typed_equal(rebuilt_grant.to_data(), archived_grant.to_data()):
            raise SkeletonGraphCalibrationRunnerError(
                "population grant cold replay differs"
            )
        archive_custody = {
            "campaign_intent_file_sha256": authorization.campaign_intent_file_sha256,
            "campaign_intent_record_digest": authorization.campaign_intent_record_digest,
            "exposure_predecessor_file_sha256": authorization.exposure_predecessor_file_sha256,
            "exposure_predecessor_ledger_digest": authorization.exposure_predecessor_ledger_digest,
            "exposure_event_digest": authorization.exposure_event_digest,
            "exposure_successor_file_sha256": authorization.exposure_successor_file_sha256,
            "exposure_successor_ledger_digest": authorization.exposure_successor_ledger_digest,
            "exposure_authorization_file_sha256": authorization.file_sha256,
            "exposure_authorization_record_digest": authorization.record_digest,
            "output_root_claim_file_sha256": _file_address(output_root_claim_raw),
            "output_root_claim_record_digest": output_root_claim["record_digest"],
            "execution_authorization_file_sha256": _file_address(execution_raw),
            "execution_authorization_record_digest": execution["record_digest"],
            "precommit_file_sha256": _file_address(precommit_raw),
            "precommit_record_digest": precommit["record_digest"],
            "prediction_attempt_file_sha256": _file_address(prediction_attempt_raw),
            "prediction_attempt_record_digest": prediction_attempt["record_digest"],
            "prediction_file_sha256": prediction_file,
            "prediction_record_digest": prediction["record_digest"],
            "label_attempt_file_sha256": _file_address(label_attempt_raw),
            "label_attempt_record_digest": label_attempt["record_digest"],
            "label_file_sha256": _file_address(labels_raw),
            "label_record_digest": labels.record_digest,
            "outcome_file_sha256": _file_address(outcome_raw),
            "outcome_record_digest": archived_grant.record_digest,
            "terminal_state_file_sha256": _file_address(terminal_state_raw),
            "terminal_state_record_digest": terminal_state["record_digest"],
        }
        receipt = SkeletonGraphCalibrationReplayReceipt._issue(
            grant=archived_grant,
            archive_custody=archive_custody,
            issuance_token=_REPLAY_RECEIPT_ISSUANCE_TOKEN,
        )
        loaded_receipt, _ = _persist_or_verify_replay(
            directory, receipt.content_data()
        )
        if not _typed_equal(
            loaded_receipt, receipt.to_data()
        ) or not receipt.verifies(archived_grant):
            raise SkeletonGraphCalibrationRunnerError("cold replay receipt differs")
        return receipt
    finally:
        directory.close()


__all__ = (
    "AUTHORIZATION_SCHEMA",
    "CAMPAIGN_ATTEMPT_AUTHORITY_SCHEMA",
    "EXECUTION_AUTHORIZATION_SCHEMA",
    "GAP_SCHEMA",
    "GENERIC_GRANT_SCHEMA",
    "LABEL_BATCH_SCHEMA",
    "LABEL_ROW_SCHEMA",
    "PREDICTION_SCHEMA",
    "PRECOMMIT_SCHEMA",
    "REPLAY_SCHEMA",
    "SAME_FAMILY_GRANT_SCHEMA",
    "SAME_FAMILY_TARGET_TASK_ID",
    "SkeletonGraphCalibrationGap",
    "SkeletonGraphCalibrationExposureAuthorization",
    "SkeletonGraphCalibrationOutcome",
    "SkeletonGraphCalibrationPanelIdentity",
    "SkeletonGraphCalibrationReplayReceipt",
    "SkeletonGraphCalibrationRunnerError",
    "SkeletonGraphCalibrationScope",
    "SkeletonGraphDelayedLabelBatch",
    "SkeletonGraphDelayedLabelRequest",
    "SkeletonGraphDelayedLabelRow",
    "SkeletonGraphFrozenInferenceAddresses",
    "SkeletonGraphPassedFitPaths",
    "SkeletonGraphPopulationGrant",
    "algorithm_digest",
    "authorize_calibration_exposure",
    "cold_replay_calibration",
    "make_verified_inference_runner",
    "run_calibration",
    "source_sha256",
    "verify_calibration_exposure_authorization",
    "verify_and_consume_delayed_label_request",
    "verify_skeleton_graph_population_grant",
)
