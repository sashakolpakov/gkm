"""Fresh exact-unused TRAIN benchmark drill for grounded anchor predicates.

This command is intentionally *not* an official benchmark.  It selects one
exact-unused TRAIN task from each ShapeBongard family, freezes the complete
runtime and source boundary before support pixels are released, and runs the
pure-Python anchor pipeline.  Lean is neither imported nor consulted.

The module owns the campaign-level safety boundary:

* every object is content-addressed and immediately reloaded;
* every model-call group has a durable claim written before the call;
* a claim without a terminal receipt is a terminal infrastructure error on
  resume, never permission to retry;
* query pixels remain sealed until the exact Python predicate freeze and
  decision commit are durable; and
* scoring occurs only after both query predictions are durable.

All live transports and preflight constructors are injectable.  Tests use
synthetic transports and synthetic PNGs only.

The two observer passes test repeatability only.  They are not calibrated
measurements, and an agreed mismatch is not claimed as scientifically
certified absence.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    attest_codex_no_tools,
)
from bongard.corpus import SplitIndex
from bongard.exposure import ExposureLedger
from bongard.historical_exposure import HistoricalExposureSeed, load_historical_exposure
from bongard.object_bongard_batch import (
    ObjectBongardBatchPlan,
    ObjectBongardTaskPlan,
    object_bongard_task_inventory_digest,
    plan_object_bongard_batch,
    verify_object_bongard_batch_plan,
)
from bongard.object_bongard_release_gate import (
    ObjectBongardExecutionPrecommit,
    ObjectBongardReleaseAuthorization,
    ObjectBongardReleaseStore,
    ObjectBongardWriteOnceReceipt,
    PreparedObjectBongardRelease,
    create_object_bongard_execution_precommit,
    prepare_object_bongard_release,
    verify_prepared_object_bongard_release,
)
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.official_panel_archive import OfficialPanelArchive
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.release import OfficialReleaseDescriptor
from bongard.transport import (
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    codex_cli_authenticated_fingerprint,
    snapshot_cloud_policy_cache,
    snapshot_pinned_model_catalog,
)


COMMAND_ID = "bongard.object-scene-anchor-benchmark/exact-unused-train-v1"
RUNTIME_SCHEMA = "gkm.object-scene-anchor-benchmark-runtime.v1"
BOOTSTRAP_SCHEMA = "gkm.object-scene-anchor-benchmark-bootstrap.v1"
CALL_CLAIM_SCHEMA = "gkm.object-scene-anchor-benchmark-call-claim.v1"
CALL_TERMINAL_SCHEMA = "gkm.object-scene-anchor-benchmark-call-terminal.v1"
TASK_RESULT_SCHEMA = "gkm.object-scene-anchor-benchmark-task-result.v1"
QUERY_SCORE_SCHEMA = "gkm.object-scene-anchor-benchmark-query-score.v1"
QUERY_STAGE_SCHEMA = "gkm.object-scene-anchor-benchmark-query-stage.v1"
CAMPAIGN_RESULT_SCHEMA = "gkm.object-scene-anchor-benchmark-result.v1"
CAMPAIGN_REPLAY_SCHEMA = "gkm.object-scene-anchor-benchmark-replay.v1"

MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
DEFAULT_MINUTES = 15
DEFAULT_REQUESTED_PER_FAMILY = 1
DEFAULT_PARALLEL_WORKERS = 3
MAX_PARALLEL_WORKERS = 3
DEFAULT_EXECUTABLE = "codex"
DEFAULT_EXPECTED_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DESCRIPTOR = _REPOSITORY_ROOT / "bongard/data/shape_bongard_v2_release_v1.json"
DEFAULT_ARCHIVE = _REPOSITORY_ROOT / "downloads/ShapeBongard_V2.zip"
DEFAULT_SPLIT = (
    _REPOSITORY_ROOT
    / "downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json"
)
DEFAULT_PREDECESSOR = (
    _REPOSITORY_ROOT
    / "downloads/ShapeBongard_V2_full/prototype_pair_python_campaign_20260807_object_v1"
    / "objects/exposure_successor"
    / "1bcde18e387539f13c4006b4a147e61c75feacb86bb031f10a6e8ba3412fe48d.json"
)
DEFAULT_HISTORICAL_EXPOSURE = _REPOSITORY_ROOT / "bongard/data/historical_exposure_v1.json"

PREDECESSOR_FILE_SHA256 = (
    "1bcde18e387539f13c4006b4a147e61c75feacb86bb031f10a6e8ba3412fe48d"
)
PREDECESSOR_LEDGER_DIGEST = (
    "sha256:73f4f6ad2cdb5413456b4298722cc26cd8de9e733e80e7b178d97b87d11fd276"
)
HISTORICAL_EXPOSURE_SEED_DIGEST = (
    "sha256:0dfa94ada526e47cfe41745125609b7b4e669e1e003d2f5366f740ff50e02ebf"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_STAGE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_MAX_RECORD_BYTES = 64 * 1024 * 1024


class ObjectSceneAnchorBenchmarkError(RuntimeError):
    """The campaign safety boundary or deterministic replay failed closed."""


class ObjectSceneAnchorBenchmarkDanglingClaim(ObjectSceneAnchorBenchmarkError):
    """A prior process may have consumed a call; retry is forbidden."""

    def __init__(self, claim: "ObjectSceneAnchorBenchmarkCallClaim") -> None:
        super().__init__(
            f"dangling call claim for {claim.stage}; retry is forbidden"
        )
        self.claim = claim


def object_scene_anchor_benchmark_command_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_selection_decision_scoring_or_replay": False,
        "official_benchmark_result": False,
        "official_test_authorized": False,
        "evaluation_kind": "exact-unused-train-targeted-engineering-drill",
        "soft_observer_calibrated": False,
        "soft_absence_semantics": "two_pass_model_agreed_mismatch",
        "scientific_certified_absence_claimed": False,
        "replay_scope": "same_python_abi_platform_and_dependency_versions",
        "dependency_binary_contents_bound": False,
    }


def _image_runtime_environment() -> dict[str, str]:
    """Exact interpreter/package-version boundary for image replay."""

    packages: dict[str, str] = {}
    for distribution, key in (
        ("numpy", "numpy_version"),
        ("scipy", "scipy_version"),
        ("Pillow", "pillow_version"),
    ):
        try:
            packages[key] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise ObjectSceneAnchorBenchmarkError(
                f"required image dependency {distribution} is unavailable"
            ) from exc
    cache_tag = sys.implementation.cache_tag
    if not isinstance(cache_tag, str) or not cache_tag:
        raise ObjectSceneAnchorBenchmarkError("Python ABI cache tag is unavailable")
    return {
        "schema": "gkm.object-scene-anchor-image-runtime-environment.v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": cache_tag,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        **packages,
    }


def _image_runtime_configuration(
    environment: Mapping[str, str],
) -> dict[str, str | bool]:
    raw = _canonical_mapping(environment, "image runtime environment")
    required = {
        "schema", "python_implementation", "python_version", "python_cache_tag",
        "platform_system", "platform_machine", "numpy_version", "scipy_version",
        "pillow_version",
    }
    if set(raw) != required or any(not isinstance(value, str) or not value for value in raw.values()):
        raise ObjectSceneAnchorBenchmarkError("image runtime environment fields differ")
    return {
        key: value for key, value in raw.items() if key != "schema"
    } | {
        "replay_scope": "same_python_abi_platform_and_dependency_versions",
        "dependency_binary_contents_bound": False,
    }


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectSceneAnchorBenchmarkError(f"{label} must be a sha256: address")
    return value


def _require_raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectSceneAnchorBenchmarkError(f"{label} must be raw lowercase SHA-256")
    return value


def _object_address(value: str) -> str:
    if _ADDRESS.fullmatch(value):
        return value
    return "sha256:" + _require_raw_digest(value, "artifact digest")


def _canonical_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObjectSceneAnchorBenchmarkError(f"{label} must be a JSON object")
    try:
        return json.loads(canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectSceneAnchorBenchmarkError(f"{label} is not canonical JSON") from exc


def _seal(body: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    raw = _canonical_mapping(body, "record body")
    if digest_field in raw:
        raise ObjectSceneAnchorBenchmarkError("record body already contains digest field")
    return {**raw, digest_field: _address(raw)}


def _verify_seal(
    value: object, *, schema: str, digest_field: str, label: str
) -> dict[str, Any]:
    raw = _canonical_mapping(value, label)
    body = {key: item for key, item in raw.items() if key != digest_field}
    if raw.get("schema") != schema or raw.get(digest_field) != _address(body):
        raise ObjectSceneAnchorBenchmarkError(f"{label} self-seal differs")
    return raw


def _read_bounded_json(path: Path, label: str) -> dict[str, Any]:
    try:
        info = path.lstat()
        payload = path.read_bytes()
    except OSError as exc:
        raise ObjectSceneAnchorBenchmarkError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or not 0 < len(payload) <= _MAX_RECORD_BYTES
    ):
        raise ObjectSceneAnchorBenchmarkError(f"{label} is not a bounded regular file")
    try:
        decoded = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectSceneAnchorBenchmarkError(f"{label} is malformed JSON") from exc
    return _canonical_mapping(decoded, label)


def _persist_typed(
    store: ObjectBongardReleaseStore,
    *,
    object_kind: str,
    value: object,
    digest: str,
    decoder: Callable[[Mapping[str, Any]], object],
) -> tuple[object, ObjectBongardWriteOnceReceipt]:
    to_data = getattr(value, "to_data", None)
    if not callable(to_data):
        raise TypeError("typed artifact must expose to_data()")
    data = _canonical_mapping(to_data(), f"{object_kind} artifact")
    receipt = store.persist(
        object_kind=object_kind,
        object_digest=_object_address(digest),
        data=data,
    )
    restored_data = store.verify(receipt, expected_data=data)
    restored = decoder(restored_data)
    if restored != value:
        raise ObjectSceneAnchorBenchmarkError(
            f"{object_kind} differs after durable reconstruction"
        )
    return restored, receipt


def _persist_record(
    store: ObjectBongardReleaseStore,
    *,
    object_kind: str,
    record: Mapping[str, Any],
    digest_field: str,
    schema: str,
) -> tuple[dict[str, Any], ObjectBongardWriteOnceReceipt]:
    raw = _verify_seal(
        record, schema=schema, digest_field=digest_field, label=object_kind
    )
    receipt = store.persist(
        object_kind=object_kind,
        object_digest=_require_address(raw[digest_field], f"{object_kind} digest"),
        data=raw,
    )
    restored = _verify_seal(
        store.verify(receipt, expected_data=raw),
        schema=schema,
        digest_field=digest_field,
        label=f"stored {object_kind}",
    )
    if restored != raw:
        raise ObjectSceneAnchorBenchmarkError(f"{object_kind} durable replay differs")
    return restored, receipt


def _load_receipted_record(
    store: ObjectBongardReleaseStore,
    receipt_data: object,
    *,
    expected_kind: str,
    schema: str | None = None,
    digest_field: str | None = None,
) -> tuple[dict[str, Any], ObjectBongardWriteOnceReceipt]:
    receipt = ObjectBongardWriteOnceReceipt.from_data(
        _canonical_mapping(receipt_data, f"{expected_kind} receipt")
    )
    if receipt.object_kind != expected_kind:
        raise ObjectSceneAnchorBenchmarkError(f"{expected_kind} receipt kind differs")
    path = store.root / receipt.relative_path
    raw = _read_bounded_json(path, expected_kind)
    restored = _canonical_mapping(store.verify(receipt, expected_data=raw), expected_kind)
    if restored != raw:
        raise ObjectSceneAnchorBenchmarkError(f"{expected_kind} receipt replay differs")
    if schema is not None and digest_field is not None:
        _verify_seal(raw, schema=schema, digest_field=digest_field, label=expected_kind)
        if receipt.object_digest != _object_address(raw[digest_field]):
            raise ObjectSceneAnchorBenchmarkError(
                f"{expected_kind} receipt identity differs"
            )
    return raw, receipt


def _verify_query_stage_record(
    store: ObjectBongardReleaseStore,
    value: object,
    *,
    expected_authorization_digest: str | None = None,
) -> dict[str, Any]:
    raw = _verify_seal(
        value,
        schema=QUERY_STAGE_SCHEMA,
        digest_field="query_stage_digest",
        label="query stage",
    )
    required = {
        "schema", "command_id", "task_plan_digest",
        "release_authorization_digest", "stage", "query_visual_plan_digest",
        "query_visual_plan_receipt", "batch_observer_artifact_digest",
        "batch_observer_artifact_receipt", "query_visual_result_digest",
        "query_visual_result_receipt", "physical_call_count",
        "complete_stage_parent_graph", *_authority_data(), "query_stage_digest",
    }
    if (
        set(raw) != required
        or raw["command_id"] != COMMAND_ID
        or raw["complete_stage_parent_graph"] is not True
        or _STAGE.fullmatch(str(raw["stage"])) is None
        or any(raw[key] != item for key, item in _authority_data().items())
    ):
        raise ObjectSceneAnchorBenchmarkError("query stage fields differ")
    _require_address(raw["task_plan_digest"], "query stage task plan")
    authorization = _require_address(
        raw["release_authorization_digest"], "query stage authorization"
    )
    if expected_authorization_digest is not None and authorization != expected_authorization_digest:
        raise ObjectSceneAnchorBenchmarkError("query stage authorization differs")
    plan_receipt = ObjectBongardWriteOnceReceipt.from_data(
        _canonical_mapping(raw["query_visual_plan_receipt"], "query plan receipt")
    )
    result_receipt = ObjectBongardWriteOnceReceipt.from_data(
        _canonical_mapping(raw["query_visual_result_receipt"], "query result receipt")
    )
    if (
        plan_receipt.object_kind != "anchor-query-visual-plan"
        or plan_receipt.object_digest != _object_address(raw["query_visual_plan_digest"])
        or result_receipt.object_kind != "anchor-query-visual-result"
        or result_receipt.object_digest != _object_address(raw["query_visual_result_digest"])
    ):
        raise ObjectSceneAnchorBenchmarkError("query stage plan/result receipt differs")
    from bongard.object_scene_anchor_batch_observer import (
        ObjectSceneAnchorBatchObserverArtifact,
    )
    from bongard.object_scene_anchor_python_query_visual_execution import (
        ObjectSceneAnchorPythonQueryVisualPlan,
        ObjectSceneAnchorPythonQueryVisualResult,
    )

    plan = _load_typed(
        store, plan_receipt, expected_kind="anchor-query-visual-plan",
        decoder=ObjectSceneAnchorPythonQueryVisualPlan.from_data,
    )
    result = _load_typed(
        store, result_receipt, expected_kind="anchor-query-visual-result",
        decoder=ObjectSceneAnchorPythonQueryVisualResult.from_data,
    )
    if (
        plan.plan_digest != raw["query_visual_plan_digest"]
        or result.result_digest != raw["query_visual_result_digest"]
        or result.plan_digest != plan.plan_digest
        or plan.physical_call_count != raw["physical_call_count"]
    ):
        raise ObjectSceneAnchorBenchmarkError("query stage plan/result parent differs")
    batch_receipt_data = raw["batch_observer_artifact_receipt"]
    batch_digest = raw["batch_observer_artifact_digest"]
    if plan.physical_call_count == 0:
        if batch_receipt_data is not None or batch_digest is not None or result.batch_artifact_digest is not None:
            raise ObjectSceneAnchorBenchmarkError("zero-call query stage has batch artifact")
    else:
        batch_receipt = ObjectBongardWriteOnceReceipt.from_data(
            _canonical_mapping(batch_receipt_data, "query batch receipt")
        )
        if (
            not isinstance(batch_digest, str)
            or batch_receipt.object_kind != "anchor-query-observer-artifact"
            or batch_receipt.object_digest != _object_address(batch_digest)
        ):
            raise ObjectSceneAnchorBenchmarkError("query batch receipt differs")
        batch = _load_typed(
            store, batch_receipt,
            expected_kind="anchor-query-observer-artifact",
            decoder=ObjectSceneAnchorBatchObserverArtifact.from_data,
        )
        if (
            batch.artifact_digest != batch_digest
            or result.batch_artifact_digest != batch.artifact_digest
            or batch.plan_digest != plan.batch_plan_digest
            or batch.observation_plan_digest != plan.observation_context_digest
            or batch.physical_call_count != plan.physical_call_count
        ):
            raise ObjectSceneAnchorBenchmarkError("query batch parent graph differs")
    return raw


def _verify_query_score_record(
    value: object,
    *,
    expected_task_plan_digest: str | None = None,
    expected_predicate_digest: str | None = None,
    expected_query_visual_result_digests: Sequence[str] | None = None,
    expected_prediction_digests: Sequence[str] | None = None,
    expected_predicted_buckets: Sequence[str] | None = None,
) -> dict[str, Any]:
    raw = _verify_seal(
        value,
        schema=QUERY_SCORE_SCHEMA,
        digest_field="score_digest",
        label="query score",
    )
    required = {
        "schema", "command_id", "task_plan_digest", "predicate_digest",
        "query_visual_result_digests", "prediction_digests",
        "predicted_buckets", "expected_buckets", "correct_count",
        "determinate_count", "abstain_count", "error_count",
        "query_denominator", "accuracy_ppm", "coverage_ppm",
        "prediction_digests_validated_before_expected_bucket_access",
        *_authority_data(), "score_digest",
    }
    counts = tuple(
        raw.get(key)
        for key in (
            "correct_count", "determinate_count", "abstain_count",
            "error_count", "query_denominator", "accuracy_ppm", "coverage_ppm",
        )
    )
    result_digests = raw.get("query_visual_result_digests")
    prediction_digests = raw.get("prediction_digests")
    predicted_buckets = raw.get("predicted_buckets")
    expected_buckets = raw.get("expected_buckets")
    allowed_buckets = {"side0_positive", "side1_positive", "abstain", "error"}
    expected_pair = ["side0_positive", "side1_positive"]
    if (
        set(raw) != required
        or raw["command_id"] != COMMAND_ID
        or any(raw[key] != item for key, item in _authority_data().items())
        or any(type(item) is not int or item < 0 for item in counts)
        or raw["query_denominator"] != 2
        or raw["determinate_count"] + raw["abstain_count"] + raw["error_count"] != 2
        or raw["correct_count"] > raw["determinate_count"]
        or raw["accuracy_ppm"] != raw["correct_count"] * 500_000
        or raw["coverage_ppm"] != raw["determinate_count"] * 500_000
        or raw["prediction_digests_validated_before_expected_bucket_access"] is not True
        or not isinstance(result_digests, list)
        or not isinstance(prediction_digests, list)
        or not isinstance(predicted_buckets, list)
        or not isinstance(expected_buckets, list)
        or any(
            len(items) != 2
            for items in (result_digests, prediction_digests, predicted_buckets, expected_buckets)
        )
        or any(not isinstance(item, str) for item in predicted_buckets)
        or any(item not in allowed_buckets for item in predicted_buckets)
        or expected_buckets != expected_pair
    ):
        raise ObjectSceneAnchorBenchmarkError("query score fields differ")
    task_digest = _require_address(raw["task_plan_digest"], "query score task plan")
    predicate_digest = _require_address(raw["predicate_digest"], "query score predicate")
    for index, digest in enumerate(result_digests):
        _require_address(digest, f"query score result {index}")
    for index, digest in enumerate(prediction_digests):
        _require_address(digest, f"query score prediction {index}")
    correct = sum(
        got == wanted
        for got, wanted in zip(predicted_buckets, expected_buckets, strict=True)
    )
    determinate = sum(item in expected_pair for item in predicted_buckets)
    abstain = sum(item == "abstain" for item in predicted_buckets)
    errors = sum(item == "error" for item in predicted_buckets)
    if (
        (raw["correct_count"], raw["determinate_count"], raw["abstain_count"], raw["error_count"])
        != (correct, determinate, abstain, errors)
        or (
            expected_task_plan_digest is not None
            and task_digest != expected_task_plan_digest
        )
        or (
            expected_predicate_digest is not None
            and predicate_digest != expected_predicate_digest
        )
        or (
            expected_query_visual_result_digests is not None
            and result_digests != list(expected_query_visual_result_digests)
        )
        or (
            expected_prediction_digests is not None
            and prediction_digests != list(expected_prediction_digests)
        )
        or (
            expected_predicted_buckets is not None
            and predicted_buckets != list(expected_predicted_buckets)
        )
    ):
        raise ObjectSceneAnchorBenchmarkError("query score parent graph differs")
    return raw


def _only_object(store: ObjectBongardReleaseStore, object_kind: str) -> dict[str, Any]:
    directory = store.root / "objects" / object_kind
    paths = tuple(sorted(directory.glob("*.json"))) if directory.is_dir() else ()
    if len(paths) != 1:
        raise ObjectSceneAnchorBenchmarkError(
            f"resume requires exactly one {object_kind} object"
        )
    return _read_bounded_json(paths[0], object_kind)


def _receipt_data(receipt: ObjectBongardWriteOnceReceipt) -> dict[str, object]:
    return receipt.to_data()


def _claim_content(value: "ObjectSceneAnchorBenchmarkCallClaim") -> dict[str, object]:
    return {
        "schema": CALL_CLAIM_SCHEMA,
        "command_id": COMMAND_ID,
        "release_authorization_digest": value.release_authorization_digest,
        "task_plan_digest": value.task_plan_digest,
        "stage": value.stage,
        "context_digest": value.context_digest,
        "expected_physical_call_count": value.expected_physical_call_count,
        "retry_allowed": False,
        "claim_written_before_call": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBenchmarkCallClaim:
    release_authorization_digest: str
    task_plan_digest: str
    stage: str
    context_digest: str
    expected_physical_call_count: int
    record_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("release authorization digest", self.release_authorization_digest),
            ("task plan digest", self.task_plan_digest),
            ("call context digest", self.context_digest),
            ("claim digest", self.record_digest),
        ):
            _require_address(value, label)
        if _STAGE.fullmatch(self.stage) is None:
            raise ObjectSceneAnchorBenchmarkError("call stage is invalid")
        if (
            type(self.expected_physical_call_count) is not int
            or self.expected_physical_call_count < 0
            or self.record_digest != _address(_claim_content(self))
        ):
            raise ObjectSceneAnchorBenchmarkError("call claim differs")

    @classmethod
    def create(
        cls,
        *,
        release_authorization_digest: str,
        task_plan_digest: str,
        stage: str,
        context_digest: str,
        expected_physical_call_count: int,
    ) -> "ObjectSceneAnchorBenchmarkCallClaim":
        values = {
            "release_authorization_digest": release_authorization_digest,
            "task_plan_digest": task_plan_digest,
            "stage": stage,
            "context_digest": context_digest,
            "expected_physical_call_count": expected_physical_call_count,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(**values, record_digest=_address(_claim_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_claim_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectSceneAnchorBenchmarkCallClaim":
        raw = _canonical_mapping(value, "call claim")
        required = {
            "schema", "command_id", "release_authorization_digest",
            "task_plan_digest", "stage", "context_digest",
            "expected_physical_call_count", "retry_allowed",
            "claim_written_before_call", *_authority_data(), "record_digest",
        }
        if (
            set(raw) != required
            or raw["schema"] != CALL_CLAIM_SCHEMA
            or raw["command_id"] != COMMAND_ID
            or raw["retry_allowed"] is not False
            or raw["claim_written_before_call"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectSceneAnchorBenchmarkError("call claim fields differ")
        result = cls(
            raw["release_authorization_digest"], raw["task_plan_digest"],
            raw["stage"], raw["context_digest"],
            raw["expected_physical_call_count"], raw["record_digest"],
        )
        if result.to_data() != raw:
            raise ObjectSceneAnchorBenchmarkError("call claim is not canonical")
        return result


def _terminal_content(
    value: "ObjectSceneAnchorBenchmarkCallTerminal",
) -> dict[str, object]:
    return {
        "schema": CALL_TERMINAL_SCHEMA,
        "command_id": COMMAND_ID,
        "claim_digest": value.claim_digest,
        "release_authorization_digest": value.release_authorization_digest,
        "task_plan_digest": value.task_plan_digest,
        "stage": value.stage,
        "context_digest": value.context_digest,
        "status": value.status,
        "physical_call_slots_consumed": value.physical_call_slots_consumed,
        "artifact_receipt": value.artifact_receipt,
        "failure_type": value.failure_type,
        "retry_allowed": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBenchmarkCallTerminal:
    claim_digest: str
    release_authorization_digest: str
    task_plan_digest: str
    stage: str
    context_digest: str
    status: str
    physical_call_slots_consumed: int
    artifact_receipt: Mapping[str, Any] | None
    failure_type: str | None
    record_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("claim digest", self.claim_digest),
            ("release authorization digest", self.release_authorization_digest),
            ("task plan digest", self.task_plan_digest),
            ("call context digest", self.context_digest),
            ("terminal digest", self.record_digest),
        ):
            _require_address(value, label)
        if _STAGE.fullmatch(self.stage) is None or self.status not in ("success", "error"):
            raise ObjectSceneAnchorBenchmarkError("call terminal stage/status differs")
        if type(self.physical_call_slots_consumed) is not int or self.physical_call_slots_consumed < 0:
            raise ObjectSceneAnchorBenchmarkError("call terminal count differs")
        if self.status == "success":
            if self.artifact_receipt is None or self.failure_type is not None:
                raise ObjectSceneAnchorBenchmarkError("successful call terminal differs")
            ObjectBongardWriteOnceReceipt.from_data(self.artifact_receipt)
        elif (
            self.artifact_receipt is not None
            or not isinstance(self.failure_type, str)
            or not self.failure_type
        ):
            raise ObjectSceneAnchorBenchmarkError("error call terminal differs")
        if self.record_digest != _address(_terminal_content(self)):
            raise ObjectSceneAnchorBenchmarkError("call terminal digest differs")

    @classmethod
    def create(
        cls,
        claim: ObjectSceneAnchorBenchmarkCallClaim,
        *,
        status: str,
        artifact_receipt: ObjectBongardWriteOnceReceipt | None,
        failure_type: str | None,
    ) -> "ObjectSceneAnchorBenchmarkCallTerminal":
        values = {
            "claim_digest": claim.record_digest,
            "release_authorization_digest": claim.release_authorization_digest,
            "task_plan_digest": claim.task_plan_digest,
            "stage": claim.stage,
            "context_digest": claim.context_digest,
            "status": status,
            "physical_call_slots_consumed": claim.expected_physical_call_count,
            "artifact_receipt": (
                None if artifact_receipt is None else artifact_receipt.to_data()
            ),
            "failure_type": failure_type,
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return cls(**values, record_digest=_address(_terminal_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_terminal_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ObjectSceneAnchorBenchmarkCallTerminal":
        raw = _canonical_mapping(value, "call terminal")
        required = {
            "schema", "command_id", "claim_digest", "release_authorization_digest",
            "task_plan_digest", "stage", "context_digest", "status",
            "physical_call_slots_consumed", "artifact_receipt", "failure_type",
            "retry_allowed", *_authority_data(), "record_digest",
        }
        if (
            set(raw) != required
            or raw["schema"] != CALL_TERMINAL_SCHEMA
            or raw["command_id"] != COMMAND_ID
            or raw["retry_allowed"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectSceneAnchorBenchmarkError("call terminal fields differ")
        result = cls(
            raw["claim_digest"], raw["release_authorization_digest"],
            raw["task_plan_digest"], raw["stage"], raw["context_digest"],
            raw["status"], raw["physical_call_slots_consumed"],
            None if raw["artifact_receipt"] is None else dict(raw["artifact_receipt"]),
            raw["failure_type"], raw["record_digest"],
        )
        if result.to_data() != raw:
            raise ObjectSceneAnchorBenchmarkError("call terminal is not canonical")
        return result


def _require_successful_stage_claim(
    store: ObjectBongardReleaseStore,
    task_result: Mapping[str, Any],
    *,
    release_authorization_digest: str,
    stage: str,
    context_digest: str,
    expected_physical_call_count: int,
    artifact_receipt: ObjectBongardWriteOnceReceipt,
) -> None:
    """Verify a result's exact durable claim, terminal, and artifact edge."""

    task_plan_digest = _require_address(
        task_result["task_plan_digest"], "stage claim task plan"
    )
    claim_digests = _canonical_mapping(
        task_result["call_claim_digests"], "task call claims"
    )
    terminal_digests = _canonical_mapping(
        task_result["call_terminal_digests"], "task call terminals"
    )
    try:
        claim_digest = _require_address(
            claim_digests[stage], f"{stage} claim digest"
        )
        terminal_digest = _require_address(
            terminal_digests[stage], f"{stage} terminal digest"
        )
    except KeyError as exc:
        raise ObjectSceneAnchorBenchmarkError(
            f"task omits the {stage} call lineage"
        ) from exc
    claim_path = (
        store.root / "objects" / "anchor-call-claim"
        / f"{claim_digest[7:]}.json"
    )
    terminal_path = (
        store.root / "objects" / "anchor-call-terminal"
        / f"{terminal_digest[7:]}.json"
    )
    claim = ObjectSceneAnchorBenchmarkCallClaim.from_data(
        _read_bounded_json(claim_path, f"{stage} call claim")
    )
    terminal = ObjectSceneAnchorBenchmarkCallTerminal.from_data(
        _read_bounded_json(terminal_path, f"{stage} call terminal")
    )
    if (
        claim.record_digest != claim_digest
        or claim.release_authorization_digest != release_authorization_digest
        or claim.task_plan_digest != task_plan_digest
        or claim.stage != stage
        or claim.context_digest != context_digest
        or claim.expected_physical_call_count != expected_physical_call_count
        or terminal.record_digest != terminal_digest
        or terminal.claim_digest != claim.record_digest
        or terminal.release_authorization_digest != release_authorization_digest
        or terminal.task_plan_digest != task_plan_digest
        or terminal.stage != stage
        or terminal.context_digest != context_digest
        or terminal.status != "success"
        or terminal.physical_call_slots_consumed != expected_physical_call_count
        or terminal.artifact_receipt != artifact_receipt.to_data()
        or terminal.failure_type is not None
    ):
        raise ObjectSceneAnchorBenchmarkError(
            f"{stage} call lineage differs from its exact parents"
        )


def _failure_type(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name) else "UnclassifiedError"


@dataclass(slots=True)
class ObjectSceneAnchorBenchmarkCallJournal:
    store: ObjectBongardReleaseStore
    release_authorization_digest: str

    def __post_init__(self) -> None:
        _require_address(self.release_authorization_digest, "release authorization digest")
        lock_root = self.store.root / "benchmark_call_journal" / "locks"
        lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if lock_root.resolve(strict=True) != lock_root:
            raise ObjectSceneAnchorBenchmarkError("call journal lock root is unsafe")

    def _terminal_for_claim(
        self, claim: ObjectSceneAnchorBenchmarkCallClaim
    ) -> ObjectSceneAnchorBenchmarkCallTerminal | None:
        directory = self.store.root / "objects" / "anchor-call-terminal"
        matches: list[ObjectSceneAnchorBenchmarkCallTerminal] = []
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                terminal = ObjectSceneAnchorBenchmarkCallTerminal.from_data(
                    _read_bounded_json(path, "call terminal")
                )
                if terminal.claim_digest == claim.record_digest:
                    matches.append(terminal)
        if len(matches) > 1:
            raise ObjectSceneAnchorBenchmarkError("call claim has multiple terminal receipts")
        return None if not matches else matches[0]

    def _claims_for_scope(
        self,
        *,
        task_plan_digest: str,
        stage: str,
    ) -> tuple[ObjectSceneAnchorBenchmarkCallClaim, ...]:
        directory = self.store.root / "objects" / "anchor-call-claim"
        if not directory.is_dir():
            return ()
        matches = []
        for path in sorted(directory.glob("*.json")):
            claim = ObjectSceneAnchorBenchmarkCallClaim.from_data(
                _read_bounded_json(path, "call claim")
            )
            if (
                claim.release_authorization_digest
                == self.release_authorization_digest
                and claim.task_plan_digest == task_plan_digest
                and claim.stage == stage
            ):
                matches.append(claim)
        return tuple(matches)

    def run(
        self,
        *,
        task_plan_digest: str,
        stage: str,
        context_digest: str,
        expected_physical_call_count: int,
        object_kind: str,
        invoke_and_persist: Callable[[], tuple[object, ObjectBongardWriteOnceReceipt]],
        load_artifact: Callable[[ObjectBongardWriteOnceReceipt], object],
    ) -> tuple[
        object,
        ObjectSceneAnchorBenchmarkCallClaim,
        ObjectSceneAnchorBenchmarkCallTerminal,
        bool,
    ]:
        """Run once, reuse a completed terminal, or reject a dangling claim."""

        claim = ObjectSceneAnchorBenchmarkCallClaim.create(
            release_authorization_digest=self.release_authorization_digest,
            task_plan_digest=task_plan_digest,
            stage=stage,
            context_digest=context_digest,
            expected_physical_call_count=expected_physical_call_count,
        )
        scope_digest = canonical_digest(
            {
                "schema": "gkm.object-scene-anchor-call-scope.v1",
                "release_authorization_digest": self.release_authorization_digest,
                "task_plan_digest": task_plan_digest,
                "stage": stage,
            }
        )
        lock_path = (
            self.store.root / "benchmark_call_journal" / "locks"
            / f"{scope_digest}.lock"
        )
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            scoped_claims = self._claims_for_scope(
                task_plan_digest=task_plan_digest, stage=stage
            )
            if len(scoped_claims) > 1:
                raise ObjectSceneAnchorBenchmarkError(
                    "call scope has multiple durable claims"
                )
            if scoped_claims and scoped_claims[0] != claim:
                raise ObjectSceneAnchorBenchmarkDanglingClaim(scoped_claims[0])
            terminal = self._terminal_for_claim(claim)
            if terminal is not None:
                if (
                    terminal.release_authorization_digest
                    != claim.release_authorization_digest
                    or terminal.task_plan_digest != claim.task_plan_digest
                    or terminal.stage != claim.stage
                    or terminal.context_digest != claim.context_digest
                    or terminal.physical_call_slots_consumed
                    != claim.expected_physical_call_count
                ):
                    raise ObjectSceneAnchorBenchmarkError(
                        "call terminal bindings differ from exact claim"
                    )
                if terminal.status != "success" or terminal.artifact_receipt is None:
                    raise ObjectSceneAnchorBenchmarkError(
                        f"prior {stage} call terminated with {terminal.failure_type}"
                    )
                receipt = ObjectBongardWriteOnceReceipt.from_data(
                    terminal.artifact_receipt
                )
                if receipt.object_kind != object_kind:
                    raise ObjectSceneAnchorBenchmarkError(
                        f"reused {stage} artifact kind differs"
                    )
                return load_artifact(receipt), claim, terminal, True
            if scoped_claims:
                raise ObjectSceneAnchorBenchmarkDanglingClaim(claim)
            _persist_typed(
                self.store,
                object_kind="anchor-call-claim",
                value=claim,
                digest=claim.record_digest,
                decoder=ObjectSceneAnchorBenchmarkCallClaim.from_data,
            )
            try:
                artifact, artifact_receipt = invoke_and_persist()
                if artifact_receipt.object_kind != object_kind:
                    raise ObjectSceneAnchorBenchmarkError(
                        f"{stage} persisted the wrong artifact kind"
                    )
            except BaseException as exc:
                terminal = ObjectSceneAnchorBenchmarkCallTerminal.create(
                    claim,
                    status="error",
                    artifact_receipt=None,
                    failure_type=_failure_type(exc),
                )
                _persist_typed(
                    self.store,
                    object_kind="anchor-call-terminal",
                    value=terminal,
                    digest=terminal.record_digest,
                    decoder=ObjectSceneAnchorBenchmarkCallTerminal.from_data,
                )
                raise
            terminal = ObjectSceneAnchorBenchmarkCallTerminal.create(
                claim,
                status="success",
                artifact_receipt=artifact_receipt,
                failure_type=None,
            )
            _persist_typed(
                self.store,
                object_kind="anchor-call-terminal",
                value=terminal,
                digest=terminal.record_digest,
                decoder=ObjectSceneAnchorBenchmarkCallTerminal.from_data,
            )
            return artifact, claim, terminal, False
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _archive_task_ids(archive: OfficialPanelArchive) -> tuple[str, ...]:
    tasks: set[str] = set()
    for member, _size, _crc in archive.members:
        parts = member.split("/")
        if (
            len(parts) == 6
            and parts[0] == "ShapeBongard_V2"
            and parts[2] == "images"
            and parts[4] in ("0", "1")
            and parts[5].endswith(".png")
        ):
            tasks.add(parts[3])
    if not tasks:
        raise ObjectSceneAnchorBenchmarkError("official archive contains no task inventory")
    return tuple(sorted(tasks))


def _runtime_record(
    runtime: ObjectBongardTurnRuntime,
    launcher_fingerprint: Mapping[str, str],
    *,
    exposure_observed_at: str,
) -> dict[str, Any]:
    cache = runtime.cloud_policy_cache_snapshot
    return _seal(
        {
            "schema": RUNTIME_SCHEMA,
            "command_id": COMMAND_ID,
            "runtime_binding": runtime.binding,
            "cloud_policy_cache_snapshot_base64": (
                None
                if cache is None or cache.data is None
                else base64.b64encode(cache.data).decode("ascii")
            ),
            "model_catalog_snapshot_base64": base64.b64encode(
                runtime.model_catalog_snapshot.data
            ).decode("ascii"),
            "no_tools_attestation": runtime.no_tools_attestation.to_dict(),
            "launcher_fingerprint": dict(launcher_fingerprint),
            "image_runtime_environment": _image_runtime_environment(),
            "exposure_observed_at": exposure_observed_at,
            "persisted_before_support_release": True,
            **_authority_data(),
        },
        "runtime_digest",
    )


def _restore_runtime(record: Mapping[str, Any]) -> ObjectBongardTurnRuntime:
    raw = _verify_seal(
        record, schema=RUNTIME_SCHEMA, digest_field="runtime_digest", label="runtime"
    )
    binding = _canonical_mapping(raw.get("runtime_binding"), "runtime binding")
    if (
        _canonical_mapping(
            raw.get("image_runtime_environment"), "image runtime environment"
        )
        != _image_runtime_environment()
        or any(raw.get(key) != item for key, item in _authority_data().items())
    ):
        raise ObjectSceneAnchorBenchmarkError(
            "runtime image environment or authority differs"
        )

    def decode(value: object, label: str) -> bytes | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ObjectSceneAnchorBenchmarkError(f"{label} snapshot differs")
        try:
            result = base64.b64decode(value.encode("ascii"), validate=True)
        except (UnicodeError, ValueError) as exc:
            raise ObjectSceneAnchorBenchmarkError(f"{label} snapshot differs") from exc
        if base64.b64encode(result).decode("ascii") != value:
            raise ObjectSceneAnchorBenchmarkError(f"{label} snapshot is not canonical")
        return result

    catalog_data = decode(raw.get("model_catalog_snapshot_base64"), "model catalog")
    if catalog_data is None:
        raise ObjectSceneAnchorBenchmarkError("model catalog snapshot is absent")
    cache_data = decode(raw.get("cloud_policy_cache_snapshot_base64"), "policy cache")
    cache_present = binding.get("cloud_policy_cache_snapshot_present")
    if type(cache_present) is not bool:
        raise ObjectSceneAnchorBenchmarkError("policy-cache presence binding differs")
    runtime = ObjectBongardTurnRuntime(
        model=binding["model"],
        reasoning_effort=binding["reasoning_effort"],
        minutes=binding["minutes"],
        verbose=binding["verbose"],
        executable=binding["executable"],
        cloud_policy_cache_snapshot=(
            CloudPolicyCacheSnapshot(cache_data) if cache_present else None
        ),
        model_catalog_snapshot=CodexModelCatalogSnapshot(catalog_data),
        expected_launcher_digest=binding["expected_launcher_digest"],
        no_tools_attestation=CodexNoToolsAttestation.from_mapping(
            raw["no_tools_attestation"]
        ),
        transport_source_digest=binding["transport_source_digest"],
    )
    if runtime.binding != binding:
        raise ObjectSceneAnchorBenchmarkError("runtime binding differs on replay")
    return runtime


def _source_bindings() -> dict[str, str]:
    """Resolve the complete active source set at the pre-exposure boundary."""

    from bongard.object_bongard_batch import object_bongard_batch_source_digest
    from bongard.object_bongard_release_gate import object_bongard_release_gate_source_digest
    from bongard.object_scene_anchor_batch_observer import (
        object_scene_anchor_batch_observer_source_digest,
    )
    from bongard.object_scene_anchor_candidate_ranker import (
        object_scene_anchor_candidate_ranker_source_digest,
        object_scene_anchor_candidate_ranker_transport_source_digest,
    )
    from bongard.object_scene_anchor_card_proposer import (
        object_scene_anchor_card_proposer_source_digest,
        object_scene_anchor_card_proposer_transport_source_digest,
    )
    from bongard.object_scene_anchor_python_bridge import (
        object_scene_anchor_python_bridge_source_digest,
    )
    from bongard.object_scene_anchor_python_predicate import (
        object_scene_anchor_python_predicate_source_digest,
    )
    from bongard.object_scene_anchor_python_query_observation import (
        object_scene_anchor_python_query_source_digest,
    )
    from bongard.object_scene_anchor_support_observation_join import (
        object_scene_anchor_support_observation_join_source_digest,
    )
    from bongard.object_scene_anchor_support_preparation import (
        object_scene_anchor_support_preparation_source_digest,
    )
    from bongard.object_scene_anchor_task_support_adapter import (
        object_scene_anchor_task_support_adapter_source_digest,
    )
    from bongard.object_scene_anchor_version_space import (
        object_scene_anchor_version_space_source_digest,
    )

    values = {
        "benchmark_command": object_scene_anchor_benchmark_command_source_digest(),
        "batch": object_bongard_batch_source_digest(),
        "release_gate": object_bongard_release_gate_source_digest(),
        "support_preparation": object_scene_anchor_support_preparation_source_digest(),
        "task_support_adapter": object_scene_anchor_task_support_adapter_source_digest(),
        "card_proposer": object_scene_anchor_card_proposer_source_digest(),
        "card_proposer_transport": object_scene_anchor_card_proposer_transport_source_digest(),
        "support_observation_join": object_scene_anchor_support_observation_join_source_digest(),
        "batch_observer": object_scene_anchor_batch_observer_source_digest(),
        "candidate_ranker": object_scene_anchor_candidate_ranker_source_digest(),
        "candidate_ranker_transport": object_scene_anchor_candidate_ranker_transport_source_digest(),
        "version_space": object_scene_anchor_version_space_source_digest(),
        "python_bridge": object_scene_anchor_python_bridge_source_digest(),
        "python_predicate": object_scene_anchor_python_predicate_source_digest(),
        "python_query_observation": object_scene_anchor_python_query_source_digest(),
    }
    # These two modules are independently landed dependencies.  Lazy imports
    # keep this source reviewable while preserving a fail-closed live launch.
    try:
        from bongard.object_scene_anchor_task_decision_custody import (
            object_scene_anchor_task_decision_custody_source_digest,
        )
        from bongard.object_scene_anchor_python_query_visual_execution import (
            object_scene_anchor_python_query_visual_execution_source_digest,
        )
    except ImportError as exc:
        raise ObjectSceneAnchorBenchmarkError(
            "query custody/execution modules are unavailable"
        ) from exc
    values["task_decision_custody"] = (
        object_scene_anchor_task_decision_custody_source_digest()
    )
    values["query_visual_execution"] = (
        object_scene_anchor_python_query_visual_execution_source_digest()
    )
    result = {key: _object_address(value) for key, value in values.items()}
    result["image_runtime_environment"] = _address(_image_runtime_environment())
    return result


@dataclass(frozen=True, slots=True)
class PreparedObjectSceneAnchorBenchmark:
    output_root: Path
    plan: ObjectBongardBatchPlan
    descriptor: OfficialReleaseDescriptor
    archive: OfficialPanelArchive = field(repr=False, compare=False)
    split: SplitIndex = field(repr=False, compare=False)
    predecessor: ExposureLedger
    historical_exposure: HistoricalExposureSeed = field(repr=False, compare=False)
    runtime: ObjectBongardTurnRuntime = field(repr=False, compare=False)
    runtime_record: Mapping[str, Any]
    runtime_receipt: ObjectBongardWriteOnceReceipt
    source_manifest: object
    source_manifest_receipt: ObjectBongardWriteOnceReceipt
    precommit: ObjectBongardExecutionPrecommit
    release: PreparedObjectBongardRelease
    bootstrap: Mapping[str, Any]
    bootstrap_receipt: ObjectBongardWriteOnceReceipt


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_runtime(
    *,
    minutes: int,
    verbose: bool,
    executable: str,
    expected_launcher_sha256: str,
    cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot],
    catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot],
    launcher_fingerprinter: Callable[..., Mapping[str, str]],
    runtime_attester: Callable[..., CodexNoToolsAttestation],
) -> tuple[ObjectBongardTurnRuntime, Mapping[str, str]]:
    cache = cache_snapshotter()
    catalog = catalog_snapshotter()
    fingerprint = launcher_fingerprinter(
        executable, expected_launcher_digest=expected_launcher_sha256
    )
    attestation = runtime_attester(
        executable=executable,
        expected_launcher_digest=expected_launcher_sha256,
        model_catalog_snapshot=catalog,
        cloud_policy_cache_snapshot=cache,
    )
    from bongard.object_scene_anchor_card_proposer import (
        object_scene_anchor_card_proposer_transport_source_digest,
    )

    runtime = ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        cloud_policy_cache_snapshot=cache,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=expected_launcher_sha256,
        no_tools_attestation=attestation,
        transport_source_digest=object_scene_anchor_card_proposer_transport_source_digest(),
    )
    return runtime, fingerprint


def prepare_object_scene_anchor_benchmark(
    *,
    output_root: str | os.PathLike[str],
    selection_seed: str,
    descriptor_path: str | os.PathLike[str] = DEFAULT_DESCRIPTOR,
    archive_path: str | os.PathLike[str] = DEFAULT_ARCHIVE,
    split_path: str | os.PathLike[str] = DEFAULT_SPLIT,
    predecessor_path: str | os.PathLike[str] = DEFAULT_PREDECESSOR,
    historical_exposure_path: str | os.PathLike[str] = DEFAULT_HISTORICAL_EXPOSURE,
    requested_per_family: int = DEFAULT_REQUESTED_PER_FAMILY,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    minutes: int = DEFAULT_MINUTES,
    verbose: bool = False,
    executable: str = DEFAULT_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_EXPECTED_LAUNCHER_SHA256,
    exposure_observed_at: str | None = None,
    resume: bool = False,
    cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot] = snapshot_cloud_policy_cache,
    catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot] = snapshot_pinned_model_catalog,
    launcher_fingerprinter: Callable[..., Mapping[str, str]] = codex_cli_authenticated_fingerprint,
    runtime_attester: Callable[..., CodexNoToolsAttestation] = attest_codex_no_tools,
) -> PreparedObjectSceneAnchorBenchmark:
    """Freeze a deterministic clean-TRAIN cohort and prepare its release."""

    if (
        not isinstance(selection_seed, str)
        or not selection_seed
        or selection_seed != selection_seed.strip()
    ):
        raise ObjectSceneAnchorBenchmarkError("a deterministic nonempty CLI seed is required")
    if type(requested_per_family) is not int or requested_per_family <= 0:
        raise ObjectSceneAnchorBenchmarkError("requested_per_family must be positive")
    if type(parallel_workers) is not int or not 1 <= parallel_workers <= MAX_PARALLEL_WORKERS:
        raise ObjectSceneAnchorBenchmarkError("parallel_workers must lie in 1..3")
    if requested_per_family == 1 and parallel_workers > 3:
        raise ObjectSceneAnchorBenchmarkError("three-task default cannot exceed three workers")
    _require_raw_digest(expected_launcher_sha256, "expected launcher digest")

    root = Path(os.path.abspath(os.path.expanduser(str(output_root))))
    if resume:
        if not root.is_dir() or root.resolve(strict=True) != root:
            raise ObjectSceneAnchorBenchmarkError("resume root is unavailable or unsafe")
    else:
        try:
            root.mkdir(mode=0o700, parents=False, exist_ok=False)
        except OSError as exc:
            raise ObjectSceneAnchorBenchmarkError("output root must be fresh") from exc
    store = ObjectBongardReleaseStore(root)

    descriptor = OfficialReleaseDescriptor.load(descriptor_path)
    archive = OfficialPanelArchive.load(
        descriptor,
        archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )
    descriptor.verify_split(split_path)
    split = SplitIndex.load(split_path)
    task_ids = _archive_task_ids(archive)
    split.validate(task_ids, official_counts=True)
    train_task_ids = tuple(split.canonical_groups["train"])

    predecessor_file = Path(predecessor_path)
    if _file_sha256(predecessor_file) != PREDECESSOR_FILE_SHA256:
        raise ObjectSceneAnchorBenchmarkError("exposure predecessor file identity differs")
    predecessor = ExposureLedger.load(predecessor_file)
    if predecessor.digest != PREDECESSOR_LEDGER_DIGEST:
        raise ObjectSceneAnchorBenchmarkError("exposure predecessor ledger differs")
    historical = load_historical_exposure(historical_exposure_path, verify_evidence=False)
    if historical.seed_digest != HISTORICAL_EXPOSURE_SEED_DIGEST:
        raise ObjectSceneAnchorBenchmarkError("historical exposure seed identity differs")
    exact_used = tuple(
        sorted(set(predecessor.exposed_task_ids) | set(historical.exact_official_task_ids))
    )
    plan = plan_object_bongard_batch(
        task_ids=task_ids,
        train_task_ids=train_task_ids,
        exact_used_task_ids=exact_used,
        selection_seed=selection_seed,
        requested_per_family=requested_per_family,
        release_descriptor_digest=descriptor.digest,
        split_source_digest=split.source_digest,
        task_inventory_digest=object_bongard_task_inventory_digest(task_ids),
        exposure_predecessor_digest=predecessor.digest,
        historical_exposure_digest=historical.seed_digest,
    )
    verify_object_bongard_batch_plan(
        plan,
        task_ids=task_ids,
        train_task_ids=train_task_ids,
        exact_used_task_ids=exact_used,
        selection_seed=selection_seed,
    )

    if resume:
        runtime_raw = _only_object(store, "anchor-benchmark-runtime")
        persisted_runtime = _restore_runtime(runtime_raw)
        timestamp = runtime_raw.get("exposure_observed_at")
        if not isinstance(timestamp, str) or not timestamp:
            raise ObjectSceneAnchorBenchmarkError("persisted exposure timestamp differs")
        runtime, fresh_fingerprint = _create_runtime(
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            expected_launcher_sha256=expected_launcher_sha256,
            cache_snapshotter=cache_snapshotter,
            catalog_snapshotter=catalog_snapshotter,
            launcher_fingerprinter=launcher_fingerprinter,
            runtime_attester=runtime_attester,
        )
        if (
            runtime != persisted_runtime
            or dict(fresh_fingerprint) != runtime_raw.get("launcher_fingerprint")
        ):
            raise ObjectSceneAnchorBenchmarkError(
                "fresh resume attestation differs from the persisted runtime"
            )
        runtime_receipt = store.persist(
            object_kind="anchor-benchmark-runtime",
            object_digest=runtime_raw["runtime_digest"],
            data=runtime_raw,
        )
        store.verify(runtime_receipt, expected_data=runtime_raw)
    else:
        timestamp = exposure_observed_at or datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        runtime, fingerprint = _create_runtime(
            minutes=minutes,
            verbose=verbose,
            executable=executable,
            expected_launcher_sha256=expected_launcher_sha256,
            cache_snapshotter=cache_snapshotter,
            catalog_snapshotter=catalog_snapshotter,
            launcher_fingerprinter=launcher_fingerprinter,
            runtime_attester=runtime_attester,
        )
        runtime_raw, runtime_receipt = _persist_record(
            store,
            object_kind="anchor-benchmark-runtime",
            record=_runtime_record(runtime, fingerprint, exposure_observed_at=timestamp),
            digest_field="runtime_digest",
            schema=RUNTIME_SCHEMA,
        )

    if (
        runtime.model != MODEL
        or runtime.reasoning_effort != REASONING_EFFORT
        or runtime.minutes != minutes
        or runtime.verbose != verbose
        or runtime.executable != executable
        or runtime.expected_launcher_digest != expected_launcher_sha256
    ):
        raise ObjectSceneAnchorBenchmarkError("runtime selectors differ from launch request")
    from bongard.object_scene_anchor_source_manifest import (
        ObjectSceneAnchorSourceManifest,
        build_object_scene_anchor_source_manifest,
        cold_verify_object_scene_anchor_source_manifest,
    )

    if resume:
        source_manifest = ObjectSceneAnchorSourceManifest.from_data(
            _only_object(store, "anchor-source-manifest")
        )
        cold_verify_object_scene_anchor_source_manifest(
            source_manifest,
            repository_root=_REPOSITORY_ROOT,
            expected_manifest_digest=source_manifest.manifest_digest,
        )
    else:
        source_manifest = build_object_scene_anchor_source_manifest(
            repository_root=_REPOSITORY_ROOT
        )
    source_manifest, source_manifest_receipt = _persist_typed(
        store,
        object_kind="anchor-source-manifest",
        value=source_manifest,
        digest=source_manifest.manifest_digest,
        decoder=ObjectSceneAnchorSourceManifest.from_data,
    )
    bindings = _source_bindings()
    bindings["runtime_record"] = runtime_raw["runtime_digest"]
    bindings["source_manifest"] = _object_address(source_manifest.manifest_digest)
    bindings["source_manifest_receipt"] = source_manifest_receipt.record_digest
    image_environment = _image_runtime_environment()
    configuration: dict[str, str | int | bool] = {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "minutes": minutes,
        "verbose": verbose,
        "requested_per_family": requested_per_family,
        "parallel_workers": parallel_workers,
        "selection_seed_digest": plan.selection_seed_digest,
        "headless": True,
        "support_only_synthesis": True,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "soft_observer_calibrated": False,
        "soft_absence_semantics": "two_pass_model_agreed_mismatch",
        "scientific_certified_absence_claimed": False,
        **_image_runtime_configuration(image_environment),
    }
    precommit = create_object_bongard_execution_precommit(
        plan=plan,
        predecessor=predecessor,
        descriptor=descriptor,
        archive=archive,
        task_ids=task_ids,
        train_task_ids=train_task_ids,
        exact_used_task_ids=exact_used,
        runtime_source_bindings=bindings,
        configuration=configuration,
        exposure_observed_at=timestamp,
        exposure_actor="headless-codex-anchor-proposer",
        exposure_purpose="exact-unused-train-anchor-engineering-drill",
        exposure_source="official-shapebongard-v2-archive",
    )
    if resume:
        persisted_bootstrap = _verify_seal(
            _only_object(store, "anchor-benchmark-bootstrap"),
            schema=BOOTSTRAP_SCHEMA,
            digest_field="bootstrap_digest",
            label="persisted bootstrap",
        )
        persisted_precommit_raw = _only_object(store, "execution-precommit")
        persisted_precommit = ObjectBongardExecutionPrecommit.from_data(
            persisted_precommit_raw
        )
        if (
            persisted_bootstrap.get("batch_plan_digest") != plan.record_digest
            or persisted_bootstrap.get("runtime_digest") != runtime_raw["runtime_digest"]
            or persisted_bootstrap.get("execution_precommit_digest")
            != precommit.record_digest
            or persisted_precommit != precommit
        ):
            raise ObjectSceneAnchorBenchmarkError(
                "resume plan, runtime, configuration, or active source differs"
            )
    release = prepare_object_bongard_release(
        store=store, plan=plan, precommit=precommit, predecessor=predecessor
    )
    verify_prepared_object_bongard_release(release)

    bootstrap_body = {
        "schema": BOOTSTRAP_SCHEMA,
        "command_id": COMMAND_ID,
        "batch_plan_digest": plan.record_digest,
        "execution_precommit_digest": precommit.record_digest,
        "release_authorization_digest": release.authorization.record_digest,
        "exposure_predecessor_digest": predecessor.digest,
        "exposure_successor_digest": release.successor.digest,
        "historical_exposure_digest": historical.seed_digest,
        "runtime_digest": runtime_raw["runtime_digest"],
        "runtime_receipt": runtime_receipt.to_data(),
        "source_manifest_digest": source_manifest.manifest_digest,
        "source_manifest_receipt": source_manifest_receipt.to_data(),
        "plan_receipt": release.plan_receipt.to_data(),
        "precommit_receipt": release.precommit_receipt.to_data(),
        "exposure_receipt": release.exposure_receipt.to_data(),
        "authorization_receipt": release.authorization_receipt.to_data(),
        "selected_task_count": len(plan.tasks),
        "requested_per_family": requested_per_family,
        "selection_seed_digest": plan.selection_seed_digest,
        "support_pixels_released": False,
        "query_pixels_released": False,
        "prepared_before_any_panel_or_model_call": True,
        **_authority_data(),
    }
    bootstrap = _seal(bootstrap_body, "bootstrap_digest")
    bootstrap, bootstrap_receipt = _persist_record(
        store,
        object_kind="anchor-benchmark-bootstrap",
        record=bootstrap,
        digest_field="bootstrap_digest",
        schema=BOOTSTRAP_SCHEMA,
    )
    return PreparedObjectSceneAnchorBenchmark(
        root, plan, descriptor, archive, split, predecessor, historical, runtime,
        runtime_raw, runtime_receipt, source_manifest, source_manifest_receipt,
        precommit, release, bootstrap,
        bootstrap_receipt,
    )


def load_prepared_object_scene_anchor_benchmark_for_replay(
    *,
    output_root: str | os.PathLike[str],
    descriptor_path: str | os.PathLike[str] = DEFAULT_DESCRIPTOR,
    archive_path: str | os.PathLike[str] = DEFAULT_ARCHIVE,
    split_path: str | os.PathLike[str] = DEFAULT_SPLIT,
    predecessor_path: str | os.PathLike[str] = DEFAULT_PREDECESSOR,
    historical_exposure_path: str | os.PathLike[str] = DEFAULT_HISTORICAL_EXPOSURE,
) -> PreparedObjectSceneAnchorBenchmark:
    """Reconstruct the prepared campaign from disk with zero live preflight."""

    root = Path(os.path.abspath(os.path.expanduser(str(output_root))))
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise ObjectSceneAnchorBenchmarkError("replay root is unavailable or unsafe")
    store = ObjectBongardReleaseStore(root)
    plan = ObjectBongardBatchPlan.from_data(_only_object(store, "batch-plan"))
    precommit = ObjectBongardExecutionPrecommit.from_data(
        _only_object(store, "execution-precommit")
    )
    successor = ExposureLedger.from_dict(_only_object(store, "exposure-successor"))
    authorization = ObjectBongardReleaseAuthorization.from_data(
        _only_object(store, "release-authorization")
    )
    runtime_raw = _verify_seal(
        _only_object(store, "anchor-benchmark-runtime"),
        schema=RUNTIME_SCHEMA,
        digest_field="runtime_digest",
        label="runtime",
    )
    runtime = _restore_runtime(runtime_raw)
    from bongard.object_scene_anchor_source_manifest import (
        ObjectSceneAnchorSourceManifest,
        cold_verify_object_scene_anchor_source_manifest,
    )

    source_manifest = ObjectSceneAnchorSourceManifest.from_data(
        _only_object(store, "anchor-source-manifest")
    )
    cold_verify_object_scene_anchor_source_manifest(
        source_manifest,
        repository_root=_REPOSITORY_ROOT,
        expected_manifest_digest=source_manifest.manifest_digest,
    )
    bootstrap = _verify_seal(
        _only_object(store, "anchor-benchmark-bootstrap"),
        schema=BOOTSTRAP_SCHEMA,
        digest_field="bootstrap_digest",
        label="bootstrap",
    )
    descriptor = OfficialReleaseDescriptor.load(descriptor_path)
    archive = OfficialPanelArchive.load(
        descriptor, archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )
    descriptor.verify_split(split_path)
    split = SplitIndex.load(split_path)
    predecessor_path = Path(predecessor_path)
    if _file_sha256(predecessor_path) != PREDECESSOR_FILE_SHA256:
        raise ObjectSceneAnchorBenchmarkError("replay predecessor file differs")
    predecessor = ExposureLedger.load(predecessor_path)
    historical = load_historical_exposure(
        historical_exposure_path, verify_evidence=False
    )
    current_bindings = _source_bindings()
    current_bindings["runtime_record"] = runtime_raw["runtime_digest"]
    current_bindings["source_manifest"] = _object_address(
        source_manifest.manifest_digest
    )
    persisted_bindings = dict(precommit.runtime_source_bindings)
    persisted_configuration = dict(precommit.configuration)
    current_image_configuration = _image_runtime_configuration(
        _image_runtime_environment()
    )
    source_manifest_receipt = store.persist(
        object_kind="anchor-source-manifest",
        object_digest=_object_address(source_manifest.manifest_digest),
        data=source_manifest.to_data(),
    )
    store.verify(source_manifest_receipt, expected_data=source_manifest.to_data())
    current_bindings["source_manifest_receipt"] = (
        source_manifest_receipt.record_digest
    )
    if (
        predecessor.digest != PREDECESSOR_LEDGER_DIGEST
        or historical.seed_digest != HISTORICAL_EXPOSURE_SEED_DIGEST
        or plan.release_descriptor_digest != descriptor.digest
        or plan.split_source_digest != split.source_digest
        or precommit.batch_plan_digest != plan.record_digest
        or any(persisted_bindings.get(key) != value for key, value in current_bindings.items())
        or any(
            persisted_configuration.get(key) != value
            for key, value in current_image_configuration.items()
        )
        or set(persisted_bindings) - set(current_bindings)
        != {"batch_source", "release_gate_source"}
        or authorization.batch_plan_digest != plan.record_digest
        or authorization.execution_precommit_digest != precommit.record_digest
        or authorization.exposure_successor_digest != successor.digest
        or bootstrap.get("batch_plan_digest") != plan.record_digest
        or bootstrap.get("execution_precommit_digest") != precommit.record_digest
        or bootstrap.get("release_authorization_digest") != authorization.record_digest
        or bootstrap.get("runtime_digest") != runtime_raw["runtime_digest"]
        or bootstrap.get("source_manifest_digest")
        != source_manifest.manifest_digest
        or bootstrap.get("source_manifest_receipt")
        != source_manifest_receipt.to_data()
    ):
        raise ObjectSceneAnchorBenchmarkError(
            "disk replay source, runtime, plan, or release lineage differs"
        )

    def receipt(kind: str, digest: str, data: Mapping[str, Any]) -> ObjectBongardWriteOnceReceipt:
        result = store.persist(object_kind=kind, object_digest=digest, data=data)
        store.verify(result, expected_data=data)
        return result

    plan_receipt = receipt("batch-plan", plan.record_digest, plan.to_data())
    precommit_receipt = receipt(
        "execution-precommit", precommit.record_digest, precommit.to_data()
    )
    exposure_receipt = receipt(
        "exposure-successor", successor.digest, successor.to_dict()
    )
    authorization_receipt = receipt(
        "release-authorization", authorization.record_digest, authorization.to_data()
    )
    runtime_receipt = receipt(
        "anchor-benchmark-runtime", runtime_raw["runtime_digest"], runtime_raw
    )
    bootstrap_receipt = receipt(
        "anchor-benchmark-bootstrap", bootstrap["bootstrap_digest"], bootstrap
    )
    release = PreparedObjectBongardRelease(
        store, plan, precommit, predecessor, successor, authorization,
        plan_receipt, precommit_receipt, exposure_receipt, authorization_receipt,
    )
    verify_prepared_object_bongard_release(release)
    return PreparedObjectSceneAnchorBenchmark(
        root, plan, descriptor, archive, split, predecessor, historical, runtime,
        runtime_raw, runtime_receipt, source_manifest, source_manifest_receipt,
        precommit, release, bootstrap,
        bootstrap_receipt,
    )


def _load_typed(
    store: ObjectBongardReleaseStore,
    receipt: ObjectBongardWriteOnceReceipt,
    *,
    expected_kind: str,
    decoder: Callable[[Mapping[str, Any]], object],
) -> object:
    if receipt.object_kind != expected_kind:
        raise ObjectSceneAnchorBenchmarkError(f"{expected_kind} artifact kind differs")
    raw = _read_bounded_json(store.root / receipt.relative_path, expected_kind)
    data = store.verify(receipt, expected_data=raw)
    restored = decoder(data)
    to_data = getattr(restored, "to_data", None)
    if not callable(to_data) or _canonical_mapping(to_data(), expected_kind) != raw:
        raise ObjectSceneAnchorBenchmarkError(f"{expected_kind} reconstruction differs")
    return restored


@dataclass(slots=True)
class _TaskState:
    task: ObjectBongardTaskPlan
    artifact_receipts: dict[str, dict[str, object]] = field(default_factory=dict)
    call_claim_digests: dict[str, str] = field(default_factory=dict)
    call_terminal_digests: dict[str, str] = field(default_factory=dict)
    physical_calls: dict[str, int] = field(default_factory=dict)
    support_release_count: int = 0
    query_release_count: int = 0
    formula_custody_verified: bool = False

    def remember(self, name: str, receipt: ObjectBongardWriteOnceReceipt) -> None:
        if name in self.artifact_receipts:
            raise ObjectSceneAnchorBenchmarkError(f"duplicate task artifact name {name}")
        self.artifact_receipts[name] = receipt.to_data()

    def called(
        self,
        claim: ObjectSceneAnchorBenchmarkCallClaim,
        terminal: ObjectSceneAnchorBenchmarkCallTerminal,
    ) -> None:
        if claim.stage in self.call_claim_digests:
            raise ObjectSceneAnchorBenchmarkError(f"duplicate call stage {claim.stage}")
        self.call_claim_digests[claim.stage] = claim.record_digest
        self.call_terminal_digests[claim.stage] = terminal.record_digest
        self.physical_calls[claim.stage] = terminal.physical_call_slots_consumed


def _validate_task_result_record(value: object) -> dict[str, Any]:
    raw = _verify_seal(
        value,
        schema=TASK_RESULT_SCHEMA,
        digest_field="task_result_digest",
        label="task result",
    )
    required = {
        "schema", "command_id", "task_id", "task_plan_digest",
        "batch_plan_digest", "execution_precommit_digest",
        "release_authorization_digest", "runtime_digest", "status",
        "terminal_stage", "diagnostic", "artifact_receipts",
        "call_claim_digests", "call_terminal_digests",
        "physical_call_slots_at_risk_by_stage", "physical_call_slots_at_risk",
        "support_release_count", "query_release_count", "query_denominator",
        "correct_count", "determinate_count", "abstain_count", "error_count",
        "coverage_ppm", "accuracy_ppm",
        "formula_frozen_and_committed_before_query_release",
        "all_terminal_outcomes_remain_in_denominator", *_authority_data(),
        "task_result_digest",
    }
    mappings = (
        raw.get("artifact_receipts"), raw.get("call_claim_digests"),
        raw.get("call_terminal_digests"),
        raw.get("physical_call_slots_at_risk_by_stage"),
    )
    counts = tuple(
        raw.get(key)
        for key in (
            "physical_call_slots_at_risk", "support_release_count",
            "query_release_count", "query_denominator", "correct_count",
            "determinate_count", "abstain_count", "error_count",
            "coverage_ppm", "accuracy_ppm",
        )
    )
    allowed_statuses = {
        "success", "query_error", "proposer_gap", "language_gap",
        "witness_gap", "capacity_gap", "pipeline_error",
        "infrastructure_error",
    }
    fixed_terminal = {
        "success": "score", "query_error": "score",
        "proposer_gap": "proposer", "language_gap": "version_space",
        "witness_gap": "version_space", "capacity_gap": "rank_input",
    }
    if (
        set(raw) != required
        or raw["command_id"] != COMMAND_ID
        or any(raw[key] != item for key, item in _authority_data().items())
        or any(not isinstance(item, Mapping) for item in mappings)
        or any(type(item) is not int or item < 0 for item in counts)
        or raw["query_denominator"] != 2
        or raw["determinate_count"] + raw["abstain_count"] + raw["error_count"] != 2
        or raw["correct_count"] > raw["determinate_count"]
        or raw["coverage_ppm"] != raw["determinate_count"] * 500_000
        or raw["accuracy_ppm"] != raw["correct_count"] * 500_000
        or raw["physical_call_slots_at_risk"]
        != sum(raw["physical_call_slots_at_risk_by_stage"].values())
        or raw["all_terminal_outcomes_remain_in_denominator"] is not True
        or type(raw["formula_frozen_and_committed_before_query_release"]) is not bool
        or raw["status"] not in allowed_statuses
        or (
            raw["status"] in fixed_terminal
            and raw["terminal_stage"] != fixed_terminal[raw["status"]]
        )
        or (
            raw["status"] not in ("pipeline_error", "infrastructure_error")
            and raw["query_release_count"] not in (0, 2)
        )
        or (
            raw["query_release_count"] > 0
            and raw["formula_frozen_and_committed_before_query_release"] is not True
        )
    ):
        raise ObjectSceneAnchorBenchmarkError("task result fields or accounting differ")
    exact_kinds = {
        "task_support_adapter": "anchor-task-support-adapter",
        "support_corpus": "anchor-support-corpus",
        "proposer_input": "anchor-proposer-input",
        "proposer_artifact": "anchor-proposer-artifact",
        "predicate_language": "anchor-predicate-language",
        "support_observation_plan": "anchor-support-observation-plan",
        "support_observer_artifact": "anchor-support-observer-artifact",
        "support_observation_result": "anchor-support-observation-result",
        "version_space_0": "anchor-support-version-space",
        "version_space_1": "anchor-support-version-space",
        "rank_input": "anchor-rank-input",
        "rank_response": "anchor-rank-response",
        "python_bridge": "anchor-python-bridge",
        "python_predicate": "anchor-python-predicate",
        "task_decision_freeze": "task-freeze",
        "task_decision_commit": "task-decision-commit",
        "query_score": "anchor-query-score",
    }
    for name, receipt_data in raw["artifact_receipts"].items():
        receipt = ObjectBongardWriteOnceReceipt.from_data(
            _canonical_mapping(receipt_data, "task artifact receipt")
        )
        expected_kind = exact_kinds.get(name)
        if name.startswith("support_release_"):
            expected_kind = "released-support-panel"
        elif name.startswith("query_release_"):
            expected_kind = "released-query-panel"
        elif name.startswith("query_visual_plan_"):
            expected_kind = "anchor-query-visual-plan"
        elif name.startswith("query_observer_artifact_"):
            expected_kind = "anchor-query-observer-artifact"
        elif name.startswith("query_visual_result_"):
            expected_kind = "anchor-query-visual-result"
        elif name.startswith("query_prediction_"):
            expected_kind = "anchor-query-prediction"
        elif name.startswith("query_stage_"):
            expected_kind = "anchor-query-stage"
        if expected_kind is not None and receipt.object_kind != expected_kind:
            raise ObjectSceneAnchorBenchmarkError(
                f"task artifact {name} kind differs"
            )
    return raw


def _find_task_result(
    store: ObjectBongardReleaseStore, task: ObjectBongardTaskPlan
) -> tuple[dict[str, Any], ObjectBongardWriteOnceReceipt] | None:
    directory = store.root / "objects" / "anchor-task-result"
    found: list[tuple[dict[str, Any], ObjectBongardWriteOnceReceipt]] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            raw = _validate_task_result_record(
                _read_bounded_json(path, "task result")
            )
            if raw.get("task_plan_digest") != task.record_digest:
                continue
            receipt = store.persist(
                object_kind="anchor-task-result",
                object_digest=raw["task_result_digest"],
                data=raw,
            )
            store.verify(receipt, expected_data=raw)
            found.append((raw, receipt))
    if len(found) > 1:
        raise ObjectSceneAnchorBenchmarkError("task has multiple terminal results")
    return None if not found else found[0]


def _finish_task(
    prepared: PreparedObjectSceneAnchorBenchmark,
    state: _TaskState,
    *,
    status: str,
    terminal_stage: str,
    correct_count: int = 0,
    determinate_count: int = 0,
    abstain_count: int = 0,
    error_count: int = 0,
    diagnostic: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ObjectBongardWriteOnceReceipt]:
    if (
        status == "success"
        and (state.query_release_count != 2 or determinate_count + abstain_count + error_count != 2)
    ):
        raise ObjectSceneAnchorBenchmarkError("successful task accounting differs")
    if status != "success" and determinate_count + abstain_count + error_count != 2:
        raise ObjectSceneAnchorBenchmarkError("terminal task must contribute two queries")
    if not 0 <= correct_count <= determinate_count <= 2:
        raise ObjectSceneAnchorBenchmarkError("task score counts differ")
    body = {
        "schema": TASK_RESULT_SCHEMA,
        "command_id": COMMAND_ID,
        "task_id": state.task.task_id,
        "task_plan_digest": state.task.record_digest,
        "batch_plan_digest": prepared.plan.record_digest,
        "execution_precommit_digest": prepared.precommit.record_digest,
        "release_authorization_digest": prepared.release.authorization.record_digest,
        "runtime_digest": prepared.runtime_record["runtime_digest"],
        "status": status,
        "terminal_stage": terminal_stage,
        "diagnostic": None if diagnostic is None else _canonical_mapping(diagnostic, "diagnostic"),
        "artifact_receipts": dict(sorted(state.artifact_receipts.items())),
        "call_claim_digests": dict(sorted(state.call_claim_digests.items())),
        "call_terminal_digests": dict(sorted(state.call_terminal_digests.items())),
        "physical_call_slots_at_risk_by_stage": dict(sorted(state.physical_calls.items())),
        "physical_call_slots_at_risk": sum(state.physical_calls.values()),
        "support_release_count": state.support_release_count,
        "query_release_count": state.query_release_count,
        "query_denominator": 2,
        "correct_count": correct_count,
        "determinate_count": determinate_count,
        "abstain_count": abstain_count,
        "error_count": error_count,
        "coverage_ppm": determinate_count * 500_000,
        "accuracy_ppm": correct_count * 500_000,
        "formula_frozen_and_committed_before_query_release": (
            state.query_release_count == 0
            or state.formula_custody_verified
        ),
        "all_terminal_outcomes_remain_in_denominator": True,
        **_authority_data(),
    }
    result = _seal(body, "task_result_digest")
    return _persist_record(
        prepared.release.store,
        object_kind="anchor-task-result",
        record=result,
        digest_field="task_result_digest",
        schema=TASK_RESULT_SCHEMA,
    )


@dataclass(frozen=True, slots=True)
class ObjectSceneAnchorBenchmarkTransports:
    proposer: Callable[..., object] | None = field(default=None, compare=False, repr=False)
    support_observer: Callable[..., object] | None = field(default=None, compare=False, repr=False)
    ranker: Callable[..., object] | None = field(default=None, compare=False, repr=False)
    query_observer: Callable[..., object] | None = field(default=None, compare=False, repr=False)


def _runtime_kwargs(runtime: ObjectBongardTurnRuntime) -> dict[str, object]:
    if runtime.cloud_policy_cache_snapshot is None:
        raise ObjectSceneAnchorBenchmarkError("frozen policy-cache snapshot is absent")
    return {
        "model": runtime.model,
        "reasoning_effort": runtime.reasoning_effort,
        "minutes": runtime.minutes,
        "verbose": runtime.verbose,
        "executable": runtime.executable,
        "cloud_policy_cache_snapshot": runtime.cloud_policy_cache_snapshot,
        "model_catalog_snapshot": runtime.model_catalog_snapshot,
        "no_tools_attestation": runtime.no_tools_attestation,
        "expected_launcher_digest": runtime.expected_launcher_digest,
    }


def _persist_lower(
    prepared: PreparedObjectSceneAnchorBenchmark,
    state: _TaskState,
    *,
    name: str,
    kind: str,
    value: object,
    digest: str,
    decoder: Callable[[Mapping[str, Any]], object],
) -> tuple[object, ObjectBongardWriteOnceReceipt]:
    restored, receipt = _persist_typed(
        prepared.release.store,
        object_kind=kind,
        value=value,
        digest=digest,
        decoder=decoder,
    )
    state.remember(name, receipt)
    return restored, receipt


def _run_object_scene_anchor_task_core(
    prepared: PreparedObjectSceneAnchorBenchmark,
    task: ObjectBongardTaskPlan,
    *,
    journal: ObjectSceneAnchorBenchmarkCallJournal,
    transports: ObjectSceneAnchorBenchmarkTransports,
) -> tuple[dict[str, Any], ObjectBongardWriteOnceReceipt]:
    from bongard.object_bongard_release_gate import (
        persist_object_bongard_task_commit,
        persist_object_bongard_task_freeze,
        release_object_bongard_query_panel,
        release_object_bongard_support_panel,
    )
    from bongard.object_scene_anchor_batch_observer import (
        ObjectSceneAnchorBatchObserverArtifact,
        observe_object_scene_anchor_batches_twice,
        verify_object_scene_anchor_batch_observer_artifact,
    )
    from bongard.object_scene_anchor_candidate_ranker import (
        ObjectSceneAnchorCandidateRanker,
        ObjectSceneAnchorRankCapacityGap,
        ObjectSceneAnchorRankInput,
        ObjectSceneAnchorRankResponse,
        freeze_object_scene_anchor_rank_input,
    )
    from bongard.object_scene_anchor_card_proposer import (
        ObjectSceneAnchorCardProposerArtifact,
        ObjectSceneAnchorCardProposerInput,
        ObjectSceneAnchorCardProposerPanelInput,
        freeze_object_scene_anchor_card_proposer_input,
        propose_object_scene_anchor_cards,
        verify_object_scene_anchor_card_proposer_artifact,
    )
    from bongard.object_scene_anchor_python_bridge import (
        ObjectSceneAnchorPythonBridgeArtifact,
        freeze_object_scene_anchor_python_bridge,
    )
    from bongard.object_scene_anchor_python_predicate import (
        ObjectSceneAnchorPythonPredicate,
    )
    from bongard.object_scene_anchor_support_observation_join import (
        ObjectSceneAnchorSupportObservationPlan,
        ObjectSceneAnchorSupportObservationResult,
        build_object_scene_anchor_support_observation_plan,
        finalize_object_scene_anchor_support_observations,
    )
    from bongard.object_scene_anchor_support_preparation import (
        ObjectSceneAnchorSupportCorpusFreeze,
    )
    from bongard.object_scene_anchor_task_decision_custody import (
        ObjectSceneAnchorTaskDecisionCommit,
        ObjectSceneAnchorTaskDecisionFreeze,
        cold_verify_object_scene_anchor_task_decision_commit,
        cold_verify_object_scene_anchor_task_decision_freeze,
        commit_object_scene_anchor_task_decision,
        freeze_object_scene_anchor_task_decision,
    )
    from bongard.object_scene_anchor_task_support_adapter import (
        ObjectSceneAnchorTaskSupportAdapter,
        build_object_scene_anchor_task_support_corpus,
        verify_object_scene_anchor_task_support_corpus,
    )
    from bongard.object_scene_anchor_version_space import (
        ObjectSceneAnchorPredicateLanguage,
        ObjectSceneAnchorSupportVersionSpace,
        project_object_scene_anchor_card_proposal,
    )

    state = _TaskState(task)
    support_ids = (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
    released = []
    for index, panel_id in enumerate(support_ids):
        panel, receipt = release_object_bongard_support_panel(
            prepared=prepared.release, archive=prepared.archive, panel_id=panel_id
        )
        released.append(panel)
        state.remember(f"support_release_{index:03d}", receipt)
    state.support_release_count = len(released)
    support_runtime = build_object_scene_anchor_task_support_corpus(
        task=task, prepared=prepared.release, released_panels=tuple(released)
    )
    verify_object_scene_anchor_task_support_corpus(
        support_runtime,
        task=task,
        prepared=prepared.release,
        expected_adapter_digest=support_runtime.adapter.adapter_digest,
    )
    adapter, _ = _persist_lower(
        prepared, state, name="task_support_adapter", kind="anchor-task-support-adapter",
        value=support_runtime.adapter, digest=support_runtime.adapter.adapter_digest,
        decoder=ObjectSceneAnchorTaskSupportAdapter.from_data,
    )
    _persist_lower(
        prepared, state, name="support_corpus", kind="anchor-support-corpus",
        value=support_runtime.support_corpus.freeze,
        digest=support_runtime.support_corpus.freeze.freeze_digest,
        decoder=ObjectSceneAnchorSupportCorpusFreeze.from_data,
    )

    proposer_rows = tuple(
        ObjectSceneAnchorCardProposerPanelInput(
            panel.freeze.support_sheet,
            panel.exact_support_sheet_png_bytes,
            panel.freeze.panel_manifest,
        )
        for panel in support_runtime.support_corpus.panels
    )
    side0, side1 = proposer_rows[:6], proposer_rows[6:]
    proposer_input = freeze_object_scene_anchor_card_proposer_input(side0, side1)
    proposer_input, _ = _persist_lower(
        prepared, state, name="proposer_input", kind="anchor-proposer-input",
        value=proposer_input, digest=proposer_input.input_digest,
        decoder=ObjectSceneAnchorCardProposerInput.from_data,
    )
    runtime_args = _runtime_kwargs(prepared.runtime)

    def invoke_proposer() -> tuple[object, ObjectBongardWriteOnceReceipt]:
        kwargs = dict(runtime_args)
        if transports.proposer is not None:
            kwargs["transport"] = transports.proposer
        artifact = propose_object_scene_anchor_cards(
            side0, side1,
            proposer_input=proposer_input,
            expected_input_digest=proposer_input.input_digest,
            **kwargs,
        )
        return _persist_lower(
            prepared, state, name="proposer_artifact", kind="anchor-proposer-artifact",
            value=artifact, digest=artifact.artifact_digest,
            decoder=ObjectSceneAnchorCardProposerArtifact.from_data,
        )

    def load_proposer(receipt: ObjectBongardWriteOnceReceipt) -> object:
        return _load_typed(
            prepared.release.store, receipt, expected_kind="anchor-proposer-artifact",
            decoder=ObjectSceneAnchorCardProposerArtifact.from_data,
        )

    proposer_artifact, claim, terminal, _reused = journal.run(
        task_plan_digest=task.record_digest,
        stage="proposer",
        context_digest=_object_address(proposer_input.input_digest),
        expected_physical_call_count=1,
        object_kind="anchor-proposer-artifact",
        invoke_and_persist=invoke_proposer,
        load_artifact=load_proposer,
    )
    state.called(claim, terminal)
    if "proposer_artifact" not in state.artifact_receipts:
        state.remember("proposer_artifact", ObjectBongardWriteOnceReceipt.from_data(terminal.artifact_receipt))
    assert isinstance(proposer_artifact, ObjectSceneAnchorCardProposerArtifact)
    verify_object_scene_anchor_card_proposer_artifact(
        proposer_artifact, side0, side1,
        expected_artifact_digest=proposer_artifact.artifact_digest,
        expected_input_digest=proposer_input.input_digest,
        **{key: runtime_args[key] for key in (
            "model", "reasoning_effort", "expected_launcher_digest",
            "cloud_policy_cache_snapshot", "model_catalog_snapshot", "no_tools_attestation",
        )},
    )
    if proposer_artifact.status != "success" or proposer_artifact.proposal is None:
        return _finish_task(
            prepared, state, status="proposer_gap", terminal_stage="proposer",
            abstain_count=2,
            diagnostic={
                "proposer_status": proposer_artifact.status,
                "failure_code": proposer_artifact.failure_code,
                "failure_type": proposer_artifact.failure_type,
            },
        )

    try:
        language = project_object_scene_anchor_card_proposal(proposer_artifact.proposal)
        language, _ = _persist_lower(
            prepared, state, name="predicate_language", kind="anchor-predicate-language",
            value=language, digest=language.language_digest,
            decoder=ObjectSceneAnchorPredicateLanguage.from_data,
        )
        support_observation_runtime = build_object_scene_anchor_support_observation_plan(
            support_runtime.support_corpus, language
        )
    except Exception as exc:
        return _finish_task(
            prepared, state, status="pipeline_error", terminal_stage="language",
            error_count=2, diagnostic={"failure_type": _failure_type(exc)},
        )
    support_plan, _ = _persist_lower(
        prepared, state, name="support_observation_plan", kind="anchor-support-observation-plan",
        value=support_observation_runtime.plan,
        digest=support_observation_runtime.plan.plan_digest,
        decoder=ObjectSceneAnchorSupportObservationPlan.from_data,
    )

    def invoke_support_observer() -> tuple[object, ObjectBongardWriteOnceReceipt]:
        kwargs = dict(runtime_args)
        if transports.support_observer is not None:
            kwargs["transport"] = transports.support_observer
        artifact = observe_object_scene_anchor_batches_twice(
            support_observation_runtime.batch_inputs,
            plan=support_plan.batch_plan,
            expected_plan_digest=support_plan.batch_plan_digest,
            observation_plan_digest=support_plan.observation_context_digest,
            **kwargs,
        )
        return _persist_lower(
            prepared, state, name="support_observer_artifact",
            kind="anchor-support-observer-artifact", value=artifact,
            digest=artifact.artifact_digest,
            decoder=ObjectSceneAnchorBatchObserverArtifact.from_data,
        )

    def load_support_observer(receipt: ObjectBongardWriteOnceReceipt) -> object:
        return _load_typed(
            prepared.release.store, receipt,
            expected_kind="anchor-support-observer-artifact",
            decoder=ObjectSceneAnchorBatchObserverArtifact.from_data,
        )

    batch_artifact, claim, terminal, _reused = journal.run(
        task_plan_digest=task.record_digest,
        stage="support_observer",
        context_digest=_object_address(support_plan.observation_context_digest),
        expected_physical_call_count=support_plan.batch_plan.physical_call_count,
        object_kind="anchor-support-observer-artifact",
        invoke_and_persist=invoke_support_observer,
        load_artifact=load_support_observer,
    )
    state.called(claim, terminal)
    if "support_observer_artifact" not in state.artifact_receipts:
        state.remember("support_observer_artifact", ObjectBongardWriteOnceReceipt.from_data(terminal.artifact_receipt))
    assert isinstance(batch_artifact, ObjectSceneAnchorBatchObserverArtifact)
    verify_object_scene_anchor_batch_observer_artifact(
        batch_artifact,
        support_observation_runtime.batch_inputs,
        expected_artifact_digest=batch_artifact.artifact_digest,
        expected_plan_digest=support_plan.batch_plan_digest,
        expected_observation_plan_digest=support_plan.observation_context_digest,
    )
    support_result = finalize_object_scene_anchor_support_observations(
        support_plan, batch_artifact
    )
    support_result, _ = _persist_lower(
        prepared, state, name="support_observation_result",
        kind="anchor-support-observation-result", value=support_result,
        digest=support_result.result_digest,
        decoder=ObjectSceneAnchorSupportObservationResult.from_data,
    )
    spaces = (
        support_result.bucket0_positive_version_space,
        support_result.bucket1_positive_version_space,
    )
    for index, space in enumerate(spaces):
        _persist_lower(
            prepared, state, name=f"version_space_{index}",
            kind="anchor-support-version-space", value=space,
            digest=space.version_space_digest,
            decoder=ObjectSceneAnchorSupportVersionSpace.from_data,
        )
    nonempty = tuple(
        sorted(
            (space for space in spaces if space.survivor_candidate_digests),
            key=lambda space: space.version_space_digest,
        )
    )
    if not nonempty:
        from bongard.object_scene_anchor_version_space import (
            ObjectSceneAnchorSupportGapKind,
        )

        exact_gaps = tuple(space.gap for space in spaces)
        if any(gap is None for gap in exact_gaps):
            raise ObjectSceneAnchorBenchmarkError(
                "empty orientation lacks its exact typed gap"
            )
        status = (
            "language_gap"
            if all(
                gap.kind is ObjectSceneAnchorSupportGapKind.LANGUAGE_GAP
                for gap in exact_gaps
            )
            else "witness_gap"
        )
        return _finish_task(
            prepared, state, status=status, terminal_stage="version_space",
            abstain_count=2,
            diagnostic={
                "orientation_gaps": [
                    None if space.gap is None else space.gap.to_data() for space in spaces
                ]
            },
        )
    try:
        rank_input = freeze_object_scene_anchor_rank_input(
            nonempty[0], None if len(nonempty) == 1 else nonempty[1]
        )
    except ObjectSceneAnchorRankCapacityGap as exc:
        return _finish_task(
            prepared, state, status="capacity_gap", terminal_stage="rank_input",
            abstain_count=2,
            diagnostic={
                "failure_type": _failure_type(exc),
                "survivor_count": exc.survivor_count,
                "maximum_survivor_count": exc.maximum_survivor_count,
                "child_version_space_digests": list(
                    exc.child_version_space_digests
                ),
                "orientation_gaps": [
                    None if space.gap is None else space.gap.to_data() for space in spaces
                ],
            },
        )
    rank_input, _ = _persist_lower(
        prepared, state, name="rank_input", kind="anchor-rank-input",
        value=rank_input, digest=rank_input.rank_input_digest,
        decoder=ObjectSceneAnchorRankInput.from_data,
    )
    ranker_kwargs = dict(
        model=prepared.runtime.model,
        expected_launcher_digest=prepared.runtime.expected_launcher_digest,
        cloud_policy_cache_snapshot=prepared.runtime.cloud_policy_cache_snapshot,
        model_catalog_snapshot=prepared.runtime.model_catalog_snapshot,
        no_tools_attestation=prepared.runtime.no_tools_attestation,
        reasoning_effort=prepared.runtime.reasoning_effort,
        minutes=prepared.runtime.minutes,
        verbose=prepared.runtime.verbose,
        executable=prepared.runtime.executable,
    )
    if transports.ranker is not None:
        ranker_kwargs["transport"] = transports.ranker
    ranker = ObjectSceneAnchorCandidateRanker(**ranker_kwargs)

    def invoke_ranker() -> tuple[object, ObjectBongardWriteOnceReceipt]:
        response = ranker(
            nonempty[0], None if len(nonempty) == 1 else nonempty[1],
            expected_rank_input_digest=rank_input.rank_input_digest,
        )
        return _persist_lower(
            prepared, state, name="rank_response", kind="anchor-rank-response",
            value=response, digest=response.response_digest,
            decoder=ObjectSceneAnchorRankResponse.from_data,
        )

    def load_ranker(receipt: ObjectBongardWriteOnceReceipt) -> object:
        return _load_typed(
            prepared.release.store, receipt, expected_kind="anchor-rank-response",
            decoder=ObjectSceneAnchorRankResponse.from_data,
        )

    rank_response, claim, terminal, _reused = journal.run(
        task_plan_digest=task.record_digest,
        stage="ranker",
        context_digest=_object_address(rank_input.rank_input_digest),
        expected_physical_call_count=1,
        object_kind="anchor-rank-response",
        invoke_and_persist=invoke_ranker,
        load_artifact=load_ranker,
    )
    state.called(claim, terminal)
    if "rank_response" not in state.artifact_receipts:
        state.remember("rank_response", ObjectBongardWriteOnceReceipt.from_data(terminal.artifact_receipt))
    assert isinstance(rank_response, ObjectSceneAnchorRankResponse)
    ranker.verify_response(
        rank_response,
        version_space=nonempty[0],
        additional_version_space=None if len(nonempty) == 1 else nonempty[1],
        expected_response_digest=rank_response.response_digest,
        expected_rank_input_digest=rank_input.rank_input_digest,
    )
    bridge = freeze_object_scene_anchor_python_bridge(
        rank_response, spaces[0], spaces[1],
        expected_response_digest=rank_response.response_digest,
        expected_rank_input_digest=rank_input.rank_input_digest,
    )
    bridge, _ = _persist_lower(
        prepared, state, name="python_bridge", kind="anchor-python-bridge",
        value=bridge, digest=bridge.bridge_digest,
        decoder=ObjectSceneAnchorPythonBridgeArtifact.from_data,
    )
    predicate, _ = _persist_lower(
        prepared, state, name="python_predicate", kind="anchor-python-predicate",
        value=bridge.predicate, digest=bridge.predicate.predicate_digest,
        decoder=ObjectSceneAnchorPythonPredicate.from_data,
    )

    # This is the only custody call site.  It intentionally names every exact
    # model/support parent so an API hardening adds no hidden runner path.
    freeze = freeze_object_scene_anchor_task_decision(
        task=task,
        execution_precommit=prepared.precommit,
        task_support_adapter=adapter,
        card_proposer_artifact=proposer_artifact,
        support_observation_plan=support_plan,
        batch_observer_artifact=batch_artifact,
        support_observation_result=support_result,
        rank_input=rank_input,
        rank_response=rank_response,
        bridge=bridge,
        predicate=predicate,
    )
    freeze_receipt = persist_object_bongard_task_freeze(
        store=prepared.release.store, freeze=freeze
    )
    freeze_raw = prepared.release.store.verify(
        freeze_receipt, expected_data=freeze.to_data()
    )
    freeze = ObjectSceneAnchorTaskDecisionFreeze.from_data(freeze_raw)
    cold_verify_object_scene_anchor_task_decision_freeze(
        freeze,
        task=task,
        execution_precommit=prepared.precommit,
        task_support_adapter=adapter,
        card_proposer_artifact=proposer_artifact,
        support_observation_plan=support_plan,
        batch_observer_artifact=batch_artifact,
        support_observation_result=support_result,
        rank_input=rank_input,
        rank_response=rank_response,
        bridge=bridge,
        predicate=predicate,
        expected_freeze_digest=freeze.record_digest,
    )
    state.remember("task_decision_freeze", freeze_receipt)
    exact_freeze_payload = canonical_json(freeze.to_data()) + b"\n"
    commit = commit_object_scene_anchor_task_decision(
        freeze=freeze,
        exact_freeze_payload=exact_freeze_payload,
        task_freeze_store_receipt=freeze_receipt,
    )
    commit_receipt = persist_object_bongard_task_commit(
        store=prepared.release.store, commit=commit
    )
    commit_raw = prepared.release.store.verify(
        commit_receipt, expected_data=commit.to_data()
    )
    commit = ObjectSceneAnchorTaskDecisionCommit.from_data(commit_raw)
    cold_verify_object_scene_anchor_task_decision_commit(
        commit,
        freeze=freeze,
        exact_freeze_payload=exact_freeze_payload,
        expected_commit_digest=commit.record_digest,
    )
    state.remember("task_decision_commit", commit_receipt)
    state.formula_custody_verified = True

    released_queries = []
    for index, panel_id in enumerate(
        (task.side_0_query_panel_id, task.side_1_query_panel_id)
    ):
        panel, receipt = release_object_bongard_query_panel(
            prepared=prepared.release,
            archive=prepared.archive,
            panel_id=panel_id,
            task_freeze=freeze,
            task_commit=commit,
            task_freeze_receipt=freeze_receipt,
            task_commit_receipt=commit_receipt,
        )
        released_queries.append((panel, receipt))
        state.remember(f"query_release_{index:03d}", receipt)
    state.query_release_count = 2
    return _run_object_scene_anchor_query_phase(
        prepared,
        state,
        predicate=predicate,
        released_queries=tuple(released_queries),
        journal=journal,
        transport=transports.query_observer,
    )


def _run_object_scene_anchor_query_phase(
    prepared: PreparedObjectSceneAnchorBenchmark,
    state: _TaskState,
    *,
    predicate: object,
    released_queries: tuple[tuple[object, ObjectBongardWriteOnceReceipt], ...],
    journal: ObjectSceneAnchorBenchmarkCallJournal,
    transport: Callable[..., object] | None,
) -> tuple[dict[str, Any], ObjectBongardWriteOnceReceipt]:
    """Freeze both visual plans, then observe, predict, and finally score."""

    from bongard.object_scene_anchor_batch_observer import (
        ObjectSceneAnchorBatchObserverArtifact,
        observe_object_scene_anchor_batches_twice,
        verify_object_scene_anchor_batch_observer_artifact,
    )
    from bongard.object_scene_anchor_python_bridge import (
        ObjectSceneAnchorPredictionBucket,
        ObjectSceneAnchorPythonPrediction,
    )
    from bongard.object_scene_anchor_python_predicate import (
        ObjectSceneAnchorPythonPredicate,
    )
    from bongard.object_scene_anchor_python_query_visual_execution import (
        ObjectSceneAnchorPythonQueryPanelInput,
        ObjectSceneAnchorPythonQueryVisualPlan,
        ObjectSceneAnchorPythonQueryVisualResult,
        build_object_scene_anchor_python_query_visual_plan,
        cold_verify_object_scene_anchor_python_query_visual_result,
        finalize_object_scene_anchor_python_query_visual_execution,
        verify_object_scene_anchor_python_query_visual_runtime,
    )

    if type(predicate) is not ObjectSceneAnchorPythonPredicate:
        raise TypeError("query phase requires exact Python predicate")
    if len(released_queries) != 2:
        raise ObjectSceneAnchorBenchmarkError("query phase requires exactly two releases")
    runtimes = []
    for index, (released, release_receipt) in enumerate(released_queries):
        source_binding = _address(
            {
                "schema": "gkm.object-scene-anchor-query-source-binding.v1",
                "task_plan_digest": state.task.record_digest,
                "query_alias": f"panel_{index:03d}",
                "released_panel_record_digest": getattr(released, "record_digest", None),
                "release_store_receipt_digest": release_receipt.record_digest,
                "predicate_digest": predicate.predicate_digest,
                "task_decision_commit_receipt_digest": state.artifact_receipts[
                    "task_decision_commit"
                ]["record_digest"],
            }
        )
        panel_input = ObjectSceneAnchorPythonQueryPanelInput(
            released, f"panel_{index:03d}", source_binding
        )
        query_runtime = build_object_scene_anchor_python_query_visual_plan(
            panel_input, predicate
        )
        verify_object_scene_anchor_python_query_visual_runtime(
            query_runtime,
            panel_input=panel_input,
            predicate=predicate,
            expected_plan_digest=query_runtime.plan.plan_digest,
        )
        plan, _ = _persist_lower(
            prepared,
            state,
            name=f"query_visual_plan_{index:03d}",
            kind="anchor-query-visual-plan",
            value=query_runtime.plan,
            digest=query_runtime.plan.plan_digest,
            decoder=ObjectSceneAnchorPythonQueryVisualPlan.from_data,
        )
        runtimes.append((query_runtime, plan))
    # Both plans above are durable before either observer group starts.

    results = []
    predictions = []
    runtime_args = _runtime_kwargs(prepared.runtime)
    for index, (query_runtime, plan) in enumerate(runtimes):
        stage = f"query_observer_{index:03d}"

        def invoke_query(
            query_runtime=query_runtime,
            plan=plan,
            index=index,
        ) -> tuple[object, ObjectBongardWriteOnceReceipt]:
            batch_artifact = None
            batch_receipt = None
            if plan.batch_plan is not None:
                kwargs = dict(runtime_args)
                if transport is not None:
                    kwargs["transport"] = transport
                batch_artifact = observe_object_scene_anchor_batches_twice(
                    query_runtime.batch_inputs,
                    plan=plan.batch_plan,
                    expected_plan_digest=plan.batch_plan_digest,
                    observation_plan_digest=plan.observation_context_digest,
                    **kwargs,
                )
                batch_artifact, batch_receipt = _persist_lower(
                    prepared,
                    state,
                    name=f"query_observer_artifact_{index:03d}",
                    kind="anchor-query-observer-artifact",
                    value=batch_artifact,
                    digest=batch_artifact.artifact_digest,
                    decoder=ObjectSceneAnchorBatchObserverArtifact.from_data,
                )
                verify_object_scene_anchor_batch_observer_artifact(
                    batch_artifact,
                    query_runtime.batch_inputs,
                    expected_artifact_digest=batch_artifact.artifact_digest,
                    expected_plan_digest=plan.batch_plan_digest,
                    expected_observation_plan_digest=plan.observation_context_digest,
                )
            result = finalize_object_scene_anchor_python_query_visual_execution(
                plan, batch_artifact
            )
            result, result_receipt = _persist_lower(
                prepared,
                state,
                name=f"query_visual_result_{index:03d}",
                kind="anchor-query-visual-result",
                value=result,
                digest=result.result_digest,
                decoder=ObjectSceneAnchorPythonQueryVisualResult.from_data,
            )
            plan_receipt = ObjectBongardWriteOnceReceipt.from_data(
                state.artifact_receipts[f"query_visual_plan_{index:03d}"]
            )
            stage_record = _seal(
                {
                    "schema": QUERY_STAGE_SCHEMA,
                    "command_id": COMMAND_ID,
                    "task_plan_digest": state.task.record_digest,
                    "release_authorization_digest": (
                        prepared.release.authorization.record_digest
                    ),
                    "stage": f"query_observer_{index:03d}",
                    "query_visual_plan_digest": plan.plan_digest,
                    "query_visual_plan_receipt": plan_receipt.to_data(),
                    "batch_observer_artifact_digest": (
                        None if batch_artifact is None else batch_artifact.artifact_digest
                    ),
                    "batch_observer_artifact_receipt": (
                        None if batch_receipt is None else batch_receipt.to_data()
                    ),
                    "query_visual_result_digest": result.result_digest,
                    "query_visual_result_receipt": result_receipt.to_data(),
                    "physical_call_count": plan.physical_call_count,
                    "complete_stage_parent_graph": True,
                    **_authority_data(),
                },
                "query_stage_digest",
            )
            stage_raw, stage_receipt = _persist_record(
                prepared.release.store,
                object_kind="anchor-query-stage",
                record=stage_record,
                digest_field="query_stage_digest",
                schema=QUERY_STAGE_SCHEMA,
            )
            _verify_query_stage_record(
                prepared.release.store,
                stage_raw,
                expected_authorization_digest=(
                    prepared.release.authorization.record_digest
                ),
            )
            state.remember(f"query_stage_{index:03d}", stage_receipt)
            return result, stage_receipt

        def load_query(receipt: ObjectBongardWriteOnceReceipt) -> object:
            stage_raw, _ = _load_receipted_record(
                prepared.release.store,
                receipt,
                expected_kind="anchor-query-stage",
                schema=QUERY_STAGE_SCHEMA,
                digest_field="query_stage_digest",
            )
            _verify_query_stage_record(
                prepared.release.store,
                stage_raw,
                expected_authorization_digest=(
                    prepared.release.authorization.record_digest
                ),
            )
            if (
                stage_raw.get("stage") != stage
                or stage_raw.get("query_visual_plan_digest") != plan.plan_digest
                or stage_raw.get("physical_call_count") != plan.physical_call_count
            ):
                raise ObjectSceneAnchorBenchmarkError("query stage replay differs")
            result_receipt = ObjectBongardWriteOnceReceipt.from_data(
                stage_raw["query_visual_result_receipt"]
            )
            return _load_typed(
                prepared.release.store,
                result_receipt,
                expected_kind="anchor-query-visual-result",
                decoder=ObjectSceneAnchorPythonQueryVisualResult.from_data,
            )

        result, claim, terminal, _reused = journal.run(
            task_plan_digest=state.task.record_digest,
            stage=stage,
            context_digest=_object_address(plan.observation_context_digest),
            expected_physical_call_count=plan.physical_call_count,
            object_kind="anchor-query-stage",
            invoke_and_persist=invoke_query,
            load_artifact=load_query,
        )
        state.called(claim, terminal)
        if f"query_stage_{index:03d}" not in state.artifact_receipts:
            state.remember(
                f"query_stage_{index:03d}",
                ObjectBongardWriteOnceReceipt.from_data(terminal.artifact_receipt),
            )
        if f"query_visual_result_{index:03d}" not in state.artifact_receipts:
            terminal_receipt = ObjectBongardWriteOnceReceipt.from_data(
                terminal.artifact_receipt
            )
            stage_raw, _ = _load_receipted_record(
                prepared.release.store,
                terminal_receipt,
                expected_kind="anchor-query-stage",
                schema=QUERY_STAGE_SCHEMA,
                digest_field="query_stage_digest",
            )
            state.artifact_receipts[f"query_visual_result_{index:03d}"] = dict(
                stage_raw["query_visual_result_receipt"]
            )
            if stage_raw["batch_observer_artifact_receipt"] is not None:
                state.artifact_receipts[f"query_observer_artifact_{index:03d}"] = dict(
                    stage_raw["batch_observer_artifact_receipt"]
                )
        if type(result) is not ObjectSceneAnchorPythonQueryVisualResult:
            raise ObjectSceneAnchorBenchmarkError("query visual result type differs")
        batch_artifact = None
        if result.batch_artifact_digest is not None:
            batch_receipt_data = state.artifact_receipts.get(
                f"query_observer_artifact_{index:03d}"
            )
            if batch_receipt_data is None:
                raise ObjectSceneAnchorBenchmarkError(
                    "query stage omits observer artifact receipt"
                )
            batch_artifact = _load_typed(
                prepared.release.store,
                ObjectBongardWriteOnceReceipt.from_data(batch_receipt_data),
                expected_kind="anchor-query-observer-artifact",
                decoder=ObjectSceneAnchorBatchObserverArtifact.from_data,
            )
        cold_verify_object_scene_anchor_python_query_visual_result(
            result,
            plan=plan,
            artifact=batch_artifact,
            expected_result_digest=result.result_digest,
        )
        prediction, _ = _persist_lower(
            prepared,
            state,
            name=f"query_prediction_{index:03d}",
            kind="anchor-query-prediction",
            value=result.prediction,
            digest=result.prediction.prediction_digest,
            decoder=ObjectSceneAnchorPythonPrediction.from_data,
        )
        results.append(result)
        predictions.append(prediction)

    expected = (
        ObjectSceneAnchorPredictionBucket.SIDE0_POSITIVE,
        ObjectSceneAnchorPredictionBucket.SIDE1_POSITIVE,
    )
    predicted = tuple(item.predicted_bucket for item in predictions)
    correct = sum(got is want for got, want in zip(predicted, expected, strict=True))
    determinate = sum(
        item
        in (
            ObjectSceneAnchorPredictionBucket.SIDE0_POSITIVE,
            ObjectSceneAnchorPredictionBucket.SIDE1_POSITIVE,
        )
        for item in predicted
    )
    abstain = sum(item is ObjectSceneAnchorPredictionBucket.ABSTAIN for item in predicted)
    errors = sum(item is ObjectSceneAnchorPredictionBucket.ERROR for item in predicted)
    score = _seal(
        {
            "schema": QUERY_SCORE_SCHEMA,
            "command_id": COMMAND_ID,
            "task_plan_digest": state.task.record_digest,
            "predicate_digest": predicate.predicate_digest,
            "query_visual_result_digests": [item.result_digest for item in results],
            "prediction_digests": [item.prediction_digest for item in predictions],
            "predicted_buckets": [item.value for item in predicted],
            "expected_buckets": [item.value for item in expected],
            "correct_count": correct,
            "determinate_count": determinate,
            "abstain_count": abstain,
            "error_count": errors,
            "query_denominator": 2,
            "accuracy_ppm": correct * 500_000,
            "coverage_ppm": determinate * 500_000,
            "prediction_digests_validated_before_expected_bucket_access": True,
            **_authority_data(),
        },
        "score_digest",
    )
    score, score_receipt = _persist_record(
        prepared.release.store,
        object_kind="anchor-query-score",
        record=score,
        digest_field="score_digest",
        schema=QUERY_SCORE_SCHEMA,
    )
    _verify_query_score_record(
        score,
        expected_task_plan_digest=state.task.record_digest,
        expected_predicate_digest=predicate.predicate_digest,
        expected_query_visual_result_digests=[
            item.result_digest for item in results
        ],
        expected_prediction_digests=[
            item.prediction_digest for item in predictions
        ],
        expected_predicted_buckets=[item.value for item in predicted],
    )
    state.remember("query_score", score_receipt)
    return _finish_task(
        prepared,
        state,
        status="query_error" if errors else "success",
        terminal_stage="score",
        correct_count=correct,
        determinate_count=determinate,
        abstain_count=abstain,
        error_count=errors,
        diagnostic={"score_digest": score["score_digest"]},
    )


def _recover_terminal_state(
    prepared: PreparedObjectSceneAnchorBenchmark,
    task: ObjectBongardTaskPlan,
) -> _TaskState:
    """Recover only durable accounting; it never authorizes another call."""

    state = _TaskState(task)
    durable_root = prepared.release.store.root
    claim_directory = durable_root / "objects" / "anchor-call-claim"
    claims: dict[str, ObjectSceneAnchorBenchmarkCallClaim] = {}
    if claim_directory.is_dir():
        for path in sorted(claim_directory.glob("*.json")):
            claim = ObjectSceneAnchorBenchmarkCallClaim.from_data(
                _read_bounded_json(path, "call claim")
            )
            if claim.task_plan_digest == task.record_digest:
                if claim.stage in claims:
                    raise ObjectSceneAnchorBenchmarkError(
                        "task has duplicate durable call stages"
                    )
                claims[claim.stage] = claim
    terminal_directory = durable_root / "objects" / "anchor-call-terminal"
    terminals: dict[str, ObjectSceneAnchorBenchmarkCallTerminal] = {}
    if terminal_directory.is_dir():
        for path in sorted(terminal_directory.glob("*.json")):
            terminal = ObjectSceneAnchorBenchmarkCallTerminal.from_data(
                _read_bounded_json(path, "call terminal")
            )
            if terminal.task_plan_digest == task.record_digest:
                if terminal.stage in terminals:
                    raise ObjectSceneAnchorBenchmarkError(
                        "task has duplicate call terminal stages"
                    )
                terminals[terminal.stage] = terminal
    for stage, claim in sorted(claims.items()):
        state.call_claim_digests[stage] = claim.record_digest
        state.physical_calls[stage] = claim.expected_physical_call_count
        terminal = terminals.get(stage)
        if terminal is not None:
            if (
                terminal.claim_digest != claim.record_digest
                or terminal.release_authorization_digest
                != claim.release_authorization_digest
                or terminal.task_plan_digest != claim.task_plan_digest
                or terminal.stage != claim.stage
                or terminal.context_digest != claim.context_digest
                or terminal.physical_call_slots_consumed
                != claim.expected_physical_call_count
            ):
                raise ObjectSceneAnchorBenchmarkError(
                    "terminal does not bind recovered claim"
                )
            state.call_terminal_digests[stage] = terminal.record_digest
            if terminal.artifact_receipt is not None:
                terminal_receipt = ObjectBongardWriteOnceReceipt.from_data(
                    terminal.artifact_receipt
                )
                state.artifact_receipts[f"terminal_{stage}"] = terminal_receipt.to_data()
                if terminal_receipt.object_kind == "anchor-query-stage":
                    stage_raw, _ = _load_receipted_record(
                        prepared.release.store,
                        terminal_receipt.to_data(),
                        expected_kind="anchor-query-stage",
                        schema=QUERY_STAGE_SCHEMA,
                        digest_field="query_stage_digest",
                    )
                    _verify_query_stage_record(
                        prepared.release.store,
                        stage_raw,
                        expected_authorization_digest=(
                            prepared.release.authorization.record_digest
                        ),
                    )
                    suffix = stage.removeprefix("query_observer_")
                    state.artifact_receipts[f"query_stage_{suffix}"] = terminal_receipt.to_data()
                    state.artifact_receipts[f"query_visual_result_{suffix}"] = dict(
                        stage_raw["query_visual_result_receipt"]
                    )
                    if stage_raw["batch_observer_artifact_receipt"] is not None:
                        state.artifact_receipts[f"query_observer_artifact_{suffix}"] = dict(
                            stage_raw["batch_observer_artifact_receipt"]
                        )
    query_ids = {task.side_0_query_panel_id, task.side_1_query_panel_id}
    for kind, is_query in (
        ("released-support-panel", False), ("released-query-panel", True)
    ):
        directory = durable_root / "objects" / kind
        count = 0
        if directory.is_dir():
            for path in directory.glob("*.json"):
                raw = _read_bounded_json(path, kind)
                panel_id = raw.get("panel_id")
                if (is_query and panel_id in query_ids) or (
                    not is_query
                    and panel_id in {
                        *task.side_0_support_panel_ids,
                        *task.side_1_support_panel_ids,
                    }
                ):
                    count += 1
        if is_query:
            state.query_release_count = count
        else:
            state.support_release_count = count
    custody_receipts = []
    custody_raw: dict[str, dict[str, Any]] = {}
    for kind, name in (
        ("task-freeze", "task_decision_freeze"),
        ("task-decision-commit", "task_decision_commit"),
    ):
        directory = durable_root / "objects" / kind
        matches = []
        if directory.is_dir():
            for path in directory.glob("*.json"):
                raw = _read_bounded_json(path, kind)
                if raw.get("task_plan_digest") == task.record_digest:
                    receipt = prepared.release.store.persist(
                        object_kind=kind,
                        object_digest=raw["record_digest"],
                        data=raw,
                    )
                    prepared.release.store.verify(receipt, expected_data=raw)
                    matches.append(receipt)
        if len(matches) > 1:
            raise ObjectSceneAnchorBenchmarkError("task custody inventory is ambiguous")
        if matches:
            state.artifact_receipts[name] = matches[0].to_data()
            custody_receipts.append(matches[0])
            custody_raw[name] = _read_bounded_json(
                prepared.release.store.root / matches[0].relative_path, kind
            )
    if len(custody_receipts) == 2:
        from bongard.object_scene_anchor_task_decision_custody import (
            ObjectSceneAnchorTaskDecisionCommit,
            ObjectSceneAnchorTaskDecisionFreeze,
            cold_verify_object_scene_anchor_task_decision_commit,
        )

        freeze = ObjectSceneAnchorTaskDecisionFreeze.from_data(
            custody_raw["task_decision_freeze"]
        )
        commit = ObjectSceneAnchorTaskDecisionCommit.from_data(
            custody_raw["task_decision_commit"]
        )
        if (
            freeze.task_id != task.task_id
            or freeze.task_plan_digest != task.record_digest
            or freeze.execution_precommit_digest != prepared.precommit.record_digest
        ):
            raise ObjectSceneAnchorBenchmarkError("recovered task freeze parents differ")
        cold_verify_object_scene_anchor_task_decision_commit(
            commit,
            freeze=freeze,
            exact_freeze_payload=canonical_json(freeze.to_data()) + b"\n",
            expected_commit_digest=commit.record_digest,
        )
        state.formula_custody_verified = True
    return state


def _partial_query_metrics(
    prepared: PreparedObjectSceneAnchorBenchmark,
    state: _TaskState,
) -> tuple[int, int, int, int]:
    from bongard.object_scene_anchor_python_bridge import (
        ObjectSceneAnchorPredictionBucket,
    )
    from bongard.object_scene_anchor_python_query_visual_execution import (
        ObjectSceneAnchorPythonQueryVisualResult,
    )

    expected = (
        ObjectSceneAnchorPredictionBucket.SIDE0_POSITIVE,
        ObjectSceneAnchorPredictionBucket.SIDE1_POSITIVE,
    )
    correct = determinate = abstain = errors = 0
    seen = 0
    for index, wanted in enumerate(expected):
        receipt_data = state.artifact_receipts.get(
            f"query_visual_result_{index:03d}"
        )
        if receipt_data is None:
            errors += 1
            continue
        result = _load_typed(
            prepared.release.store,
            ObjectBongardWriteOnceReceipt.from_data(receipt_data),
            expected_kind="anchor-query-visual-result",
            decoder=ObjectSceneAnchorPythonQueryVisualResult.from_data,
        )
        got = result.prediction.predicted_bucket
        seen += 1
        correct += got is wanted
        if got in (
            ObjectSceneAnchorPredictionBucket.SIDE0_POSITIVE,
            ObjectSceneAnchorPredictionBucket.SIDE1_POSITIVE,
        ):
            determinate += 1
        elif got is ObjectSceneAnchorPredictionBucket.ABSTAIN:
            abstain += 1
        else:
            errors += 1
    if seen > state.query_release_count:
        raise ObjectSceneAnchorBenchmarkError("prediction exists without query release")
    return correct, determinate, abstain, errors


def _latest_stage(stages: Sequence[str]) -> str:
    def key(stage: str) -> tuple[int, str]:
        if stage == "proposer":
            return (0, stage)
        if stage == "support_observer":
            return (1, stage)
        if stage == "ranker":
            return (2, stage)
        if stage.startswith("query_observer_"):
            return (3, stage)
        return (-1, stage)

    return max(stages, key=key, default="pre_model")


def run_object_scene_anchor_benchmark_task(
    prepared: PreparedObjectSceneAnchorBenchmark,
    task: ObjectBongardTaskPlan,
    *,
    transports: ObjectSceneAnchorBenchmarkTransports = ObjectSceneAnchorBenchmarkTransports(),
) -> tuple[dict[str, Any], ObjectBongardWriteOnceReceipt]:
    """Run or reload one task; every exception becomes a denominator result."""

    if type(prepared) is not PreparedObjectSceneAnchorBenchmark:
        raise TypeError("prepared must be exact benchmark preparation")
    if type(task) is not ObjectBongardTaskPlan or task not in prepared.plan.tasks:
        raise TypeError("task must be an exact member of the prepared plan")
    completed = _find_task_result(prepared.release.store, task)
    if completed is not None:
        return completed
    journal = ObjectSceneAnchorBenchmarkCallJournal(
        prepared.release.store, prepared.release.authorization.record_digest
    )
    try:
        return _run_object_scene_anchor_task_core(
            prepared, task, journal=journal, transports=transports
        )
    except ObjectSceneAnchorBenchmarkDanglingClaim as exc:
        state = _recover_terminal_state(prepared, task)
        correct, determinate, abstain, errors = _partial_query_metrics(
            prepared, state
        )
        return _finish_task(
            prepared,
            state,
            status="infrastructure_error",
            terminal_stage=exc.claim.stage,
            correct_count=correct,
            determinate_count=determinate,
            abstain_count=abstain,
            error_count=errors,
            diagnostic={
                "failure_type": type(exc).__name__,
                "dangling_claim_digest": exc.claim.record_digest,
                "retry_permitted": False,
            },
        )
    except Exception as exc:
        state = _recover_terminal_state(prepared, task)
        correct, determinate, abstain, errors = _partial_query_metrics(
            prepared, state
        )
        return _finish_task(
            prepared,
            state,
            status="pipeline_error",
            terminal_stage=_latest_stage(tuple(state.call_claim_digests)),
            correct_count=correct,
            determinate_count=determinate,
            abstain_count=abstain,
            error_count=errors,
            diagnostic={"failure_type": _failure_type(exc)},
        )


def _campaign_from_tasks(
    prepared: PreparedObjectSceneAnchorBenchmark,
    ordered: Sequence[tuple[Mapping[str, Any], ObjectBongardWriteOnceReceipt]],
) -> dict[str, Any]:
    rows = tuple(ordered)
    if len(rows) != len(prepared.plan.tasks):
        raise ObjectSceneAnchorBenchmarkError("campaign task result count differs")
    task_results = tuple(_validate_task_result_record(item) for item, _ in rows)
    expected_order = tuple(task.task_id for task in prepared.plan.tasks)
    if tuple(item.get("task_id") for item in task_results) != expected_order:
        raise ObjectSceneAnchorBenchmarkError("campaign result order differs from batch plan")
    denominator = sum(int(item["query_denominator"]) for item in task_results)
    correct = sum(int(item["correct_count"]) for item in task_results)
    determinate = sum(int(item["determinate_count"]) for item in task_results)
    abstain = sum(int(item["abstain_count"]) for item in task_results)
    errors = sum(int(item["error_count"]) for item in task_results)
    if denominator != 2 * len(task_results) or determinate + abstain + errors != denominator:
        raise ObjectSceneAnchorBenchmarkError("campaign denominator accounting differs")
    statuses = tuple(str(item["status"]) for item in task_results)
    status = (
        "success"
        if all(item == "success" for item in statuses) and errors == 0
        else "completed_with_errors"
        if errors or any(item.endswith("error") for item in statuses)
        else "completed_with_gaps"
    )
    return _seal(
        {
            "schema": CAMPAIGN_RESULT_SCHEMA,
            "command_id": COMMAND_ID,
            "status": status,
            "batch_plan_digest": prepared.plan.record_digest,
            "execution_precommit_digest": prepared.precommit.record_digest,
            "release_authorization_digest": prepared.release.authorization.record_digest,
            "exposure_predecessor_digest": prepared.predecessor.digest,
            "exposure_successor_digest": prepared.release.successor.digest,
            "runtime_digest": prepared.runtime_record["runtime_digest"],
            "bootstrap_receipt": prepared.bootstrap_receipt.to_data(),
            "task_result_receipts": [receipt.to_data() for _item, receipt in rows],
            "task_result_digests": [item["task_result_digest"] for item in task_results],
            "task_statuses": list(statuses),
            "task_count": len(task_results),
            "query_denominator": denominator,
            "correct_count": correct,
            "determinate_count": determinate,
            "abstain_count": abstain,
            "error_count": errors,
            "accuracy_ppm": 0 if denominator == 0 else correct * 1_000_000 // denominator,
            "coverage_ppm": 0 if denominator == 0 else determinate * 1_000_000 // denominator,
            "physical_call_slots_at_risk_by_stage": {
                stage: sum(
                    int(item["physical_call_slots_at_risk_by_stage"].get(stage, 0))
                    for item in task_results
                )
                for stage in sorted(
                    {
                        stage
                        for item in task_results
                        for stage in item["physical_call_slots_at_risk_by_stage"]
                    }
                )
            },
            "physical_call_slots_at_risk": sum(
                int(item["physical_call_slots_at_risk"]) for item in task_results
            ),
            "result_order_is_batch_plan_order": True,
            "all_terminal_outcomes_remain_in_denominator": True,
            **_authority_data(),
        },
        "campaign_result_digest",
    )


def _cold_verify_task_result(
    prepared: PreparedObjectSceneAnchorBenchmark,
    task: ObjectBongardTaskPlan,
    value: object,
) -> dict[str, Any]:
    """Recompute one durable task outcome from exact parents, with no calls."""

    from bongard.official_panel_archive import ReleasedOfficialPanel
    from bongard.object_scene_anchor_batch_observer import (
        ObjectSceneAnchorBatchObserverArtifact,
        verify_object_scene_anchor_batch_observer_artifact,
    )
    from bongard.object_scene_anchor_candidate_ranker import (
        ObjectSceneAnchorCandidateRanker,
        ObjectSceneAnchorRankCapacityGap,
        ObjectSceneAnchorRankInput,
        ObjectSceneAnchorRankResponse,
        freeze_object_scene_anchor_rank_input,
    )
    from bongard.object_scene_anchor_card_proposer import (
        ObjectSceneAnchorCardProposerArtifact,
        ObjectSceneAnchorCardProposerInput,
        ObjectSceneAnchorCardProposerPanelInput,
        freeze_object_scene_anchor_card_proposer_input,
        verify_object_scene_anchor_card_proposer_artifact,
    )
    from bongard.object_scene_anchor_python_bridge import (
        ObjectSceneAnchorPredictionBucket,
        ObjectSceneAnchorPythonBridgeArtifact,
        ObjectSceneAnchorPythonPrediction,
        cold_verify_object_scene_anchor_python_bridge,
    )
    from bongard.object_scene_anchor_python_predicate import (
        ObjectSceneAnchorPythonPredicate,
    )
    from bongard.object_scene_anchor_python_query_visual_execution import (
        ObjectSceneAnchorPythonQueryPanelInput,
        ObjectSceneAnchorPythonQueryVisualPlan,
        ObjectSceneAnchorPythonQueryVisualResult,
        build_object_scene_anchor_python_query_visual_plan,
        cold_verify_object_scene_anchor_python_query_visual_result,
        verify_object_scene_anchor_python_query_visual_runtime,
    )
    from bongard.object_scene_anchor_support_observation_join import (
        ObjectSceneAnchorSupportObservationPlan,
        ObjectSceneAnchorSupportObservationResult,
        build_object_scene_anchor_support_observation_plan,
        cold_verify_object_scene_anchor_support_observation_result,
    )
    from bongard.object_scene_anchor_support_preparation import (
        ObjectSceneAnchorSupportCorpusFreeze,
    )
    from bongard.object_scene_anchor_task_decision_custody import (
        ObjectSceneAnchorTaskDecisionCommit,
        ObjectSceneAnchorTaskDecisionFreeze,
        cold_verify_object_scene_anchor_task_decision_commit,
        cold_verify_object_scene_anchor_task_decision_freeze,
    )
    from bongard.object_scene_anchor_task_support_adapter import (
        ObjectSceneAnchorTaskSupportAdapter,
        build_object_scene_anchor_task_support_corpus,
        verify_object_scene_anchor_task_support_corpus,
    )
    from bongard.object_scene_anchor_version_space import (
        ObjectSceneAnchorPredicateLanguage,
        ObjectSceneAnchorSupportGapKind,
        ObjectSceneAnchorSupportVersionSpace,
        project_object_scene_anchor_card_proposal,
    )

    raw = _validate_task_result_record(value)
    if (
        raw["task_id"] != task.task_id
        or raw["task_plan_digest"] != task.record_digest
        or raw["batch_plan_digest"] != prepared.plan.record_digest
        or raw["execution_precommit_digest"] != prepared.precommit.record_digest
        or raw["release_authorization_digest"]
        != prepared.release.authorization.record_digest
        or raw["runtime_digest"] != prepared.runtime_record["runtime_digest"]
    ):
        raise ObjectSceneAnchorBenchmarkError("task result prepared parents differ")
    receipts = {
        name: ObjectBongardWriteOnceReceipt.from_data(data)
        for name, data in raw["artifact_receipts"].items()
    }
    for name, receipt in receipts.items():
        stored = _read_bounded_json(
            prepared.release.store.root / receipt.relative_path,
            f"task artifact {name}",
        )
        prepared.release.store.verify(receipt, expected_data=stored)

    recovered = _recover_terminal_state(prepared, task)
    if (
        raw["call_claim_digests"] != dict(sorted(recovered.call_claim_digests.items()))
        or raw["call_terminal_digests"]
        != dict(sorted(recovered.call_terminal_digests.items()))
        or raw["physical_call_slots_at_risk_by_stage"]
        != dict(sorted(recovered.physical_calls.items()))
        or raw["support_release_count"] != recovered.support_release_count
        or raw["query_release_count"] != recovered.query_release_count
    ):
        raise ObjectSceneAnchorBenchmarkError("task journal/release accounting differs")

    # Infrastructure and deterministic pipeline errors can end at any stage.
    # Their exact durable call/release graph and any completed query side are
    # the complete replayable outcome; no exception is re-executed.
    if raw["status"] in ("infrastructure_error", "pipeline_error"):
        correct, determinate, abstain, errors = _partial_query_metrics(
            prepared, recovered
        )
        if (
            (raw["correct_count"], raw["determinate_count"], raw["abstain_count"], raw["error_count"])
            != (correct, determinate, abstain, errors)
            or raw["formula_frozen_and_committed_before_query_release"]
            != (raw["query_release_count"] == 0 or recovered.formula_custody_verified)
        ):
            raise ObjectSceneAnchorBenchmarkError("error task replay accounting differs")
        return raw

    def typed(
        name: str,
        kind: str,
        decoder: Callable[[Mapping[str, Any]], object],
        digest_attribute: str,
    ) -> object:
        try:
            receipt = receipts[name]
        except KeyError as exc:
            raise ObjectSceneAnchorBenchmarkError(f"task omits {name}") from exc
        result = _load_typed(
            prepared.release.store, receipt,
            expected_kind=kind, decoder=decoder,
        )
        if receipt.object_digest != _object_address(getattr(result, digest_attribute)):
            raise ObjectSceneAnchorBenchmarkError(f"{name} receipt identity differs")
        return result

    released_support = tuple(
        typed(
            f"support_release_{index:03d}", "released-support-panel",
            ReleasedOfficialPanel.from_data, "record_digest",
        )
        for index in range(12)
    )
    support_runtime = build_object_scene_anchor_task_support_corpus(
        task=task,
        prepared=prepared.release,
        released_panels=released_support,
    )
    adapter = typed(
        "task_support_adapter", "anchor-task-support-adapter",
        ObjectSceneAnchorTaskSupportAdapter.from_data, "adapter_digest",
    )
    corpus = typed(
        "support_corpus", "anchor-support-corpus",
        ObjectSceneAnchorSupportCorpusFreeze.from_data, "freeze_digest",
    )
    if adapter != support_runtime.adapter or corpus != support_runtime.support_corpus.freeze:
        raise ObjectSceneAnchorBenchmarkError("support adapter/corpus replay differs")
    verify_object_scene_anchor_task_support_corpus(
        support_runtime, task=task, prepared=prepared.release,
        expected_adapter_digest=adapter.adapter_digest,
    )
    proposer_rows = tuple(
        ObjectSceneAnchorCardProposerPanelInput(
            panel.freeze.support_sheet,
            panel.exact_support_sheet_png_bytes,
            panel.freeze.panel_manifest,
        )
        for panel in support_runtime.support_corpus.panels
    )
    side0, side1 = proposer_rows[:6], proposer_rows[6:]
    proposer_input = typed(
        "proposer_input", "anchor-proposer-input",
        ObjectSceneAnchorCardProposerInput.from_data, "input_digest",
    )
    if proposer_input != freeze_object_scene_anchor_card_proposer_input(side0, side1):
        raise ObjectSceneAnchorBenchmarkError("proposer input cold replay differs")
    proposer_artifact = typed(
        "proposer_artifact", "anchor-proposer-artifact",
        ObjectSceneAnchorCardProposerArtifact.from_data, "artifact_digest",
    )
    verify_object_scene_anchor_card_proposer_artifact(
        proposer_artifact, side0, side1,
        expected_artifact_digest=proposer_artifact.artifact_digest,
        expected_input_digest=proposer_input.input_digest,
        model=prepared.runtime.model,
        reasoning_effort=prepared.runtime.reasoning_effort,
        expected_launcher_digest=prepared.runtime.expected_launcher_digest,
        cloud_policy_cache_snapshot=prepared.runtime.cloud_policy_cache_snapshot,
        model_catalog_snapshot=prepared.runtime.model_catalog_snapshot,
        no_tools_attestation=prepared.runtime.no_tools_attestation,
    )
    _require_successful_stage_claim(
        prepared.release.store,
        raw,
        release_authorization_digest=(
            prepared.release.authorization.record_digest
        ),
        stage="proposer",
        context_digest=_object_address(proposer_input.input_digest),
        expected_physical_call_count=1,
        artifact_receipt=receipts["proposer_artifact"],
    )
    if proposer_artifact.status != "success":
        if (
            raw["status"] != "proposer_gap"
            or (raw["correct_count"], raw["determinate_count"], raw["abstain_count"], raw["error_count"])
            != (0, 0, 2, 0)
        ):
            raise ObjectSceneAnchorBenchmarkError("proposer gap result differs")
        return raw
    if proposer_artifact.proposal is None:
        raise ObjectSceneAnchorBenchmarkError("successful proposer has no proposal")
    language = typed(
        "predicate_language", "anchor-predicate-language",
        ObjectSceneAnchorPredicateLanguage.from_data, "language_digest",
    )
    if language != project_object_scene_anchor_card_proposal(proposer_artifact.proposal):
        raise ObjectSceneAnchorBenchmarkError("predicate language replay differs")
    support_bundle = build_object_scene_anchor_support_observation_plan(
        support_runtime.support_corpus, language
    )
    support_plan = typed(
        "support_observation_plan", "anchor-support-observation-plan",
        ObjectSceneAnchorSupportObservationPlan.from_data, "plan_digest",
    )
    if support_plan != support_bundle.plan:
        raise ObjectSceneAnchorBenchmarkError("support observation plan replay differs")
    batch_artifact = typed(
        "support_observer_artifact", "anchor-support-observer-artifact",
        ObjectSceneAnchorBatchObserverArtifact.from_data, "artifact_digest",
    )
    verify_object_scene_anchor_batch_observer_artifact(
        batch_artifact, support_bundle.batch_inputs,
        expected_artifact_digest=batch_artifact.artifact_digest,
        expected_plan_digest=support_plan.batch_plan_digest,
        expected_observation_plan_digest=support_plan.observation_context_digest,
    )
    _require_successful_stage_claim(
        prepared.release.store,
        raw,
        release_authorization_digest=(
            prepared.release.authorization.record_digest
        ),
        stage="support_observer",
        context_digest=_object_address(support_plan.observation_context_digest),
        expected_physical_call_count=support_plan.batch_plan.physical_call_count,
        artifact_receipt=receipts["support_observer_artifact"],
    )
    support_result = typed(
        "support_observation_result", "anchor-support-observation-result",
        ObjectSceneAnchorSupportObservationResult.from_data, "result_digest",
    )
    cold_verify_object_scene_anchor_support_observation_result(
        support_result, plan=support_plan, artifact=batch_artifact
    )
    spaces = (
        support_result.bucket0_positive_version_space,
        support_result.bucket1_positive_version_space,
    )
    for index, space in enumerate(spaces):
        stored = typed(
            f"version_space_{index}", "anchor-support-version-space",
            ObjectSceneAnchorSupportVersionSpace.from_data, "version_space_digest",
        )
        if stored != space:
            raise ObjectSceneAnchorBenchmarkError("orientation version space replay differs")
    nonempty = tuple(sorted(
        (space for space in spaces if space.survivor_candidate_digests),
        key=lambda item: item.version_space_digest,
    ))
    if not nonempty:
        gaps = tuple(space.gap for space in spaces)
        expected_status = (
            "language_gap"
            if all(gap is not None and gap.kind is ObjectSceneAnchorSupportGapKind.LANGUAGE_GAP for gap in gaps)
            else "witness_gap"
        )
        if raw["status"] != expected_status or raw["abstain_count"] != 2:
            raise ObjectSceneAnchorBenchmarkError("typed orientation gap replay differs")
        return raw
    try:
        rebuilt_rank_input = freeze_object_scene_anchor_rank_input(
            nonempty[0], None if len(nonempty) == 1 else nonempty[1]
        )
    except ObjectSceneAnchorRankCapacityGap:
        if raw["status"] != "capacity_gap" or raw["abstain_count"] != 2:
            raise ObjectSceneAnchorBenchmarkError("rank capacity gap replay differs")
        return raw
    rank_input = typed(
        "rank_input", "anchor-rank-input",
        ObjectSceneAnchorRankInput.from_data, "rank_input_digest",
    )
    if rank_input != rebuilt_rank_input:
        raise ObjectSceneAnchorBenchmarkError("rank input replay differs")
    response = typed(
        "rank_response", "anchor-rank-response",
        ObjectSceneAnchorRankResponse.from_data, "response_digest",
    )
    ranker = ObjectSceneAnchorCandidateRanker(
        model=prepared.runtime.model,
        expected_launcher_digest=prepared.runtime.expected_launcher_digest,
        cloud_policy_cache_snapshot=prepared.runtime.cloud_policy_cache_snapshot,
        model_catalog_snapshot=prepared.runtime.model_catalog_snapshot,
        no_tools_attestation=prepared.runtime.no_tools_attestation,
        reasoning_effort=prepared.runtime.reasoning_effort,
        minutes=prepared.runtime.minutes,
        verbose=prepared.runtime.verbose,
        executable=prepared.runtime.executable,
    )
    ranker.verify_response(
        response, version_space=nonempty[0],
        additional_version_space=None if len(nonempty) == 1 else nonempty[1],
        expected_response_digest=response.response_digest,
        expected_rank_input_digest=rank_input.rank_input_digest,
    )
    _require_successful_stage_claim(
        prepared.release.store,
        raw,
        release_authorization_digest=(
            prepared.release.authorization.record_digest
        ),
        stage="ranker",
        context_digest=_object_address(rank_input.rank_input_digest),
        expected_physical_call_count=1,
        artifact_receipt=receipts["rank_response"],
    )
    bridge = typed(
        "python_bridge", "anchor-python-bridge",
        ObjectSceneAnchorPythonBridgeArtifact.from_data, "bridge_digest",
    )
    cold_verify_object_scene_anchor_python_bridge(
        bridge, response=response, first_version_space=spaces[0],
        second_version_space=spaces[1],
        expected_bridge_digest=bridge.bridge_digest,
        expected_response_digest=response.response_digest,
        expected_rank_input_digest=rank_input.rank_input_digest,
    )
    predicate = typed(
        "python_predicate", "anchor-python-predicate",
        ObjectSceneAnchorPythonPredicate.from_data, "predicate_digest",
    )
    if predicate != bridge.predicate:
        raise ObjectSceneAnchorBenchmarkError("selected predicate replay differs")
    freeze = typed(
        "task_decision_freeze", "task-freeze",
        ObjectSceneAnchorTaskDecisionFreeze.from_data, "record_digest",
    )
    cold_verify_object_scene_anchor_task_decision_freeze(
        freeze, task=task, execution_precommit=prepared.precommit,
        task_support_adapter=adapter, card_proposer_artifact=proposer_artifact,
        support_observation_plan=support_plan,
        batch_observer_artifact=batch_artifact,
        support_observation_result=support_result, rank_input=rank_input,
        rank_response=response, bridge=bridge, predicate=predicate,
        expected_freeze_digest=freeze.record_digest,
    )
    commit = typed(
        "task_decision_commit", "task-decision-commit",
        ObjectSceneAnchorTaskDecisionCommit.from_data, "record_digest",
    )
    cold_verify_object_scene_anchor_task_decision_commit(
        commit, freeze=freeze,
        exact_freeze_payload=canonical_json(freeze.to_data()) + b"\n",
        expected_commit_digest=commit.record_digest,
    )
    if raw["query_release_count"] != 2 or raw["formula_frozen_and_committed_before_query_release"] is not True:
        raise ObjectSceneAnchorBenchmarkError("query release chronology differs")
    released_queries = tuple(
        typed(
            f"query_release_{index:03d}", "released-query-panel",
            ReleasedOfficialPanel.from_data, "record_digest",
        )
        for index in range(2)
    )
    expected_panel_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    if tuple(item.panel_id for item in released_queries) != expected_panel_ids:
        raise ObjectSceneAnchorBenchmarkError("released query panel order differs")
    predictions = []
    result_digests = []
    for index, released in enumerate(released_queries):
        plan = typed(
            f"query_visual_plan_{index:03d}", "anchor-query-visual-plan",
            ObjectSceneAnchorPythonQueryVisualPlan.from_data, "plan_digest",
        )
        expected_source_binding = _address(
            {
                "schema": "gkm.object-scene-anchor-query-source-binding.v1",
                "task_plan_digest": task.record_digest,
                "query_alias": f"panel_{index:03d}",
                "released_panel_record_digest": released.record_digest,
                "release_store_receipt_digest": receipts[
                    f"query_release_{index:03d}"
                ].record_digest,
                "predicate_digest": predicate.predicate_digest,
                "task_decision_commit_receipt_digest": receipts[
                    "task_decision_commit"
                ].record_digest,
            }
        )
        if plan.source_binding_digest != expected_source_binding:
            raise ObjectSceneAnchorBenchmarkError(
                "query source binding replay differs"
            )
        panel_input = ObjectSceneAnchorPythonQueryPanelInput(
            released, f"panel_{index:03d}", plan.source_binding_digest
        )
        query_runtime = build_object_scene_anchor_python_query_visual_plan(
            panel_input, predicate
        )
        if query_runtime.plan != plan:
            raise ObjectSceneAnchorBenchmarkError("query visual plan cold replay differs")
        verify_object_scene_anchor_python_query_visual_runtime(
            query_runtime, panel_input=panel_input, predicate=predicate,
            expected_plan_digest=plan.plan_digest,
        )
        stage_receipt = receipts[f"query_stage_{index:03d}"]
        stage_raw, _ = _load_receipted_record(
            prepared.release.store, stage_receipt.to_data(),
            expected_kind="anchor-query-stage", schema=QUERY_STAGE_SCHEMA,
            digest_field="query_stage_digest",
        )
        _verify_query_stage_record(
            prepared.release.store, stage_raw,
            expected_authorization_digest=prepared.release.authorization.record_digest,
        )
        if stage_raw["task_plan_digest"] != task.record_digest:
            raise ObjectSceneAnchorBenchmarkError("query stage task parent differs")
        if (
            stage_raw["query_visual_plan_receipt"]
            != receipts[f"query_visual_plan_{index:03d}"].to_data()
            or stage_raw["query_visual_result_receipt"]
            != receipts[f"query_visual_result_{index:03d}"].to_data()
            or stage_raw["batch_observer_artifact_receipt"]
            != (
                None
                if f"query_observer_artifact_{index:03d}" not in receipts
                else receipts[f"query_observer_artifact_{index:03d}"].to_data()
            )
        ):
            raise ObjectSceneAnchorBenchmarkError(
                "query stage receipts differ from task graph"
            )
        _require_successful_stage_claim(
            prepared.release.store,
            raw,
            release_authorization_digest=(
                prepared.release.authorization.record_digest
            ),
            stage=f"query_observer_{index:03d}",
            context_digest=_object_address(plan.observation_context_digest),
            expected_physical_call_count=plan.physical_call_count,
            artifact_receipt=stage_receipt,
        )
        batch = None
        if stage_raw["batch_observer_artifact_receipt"] is not None:
            batch = _load_typed(
                prepared.release.store,
                ObjectBongardWriteOnceReceipt.from_data(stage_raw["batch_observer_artifact_receipt"]),
                expected_kind="anchor-query-observer-artifact",
                decoder=ObjectSceneAnchorBatchObserverArtifact.from_data,
            )
            verify_object_scene_anchor_batch_observer_artifact(
                batch, query_runtime.batch_inputs,
                expected_artifact_digest=batch.artifact_digest,
                expected_plan_digest=plan.batch_plan_digest,
                expected_observation_plan_digest=plan.observation_context_digest,
            )
        result = _load_typed(
            prepared.release.store,
            ObjectBongardWriteOnceReceipt.from_data(stage_raw["query_visual_result_receipt"]),
            expected_kind="anchor-query-visual-result",
            decoder=ObjectSceneAnchorPythonQueryVisualResult.from_data,
        )
        cold_verify_object_scene_anchor_python_query_visual_result(
            result, plan=plan, artifact=batch,
            expected_result_digest=result.result_digest,
        )
        prediction = typed(
            f"query_prediction_{index:03d}", "anchor-query-prediction",
            ObjectSceneAnchorPythonPrediction.from_data, "prediction_digest",
        )
        if prediction != result.prediction:
            raise ObjectSceneAnchorBenchmarkError("query prediction replay differs")
        predictions.append(prediction)
        result_digests.append(result.result_digest)
    expected = (
        ObjectSceneAnchorPredictionBucket.SIDE0_POSITIVE,
        ObjectSceneAnchorPredictionBucket.SIDE1_POSITIVE,
    )
    predicted = tuple(item.predicted_bucket for item in predictions)
    recomputed = {
        "correct_count": sum(got is want for got, want in zip(predicted, expected, strict=True)),
        "determinate_count": sum(got in expected for got in predicted),
        "abstain_count": sum(got is ObjectSceneAnchorPredictionBucket.ABSTAIN for got in predicted),
        "error_count": sum(got is ObjectSceneAnchorPredictionBucket.ERROR for got in predicted),
    }
    score_receipt = receipts["query_score"]
    score_raw, _ = _load_receipted_record(
        prepared.release.store, score_receipt.to_data(),
        expected_kind="anchor-query-score", schema=QUERY_SCORE_SCHEMA,
        digest_field="score_digest",
    )
    _verify_query_score_record(
        score_raw,
        expected_task_plan_digest=task.record_digest,
        expected_predicate_digest=predicate.predicate_digest,
        expected_query_visual_result_digests=result_digests,
        expected_prediction_digests=[
            item.prediction_digest for item in predictions
        ],
        expected_predicted_buckets=[item.value for item in predicted],
    )
    if (
        any(score_raw.get(key) != item for key, item in recomputed.items())
        or any(raw[key] != item for key, item in recomputed.items())
        or raw["status"] != ("query_error" if recomputed["error_count"] else "success")
        or raw["terminal_stage"] != "score"
        or raw["diagnostic"] != {"score_digest": score_raw["score_digest"]}
    ):
        raise ObjectSceneAnchorBenchmarkError("query score/task counters differ on replay")
    return raw


def cold_replay_object_scene_anchor_benchmark(
    prepared: PreparedObjectSceneAnchorBenchmark,
    campaign: Mapping[str, Any],
) -> dict[str, Any]:
    """Cold-replay the complete durable result graph without any model call."""

    raw = _verify_seal(
        campaign,
        schema=CAMPAIGN_RESULT_SCHEMA,
        digest_field="campaign_result_digest",
        label="campaign result",
    )
    receipts = raw.get("task_result_receipts")
    if not isinstance(receipts, list):
        raise ObjectSceneAnchorBenchmarkError("campaign task receipts differ")
    rows = []
    for task, receipt_data in zip(prepared.plan.tasks, receipts, strict=True):
        task_raw, receipt = _load_receipted_record(
            prepared.release.store,
            receipt_data,
            expected_kind="anchor-task-result",
            schema=TASK_RESULT_SCHEMA,
            digest_field="task_result_digest",
        )
        rows.append((_cold_verify_task_result(prepared, task, task_raw), receipt))
    rebuilt = _campaign_from_tasks(prepared, rows)
    if rebuilt != raw:
        raise ObjectSceneAnchorBenchmarkError("campaign aggregate differs on cold replay")
    return _seal(
        {
            "schema": CAMPAIGN_REPLAY_SCHEMA,
            "command_id": COMMAND_ID,
            "campaign_result_digest": raw["campaign_result_digest"],
            "task_result_digests": raw["task_result_digests"],
            "model_calls": 0,
            "model_free": True,
            "tamper_detecting": True,
            "completed": True,
            **_authority_data(),
        },
        "replay_digest",
    )


def run_prepared_object_scene_anchor_benchmark(
    prepared: PreparedObjectSceneAnchorBenchmark,
    *,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    transports: ObjectSceneAnchorBenchmarkTransports = ObjectSceneAnchorBenchmarkTransports(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(parallel_workers) is not int or not 1 <= parallel_workers <= MAX_PARALLEL_WORKERS:
        raise ObjectSceneAnchorBenchmarkError("parallel_workers must lie in 1..3")
    with ThreadPoolExecutor(max_workers=min(parallel_workers, len(prepared.plan.tasks))) as pool:
        futures = {
            task.record_digest: pool.submit(
                run_object_scene_anchor_benchmark_task,
                prepared,
                task,
                transports=transports,
            )
            for task in prepared.plan.tasks
        }
        ordered = [futures[task.record_digest].result() for task in prepared.plan.tasks]
    campaign = _campaign_from_tasks(prepared, ordered)
    campaign, campaign_receipt = _persist_record(
        prepared.release.store,
        object_kind="anchor-campaign-result",
        record=campaign,
        digest_field="campaign_result_digest",
        schema=CAMPAIGN_RESULT_SCHEMA,
    )
    replay = cold_replay_object_scene_anchor_benchmark(prepared, campaign)
    replay, replay_receipt = _persist_record(
        prepared.release.store,
        object_kind="anchor-campaign-replay",
        record=replay,
        digest_field="replay_digest",
        schema=CAMPAIGN_REPLAY_SCHEMA,
    )
    return (
        {**campaign, "campaign_store_receipt": campaign_receipt.to_data()},
        {**replay, "replay_store_receipt": replay_receipt.to_data()},
    )


def run_object_scene_anchor_benchmark(
    *,
    output_root: str | os.PathLike[str],
    selection_seed: str,
    descriptor_path: str | os.PathLike[str] = DEFAULT_DESCRIPTOR,
    archive_path: str | os.PathLike[str] = DEFAULT_ARCHIVE,
    split_path: str | os.PathLike[str] = DEFAULT_SPLIT,
    predecessor_path: str | os.PathLike[str] = DEFAULT_PREDECESSOR,
    historical_exposure_path: str | os.PathLike[str] = DEFAULT_HISTORICAL_EXPOSURE,
    requested_per_family: int = DEFAULT_REQUESTED_PER_FAMILY,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    minutes: int = DEFAULT_MINUTES,
    verbose: bool = False,
    executable: str = DEFAULT_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_EXPECTED_LAUNCHER_SHA256,
    resume: bool = False,
    transports: ObjectSceneAnchorBenchmarkTransports = ObjectSceneAnchorBenchmarkTransports(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = prepare_object_scene_anchor_benchmark(
        output_root=output_root,
        selection_seed=selection_seed,
        descriptor_path=descriptor_path,
        archive_path=archive_path,
        split_path=split_path,
        predecessor_path=predecessor_path,
        historical_exposure_path=historical_exposure_path,
        requested_per_family=requested_per_family,
        parallel_workers=parallel_workers,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
        resume=resume,
    )
    return run_prepared_object_scene_anchor_benchmark(
        prepared, parallel_workers=parallel_workers, transports=transports
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or cold-replay the exact-unused TRAIN anchor predicate drill "
            "(never an official benchmark)."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", help="required deterministic cohort seed for run/resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument(
        "--expected-campaign-digest",
        help="required sha256: campaign address for replay-only",
    )
    parser.add_argument("--descriptor", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--predecessor", type=Path, default=DEFAULT_PREDECESSOR)
    parser.add_argument(
        "--historical-exposure", type=Path, default=DEFAULT_HISTORICAL_EXPOSURE
    )
    parser.add_argument(
        "--requested-per-family", type=int, default=DEFAULT_REQUESTED_PER_FAMILY
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
    parser.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--executable", default=DEFAULT_EXECUTABLE)
    parser.add_argument(
        "--expected-launcher-sha256", default=DEFAULT_EXPECTED_LAUNCHER_SHA256
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.replay_only:
        if args.seed is not None or args.resume:
            parser.error("--replay-only does not accept --seed or --resume")
        if args.expected_campaign_digest is None:
            parser.error("--replay-only requires --expected-campaign-digest")
        expected = _require_address(
            args.expected_campaign_digest, "expected campaign digest"
        )
        prepared = load_prepared_object_scene_anchor_benchmark_for_replay(
            output_root=args.output_root,
            descriptor_path=args.descriptor,
            archive_path=args.archive,
            split_path=args.split,
            predecessor_path=args.predecessor,
            historical_exposure_path=args.historical_exposure,
        )
        campaign = _verify_seal(
            _only_object(prepared.release.store, "anchor-campaign-result"),
            schema=CAMPAIGN_RESULT_SCHEMA,
            digest_field="campaign_result_digest",
            label="campaign result",
        )
        if campaign["campaign_result_digest"] != expected:
            raise ObjectSceneAnchorBenchmarkError(
                "campaign digest differs from external replay commitment"
            )
        replay = cold_replay_object_scene_anchor_benchmark(prepared, campaign)
        print(canonical_json(replay).decode("utf-8"))
        return 0
    if args.seed is None:
        parser.error("run/resume requires --seed")
    if args.expected_campaign_digest is not None:
        parser.error("--expected-campaign-digest is replay-only")
    campaign, replay = run_object_scene_anchor_benchmark(
        output_root=args.output_root,
        selection_seed=args.seed,
        descriptor_path=args.descriptor,
        archive_path=args.archive,
        split_path=args.split,
        predecessor_path=args.predecessor,
        historical_exposure_path=args.historical_exposure,
        requested_per_family=args.requested_per_family,
        parallel_workers=args.workers,
        minutes=args.minutes,
        verbose=args.verbose,
        executable=args.executable,
        expected_launcher_sha256=args.expected_launcher_sha256,
        resume=args.resume,
    )
    print(canonical_json({"campaign": campaign, "replay": replay}).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke tests.
    raise SystemExit(main())
