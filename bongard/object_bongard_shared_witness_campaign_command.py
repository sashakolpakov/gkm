"""Sealed 12-task TRAIN campaign for structured shared-witness predicates.

An explicitly supplied two-pass calibration must cold-verify as accepted before
the preregistered cohort, archive, output root, panel reader, or model transport
can be reached.  Each task makes one structured vision proposal and 24 support
observer calls, then delegates rank selection and durable freeze-before-query
to the Python shared-witness runner.  A successful campaign makes exactly two
query observer calls per task and scores the fixed denominator of 24.

The completed directory cold-replays journals, observations, predicates,
freezes, and scores with transport forbidden.  Lean is absent and removable.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import CodexNoToolsAttestation, attest_codex_no_tools
from bongard.object_bongard_batch import ObjectBongardBatchPlan, ObjectBongardTaskPlan
from bongard.object_bongard_shared_witness import (
    ObjectBongardSharedWitnessRubricSpec,
    build_shared_witness_rubric_specs,
)
from bongard.object_bongard_shared_witness_calibration_command import (
    DEFAULT_STRUCTURED_NOMINATION_ROOT,
    RESULT_FILENAME as CALIBRATION_RESULT_FILENAME,
    RESULT_SCHEMA as CALIBRATION_RESULT_SCHEMA,
    VerifiedObjectBongardSharedWitnessCalibration,
    object_bongard_shared_witness_calibration_command_source_digest,
    verify_object_bongard_shared_witness_calibration,
)
from bongard.object_bongard_panel_rubric_calibration import (
    DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE,
)
from bongard.object_bongard_shared_witness_observer import (
    ObjectBongardSharedWitnessPanelArtifact,
    observe_object_bongard_shared_witness_panel,
    prepare_object_bongard_shared_witness_panel_inputs,
    verify_object_bongard_shared_witness_panel_artifact,
)
from bongard.object_bongard_shared_witness_semantics import (
    ObjectBongardSharedWitnessSemanticArtifact,
    describe_object_bongard_shared_witness_support,
    object_bongard_shared_witness_semantics_output_schema,
    object_bongard_shared_witness_semantics_prompt,
    verify_object_bongard_shared_witness_semantic_artifact,
)
from bongard.object_bongard_shared_witness_task_runner import (
    ObjectBongardSharedWitnessTaskFreeze,
    ObjectBongardSharedWitnessTaskFreezeCommit,
    ObjectBongardSharedWitnessTaskRunArchive,
    ObjectBongardSharedWitnessTaskRunStatus,
    cold_replay_object_bongard_shared_witness_task,
    run_object_bongard_shared_witness_task,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTurnRuntime,
)
from bongard.official_panel_archive import OfficialPanelArchive, OfficialPanelReceipt
from bongard.prototype_scene_observer import (
    PrototypeSceneObserverStatus,
    prototype_scene_transport_source_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.release import OfficialReleaseDescriptor
from bongard.transport import (
    PINNED_CODEX_CLI_VERSION,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexStructuredResult,
    codex_cli_authenticated_fingerprint,
    run_codex_named_images_structured,
    snapshot_cloud_policy_cache,
    snapshot_pinned_model_catalog,
)


COMMAND_ID = "bongard.shared-witness-campaign/exact-train-12-v1"
AUTHORIZATION_SCHEMA = "gkm.bongard-shared-witness-campaign-authorization.v1"
PRECOMMIT_SCHEMA = "gkm.bongard-shared-witness-campaign-precommit.v1"
TASK_RECORD_SCHEMA = "gkm.bongard-shared-witness-campaign-task-record.v1"
CAMPAIGN_SCHEMA = "gkm.bongard-shared-witness-campaign-result.v1"
REPLAY_SCHEMA = "gkm.bongard-shared-witness-campaign-cold-replay.v1"
CALIBRATION_PARENT_SCHEMA = (
    "gkm.bongard-shared-witness-campaign-accepted-calibration-parent.v1"
)

MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
DEFAULT_MINUTES = 15
DEFAULT_PARALLEL_WORKERS = 12
DEFAULT_EXECUTABLE = "codex"
DEFAULT_EXPECTED_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)
TASK_COUNT = 12
SUPPORT_CALLS_PER_TASK = 24
QUERY_CALLS_PER_TASK = 2
QUERY_DENOMINATOR = 24
MAX_PHYSICAL_CALLS = TASK_COUNT * (1 + SUPPORT_CALLS_PER_TASK + QUERY_CALLS_PER_TASK)

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREREGISTRATION = _REPOSITORY_ROOT / "bongard/data/object_bongard_rubric_train_20260808.prereg.json"
DEFAULT_PLAN = _REPOSITORY_ROOT / "bongard/data/object_bongard_rubric_train_20260808.plan.json"
DEFAULT_DESCRIPTOR = _REPOSITORY_ROOT / "bongard/data/shape_bongard_v2_release_v1.json"
DEFAULT_ARCHIVE = _REPOSITORY_ROOT / "downloads/ShapeBongard_V2.zip"
DEFAULT_CALIBRATION_ROOT = (
    _REPOSITORY_ROOT
    / "downloads/ShapeBongard_V2_full/object_shared_witness_calibration_20260808_v1"
)
DEFAULT_CALIBRATION_SOURCE_ROOT = (
    _REPOSITORY_ROOT / DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    if not DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE.is_absolute()
    else DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
)
DEFAULT_CALIBRATION_NOMINATION_ROOT = (
    _REPOSITORY_ROOT / DEFAULT_STRUCTURED_NOMINATION_ROOT
    if not DEFAULT_STRUCTURED_NOMINATION_ROOT.is_absolute()
    else DEFAULT_STRUCTURED_NOMINATION_ROOT
)

PREREGISTRATION_FILE_SHA256 = "10d52f9eec047063e1861cd7c151fa6600cf2c4ef4ad6423784cc419db0fb76e"
PLAN_FILE_SHA256 = "c2f07c7885a42f4125f397ddf5bf7f8827b3ef1a6c1fb77e82f08a6ab2b3d523"
PREREGISTRATION_DIGEST = "sha256:b4e29960a9524f5785139a3ddf462d5ddec784d52eb0f2678cb1674820dd8107"
PLAN_DIGEST = "sha256:760edd40d91c67fd3c5e3b6f94119754f5368441b479f0940c2c7bd77c17b941"

PLAN_FILENAME = "plan.json"
AUTHORIZATION_FILENAME = "authorization.json"
PRECOMMIT_FILENAME = "execution_precommit.json"
CAMPAIGN_FILENAME = "campaign.json"
REPLAY_FILENAME = "cold_replay.json"
TASKS_DIRECTORY = "tasks"
JOURNALS_DIRECTORY = "journals"

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_RECORD_BYTES = 256 * 1024 * 1024

PanelReader = Callable[[str], tuple[bytes, OfficialPanelReceipt | Mapping[str, Any]]]
NamedImageTransport = Callable[..., CodexStructuredResult]


class ObjectBongardSharedWitnessCampaignCommandError(RuntimeError):
    """The calibration gate, launch boundary, or cold replay failed closed."""


def object_bongard_shared_witness_campaign_command_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "structured_shared_witness_predicates_only": True,
        "full_ir_and_entity_evidence_persisted": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_selection_decision_scoring_or_replay": False,
        "lean_removal_changes_decision": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "threshold_retries_allowed": False,
        "model_selects_candidate": False,
    }


def _is_raw_digest(value: object) -> bool:
    return isinstance(value, str) and _RAW_DIGEST.fullmatch(value) is not None


def _is_address(value: object) -> bool:
    return isinstance(value, str) and _ADDRESS.fullmatch(value) is not None


def _seal(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = json.loads(canonical_json(dict(body)).decode("utf-8"))
    result[field] = "sha256:" + canonical_digest(result)
    return result


def _validate_seal(value: object, field: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ObjectBongardSharedWitnessCampaignCommandError(f"{label} must be an object")
    raw = json.loads(canonical_json(dict(value)).decode("utf-8"))
    expected = "sha256:" + canonical_digest({key: item for key, item in raw.items() if key != field})
    if raw.get(field) != expected:
        raise ObjectBongardSharedWitnessCampaignCommandError(f"{label} digest differs")
    return raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, value: Mapping[str, Any]) -> bytes:
    payload = canonical_json(dict(value)) + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return payload


def _write_exact_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        info = path.lstat()
        payload = path.read_bytes()
    except OSError as exc:
        raise ObjectBongardSharedWitnessCampaignCommandError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
        or not 0 < len(payload) <= _MAX_RECORD_BYTES
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            f"{label} is not a bounded regular file"
        )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardSharedWitnessCampaignCommandError(f"{label} is malformed") from exc
    if not isinstance(value, dict):
        raise ObjectBongardSharedWitnessCampaignCommandError(f"{label} must be an object")
    return value


def _read_exact_input(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ObjectBongardSharedWitnessCampaignCommandError(f"{label} is unavailable") from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ObjectBongardSharedWitnessCampaignCommandError(f"{label} identity differs")
    return _read_json(path, label)


def _load_exact_cohort(
    preregistration_path: Path, plan_path: Path
) -> tuple[dict[str, Any], ObjectBongardBatchPlan]:
    prereg = _read_exact_input(
        preregistration_path, PREREGISTRATION_FILE_SHA256, "preregistration"
    )
    plan = ObjectBongardBatchPlan.from_data(
        _read_exact_input(plan_path, PLAN_FILE_SHA256, "batch plan")
    )
    body = {key: item for key, item in prereg.items() if key != "record_digest"}
    families = tuple(task.family for task in plan.tasks)
    if (
        prereg.get("record_digest") != PREREGISTRATION_DIGEST
        or "sha256:" + canonical_digest(body) != PREREGISTRATION_DIGEST
        or plan.record_digest != PLAN_DIGEST
        or prereg.get("batch_plan_digest") != plan.record_digest
        or prereg.get("scope") != "exact-unused-train-targeted-engineering-not-official-benchmark"
        or prereg.get("query_identities_sealed_before_support_pixels") is not True
        or prereg.get("panel_bytes_opened_before_preregistration") is not False
        or prereg.get("official_test_authorized") is not False
        or len(plan.tasks) != TASK_COUNT
        or any(families.count(family) != 4 for family in ("bd", "ff", "hd"))
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "preregistered TRAIN cohort differs"
        )
    return prereg, plan


def _validate_accepted_calibration_parent(value: object) -> dict[str, Any]:
    fields = {
        "schema", "calibration_verifier_source_sha256", "calibration_result_digest",
        "calibration_cold_replay_digest", "calibration_historical_source_digest",
        "calibration_authorization_digest", "calibration_execution_precommit_digest",
        "calibration_batch_digest", "calibration_freeze_digest",
        "calibration_assessment_digest", "calibration_nomination_artifact_digest",
        "selected_candidate_rank", "selected_spec_digest", "selected_candidate_digest",
        "fresh_call_count", "reused_call_count", "accepted",
        "cold_verified_before_campaign_cohort_access", "parent_digest",
    }
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != fields
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "accepted calibration parent fields differ"
        )
    raw = _validate_seal(value, "parent_digest", "accepted calibration parent")
    address_fields = (
        "calibration_result_digest", "calibration_cold_replay_digest",
        "calibration_authorization_digest", "calibration_execution_precommit_digest",
        "calibration_batch_digest", "calibration_freeze_digest",
        "calibration_assessment_digest",
    )
    if (
        raw["schema"] != CALIBRATION_PARENT_SCHEMA
        or not _is_raw_digest(raw["calibration_verifier_source_sha256"])
        or not _is_raw_digest(raw["calibration_historical_source_digest"])
        or not _is_raw_digest(raw["calibration_nomination_artifact_digest"])
        or any(not _is_address(raw[name]) for name in address_fields)
        or type(raw["selected_candidate_rank"]) is not int
        or raw["selected_candidate_rank"] not in (0, 1)
        or not _is_raw_digest(raw["selected_spec_digest"])
        or not _is_raw_digest(raw["selected_candidate_digest"])
        or raw["fresh_call_count"] != 48 or raw["reused_call_count"] != 0
        or raw["accepted"] is not True
        or raw["cold_verified_before_campaign_cohort_access"] is not True
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "accepted calibration parent policy differs"
        )
    return raw


def _cold_verify_accepted_calibration_parent(
    calibration_root: str | os.PathLike[str],
) -> dict[str, Any]:
    candidate = Path(calibration_root).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "accepted calibration root is unavailable or linked"
        )
    root = candidate.resolve(strict=True)
    verified = verify_object_bongard_shared_witness_calibration(
        root,
        nomination_root=DEFAULT_CALIBRATION_NOMINATION_ROOT,
        source_root=DEFAULT_CALIBRATION_SOURCE_ROOT,
    )
    if not isinstance(verified, VerifiedObjectBongardSharedWitnessCalibration):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "calibration verifier returned the wrong type"
        )
    result = _validate_seal(
        _read_json(root / CALIBRATION_RESULT_FILENAME, "calibration result"),
        "result_digest",
        "calibration result",
    )
    selected_candidate_digest = result.get("selected_candidate_digest")
    if (
        verified.output_root != root or verified.accepted is not True
        or verified.selected_candidate_rank not in (0, 1)
        or not _is_raw_digest(verified.selected_spec_digest)
        or verified.fresh_call_count != 48 or verified.reused_call_count != 0
        or result.get("schema") != CALIBRATION_RESULT_SCHEMA
        or result.get("result_digest") != verified.result_digest
        or result.get("cold_replay_digest") != verified.replay_digest
        or result.get("historical_source_digest") != verified.source_digest
        or result.get("authorization_digest") != verified.authorization_digest
        or result.get("execution_precommit_digest") != verified.execution_precommit_digest
        or result.get("batch_digest") != verified.batch_digest
        or result.get("freeze_digest") != verified.freeze_digest
        or result.get("assessment_digest") != verified.assessment_digest
        or result.get("nomination_artifact_digest") != verified.nomination_artifact_digest
        or result.get("accepted") is not True
        or result.get("selected_candidate_rank") != verified.selected_candidate_rank
        or result.get("selected_spec_digest") != verified.selected_spec_digest
        or not _is_raw_digest(selected_candidate_digest)
        or result.get("fresh_call_count") != 48 or result.get("reused_call_count") != 0
        or result.get("physical_call_denominator") != 48
        or result.get("campaign_gate_lineage_complete") is not True
        or result.get("all_48_artifacts_frozen_and_reloaded_before_assessment") is not True
        or result.get("model_calls_during_assessment_or_replay") != 0
        or result.get("query_pixels_used") is not False
        or result.get("fresh_broad_cohort_pixels_used") is not False
        or result.get("official_test_pixels_used") is not False
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "calibration is not an accepted cold-verified campaign parent"
        )
    return _validate_accepted_calibration_parent(
        _seal(
            {
                "schema": CALIBRATION_PARENT_SCHEMA,
                "calibration_verifier_source_sha256": (
                    object_bongard_shared_witness_calibration_command_source_digest()
                ),
                "calibration_result_digest": verified.result_digest,
                "calibration_cold_replay_digest": verified.replay_digest,
                "calibration_historical_source_digest": verified.source_digest,
                "calibration_authorization_digest": verified.authorization_digest,
                "calibration_execution_precommit_digest": verified.execution_precommit_digest,
                "calibration_batch_digest": verified.batch_digest,
                "calibration_freeze_digest": verified.freeze_digest,
                "calibration_assessment_digest": verified.assessment_digest,
                "calibration_nomination_artifact_digest": verified.nomination_artifact_digest,
                "selected_candidate_rank": verified.selected_candidate_rank,
                "selected_spec_digest": verified.selected_spec_digest,
                "selected_candidate_digest": selected_candidate_digest,
                "fresh_call_count": verified.fresh_call_count,
                "reused_call_count": verified.reused_call_count,
                "accepted": True,
                "cold_verified_before_campaign_cohort_access": True,
            },
            "parent_digest",
        )
    )


def _ensure_fresh_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value).absolute()
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "campaign output root must be fresh"
        ) from exc
    _fsync_directory(root.parent)
    return root


def _runtime_kwargs(runtime: ObjectBongardTurnRuntime) -> dict[str, object]:
    return {
        "model": runtime.model,
        "reasoning_effort": runtime.reasoning_effort,
        "minutes": runtime.minutes,
        "verbose": runtime.verbose,
        "executable": runtime.executable,
        "cloud_policy_cache_snapshot": runtime.cloud_policy_cache_snapshot,
        "expected_launcher_digest": runtime.expected_launcher_digest,
        "model_catalog_snapshot": runtime.model_catalog_snapshot,
        "no_tools_attestation": runtime.no_tools_attestation,
    }


def _authorization_record(
    *,
    plan: ObjectBongardBatchPlan,
    preregistration: Mapping[str, Any],
    accepted_calibration_parent: Mapping[str, Any],
    archive_identity: Mapping[str, Any],
    minutes: int,
    parallel_workers: int,
    expected_launcher_sha256: str,
) -> dict[str, Any]:
    parent = _validate_accepted_calibration_parent(accepted_calibration_parent)
    jobs = [
        {
            "task_index": index,
            "task_id": task.task_id,
            "task_plan_digest": task.record_digest,
            "semantic_calls": 1,
            "support_observer_calls": SUPPORT_CALLS_PER_TASK,
            "query_observer_calls": QUERY_CALLS_PER_TASK,
            "sealed_query_panel_ids": [
                task.side_0_query_panel_id, task.side_1_query_panel_id
            ],
        }
        for index, task in enumerate(plan.tasks)
    ]
    return _seal(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "command_id": COMMAND_ID,
            "command_source_sha256": object_bongard_shared_witness_campaign_command_source_digest(),
            "preregistration_digest": preregistration["record_digest"],
            "batch_plan_digest": plan.record_digest,
            "selected_task_ids_digest": preregistration["selected_task_ids_digest"],
            "sealed_query_panel_ids_digest": preregistration["sealed_query_panel_ids_digest"],
            "scope": preregistration["scope"],
            "accepted_calibration_parent": parent,
            "accepted_calibration_parent_digest": parent["parent_digest"],
            "calibration_result_digest": parent["calibration_result_digest"],
            "calibration_cold_replay_digest": parent["calibration_cold_replay_digest"],
            "calibration_historical_source_digest": parent[
                "calibration_historical_source_digest"
            ],
            "archive_identity": dict(archive_identity),
            "jobs": jobs,
            "task_count": TASK_COUNT,
            "semantic_calls_per_task": 1,
            "support_observer_calls_per_task": SUPPORT_CALLS_PER_TASK,
            "query_observer_calls_per_task": QUERY_CALLS_PER_TASK,
            "exact_query_observer_calls": QUERY_DENOMINATOR,
            "maximum_physical_model_calls": MAX_PHYSICAL_CALLS,
            "fixed_campaign_score_denominator": QUERY_DENOMINATOR,
            "complete_task_score_denominator": 2,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "minutes": minutes,
            "parallel_workers": parallel_workers,
            "expected_launcher_sha256": expected_launcher_sha256,
            "fresh_journals_required": True,
            "support_and_query_journals_disjoint": True,
            "query_bytes_only_after_selected_rank_formula_freeze": True,
            "official_test_authorized": False,
            **_authority_data(),
        },
        "authorization_digest",
    )


def _prepare_runtime_precommit(
    *,
    authorization: Mapping[str, Any],
    plan: ObjectBongardBatchPlan,
    accepted_calibration_parent: Mapping[str, Any],
    minutes: int,
    executable: str,
    expected_launcher_sha256: str,
    cloud_policy_cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot],
    model_catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot],
    launcher_fingerprinter: Callable[..., Mapping[str, str]],
    runtime_attester: Callable[..., CodexNoToolsAttestation],
) -> tuple[dict[str, Any], ObjectBongardTurnRuntime]:
    parent = _validate_accepted_calibration_parent(accepted_calibration_parent)
    if authorization.get("accepted_calibration_parent") != parent:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "authorization calibration parent differs"
        )
    cache = cloud_policy_cache_snapshotter()
    catalog = model_catalog_snapshotter()
    if not isinstance(cache, CloudPolicyCacheSnapshot) or not isinstance(
        catalog, CodexModelCatalogSnapshot
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "runtime snapshot type differs"
        )
    fingerprint = dict(
        launcher_fingerprinter(
            executable, expected_launcher_digest=expected_launcher_sha256
        )
    )
    if fingerprint != {
        "version": PINNED_CODEX_CLI_VERSION,
        "launcher_digest": expected_launcher_sha256,
    }:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "authenticated launcher fingerprint differs"
        )
    attestation = runtime_attester(
        executable=executable,
        expected_launcher_digest=expected_launcher_sha256,
        model_catalog_snapshot=catalog,
        cloud_policy_cache_snapshot=cache,
    )
    if not isinstance(attestation, CodexNoToolsAttestation):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "no-tools attestation type differs"
        )
    runtime = ObjectBongardTurnRuntime(
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        minutes=minutes,
        verbose=False,
        executable=executable,
        cloud_policy_cache_snapshot=cache,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=expected_launcher_sha256,
        no_tools_attestation=attestation,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    precommit = _seal(
        {
            "schema": PRECOMMIT_SCHEMA,
            "command_id": COMMAND_ID,
            "authorization_digest": authorization["authorization_digest"],
            "batch_plan_digest": plan.record_digest,
            "accepted_calibration_parent": parent,
            "accepted_calibration_parent_digest": parent["parent_digest"],
            "calibration_result_digest": parent["calibration_result_digest"],
            "calibration_cold_replay_digest": parent["calibration_cold_replay_digest"],
            "calibration_historical_source_digest": parent[
                "calibration_historical_source_digest"
            ],
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "minutes": minutes,
            "verbose": False,
            "executable": executable,
            "cloud_policy_cache_base64": (
                None if cache.data is None else base64.b64encode(cache.data).decode("ascii")
            ),
            "model_catalog_base64": base64.b64encode(catalog.data).decode("ascii"),
            "launcher_fingerprint": fingerprint,
            "no_tools_attestation": attestation.to_dict(),
            "runtime_binding": runtime.binding,
            "transport_source_sha256": prototype_scene_transport_source_digest(),
            "snapshots_captured_once_for_entire_campaign": True,
            "authorization_and_precommit_fsynced_before_model_calls": True,
            "maximum_physical_model_calls": MAX_PHYSICAL_CALLS,
            "exact_query_observer_calls": QUERY_DENOMINATOR,
            **_authority_data(),
        },
        "precommit_digest",
    )
    return precommit, runtime


def _runtime_from_precommit(value: Mapping[str, Any]) -> ObjectBongardTurnRuntime:
    raw = _validate_seal(value, "precommit_digest", "execution precommit")
    if raw.get("schema") != PRECOMMIT_SCHEMA:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "execution precommit schema differs"
        )
    parent = _validate_accepted_calibration_parent(raw.get("accepted_calibration_parent"))
    if (
        raw.get("accepted_calibration_parent_digest") != parent["parent_digest"]
        or raw.get("calibration_result_digest") != parent["calibration_result_digest"]
        or raw.get("calibration_cold_replay_digest")
        != parent["calibration_cold_replay_digest"]
        or raw.get("calibration_historical_source_digest")
        != parent["calibration_historical_source_digest"]
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "precommit calibration binding differs"
        )
    try:
        cache_data = (
            None
            if raw["cloud_policy_cache_base64"] is None
            else base64.b64decode(raw["cloud_policy_cache_base64"], validate=True)
        )
        catalog_data = base64.b64decode(raw["model_catalog_base64"], validate=True)
        cache = CloudPolicyCacheSnapshot(cache_data)
        catalog = CodexModelCatalogSnapshot(catalog_data)
        attestation = CodexNoToolsAttestation.from_mapping(
            raw["no_tools_attestation"]
        )
        fingerprint = raw["launcher_fingerprint"]
        if (
            not isinstance(fingerprint, Mapping)
            or fingerprint.get("version") != PINNED_CODEX_CLI_VERSION
            or not _is_raw_digest(fingerprint.get("launcher_digest"))
        ):
            raise ValueError("launcher fingerprint differs")
        runtime = ObjectBongardTurnRuntime(
            model=raw["model"],
            reasoning_effort=raw["reasoning_effort"],
            minutes=raw["minutes"],
            verbose=raw["verbose"],
            executable=raw["executable"],
            cloud_policy_cache_snapshot=cache,
            model_catalog_snapshot=catalog,
            expected_launcher_digest=fingerprint["launcher_digest"],
            no_tools_attestation=attestation,
            transport_source_digest=raw["transport_source_sha256"],
        )
    except Exception as exc:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "execution runtime snapshot is malformed"
        ) from exc
    if runtime.binding != raw.get("runtime_binding"):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "execution runtime binding differs"
        )
    return runtime


def _snapshot_panel(
    task_root: Path,
    *,
    phase: str,
    panel_id: str,
    panel_reader: PanelReader,
) -> tuple[bytes, dict[str, Any]]:
    payload, receipt = panel_reader(panel_id)
    if not isinstance(payload, bytes) or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "panel source returned non-PNG bytes"
        )
    receipt_data = (
        receipt.to_data() if isinstance(receipt, OfficialPanelReceipt) else dict(receipt)
    )
    digest = hashlib.sha256(payload).hexdigest()
    record = _seal(
        {
            "schema": "gkm.bongard-shared-witness-campaign-panel-snapshot.v1",
            "phase": phase,
            "panel_id": panel_id,
            "png_sha256": digest,
            "exact_png_base64": base64.b64encode(payload).decode("ascii"),
            "source_receipt": receipt_data,
            "query_pixel": phase.startswith("query"),
            "opened_after_durable_rank_formula_freeze": phase.startswith("query"),
            **_authority_data(),
        },
        "record_digest",
    )
    panel_key = hashlib.sha256(panel_id.encode("utf-8")).hexdigest()
    target = task_root / "panels" / phase / f"{panel_key}_{digest}.json"
    _write_once(target, record)
    if _read_json(target, "panel snapshot") != record:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "panel snapshot reload differs"
        )
    return payload, record


class _PhysicalCallBudget:
    def __init__(self, limit: int, transport: NamedImageTransport) -> None:
        self.limit = limit
        self.transport = transport
        self._count = 0
        self._lock = Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def __call__(self, *args: Any, **kwargs: Any) -> CodexStructuredResult:
        with self._lock:
            if self._count >= self.limit:
                raise ObjectBongardSharedWitnessCampaignCommandError(
                    "physical model-call budget exhausted"
                )
            self._count += 1
        return self.transport(*args, **kwargs)


def _assert_one_fresh_call(
    journal: ObjectBongardNamedImageTurnJournalTransport,
) -> None:
    if journal.fresh_call_count != 1 or journal.reused_call_count != 0:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "production journal did not make exactly one fresh call"
        )


def _task_record(
    *,
    task_index: int,
    task: ObjectBongardTaskPlan,
    status: str,
    semantic: ObjectBongardSharedWitnessSemanticArtifact | None,
    archive: ObjectBongardSharedWitnessTaskRunArchive | None,
    panel_snapshots: Sequence[Mapping[str, Any]],
    journal_directories: Sequence[str],
    physical_call_count: int,
    exception_type: str | None = None,
) -> dict[str, Any]:
    correct = archive.correct_count if archive is not None and status == "complete" else 0
    query_calls = 0 if archive is None else archive.query_source_calls_made * 2
    return _seal(
        {
            "schema": TASK_RECORD_SCHEMA,
            "command_id": COMMAND_ID,
            "task_index": task_index,
            "task_id": task.task_id,
            "task_plan": task.to_data(),
            "status": status,
            "semantic_artifact": None if semantic is None else semantic.to_data(),
            "task_archive": None if archive is None else archive.to_data(),
            "panel_snapshots": [dict(item) for item in panel_snapshots],
            "journal_directories": list(journal_directories),
            "physical_call_count": physical_call_count,
            "correct_count": correct,
            "incorrect_count": 2 - correct,
            "score_denominator": 2,
            "query_observer_calls_made": query_calls,
            "query_pixels_opened": query_calls == 2,
            "query_pixels_opened_only_after_durable_rank_formula_freeze": query_calls == 2,
            "exception_type": exception_type,
            "exception_message_persisted": False,
            "noncomplete_task_is_typed_zero_of_two": status != "complete",
            **_authority_data(),
        },
        "record_digest",
    )


def _execute_task(
    *,
    root: Path,
    task_index: int,
    task: ObjectBongardTaskPlan,
    authorization_digest: str,
    precommit_digest: str,
    runtime: ObjectBongardTurnRuntime,
    panel_reader: PanelReader,
    transport: NamedImageTransport,
) -> dict[str, Any]:
    task_root = root / TASKS_DIRECTORY / f"{task_index:02d}_{task.task_id}"
    task_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    journal_root = root / JOURNALS_DIRECTORY / f"{task_index:02d}_{task.task_id}"
    panel_snapshots: list[dict[str, Any]] = []
    journal_directories: list[str] = []
    physical_calls = 0

    support_ids = (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
    support_png: dict[str, bytes] = {}
    for panel_id in support_ids:
        payload, snapshot = _snapshot_panel(
            task_root, phase="support", panel_id=panel_id, panel_reader=panel_reader
        )
        support_png[panel_id] = payload
        panel_snapshots.append(snapshot)

    semantic_images = tuple(
        (f"group_{group_index}_ref_{image_index:02d}.png", support_png[panel_id])
        for group_index, group in enumerate(
            (task.side_0_support_panel_ids, task.side_1_support_panel_ids)
        )
        for image_index, panel_id in enumerate(group)
    )
    semantic_directory = journal_root / "semantic"
    semantic_journal = ObjectBongardNamedImageTurnJournalTransport(
        semantic_directory,
        authorization_digest=authorization_digest,
        execution_precommit_digest=precommit_digest,
        task_id=task.task_id,
        turn_kind="structured_semantic_proposal",
        expected_prompt=object_bongard_shared_witness_semantics_prompt(),
        expected_images=semantic_images,
        expected_output_schema=object_bongard_shared_witness_semantics_output_schema(),
        runtime=runtime,
        underlying_transport=transport,
    )
    semantic = describe_object_bongard_shared_witness_support(
        task_id=task.task_id,
        group_0_panel_ids=task.side_0_support_panel_ids,
        group_1_panel_ids=task.side_1_support_panel_ids,
        support_png_by_panel_id=support_png,
        observation_context_digest=precommit_digest,
        **_runtime_kwargs(runtime),
        transport=semantic_journal,
    )
    _assert_one_fresh_call(semantic_journal)
    physical_calls += 1
    journal_directories.append(str(semantic_directory.relative_to(root)))
    _write_once(task_root / "semantic_artifact.json", semantic.to_data())
    if semantic.status is not PrototypeSceneObserverStatus.SUCCESS:
        record = _task_record(
            task_index=task_index,
            task=task,
            status="language_gap",
            semantic=semantic,
            archive=None,
            panel_snapshots=panel_snapshots,
            journal_directories=journal_directories,
            physical_call_count=physical_calls,
        )
        _write_once(task_root / "task_record.json", record)
        return record

    specs = build_shared_witness_rubric_specs(
        semantic, expected_artifact_digest=semantic.artifact_digest
    )
    side_0_by_rank: list[tuple[ObjectBongardSharedWitnessPanelArtifact, ...]] = []
    side_1_by_rank: list[tuple[ObjectBongardSharedWitnessPanelArtifact, ...]] = []
    for rank, spec in enumerate(specs):
        rank_blocks: list[tuple[ObjectBongardSharedWitnessPanelArtifact, ...]] = []
        for side, panel_ids in (
            ("side_0", task.side_0_support_panel_ids),
            ("side_1", task.side_1_support_panel_ids),
        ):
            artifacts: list[ObjectBongardSharedWitnessPanelArtifact] = []
            for panel_index, panel_id in enumerate(panel_ids):
                payload = support_png[panel_id]
                prepared = prepare_object_bongard_shared_witness_panel_inputs(
                    payload, spec
                )
                directory = (
                    journal_root / "support" / f"rank_{rank}" / side
                    / f"panel_{panel_index:02d}"
                )
                turn_kind = f"support_rank_{rank}_{side}_{panel_index:02d}"
                journal = ObjectBongardNamedImageTurnJournalTransport(
                    directory,
                    authorization_digest=authorization_digest,
                    execution_precommit_digest=precommit_digest,
                    task_id=task.task_id,
                    turn_kind=turn_kind,
                    expected_prompt=prepared.prompt,
                    expected_images=(("panel.png", payload),),
                    expected_output_schema=dict(prepared.output_schema),
                    runtime=runtime,
                    underlying_transport=transport,
                )
                artifact = observe_object_bongard_shared_witness_panel(
                    payload,
                    panel_id=panel_id,
                    rubric_spec=spec,
                    expected_panel_sha256=hashlib.sha256(payload).hexdigest(),
                    expected_rubric_spec_digest=spec.spec_digest,
                    observation_context_digest=precommit_digest,
                    **_runtime_kwargs(runtime),
                    transport=journal,
                )
                _assert_one_fresh_call(journal)
                physical_calls += 1
                journal_directories.append(str(directory.relative_to(root)))
                artifacts.append(artifact)
            rank_blocks.append(tuple(artifacts))
        side_0_by_rank.append(rank_blocks[0])
        side_1_by_rank.append(rank_blocks[1])
    if physical_calls != 1 + SUPPORT_CALLS_PER_TASK:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "task support call count differs"
        )

    freeze_path = task_root / "freeze.json"
    commit_path = task_root / "freeze_commit.json"

    def freeze_committer(payload: bytes) -> ObjectBongardSharedWitnessTaskFreezeCommit:
        freeze = ObjectBongardSharedWitnessTaskFreeze.from_data(json.loads(payload))
        _write_exact_once(freeze_path, payload)
        store_receipt = "sha256:" + canonical_digest(
            {
                "path": str(freeze_path.relative_to(root)),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
        commit = ObjectBongardSharedWitnessTaskFreezeCommit.seal(
            freeze, payload, task_freeze_store_receipt_digest=store_receipt
        )
        _write_once(commit_path, commit.to_data())
        return commit

    def freeze_reloader(commit_data: Mapping[str, Any]) -> bytes:
        commit = ObjectBongardSharedWitnessTaskFreezeCommit.from_data(commit_data)
        if _read_json(commit_path, "task freeze commit") != commit.to_data():
            raise ObjectBongardSharedWitnessCampaignCommandError(
                "task freeze commit reload differs"
            )
        payload = freeze_path.read_bytes()
        freeze = ObjectBongardSharedWitnessTaskFreeze.from_data(json.loads(payload))
        commit.assert_matches(freeze, payload)
        return payload

    def query_source(
        freeze_data: Mapping[str, Any], commit_data: Mapping[str, Any]
    ) -> Mapping[str, ObjectBongardSharedWitnessPanelArtifact]:
        nonlocal physical_calls
        freeze = ObjectBongardSharedWitnessTaskFreeze.from_data(freeze_data)
        commit = ObjectBongardSharedWitnessTaskFreezeCommit.from_data(commit_data)
        if not freeze_path.is_file() or not commit_path.is_file():
            raise ObjectBongardSharedWitnessCampaignCommandError(
                "query source preceded durable rank/formula freeze"
            )
        commit.assert_matches(freeze, freeze_path.read_bytes())
        selected_spec = freeze.selected_rubric_spec
        result: dict[str, ObjectBongardSharedWitnessPanelArtifact] = {}
        for side, panel_id in (
            ("side_0", task.side_0_query_panel_id),
            ("side_1", task.side_1_query_panel_id),
        ):
            payload, snapshot = _snapshot_panel(
                task_root,
                phase=f"query_{side}",
                panel_id=panel_id,
                panel_reader=panel_reader,
            )
            panel_snapshots.append(snapshot)
            prepared = prepare_object_bongard_shared_witness_panel_inputs(
                payload, selected_spec
            )
            directory = journal_root / "query" / side
            journal = ObjectBongardNamedImageTurnJournalTransport(
                directory,
                authorization_digest=authorization_digest,
                execution_precommit_digest=precommit_digest,
                task_id=task.task_id,
                turn_kind=f"selected_query_{side}",
                expected_prompt=prepared.prompt,
                expected_images=(("panel.png", payload),),
                expected_output_schema=dict(prepared.output_schema),
                runtime=runtime,
                underlying_transport=transport,
            )
            artifact = observe_object_bongard_shared_witness_panel(
                payload,
                panel_id=panel_id,
                rubric_spec=selected_spec,
                expected_panel_sha256=hashlib.sha256(payload).hexdigest(),
                expected_rubric_spec_digest=selected_spec.spec_digest,
                observation_context_digest=precommit_digest,
                **_runtime_kwargs(runtime),
                transport=journal,
            )
            _assert_one_fresh_call(journal)
            physical_calls += 1
            journal_directories.append(str(directory.relative_to(root)))
            result[side] = artifact
        return result

    archive = run_object_bongard_shared_witness_task(
        task,
        semantic,
        side_0_by_rank,
        side_1_by_rank,
        execution_precommit_digest=precommit_digest,
        freeze_committer=freeze_committer,
        freeze_reloader=freeze_reloader,
        query_source=query_source,
    )
    status = archive.status.value
    expected_calls = 27 if status == "complete" else 25
    if physical_calls != expected_calls:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "task physical call count differs"
        )
    record = _task_record(
        task_index=task_index,
        task=task,
        status=status,
        semantic=semantic,
        archive=archive,
        panel_snapshots=panel_snapshots,
        journal_directories=journal_directories,
        physical_call_count=physical_calls,
    )
    _write_once(task_root / "task_archive.json", archive.to_data())
    _write_once(task_root / "task_record.json", record)
    return record


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessCampaignCommandResult:
    output_root: Path
    campaign: Mapping[str, Any]
    replay: Mapping[str, Any]

    @property
    def correct_count(self) -> int:
        return int(self.campaign["correct_count"])

    @property
    def score_denominator(self) -> int:
        return int(self.campaign["score_denominator"])

    @property
    def physical_model_calls(self) -> int:
        return int(self.campaign["physical_model_calls"])

    @property
    def query_observer_calls(self) -> int:
        return int(self.campaign["query_observer_calls"])


def _campaign_record(
    *,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    plan: ObjectBongardBatchPlan,
    task_records: Sequence[Mapping[str, Any]],
    physical_model_calls: int,
) -> dict[str, Any]:
    parent = _validate_accepted_calibration_parent(
        authorization.get("accepted_calibration_parent")
    )
    if precommit.get("accepted_calibration_parent") != parent:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "campaign calibration parent differs across seals"
        )
    records = tuple(dict(item) for item in task_records)
    if len(records) != TASK_COUNT:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "campaign task record count differs"
        )
    correct = sum(int(item["correct_count"]) for item in records)
    query_calls = sum(int(item["query_observer_calls_made"]) for item in records)
    status_counts = {
        status: sum(item["status"] == status for item in records)
        for status in (
            "complete", "language_gap", "witness_gap", "error_gap", "task_exception"
        )
    }
    if (
        query_calls != QUERY_DENOMINATOR
        or status_counts["complete"] != TASK_COUNT
        or any(status_counts[name] != 0 for name in status_counts if name != "complete")
        or physical_model_calls != MAX_PHYSICAL_CALLS
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "successful campaign requires exactly 24 post-freeze query calls"
        )
    return _seal(
        {
            "schema": CAMPAIGN_SCHEMA,
            "command_id": COMMAND_ID,
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "batch_plan_digest": plan.record_digest,
            "accepted_calibration_parent_digest": parent["parent_digest"],
            "calibration_result_digest": parent["calibration_result_digest"],
            "calibration_cold_replay_digest": parent["calibration_cold_replay_digest"],
            "calibration_historical_source_digest": parent[
                "calibration_historical_source_digest"
            ],
            "task_records": list(records),
            "task_record_digests": [item["record_digest"] for item in records],
            "task_count": TASK_COUNT,
            "status_counts": status_counts,
            "correct_count": correct,
            "incorrect_count": QUERY_DENOMINATOR - correct,
            "score_denominator": QUERY_DENOMINATOR,
            "accuracy_ppm": correct * 1_000_000 // QUERY_DENOMINATOR,
            "query_observer_calls": query_calls,
            "exact_query_observer_calls_required": QUERY_DENOMINATOR,
            "physical_model_calls": physical_model_calls,
            "maximum_physical_model_calls": MAX_PHYSICAL_CALLS,
            "all_query_calls_follow_task_rank_formula_freeze": True,
            "all_task_archives_model_free_replayable": True,
            "official_test_authorized": False,
            **_authority_data(),
        },
        "campaign_digest",
    )


def _load_default_archive(
    descriptor_path: Path, archive_path: Path
) -> OfficialPanelArchive:
    descriptor = OfficialReleaseDescriptor.from_dict(
        _read_json(descriptor_path, "official release descriptor")
    )
    return OfficialPanelArchive.load(
        descriptor,
        archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )


def run_object_bongard_shared_witness_campaign_command(
    output_root: str | os.PathLike[str],
    *,
    calibration_root: str | os.PathLike[str] = DEFAULT_CALIBRATION_ROOT,
    preregistration_path: str | os.PathLike[str] = DEFAULT_PREREGISTRATION,
    plan_path: str | os.PathLike[str] = DEFAULT_PLAN,
    descriptor_path: str | os.PathLike[str] = DEFAULT_DESCRIPTOR,
    archive_path: str | os.PathLike[str] = DEFAULT_ARCHIVE,
    minutes: int = DEFAULT_MINUTES,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    executable: str = DEFAULT_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_EXPECTED_LAUNCHER_SHA256,
    cloud_policy_cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot] = snapshot_cloud_policy_cache,
    model_catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot] = snapshot_pinned_model_catalog,
    launcher_fingerprinter: Callable[..., Mapping[str, str]] = codex_cli_authenticated_fingerprint,
    runtime_attester: Callable[..., CodexNoToolsAttestation] = attest_codex_no_tools,
    underlying_transport: NamedImageTransport = run_codex_named_images_structured,
    panel_reader: PanelReader | None = None,
    archive_identity: Mapping[str, Any] | None = None,
) -> ObjectBongardSharedWitnessCampaignCommandResult:
    """Run the exact preregistered 12-task structured TRAIN campaign."""

    if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 120:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "minutes must lie in 1..120"
        )
    if (
        isinstance(parallel_workers, bool)
        or not isinstance(parallel_workers, int)
        or not 1 <= parallel_workers <= TASK_COUNT
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "parallel_workers must lie in 1..12"
        )
    if not _is_raw_digest(expected_launcher_sha256):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "launcher SHA-256 differs"
        )

    # First external-data gate.  Failure occurs before cohort files, archive
    # construction, output-root creation, panel callbacks, or model transport.
    accepted_parent = _cold_verify_accepted_calibration_parent(calibration_root)
    prereg, plan = _load_exact_cohort(Path(preregistration_path), Path(plan_path))
    if panel_reader is None:
        official_archive = _load_default_archive(
            Path(descriptor_path), Path(archive_path)
        )
        panel_reader = official_archive.read_panel
        archive_identity = {
            **official_archive.identity_data(),
            "record_digest": official_archive.record_digest,
        }
    elif archive_identity is None:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "an injected panel reader requires an archive identity"
        )
    assert archive_identity is not None
    root = _ensure_fresh_root(output_root)
    plan_record = _seal(
        {
            "schema": "gkm.bongard-shared-witness-campaign-plan-binding.v1",
            "preregistration": prereg,
            "batch_plan": plan.to_data(),
            "preregistration_file_sha256": PREREGISTRATION_FILE_SHA256,
            "batch_plan_file_sha256": PLAN_FILE_SHA256,
            "accepted_calibration_parent": accepted_parent,
            "exact_preregistered_train_tasks": True,
            "task_count": TASK_COUNT,
            **_authority_data(),
        },
        "record_digest",
    )
    _write_once(root / PLAN_FILENAME, plan_record)
    authorization = _authorization_record(
        plan=plan,
        preregistration=prereg,
        accepted_calibration_parent=accepted_parent,
        archive_identity=archive_identity,
        minutes=minutes,
        parallel_workers=parallel_workers,
        expected_launcher_sha256=expected_launcher_sha256,
    )
    _write_once(root / AUTHORIZATION_FILENAME, authorization)
    if _read_json(root / AUTHORIZATION_FILENAME, "authorization") != authorization:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "authorization reload differs"
        )
    precommit, _ = _prepare_runtime_precommit(
        authorization=authorization,
        plan=plan,
        accepted_calibration_parent=accepted_parent,
        minutes=minutes,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
        cloud_policy_cache_snapshotter=cloud_policy_cache_snapshotter,
        model_catalog_snapshotter=model_catalog_snapshotter,
        launcher_fingerprinter=launcher_fingerprinter,
        runtime_attester=runtime_attester,
    )
    _write_once(root / PRECOMMIT_FILENAME, precommit)
    runtime = _runtime_from_precommit(
        _read_json(root / PRECOMMIT_FILENAME, "execution precommit")
    )
    (root / TASKS_DIRECTORY).mkdir(mode=0o700)
    (root / JOURNALS_DIRECTORY).mkdir(mode=0o700)
    _fsync_directory(root)
    budget = _PhysicalCallBudget(MAX_PHYSICAL_CALLS, underlying_transport)

    def safe_execute(index_and_task: tuple[int, ObjectBongardTaskPlan]) -> dict[str, Any]:
        index, task = index_and_task
        try:
            return _execute_task(
                root=root,
                task_index=index,
                task=task,
                authorization_digest=authorization["authorization_digest"],
                precommit_digest=precommit["precommit_digest"],
                runtime=runtime,
                panel_reader=panel_reader,  # type: ignore[arg-type]
                transport=budget,
            )
        except Exception as exc:
            task_root = root / TASKS_DIRECTORY / f"{index:02d}_{task.task_id}"
            task_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            record_path = task_root / "task_record.json"
            if record_path.exists():
                return _read_json(record_path, "task record")
            snapshots = tuple(
                _read_json(path, "partial panel snapshot")
                for path in sorted((task_root / "panels").glob("*/*.json"))
            ) if (task_root / "panels").exists() else ()
            journal_root = root / JOURNALS_DIRECTORY / f"{index:02d}_{task.task_id}"
            directories = tuple(
                str(path.parent.relative_to(root))
                for path in sorted(journal_root.rglob("manifest.json"))
            ) if journal_root.exists() else ()
            semantic_path = task_root / "semantic_artifact.json"
            semantic = (
                ObjectBongardSharedWitnessSemanticArtifact.from_data(
                    _read_json(semantic_path, "partial semantic artifact")
                )
                if semantic_path.exists() else None
            )
            record = _task_record(
                task_index=index,
                task=task,
                status="task_exception",
                semantic=semantic,
                archive=None,
                panel_snapshots=snapshots,
                journal_directories=directories,
                physical_call_count=len(directories),
                exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            )
            _write_once(record_path, record)
            return record

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        task_records = tuple(executor.map(safe_execute, tuple(enumerate(plan.tasks))))
    campaign = _campaign_record(
        authorization=authorization,
        precommit=precommit,
        plan=plan,
        task_records=task_records,
        physical_model_calls=budget.count,
    )
    _write_once(root / CAMPAIGN_FILENAME, campaign)
    replay = _seal(
        {
            "schema": REPLAY_SCHEMA,
            "command_id": COMMAND_ID,
            "campaign_digest": campaign["campaign_digest"],
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "accepted_calibration_parent_digest": accepted_parent["parent_digest"],
            "calibration_result_digest": accepted_parent["calibration_result_digest"],
            "calibration_cold_replay_digest": accepted_parent[
                "calibration_cold_replay_digest"
            ],
            "calibration_historical_source_digest": accepted_parent[
                "calibration_historical_source_digest"
            ],
            "model_calls_during_replay": 0,
            "new_pixels_opened_during_replay": 0,
            "all_task_archives_cold_replayed": True,
            "exact_query_observer_calls": QUERY_DENOMINATOR,
            **_authority_data(),
        },
        "replay_digest",
    )
    _write_once(root / REPLAY_FILENAME, replay)
    return verify_object_bongard_shared_witness_campaign_command_directory(
        root, calibration_root=calibration_root
    )


def _decode_panel_snapshots(
    task_root: Path,
    values: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[bytes, dict[str, Any]]]:
    result: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for value in values:
        raw = _validate_seal(value, "record_digest", "panel snapshot")
        try:
            payload = base64.b64decode(raw["exact_png_base64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise ObjectBongardSharedWitnessCampaignCommandError(
                "panel snapshot bytes are malformed"
            ) from exc
        digest = hashlib.sha256(payload).hexdigest()
        phase = raw.get("phase")
        query = isinstance(phase, str) and phase.startswith("query")
        if (
            raw.get("schema")
            != "gkm.bongard-shared-witness-campaign-panel-snapshot.v1"
            or raw.get("png_sha256") != digest
            or not payload.startswith(b"\x89PNG\r\n\x1a\n")
            or raw.get("panel_id") in result
            or raw.get("query_pixel") is not query
            or raw.get("opened_after_durable_rank_formula_freeze") is not query
            or any(raw.get(key) != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardSharedWitnessCampaignCommandError(
                "panel snapshot identity differs"
            )
        panel_key = hashlib.sha256(raw["panel_id"].encode("utf-8")).hexdigest()
        path = task_root / "panels" / phase / f"{panel_key}_{digest}.json"
        if _read_json(path, "durable panel snapshot") != raw:
            raise ObjectBongardSharedWitnessCampaignCommandError(
                "durable panel snapshot differs"
            )
        result[raw["panel_id"]] = (payload, raw)
    return result


def _forbidden_transport(*_args: Any, **_kwargs: Any) -> CodexStructuredResult:
    raise AssertionError("cold replay attempted a model transport")


def _replay_semantic_journal(
    *,
    root: Path,
    task: ObjectBongardTaskPlan,
    semantic: ObjectBongardSharedWitnessSemanticArtifact,
    support_png: Mapping[str, bytes],
    runtime: ObjectBongardTurnRuntime,
    authorization_digest: str,
    precommit_digest: str,
    relative_directory: str,
) -> None:
    expected_images = tuple(
        (f"group_{group_index}_ref_{image_index:02d}.png", support_png[panel_id])
        for group_index, group in enumerate(
            (task.side_0_support_panel_ids, task.side_1_support_panel_ids)
        )
        for image_index, panel_id in enumerate(group)
    )
    journal = ObjectBongardNamedImageTurnJournalTransport(
        root / relative_directory,
        authorization_digest=authorization_digest,
        execution_precommit_digest=precommit_digest,
        task_id=task.task_id,
        turn_kind="structured_semantic_proposal",
        expected_prompt=object_bongard_shared_witness_semantics_prompt(),
        expected_images=expected_images,
        expected_output_schema=object_bongard_shared_witness_semantics_output_schema(),
        runtime=runtime,
        underlying_transport=_forbidden_transport,
    )
    replayed = describe_object_bongard_shared_witness_support(
        task_id=task.task_id,
        group_0_panel_ids=task.side_0_support_panel_ids,
        group_1_panel_ids=task.side_1_support_panel_ids,
        support_png_by_panel_id=support_png,
        observation_context_digest=precommit_digest,
        **_runtime_kwargs(runtime),
        transport=journal,
    )
    if (
        replayed != semantic
        or journal.fresh_call_count != 0
        or journal.reused_call_count != 1
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "semantic journal cold replay differs"
        )


def _replay_panel_journal(
    *,
    root: Path,
    task: ObjectBongardTaskPlan,
    artifact: ObjectBongardSharedWitnessPanelArtifact,
    payload: bytes,
    runtime: ObjectBongardTurnRuntime,
    authorization_digest: str,
    precommit_digest: str,
    relative_directory: str,
    turn_kind: str,
) -> None:
    prepared = prepare_object_bongard_shared_witness_panel_inputs(
        payload, artifact.rubric_spec
    )
    journal = ObjectBongardNamedImageTurnJournalTransport(
        root / relative_directory,
        authorization_digest=authorization_digest,
        execution_precommit_digest=precommit_digest,
        task_id=task.task_id,
        turn_kind=turn_kind,
        expected_prompt=prepared.prompt,
        expected_images=(("panel.png", payload),),
        expected_output_schema=dict(prepared.output_schema),
        runtime=runtime,
        underlying_transport=_forbidden_transport,
    )
    replayed = observe_object_bongard_shared_witness_panel(
        payload,
        panel_id=artifact.panel_id,
        rubric_spec=artifact.rubric_spec,
        expected_panel_sha256=hashlib.sha256(payload).hexdigest(),
        expected_rubric_spec_digest=artifact.rubric_spec_digest,
        observation_context_digest=precommit_digest,
        **_runtime_kwargs(runtime),
        transport=journal,
    )
    if (
        replayed != artifact
        or journal.fresh_call_count != 0
        or journal.reused_call_count != 1
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "panel journal cold replay differs"
        )


def _cold_replay_task_record(
    *,
    root: Path,
    task_index: int,
    task: ObjectBongardTaskPlan,
    record: Mapping[str, Any],
    runtime: ObjectBongardTurnRuntime,
    authorization_digest: str,
    precommit_digest: str,
) -> dict[str, Any]:
    raw = _validate_seal(record, "record_digest", "task record")
    if (
        raw.get("schema") != TASK_RECORD_SCHEMA
        or raw.get("task_index") != task_index
        or raw.get("task_id") != task.task_id
        or raw.get("task_plan") != task.to_data()
        or raw.get("status") != "complete"
        or raw.get("score_denominator") != 2
        or raw.get("correct_count") not in (0, 1, 2)
        or raw.get("incorrect_count") != 2 - raw["correct_count"]
        or raw.get("query_observer_calls_made") != 2
        or raw.get("query_pixels_opened") is not True
        or raw.get("query_pixels_opened_only_after_durable_rank_formula_freeze") is not True
        or any(raw.get(key) != item for key, item in _authority_data().items())
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "task record scoring or identity differs"
        )
    task_root = root / TASKS_DIRECTORY / f"{task_index:02d}_{task.task_id}"
    if _read_json(task_root / "task_record.json", "task record") != raw:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "durable task record differs"
        )
    snapshots = _decode_panel_snapshots(task_root, raw["panel_snapshots"])
    support_ids = (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
    query_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    if set(snapshots) != set((*support_ids, *query_ids)):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "task snapshot inventory differs"
        )
    support_png = {panel_id: snapshots[panel_id][0] for panel_id in support_ids}
    semantic = ObjectBongardSharedWitnessSemanticArtifact.from_data(
        raw["semantic_artifact"]
    )
    verify_object_bongard_shared_witness_semantic_artifact(
        semantic,
        support_png_by_panel_id=support_png,
        expected_task_id=task.task_id,
        expected_observation_context_digest=precommit_digest,
        expected_artifact_digest=semantic.artifact_digest,
    )
    directories = tuple(raw["journal_directories"])
    expected_semantic_directory = str(
        Path(JOURNALS_DIRECTORY) / f"{task_index:02d}_{task.task_id}" / "semantic"
    )
    if not directories or directories[0] != expected_semantic_directory:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "semantic journal inventory differs"
        )
    _replay_semantic_journal(
        root=root,
        task=task,
        semantic=semantic,
        support_png=support_png,
        runtime=runtime,
        authorization_digest=authorization_digest,
        precommit_digest=precommit_digest,
        relative_directory=directories[0],
    )
    archive = ObjectBongardSharedWitnessTaskRunArchive.from_data(raw["task_archive"])
    cold_replay_object_bongard_shared_witness_task(
        archive, expected_archive_digest=archive.record_digest
    )
    directory_index = 1
    for rank in (0, 1):
        for side, panel_ids, block in (
            ("side_0", task.side_0_support_panel_ids, archive.side_0_support_by_rank[rank]),
            ("side_1", task.side_1_support_panel_ids, archive.side_1_support_by_rank[rank]),
        ):
            for panel_index, (panel_id, artifact) in enumerate(
                zip(panel_ids, block, strict=True)
            ):
                expected_directory = str(
                    Path(JOURNALS_DIRECTORY)
                    / f"{task_index:02d}_{task.task_id}"
                    / "support" / f"rank_{rank}" / side
                    / f"panel_{panel_index:02d}"
                )
                if directories[directory_index] != expected_directory:
                    raise ObjectBongardSharedWitnessCampaignCommandError(
                        "support journal order differs"
                    )
                payload = snapshots[panel_id][0]
                verify_object_bongard_shared_witness_panel_artifact(
                    artifact,
                    payload,
                    panel_id=panel_id,
                    rubric_spec=artifact.rubric_spec,
                    expected_artifact_digest=artifact.artifact_digest,
                    expected_runtime_identity_digest=(
                        archive.version_spaces[0].observer_runtime_identity_digest
                    ),
                )
                _replay_panel_journal(
                    root=root,
                    task=task,
                    artifact=artifact,
                    payload=payload,
                    runtime=runtime,
                    authorization_digest=authorization_digest,
                    precommit_digest=precommit_digest,
                    relative_directory=expected_directory,
                    turn_kind=f"support_rank_{rank}_{side}_{panel_index:02d}",
                )
                directory_index += 1
    if (
        archive.status is not ObjectBongardSharedWitnessTaskRunStatus.COMPLETE
        or archive.side_0_query is None
        or archive.side_1_query is None
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "completed campaign task lacks query archive"
        )
    for side, artifact in (
        ("side_0", archive.side_0_query),
        ("side_1", archive.side_1_query),
    ):
        directory = str(
            Path(JOURNALS_DIRECTORY)
            / f"{task_index:02d}_{task.task_id}" / "query" / side
        )
        if directories[directory_index] != directory:
            raise ObjectBongardSharedWitnessCampaignCommandError(
                "query journal order differs"
            )
        payload = snapshots[artifact.panel_id][0]
        verify_object_bongard_shared_witness_panel_artifact(
            artifact,
            payload,
            panel_id=artifact.panel_id,
            rubric_spec=artifact.rubric_spec,
            expected_artifact_digest=artifact.artifact_digest,
            expected_runtime_identity_digest=(
                archive.version_spaces[0].observer_runtime_identity_digest
            ),
        )
        _replay_panel_journal(
            root=root,
            task=task,
            artifact=artifact,
            payload=payload,
            runtime=runtime,
            authorization_digest=authorization_digest,
            precommit_digest=precommit_digest,
            relative_directory=directory,
            turn_kind=f"selected_query_{side}",
        )
        directory_index += 1
    freeze_payload = (task_root / "freeze.json").read_bytes()
    freeze = ObjectBongardSharedWitnessTaskFreeze.from_data(json.loads(freeze_payload))
    commit = ObjectBongardSharedWitnessTaskFreezeCommit.from_data(
        _read_json(task_root / "freeze_commit.json", "task freeze commit")
    )
    commit.assert_matches(freeze, freeze_payload)
    if directory_index != len(directories) or len(directories) != 27:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "task journal inventory differs"
        )
    return raw


def verify_object_bongard_shared_witness_campaign_command_directory(
    output_root: str | os.PathLike[str],
    *,
    calibration_root: str | os.PathLike[str] = DEFAULT_CALIBRATION_ROOT,
) -> ObjectBongardSharedWitnessCampaignCommandResult:
    """Cold replay a completed campaign with model transport forbidden."""

    accepted_parent = _cold_verify_accepted_calibration_parent(calibration_root)
    root = Path(output_root).absolute()
    expected_root = {
        PLAN_FILENAME, AUTHORIZATION_FILENAME, PRECOMMIT_FILENAME,
        CAMPAIGN_FILENAME, REPLAY_FILENAME, TASKS_DIRECTORY, JOURNALS_DIRECTORY,
    }
    if not root.is_dir() or {item.name for item in root.iterdir()} != expected_root:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "campaign root inventory differs"
        )
    plan_record = _validate_seal(
        _read_json(root / PLAN_FILENAME, "campaign plan"),
        "record_digest",
        "campaign plan",
    )
    plan = ObjectBongardBatchPlan.from_data(plan_record["batch_plan"])
    prereg = plan_record["preregistration"]
    if (
        plan.record_digest != PLAN_DIGEST
        or prereg.get("record_digest") != PREREGISTRATION_DIGEST
        or plan_record.get("preregistration_file_sha256") != PREREGISTRATION_FILE_SHA256
        or plan_record.get("batch_plan_file_sha256") != PLAN_FILE_SHA256
        or plan_record.get("accepted_calibration_parent") != accepted_parent
        or len(plan.tasks) != TASK_COUNT
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "campaign cohort replay differs"
        )
    authorization = _validate_seal(
        _read_json(root / AUTHORIZATION_FILENAME, "authorization"),
        "authorization_digest",
        "authorization",
    )
    expected_authorization = _authorization_record(
        plan=plan,
        preregistration=prereg,
        accepted_calibration_parent=accepted_parent,
        archive_identity=authorization["archive_identity"],
        minutes=authorization["minutes"],
        parallel_workers=authorization["parallel_workers"],
        expected_launcher_sha256=authorization["expected_launcher_sha256"],
    )
    if authorization != expected_authorization:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "authorization replay differs"
        )
    precommit = _validate_seal(
        _read_json(root / PRECOMMIT_FILENAME, "execution precommit"),
        "precommit_digest",
        "execution precommit",
    )
    if (
        precommit.get("authorization_digest") != authorization["authorization_digest"]
        or precommit.get("batch_plan_digest") != plan.record_digest
        or precommit.get("accepted_calibration_parent") != accepted_parent
        or precommit.get("accepted_calibration_parent_digest") != accepted_parent["parent_digest"]
    ):
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "precommit parents differ"
        )
    runtime = _runtime_from_precommit(precommit)
    campaign = _validate_seal(
        _read_json(root / CAMPAIGN_FILENAME, "campaign result"),
        "campaign_digest",
        "campaign result",
    )
    if not isinstance(campaign.get("task_records"), list) or len(
        campaign["task_records"]
    ) != TASK_COUNT:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "campaign task inventory differs"
        )
    replayed_records = tuple(
        _cold_replay_task_record(
            root=root,
            task_index=index,
            task=task,
            record=campaign["task_records"][index],
            runtime=runtime,
            authorization_digest=authorization["authorization_digest"],
            precommit_digest=precommit["precommit_digest"],
        )
        for index, task in enumerate(plan.tasks)
    )
    expected_campaign = _campaign_record(
        authorization=authorization,
        precommit=precommit,
        plan=plan,
        task_records=replayed_records,
        physical_model_calls=campaign["physical_model_calls"],
    )
    if campaign != expected_campaign:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "campaign aggregate replay differs"
        )
    expected_manifests = {
        str((root / directory / "manifest.json").relative_to(root))
        for record in replayed_records
        for directory in record["journal_directories"]
    }
    actual_manifests = {
        str(path.relative_to(root))
        for path in (root / JOURNALS_DIRECTORY).rglob("manifest.json")
    }
    if actual_manifests != expected_manifests:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "physical journal tree differs"
        )
    replay = _validate_seal(
        _read_json(root / REPLAY_FILENAME, "campaign cold replay"),
        "replay_digest",
        "campaign cold replay",
    )
    expected_replay = _seal(
        {
            "schema": REPLAY_SCHEMA,
            "command_id": COMMAND_ID,
            "campaign_digest": campaign["campaign_digest"],
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "accepted_calibration_parent_digest": accepted_parent["parent_digest"],
            "calibration_result_digest": accepted_parent["calibration_result_digest"],
            "calibration_cold_replay_digest": accepted_parent[
                "calibration_cold_replay_digest"
            ],
            "calibration_historical_source_digest": accepted_parent[
                "calibration_historical_source_digest"
            ],
            "model_calls_during_replay": 0,
            "new_pixels_opened_during_replay": 0,
            "all_task_archives_cold_replayed": True,
            "exact_query_observer_calls": QUERY_DENOMINATOR,
            **_authority_data(),
        },
        "replay_digest",
    )
    if replay != expected_replay:
        raise ObjectBongardSharedWitnessCampaignCommandError(
            "campaign replay record differs"
        )
    return ObjectBongardSharedWitnessCampaignCommandResult(root, campaign, replay)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    launch = commands.add_parser("launch")
    verify = commands.add_parser("verify")
    for command in (launch, verify):
        command.add_argument("--output-root", required=True, type=Path)
        command.add_argument(
            "--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT
        )
    launch.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
    launch.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    launch.add_argument("--executable", default=DEFAULT_EXECUTABLE)
    launch.add_argument(
        "--expected-launcher-sha256", default=DEFAULT_EXPECTED_LAUNCHER_SHA256
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        if args.operation == "launch":
            result = run_object_bongard_shared_witness_campaign_command(
                args.output_root,
                calibration_root=args.calibration_root,
                parallel_workers=args.parallel_workers,
                minutes=args.minutes,
                executable=args.executable,
                expected_launcher_sha256=args.expected_launcher_sha256,
            )
        else:
            result = verify_object_bongard_shared_witness_campaign_command_directory(
                args.output_root, calibration_root=args.calibration_root
            )
    except Exception as exc:
        print(
            canonical_json(
                {
                    "schema": "gkm.bongard-shared-witness-campaign-command-error.v1",
                    "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "raw_message_persisted": False,
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 1
    print(
        canonical_json(
            {
                "schema": "gkm.bongard-shared-witness-campaign-summary.v1",
                "output_root": str(result.output_root),
                "campaign_digest": result.campaign["campaign_digest"],
                "replay_digest": result.replay["replay_digest"],
                "correct_count": result.correct_count,
                "score_denominator": result.score_denominator,
                "query_observer_calls": result.query_observer_calls,
                **_authority_data(),
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "DEFAULT_CALIBRATION_ROOT",
    "MAX_PHYSICAL_CALLS",
    "ObjectBongardSharedWitnessCampaignCommandError",
    "ObjectBongardSharedWitnessCampaignCommandResult",
    "QUERY_DENOMINATOR",
    "run_object_bongard_shared_witness_campaign_command",
    "verify_object_bongard_shared_witness_campaign_command_directory",
    "object_bongard_shared_witness_campaign_command_source_digest",
    "main",
)
