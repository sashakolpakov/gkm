"""Sealed production command for the preregistered panel-rubric campaign.

The command first cold-verifies an explicitly supplied accepted calibration
parent, then opens only the twelve exact-unused TRAIN tasks committed by the
checked-in preregistration.  For each task it obtains one two-rank semantic
proposal, judges both ranks on the twelve support panels, and delegates all
selection, freeze-before-query, and scoring decisions to
``object_bongard_panel_rubric_task_runner``.  Python is the sole predicate
authority.  Lean is absent, removable, and decision-inert.
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
from bongard.object_bongard_panel_rubric_calibration_command import (
    CALIBRATION_RESULT_SCHEMA,
    RESULT_FILENAME as CALIBRATION_RESULT_FILENAME,
    VerifiedObjectBongardPanelRubricCalibration,
    object_bongard_panel_rubric_calibration_command_source_digest,
    verify_object_bongard_panel_rubric_calibration,
)
from bongard.object_bongard_panel_rubric_observer import (
    ObjectBongardPanelRubricArtifact,
    object_bongard_panel_rubric_output_schema,
    object_bongard_panel_rubric_prompt,
    observe_object_bongard_panel_rubric,
    verify_object_bongard_panel_rubric_artifact,
)
from bongard.object_bongard_panel_rubric_task_runner import (
    ObjectBongardPanelRubricTaskFreeze,
    ObjectBongardPanelRubricTaskFreezeCommit,
    ObjectBongardPanelRubricTaskRunArchive,
    ObjectBongardPanelRubricTaskRunStatus,
    cold_replay_object_bongard_panel_rubric_task,
    run_object_bongard_panel_rubric_task,
)
from bongard.object_bongard_rubric_language import ObjectBongardRubricSpec
from bongard.object_bongard_semantics import (
    ObjectBongardSemanticArtifact,
    describe_object_bongard_support,
    object_bongard_semantics_output_schema,
    object_bongard_semantics_prompt,
    verify_object_bongard_semantic_artifact,
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


COMMAND_ID = "bongard.panel-rubric-campaign-command/exact-train-12-v2"
AUTHORIZATION_SCHEMA = "gkm.bongard-panel-rubric-campaign-authorization.v2"
PRECOMMIT_SCHEMA = "gkm.bongard-panel-rubric-campaign-precommit.v2"
TASK_RECORD_SCHEMA = "gkm.bongard-panel-rubric-campaign-task-record.v1"
CAMPAIGN_SCHEMA = "gkm.bongard-panel-rubric-campaign-result.v2"
REPLAY_SCHEMA = "gkm.bongard-panel-rubric-campaign-replay.v2"
CALIBRATION_PARENT_SCHEMA = (
    "gkm.bongard-panel-rubric-campaign-accepted-calibration-parent.v1"
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
QUERY_DENOMINATOR = 24
MAX_PHYSICAL_CALLS = TASK_COUNT * (1 + SUPPORT_CALLS_PER_TASK + 2)

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREREGISTRATION = _REPOSITORY_ROOT / "bongard/data/object_bongard_rubric_train_20260808.prereg.json"
DEFAULT_PLAN = _REPOSITORY_ROOT / "bongard/data/object_bongard_rubric_train_20260808.plan.json"
DEFAULT_DESCRIPTOR = _REPOSITORY_ROOT / "bongard/data/shape_bongard_v2_release_v1.json"
DEFAULT_ARCHIVE = _REPOSITORY_ROOT / "downloads/ShapeBongard_V2.zip"
DEFAULT_CALIBRATION_NOMINATION_ROOT = (
    _REPOSITORY_ROOT
    / "downloads/ShapeBongard_V2_full/object_rubric_nomination_20260808_all_support_v10"
)
DEFAULT_CALIBRATION_SOURCE_DIRECTORY = (
    _REPOSITORY_ROOT
    / "downloads/ShapeBongard_V2_full/"
    "prototype_pair_python_campaign_20260807_object_v1/objects"
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

PanelReader = Callable[[str], tuple[bytes, OfficialPanelReceipt]]
NamedImageTransport = Callable[..., CodexStructuredResult]


class ObjectBongardPanelRubricCampaignCommandError(RuntimeError):
    """The launch boundary, execution, or cold replay failed closed."""


def _is_raw_digest(value: object) -> bool:
    return isinstance(value, str) and _RAW_DIGEST.fullmatch(value) is not None


def _is_address(value: object) -> bool:
    return isinstance(value, str) and _ADDRESS.fullmatch(value) is not None


def object_bongard_panel_rubric_campaign_command_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
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


def _seal(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(body)
    result[field] = "sha256:" + canonical_digest(result)
    return result


def _validate_seal(value: Mapping[str, Any], field: str, label: str) -> dict[str, Any]:
    raw = dict(value)
    expected = "sha256:" + canonical_digest({k: v for k, v in raw.items() if k != field})
    if raw.get(field) != expected:
        raise ObjectBongardPanelRubricCampaignCommandError(f"{label} digest differs")
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
        raise ObjectBongardPanelRubricCampaignCommandError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or not 0 < len(payload) <= _MAX_RECORD_BYTES:
        raise ObjectBongardPanelRubricCampaignCommandError(f"{label} is not a bounded regular file")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardPanelRubricCampaignCommandError(f"{label} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ObjectBongardPanelRubricCampaignCommandError(f"{label} must be a JSON object")
    return value


def _read_exact_input(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ObjectBongardPanelRubricCampaignCommandError(f"{label} file identity differs")
    return _read_json(path, label)


def _load_exact_cohort(preregistration_path: Path, plan_path: Path) -> tuple[dict[str, Any], ObjectBongardBatchPlan]:
    prereg = _read_exact_input(preregistration_path, PREREGISTRATION_FILE_SHA256, "preregistration")
    plan_raw = _read_exact_input(plan_path, PLAN_FILE_SHA256, "batch plan")
    plan = ObjectBongardBatchPlan.from_data(plan_raw)
    body = {key: item for key, item in prereg.items() if key != "record_digest"}
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
        or tuple(task.family for task in plan.tasks).count("bd") != 4
        or tuple(task.family for task in plan.tasks).count("ff") != 4
        or tuple(task.family for task in plan.tasks).count("hd") != 4
    ):
        raise ObjectBongardPanelRubricCampaignCommandError("preregistered TRAIN cohort differs")
    return prereg, plan


def _validate_accepted_calibration_parent(value: object) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "calibration_verifier_source_sha256",
        "calibration_result_digest",
        "calibration_cold_replay_digest",
        "calibration_source_digest",
        "calibration_plan_digest",
        "calibration_assessment_digest",
        "selected_candidate_rank",
        "selected_candidate_digest",
        "fresh_call_count",
        "reused_call_count",
        "accepted",
        "cold_verified_before_campaign_archive_access",
        "parent_digest",
    }
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected_fields
    ):
        raise ObjectBongardPanelRubricCampaignCommandError(
            "accepted calibration parent fields differ"
        )
    raw = _validate_seal(value, "parent_digest", "accepted calibration parent")
    if (
        raw["schema"] != CALIBRATION_PARENT_SCHEMA
        or not _is_raw_digest(raw["calibration_verifier_source_sha256"])
        or not _is_address(raw["calibration_result_digest"])
        or not _is_address(raw["calibration_cold_replay_digest"])
        or not _is_raw_digest(raw["calibration_source_digest"])
        or not _is_raw_digest(raw["calibration_plan_digest"])
        or not _is_raw_digest(raw["calibration_assessment_digest"])
        or type(raw["selected_candidate_rank"]) is not int
        or raw["selected_candidate_rank"] not in (0, 1)
        or not _is_raw_digest(raw["selected_candidate_digest"])
        or raw["fresh_call_count"] != 24
        or raw["reused_call_count"] != 0
        or raw["accepted"] is not True
        or raw["cold_verified_before_campaign_archive_access"] is not True
    ):
        raise ObjectBongardPanelRubricCampaignCommandError(
            "accepted calibration parent policy differs"
        )
    return raw


def _cold_verify_accepted_calibration_parent(
    calibration_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Require a successful calibration before any fresh campaign access."""

    candidate = Path(calibration_root).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ObjectBongardPanelRubricCampaignCommandError(
            "accepted calibration root is unavailable or linked"
        )
    root = candidate.resolve(strict=True)
    verified = verify_object_bongard_panel_rubric_calibration(
        root,
        nomination_root=DEFAULT_CALIBRATION_NOMINATION_ROOT,
        source_directory=DEFAULT_CALIBRATION_SOURCE_DIRECTORY,
    )
    if not isinstance(verified, VerifiedObjectBongardPanelRubricCalibration):
        raise ObjectBongardPanelRubricCampaignCommandError(
            "calibration verifier returned the wrong type"
        )
    result = _validate_seal(
        _read_json(root / CALIBRATION_RESULT_FILENAME, "calibration result"),
        "record_digest",
        "calibration result",
    )
    selected_digest = result.get("selected_candidate_digest")
    if (
        verified.output_root != root
        or verified.accepted is not True
        or verified.selected_candidate_rank not in (0, 1)
        or verified.fresh_call_count != 24
        or verified.reused_call_count != 0
        or result.get("schema") != CALIBRATION_RESULT_SCHEMA
        or result.get("record_digest") != verified.result_digest
        or result.get("cold_replay_digest") != verified.replay_digest
        or result.get("plan_digest") != verified.plan_digest
        or result.get("assessment_digest") != verified.assessment_digest
        or result.get("accepted") is not True
        or result.get("selected_candidate_rank")
        != verified.selected_candidate_rank
        or not isinstance(selected_digest, str)
        or _RAW_DIGEST.fullmatch(selected_digest) is None
        or result.get("fresh_call_count") != 24
        or result.get("reused_call_count") != 0
        or result.get("physical_call_denominator") != 24
        or result.get("all_24_artifacts_frozen_before_support_labels") is not True
        or result.get("model_calls_during_assessment_or_replay") != 0
        or result.get("query_pixels_opened") is not False
        or result.get("broad_cohort_pixels_opened") is not False
        or result.get("official_test_pixels_opened") is not False
        or not isinstance(result.get("source_digest"), str)
        or _RAW_DIGEST.fullmatch(result["source_digest"]) is None
    ):
        raise ObjectBongardPanelRubricCampaignCommandError(
            "calibration is not an accepted cold-verified campaign parent"
        )
    return _validate_accepted_calibration_parent(
        _seal(
            {
                "schema": CALIBRATION_PARENT_SCHEMA,
                "calibration_verifier_source_sha256": (
                    object_bongard_panel_rubric_calibration_command_source_digest()
                ),
                "calibration_result_digest": verified.result_digest,
                "calibration_cold_replay_digest": verified.replay_digest,
                "calibration_source_digest": result["source_digest"],
                "calibration_plan_digest": verified.plan_digest,
                "calibration_assessment_digest": verified.assessment_digest,
                "selected_candidate_rank": verified.selected_candidate_rank,
                "selected_candidate_digest": selected_digest,
                "fresh_call_count": verified.fresh_call_count,
                "reused_call_count": verified.reused_call_count,
                "accepted": True,
                "cold_verified_before_campaign_archive_access": True,
            },
            "parent_digest",
        )
    )


def _ensure_fresh_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value).absolute()
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ObjectBongardPanelRubricCampaignCommandError(
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
    calibration_parent = _validate_accepted_calibration_parent(
        accepted_calibration_parent
    )
    jobs = [
        {
            "task_index": index,
            "task_id": task.task_id,
            "task_plan_digest": task.record_digest,
            "semantic_calls": 1,
            "support_calls": SUPPORT_CALLS_PER_TASK,
            "maximum_query_calls": 2,
            "sealed_query_panel_ids": [
                task.side_0_query_panel_id,
                task.side_1_query_panel_id,
            ],
        }
        for index, task in enumerate(plan.tasks)
    ]
    return _seal(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "command_id": COMMAND_ID,
            "command_source_sha256": object_bongard_panel_rubric_campaign_command_source_digest(),
            "preregistration_digest": preregistration["record_digest"],
            "batch_plan_digest": plan.record_digest,
            "selected_task_ids_digest": preregistration["selected_task_ids_digest"],
            "sealed_query_panel_ids_digest": preregistration["sealed_query_panel_ids_digest"],
            "scope": preregistration["scope"],
            "accepted_calibration_parent": calibration_parent,
            "calibration_result_digest": calibration_parent[
                "calibration_result_digest"
            ],
            "calibration_cold_replay_digest": calibration_parent[
                "calibration_cold_replay_digest"
            ],
            "calibration_source_digest": calibration_parent[
                "calibration_source_digest"
            ],
            "archive_identity": dict(archive_identity),
            "jobs": jobs,
            "task_count": TASK_COUNT,
            "semantic_calls_per_task": 1,
            "support_calls_per_task": SUPPORT_CALLS_PER_TASK,
            "query_calls_only_after_survivor_and_durable_freeze": True,
            "maximum_physical_model_calls": MAX_PHYSICAL_CALLS,
            "fixed_campaign_score_denominator": QUERY_DENOMINATOR,
            "complete_task_score_denominator": 2,
            "gap_or_task_exception_score": "0/2",
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "minutes": minutes,
            "parallel_workers": parallel_workers,
            "expected_launcher_sha256": expected_launcher_sha256,
            "fresh_journals_required": True,
            "support_and_query_journals_disjoint": True,
            "official_test_authorized": False,
            "query_pixels_opened_before_task_freeze": False,
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
    calibration_parent = _validate_accepted_calibration_parent(
        accepted_calibration_parent
    )
    if authorization.get("accepted_calibration_parent") != calibration_parent:
        raise ObjectBongardPanelRubricCampaignCommandError(
            "authorization calibration parent differs"
        )
    cache = cloud_policy_cache_snapshotter()
    catalog = model_catalog_snapshotter()
    if not isinstance(cache, CloudPolicyCacheSnapshot) or not isinstance(catalog, CodexModelCatalogSnapshot):
        raise ObjectBongardPanelRubricCampaignCommandError("runtime snapshot type differs")
    fingerprint = dict(
        launcher_fingerprinter(
            executable, expected_launcher_digest=expected_launcher_sha256
        )
    )
    if fingerprint != {
        "version": PINNED_CODEX_CLI_VERSION,
        "launcher_digest": expected_launcher_sha256,
    }:
        raise ObjectBongardPanelRubricCampaignCommandError("authenticated launcher fingerprint differs")
    attestation = runtime_attester(
        executable=executable,
        expected_launcher_digest=expected_launcher_sha256,
        model_catalog_snapshot=catalog,
        cloud_policy_cache_snapshot=cache,
    )
    if not isinstance(attestation, CodexNoToolsAttestation):
        raise ObjectBongardPanelRubricCampaignCommandError("no-tools attestation type differs")
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
            "accepted_calibration_parent": calibration_parent,
            "calibration_result_digest": calibration_parent[
                "calibration_result_digest"
            ],
            "calibration_cold_replay_digest": calibration_parent[
                "calibration_cold_replay_digest"
            ],
            "calibration_source_digest": calibration_parent[
                "calibration_source_digest"
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
            **_authority_data(),
        },
        "precommit_digest",
    )
    return precommit, runtime


def _runtime_from_precommit(value: Mapping[str, Any]) -> ObjectBongardTurnRuntime:
    raw = _validate_seal(value, "precommit_digest", "execution precommit")
    if raw.get("schema") != PRECOMMIT_SCHEMA:
        raise ObjectBongardPanelRubricCampaignCommandError("execution precommit schema differs")
    calibration_parent = _validate_accepted_calibration_parent(
        raw.get("accepted_calibration_parent")
    )
    if (
        raw.get("calibration_result_digest")
        != calibration_parent["calibration_result_digest"]
        or raw.get("calibration_cold_replay_digest")
        != calibration_parent["calibration_cold_replay_digest"]
        or raw.get("calibration_source_digest")
        != calibration_parent["calibration_source_digest"]
    ):
        raise ObjectBongardPanelRubricCampaignCommandError(
            "execution precommit calibration binding differs"
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
        attestation = CodexNoToolsAttestation.from_mapping(raw["no_tools_attestation"])
        runtime = ObjectBongardTurnRuntime(
            model=raw["model"],
            reasoning_effort=raw["reasoning_effort"],
            minutes=raw["minutes"],
            verbose=raw["verbose"],
            executable=raw["executable"],
            cloud_policy_cache_snapshot=cache,
            model_catalog_snapshot=catalog,
            expected_launcher_digest=raw["launcher_fingerprint"]["launcher_digest"],
            no_tools_attestation=attestation,
            transport_source_digest=raw["transport_source_sha256"],
        )
    except Exception as exc:
        raise ObjectBongardPanelRubricCampaignCommandError("execution precommit cannot be reconstructed") from exc
    if runtime.binding != raw.get("runtime_binding"):
        raise ObjectBongardPanelRubricCampaignCommandError("execution runtime binding differs")
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
        raise ObjectBongardPanelRubricCampaignCommandError("panel source returned non-PNG bytes")
    receipt_data = receipt.to_data() if isinstance(receipt, OfficialPanelReceipt) else dict(receipt)
    digest = hashlib.sha256(payload).hexdigest()
    record = _seal(
        {
            "schema": "gkm.bongard-panel-rubric-campaign-panel-snapshot.v1",
            "phase": phase,
            "panel_id": panel_id,
            "png_sha256": digest,
            "exact_png_base64": base64.b64encode(payload).decode("ascii"),
            "source_receipt": receipt_data,
            "query_pixel": phase.startswith("query"),
            "opened_after_durable_task_freeze": phase.startswith("query"),
            **_authority_data(),
        },
        "record_digest",
    )
    panel_key = hashlib.sha256(panel_id.encode("utf-8")).hexdigest()
    target = task_root / "panels" / phase / f"{panel_key}_{digest}.json"
    _write_once(target, record)
    restored = _read_json(target, "panel snapshot")
    if restored != record:
        raise ObjectBongardPanelRubricCampaignCommandError("panel snapshot reload differs")
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
                raise ObjectBongardPanelRubricCampaignCommandError("physical model-call budget exhausted")
            self._count += 1
        return self.transport(*args, **kwargs)


def _assert_one_fresh_call(
    journal: ObjectBongardNamedImageTurnJournalTransport,
) -> None:
    if journal.fresh_call_count != 1 or journal.reused_call_count != 0:
        raise ObjectBongardPanelRubricCampaignCommandError(
            "production journal did not make exactly one fresh call"
        )


def _task_record(
    *,
    task_index: int,
    task: ObjectBongardTaskPlan,
    status: str,
    semantic: ObjectBongardSemanticArtifact | None,
    archive: ObjectBongardPanelRubricTaskRunArchive | None,
    panel_snapshots: Sequence[Mapping[str, Any]],
    journal_directories: Sequence[str],
    physical_call_count: int,
    exception_type: str | None = None,
) -> dict[str, Any]:
    correct = 0 if archive is None else archive.correct_count
    if status != "complete":
        correct = 0
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
            "query_calls_made": 0 if archive is None else archive.query_source_calls_made * 2,
            "query_pixels_opened": bool(
                archive is not None
                and archive.status is ObjectBongardPanelRubricTaskRunStatus.COMPLETE
            ),
            "exception_type": exception_type,
            "exception_message_persisted": False,
            "gap_or_exception_is_typed_zero_of_two": status != "complete",
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
        turn_kind="semantic_proposal",
        expected_prompt=object_bongard_semantics_prompt(),
        expected_images=semantic_images,
        expected_output_schema=object_bongard_semantics_output_schema(),
        runtime=runtime,
        underlying_transport=transport,
    )
    semantic = describe_object_bongard_support(
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

    specs = tuple(
        ObjectBongardRubricSpec.from_semantic_artifact(
            semantic,
            expected_artifact_digest=semantic.artifact_digest,
            candidate_rank=rank,
        )
        for rank in (0, 1)
    )
    side_0_by_rank: list[tuple[ObjectBongardPanelRubricArtifact, ...]] = []
    side_1_by_rank: list[tuple[ObjectBongardPanelRubricArtifact, ...]] = []
    for rank, spec in enumerate(specs):
        rank_blocks: list[tuple[ObjectBongardPanelRubricArtifact, ...]] = []
        for side, panel_ids in (
            ("side_0", task.side_0_support_panel_ids),
            ("side_1", task.side_1_support_panel_ids),
        ):
            artifacts: list[ObjectBongardPanelRubricArtifact] = []
            for panel_index, panel_id in enumerate(panel_ids):
                payload = support_png[panel_id]
                directory = journal_root / "support" / f"rank_{rank}" / side / f"panel_{panel_index:02d}"
                journal = ObjectBongardNamedImageTurnJournalTransport(
                    directory,
                    authorization_digest=authorization_digest,
                    execution_precommit_digest=precommit_digest,
                    task_id=task.task_id,
                    turn_kind=f"support_rank_{rank}_{side}_{panel_index:02d}",
                    expected_prompt=object_bongard_panel_rubric_prompt(spec),
                    expected_images=(("panel.png", payload),),
                    expected_output_schema=object_bongard_panel_rubric_output_schema(),
                    runtime=runtime,
                    underlying_transport=transport,
                )
                artifact = observe_object_bongard_panel_rubric(
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
        raise ObjectBongardPanelRubricCampaignCommandError("task support call count differs")

    freeze_path = task_root / "freeze.json"
    commit_path = task_root / "freeze_commit.json"

    def freeze_committer(payload: bytes) -> ObjectBongardPanelRubricTaskFreezeCommit:
        freeze = ObjectBongardPanelRubricTaskFreeze.from_data(json.loads(payload))
        _write_exact_once(freeze_path, payload)
        store_receipt = "sha256:" + canonical_digest(
            {
                "path": str(freeze_path.relative_to(root)),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
        commit = ObjectBongardPanelRubricTaskFreezeCommit.seal(
            freeze,
            payload,
            task_freeze_store_receipt_digest=store_receipt,
        )
        _write_once(commit_path, commit.to_data())
        return commit

    def freeze_reloader(commit_data: Mapping[str, Any]) -> bytes:
        commit = ObjectBongardPanelRubricTaskFreezeCommit.from_data(commit_data)
        if _read_json(commit_path, "task freeze commit") != commit.to_data():
            raise ObjectBongardPanelRubricCampaignCommandError("task freeze commit reload differs")
        payload = freeze_path.read_bytes()
        freeze = ObjectBongardPanelRubricTaskFreeze.from_data(json.loads(payload))
        commit.assert_matches(freeze, payload)
        return payload

    def query_source(
        freeze_data: Mapping[str, Any], commit_data: Mapping[str, Any]
    ) -> Mapping[str, ObjectBongardPanelRubricArtifact]:
        nonlocal physical_calls
        freeze = ObjectBongardPanelRubricTaskFreeze.from_data(freeze_data)
        commit = ObjectBongardPanelRubricTaskFreezeCommit.from_data(commit_data)
        if not freeze_path.exists() or not commit_path.exists():
            raise ObjectBongardPanelRubricCampaignCommandError("query source preceded durable freeze")
        commit.assert_matches(freeze, freeze_path.read_bytes())
        selected_spec = freeze.selected_rubric_spec
        result: dict[str, ObjectBongardPanelRubricArtifact] = {}
        for side, panel_id in (
            ("side_0", task.side_0_query_panel_id),
            ("side_1", task.side_1_query_panel_id),
        ):
            payload, snapshot = _snapshot_panel(
                task_root, phase=f"query_{side}", panel_id=panel_id, panel_reader=panel_reader
            )
            panel_snapshots.append(snapshot)
            directory = journal_root / "query" / side
            journal = ObjectBongardNamedImageTurnJournalTransport(
                directory,
                authorization_digest=authorization_digest,
                execution_precommit_digest=precommit_digest,
                task_id=task.task_id,
                turn_kind=f"selected_query_{side}",
                expected_prompt=object_bongard_panel_rubric_prompt(selected_spec),
                expected_images=(("panel.png", payload),),
                expected_output_schema=object_bongard_panel_rubric_output_schema(),
                runtime=runtime,
                underlying_transport=transport,
            )
            artifact = observe_object_bongard_panel_rubric(
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

    archive = run_object_bongard_panel_rubric_task(
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
        raise ObjectBongardPanelRubricCampaignCommandError("task physical call count differs")
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
class ObjectBongardPanelRubricCampaignCommandResult:
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


def _campaign_record(
    *,
    authorization: Mapping[str, Any],
    precommit: Mapping[str, Any],
    plan: ObjectBongardBatchPlan,
    task_records: Sequence[Mapping[str, Any]],
    physical_model_calls: int,
) -> dict[str, Any]:
    calibration_parent = _validate_accepted_calibration_parent(
        authorization.get("accepted_calibration_parent")
    )
    if precommit.get("accepted_calibration_parent") != calibration_parent:
        raise ObjectBongardPanelRubricCampaignCommandError(
            "campaign calibration parent differs across sealed records"
        )
    records = tuple(dict(item) for item in task_records)
    correct = sum(int(item["correct_count"]) for item in records)
    status_counts = {
        status: sum(item["status"] == status for item in records)
        for status in ("complete", "language_gap", "witness_gap", "task_exception")
    }
    return _seal(
        {
            "schema": CAMPAIGN_SCHEMA,
            "command_id": COMMAND_ID,
            "authorization_digest": authorization["authorization_digest"],
            "execution_precommit_digest": precommit["precommit_digest"],
            "batch_plan_digest": plan.record_digest,
            "accepted_calibration_parent_digest": calibration_parent[
                "parent_digest"
            ],
            "calibration_result_digest": calibration_parent[
                "calibration_result_digest"
            ],
            "calibration_cold_replay_digest": calibration_parent[
                "calibration_cold_replay_digest"
            ],
            "calibration_source_digest": calibration_parent[
                "calibration_source_digest"
            ],
            "task_records": list(records),
            "task_record_digests": [item["record_digest"] for item in records],
            "task_count": TASK_COUNT,
            "status_counts": status_counts,
            "correct_count": correct,
            "incorrect_count": QUERY_DENOMINATOR - correct,
            "score_denominator": QUERY_DENOMINATOR,
            "accuracy_ppm": correct * 1_000_000 // QUERY_DENOMINATOR,
            "physical_model_calls": physical_model_calls,
            "maximum_physical_model_calls": MAX_PHYSICAL_CALLS,
            "complete_task_scores_lie_in_zero_to_two": True,
            "language_witness_and_exception_are_zero_of_two": True,
            "query_calls_only_for_survivors": True,
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


def run_object_bongard_panel_rubric_campaign_command(
    output_root: str | os.PathLike[str],
    *,
    calibration_root: str | os.PathLike[str],
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
) -> ObjectBongardPanelRubricCampaignCommandResult:
    """Run the exact twelve-task TRAIN campaign from a fresh output root."""

    if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 120:
        raise ObjectBongardPanelRubricCampaignCommandError("minutes must lie in 1..120")
    if (
        isinstance(parallel_workers, bool)
        or not isinstance(parallel_workers, int)
        or not 1 <= parallel_workers <= TASK_COUNT
    ):
        raise ObjectBongardPanelRubricCampaignCommandError("parallel_workers must lie in 1..12")
    if _RAW_DIGEST.fullmatch(expected_launcher_sha256) is None:
        raise ObjectBongardPanelRubricCampaignCommandError("launcher SHA-256 differs")
    # This is deliberately the first external-data gate.  A missing, rejected,
    # or tampered calibration fails before the cohort archive is constructed,
    # before the fresh output root exists, and before a panel/model callback can
    # be reached.
    accepted_calibration_parent = _cold_verify_accepted_calibration_parent(
        calibration_root
    )
    prereg, plan = _load_exact_cohort(Path(preregistration_path), Path(plan_path))
    if panel_reader is None:
        official_archive = _load_default_archive(Path(descriptor_path), Path(archive_path))
        panel_reader = official_archive.read_panel
        archive_identity = {
            **official_archive.identity_data(),
            "record_digest": official_archive.record_digest,
        }
    elif archive_identity is None:
        raise ObjectBongardPanelRubricCampaignCommandError(
            "an injected panel reader requires an archive identity"
        )
    assert archive_identity is not None
    root = _ensure_fresh_root(output_root)
    plan_record = _seal(
        {
            "schema": "gkm.bongard-panel-rubric-campaign-plan-binding.v2",
            "preregistration": prereg,
            "batch_plan": plan.to_data(),
            "preregistration_file_sha256": PREREGISTRATION_FILE_SHA256,
            "batch_plan_file_sha256": PLAN_FILE_SHA256,
            "accepted_calibration_parent": accepted_calibration_parent,
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
        accepted_calibration_parent=accepted_calibration_parent,
        archive_identity=archive_identity,
        minutes=minutes,
        parallel_workers=parallel_workers,
        expected_launcher_sha256=expected_launcher_sha256,
    )
    _write_once(root / AUTHORIZATION_FILENAME, authorization)
    if _read_json(root / AUTHORIZATION_FILENAME, "authorization") != authorization:
        raise ObjectBongardPanelRubricCampaignCommandError("authorization reload differs")
    precommit, _runtime = _prepare_runtime_precommit(
        authorization=authorization,
        plan=plan,
        accepted_calibration_parent=accepted_calibration_parent,
        minutes=minutes,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
        cloud_policy_cache_snapshotter=cloud_policy_cache_snapshotter,
        model_catalog_snapshotter=model_catalog_snapshotter,
        launcher_fingerprinter=launcher_fingerprinter,
        runtime_attester=runtime_attester,
    )
    _write_once(root / PRECOMMIT_FILENAME, precommit)
    persisted_precommit = _read_json(root / PRECOMMIT_FILENAME, "execution precommit")
    runtime = _runtime_from_precommit(persisted_precommit)
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
            existing = task_root / "task_record.json"
            if existing.exists():
                return _read_json(existing, "task record")
            snapshot_records = tuple(
                _read_json(path, "partial panel snapshot")
                for path in sorted((task_root / "panels").glob("*/*.json"))
            ) if (task_root / "panels").exists() else ()
            journal_directories = tuple(
                str(path.parent.relative_to(root))
                for path in sorted(
                    (root / JOURNALS_DIRECTORY / f"{index:02d}_{task.task_id}").rglob("manifest.json")
                )
            ) if (root / JOURNALS_DIRECTORY / f"{index:02d}_{task.task_id}").exists() else ()
            semantic_path = task_root / "semantic_artifact.json"
            semantic = (
                ObjectBongardSemanticArtifact.from_data(
                    _read_json(semantic_path, "partial semantic artifact")
                )
                if semantic_path.exists()
                else None
            )
            record = _task_record(
                task_index=index,
                task=task,
                status="task_exception",
                semantic=semantic,
                archive=None,
                panel_snapshots=snapshot_records,
                journal_directories=journal_directories,
                physical_call_count=len(journal_directories),
                exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            )
            _write_once(existing, record)
            return record

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        task_records = tuple(
            executor.map(safe_execute, tuple(enumerate(plan.tasks)))
        )
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
            "accepted_calibration_parent_digest": accepted_calibration_parent[
                "parent_digest"
            ],
            "calibration_result_digest": accepted_calibration_parent[
                "calibration_result_digest"
            ],
            "calibration_cold_replay_digest": accepted_calibration_parent[
                "calibration_cold_replay_digest"
            ],
            "calibration_source_digest": accepted_calibration_parent[
                "calibration_source_digest"
            ],
            "model_calls_during_replay": 0,
            "new_pixels_opened_during_replay": 0,
            "all_task_archives_cold_replayed": True,
            **_authority_data(),
        },
        "replay_digest",
    )
    _write_once(root / REPLAY_FILENAME, replay)
    return verify_object_bongard_panel_rubric_campaign_command_directory(
        root, calibration_root=calibration_root
    )


def _decode_panel_snapshots(
    root: Path,
    task_root: Path,
    values: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[bytes, dict[str, Any]]]:
    result: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for value in values:
        raw = _validate_seal(value, "record_digest", "panel snapshot")
        try:
            payload = base64.b64decode(raw["exact_png_base64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise ObjectBongardPanelRubricCampaignCommandError("panel snapshot bytes are malformed") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if (
            raw.get("schema") != "gkm.bongard-panel-rubric-campaign-panel-snapshot.v1"
            or raw.get("png_sha256") != digest
            or not payload.startswith(b"\x89PNG\r\n\x1a\n")
            or raw.get("panel_id") in result
        ):
            raise ObjectBongardPanelRubricCampaignCommandError("panel snapshot identity differs")
        panel_key = hashlib.sha256(raw["panel_id"].encode("utf-8")).hexdigest()
        path = task_root / "panels" / raw["phase"] / f"{panel_key}_{digest}.json"
        if _read_json(path, "durable panel snapshot") != raw:
            raise ObjectBongardPanelRubricCampaignCommandError("durable panel snapshot differs")
        result[raw["panel_id"]] = (payload, raw)
    return result


def _forbidden_transport(*_args: Any, **_kwargs: Any) -> CodexStructuredResult:
    raise AssertionError("cold replay attempted a model transport")


def _replay_semantic_journal(
    *,
    root: Path,
    task: ObjectBongardTaskPlan,
    semantic: ObjectBongardSemanticArtifact,
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
        turn_kind="semantic_proposal",
        expected_prompt=object_bongard_semantics_prompt(),
        expected_images=expected_images,
        expected_output_schema=object_bongard_semantics_output_schema(),
        runtime=runtime,
        underlying_transport=_forbidden_transport,
    )
    replayed = describe_object_bongard_support(
        task_id=task.task_id,
        group_0_panel_ids=task.side_0_support_panel_ids,
        group_1_panel_ids=task.side_1_support_panel_ids,
        support_png_by_panel_id=support_png,
        observation_context_digest=precommit_digest,
        **_runtime_kwargs(runtime),
        transport=journal,
    )
    if replayed != semantic or journal.fresh_call_count != 0 or journal.reused_call_count != 1:
        raise ObjectBongardPanelRubricCampaignCommandError("semantic journal cold replay differs")


def _replay_panel_journal(
    *,
    root: Path,
    task: ObjectBongardTaskPlan,
    artifact: ObjectBongardPanelRubricArtifact,
    payload: bytes,
    runtime: ObjectBongardTurnRuntime,
    authorization_digest: str,
    precommit_digest: str,
    relative_directory: str,
    turn_kind: str,
) -> None:
    journal = ObjectBongardNamedImageTurnJournalTransport(
        root / relative_directory,
        authorization_digest=authorization_digest,
        execution_precommit_digest=precommit_digest,
        task_id=task.task_id,
        turn_kind=turn_kind,
        expected_prompt=object_bongard_panel_rubric_prompt(artifact.rubric_spec),
        expected_images=(("panel.png", payload),),
        expected_output_schema=object_bongard_panel_rubric_output_schema(),
        runtime=runtime,
        underlying_transport=_forbidden_transport,
    )
    replayed = observe_object_bongard_panel_rubric(
        payload,
        panel_id=artifact.panel_id,
        rubric_spec=artifact.rubric_spec,
        expected_panel_sha256=hashlib.sha256(payload).hexdigest(),
        expected_rubric_spec_digest=artifact.rubric_spec_digest,
        observation_context_digest=precommit_digest,
        **_runtime_kwargs(runtime),
        transport=journal,
    )
    if replayed != artifact or journal.fresh_call_count != 0 or journal.reused_call_count != 1:
        raise ObjectBongardPanelRubricCampaignCommandError("panel journal cold replay differs")


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
        or raw.get("score_denominator") != 2
        or raw.get("correct_count") not in (0, 1, 2)
        or raw.get("incorrect_count") != 2 - raw["correct_count"]
    ):
        raise ObjectBongardPanelRubricCampaignCommandError("task record scoring or identity differs")
    task_root = root / TASKS_DIRECTORY / f"{task_index:02d}_{task.task_id}"
    if _read_json(task_root / "task_record.json", "task record") != raw:
        raise ObjectBongardPanelRubricCampaignCommandError("durable task record differs")
    if raw["status"] == "task_exception":
        if raw["correct_count"] != 0 or raw["task_archive"] is not None:
            raise ObjectBongardPanelRubricCampaignCommandError("task exception escaped fixed scoring")
        _decode_panel_snapshots(root, task_root, raw["panel_snapshots"])
        if any(
            not (root / directory / "manifest.json").is_file()
            for directory in raw["journal_directories"]
        ):
            raise ObjectBongardPanelRubricCampaignCommandError("task exception journal inventory differs")
        return raw
    snapshots = _decode_panel_snapshots(root, task_root, raw["panel_snapshots"])
    support_ids = (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
    if not set(support_ids) <= set(snapshots):
        raise ObjectBongardPanelRubricCampaignCommandError("task support snapshot inventory differs")
    support_png = {panel_id: snapshots[panel_id][0] for panel_id in support_ids}
    semantic = ObjectBongardSemanticArtifact.from_data(raw["semantic_artifact"])
    verify_object_bongard_semantic_artifact(
        semantic,
        support_png_by_panel_id=support_png,
        expected_task_id=task.task_id,
        expected_observation_context_digest=precommit_digest,
        expected_artifact_digest=semantic.artifact_digest,
    )
    directories = tuple(raw["journal_directories"])
    if not directories or directories[0] != str(
        (Path(JOURNALS_DIRECTORY) / f"{task_index:02d}_{task.task_id}" / "semantic")
    ):
        raise ObjectBongardPanelRubricCampaignCommandError("semantic journal inventory differs")
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
    if semantic.status is not PrototypeSceneObserverStatus.SUCCESS:
        if raw["status"] != "language_gap" or raw["task_archive"] is not None or len(directories) != 1:
            raise ObjectBongardPanelRubricCampaignCommandError("language gap crossed into support or query")
        return raw
    archive = ObjectBongardPanelRubricTaskRunArchive.from_data(raw["task_archive"])
    cold_replay_object_bongard_panel_rubric_task(
        archive, expected_archive_digest=archive.record_digest
    )
    artifacts: list[tuple[ObjectBongardPanelRubricArtifact, str, str]] = []
    directory_index = 1
    for rank in (0, 1):
        for side, panel_ids, block in (
            ("side_0", task.side_0_support_panel_ids, archive.side_0_support_by_rank[rank]),
            ("side_1", task.side_1_support_panel_ids, archive.side_1_support_by_rank[rank]),
        ):
            for panel_index, (panel_id, artifact) in enumerate(zip(panel_ids, block, strict=True)):
                expected_directory = str(
                    Path(JOURNALS_DIRECTORY)
                    / f"{task_index:02d}_{task.task_id}"
                    / "support"
                    / f"rank_{rank}"
                    / side
                    / f"panel_{panel_index:02d}"
                )
                if directories[directory_index] != expected_directory:
                    raise ObjectBongardPanelRubricCampaignCommandError("support journal order differs")
                artifacts.append((artifact, expected_directory, f"support_rank_{rank}_{side}_{panel_index:02d}"))
                directory_index += 1
    for artifact, directory, turn_kind in artifacts:
        payload = snapshots[artifact.panel_id][0]
        verify_object_bongard_panel_rubric_artifact(
            artifact,
            payload,
            panel_id=artifact.panel_id,
            rubric_spec=artifact.rubric_spec,
            expected_artifact_digest=artifact.artifact_digest,
            expected_runtime_identity_digest=archive.version_spaces[0].observer_runtime_identity_digest,
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
            turn_kind=turn_kind,
        )
    if archive.status is ObjectBongardPanelRubricTaskRunStatus.COMPLETE:
        assert archive.side_0_query is not None and archive.side_1_query is not None
        for side, artifact in (
            ("side_0", archive.side_0_query),
            ("side_1", archive.side_1_query),
        ):
            directory = str(
                Path(JOURNALS_DIRECTORY)
                / f"{task_index:02d}_{task.task_id}"
                / "query"
                / side
            )
            if directories[directory_index] != directory:
                raise ObjectBongardPanelRubricCampaignCommandError("query journal order differs")
            payload = snapshots[artifact.panel_id][0]
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
        freeze = ObjectBongardPanelRubricTaskFreeze.from_data(json.loads(freeze_payload))
        commit = ObjectBongardPanelRubricTaskFreezeCommit.from_data(
            _read_json(task_root / "freeze_commit.json", "task freeze commit")
        )
        commit.assert_matches(freeze, freeze_payload)
    elif set(snapshots) != set(support_ids):
        raise ObjectBongardPanelRubricCampaignCommandError("gap task opened query pixels")
    if directory_index != len(directories):
        raise ObjectBongardPanelRubricCampaignCommandError("task journal inventory has extras")
    return raw


def verify_object_bongard_panel_rubric_campaign_command_directory(
    output_root: str | os.PathLike[str],
    *,
    calibration_root: str | os.PathLike[str],
) -> ObjectBongardPanelRubricCampaignCommandResult:
    """Cold replay a completed campaign without transport or new pixel access."""

    accepted_calibration_parent = _cold_verify_accepted_calibration_parent(
        calibration_root
    )
    root = Path(output_root).absolute()
    expected_root = {
        PLAN_FILENAME,
        AUTHORIZATION_FILENAME,
        PRECOMMIT_FILENAME,
        CAMPAIGN_FILENAME,
        REPLAY_FILENAME,
        TASKS_DIRECTORY,
        JOURNALS_DIRECTORY,
    }
    if not root.is_dir() or {item.name for item in root.iterdir()} != expected_root:
        raise ObjectBongardPanelRubricCampaignCommandError("campaign root inventory differs")
    plan_record = _validate_seal(
        _read_json(root / PLAN_FILENAME, "campaign plan"), "record_digest", "campaign plan"
    )
    plan = ObjectBongardBatchPlan.from_data(plan_record["batch_plan"])
    prereg = plan_record["preregistration"]
    if (
        plan.record_digest != PLAN_DIGEST
        or prereg.get("record_digest") != PREREGISTRATION_DIGEST
        or plan_record.get("preregistration_file_sha256") != PREREGISTRATION_FILE_SHA256
        or plan_record.get("batch_plan_file_sha256") != PLAN_FILE_SHA256
        or plan_record.get("accepted_calibration_parent")
        != accepted_calibration_parent
        or len(plan.tasks) != TASK_COUNT
    ):
        raise ObjectBongardPanelRubricCampaignCommandError("campaign cohort replay differs")
    authorization = _validate_seal(
        _read_json(root / AUTHORIZATION_FILENAME, "authorization"),
        "authorization_digest",
        "authorization",
    )
    expected_authorization = _authorization_record(
        plan=plan,
        preregistration=prereg,
        accepted_calibration_parent=accepted_calibration_parent,
        archive_identity=authorization["archive_identity"],
        minutes=authorization["minutes"],
        parallel_workers=authorization["parallel_workers"],
        expected_launcher_sha256=authorization["expected_launcher_sha256"],
    )
    if authorization != expected_authorization:
        raise ObjectBongardPanelRubricCampaignCommandError("authorization replay differs")
    precommit = _validate_seal(
        _read_json(root / PRECOMMIT_FILENAME, "execution precommit"),
        "precommit_digest",
        "execution precommit",
    )
    if (
        precommit.get("authorization_digest") != authorization["authorization_digest"]
        or precommit.get("batch_plan_digest") != plan.record_digest
        or precommit.get("accepted_calibration_parent")
        != accepted_calibration_parent
        or precommit.get("calibration_result_digest")
        != accepted_calibration_parent["calibration_result_digest"]
        or precommit.get("calibration_cold_replay_digest")
        != accepted_calibration_parent["calibration_cold_replay_digest"]
        or precommit.get("calibration_source_digest")
        != accepted_calibration_parent["calibration_source_digest"]
    ):
        raise ObjectBongardPanelRubricCampaignCommandError("precommit parents differ")
    runtime = _runtime_from_precommit(precommit)
    campaign = _validate_seal(
        _read_json(root / CAMPAIGN_FILENAME, "campaign result"),
        "campaign_digest",
        "campaign result",
    )
    if not isinstance(campaign.get("task_records"), list) or len(campaign["task_records"]) != TASK_COUNT:
        raise ObjectBongardPanelRubricCampaignCommandError("campaign task inventory differs")
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
        raise ObjectBongardPanelRubricCampaignCommandError("campaign aggregate replay differs")
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
        raise ObjectBongardPanelRubricCampaignCommandError("physical journal tree differs")
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
            "accepted_calibration_parent_digest": accepted_calibration_parent[
                "parent_digest"
            ],
            "calibration_result_digest": accepted_calibration_parent[
                "calibration_result_digest"
            ],
            "calibration_cold_replay_digest": accepted_calibration_parent[
                "calibration_cold_replay_digest"
            ],
            "calibration_source_digest": accepted_calibration_parent[
                "calibration_source_digest"
            ],
            "model_calls_during_replay": 0,
            "new_pixels_opened_during_replay": 0,
            "all_task_archives_cold_replayed": True,
            **_authority_data(),
        },
        "replay_digest",
    )
    if replay != expected_replay:
        raise ObjectBongardPanelRubricCampaignCommandError("campaign replay record differs")
    return ObjectBongardPanelRubricCampaignCommandResult(root, campaign, replay)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--output-root", required=True)
    launch.add_argument("--calibration-root", required=True)
    launch.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    launch.add_argument("--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS)
    launch.add_argument("--executable", default=DEFAULT_EXECUTABLE)
    launch.add_argument("--expected-launcher-sha256", default=DEFAULT_EXPECTED_LAUNCHER_SHA256)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-root", required=True)
    verify.add_argument("--calibration-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = (
            run_object_bongard_panel_rubric_campaign_command(
                args.output_root,
                calibration_root=args.calibration_root,
                minutes=args.minutes,
                parallel_workers=args.parallel_workers,
                executable=args.executable,
                expected_launcher_sha256=args.expected_launcher_sha256,
            )
            if args.action == "launch"
            else verify_object_bongard_panel_rubric_campaign_command_directory(
                args.output_root, calibration_root=args.calibration_root
            )
        )
    except Exception as exc:
        print(f"panel-rubric campaign failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "output_root": str(result.output_root),
        "correct_count": result.correct_count,
        "score_denominator": result.score_denominator,
        "physical_model_calls": result.physical_model_calls,
        "campaign_digest": result.campaign["campaign_digest"],
        "replay_digest": result.replay["replay_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DEFAULT_EXPECTED_LAUNCHER_SHA256",
    "MAX_PHYSICAL_CALLS",
    "ObjectBongardPanelRubricCampaignCommandError",
    "ObjectBongardPanelRubricCampaignCommandResult",
    "QUERY_DENOMINATOR",
    "TASK_COUNT",
    "object_bongard_panel_rubric_campaign_command_source_digest",
    "run_object_bongard_panel_rubric_campaign_command",
    "verify_object_bongard_panel_rubric_campaign_command_directory",
)
