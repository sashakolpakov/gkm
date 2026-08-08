"""Launch and cold-verify the sealed whole-panel rubric calibration.

This is the production adapter around :mod:`object_bongard_panel_rubric_calibration`.
It accepts only the exact five frozen v10 nomination records, creates a fresh
runtime authorization and precommit, runs the fixed 24 whole-panel jobs, and
persists the full observation batch before support labels enter assessment.
Verification reconstructs every decision from disk with a transport that
raises if called.  Python is the sole predicate and replay authority; Lean is
absent and removable.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import argparse
import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.codex_no_tools_preflight import (
    CodexNoToolsAttestation,
    attest_codex_no_tools,
)
from bongard.object_bongard_panel_rubric_calibration import (
    DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE,
    ObjectBongardPanelRubricCalibrationAssessment,
    ObjectBongardPanelRubricCalibrationDurableFreeze,
    ObjectBongardPanelRubricCalibrationObservationBatch,
    ObjectBongardPanelRubricCalibrationPlan,
    PANEL_RUBRIC_CALIBRATION_JOB_COUNT,
    assess_object_bongard_panel_rubric_calibration,
    bind_object_bongard_panel_rubric_calibration_nomination,
    cold_verify_object_bongard_panel_rubric_calibration,
    load_object_bongard_panel_rubric_calibration_source,
    object_bongard_panel_rubric_calibration_source_digest,
    persist_and_reload_object_bongard_panel_rubric_calibration_batch,
    run_object_bongard_panel_rubric_calibration_observation,
    run_object_bongard_panel_rubric_calibration_observations,
)
from bongard.object_bongard_semantics import (
    ObjectBongardSemanticArtifact,
    verify_object_bongard_semantic_artifact,
)
from bongard.object_bongard_turn_journal import ObjectBongardTurnRuntime
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
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


PANEL_RUBRIC_CALIBRATION_COMMAND_SCHEMA = (
    "gkm.bongard-panel-rubric-calibration-command.v1"
)
CALIBRATION_AUTHORIZATION_SCHEMA = (
    "gkm.bongard-panel-rubric-calibration-authorization.v1"
)
CALIBRATION_PRECOMMIT_SCHEMA = (
    "gkm.bongard-panel-rubric-calibration-runtime-precommit.v1"
)
CALIBRATION_REPLAY_SCHEMA = "gkm.bongard-panel-rubric-calibration-replay.v1"
CALIBRATION_RESULT_SCHEMA = "gkm.bongard-panel-rubric-calibration-result.v1"

AUTHORIZATION_FILENAME = "authorization.json"
PRECOMMIT_FILENAME = "execution_precommit.json"
PLAN_FILENAME = "plan.json"
BATCH_FILENAME = "observation_batch.json"
FREEZE_FILENAME = "durable_freeze.json"
ASSESSMENT_FILENAME = "assessment.json"
REPLAY_FILENAME = "cold_replay.json"
RESULT_FILENAME = "result.json"
JOURNAL_DIRECTORY = "journals"

DEFAULT_V10_NOMINATION_ROOT = Path(
    "downloads/ShapeBongard_V2_full/"
    "object_rubric_nomination_20260808_all_support_v10"
)
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MINUTES = 15
DEFAULT_EXECUTABLE = "codex"
DEFAULT_EXPECTED_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)
DEFAULT_PARALLEL_WORKERS = 4
MAX_PARALLEL_WORKERS = 4

V10_NOMINATION_ARTIFACT_DIGEST = (
    "c765cdfaba7315ce04265e2151490a86f25d042347eac5cba8a7fc1282dc7c29"
)
V10_NOMINATION_AUTHORIZATION_DIGEST = (
    "sha256:65d2c58cb09bd3e7aeecde0093a50047ccb1676af105559758b589e5cdd368fe"
)
V10_NOMINATION_PRECOMMIT_DIGEST = (
    "sha256:caaa7aea85d3c35838c0abfbc052743f7fe05a7e52ff817c2a3a1c2e2ba992bd"
)
V10_NOMINATION_REPLAY_DIGEST = (
    "sha256:b1c20a920e12f4d2e85f42a3cee06d7565e308f52378e5edfb6bc4ee7c9ed6c4"
)
V10_NOMINATION_RESULT_DIGEST = (
    "sha256:2e0bcd7e0792641265806ccde66bac1af7f791746cf02051454f57ebf7fac4cf"
)
V10_NOMINATION_SOURCE_DIGEST = (
    "78c0228d4326dc5e9335fd506e9dce23ec08d2ce4fef6d9a53653b8ab4cbefbe"
)
V10_CONTEXT_TASK_ID = "bd_two_mismatch_sectors8-thin_seven_lines2_0000"

# Filename, exact canonical file SHA-256, internal digest field, internal digest.
_V10_NOMINATION_PARENT_FILES = (
    ("authorization.json", "4ad41a8dd69b30661ea47394a3582cdacd90fcfd934d4755c6b7c2e277a6a586", "authorization_digest", V10_NOMINATION_AUTHORIZATION_DIGEST),
    ("execution_precommit.json", "26dea77e646ee138c2e96ff8bf87f0cf7919e76e4d1f3b9a4da81d305717691a", "precommit_digest", V10_NOMINATION_PRECOMMIT_DIGEST),
    ("semantic_artifact.json", "2906a76c6dd971301ce99671720476c0f34c62206f2dfaeaf1025cb03bf923c8", "artifact_digest", V10_NOMINATION_ARTIFACT_DIGEST),
    ("cold_replay.json", "07f2a56448ad4e8eabb4462d5b9602462407596bd30ed7ff3b942c478e4ff3d1", "replay_digest", V10_NOMINATION_REPLAY_DIGEST),
    ("result.json", "e3a3312eb30a71fc0e531937706edd86d54699f76850eefea064fd40d3f60bf5", "result_digest", V10_NOMINATION_RESULT_DIGEST),
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RECORD_BYTES = 16 * 1024 * 1024


class ObjectBongardPanelRubricCalibrationCommandError(RuntimeError):
    """A command commitment, runtime seal, or cold replay differs."""


@dataclass(frozen=True, slots=True)
class _PinnedV10Nomination:
    artifact: ObjectBongardSemanticArtifact
    authorization_digest: str
    execution_precommit_digest: str
    cold_replay_digest: str
    result_digest: str
    source_digest: str
    accepted: bool
    parent_file_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _FreshRuntimePrecommit:
    record: Mapping[str, Any]
    runtime: ObjectBongardTurnRuntime

    @property
    def precommit_digest(self) -> str:
        value = self.record.get("record_digest")
        if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
            raise ObjectBongardPanelRubricCalibrationCommandError(
                "runtime precommit digest is malformed"
            )
        return value


@dataclass(frozen=True, slots=True)
class VerifiedObjectBongardPanelRubricCalibration:
    output_root: Path
    authorization_digest: str
    execution_precommit_digest: str
    plan_digest: str
    batch_digest: str
    freeze_digest: str
    assessment_digest: str
    replay_digest: str
    result_digest: str
    accepted: bool
    selected_candidate_rank: int | None
    fresh_call_count: int
    reused_call_count: int

    def summary_data(self) -> dict[str, object]:
        return {
            "schema": "gkm.bongard-panel-rubric-calibration-summary.v1",
            "output_root": str(self.output_root),
            "authorization_digest": self.authorization_digest,
            "execution_precommit_digest": self.execution_precommit_digest,
            "plan_digest": self.plan_digest,
            "batch_digest": self.batch_digest,
            "freeze_digest": self.freeze_digest,
            "assessment_digest": self.assessment_digest,
            "cold_replay_digest": self.replay_digest,
            "result_digest": self.result_digest,
            "accepted": self.accepted,
            "selected_candidate_rank": self.selected_candidate_rank,
            "fresh_call_count": self.fresh_call_count,
            "reused_call_count": self.reused_call_count,
            **_authority_data(),
        }


Transport = Callable[..., CodexStructuredResult]


def object_bongard_panel_rubric_calibration_command_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def _record(body: Mapping[str, Any]) -> dict[str, Any]:
    frozen = json.loads(canonical_json(dict(body)).decode("utf-8"))
    if not isinstance(frozen, dict):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "record body is not an object"
        )
    return {**frozen, "record_digest": "sha256:" + canonical_digest(frozen)}


def _verify_record(
    value: object, *, schema: str, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"{label} is not an object"
        )
    raw = dict(value)
    digest = raw.pop("record_digest", None)
    if raw.get("schema") != schema or digest != "sha256:" + canonical_digest(raw):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"{label} schema or digest differs"
        )
    return {**raw, "record_digest": digest}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, value: Mapping[str, Any], label: str) -> str:
    payload = canonical_json(dict(value)) + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"{label} already exists"
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if path.read_bytes() != payload:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"persisted {label} differs"
        )
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"cannot read {label}"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= _MAX_RECORD_BYTES
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"{label} is not a bounded singly-linked regular file"
        )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened.st_nlink != 1 or identity != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise ObjectBongardPanelRubricCalibrationCommandError(
                f"{label} changed while opening"
            )
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                raise ObjectBongardPanelRubricCalibrationCommandError(
                    f"{label} was truncated"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if after.st_nlink != 1 or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != identity:
            raise ObjectBongardPanelRubricCalibrationCommandError(
                f"{label} changed while reading"
            )
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"{label} is malformed JSON"
        ) from exc
    if not isinstance(value, dict) or payload != canonical_json(value) + b"\n":
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"{label} is not canonical JSON plus newline"
        )
    return value


def _fresh_output_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    parent = candidate.parent.resolve(strict=True)
    root = parent / candidate.name
    if not candidate.name or root.exists() or root.is_symlink():
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "output root must be fresh"
        )
    root.mkdir(mode=0o700)
    (root / JOURNAL_DIRECTORY).mkdir(mode=0o700)
    _fsync_directory(root)
    _fsync_directory(parent)
    return root


def _existing_output_root(value: str | os.PathLike[str]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "calibration root cannot be a symlink"
        )
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "calibration root is not a directory"
        )
    return root


def _read_pinned_json(
    path: Path, *, expected_file_sha256: str, label: str
) -> dict[str, Any]:
    if _RAW_DIGEST.fullmatch(expected_file_sha256) is None:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"{label} file digest is malformed"
        )
    try:
        before = path.lstat()
    except OSError as exc:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"cannot inspect pinned {label}"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or not 0 < before.st_size <= _MAX_RECORD_BYTES
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"pinned {label} is not a bounded regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        payload = bytearray()
        while len(payload) < opened.st_size:
            chunk = os.read(descriptor, min(65536, opened.st_size - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    frozen = bytes(payload)
    if (
        (opened.st_dev, opened.st_ino, opened.st_size)
        != (before.st_dev, before.st_ino, before.st_size)
        or (after.st_dev, after.st_ino, after.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or len(frozen) != opened.st_size
        or hashlib.sha256(frozen).hexdigest() != expected_file_sha256
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"pinned {label} bytes differ"
        )
    try:
        value = json.loads(frozen.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"pinned {label} is malformed JSON"
        ) from exc
    if not isinstance(value, dict) or frozen != canonical_json(value) + b"\n":
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"pinned {label} is not canonical"
        )
    return value


def _verify_internal_digest(
    value: Mapping[str, Any], *, field: str, expected: str, label: str
) -> None:
    body = dict(value)
    observed = body.pop(field, None)
    addressed = expected.startswith("sha256:")
    computed = canonical_digest(body)
    if observed != expected or expected.removeprefix("sha256:") != computed:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"pinned {label} content digest differs"
        )


def _load_pinned_v10_nomination(
    nomination_root: str | os.PathLike[str],
    plan_source: object,
) -> _PinnedV10Nomination:
    root = Path(nomination_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "v10 nomination root is not a directory"
        )
    records: dict[str, dict[str, Any]] = {}
    files: dict[str, str] = {}
    for filename, file_sha, digest_field, digest in _V10_NOMINATION_PARENT_FILES:
        record = _read_pinned_json(
            root / filename,
            expected_file_sha256=file_sha,
            label=f"v10 nomination {filename}",
        )
        _verify_internal_digest(
            record,
            field=digest_field,
            expected=digest,
            label=f"v10 nomination {filename}",
        )
        records[filename] = record
        files[filename] = file_sha
    authorization = records["authorization.json"]
    precommit = records["execution_precommit.json"]
    replay = records["cold_replay.json"]
    result = records["result.json"]
    if (
        authorization.get("schema")
        != "gkm.bongard-object-rubric-nomination-authorization.v4"
        or precommit.get("schema")
        != "gkm.bongard-object-rubric-nomination-precommit.v4"
        or replay.get("schema")
        != "gkm.bongard-object-rubric-nomination-cold-replay.v4"
        or result.get("schema")
        != "gkm.bongard-object-rubric-nomination-result.v4"
        or any(
            item.get("source_digest") != V10_NOMINATION_SOURCE_DIGEST
            for item in (authorization, precommit, replay, result)
        )
        or precommit.get("authorization_digest")
        != V10_NOMINATION_AUTHORIZATION_DIGEST
        or replay.get("authorization_digest")
        != V10_NOMINATION_AUTHORIZATION_DIGEST
        or result.get("authorization_digest")
        != V10_NOMINATION_AUTHORIZATION_DIGEST
        or replay.get("execution_precommit_digest")
        != V10_NOMINATION_PRECOMMIT_DIGEST
        or result.get("execution_precommit_digest")
        != V10_NOMINATION_PRECOMMIT_DIGEST
        or result.get("cold_replay_digest") != V10_NOMINATION_REPLAY_DIGEST
        or replay.get("semantic_artifact_digest")
        != V10_NOMINATION_ARTIFACT_DIGEST
        or result.get("semantic_artifact_digest")
        != V10_NOMINATION_ARTIFACT_DIGEST
        or result.get("accepted") is not True
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "pinned v10 nomination linkage differs"
        )
    artifact = ObjectBongardSemanticArtifact.from_data(
        records["semantic_artifact.json"],
        expected_artifact_digest=V10_NOMINATION_ARTIFACT_DIGEST,
    )
    panels = getattr(plan_source, "panels", None)
    group_0 = getattr(plan_source, "group_0_panels", None)
    group_1 = getattr(plan_source, "group_1_panels", None)
    if not isinstance(panels, tuple) or not isinstance(group_0, tuple) or not isinstance(group_1, tuple):
        raise TypeError("plan_source must be a typed panel calibration source")
    expected_groups = (
        tuple(sorted(item.panel_id for item in group_0)),
        tuple(sorted(item.panel_id for item in group_1)),
    )
    if artifact.group_panel_ids != expected_groups:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "pinned nomination groups differ from the calibration source"
        )
    verify_object_bongard_semantic_artifact(
        artifact,
        support_png_by_panel_id={item.panel_id: item.exact_png_bytes for item in panels},
        expected_task_id=V10_CONTEXT_TASK_ID,
        expected_observation_context_digest=V10_NOMINATION_PRECOMMIT_DIGEST,
        expected_artifact_digest=V10_NOMINATION_ARTIFACT_DIGEST,
    )
    return _PinnedV10Nomination(
        artifact,
        V10_NOMINATION_AUTHORIZATION_DIGEST,
        V10_NOMINATION_PRECOMMIT_DIGEST,
        V10_NOMINATION_REPLAY_DIGEST,
        V10_NOMINATION_RESULT_DIGEST,
        V10_NOMINATION_SOURCE_DIGEST,
        True,
        files,
    )


def _authorization(
    plan: ObjectBongardPanelRubricCalibrationPlan,
    nomination: _PinnedV10Nomination,
    *,
    parallel_workers: int,
    minutes: int,
    verbose: bool,
    executable: str,
    expected_launcher_sha256: str,
) -> dict[str, Any]:
    if (
        not isinstance(plan, ObjectBongardPanelRubricCalibrationPlan)
        or isinstance(parallel_workers, bool)
        or not isinstance(parallel_workers, int)
        or not 1 <= parallel_workers <= MAX_PARALLEL_WORKERS
        or isinstance(minutes, bool)
        or not isinstance(minutes, int)
        or not 1 <= minutes <= 120
        or not isinstance(verbose, bool)
        or not isinstance(executable, str)
        or not executable
        or not isinstance(expected_launcher_sha256, str)
        or _RAW_DIGEST.fullmatch(expected_launcher_sha256) is None
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "calibration authorization selectors are invalid"
        )
    return _record(
        {
            "schema": CALIBRATION_AUTHORIZATION_SCHEMA,
            "command_schema": PANEL_RUBRIC_CALIBRATION_COMMAND_SCHEMA,
            "purpose": "sealed-two-rank-whole-panel-soft-rubric-calibration",
            "plan_digest": plan.plan_digest,
            "source_digest": plan.source.source_digest,
            "nomination": {
                "artifact_digest": nomination.artifact.artifact_digest,
                "authorization_digest": nomination.authorization_digest,
                "execution_precommit_digest": nomination.execution_precommit_digest,
                "cold_replay_digest": nomination.cold_replay_digest,
                "result_digest": nomination.result_digest,
                "source_digest": nomination.source_digest,
                "parent_file_sha256": dict(nomination.parent_file_sha256),
            },
            "rubric_specs": [item.to_data() for item in plan.rubric_specs],
            "panel_inventory": [item.commitment_data() for item in plan.source.panels],
            "job_count": PANEL_RUBRIC_CALIBRATION_JOB_COUNT,
            "job_order": "candidate-rank-then-source-ordinal",
            "parallel_workers": parallel_workers,
            "runtime_policy": {
                "model": DEFAULT_MODEL,
                "reasoning_effort": DEFAULT_REASONING_EFFORT,
                "minutes": minutes,
                "verbose": verbose,
                "executable": executable,
                "expected_launcher_sha256": expected_launcher_sha256,
                "fresh_cloud_policy_cache_snapshot_required": True,
                "fresh_model_catalog_snapshot_required": True,
                "fresh_no_tools_attestation_required": True,
                "authenticated_launcher_fingerprint_required": True,
            },
            "source_identities": {
                "command_source_sha256": (
                    object_bongard_panel_rubric_calibration_command_source_digest()
                ),
                "calibration_source_sha256": (
                    object_bongard_panel_rubric_calibration_source_digest()
                ),
                "runtime_transport_source_sha256": (
                    prototype_scene_transport_source_digest()
                ),
            },
            "all_24_artifacts_frozen_before_support_labels": True,
            "support_roles_visible_to_observer": False,
            "query_pixels_authorized": False,
            "broad_cohort_pixels_authorized": False,
            "official_test_pixels_authorized": False,
            "fresh_output_root_required": True,
            "resume_of_failed_root_authorized": False,
            **_authority_data(),
        }
    )


def _decode_snapshot(value: object, label: str) -> bytes | None:
    if value is None:
        return None
    try:
        return base64.b64decode(value, validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            f"{label} snapshot is malformed"
        ) from exc


def _runtime_from_precommit(
    record: Mapping[str, Any], *, expected_authorization_digest: str
) -> ObjectBongardTurnRuntime:
    verified = _verify_record(
        record,
        schema=CALIBRATION_PRECOMMIT_SCHEMA,
        label="calibration execution precommit",
    )
    expected_fields = {
        "schema",
        "command_schema",
        "authorization_digest",
        "command_source_sha256",
        "calibration_source_sha256",
        "runtime_binding",
        "cloud_policy_cache_snapshot_base64",
        "model_catalog_snapshot_base64",
        "no_tools_attestation",
        "launcher_fingerprint",
        "fresh_runtime_snapshots_captured",
        "precommit_fsynced_before_calls",
        "panel_calls_started",
        "support_labels_introduced",
        "query_pixels_opened",
        "broad_cohort_pixels_opened",
        "official_test_pixels_opened",
        *_authority_data(),
        "record_digest",
    }
    binding = verified.get("runtime_binding")
    fingerprint = verified.get("launcher_fingerprint")
    attestation_data = verified.get("no_tools_attestation")
    if (
        set(verified) != expected_fields
        or verified.get("command_schema")
        != PANEL_RUBRIC_CALIBRATION_COMMAND_SCHEMA
        or verified.get("authorization_digest")
        != expected_authorization_digest
        or verified.get("command_source_sha256")
        != object_bongard_panel_rubric_calibration_command_source_digest()
        or verified.get("calibration_source_sha256")
        != object_bongard_panel_rubric_calibration_source_digest()
        or verified.get("fresh_runtime_snapshots_captured") is not True
        or verified.get("precommit_fsynced_before_calls") is not True
        or verified.get("panel_calls_started") is not False
        or verified.get("support_labels_introduced") is not False
        or verified.get("query_pixels_opened") is not False
        or verified.get("broad_cohort_pixels_opened") is not False
        or verified.get("official_test_pixels_opened") is not False
        or any(verified.get(key) != item for key, item in _authority_data().items())
        or not isinstance(binding, Mapping)
        or not isinstance(fingerprint, Mapping)
        or set(fingerprint) != {"version", "launcher_digest"}
        or fingerprint.get("version") != PINNED_CODEX_CLI_VERSION
        or not isinstance(attestation_data, Mapping)
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "calibration execution precommit policy differs"
        )
    cache = CloudPolicyCacheSnapshot(
        _decode_snapshot(
            verified.get("cloud_policy_cache_snapshot_base64"), "policy cache"
        )
    )
    catalog_bytes = _decode_snapshot(
        verified.get("model_catalog_snapshot_base64"), "model catalog"
    )
    if catalog_bytes is None:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "model catalog snapshot is absent"
        )
    catalog = CodexModelCatalogSnapshot(catalog_bytes)
    attestation = CodexNoToolsAttestation.from_mapping(attestation_data)
    try:
        runtime = ObjectBongardTurnRuntime(
            model=binding["model"],
            reasoning_effort=binding["reasoning_effort"],
            minutes=binding["minutes"],
            verbose=binding["verbose"],
            executable=binding["executable"],
            cloud_policy_cache_snapshot=cache,
            model_catalog_snapshot=catalog,
            expected_launcher_digest=binding["expected_launcher_digest"],
            no_tools_attestation=attestation,
            transport_source_digest=binding["transport_source_digest"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "runtime binding is malformed"
        ) from exc
    if (
        runtime.binding != dict(binding)
        or fingerprint.get("launcher_digest") != runtime.expected_launcher_digest
        or runtime.transport_source_digest
        != prototype_scene_transport_source_digest()
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "runtime differs from its execution precommit"
        )
    return runtime


def _prepare_runtime_precommit(
    authorization: Mapping[str, Any],
    *,
    cloud_policy_cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot],
    model_catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot],
    launcher_fingerprinter: Callable[..., Mapping[str, str]],
    runtime_attester: Callable[..., CodexNoToolsAttestation],
) -> _FreshRuntimePrecommit:
    auth = _verify_record(
        authorization,
        schema=CALIBRATION_AUTHORIZATION_SCHEMA,
        label="calibration authorization",
    )
    policy = auth.get("runtime_policy")
    if not isinstance(policy, Mapping):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "authorization runtime policy is malformed"
        )
    cache = cloud_policy_cache_snapshotter()
    catalog = model_catalog_snapshotter()
    if not isinstance(cache, CloudPolicyCacheSnapshot) or not isinstance(
        catalog, CodexModelCatalogSnapshot
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "fresh runtime snapshot has the wrong type"
        )
    fingerprint = launcher_fingerprinter(
        policy["executable"],
        expected_launcher_digest=policy["expected_launcher_sha256"],
    )
    attestation = runtime_attester(
        executable=policy["executable"],
        expected_launcher_digest=policy["expected_launcher_sha256"],
        model_catalog_snapshot=catalog,
        cloud_policy_cache_snapshot=cache,
    )
    if not isinstance(attestation, CodexNoToolsAttestation):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "fresh no-tools attestation has the wrong type"
        )
    runtime = ObjectBongardTurnRuntime(
        model=policy["model"],
        reasoning_effort=policy["reasoning_effort"],
        minutes=policy["minutes"],
        verbose=policy["verbose"],
        executable=policy["executable"],
        cloud_policy_cache_snapshot=cache,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=policy["expected_launcher_sha256"],
        no_tools_attestation=attestation,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    record = _record(
        {
            "schema": CALIBRATION_PRECOMMIT_SCHEMA,
            "command_schema": PANEL_RUBRIC_CALIBRATION_COMMAND_SCHEMA,
            "authorization_digest": auth["record_digest"],
            "command_source_sha256": (
                object_bongard_panel_rubric_calibration_command_source_digest()
            ),
            "calibration_source_sha256": (
                object_bongard_panel_rubric_calibration_source_digest()
            ),
            "runtime_binding": runtime.binding,
            "cloud_policy_cache_snapshot_base64": (
                None
                if cache.data is None
                else base64.b64encode(cache.data).decode("ascii")
            ),
            "model_catalog_snapshot_base64": base64.b64encode(catalog.data).decode(
                "ascii"
            ),
            "no_tools_attestation": attestation.to_dict(),
            "launcher_fingerprint": dict(fingerprint),
            "fresh_runtime_snapshots_captured": True,
            "precommit_fsynced_before_calls": True,
            "panel_calls_started": False,
            "support_labels_introduced": False,
            "query_pixels_opened": False,
            "broad_cohort_pixels_opened": False,
            "official_test_pixels_opened": False,
            **_authority_data(),
        }
    )
    restored = _runtime_from_precommit(
        record, expected_authorization_digest=auth["record_digest"]
    )
    if restored != runtime:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "runtime precommit typed round trip differs"
        )
    return _FreshRuntimePrecommit(record, restored)


def _seal_fresh_runtime(
    root: Path,
    plan: ObjectBongardPanelRubricCalibrationPlan,
    nomination: _PinnedV10Nomination,
    *,
    parallel_workers: int,
    minutes: int,
    verbose: bool,
    executable: str,
    expected_launcher_sha256: str,
    cloud_policy_cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot],
    model_catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot],
    launcher_fingerprinter: Callable[..., Mapping[str, str]],
    runtime_attester: Callable[..., CodexNoToolsAttestation],
) -> tuple[dict[str, Any], _FreshRuntimePrecommit]:
    authorization = _authorization(
        plan,
        nomination,
        parallel_workers=parallel_workers,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
    )
    _write_once(root / AUTHORIZATION_FILENAME, authorization, "authorization")
    stored_authorization = _verify_record(
        _read_json(root / AUTHORIZATION_FILENAME, "authorization"),
        schema=CALIBRATION_AUTHORIZATION_SCHEMA,
        label="authorization",
    )
    if stored_authorization != authorization:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "persisted authorization differs"
        )
    prepared = _prepare_runtime_precommit(
        stored_authorization,
        cloud_policy_cache_snapshotter=cloud_policy_cache_snapshotter,
        model_catalog_snapshotter=model_catalog_snapshotter,
        launcher_fingerprinter=launcher_fingerprinter,
        runtime_attester=runtime_attester,
    )
    _write_once(
        root / PRECOMMIT_FILENAME,
        prepared.record,
        "execution precommit",
    )
    stored_record = _verify_record(
        _read_json(root / PRECOMMIT_FILENAME, "execution precommit"),
        schema=CALIBRATION_PRECOMMIT_SCHEMA,
        label="execution precommit",
    )
    if stored_record != prepared.record:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "persisted execution precommit differs"
        )
    runtime = _runtime_from_precommit(
        stored_record,
        expected_authorization_digest=stored_authorization["record_digest"],
    )
    return stored_authorization, _FreshRuntimePrecommit(stored_record, runtime)


def _forbidden_transport(*_args: Any, **_kwargs: Any) -> CodexStructuredResult:
    raise AssertionError("cold calibration verification attempted model transport")


def _verify_all_journals(
    root: Path,
    plan: ObjectBongardPanelRubricCalibrationPlan,
    batch: ObjectBongardPanelRubricCalibrationObservationBatch,
    *,
    runtime: ObjectBongardTurnRuntime,
    authorization_digest: str,
    execution_precommit_digest: str,
) -> tuple[str, ...]:
    expected_pairs = tuple(
        (panel.panel_binding_digest, spec.spec_digest)
        for spec in plan.rubric_specs
        for panel in plan.source.panels
    )
    actual_pairs = tuple(
        (run.panel_binding_digest, run.rubric_spec_digest) for run in batch.runs
    )
    if actual_pairs != expected_pairs:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "batch rank-by-ordinal order differs"
        )
    expected_ranks = {f"rank_{spec.candidate_rank}" for spec in plan.rubric_specs}
    if {item.name for item in (root / JOURNAL_DIRECTORY).iterdir()} != expected_ranks:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "journal rank inventory differs"
        )
    for spec in plan.rubric_specs:
        rank_root = root / JOURNAL_DIRECTORY / f"rank_{spec.candidate_rank}"
        if not rank_root.is_dir() or rank_root.is_symlink():
            raise ObjectBongardPanelRubricCalibrationCommandError(
                "journal rank path is not a real directory"
            )
        expected_ordinals = {
            f"ordinal_{panel.ordinal:03d}" for panel in plan.source.panels
        }
        if {item.name for item in rank_root.iterdir()} != expected_ordinals:
            raise ObjectBongardPanelRubricCalibrationCommandError(
                "journal ordinal inventory differs"
            )
    journal_digests: list[str] = []
    for run, (panel_digest, spec_digest) in zip(
        batch.runs, expected_pairs, strict=True
    ):
        panel = next(
            item
            for item in plan.source.panels
            if item.panel_binding_digest == panel_digest
        )
        spec = next(
            item for item in plan.rubric_specs if item.spec_digest == spec_digest
        )
        ordinal_root = (
            root
            / JOURNAL_DIRECTORY
            / f"rank_{spec.candidate_rank}"
            / f"ordinal_{panel.ordinal:03d}"
        )
        expected_ordinal_inventory = {"turn"}
        if run.failure_evidence is not None:
            expected_ordinal_inventory.add("failure_evidence.json")
        if (
            not ordinal_root.is_dir()
            or ordinal_root.is_symlink()
            or {item.name for item in ordinal_root.iterdir()}
            != expected_ordinal_inventory
        ):
            raise ObjectBongardPanelRubricCalibrationCommandError(
                "journal job inventory differs"
            )
        turn_root = ordinal_root / "turn"
        if (
            not turn_root.is_dir()
            or turn_root.is_symlink()
            or {item.name for item in turn_root.iterdir()}
            != {"manifest.json", "claim.json", "result.json", "outcome.json"}
        ):
            raise ObjectBongardPanelRubricCalibrationCommandError(
                "terminal turn journal inventory differs"
            )
        replayed_run = run_object_bongard_panel_rubric_calibration_observation(
            plan,
            panel,
            spec,
            runtime=runtime,
            journal_root=root / JOURNAL_DIRECTORY,
            authorization_digest=authorization_digest,
            execution_precommit_digest=execution_precommit_digest,
            underlying_transport=_forbidden_transport,
        )
        if (
            replayed_run.artifact != run.artifact
            or replayed_run.journal_summary != run.journal_summary
            or replayed_run.failure_evidence != run.failure_evidence
            or replayed_run.fresh_call_count != 0
            or replayed_run.reused_call_count != 1
        ):
            raise ObjectBongardPanelRubricCalibrationCommandError(
                "cold journal replay differs from frozen batch"
            )
        summary = replayed_run.journal_summary
        if summary.record_digest is None:
            raise ObjectBongardPanelRubricCalibrationCommandError(
                "terminal journal summary lacks a record digest"
            )
        journal_digests.append(summary.record_digest)
    return tuple(journal_digests)


def _replay_record(
    plan: ObjectBongardPanelRubricCalibrationPlan,
    frozen: ObjectBongardPanelRubricCalibrationDurableFreeze,
    assessment: ObjectBongardPanelRubricCalibrationAssessment,
    *,
    authorization_digest: str,
    execution_precommit_digest: str,
    journal_summary_digests: Sequence[str],
) -> dict[str, Any]:
    replayed = cold_verify_object_bongard_panel_rubric_calibration(
        assessment, plan, frozen
    )
    if replayed != assessment:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "core calibration replay differs"
        )
    if len(journal_summary_digests) != PANEL_RUBRIC_CALIBRATION_JOB_COUNT:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "cold journal replay count differs"
        )
    return _record(
        {
            "schema": CALIBRATION_REPLAY_SCHEMA,
            "command_schema": PANEL_RUBRIC_CALIBRATION_COMMAND_SCHEMA,
            "authorization_digest": authorization_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "plan_digest": plan.plan_digest,
            "source_digest": plan.source.source_digest,
            "batch_digest": frozen.batch.batch_digest,
            "freeze_digest": frozen.freeze_digest,
            "assessment_digest": assessment.assessment_digest,
            "journal_summary_digests": list(journal_summary_digests),
            "journal_replay_count": PANEL_RUBRIC_CALIBRATION_JOB_COUNT,
            "artifact_replay_count": PANEL_RUBRIC_CALIBRATION_JOB_COUNT,
            "model_calls_during_replay": 0,
            "transport_forbidden_during_replay": True,
            "support_projection_recomputed": True,
            "slate_selection_recomputed": True,
            "query_pixels_opened": False,
            "broad_cohort_pixels_opened": False,
            "official_test_pixels_opened": False,
            **_authority_data(),
        }
    )


def _result_record(
    plan: ObjectBongardPanelRubricCalibrationPlan,
    frozen: ObjectBongardPanelRubricCalibrationDurableFreeze,
    assessment: ObjectBongardPanelRubricCalibrationAssessment,
    replay: Mapping[str, Any],
    *,
    authorization_digest: str,
    execution_precommit_digest: str,
) -> dict[str, Any]:
    selection = assessment.slate_selection
    selected_rank = (
        None
        if selection.selected_rubric_spec is None
        else selection.selected_rubric_spec.candidate_rank
    )
    return _record(
        {
            "schema": CALIBRATION_RESULT_SCHEMA,
            "command_schema": PANEL_RUBRIC_CALIBRATION_COMMAND_SCHEMA,
            "authorization_digest": authorization_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "plan_digest": plan.plan_digest,
            "source_digest": plan.source.source_digest,
            "batch_digest": frozen.batch.batch_digest,
            "freeze_digest": frozen.freeze_digest,
            "assessment_digest": assessment.assessment_digest,
            "cold_replay_digest": replay["record_digest"],
            "accepted": selection.selected_candidate_digest is not None,
            "selected_candidate_rank": selected_rank,
            "selected_candidate_digest": selection.selected_candidate_digest,
            "fresh_call_count": frozen.batch.fresh_call_count,
            "reused_call_count": frozen.batch.reused_call_count,
            "physical_call_denominator": PANEL_RUBRIC_CALIBRATION_JOB_COUNT,
            "all_24_artifacts_frozen_before_support_labels": True,
            "model_calls_during_assessment_or_replay": 0,
            "query_pixels_opened": False,
            "broad_cohort_pixels_opened": False,
            "official_test_pixels_opened": False,
            **_authority_data(),
        }
    )


def _verification(
    root: Path,
    authorization: Mapping[str, Any],
    precommit: _FreshRuntimePrecommit,
    plan: ObjectBongardPanelRubricCalibrationPlan,
    frozen: ObjectBongardPanelRubricCalibrationDurableFreeze,
    assessment: ObjectBongardPanelRubricCalibrationAssessment,
    replay: Mapping[str, Any],
    result: Mapping[str, Any],
) -> VerifiedObjectBongardPanelRubricCalibration:
    selected = assessment.slate_selection.selected_rubric_spec
    selected_rank = None if selected is None else selected.candidate_rank
    accepted = assessment.slate_selection.selected_candidate_digest is not None
    if (
        result.get("accepted") is not accepted
        or result.get("selected_candidate_rank") != selected_rank
        or result.get("fresh_call_count") != frozen.batch.fresh_call_count
        or result.get("reused_call_count") != frozen.batch.reused_call_count
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "result summary differs from replayed assessment"
        )
    return VerifiedObjectBongardPanelRubricCalibration(
        root,
        authorization["record_digest"],
        precommit.precommit_digest,
        plan.plan_digest,
        frozen.batch.batch_digest,
        frozen.freeze_digest,
        assessment.assessment_digest,
        replay["record_digest"],
        result["record_digest"],
        accepted,
        selected_rank,
        frozen.batch.fresh_call_count,
        frozen.batch.reused_call_count,
    )


def _load_plan_inputs(
    *,
    nomination_root: str | os.PathLike[str],
    source_directory: str | os.PathLike[str],
) -> tuple[ObjectBongardPanelRubricCalibrationPlan, _PinnedV10Nomination]:
    source = load_object_bongard_panel_rubric_calibration_source(source_directory)
    nomination = _load_pinned_v10_nomination(nomination_root, source)
    plan = bind_object_bongard_panel_rubric_calibration_nomination(
        source, nomination
    )
    return plan, nomination


def _run_loaded_calibration(
    output_root: str | os.PathLike[str],
    plan: ObjectBongardPanelRubricCalibrationPlan,
    nomination: _PinnedV10Nomination,
    *,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    minutes: int = DEFAULT_MINUTES,
    verbose: bool = False,
    executable: str = DEFAULT_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_EXPECTED_LAUNCHER_SHA256,
    transport: Transport = run_codex_named_images_structured,
    cloud_policy_cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot] = (
        snapshot_cloud_policy_cache
    ),
    model_catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot] = (
        snapshot_pinned_model_catalog
    ),
    launcher_fingerprinter: Callable[..., Mapping[str, str]] = (
        codex_cli_authenticated_fingerprint
    ),
    runtime_attester: Callable[..., CodexNoToolsAttestation] = attest_codex_no_tools,
) -> VerifiedObjectBongardPanelRubricCalibration:
    root = _fresh_output_root(output_root)
    _write_once(root / PLAN_FILENAME, plan.to_data(), "calibration plan")
    if _read_json(root / PLAN_FILENAME, "calibration plan") != plan.to_data():
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "persisted calibration plan differs"
        )
    authorization, precommit = _seal_fresh_runtime(
        root,
        plan,
        nomination,
        parallel_workers=parallel_workers,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
        cloud_policy_cache_snapshotter=cloud_policy_cache_snapshotter,
        model_catalog_snapshotter=model_catalog_snapshotter,
        launcher_fingerprinter=launcher_fingerprinter,
        runtime_attester=runtime_attester,
    )
    batch = run_object_bongard_panel_rubric_calibration_observations(
        plan,
        runtime=precommit.runtime,
        journal_root=root / JOURNAL_DIRECTORY,
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit.precommit_digest,
        parallel_workers=parallel_workers,
        underlying_transport=transport,
    )
    if (
        batch.fresh_call_count != PANEL_RUBRIC_CALIBRATION_JOB_COUNT
        or batch.reused_call_count != 0
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "fresh calibration did not execute exactly 24 new calls"
        )
    frozen = persist_and_reload_object_bongard_panel_rubric_calibration_batch(
        batch, root / BATCH_FILENAME
    )
    _write_once(root / FREEZE_FILENAME, frozen.to_data(), "durable freeze")
    restored_freeze = ObjectBongardPanelRubricCalibrationDurableFreeze.from_data(
        _read_json(root / FREEZE_FILENAME, "durable freeze")
    )
    if restored_freeze != frozen:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "persisted durable freeze differs"
        )
    assessment = assess_object_bongard_panel_rubric_calibration(plan, restored_freeze)
    _write_once(root / ASSESSMENT_FILENAME, assessment.to_data(), "assessment")
    restored_assessment = ObjectBongardPanelRubricCalibrationAssessment.from_data(
        _read_json(root / ASSESSMENT_FILENAME, "assessment")
    )
    journal_digests = _verify_all_journals(
        root,
        plan,
        restored_freeze.batch,
        runtime=precommit.runtime,
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit.precommit_digest,
    )
    replay = _replay_record(
        plan,
        restored_freeze,
        restored_assessment,
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit.precommit_digest,
        journal_summary_digests=journal_digests,
    )
    _write_once(root / REPLAY_FILENAME, replay, "cold replay")
    result = _result_record(
        plan,
        restored_freeze,
        restored_assessment,
        replay,
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit.precommit_digest,
    )
    _write_once(root / RESULT_FILENAME, result, "calibration result")
    return _verify_loaded_calibration(root, plan, nomination)


def run_object_bongard_panel_rubric_calibration(
    output_root: str | os.PathLike[str],
    *,
    nomination_root: str | os.PathLike[str] = DEFAULT_V10_NOMINATION_ROOT,
    source_directory: str | os.PathLike[str] = (
        DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    ),
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    minutes: int = DEFAULT_MINUTES,
    verbose: bool = False,
    executable: str = DEFAULT_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_EXPECTED_LAUNCHER_SHA256,
    transport: Transport = run_codex_named_images_structured,
) -> VerifiedObjectBongardPanelRubricCalibration:
    """Launch one immutable 24-call calibration in a fresh output root."""

    plan, nomination = _load_plan_inputs(
        nomination_root=nomination_root,
        source_directory=source_directory,
    )
    return _run_loaded_calibration(
        output_root,
        plan,
        nomination,
        parallel_workers=parallel_workers,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
        transport=transport,
    )


def _verify_loaded_calibration(
    output_root: str | os.PathLike[str],
    plan: ObjectBongardPanelRubricCalibrationPlan,
    nomination: _PinnedV10Nomination,
) -> VerifiedObjectBongardPanelRubricCalibration:
    root = _existing_output_root(output_root)
    expected_inventory = {
        AUTHORIZATION_FILENAME,
        PRECOMMIT_FILENAME,
        PLAN_FILENAME,
        BATCH_FILENAME,
        FREEZE_FILENAME,
        ASSESSMENT_FILENAME,
        REPLAY_FILENAME,
        RESULT_FILENAME,
        JOURNAL_DIRECTORY,
    }
    if {item.name for item in root.iterdir()} != expected_inventory:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "calibration root inventory differs"
        )
    journal_root = root / JOURNAL_DIRECTORY
    if not journal_root.is_dir() or journal_root.is_symlink():
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "journal root is not a real directory"
        )
    if _read_json(root / PLAN_FILENAME, "calibration plan") != plan.to_data():
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "calibration plan differs on replay"
        )
    authorization = _verify_record(
        _read_json(root / AUTHORIZATION_FILENAME, "authorization"),
        schema=CALIBRATION_AUTHORIZATION_SCHEMA,
        label="authorization",
    )
    policy = authorization.get("runtime_policy")
    if not isinstance(policy, Mapping):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "authorization runtime policy is malformed"
        )
    expected_authorization = _authorization(
        plan,
        nomination,
        parallel_workers=authorization.get("parallel_workers"),
        minutes=policy.get("minutes"),
        verbose=policy.get("verbose"),
        executable=policy.get("executable"),
        expected_launcher_sha256=policy.get("expected_launcher_sha256"),
    )
    if authorization != expected_authorization:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "authorization differs on replay"
        )
    precommit_record = _verify_record(
        _read_json(root / PRECOMMIT_FILENAME, "execution precommit"),
        schema=CALIBRATION_PRECOMMIT_SCHEMA,
        label="execution precommit",
    )
    runtime = _runtime_from_precommit(
        precommit_record,
        expected_authorization_digest=authorization["record_digest"],
    )
    precommit = _FreshRuntimePrecommit(precommit_record, runtime)
    batch = ObjectBongardPanelRubricCalibrationObservationBatch.from_data(
        _read_json(root / BATCH_FILENAME, "observation batch")
    )
    if (
        batch.plan_digest != plan.plan_digest
        or batch.source_digest != plan.source.source_digest
        or batch.authorization_digest != authorization["record_digest"]
        or batch.execution_precommit_digest != precommit.precommit_digest
        or batch.parallel_workers != authorization["parallel_workers"]
        or batch.fresh_call_count != PANEL_RUBRIC_CALIBRATION_JOB_COUNT
        or batch.reused_call_count != 0
    ):
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "frozen observation batch command binding differs"
        )
    frozen = ObjectBongardPanelRubricCalibrationDurableFreeze.from_data(
        _read_json(root / FREEZE_FILENAME, "durable freeze")
    )
    if frozen.batch != batch:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "durable freeze batch differs from the standalone batch"
        )
    assessment = ObjectBongardPanelRubricCalibrationAssessment.from_data(
        _read_json(root / ASSESSMENT_FILENAME, "assessment")
    )
    journal_digests = _verify_all_journals(
        root,
        plan,
        batch,
        runtime=runtime,
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit.precommit_digest,
    )
    expected_replay = _replay_record(
        plan,
        frozen,
        assessment,
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit.precommit_digest,
        journal_summary_digests=journal_digests,
    )
    replay = _verify_record(
        _read_json(root / REPLAY_FILENAME, "cold replay"),
        schema=CALIBRATION_REPLAY_SCHEMA,
        label="cold replay",
    )
    if replay != expected_replay:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "cold replay record differs"
        )
    expected_result = _result_record(
        plan,
        frozen,
        assessment,
        replay,
        authorization_digest=authorization["record_digest"],
        execution_precommit_digest=precommit.precommit_digest,
    )
    result = _verify_record(
        _read_json(root / RESULT_FILENAME, "calibration result"),
        schema=CALIBRATION_RESULT_SCHEMA,
        label="calibration result",
    )
    if result != expected_result:
        raise ObjectBongardPanelRubricCalibrationCommandError(
            "calibration result differs"
        )
    return _verification(
        root,
        authorization,
        precommit,
        plan,
        frozen,
        assessment,
        replay,
        result,
    )


def verify_object_bongard_panel_rubric_calibration(
    output_root: str | os.PathLike[str],
    *,
    nomination_root: str | os.PathLike[str] = DEFAULT_V10_NOMINATION_ROOT,
    source_directory: str | os.PathLike[str] = (
        DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE
    ),
) -> VerifiedObjectBongardPanelRubricCalibration:
    """Cold replay a completed calibration without any model transport."""

    plan, nomination = _load_plan_inputs(
        nomination_root=nomination_root,
        source_directory=source_directory,
    )
    return _verify_loaded_calibration(output_root, plan, nomination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m bongard.object_bongard_panel_rubric_calibration_command",
        description="Launch or cold-verify the sealed 24-call panel calibration",
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("launch", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", required=True, type=Path)
        command.add_argument(
            "--nomination-root", type=Path, default=DEFAULT_V10_NOMINATION_ROOT
        )
        command.add_argument(
            "--source-directory",
            type=Path,
            default=DEFAULT_OBJECT_BONGARD_PANEL_RUBRIC_CALIBRATION_SOURCE,
        )
    launch = commands.choices["launch"]
    launch.add_argument(
        "--parallel-workers", type=int, default=DEFAULT_PARALLEL_WORKERS
    )
    launch.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    launch.add_argument("--verbose", action="store_true")
    launch.add_argument("--executable", default=DEFAULT_EXECUTABLE)
    launch.add_argument(
        "--expected-launcher-sha256",
        default=DEFAULT_EXPECTED_LAUNCHER_SHA256,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        common = {
            "nomination_root": args.nomination_root,
            "source_directory": args.source_directory,
        }
        if args.operation == "launch":
            verified = run_object_bongard_panel_rubric_calibration(
                args.output_root,
                parallel_workers=args.parallel_workers,
                minutes=args.minutes,
                verbose=args.verbose,
                executable=args.executable,
                expected_launcher_sha256=args.expected_launcher_sha256,
                **common,
            )
        else:
            verified = verify_object_bongard_panel_rubric_calibration(
                args.output_root, **common
            )
    except Exception as exc:
        try:
            prefix = str(exc).encode("utf-8", errors="replace")[:4096]
        except Exception:
            prefix = b""
        print(
            canonical_json(
                {
                    "schema": "gkm.bongard-panel-rubric-calibration-command-error.v1",
                    "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message_prefix_sha256": (
                        None if not prefix else hashlib.sha256(prefix).hexdigest()
                    ),
                    "raw_message_persisted": False,
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 1
    print(canonical_json(verified.summary_data()).decode("utf-8"))
    return 0 if verified.accepted else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "DEFAULT_EXPECTED_LAUNCHER_SHA256",
    "DEFAULT_PARALLEL_WORKERS",
    "DEFAULT_V10_NOMINATION_ROOT",
    "ObjectBongardPanelRubricCalibrationCommandError",
    "VerifiedObjectBongardPanelRubricCalibration",
    "main",
    "object_bongard_panel_rubric_calibration_command_source_digest",
    "run_object_bongard_panel_rubric_calibration",
    "verify_object_bongard_panel_rubric_calibration",
)
