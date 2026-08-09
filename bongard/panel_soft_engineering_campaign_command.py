"""Headless exact-unused TRAIN campaign for the panel-soft engineering lane.

This command is an engineering diagnostic, never an official benchmark.  Its
standalone task executor requires a durably reloaded release-authority record
before invoking any panel releaser.  It shows only the twelve support panels
to the proposer, journals every Codex turn globally by task/turn identity,
persists and reloads the exact Python predicate freeze, and releases query
pixels only from the runner's post-reload callback.  Lean is absent, optional,
and removable.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

import base64
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Callable, Mapping, Sequence, TextIO

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
    ObjectBongardReleaseStore,
    ObjectBongardWriteOnceReceipt,
    PreparedObjectBongardRelease,
    create_object_bongard_execution_precommit,
    prepare_object_bongard_release,
    release_object_bongard_support_panel,
    verify_prepared_object_bongard_release,
)
from bongard.object_bongard_turn_journal import (
    ObjectBongardNamedImageTurnJournalTransport,
    ObjectBongardTextTurnJournalTransport,
    ObjectBongardTurnJournalSummary,
    ObjectBongardTurnRuntime,
    TURN_JOURNAL_MANIFEST_SCHEMA,
    TURN_JOURNAL_SUMMARY_SCHEMA,
    object_bongard_turn_journal_source_digest,
)
from bongard.object_scene_anchor_source_manifest import (
    ObjectSceneAnchorSourceManifest,
    build_object_scene_anchor_source_manifest,
    cold_verify_object_scene_anchor_source_manifest,
    object_scene_anchor_source_manifest_source_digest,
)
from bongard.official_panel_archive import OfficialPanelArchive, ReleasedOfficialPanel
from bongard.panel_soft_engineering_task_runner import (
    PANEL_SOFT_ENGINEERING_PROPOSER_TERMINAL_SCHEMA,
    PANEL_SOFT_ENGINEERING_TASK_ARCHIVE_SCHEMA,
    PanelSoftEngineeringProposerTerminal,
    PanelSoftEngineeringTaskFreeze,
    PanelSoftEngineeringTaskFreezeCommit,
    PanelSoftEngineeringTaskRunArchive,
    PanelSoftEngineeringTaskRunStatus,
    PanelSoftEngineeringTaskRunnerError,
    cold_replay_panel_soft_engineering_task,
    panel_soft_engineering_task_runner_source_digest,
    run_panel_soft_engineering_task,
)
from bongard.panel_soft_observer import (
    PanelSoftObserverArtifact,
    aggregate_panel_soft_observer_artifacts,
    observe_panel_soft_vocabulary,
    panel_soft_observer_output_schema,
    panel_soft_observer_prompt,
    panel_soft_observer_source_digest,
)
from bongard.panel_soft_predicate import (
    PanelSoftEngineeringVersionSpace,
    panel_soft_predicate_source_digest,
)
from bongard.panel_soft_ranker import (
    PanelSoftRankArtifact,
    PanelSoftRankInput,
    PanelSoftRankTransportProvenance,
    panel_soft_rank_transport_provenance,
    panel_soft_ranker_output_schema,
    panel_soft_ranker_prompt,
    panel_soft_ranker_source_digest,
    rank_panel_soft_version_space,
    verify_panel_soft_rank_artifact,
)
from bongard.panel_soft_proposer import (
    PANEL_SOFT_PROPOSER_PRESENTATION_NAMES,
    PanelSoftProposerArtifact,
    panel_soft_proposer_output_schema,
    panel_soft_proposer_prompt,
    panel_soft_proposer_source_digest,
    propose_panel_soft_atoms,
    verify_panel_soft_proposer_artifact,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.prototype_scene_observer import prototype_scene_transport_source_digest
from bongard.release import OfficialReleaseDescriptor
from bongard.transport import (
    DEFAULT_CODEX_MODEL,
    PINNED_CODEX_CLI_VERSION,
    REASONING_EFFORTS,
    CloudPolicyCacheSnapshot,
    CodexModelCatalogSnapshot,
    CodexStructuredResult,
    codex_cli_authenticated_fingerprint,
    run_codex_named_images_structured,
    run_codex_text_structured,
    snapshot_cloud_policy_cache,
    snapshot_pinned_model_catalog,
)


PANEL_SOFT_CAMPAIGN_AUTHORITY_SCHEMA = (
    "gkm.bongard-panel-soft-campaign-release-authority.v1"
)
PANEL_SOFT_CAMPAIGN_TASK_SCHEMA = "gkm.bongard-panel-soft-campaign-task.v1"
PANEL_SOFT_CAMPAIGN_SCHEMA = "gkm.bongard-panel-soft-campaign.v1"
PANEL_SOFT_RUNTIME_EVIDENCE_SCHEMA = (
    "gkm.bongard-panel-soft-runtime-evidence.v1"
)
PANEL_SOFT_SUCCESSOR_MIRROR_SCHEMA = (
    "gkm.bongard-panel-soft-successor-mirror.v1"
)
PANEL_SOFT_CAMPAIGN_REPLAY_SCHEMA = (
    "gkm.bongard-panel-soft-campaign-replay-receipt.v1"
)
PANEL_SOFT_RANK_JOURNAL_EVIDENCE_SCHEMA = (
    "gkm.bongard-panel-soft-rank-journal-evidence.v1"
)
PANEL_SOFT_RANK_FAILURE_EVIDENCE_SCHEMA = (
    "gkm.bongard-panel-soft-rank-failure-evidence.v1"
)
PANEL_SOFT_RANK_TERMINAL_SCHEMA = (
    "gkm.bongard-panel-soft-rank-terminal.v1"
)
PANEL_SOFT_CAMPAIGN_COMPLETION_SUMMARY_SCHEMA = (
    "gkm.bongard-panel-soft-campaign-completion-summary.v1"
)
PANEL_SOFT_CAMPAIGN_REPLAY_SUMMARY_SCHEMA = (
    "gkm.bongard-panel-soft-campaign-replay-summary.v1"
)
PANEL_SOFT_CAMPAIGN_ID = "bongard.panel-soft/exact-unused-train-engineering-v1"
PANEL_SOFT_CAMPAIGN_MODULE_NAME = (
    "bongard.panel_soft_engineering_campaign_command"
)
DEFAULT_SELECTION_SEED = "panel-soft-exact-unused-train-20260809-v1"
DEFAULT_SELECTED_TASK_IDS = (
    "bd_open_s-exist_quadrangle_five_lines9_0000",
    "ff_nact8_0041",
    "hd_convex-has_four_straight_lines_0001",
)
DEFAULT_PLAN_DIGEST = (
    "sha256:b342bc829f20a9e825ced42e051ac12a05d470c5c19130977d5a069878a1bc88"
)
DEFAULT_PREDECESSOR_LEDGER_DIGEST = (
    "sha256:1ed439c5e40dca1da1cafd67171dc97caad191561c1f39664d8ea0a641447546"
)
DEFAULT_PREDECESSOR_FILE_SHA256 = (
    "452e80d26eae6182386f751019263e8a5e5f2d307151a1707a5353311de9f3ea"
)
PANEL_SOFT_CAMPAIGN_TASK_COUNT = 3
PANEL_SOFT_CAMPAIGN_QUERY_DENOMINATOR = 6
PANEL_SOFT_SELECTION_MODES = (
    "deterministic_baseline",
    "support_only_codex_ranker",
)
DEFAULT_CODEX_EXECUTABLE = "codex"
DEFAULT_CODEX_LAUNCHER_SHA256 = (
    "19c4f144c5226a9f17c58e6f0fa854843b0f77a6eb420f40e2745a12f10f5d37"
)
DEFAULT_MODEL = DEFAULT_CODEX_MODEL
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MINUTES = 15
DEFAULT_WORKERS = 3

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DESCRIPTOR = _ROOT / "bongard/data/shape_bongard_v2_release_v1.json"
DEFAULT_ARCHIVE = _ROOT / "downloads/ShapeBongard_V2.zip"
DEFAULT_SPLIT = _ROOT / "downloads/ShapeBongard_V2_full/ShapeBongard_V2/ShapeBongard_V2_split.json"
DEFAULT_PREDECESSOR = (
    _ROOT
    / "downloads/ShapeBongard_V2_full/object_scene_anchor_exact_unused_train_20260809_python_v1"
    / "research-exposure-successors"
    / "1ed439c5e40dca1da1cafd67171dc97caad191561c1f39664d8ea0a641447546.exposure.json"
)
DEFAULT_HISTORICAL_EXPOSURE = _ROOT / "bongard/data/historical_exposure_v1.json"

_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")


class PanelSoftEngineeringCampaignError(RuntimeError):
    """Campaign selection, chronology, persistence, or replay failed closed."""


def panel_soft_engineering_campaign_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _selection_mode(value: object) -> str:
    if not isinstance(value, str) or value not in PANEL_SOFT_SELECTION_MODES:
        raise PanelSoftEngineeringCampaignError(
            "predicate-pair selection mode is unsupported"
        )
    return value


def _worker_count(value: object) -> int:
    if type(value) is not int or not 1 <= value <= PANEL_SOFT_CAMPAIGN_TASK_COUNT:
        raise PanelSoftEngineeringCampaignError(
            "workers must be an exact integer in 1..3"
        )
    return value


def _authority_data(
    selection_mode: str = "support_only_codex_ranker",
) -> dict[str, object]:
    mode = _selection_mode(selection_mode)
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "evaluation_kind": "exact-unused-train-engineering-diagnostic",
        "engineering_only": True,
        "uncalibrated": True,
        "scientific_evidence": False,
        "official_benchmark_result": False,
        "official_test_authorized": False,
        "predicate_pair_selection_mode": mode,
        "support_only_codex_ranker_present": mode == "support_only_codex_ranker",
        "deterministic_selector_baseline": mode == "deterministic_baseline",
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_selection_decision_scoring_or_replay": False,
        "post_rank_custody_rejection_is_campaign_integrity_fatal": True,
    }


def _raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
        raise PanelSoftEngineeringCampaignError(f"{label} must be a raw SHA-256")
    return value


def _address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise PanelSoftEngineeringCampaignError(f"{label} must be a sha256: address")
    return value


def _content_address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value) or set(value) != expected:
        raise PanelSoftEngineeringCampaignError(f"{label} fields differ")
    return value


_TURN_JOURNAL_AUTHORITY = {
    "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
    "python_is_canonical_authority": True,
    "lean_present": False,
    "lean_required": False,
    "lean_removable": True,
    "lean_affects_identity_or_replay": False,
}


def _turn_journal_summary_from_data(
    value: object,
) -> ObjectBongardTurnJournalSummary:
    expected = {
        "schema", "manifest_digest", "turn_key", "terminal_status",
        "claim_digest", "result_digest", "outcome_digest", "record_digest",
        *_TURN_JOURNAL_AUTHORITY,
    }
    raw = _fields(value, expected, "turn journal summary")
    if (
        raw["schema"] != TURN_JOURNAL_SUMMARY_SCHEMA
        or any(
            type(raw[key]) is not type(item) or raw[key] != item
            for key, item in _TURN_JOURNAL_AUTHORITY.items()
        )
        or raw["terminal_status"] not in {"success", "failure", "unclaimed"}
    ):
        raise PanelSoftEngineeringCampaignError("turn journal summary policy differs")
    for key in ("manifest_digest", "turn_key", "record_digest"):
        _address(raw[key], f"turn journal {key}")
    for key in ("claim_digest", "result_digest", "outcome_digest"):
        if raw[key] is not None:
            _address(raw[key], f"turn journal {key}")
    terminal = raw["terminal_status"] in {"success", "failure"}
    if terminal != all(raw[key] is not None for key in (
        "claim_digest", "result_digest", "outcome_digest"
    )):
        raise PanelSoftEngineeringCampaignError(
            "turn journal terminal disposition differs"
        )
    content = {key: item for key, item in raw.items() if key != "record_digest"}
    if raw["record_digest"] != _content_address(content):
        raise PanelSoftEngineeringCampaignError("turn journal summary digest differs")
    result = ObjectBongardTurnJournalSummary(
        manifest_digest=raw["manifest_digest"],
        turn_key=raw["turn_key"],
        terminal_status=raw["terminal_status"],
        claim_digest=raw["claim_digest"],
        result_digest=raw["result_digest"],
        outcome_digest=raw["outcome_digest"],
        record_digest=raw["record_digest"],
    )
    if canonical_json(result.to_data()) != canonical_json(dict(raw)):
        raise PanelSoftEngineeringCampaignError(
            "turn journal summary is not canonical"
        )
    return result


def _stable_private_read(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.lstat(path)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PanelSoftEngineeringCampaignError("durable mirror is unavailable") from exc
    try:
        opened = os.fstat(descriptor)
        identity = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or opened.st_nlink != 1
            or not 0 < opened.st_size <= maximum_bytes
            or identity != (
                opened.st_dev, opened.st_ino, opened.st_size,
                opened.st_mtime_ns, opened.st_ctime_ns,
            )
        ):
            raise PanelSoftEngineeringCampaignError(
                "durable mirror is not a stable private file"
            )
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1_048_576))
            if not block:
                raise PanelSoftEngineeringCampaignError("durable mirror was truncated")
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        if after.st_nlink != 1 or identity != (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise PanelSoftEngineeringCampaignError(
                "durable mirror changed while reading"
            )
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_write_once(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= 16 * 1024 * 1024:
        raise PanelSoftEngineeringCampaignError("durable mirror payload is invalid")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.resolve(strict=True) != path.parent or not stat.S_ISDIR(
        os.lstat(path.parent).st_mode
    ):
        raise PanelSoftEngineeringCampaignError("durable mirror directory is unsafe")
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PanelSoftEngineeringCampaignError(
                    "durable mirror write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    if _stable_private_read(path, maximum_bytes=16 * 1024 * 1024) != payload:
        raise PanelSoftEngineeringCampaignError(
            "durable mirror collision or tamper"
        )


def _expected_store_receipt(
    *, object_kind: str, object_digest: str, data: Mapping[str, Any]
) -> ObjectBongardWriteOnceReceipt:
    payload = canonical_json(dict(data)) + b"\n"
    relative_path = f"objects/{object_kind}/{object_digest[7:]}.json"
    content = {
        "schema": "gkm.bongard-object-write-once-receipt.v1",
        "object_kind": object_kind,
        "object_digest": object_digest,
        "payload_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "relative_path": relative_path,
        "persisted_and_reloaded": True,
    }
    return ObjectBongardWriteOnceReceipt(
        object_kind=object_kind,
        object_digest=object_digest,
        payload_digest=content["payload_digest"],
        size_bytes=len(payload),
        relative_path=relative_path,
        record_digest=_content_address(content),
    )


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
        raise PanelSoftEngineeringCampaignError("official archive contains no task inventory")
    return tuple(sorted(tasks))


def _prepared_task(
    prepared: PreparedObjectBongardRelease,
    task_plan: ObjectBongardTaskPlan,
) -> ObjectBongardTaskPlan:
    """Return the exact canonical task sealed by a prepared release."""

    verify_prepared_object_bongard_release(prepared)
    task = ObjectBongardTaskPlan.from_data(task_plan.to_data())
    matches = tuple(
        item for item in prepared.plan.tasks if item.task_id == task.task_id
    )
    if len(matches) != 1 or matches[0] != task:
        raise PanelSoftEngineeringCampaignError(
            "task is not the exact task sealed by the prepared release"
        )
    return task


def _persist_mapping(
    store: ObjectBongardReleaseStore,
    *,
    kind: str,
    digest: str,
    data: Mapping[str, Any],
) -> ObjectBongardWriteOnceReceipt:
    receipt = store.persist(object_kind=kind, object_digest=digest, data=data)
    if dict(store.verify(receipt, expected_data=data)) != dict(data):
        raise PanelSoftEngineeringCampaignError("write-once reload differs")
    return receipt


def _persist_exposure_successor_mirror(
    *,
    store: ObjectBongardReleaseStore,
    successor: ExposureLedger,
) -> tuple[Path, Mapping[str, Any], ObjectBongardWriteOnceReceipt]:
    relative_path = (
        "research-exposure-successors/"
        + successor.digest.removeprefix("sha256:")
        + ".exposure.json"
    )
    path = store.root / relative_path
    payload = successor.to_json().encode("utf-8", errors="strict")
    _durable_write_once(path, payload)
    content: dict[str, object] = {
        "schema": PANEL_SOFT_SUCCESSOR_MIRROR_SCHEMA,
        "exposure_successor_digest": successor.digest,
        "mirror_relative_path": relative_path,
        "mirror_payload_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "mirror_size_bytes": len(payload),
        "write_protocol": "fsync-temp-link-fsync-directory-stable-private-read",
        "content_addressed": True,
        "write_once": True,
        "persisted_and_reloaded": True,
    }
    evidence = {**content, "record_digest": _content_address(content)}
    receipt = _persist_mapping(
        store,
        kind="panel-soft-successor-mirror",
        digest=evidence["record_digest"],  # type: ignore[arg-type]
        data=evidence,
    )
    _verify_exposure_successor_mirror(
        store=store,
        successor=successor,
        path=path,
        evidence=evidence,
        receipt=receipt,
    )
    return path, evidence, receipt


def _verify_exposure_successor_mirror(
    *,
    store: ObjectBongardReleaseStore,
    successor: ExposureLedger,
    path: Path,
    evidence: Mapping[str, Any],
    receipt: ObjectBongardWriteOnceReceipt,
) -> None:
    raw = _fields(
        evidence,
        {
            "schema", "exposure_successor_digest", "mirror_relative_path",
            "mirror_payload_digest", "mirror_size_bytes", "write_protocol",
            "content_addressed", "write_once", "persisted_and_reloaded",
            "record_digest",
        },
        "exposure successor mirror evidence",
    )
    payload = successor.to_json().encode("utf-8", errors="strict")
    relative_path = (
        "research-exposure-successors/"
        + successor.digest.removeprefix("sha256:")
        + ".exposure.json"
    )
    content = {key: item for key, item in raw.items() if key != "record_digest"}
    if (
        raw["schema"] != PANEL_SOFT_SUCCESSOR_MIRROR_SCHEMA
        or raw["exposure_successor_digest"] != successor.digest
        or raw["mirror_relative_path"] != relative_path
        or raw["mirror_payload_digest"]
        != "sha256:" + hashlib.sha256(payload).hexdigest()
        or raw["mirror_size_bytes"] != len(payload)
        or raw["write_protocol"]
        != "fsync-temp-link-fsync-directory-stable-private-read"
        or raw["content_addressed"] is not True
        or raw["write_once"] is not True
        or raw["persisted_and_reloaded"] is not True
        or raw["record_digest"] != _content_address(content)
        or path != store.root / relative_path
        or _stable_private_read(path, maximum_bytes=16 * 1024 * 1024) != payload
        or ExposureLedger.load(path) != successor
        or receipt.object_kind != "panel-soft-successor-mirror"
        or receipt.object_digest != raw["record_digest"]
        or dict(store.verify(receipt, expected_data=raw)) != dict(raw)
    ):
        raise PanelSoftEngineeringCampaignError(
            "research exposure successor durable replay differs"
        )


def _authority_content(value: "PanelSoftCampaignReleaseAuthority") -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_CAMPAIGN_AUTHORITY_SCHEMA,
        "batch_plan_digest": value.batch_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "exposure_predecessor_digest": value.exposure_predecessor_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "release_authorization_digest": value.release_authorization_digest,
        "exposure_successor_store_receipt_digest": value.exposure_successor_store_receipt_digest,
        "exposure_successor_persisted_and_reloaded_before_selected_pixel_read": True,
        "query_requires_exact_runner_freeze_reload": True,
        **_authority_data(value.selection_mode),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftCampaignReleaseAuthority:
    selection_mode: str
    batch_plan_digest: str
    execution_precommit_digest: str
    exposure_predecessor_digest: str
    exposure_successor_digest: str
    release_authorization_digest: str
    exposure_successor_store_receipt_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        _selection_mode(self.selection_mode)
        for name in (
            "batch_plan_digest", "execution_precommit_digest",
            "exposure_predecessor_digest", "exposure_successor_digest",
            "release_authorization_digest",
            "exposure_successor_store_receipt_digest", "record_digest",
        ):
            _address(getattr(self, name), name)
        if self.record_digest != _content_address(_authority_content(self)):
            raise PanelSoftEngineeringCampaignError("campaign release authority differs")

    @classmethod
    def seal(
        cls,
        *,
        selection_mode: str = "support_only_codex_ranker",
        batch_plan_digest: str,
        execution_precommit_digest: str,
        exposure_predecessor_digest: str,
        exposure_successor_digest: str,
        release_authorization_digest: str,
        exposure_successor_store_receipt_digest: str,
    ) -> "PanelSoftCampaignReleaseAuthority":
        values = dict(
            selection_mode=_selection_mode(selection_mode),
            batch_plan_digest=batch_plan_digest,
            execution_precommit_digest=execution_precommit_digest,
            exposure_predecessor_digest=exposure_predecessor_digest,
            exposure_successor_digest=exposure_successor_digest,
            release_authorization_digest=release_authorization_digest,
            exposure_successor_store_receipt_digest=exposure_successor_store_receipt_digest,
        )
        provisional = object.__new__(cls)
        for key, item in values.items():
            object.__setattr__(provisional, key, item)
        return cls(**values, record_digest=_content_address(_authority_content(provisional)))

    @classmethod
    def from_prepared_release(
        cls,
        prepared: PreparedObjectBongardRelease,
        *,
        selection_mode: str = "support_only_codex_ranker",
    ) -> "PanelSoftCampaignReleaseAuthority":
        verify_prepared_object_bongard_release(prepared)
        return cls.seal(
            selection_mode=selection_mode,
            batch_plan_digest=prepared.plan.record_digest,
            execution_precommit_digest=prepared.precommit.record_digest,
            exposure_predecessor_digest=prepared.predecessor.digest,
            exposure_successor_digest=prepared.successor.digest,
            release_authorization_digest=prepared.authorization.record_digest,
            exposure_successor_store_receipt_digest=prepared.exposure_receipt.record_digest,
        )

    def to_data(self) -> dict[str, object]:
        return {**_authority_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftCampaignReleaseAuthority":
        raw = _fields(value, set(_authority_content_fields()) | {"record_digest"}, "campaign authority")
        if (
            raw["schema"] != PANEL_SOFT_CAMPAIGN_AUTHORITY_SCHEMA
            or raw["exposure_successor_persisted_and_reloaded_before_selected_pixel_read"] is not True
            or raw["query_requires_exact_runner_freeze_reload"] is not True
            or any(
                type(raw[key]) is not type(item) or raw[key] != item
                for key, item in _authority_data(raw["predicate_pair_selection_mode"]).items()
            )
        ):
            raise PanelSoftEngineeringCampaignError("campaign authority policy differs")
        result = cls(
            raw["predicate_pair_selection_mode"],
            raw["batch_plan_digest"], raw["execution_precommit_digest"],
            raw["exposure_predecessor_digest"], raw["exposure_successor_digest"],
            raw["release_authorization_digest"],
            raw["exposure_successor_store_receipt_digest"], raw["record_digest"],
        )
        if canonical_json(result.to_data()) != canonical_json(dict(raw)):
            raise PanelSoftEngineeringCampaignError("campaign authority is not canonical")
        return result


def _authority_content_fields() -> tuple[str, ...]:
    return (
        "schema", "batch_plan_digest", "execution_precommit_digest",
        "exposure_predecessor_digest", "exposure_successor_digest",
        "release_authorization_digest", "exposure_successor_store_receipt_digest",
        "exposure_successor_persisted_and_reloaded_before_selected_pixel_read",
        "query_requires_exact_runner_freeze_reload", *_authority_data(),
    )


RunnerRecord = PanelSoftEngineeringTaskRunArchive | PanelSoftEngineeringProposerTerminal


def _rank_journal_evidence_content(
    value: "PanelSoftRankJournalEvidence",
) -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_RANK_JOURNAL_EVIDENCE_SCHEMA,
        "task_id": value.task_id,
        "authorization_digest": value.authorization_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "turn_kind": value.turn_kind,
        "rank_input_digest": value.rank_input_digest,
        "prompt_sha256": value.prompt_sha256,
        "output_schema_digest": value.output_schema_digest,
        "rank_artifact_digest": value.rank_artifact_digest,
        "rank_receipt_digest": value.rank_receipt_digest,
        "rank_thread_id": value.rank_thread_id,
        "transport_provenance": value.transport_provenance.to_data(),
        "journal_summary": value.journal_summary.to_data(),
        "rank_artifact_store_receipt_digest": (
            value.rank_artifact_store_receipt_digest
        ),
        "terminal_attempt_count": 1,
        "exact_prompt_input_schema_and_receipt_correlated": True,
        "persisted_and_reloaded_before_freeze_or_query": True,
        **_authority_data("support_only_codex_ranker"),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftRankJournalEvidence:
    task_id: str
    authorization_digest: str
    execution_precommit_digest: str
    turn_kind: str
    rank_input_digest: str
    prompt_sha256: str
    output_schema_digest: str
    rank_artifact_digest: str
    rank_receipt_digest: str
    rank_thread_id: str
    transport_provenance: PanelSoftRankTransportProvenance
    journal_summary: ObjectBongardTurnJournalSummary
    rank_artifact_store_receipt_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise PanelSoftEngineeringCampaignError("rank evidence task ID differs")
        for name in (
            "authorization_digest", "execution_precommit_digest",
            "output_schema_digest", "rank_artifact_store_receipt_digest",
            "record_digest",
        ):
            _address(getattr(self, name), name)
        for name in (
            "rank_input_digest", "prompt_sha256", "rank_artifact_digest",
            "rank_receipt_digest",
        ):
            _raw_digest(getattr(self, name), name)
        provenance = PanelSoftRankTransportProvenance.from_data(
            self.transport_provenance.to_data()
        )
        summary = _turn_journal_summary_from_data(self.journal_summary.to_data())
        if (
            self.turn_kind != "support-rank"
            or not isinstance(self.rank_thread_id, str)
            or not self.rank_thread_id
            or provenance != self.transport_provenance
            or provenance.kind != "production_exactly_once_journal"
            or provenance.benchmark_sealable is not True
            or summary != self.journal_summary
            or summary.terminal_status != "success"
            or self.record_digest
            != _content_address(_rank_journal_evidence_content(self))
        ):
            raise PanelSoftEngineeringCampaignError(
                "rank journal evidence differs"
            )

    @classmethod
    def seal(
        cls,
        *,
        task_id: str,
        authorization_digest: str,
        execution_precommit_digest: str,
        rank_input: PanelSoftRankInput,
        prompt: str,
        output_schema: Mapping[str, Any],
        artifact: PanelSoftRankArtifact,
        transport_provenance: PanelSoftRankTransportProvenance,
        journal_summary: ObjectBongardTurnJournalSummary,
        rank_artifact_store_receipt: ObjectBongardWriteOnceReceipt,
    ) -> "PanelSoftRankJournalEvidence":
        values = {
            "task_id": task_id,
            "authorization_digest": authorization_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "turn_kind": "support-rank",
            "rank_input_digest": rank_input.rank_input_digest,
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8", errors="strict")
            ).hexdigest(),
            "output_schema_digest": _content_address(dict(output_schema)),
            "rank_artifact_digest": artifact.artifact_digest,
            "rank_receipt_digest": artifact.receipt.receipt_digest,
            "rank_thread_id": artifact.receipt.thread_id,
            "transport_provenance": transport_provenance,
            "journal_summary": journal_summary,
            "rank_artifact_store_receipt_digest": (
                rank_artifact_store_receipt.record_digest
            ),
        }
        provisional = object.__new__(cls)
        for key, item in values.items():
            object.__setattr__(provisional, key, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_content_address(
                _rank_journal_evidence_content(provisional)
            ),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_rank_journal_evidence_content(self),
            "record_digest": self.record_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftRankJournalEvidence":
        raw = _fields(
            value,
            set(_rank_journal_evidence_content_fields()) | {"record_digest"},
            "rank journal evidence",
        )
        if (
            raw["schema"] != PANEL_SOFT_RANK_JOURNAL_EVIDENCE_SCHEMA
            or type(raw["terminal_attempt_count"]) is not int
            or raw["terminal_attempt_count"] != 1
            or raw["exact_prompt_input_schema_and_receipt_correlated"] is not True
            or raw["persisted_and_reloaded_before_freeze_or_query"] is not True
            or any(
                type(raw[key]) is not type(item) or raw[key] != item
                for key, item in _authority_data(
                    "support_only_codex_ranker"
                ).items()
            )
        ):
            raise PanelSoftEngineeringCampaignError(
                "rank journal evidence policy differs"
            )
        result = cls(
            raw["task_id"],
            raw["authorization_digest"],
            raw["execution_precommit_digest"],
            raw["turn_kind"],
            raw["rank_input_digest"],
            raw["prompt_sha256"],
            raw["output_schema_digest"],
            raw["rank_artifact_digest"],
            raw["rank_receipt_digest"],
            raw["rank_thread_id"],
            PanelSoftRankTransportProvenance.from_data(
                raw["transport_provenance"]
            ),
            _turn_journal_summary_from_data(raw["journal_summary"]),
            raw["rank_artifact_store_receipt_digest"],
            raw["record_digest"],
        )
        if canonical_json(result.to_data()) != canonical_json(dict(raw)):
            raise PanelSoftEngineeringCampaignError(
                "rank journal evidence is not canonical"
            )
        return result


def _rank_journal_evidence_content_fields() -> tuple[str, ...]:
    return (
        "schema", "task_id", "authorization_digest",
        "execution_precommit_digest", "turn_kind", "rank_input_digest",
        "prompt_sha256", "output_schema_digest", "rank_artifact_digest",
        "rank_receipt_digest", "rank_thread_id", "transport_provenance",
        "journal_summary", "rank_artifact_store_receipt_digest",
        "terminal_attempt_count",
        "exact_prompt_input_schema_and_receipt_correlated",
        "persisted_and_reloaded_before_freeze_or_query",
        *_authority_data("support_only_codex_ranker"),
    )


def _rank_terminal_journal_successful_identity(
    journal: ObjectBongardTextTurnJournalTransport,
) -> tuple[str, str] | None:
    """Return the receipt identity authenticated by a terminal rank journal."""

    summary = journal.verify()
    result = _read_canonical_durable_mapping(
        journal.result_path, "rank terminal durable result"
    )
    if summary.terminal_status == "failure":
        if (
            result.get("status") != "failure"
            or result.get("codex_structured_result") is not None
            or result.get("receipt_digest") is not None
        ):
            raise PanelSoftEngineeringCampaignError(
                "failed rank journal contains successful receipt material"
            )
        return None
    if summary.terminal_status != "success" or result.get("status") != "success":
        raise PanelSoftEngineeringCampaignError(
            "rank journal is not a typed terminal"
        )
    structured = result.get("codex_structured_result")
    receipt = (
        structured.get("receipt") if isinstance(structured, Mapping) else None
    )
    receipt_digest = result.get("receipt_digest")
    thread_id = receipt.get("thread_id") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(receipt, Mapping)
        or not isinstance(receipt_digest, str)
        or _RAW_DIGEST.fullmatch(receipt_digest) is None
        or receipt.get("receipt_digest") != receipt_digest
        or not isinstance(thread_id, str)
        or not thread_id
    ):
        raise PanelSoftEngineeringCampaignError(
            "successful rank journal receipt identity differs"
        )
    return receipt_digest, thread_id


def _rank_failure_evidence_content(
    value: "PanelSoftRankFailureEvidence",
) -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_RANK_FAILURE_EVIDENCE_SCHEMA,
        "task_id": value.task_id,
        "authorization_digest": value.authorization_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "turn_kind": "support-rank",
        "rank_input_digest": value.rank_input_digest,
        "prompt_sha256": value.prompt_sha256,
        "output_schema_digest": value.output_schema_digest,
        "transport_provenance": value.transport_provenance.to_data(),
        "journal_summary": value.journal_summary.to_data(),
        "failure_disposition": value.failure_disposition,
        "source_exception_type": value.source_exception_type,
        "successful_call_identity": (
            None
            if value.successful_call_identity is None
            else list(value.successful_call_identity)
        ),
        "terminal_attempt_count": 1,
        "no_formula_selected": True,
        "no_baseline_fallback": True,
        "query_release_authorized": False,
        "persisted_and_reloaded_before_task_terminal": True,
        **_authority_data("support_only_codex_ranker"),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftRankFailureEvidence:
    task_id: str
    authorization_digest: str
    execution_precommit_digest: str
    rank_input_digest: str
    prompt_sha256: str
    output_schema_digest: str
    transport_provenance: PanelSoftRankTransportProvenance
    journal_summary: ObjectBongardTurnJournalSummary
    failure_disposition: str
    source_exception_type: str
    successful_call_identity: tuple[str, str] | None
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise PanelSoftEngineeringCampaignError(
                "rank failure task ID differs"
            )
        for name in (
            "authorization_digest", "execution_precommit_digest",
            "output_schema_digest", "record_digest",
        ):
            _address(getattr(self, name), name)
        for name in ("rank_input_digest", "prompt_sha256"):
            _raw_digest(getattr(self, name), name)
        provenance = PanelSoftRankTransportProvenance.from_data(
            self.transport_provenance.to_data()
        )
        summary = _turn_journal_summary_from_data(self.journal_summary.to_data())
        if (
            provenance.kind != "production_exactly_once_journal"
            or provenance.benchmark_sealable is not True
            or summary != self.journal_summary
            or summary.terminal_status not in {"success", "failure"}
            or self.failure_disposition
            not in {"transport_failure", "invalid_rank_result"}
            or (summary.terminal_status == "failure")
            != (self.failure_disposition == "transport_failure")
            or not isinstance(self.source_exception_type, str)
            or re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_.]{0,255}",
                self.source_exception_type,
            )
            is None
        ):
            raise PanelSoftEngineeringCampaignError(
                "rank failure evidence disposition differs"
            )
        if self.successful_call_identity is not None:
            if (
                type(self.successful_call_identity) is not tuple
                or len(self.successful_call_identity) != 2
                or _RAW_DIGEST.fullmatch(self.successful_call_identity[0]) is None
                or not isinstance(self.successful_call_identity[1], str)
                or not self.successful_call_identity[1]
                or summary.terminal_status != "success"
            ):
                raise PanelSoftEngineeringCampaignError(
                    "rank failure successful-call identity differs"
                )
        elif summary.terminal_status == "success":
            raise PanelSoftEngineeringCampaignError(
                "successful rank journal lacks its call identity"
            )
        if self.record_digest != _content_address(
            _rank_failure_evidence_content(self)
        ):
            raise PanelSoftEngineeringCampaignError(
                "rank failure evidence digest differs"
            )

    @classmethod
    def seal(
        cls,
        *,
        task_id: str,
        authorization_digest: str,
        execution_precommit_digest: str,
        rank_input: PanelSoftRankInput,
        prompt: str,
        output_schema: Mapping[str, Any],
        transport_provenance: PanelSoftRankTransportProvenance,
        journal: ObjectBongardTextTurnJournalTransport,
        journal_summary: ObjectBongardTurnJournalSummary,
        source_exception: Exception,
    ) -> "PanelSoftRankFailureEvidence":
        identity = _rank_terminal_journal_successful_identity(journal)
        if (identity is not None) != (journal_summary.terminal_status == "success"):
            raise PanelSoftEngineeringCampaignError(
                "rank failure journal identity and terminal status differ"
            )
        values = {
            "task_id": task_id,
            "authorization_digest": authorization_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "rank_input_digest": rank_input.rank_input_digest,
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8", errors="strict")
            ).hexdigest(),
            "output_schema_digest": _content_address(dict(output_schema)),
            "transport_provenance": transport_provenance,
            "journal_summary": journal_summary,
            "failure_disposition": (
                "transport_failure"
                if journal_summary.terminal_status == "failure"
                else "invalid_rank_result"
            ),
            "source_exception_type": (
                f"{type(source_exception).__module__}."
                f"{type(source_exception).__qualname__}"
            ),
            "successful_call_identity": identity,
        }
        provisional = object.__new__(cls)
        for key, item in values.items():
            object.__setattr__(provisional, key, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_content_address(
                _rank_failure_evidence_content(provisional)
            ),
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_rank_failure_evidence_content(self),
            "record_digest": self.record_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftRankFailureEvidence":
        raw = _fields(
            value,
            set(_rank_failure_evidence_content_fields()) | {"record_digest"},
            "rank failure evidence",
        )
        if (
            raw["schema"] != PANEL_SOFT_RANK_FAILURE_EVIDENCE_SCHEMA
            or type(raw["terminal_attempt_count"]) is not int
            or raw["terminal_attempt_count"] != 1
            or raw["no_formula_selected"] is not True
            or raw["no_baseline_fallback"] is not True
            or raw["query_release_authorized"] is not False
            or raw["persisted_and_reloaded_before_task_terminal"] is not True
            or any(
                type(raw[key]) is not type(item) or raw[key] != item
                for key, item in _authority_data(
                    "support_only_codex_ranker"
                ).items()
            )
        ):
            raise PanelSoftEngineeringCampaignError(
                "rank failure evidence policy differs"
            )
        result = cls(
            raw["task_id"],
            raw["authorization_digest"],
            raw["execution_precommit_digest"],
            raw["rank_input_digest"],
            raw["prompt_sha256"],
            raw["output_schema_digest"],
            PanelSoftRankTransportProvenance.from_data(
                raw["transport_provenance"]
            ),
            _turn_journal_summary_from_data(raw["journal_summary"]),
            raw["failure_disposition"],
            raw["source_exception_type"],
            (
                None
                if raw["successful_call_identity"] is None
                else tuple(raw["successful_call_identity"])
            ),
            raw["record_digest"],
        )
        if canonical_json(result.to_data()) != canonical_json(dict(raw)):
            raise PanelSoftEngineeringCampaignError(
                "rank failure evidence is not canonical"
            )
        return result


def _rank_failure_evidence_content_fields() -> tuple[str, ...]:
    return (
        "schema", "task_id", "authorization_digest",
        "execution_precommit_digest", "turn_kind", "rank_input_digest",
        "prompt_sha256", "output_schema_digest", "transport_provenance",
        "journal_summary", "failure_disposition", "source_exception_type",
        "successful_call_identity", "terminal_attempt_count",
        "no_formula_selected", "no_baseline_fallback",
        "query_release_authorized",
        "persisted_and_reloaded_before_task_terminal",
        *_authority_data("support_only_codex_ranker"),
    )


def _rank_terminal_content(
    value: "PanelSoftEngineeringRankTerminal",
) -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_RANK_TERMINAL_SCHEMA,
        "campaign_source_digest": value.campaign_source_digest,
        "task_plan": value.task_plan.to_data(),
        "execution_precommit_digest": value.execution_precommit_digest,
        "selection_mode": "support_only_codex_ranker",
        "proposer_artifact": value.proposer_artifact.to_data(),
        "support_png_base64_by_panel_id": dict(
            value.support_png_base64_by_panel_id
        ),
        "support_artifacts": [item.to_data() for item in value.support_artifacts],
        "engineering_version_space": value.engineering_version_space.to_data(),
        "rank_failure_evidence": value.rank_failure_evidence.to_data(),
        "rank_failure_evidence_store_receipt": (
            value.rank_failure_evidence_store_receipt.to_data()
        ),
        "status": "rank_error",
        "correct_count": 0,
        "determinate_count": 0,
        "abstain_count": 0,
        "error_count": 2,
        "query_denominator": 2,
        "query_release_count": 0,
        "ranker_callback_invocations": 1,
        "allow_unverified_rank_artifact": False,
        "rank_artifact": None,
        "no_baseline_fallback": True,
        "cold_replay_model_calls": 0,
        **_authority_data("support_only_codex_ranker"),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringRankTerminal:
    campaign_source_digest: str
    task_plan: ObjectBongardTaskPlan
    execution_precommit_digest: str
    proposer_artifact: PanelSoftProposerArtifact
    support_png_base64_by_panel_id: tuple[tuple[str, str], ...]
    support_artifacts: tuple[PanelSoftObserverArtifact, ...]
    engineering_version_space: PanelSoftEngineeringVersionSpace
    rank_failure_evidence: PanelSoftRankFailureEvidence
    rank_failure_evidence_store_receipt: ObjectBongardWriteOnceReceipt
    record_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.campaign_source_digest, "rank terminal source digest")
        _address(self.execution_precommit_digest, "rank terminal precommit")
        _raw_digest(self.record_digest, "rank terminal record digest")
        task = ObjectBongardTaskPlan.from_data(self.task_plan.to_data())
        proposer = PanelSoftProposerArtifact.from_data(
            self.proposer_artifact.to_data()
        )
        artifacts = tuple(
            PanelSoftObserverArtifact.from_data(item.to_data())
            for item in self.support_artifacts
        )
        space = PanelSoftEngineeringVersionSpace.from_data(
            self.engineering_version_space.to_data()
        )
        evidence = PanelSoftRankFailureEvidence.from_data(
            self.rank_failure_evidence.to_data()
        )
        support_ids = (
            *task.side_0_support_panel_ids,
            *task.side_1_support_panel_ids,
        )
        if (
            type(self.support_png_base64_by_panel_id) is not tuple
            or tuple(item[0] for item in self.support_png_base64_by_panel_id)
            != support_ids
        ):
            raise PanelSoftEngineeringCampaignError(
                "rank terminal support PNG inventory differs"
            )
        try:
            support_pngs = tuple(
                base64.b64decode(item[1], validate=True)
                for item in self.support_png_base64_by_panel_id
            )
        except (TypeError, ValueError) as exc:
            raise PanelSoftEngineeringCampaignError(
                "rank terminal support PNG encoding differs"
            ) from exc
        if proposer.vocabulary is None:
            raise PanelSoftEngineeringCampaignError(
                "rank terminal lacks a successful proposer vocabulary"
            )
        verify_panel_soft_proposer_artifact(
            proposer,
            support_pngs,
            support_panel_ids=support_ids,
            expected_artifact_digest=proposer.artifact_digest,
        )
        if len(artifacts) != 12:
            raise PanelSoftEngineeringCampaignError(
                "rank terminal support artifact count differs"
            )
        table = aggregate_panel_soft_observer_artifacts(
            artifacts,
            ordered_panel_commitments=tuple(
                (item.panel_id, item.panel_png_digest) for item in artifacts
            ),
            expected_vocabulary=proposer.vocabulary,
            expected_contract=artifacts[0].contract,
        )
        rebuilt_space = PanelSoftEngineeringVersionSpace.create(
            table,
            task.side_0_support_panel_ids,
            task.side_1_support_panel_ids,
        )
        if (
            task != self.task_plan
            or proposer != self.proposer_artifact
            or artifacts != self.support_artifacts
            or space != self.engineering_version_space
            or rebuilt_space != space
            or evidence != self.rank_failure_evidence
            or evidence.task_id != task.task_id
            or evidence.execution_precommit_digest
            != self.execution_precommit_digest
            or evidence.rank_input_digest
            != PanelSoftRankInput.freeze(space).rank_input_digest
            or self.rank_failure_evidence_store_receipt.object_kind
            != "panel-soft-rank-failure-evidence"
            or self.rank_failure_evidence_store_receipt.object_digest
            != evidence.record_digest
            or self.record_digest != canonical_digest(_rank_terminal_content(self))
        ):
            raise PanelSoftEngineeringCampaignError(
                "rank terminal custody differs"
            )

    @classmethod
    def seal(
        cls,
        *,
        task_plan: ObjectBongardTaskPlan,
        execution_precommit_digest: str,
        proposer_artifact: PanelSoftProposerArtifact,
        support_png_by_panel_id: Mapping[str, bytes],
        support_artifacts: Sequence[PanelSoftObserverArtifact],
        engineering_version_space: PanelSoftEngineeringVersionSpace,
        rank_failure_evidence: PanelSoftRankFailureEvidence,
        rank_failure_evidence_store_receipt: ObjectBongardWriteOnceReceipt,
    ) -> "PanelSoftEngineeringRankTerminal":
        task = ObjectBongardTaskPlan.from_data(task_plan.to_data())
        support_ids = (
            *task.side_0_support_panel_ids,
            *task.side_1_support_panel_ids,
        )
        values = {
            "campaign_source_digest": (
                panel_soft_engineering_campaign_source_digest()
            ),
            "task_plan": task,
            "execution_precommit_digest": execution_precommit_digest,
            "proposer_artifact": proposer_artifact,
            "support_png_base64_by_panel_id": tuple(
                (
                    panel_id,
                    base64.b64encode(support_png_by_panel_id[panel_id]).decode(
                        "ascii"
                    ),
                )
                for panel_id in support_ids
            ),
            "support_artifacts": tuple(support_artifacts),
            "engineering_version_space": engineering_version_space,
            "rank_failure_evidence": rank_failure_evidence,
            "rank_failure_evidence_store_receipt": (
                rank_failure_evidence_store_receipt
            ),
        }
        provisional = object.__new__(cls)
        for key, item in values.items():
            object.__setattr__(provisional, key, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=canonical_digest(_rank_terminal_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_rank_terminal_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftEngineeringRankTerminal":
        raw = _fields(
            value,
            set(_rank_terminal_content_fields()) | {"record_digest"},
            "rank terminal",
        )
        mode = _selection_mode(raw["predicate_pair_selection_mode"])
        if (
            raw["schema"] != PANEL_SOFT_RANK_TERMINAL_SCHEMA
            or raw["selection_mode"] != "support_only_codex_ranker"
            or raw["status"] != "rank_error"
            or any(
                type(raw[key]) is not int or raw[key] != expected
                for key, expected in (
                    ("correct_count", 0), ("determinate_count", 0),
                    ("abstain_count", 0), ("error_count", 2),
                    ("query_denominator", 2), ("query_release_count", 0),
                    ("ranker_callback_invocations", 1),
                    ("cold_replay_model_calls", 0),
                )
            )
            or raw["allow_unverified_rank_artifact"] is not False
            or raw["rank_artifact"] is not None
            or raw["no_baseline_fallback"] is not True
            or mode != "support_only_codex_ranker"
            or any(
                type(raw[key]) is not type(item) or raw[key] != item
                for key, item in _authority_data(mode).items()
            )
            or not isinstance(raw["support_png_base64_by_panel_id"], Mapping)
            or not isinstance(raw["support_artifacts"], list)
        ):
            raise PanelSoftEngineeringCampaignError(
                "rank terminal policy differs"
            )
        task = ObjectBongardTaskPlan.from_data(raw["task_plan"])
        support_ids = (
            *task.side_0_support_panel_ids,
            *task.side_1_support_panel_ids,
        )
        encoded = raw["support_png_base64_by_panel_id"]
        if set(encoded) != set(support_ids):
            raise PanelSoftEngineeringCampaignError(
                "rank terminal support keys differ"
            )
        result = cls(
            raw["campaign_source_digest"],
            task,
            raw["execution_precommit_digest"],
            PanelSoftProposerArtifact.from_data(raw["proposer_artifact"]),
            tuple((item, encoded[item]) for item in support_ids),
            tuple(
                PanelSoftObserverArtifact.from_data(item)
                for item in raw["support_artifacts"]
            ),
            PanelSoftEngineeringVersionSpace.from_data(
                raw["engineering_version_space"]
            ),
            PanelSoftRankFailureEvidence.from_data(
                raw["rank_failure_evidence"]
            ),
            ObjectBongardWriteOnceReceipt.from_data(
                raw["rank_failure_evidence_store_receipt"]
            ),
            raw["record_digest"],
        )
        if canonical_json(result.to_data()) != canonical_json(dict(raw)):
            raise PanelSoftEngineeringCampaignError(
                "rank terminal is not canonical"
            )
        return result


def _rank_terminal_content_fields() -> tuple[str, ...]:
    return (
        "schema", "campaign_source_digest", "task_plan",
        "execution_precommit_digest", "selection_mode", "proposer_artifact",
        "support_png_base64_by_panel_id", "support_artifacts",
        "engineering_version_space", "rank_failure_evidence",
        "rank_failure_evidence_store_receipt", "status", "correct_count",
        "determinate_count", "abstain_count", "error_count",
        "query_denominator", "query_release_count",
        "ranker_callback_invocations", "allow_unverified_rank_artifact",
        "rank_artifact", "no_baseline_fallback", "cold_replay_model_calls",
        *_authority_data("support_only_codex_ranker"),
    )


RunnerRecord = (
    PanelSoftEngineeringTaskRunArchive
    | PanelSoftEngineeringProposerTerminal
    | PanelSoftEngineeringRankTerminal
)


def _runner_from_data(value: object) -> RunnerRecord:
    if not isinstance(value, Mapping):
        raise PanelSoftEngineeringCampaignError("runner record must be an object")
    schema = value.get("schema")
    if schema == PANEL_SOFT_ENGINEERING_TASK_ARCHIVE_SCHEMA:
        return PanelSoftEngineeringTaskRunArchive.from_data(value)
    if schema == PANEL_SOFT_ENGINEERING_PROPOSER_TERMINAL_SCHEMA:
        return PanelSoftEngineeringProposerTerminal.from_data(value)
    if schema == PANEL_SOFT_RANK_TERMINAL_SCHEMA:
        return PanelSoftEngineeringRankTerminal.from_data(value)
    raise PanelSoftEngineeringCampaignError("runner record schema differs")


def _runner_metrics(value: RunnerRecord) -> tuple[int, int, int, int]:
    if isinstance(
        value,
        (PanelSoftEngineeringProposerTerminal, PanelSoftEngineeringRankTerminal),
    ):
        return (0, 0, 0, 2)
    return (
        value.correct_count,
        value.determinate_count,
        value.abstain_count,
        value.error_count,
    )


def _receipt_identities(value: RunnerRecord) -> tuple[tuple[str, str], ...]:
    artifacts: list[object] = [value.proposer_artifact]
    if isinstance(
        value,
        (PanelSoftEngineeringTaskRunArchive, PanelSoftEngineeringRankTerminal),
    ):
        artifacts.extend(value.support_artifacts)
    if isinstance(value, PanelSoftEngineeringTaskRunArchive):
        artifacts.extend(value.query_artifacts)
    rows: list[tuple[str, str]] = []
    for artifact in artifacts:
        if isinstance(artifact, PanelSoftProposerArtifact):
            if artifact.receipt is not None:
                rows.append((artifact.receipt.receipt_digest, artifact.receipt.thread_id))
        elif isinstance(artifact, PanelSoftObserverArtifact):
            rows.extend(
                (repeat.receipt.receipt_digest, repeat.receipt.thread_id)
                for repeat in artifact.repeats
                if repeat.receipt is not None
            )
    if (
        isinstance(value, PanelSoftEngineeringRankTerminal)
        and value.rank_failure_evidence.successful_call_identity is not None
    ):
        rows.append(value.rank_failure_evidence.successful_call_identity)
    if len({row[0] for row in rows}) != len(rows) or len({row[1] for row in rows}) != len(rows):
        raise PanelSoftEngineeringCampaignError("model-call identity is reused")
    return tuple(rows)


def _selection_model_attempt_count(value: RunnerRecord) -> int:
    if isinstance(value, PanelSoftEngineeringRankTerminal):
        return 1
    if (
        isinstance(value, PanelSoftEngineeringTaskRunArchive)
        and value.rank_artifact is not None
    ):
        return 1
    return 0


def _successful_selection_model_call_count(value: RunnerRecord) -> int:
    if isinstance(value, PanelSoftEngineeringRankTerminal):
        return int(value.rank_failure_evidence.successful_call_identity is not None)
    if (
        isinstance(value, PanelSoftEngineeringTaskRunArchive)
        and value.rank_artifact is not None
    ):
        return 1
    return 0


def _task_call_identities(
    runner: RunnerRecord,
    selector_call_identity: tuple[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    rows = _receipt_identities(runner)
    if selector_call_identity is None:
        return rows
    if (
        type(selector_call_identity) is not tuple
        or len(selector_call_identity) != 2
        or _RAW_DIGEST.fullmatch(selector_call_identity[0]) is None
        or not isinstance(selector_call_identity[1], str)
        or not selector_call_identity[1]
    ):
        raise PanelSoftEngineeringCampaignError(
            "selector call identity is malformed"
        )
    combined = (*rows, selector_call_identity)
    if (
        len({row[0] for row in combined}) != len(combined)
        or len({row[1] for row in combined}) != len(combined)
    ):
        raise PanelSoftEngineeringCampaignError("model-call identity is reused")
    return combined


def _task_content(value: "PanelSoftEngineeringCampaignTaskRecord") -> dict[str, object]:
    correct, determinate, abstain, errors = _runner_metrics(value.runner_record)
    return {
        "schema": PANEL_SOFT_CAMPAIGN_TASK_SCHEMA,
        "campaign_source_digest": value.campaign_source_digest,
        "predicate_pair_selection_mode": value.selection_mode,
        "task_plan": value.task_plan.to_data(),
        "release_authority_digest": value.release_authority_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "selector_artifact_digest": value.selector_artifact_digest,
        "selector_call_identity": (
            None
            if value.selector_call_identity is None
            else list(value.selector_call_identity)
        ),
        "selection_model_attempt_count": _selection_model_attempt_count(
            value.runner_record
        ),
        "successful_selection_model_call_count": (
            _successful_selection_model_call_count(value.runner_record)
        ),
        "rank_artifact_store_receipt": (
            None
            if value.rank_artifact_store_receipt is None
            else value.rank_artifact_store_receipt.to_data()
        ),
        "rank_journal_evidence": (
            None
            if value.rank_journal_evidence is None
            else value.rank_journal_evidence.to_data()
        ),
        "rank_journal_evidence_store_receipt": (
            None
            if value.rank_journal_evidence_store_receipt is None
            else value.rank_journal_evidence_store_receipt.to_data()
        ),
        "released_panels": [item.to_data() for item in value.released_panels],
        "release_store_receipts": [
            item.to_data() for item in value.release_store_receipts
        ],
        "runner_record": value.runner_record.to_data(),
        "runner_record_digest": value.runner_record.record_digest,
        "turn_journal_summaries": [
            item.to_data() for item in value.turn_journal_summaries
        ],
        "terminal_turn_count": len(value.turn_journal_summaries),
        "successful_terminal_turn_count": sum(
            item.terminal_status == "success"
            for item in value.turn_journal_summaries
        ),
        "failed_terminal_turn_count": sum(
            item.terminal_status == "failure"
            for item in value.turn_journal_summaries
        ),
        "successful_call_identities": [list(row) for row in value.successful_call_identities],
        "correct_count": correct,
        "determinate_count": determinate,
        "abstain_count": abstain,
        "error_count": errors,
        "query_denominator": 2,
        "query_release_count": len(value.released_panels) - 12,
        "semantic_side_0_uses_task_plan_order_not_archive_numeral": True,
        "query_release_occurs_only_inside_post_freeze_runner_callback": True,
        "cold_replay_model_calls": 0,
        **_authority_data(value.selection_mode),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringCampaignTaskRecord:
    campaign_source_digest: str
    selection_mode: str
    task_plan: ObjectBongardTaskPlan
    release_authority_digest: str
    execution_precommit_digest: str
    exposure_successor_digest: str
    selector_artifact_digest: str | None
    selector_call_identity: tuple[str, str] | None
    rank_artifact_store_receipt: ObjectBongardWriteOnceReceipt | None
    rank_journal_evidence: PanelSoftRankJournalEvidence | None
    rank_journal_evidence_store_receipt: ObjectBongardWriteOnceReceipt | None
    released_panels: tuple[ReleasedOfficialPanel, ...]
    release_store_receipts: tuple[ObjectBongardWriteOnceReceipt, ...]
    runner_record: RunnerRecord
    turn_journal_summaries: tuple[ObjectBongardTurnJournalSummary, ...]
    successful_call_identities: tuple[tuple[str, str], ...]
    record_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.campaign_source_digest, "campaign source digest")
        _selection_mode(self.selection_mode)
        _address(self.release_authority_digest, "release authority digest")
        _address(self.execution_precommit_digest, "execution precommit digest")
        _address(self.exposure_successor_digest, "exposure successor digest")
        if self.selector_artifact_digest is not None:
            _raw_digest(self.selector_artifact_digest, "selector artifact digest")
        ranked = self.selection_mode == "support_only_codex_ranker"
        rank_artifact = (
            self.runner_record.rank_artifact
            if isinstance(self.runner_record, PanelSoftEngineeringTaskRunArchive)
            else None
        )
        if (self.selector_artifact_digest is None) != (
            self.selector_call_identity is None
        ) or (not ranked and self.selector_artifact_digest is not None):
            raise PanelSoftEngineeringCampaignError(
                "selection mode and selector evidence differ"
            )
        if self.selector_call_identity is not None:
            _task_call_identities(self.runner_record, self.selector_call_identity)
        rank_custody = (
            self.rank_artifact_store_receipt,
            self.rank_journal_evidence,
            self.rank_journal_evidence_store_receipt,
        )
        if (rank_artifact is None) != all(item is None for item in rank_custody):
            raise PanelSoftEngineeringCampaignError(
                "rank artifact and durable journal custody differ"
            )
        if rank_artifact is not None:
            if (
                rank_artifact.transport_provenance.benchmark_sealable is not True
                or self.runner_record.allow_unverified_rank_artifact is not False
                or self.selector_artifact_digest != rank_artifact.artifact_digest
                or self.selector_call_identity
                != (
                    rank_artifact.receipt.receipt_digest,
                    rank_artifact.receipt.thread_id,
                )
                or not isinstance(
                    self.rank_artifact_store_receipt,
                    ObjectBongardWriteOnceReceipt,
                )
                or self.rank_artifact_store_receipt.object_kind
                != "panel-soft-rank-artifact"
                or self.rank_artifact_store_receipt.object_digest
                != "sha256:" + rank_artifact.artifact_digest
                or not isinstance(
                    self.rank_journal_evidence, PanelSoftRankJournalEvidence
                )
                or self.rank_journal_evidence.task_id != self.task_plan.task_id
                or self.rank_journal_evidence.rank_artifact_digest
                != rank_artifact.artifact_digest
                or self.rank_journal_evidence.rank_receipt_digest
                != rank_artifact.receipt.receipt_digest
                or self.rank_journal_evidence.rank_thread_id
                != rank_artifact.receipt.thread_id
                or self.rank_journal_evidence.rank_artifact_store_receipt_digest
                != self.rank_artifact_store_receipt.record_digest
                or not isinstance(
                    self.rank_journal_evidence_store_receipt,
                    ObjectBongardWriteOnceReceipt,
                )
                or self.rank_journal_evidence_store_receipt.object_kind
                != "panel-soft-rank-journal-evidence"
                or self.rank_journal_evidence_store_receipt.object_digest
                != self.rank_journal_evidence.record_digest
            ):
                raise PanelSoftEngineeringCampaignError(
                    "rank artifact durable custody differs"
                )
        if (
            ranked
            and isinstance(self.runner_record, PanelSoftEngineeringTaskRunArchive)
            and self.runner_record.status is PanelSoftEngineeringTaskRunStatus.COMPLETE
            and self.selector_artifact_digest is None
        ):
            raise PanelSoftEngineeringCampaignError(
                "ranked complete task lacks selector evidence"
            )
        _raw_digest(self.record_digest, "task record digest")
        task = ObjectBongardTaskPlan.from_data(self.task_plan.to_data())
        runner = _runner_from_data(self.runner_record.to_data())
        expected_support = (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
        expected_query = (task.side_0_query_panel_id, task.side_1_query_panel_id)
        complete = isinstance(runner, PanelSoftEngineeringTaskRunArchive) and runner.status is PanelSoftEngineeringTaskRunStatus.COMPLETE
        expected_ids = (*expected_support, *expected_query) if complete else expected_support
        if (
            task != self.task_plan
            or runner != self.runner_record
            or runner.task_plan != task
            or runner.execution_precommit_digest != self.execution_precommit_digest
            or type(self.released_panels) is not tuple
            or tuple(item.panel_id for item in self.released_panels) != expected_ids
            or any(not isinstance(item, ReleasedOfficialPanel) for item in self.released_panels)
            or type(self.release_store_receipts) is not tuple
            or len(self.release_store_receipts) != len(self.released_panels)
            or any(
                not isinstance(item, ObjectBongardWriteOnceReceipt)
                for item in self.release_store_receipts
            )
            or any(
                receipt.object_kind
                != (
                    "released-support-panel"
                    if index < 12
                    else "panel-soft-released-query-panel"
                )
                or receipt.object_digest != panel.record_digest
                for index, (panel, receipt) in enumerate(
                    zip(
                        self.released_panels,
                        self.release_store_receipts,
                        strict=True,
                    )
                )
            )
            or any(
                item.execution_precommit_digest != self.execution_precommit_digest
                or item.exposure_successor_digest != self.exposure_successor_digest
                for item in self.released_panels
            )
            or self.successful_call_identities
            != _task_call_identities(runner, self.selector_call_identity)
            or type(self.turn_journal_summaries) is not tuple
            or any(
                not isinstance(item, ObjectBongardTurnJournalSummary)
                or _turn_journal_summary_from_data(item.to_data()) != item
                or item.terminal_status not in {"success", "failure"}
                for item in self.turn_journal_summaries
            )
            or sum(
                item.terminal_status == "success"
                for item in self.turn_journal_summaries
            )
            != len(self.successful_call_identities)
            or len({item.turn_key for item in self.turn_journal_summaries})
            != len(self.turn_journal_summaries)
            or len({item.manifest_digest for item in self.turn_journal_summaries})
            != len(self.turn_journal_summaries)
            or len({item.record_digest for item in self.turn_journal_summaries})
            != len(self.turn_journal_summaries)
            or (
                self.rank_journal_evidence is not None
                and sum(
                    item == self.rank_journal_evidence.journal_summary
                    for item in self.turn_journal_summaries
                )
                != 1
            )
            or (
                isinstance(runner, PanelSoftEngineeringRankTerminal)
                and (
                    self.selection_mode != "support_only_codex_ranker"
                    or self.selector_artifact_digest is not None
                    or sum(
                        item == runner.rank_failure_evidence.journal_summary
                        for item in self.turn_journal_summaries
                    )
                    != 1
                )
            )
            or self.record_digest != canonical_digest(_task_content(self))
        ):
            raise PanelSoftEngineeringCampaignError("campaign task record differs")
        support_bytes = {
            item.panel_id: item.exact_png_bytes for item in self.released_panels[:12]
        }
        encoded_support = dict(runner.support_png_base64_by_panel_id)
        if any(base64.b64decode(encoded_support[key]) != value for key, value in support_bytes.items()):
            raise PanelSoftEngineeringCampaignError("runner support bytes differ from official release")
        if complete:
            assert isinstance(runner, PanelSoftEngineeringTaskRunArchive)
            encoded_query = dict(runner.query_png_base64_by_side)
            if any(
                base64.b64decode(encoded_query[side]) != panel.exact_png_bytes
                for side, panel in zip(("side_0", "side_1"), self.released_panels[12:], strict=True)
            ):
                raise PanelSoftEngineeringCampaignError("runner query bytes differ from official release")

    @classmethod
    def seal(
        cls,
        *,
        task_plan: ObjectBongardTaskPlan,
        selection_mode: str = "support_only_codex_ranker",
        release_authority_digest: str,
        execution_precommit_digest: str,
        exposure_successor_digest: str,
        selector_artifact_digest: str | None = None,
        selector_call_identity: tuple[str, str] | None = None,
        rank_artifact_store_receipt: ObjectBongardWriteOnceReceipt | None = None,
        rank_journal_evidence: PanelSoftRankJournalEvidence | None = None,
        rank_journal_evidence_store_receipt: ObjectBongardWriteOnceReceipt | None = None,
        released_panels: Sequence[ReleasedOfficialPanel],
        release_store_receipts: Sequence[ObjectBongardWriteOnceReceipt],
        runner_record: RunnerRecord,
        turn_journal_summaries: Sequence[ObjectBongardTurnJournalSummary],
    ) -> "PanelSoftEngineeringCampaignTaskRecord":
        values = {
            "campaign_source_digest": panel_soft_engineering_campaign_source_digest(),
            "selection_mode": _selection_mode(selection_mode),
            "task_plan": task_plan,
            "release_authority_digest": release_authority_digest,
            "execution_precommit_digest": execution_precommit_digest,
            "exposure_successor_digest": exposure_successor_digest,
            "selector_artifact_digest": selector_artifact_digest,
            "selector_call_identity": selector_call_identity,
            "rank_artifact_store_receipt": rank_artifact_store_receipt,
            "rank_journal_evidence": rank_journal_evidence,
            "rank_journal_evidence_store_receipt": (
                rank_journal_evidence_store_receipt
            ),
            "released_panels": tuple(released_panels),
            "release_store_receipts": tuple(release_store_receipts),
            "runner_record": runner_record,
            "turn_journal_summaries": tuple(turn_journal_summaries),
            "successful_call_identities": _task_call_identities(
                runner_record, selector_call_identity
            ),
        }
        provisional = object.__new__(cls)
        for key, item in values.items():
            object.__setattr__(provisional, key, item)
        return cls(**values, record_digest=canonical_digest(_task_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_task_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftEngineeringCampaignTaskRecord":
        raw = _fields(value, set(_task_content_fields()) | {"record_digest"}, "campaign task")
        if (
            raw["schema"] != PANEL_SOFT_CAMPAIGN_TASK_SCHEMA
            or raw["semantic_side_0_uses_task_plan_order_not_archive_numeral"] is not True
            or raw["query_release_occurs_only_inside_post_freeze_runner_callback"] is not True
            or type(raw["cold_replay_model_calls"]) is not int
            or raw["cold_replay_model_calls"] != 0
            or any(
                type(raw[key]) is not type(item) or raw[key] != item
                for key, item in _authority_data(raw["predicate_pair_selection_mode"]).items()
            )
            or not isinstance(raw["released_panels"], list)
            or not isinstance(raw["release_store_receipts"], list)
            or not isinstance(raw["turn_journal_summaries"], list)
            or not isinstance(raw["successful_call_identities"], list)
        ):
            raise PanelSoftEngineeringCampaignError("campaign task policy differs")
        runner = _runner_from_data(raw["runner_record"])
        result = cls(
            raw["campaign_source_digest"], raw["predicate_pair_selection_mode"],
            ObjectBongardTaskPlan.from_data(raw["task_plan"]),
            raw["release_authority_digest"],
            raw["execution_precommit_digest"], raw["exposure_successor_digest"],
            raw["selector_artifact_digest"],
            (
                None
                if raw["selector_call_identity"] is None
                else tuple(raw["selector_call_identity"])
            ),
            (
                None
                if raw["rank_artifact_store_receipt"] is None
                else ObjectBongardWriteOnceReceipt.from_data(
                    raw["rank_artifact_store_receipt"]
                )
            ),
            (
                None
                if raw["rank_journal_evidence"] is None
                else PanelSoftRankJournalEvidence.from_data(
                    raw["rank_journal_evidence"]
                )
            ),
            (
                None
                if raw["rank_journal_evidence_store_receipt"] is None
                else ObjectBongardWriteOnceReceipt.from_data(
                    raw["rank_journal_evidence_store_receipt"]
                )
            ),
            tuple(ReleasedOfficialPanel.from_data(item) for item in raw["released_panels"]),
            tuple(
                ObjectBongardWriteOnceReceipt.from_data(item)
                for item in raw["release_store_receipts"]
            ),
            runner,
            tuple(
                _turn_journal_summary_from_data(item)
                for item in raw["turn_journal_summaries"]
            ),
            tuple(tuple(item) for item in raw["successful_call_identities"]),
            raw["record_digest"],
        )
        correct, determinate, abstain, errors = _runner_metrics(runner)
        if (
            raw["runner_record_digest"] != runner.record_digest
            or (raw["correct_count"], raw["determinate_count"], raw["abstain_count"], raw["error_count"])
            != (correct, determinate, abstain, errors)
            or any(
                type(raw[key]) is not int
                for key in (
                    "correct_count", "determinate_count", "abstain_count",
                    "error_count", "query_denominator", "query_release_count",
                    "selection_model_attempt_count",
                    "successful_selection_model_call_count",
                    "terminal_turn_count", "successful_terminal_turn_count",
                    "failed_terminal_turn_count",
                )
            )
            or raw["query_denominator"] != 2
            or raw["query_release_count"] != len(result.released_panels) - 12
            or raw["selection_model_attempt_count"]
            != _selection_model_attempt_count(runner)
            or raw["successful_selection_model_call_count"]
            != _successful_selection_model_call_count(runner)
            or raw["terminal_turn_count"] != len(result.turn_journal_summaries)
            or raw["successful_terminal_turn_count"]
            != sum(
                item.terminal_status == "success"
                for item in result.turn_journal_summaries
            )
            or raw["failed_terminal_turn_count"]
            != sum(
                item.terminal_status == "failure"
                for item in result.turn_journal_summaries
            )
            or canonical_json(result.to_data()) != canonical_json(dict(raw))
        ):
            raise PanelSoftEngineeringCampaignError("campaign task metrics differ")
        return result


def _task_content_fields() -> tuple[str, ...]:
    return (
        "schema", "campaign_source_digest", "predicate_pair_selection_mode",
        "task_plan", "release_authority_digest",
        "execution_precommit_digest", "exposure_successor_digest",
        "selector_artifact_digest", "selector_call_identity",
        "selection_model_attempt_count", "successful_selection_model_call_count",
        "rank_artifact_store_receipt", "rank_journal_evidence",
        "rank_journal_evidence_store_receipt",
        "released_panels", "release_store_receipts", "runner_record",
        "runner_record_digest", "turn_journal_summaries", "terminal_turn_count",
        "successful_terminal_turn_count", "failed_terminal_turn_count",
        "successful_call_identities", "correct_count",
        "determinate_count", "abstain_count", "error_count", "query_denominator",
        "query_release_count", "semantic_side_0_uses_task_plan_order_not_archive_numeral",
        "query_release_occurs_only_inside_post_freeze_runner_callback",
        "cold_replay_model_calls", *_authority_data(),
    )


def panel_soft_engineering_campaign_source_bindings() -> dict[str, str]:
    """Return active Python authorities committed by campaign preparation."""

    return {
        "panel_soft_campaign": (
            "sha256:" + panel_soft_engineering_campaign_source_digest()
        ),
        "panel_soft_runner": (
            "sha256:" + panel_soft_engineering_task_runner_source_digest()
        ),
        "panel_soft_proposer": "sha256:" + panel_soft_proposer_source_digest(),
        "panel_soft_observer": "sha256:" + panel_soft_observer_source_digest(),
        "panel_soft_predicate": "sha256:" + panel_soft_predicate_source_digest(),
        "panel_soft_ranker": "sha256:" + panel_soft_ranker_source_digest(),
        "panel_soft_turn_journal": (
            "sha256:" + object_bongard_turn_journal_source_digest()
        ),
        "panel_soft_source_manifest_algorithm": (
            "sha256:" + object_scene_anchor_source_manifest_source_digest()
        ),
    }


def _build_panel_soft_engineering_campaign_source_manifest(
) -> ObjectSceneAnchorSourceManifest:
    """Build the closure under the canonical import name, including `-m` runs."""

    return build_object_scene_anchor_source_manifest(
        root_module=PANEL_SOFT_CAMPAIGN_MODULE_NAME,
        repository_root=_ROOT,
    )


@dataclass(frozen=True, slots=True)
class PreparedPanelSoftEngineeringCampaign:
    output_root: Path
    selection_mode: str
    workers: int
    selection_seed: str
    plan: ObjectBongardBatchPlan
    descriptor: OfficialReleaseDescriptor
    archive: OfficialPanelArchive = field(repr=False, compare=False)
    split: SplitIndex = field(repr=False, compare=False)
    predecessor: ExposureLedger
    historical_exposure: HistoricalExposureSeed = field(repr=False, compare=False)
    source_manifest: ObjectSceneAnchorSourceManifest
    source_manifest_receipt: ObjectBongardWriteOnceReceipt
    runtime: ObjectBongardTurnRuntime = field(repr=False, compare=False)
    runtime_evidence: Mapping[str, Any]
    runtime_evidence_receipt: ObjectBongardWriteOnceReceipt
    precommit: ObjectBongardExecutionPrecommit
    release: PreparedObjectBongardRelease
    release_authority: PanelSoftCampaignReleaseAuthority
    release_authority_receipt: ObjectBongardWriteOnceReceipt
    research_exposure_successor_path: Path
    research_exposure_successor_evidence: Mapping[str, Any]
    research_exposure_successor_receipt: ObjectBongardWriteOnceReceipt

    def __post_init__(self) -> None:
        mode = _selection_mode(self.selection_mode)
        workers = _worker_count(self.workers)
        configuration = dict(self.precommit.configuration)
        source_bindings = dict(self.precommit.runtime_source_bindings)
        verified_runtime, _fingerprint = _verify_runtime_evidence(
            self.runtime_evidence
        )
        verified_manifest = cold_verify_object_scene_anchor_source_manifest(
            self.source_manifest,
            repository_root=_ROOT,
            expected_manifest_digest=self.source_manifest.manifest_digest,
        )
        manifest_address = "sha256:" + verified_manifest.manifest_digest
        if (
            not isinstance(self.output_root, Path)
            or not self.output_root.is_absolute()
            or self.release.store.root != self.output_root
            or self.release.plan != self.plan
            or self.release.precommit != self.precommit
            or self.archive.record_digest != self.precommit.archive_record_digest
            or verified_runtime != self.runtime
            or verified_manifest != self.source_manifest
            or self.runtime_evidence.get("source_manifest_digest")
            != manifest_address
            or configuration.get("runtime_binding_digest")
            != _content_address(self.runtime.binding)
            or configuration.get("runtime_evidence_digest")
            != self.runtime_evidence["record_digest"]
            or configuration.get("runtime_evidence_receipt_digest")
            != self.runtime_evidence_receipt.record_digest
            or configuration.get("source_manifest_digest") != manifest_address
            or configuration.get("source_manifest_receipt_digest")
            != self.source_manifest_receipt.record_digest
            or configuration.get("predicate_pair_selection_mode") != mode
            or configuration.get("workers") != workers
            or configuration.get("task_execution_model")
            != "thread-pool-task-isolated-deterministic-plan-order"
            or configuration.get("support_only_ranker_present")
            is not (mode == "support_only_codex_ranker")
            or configuration.get("deterministic_selector_baseline")
            is not (mode == "deterministic_baseline")
            or source_bindings.get("panel_soft_runtime_evidence")
            != self.runtime_evidence["record_digest"]
            or source_bindings.get("panel_soft_runtime_evidence_receipt")
            != self.runtime_evidence_receipt.record_digest
            or source_bindings.get("panel_soft_source_manifest")
            != manifest_address
            or source_bindings.get("panel_soft_source_manifest_receipt")
            != self.source_manifest_receipt.record_digest
            or self.release_authority
            != PanelSoftCampaignReleaseAuthority.from_prepared_release(
                self.release, selection_mode=mode
            )
            or len(self.plan.tasks) != PANEL_SOFT_CAMPAIGN_TASK_COUNT
            or tuple(task.family for task in self.plan.tasks) != ("bd", "ff", "hd")
        ):
            raise PanelSoftEngineeringCampaignError(
                "prepared panel-soft campaign parents differ"
            )
        verify_object_bongard_batch_plan(
            self.plan,
            task_ids=_archive_task_ids(self.archive),
            train_task_ids=tuple(self.split.canonical_groups["train"]),
            exact_used_task_ids=tuple(
                sorted(
                    set(self.predecessor.exposed_task_ids)
                    | set(self.historical_exposure.exact_official_task_ids)
                )
            ),
            selection_seed=self.selection_seed,
        )
        verify_prepared_object_bongard_release(self.release)
        durable_runtime = self.release.store.verify(
            self.runtime_evidence_receipt,
            expected_data=self.runtime_evidence,
        )
        reloaded_runtime, _ = _verify_runtime_evidence(durable_runtime)
        durable_manifest = ObjectSceneAnchorSourceManifest.from_data(
            self.release.store.verify(
                self.source_manifest_receipt,
                expected_data=self.source_manifest.to_data(),
            )
        )
        decoded = PanelSoftCampaignReleaseAuthority.from_data(
            self.release.store.verify(
                self.release_authority_receipt,
                expected_data=self.release_authority.to_data(),
            )
        )
        if (
            reloaded_runtime != self.runtime
            or durable_manifest != self.source_manifest
            or self.source_manifest_receipt.object_kind
            != "panel-soft-source-manifest"
            or self.source_manifest_receipt.object_digest != manifest_address
            or self.runtime_evidence_receipt.object_kind
            != "panel-soft-runtime-evidence"
            or self.runtime_evidence_receipt.object_digest
            != self.runtime_evidence["record_digest"]
            or decoded != self.release_authority
            or self.release_authority_receipt.object_kind
            != "panel-soft-release-authority"
            or self.release_authority_receipt.object_digest
            != self.release_authority.record_digest
        ):
            raise PanelSoftEngineeringCampaignError(
                "campaign release authority durable replay differs"
            )
        expected_successor_path = (
            self.output_root
            / "research-exposure-successors"
            / (
                self.release.successor.digest.removeprefix("sha256:")
                + ".exposure.json"
            )
        )
        if self.research_exposure_successor_path != expected_successor_path:
            raise PanelSoftEngineeringCampaignError(
                "research exposure successor path differs"
            )
        _verify_exposure_successor_mirror(
            store=self.release.store,
            successor=self.release.successor,
            path=self.research_exposure_successor_path,
            evidence=self.research_exposure_successor_evidence,
            receipt=self.research_exposure_successor_receipt,
        )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


FreshRuntimeFactory = Callable[
    ...,
    tuple[ObjectBongardTurnRuntime, Mapping[str, str]],
]


def _validate_runtime_request(
    runtime: ObjectBongardTurnRuntime,
    launcher_fingerprint: Mapping[str, str],
    *,
    model: object,
    reasoning_effort: object,
    minutes: object,
    verbose: object,
    executable: object,
    expected_launcher_sha256: object,
) -> dict[str, str]:
    if not isinstance(runtime, ObjectBongardTurnRuntime):
        raise PanelSoftEngineeringCampaignError("runtime evidence type differs")
    expected = (
        model, reasoning_effort, minutes, verbose, executable,
        expected_launcher_sha256,
    )
    actual = (
        runtime.model, runtime.reasoning_effort, runtime.minutes,
        runtime.verbose, runtime.executable, runtime.expected_launcher_digest,
    )
    if any(
        type(got) is not type(want) or got != want
        for got, want in zip(actual, expected, strict=True)
    ):
        raise PanelSoftEngineeringCampaignError(
            "fresh runtime differs from the exact requested invocation"
        )
    if (
        not isinstance(launcher_fingerprint, Mapping)
        or set(launcher_fingerprint) != {"version", "launcher_digest"}
        or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in launcher_fingerprint.items()
        )
    ):
        raise PanelSoftEngineeringCampaignError(
            "authenticated launcher fingerprint fields differ"
        )
    fingerprint = dict(launcher_fingerprint)
    attestation = runtime.no_tools_attestation.to_dict()
    if (
        fingerprint["version"] != PINNED_CODEX_CLI_VERSION
        or fingerprint["version"] != attestation.get("launcher_version")
        or fingerprint["launcher_digest"] != runtime.expected_launcher_digest
        or fingerprint["launcher_digest"] != attestation.get("launcher_digest")
        or runtime.transport_source_digest
        != prototype_scene_transport_source_digest()
    ):
        raise PanelSoftEngineeringCampaignError(
            "runtime launcher version, bytes, attestation, or transport differs"
        )
    return fingerprint


def _create_fresh_panel_soft_runtime(
    *,
    model: str,
    reasoning_effort: str,
    minutes: int,
    verbose: bool,
    executable: str,
    expected_launcher_sha256: str,
    cache_snapshotter: Callable[[], CloudPolicyCacheSnapshot] = (
        snapshot_cloud_policy_cache
    ),
    catalog_snapshotter: Callable[[], CodexModelCatalogSnapshot] = (
        snapshot_pinned_model_catalog
    ),
    launcher_fingerprinter: Callable[..., Mapping[str, str]] = (
        codex_cli_authenticated_fingerprint
    ),
    runtime_attester: Callable[..., CodexNoToolsAttestation] = (
        attest_codex_no_tools
    ),
) -> tuple[ObjectBongardTurnRuntime, Mapping[str, str]]:
    """Take fresh launcher, catalog, cloud-policy, and no-tools evidence."""

    cache = cache_snapshotter()
    catalog = catalog_snapshotter()
    fingerprint = launcher_fingerprinter(
        executable, expected_launcher_digest=expected_launcher_sha256
    )
    if (
        not isinstance(fingerprint, Mapping)
        or fingerprint.get("launcher_digest") != expected_launcher_sha256
    ):
        raise PanelSoftEngineeringCampaignError(
            "fresh authenticated Codex launcher fingerprint differs"
        )
    attestation = runtime_attester(
        executable=executable,
        expected_launcher_digest=expected_launcher_sha256,
        model_catalog_snapshot=catalog,
        cloud_policy_cache_snapshot=cache,
    )
    runtime = ObjectBongardTurnRuntime(
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        cloud_policy_cache_snapshot=cache,
        model_catalog_snapshot=catalog,
        expected_launcher_digest=expected_launcher_sha256,
        no_tools_attestation=attestation,
        transport_source_digest=prototype_scene_transport_source_digest(),
    )
    return runtime, _validate_runtime_request(
        runtime,
        fingerprint,
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
    )


def _runtime_evidence_content(
    runtime: ObjectBongardTurnRuntime,
    launcher_fingerprint: Mapping[str, str],
    *,
    source_manifest_digest: str,
) -> dict[str, object]:
    cache = runtime.cloud_policy_cache_snapshot
    return {
        "schema": PANEL_SOFT_RUNTIME_EVIDENCE_SCHEMA,
        "runtime_binding": runtime.binding,
        "runtime_binding_digest": _content_address(runtime.binding),
        "source_manifest_digest": source_manifest_digest,
        "launcher_fingerprint": dict(launcher_fingerprint),
        "model_catalog_base64": base64.b64encode(
            runtime.model_catalog_snapshot.data
        ).decode("ascii"),
        "model_catalog_raw_digest": runtime.model_catalog_snapshot.raw_digest,
        "model_catalog_canonical_digest": (
            runtime.model_catalog_snapshot.canonical_digest
        ),
        "cloud_policy_cache_base64": (
            None
            if cache is None or cache.data is None
            else base64.b64encode(cache.data).decode("ascii")
        ),
        "cloud_policy_cache_snapshot_present": cache is not None,
        "cloud_policy_cache_binding": runtime.policy_cache_binding,
        "no_tools_attestation_base64": base64.b64encode(
            runtime.no_tools_attestation.canonical_bytes
        ).decode("ascii"),
        "no_tools_attestation": runtime.no_tools_attestation.to_dict(),
        "no_tools_attestation_digest": (
            runtime.no_tools_attestation.attestation_digest
        ),
        "fresh_launcher_catalog_cache_and_no_tools_evidence": True,
        "persisted_and_reloaded_before_exposure_successor": True,
    }


def _seal_runtime_evidence(
    runtime: ObjectBongardTurnRuntime,
    launcher_fingerprint: Mapping[str, str],
    *,
    source_manifest_digest: str,
) -> dict[str, object]:
    content = _runtime_evidence_content(
        runtime,
        launcher_fingerprint,
        source_manifest_digest=source_manifest_digest,
    )
    return {**content, "record_digest": _content_address(content)}


def _verify_runtime_evidence(
    value: object,
) -> tuple[ObjectBongardTurnRuntime, dict[str, str]]:
    expected = {
        "schema", "runtime_binding", "runtime_binding_digest",
        "source_manifest_digest",
        "launcher_fingerprint", "model_catalog_base64",
        "model_catalog_raw_digest", "model_catalog_canonical_digest",
        "cloud_policy_cache_base64", "cloud_policy_cache_binding",
        "cloud_policy_cache_snapshot_present",
        "no_tools_attestation_base64", "no_tools_attestation",
        "no_tools_attestation_digest",
        "fresh_launcher_catalog_cache_and_no_tools_evidence",
        "persisted_and_reloaded_before_exposure_successor", "record_digest",
    }
    raw = _fields(value, expected, "runtime evidence")
    if (
        raw["schema"] != PANEL_SOFT_RUNTIME_EVIDENCE_SCHEMA
        or raw["fresh_launcher_catalog_cache_and_no_tools_evidence"] is not True
        or raw["persisted_and_reloaded_before_exposure_successor"] is not True
        or not isinstance(raw["runtime_binding"], Mapping)
        or not isinstance(raw["launcher_fingerprint"], Mapping)
        or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in raw["launcher_fingerprint"].items()
        )
    ):
        raise PanelSoftEngineeringCampaignError("runtime evidence policy differs")
    try:
        catalog = CodexModelCatalogSnapshot(
            base64.b64decode(raw["model_catalog_base64"], validate=True)
        )
        cache_encoded = raw["cloud_policy_cache_base64"]
        cache_data = (
            None
            if cache_encoded is None
            else base64.b64decode(cache_encoded, validate=True)
        )
        cache = (
            CloudPolicyCacheSnapshot(cache_data)
            if raw["cloud_policy_cache_snapshot_present"] is True
            else None
        )
        attestation_bytes = base64.b64decode(
            raw["no_tools_attestation_base64"], validate=True
        )
        attestation = CodexNoToolsAttestation(attestation_bytes)
    except Exception as exc:
        raise PanelSoftEngineeringCampaignError(
            "runtime evidence exact bytes are invalid"
        ) from exc
    binding = dict(raw["runtime_binding"])
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
    fingerprint = dict(raw["launcher_fingerprint"])
    content = {key: item for key, item in raw.items() if key != "record_digest"}
    source_manifest_digest = _address(
        raw["source_manifest_digest"], "runtime source manifest digest"
    )
    validated_fingerprint = _validate_runtime_request(
        runtime,
        fingerprint,
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        minutes=runtime.minutes,
        verbose=runtime.verbose,
        executable=runtime.executable,
        expected_launcher_sha256=runtime.expected_launcher_digest,
    )
    if (
        runtime.binding != binding
        or raw["runtime_binding_digest"] != _content_address(binding)
        or fingerprint.get("launcher_digest") != runtime.expected_launcher_digest
        or raw["model_catalog_raw_digest"] != catalog.raw_digest
        or raw["model_catalog_canonical_digest"] != catalog.canonical_digest
        or raw["cloud_policy_cache_binding"] != runtime.policy_cache_binding
        or raw["no_tools_attestation"] != attestation.to_dict()
        or raw["no_tools_attestation_digest"] != attestation.attestation_digest
        or raw["record_digest"] != _content_address(content)
        or _seal_runtime_evidence(
            runtime,
            validated_fingerprint,
            source_manifest_digest=source_manifest_digest,
        ) != dict(raw)
    ):
        raise PanelSoftEngineeringCampaignError("runtime evidence differs")
    return runtime, fingerprint


def prepare_panel_soft_engineering_campaign(
    *,
    output_root: str | os.PathLike[str],
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = DEFAULT_MINUTES,
    verbose: bool = False,
    executable: str = DEFAULT_CODEX_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_CODEX_LAUNCHER_SHA256,
    fresh_runtime_factory: FreshRuntimeFactory = _create_fresh_panel_soft_runtime,
    selection_mode: str = "support_only_codex_ranker",
    workers: int = DEFAULT_WORKERS,
    descriptor_path: str | os.PathLike[str] = DEFAULT_DESCRIPTOR,
    archive_path: str | os.PathLike[str] = DEFAULT_ARCHIVE,
    split_path: str | os.PathLike[str] = DEFAULT_SPLIT,
    predecessor_path: str | os.PathLike[str] = DEFAULT_PREDECESSOR,
    historical_exposure_path: str | os.PathLike[str] = DEFAULT_HISTORICAL_EXPOSURE,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    expected_selected_task_ids: Sequence[str] | None = DEFAULT_SELECTED_TASK_IDS,
    expected_plan_digest: str | None = DEFAULT_PLAN_DIGEST,
    expected_predecessor_digest: str | None = DEFAULT_PREDECESSOR_LEDGER_DIGEST,
    expected_predecessor_file_sha256: str | None = DEFAULT_PREDECESSOR_FILE_SHA256,
    require_official_split_counts: bool = True,
    exposure_observed_at: str | None = None,
    additional_source_bindings: Mapping[str, str] | None = None,
    additional_configuration: Mapping[str, str | int | bool] | None = None,
) -> PreparedPanelSoftEngineeringCampaign:
    """Prepare the exact-unused three-task campaign without reading a panel."""

    mode = _selection_mode(selection_mode)
    worker_count = _worker_count(workers)
    if not isinstance(selection_seed, str) or not selection_seed.strip():
        raise PanelSoftEngineeringCampaignError("selection seed must be nonempty")
    _raw_digest(expected_launcher_sha256, "expected launcher digest")
    fresh_runtime = fresh_runtime_factory(
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
    )
    if (
        not isinstance(fresh_runtime, tuple)
        or len(fresh_runtime) != 2
        or not isinstance(fresh_runtime[0], ObjectBongardTurnRuntime)
        or not isinstance(fresh_runtime[1], Mapping)
    ):
        raise PanelSoftEngineeringCampaignError(
            "fresh runtime factory returned the wrong evidence envelope"
        )
    runtime, launcher_fingerprint = fresh_runtime
    launcher_fingerprint = _validate_runtime_request(
        runtime,
        launcher_fingerprint,
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
    )
    source_manifest = _build_panel_soft_engineering_campaign_source_manifest()
    cold_verify_object_scene_anchor_source_manifest(
        source_manifest,
        repository_root=_ROOT,
        expected_manifest_digest=source_manifest.manifest_digest,
    )
    source_manifest_address = "sha256:" + source_manifest.manifest_digest
    if (
        mode == "support_only_codex_ranker"
        and not isinstance(runtime.cloud_policy_cache_snapshot, CloudPolicyCacheSnapshot)
    ):
        raise PanelSoftEngineeringCampaignError(
            "ranked campaign requires an exact cloud-policy cache snapshot"
        )
    descriptor = OfficialReleaseDescriptor.load(descriptor_path)
    archive = OfficialPanelArchive.load(
        descriptor,
        archive_path,
        expected_release_descriptor_digest=descriptor.digest,
    )
    descriptor.verify_split(split_path)
    split = SplitIndex.load(split_path)
    task_ids = _archive_task_ids(archive)
    split.validate(task_ids, official_counts=require_official_split_counts)
    train_task_ids = tuple(split.canonical_groups["train"])
    predecessor_file = Path(predecessor_path).expanduser().resolve()
    if (
        expected_predecessor_file_sha256 is not None
        and _file_sha256(predecessor_file) != expected_predecessor_file_sha256
    ):
        raise PanelSoftEngineeringCampaignError(
            "exposure predecessor file identity differs"
        )
    predecessor = ExposureLedger.load(predecessor_file)
    if (
        expected_predecessor_digest is not None
        and predecessor.digest != expected_predecessor_digest
    ):
        raise PanelSoftEngineeringCampaignError("exposure predecessor ledger differs")
    historical = load_historical_exposure(
        historical_exposure_path, verify_evidence=False
    )
    exact_used = tuple(
        sorted(
            set(predecessor.exposed_task_ids)
            | set(historical.exact_official_task_ids)
        )
    )
    plan = plan_object_bongard_batch(
        task_ids=task_ids,
        train_task_ids=train_task_ids,
        exact_used_task_ids=exact_used,
        selection_seed=selection_seed,
        requested_per_family=1,
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
    selected = tuple(task.task_id for task in plan.tasks)
    if expected_selected_task_ids is not None and selected != tuple(
        expected_selected_task_ids
    ):
        raise PanelSoftEngineeringCampaignError(
            "selected task IDs differ from the exact campaign commitment"
        )
    if expected_plan_digest is not None and plan.record_digest != expected_plan_digest:
        raise PanelSoftEngineeringCampaignError(
            "batch plan differs from the exact campaign commitment"
        )
    root = Path(os.path.abspath(os.path.expanduser(str(output_root))))
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError as exc:
        raise PanelSoftEngineeringCampaignError(
            "campaign output root must be fresh"
        ) from exc
    store = ObjectBongardReleaseStore(root)
    source_manifest_receipt = _persist_mapping(
        store,
        kind="panel-soft-source-manifest",
        digest=source_manifest_address,
        data=source_manifest.to_data(),
    )
    runtime_evidence = _seal_runtime_evidence(
        runtime,
        launcher_fingerprint,
        source_manifest_digest=source_manifest_address,
    )
    runtime_evidence_receipt = _persist_mapping(
        store,
        kind="panel-soft-runtime-evidence",
        digest=runtime_evidence["record_digest"],  # type: ignore[arg-type]
        data=runtime_evidence,
    )
    bindings = panel_soft_engineering_campaign_source_bindings()
    bindings["panel_soft_runtime_evidence"] = runtime_evidence[  # type: ignore[assignment]
        "record_digest"
    ]
    bindings["panel_soft_runtime_evidence_receipt"] = (
        runtime_evidence_receipt.record_digest
    )
    bindings["panel_soft_source_manifest"] = source_manifest_address
    bindings["panel_soft_source_manifest_receipt"] = (
        source_manifest_receipt.record_digest
    )
    if mode == "support_only_codex_ranker":
        bindings["panel_soft_ranker"] = (
            "sha256:" + panel_soft_ranker_source_digest()
        )
    for key, value in dict(additional_source_bindings or {}).items():
        if key in bindings and bindings[key] != value:
            raise PanelSoftEngineeringCampaignError(
                f"additional source binding {key} conflicts"
            )
        bindings[key] = value
    runtime_binding_digest = _content_address(runtime.binding)
    configuration: dict[str, str | int | bool] = {
        "runtime_binding_digest": runtime_binding_digest,
        "runtime_evidence_digest": runtime_evidence["record_digest"],  # type: ignore[dict-item]
        "runtime_evidence_receipt_digest": runtime_evidence_receipt.record_digest,
        "source_manifest_digest": source_manifest_address,
        "source_manifest_receipt_digest": source_manifest_receipt.record_digest,
        "predicate_pair_selection_mode": mode,
        "workers": worker_count,
        "task_execution_model": (
            "thread-pool-task-isolated-deterministic-plan-order"
        ),
        "support_only_ranker_present": mode == "support_only_codex_ranker",
        "deterministic_selector_baseline": mode == "deterministic_baseline",
        "headless": True,
        "requested_per_family": 1,
        "query_denominator": PANEL_SOFT_CAMPAIGN_QUERY_DENOMINATOR,
        "official_test_authorized": False,
        "python_is_canonical_authority": True,
        "lean_required": False,
        "lean_removable": True,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
    }
    if mode == "support_only_codex_ranker":
        configuration["selector_runtime_binding_digest"] = (
            runtime_binding_digest
        )
    for key, value in dict(additional_configuration or {}).items():
        if key in configuration and configuration[key] != value:
            raise PanelSoftEngineeringCampaignError(
                f"additional configuration {key} conflicts"
            )
        configuration[key] = value
    timestamp = exposure_observed_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
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
        exposure_actor="headless-codex-panel-soft-proposer",
        exposure_purpose="exact-unused-train-panel-soft-engineering-campaign",
        exposure_source="official-shapebongard-v2-archive",
    )
    release = prepare_object_bongard_release(
        store=store,
        plan=plan,
        precommit=precommit,
        predecessor=predecessor,
    )
    successor_path, successor_evidence, successor_receipt = (
        _persist_exposure_successor_mirror(store=store, successor=release.successor)
    )
    authority = PanelSoftCampaignReleaseAuthority.from_prepared_release(
        release, selection_mode=mode
    )
    authority_receipt = _persist_mapping(
        store,
        kind="panel-soft-release-authority",
        digest=authority.record_digest,
        data=authority.to_data(),
    )
    return PreparedPanelSoftEngineeringCampaign(
        output_root=root,
        selection_mode=mode,
        workers=worker_count,
        selection_seed=selection_seed,
        plan=plan,
        descriptor=descriptor,
        archive=archive,
        split=split,
        predecessor=predecessor,
        historical_exposure=historical,
        source_manifest=source_manifest,
        source_manifest_receipt=source_manifest_receipt,
        runtime=runtime,
        runtime_evidence=runtime_evidence,
        runtime_evidence_receipt=runtime_evidence_receipt,
        precommit=precommit,
        release=release,
        release_authority=authority,
        release_authority_receipt=authority_receipt,
        research_exposure_successor_path=successor_path,
        research_exposure_successor_evidence=successor_evidence,
        research_exposure_successor_receipt=successor_receipt,
    )


class _SequencedTransport:
    def __init__(self, transports: Sequence[Callable[..., CodexStructuredResult]]) -> None:
        self.transports = tuple(transports)
        self.index = 0

    def __call__(self, *args: Any, **kwargs: Any) -> CodexStructuredResult:
        if self.index >= len(self.transports):
            raise PanelSoftEngineeringCampaignError("observer transport exceeded frozen repeat count")
        transport = self.transports[self.index]
        self.index += 1
        return transport(*args, **kwargs)


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


def _release_panel_soft_engineering_query_panel(
    *,
    prepared: PreparedObjectBongardRelease,
    archive: OfficialPanelArchive,
    panel_id: str,
    freeze: PanelSoftEngineeringTaskFreeze,
    commit: PanelSoftEngineeringTaskFreezeCommit,
    exact_freeze_payload: bytes,
    freeze_store_receipt: ObjectBongardWriteOnceReceipt,
    commit_store_receipt: ObjectBongardWriteOnceReceipt,
) -> tuple[ReleasedOfficialPanel, ObjectBongardWriteOnceReceipt]:
    """Release one sealed query only after the panel-soft freeze was reloaded.

    This narrow gate binds the authenticated panel directly to the runner's
    Python predicate-pair freeze and commit.  Ranked mode performs its
    separately typed artifact/journal custody checks before entering here.
    """

    verify_prepared_object_bongard_release(prepared)
    if not isinstance(archive, OfficialPanelArchive):
        raise TypeError("archive must be OfficialPanelArchive")
    if not isinstance(freeze, PanelSoftEngineeringTaskFreeze):
        raise TypeError("freeze must be PanelSoftEngineeringTaskFreeze")
    if not isinstance(commit, PanelSoftEngineeringTaskFreezeCommit):
        raise TypeError("commit must be PanelSoftEngineeringTaskFreezeCommit")
    if not isinstance(exact_freeze_payload, bytes):
        raise TypeError("exact_freeze_payload must be bytes")
    restored_freeze = PanelSoftEngineeringTaskFreeze.from_data(freeze.to_data())
    restored_commit = PanelSoftEngineeringTaskFreezeCommit.from_data(commit.to_data())
    if restored_freeze != freeze or restored_commit != commit:
        raise PanelSoftEngineeringCampaignError("freeze or commit is not canonical")
    matches = tuple(
        task
        for task in prepared.plan.tasks
        if panel_id in (task.side_0_query_panel_id, task.side_1_query_panel_id)
    )
    if len(matches) != 1:
        raise PanelSoftEngineeringCampaignError(
            "panel is not one sealed query in the prepared release"
        )
    task = matches[0]
    expected_queries = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    if (
        archive.record_digest != prepared.authorization.archive_record_digest
        or panel_id not in prepared.authorization.sealed_query_panel_ids
        or freeze.task_id != task.task_id
        or freeze.task_plan_digest != task.record_digest
        or freeze.execution_precommit_digest != prepared.precommit.record_digest
        or freeze.sealed_query_panel_ids != expected_queries
    ):
        raise PanelSoftEngineeringCampaignError(
            "query archive, task, precommit, or sealed identities differ"
        )
    expected_payload = canonical_json(freeze.to_data()) + b"\n"
    if exact_freeze_payload != expected_payload:
        raise PanelSoftEngineeringCampaignError("query gate did not receive exact freeze bytes")
    commit.assert_matches(freeze, exact_freeze_payload)
    prepared.store.verify(freeze_store_receipt, expected_data=freeze.to_data())
    prepared.store.verify(commit_store_receipt, expected_data=commit.to_data())
    if (
        freeze_store_receipt.object_kind != "panel-soft-task-freeze"
        or freeze_store_receipt.object_digest != freeze.record_digest
        or freeze_store_receipt.payload_digest != commit.exact_freeze_payload_digest
        or commit_store_receipt.object_kind != "panel-soft-task-freeze-commit"
        or commit_store_receipt.object_digest != commit.record_digest
        or commit.task_freeze_store_receipt_digest
        != freeze_store_receipt.record_digest
    ):
        raise PanelSoftEngineeringCampaignError(
            "query gate receipts do not bind the exact durable freeze"
        )
    released = ReleasedOfficialPanel.release(
        archive,
        panel_id,
        execution_precommit_digest=prepared.precommit.record_digest,
        exposure_successor_digest=prepared.successor.digest,
        expected_execution_precommit_digest=(
            prepared.authorization.execution_precommit_digest
        ),
        expected_exposure_successor_digest=(
            prepared.authorization.exposure_successor_digest
        ),
    )
    receipt = prepared.store.persist(
        object_kind="panel-soft-released-query-panel",
        object_digest=released.record_digest,
        data=released.to_data(),
    )
    reloaded = ReleasedOfficialPanel.from_data(
        prepared.store.verify(receipt, expected_data=released.to_data())
    )
    if reloaded != released:
        raise PanelSoftEngineeringCampaignError("query panel durable replay differs")
    return released, receipt


def _read_canonical_durable_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = _stable_private_read(path, maximum_bytes=24 * 1024 * 1024)
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise PanelSoftEngineeringCampaignError(f"{label} encoding differs")
    try:
        value = json.loads(payload[:-1].decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PanelSoftEngineeringCampaignError(f"{label} is malformed") from exc
    if not isinstance(value, dict) or canonical_json(value) + b"\n" != payload:
        raise PanelSoftEngineeringCampaignError(
            f"{label} is not canonical JSON"
        )
    return value


def _verify_rank_journal_binding(
    *,
    journal: ObjectBongardTextTurnJournalTransport,
    evidence: PanelSoftRankJournalEvidence,
    artifact: PanelSoftRankArtifact,
    rank_input: PanelSoftRankInput,
    prompt: str,
    output_schema: Mapping[str, Any],
    campaign: PreparedPanelSoftEngineeringCampaign,
    require_one_fresh_attempt: bool,
) -> ObjectBongardTurnJournalSummary:
    if type(journal) is not ObjectBongardTextTurnJournalTransport:
        raise TypeError("rank journal must be the exact typed text journal")
    provenance = panel_soft_rank_transport_provenance(journal)
    if (
        provenance.kind != "production_exactly_once_journal"
        or provenance.benchmark_sealable is not True
        or getattr(journal, "_underlying_transport", None)
        is not run_codex_text_structured
    ):
        raise PanelSoftEngineeringCampaignError(
            "rank journal is not externally classified benchmark-sealable custody"
        )
    summary = journal.verify()
    manifest = _read_canonical_durable_mapping(
        journal.manifest_path, "rank journal manifest"
    )
    result = _read_canonical_durable_mapping(
        journal.result_path, "rank journal result"
    )
    structured = result.get("codex_structured_result")
    runtime = campaign.runtime
    if (
        summary != evidence.journal_summary
        or summary.terminal_status != "success"
        or manifest.get("schema") != TURN_JOURNAL_MANIFEST_SCHEMA
        or manifest.get("record_digest") != summary.manifest_digest
        or manifest.get("authorization_digest")
        != campaign.release_authority.release_authorization_digest
        or manifest.get("execution_precommit_digest")
        != campaign.precommit.record_digest
        or manifest.get("task_id") != evidence.task_id
        or manifest.get("turn_kind") != evidence.turn_kind
        or manifest.get("modality") != "text_structured"
        or manifest.get("prompt") != prompt
        or manifest.get("prompt_sha256") != evidence.prompt_sha256
        or manifest.get("output_schema") != dict(output_schema)
        or manifest.get("output_schema_digest") != evidence.output_schema_digest
        or manifest.get("named_images") != []
        or manifest.get("runtime_binding") != runtime.binding
        or result.get("status") != "success"
        or result.get("record_digest") != summary.result_digest
        or not isinstance(structured, Mapping)
        or structured.get("payload") != dict(artifact.model_payload)
        or structured.get("receipt") != artifact.receipt.to_dict()
        or result.get("receipt_digest") != artifact.receipt.receipt_digest
        or result.get("payload_digest")
        != _content_address(dict(artifact.model_payload))
        or evidence.authorization_digest
        != campaign.release_authority.release_authorization_digest
        or evidence.execution_precommit_digest != campaign.precommit.record_digest
        or evidence.rank_input_digest != rank_input.rank_input_digest
        or evidence.rank_artifact_digest != artifact.artifact_digest
        or evidence.rank_receipt_digest != artifact.receipt.receipt_digest
        or evidence.rank_thread_id != artifact.receipt.thread_id
        or evidence.transport_provenance != provenance
        or (
            require_one_fresh_attempt
            and (
                journal.attempted_call_count,
                journal.fresh_call_count,
                journal.reused_call_count,
            )
            != (1, 1, 0)
        )
    ):
        raise PanelSoftEngineeringCampaignError(
            "rank journal manifest, terminal, prompt, input, or receipt differs"
        )
    verify_panel_soft_rank_artifact(
        artifact,
        version_space=rank_input.engineering_version_space,
        expected_artifact_digest=artifact.artifact_digest,
        model=runtime.model,
        reasoning_effort=runtime.reasoning_effort,
        expected_launcher_digest=runtime.expected_launcher_digest,
        cloud_policy_cache_snapshot=runtime.cloud_policy_cache_snapshot,
        model_catalog_snapshot=runtime.model_catalog_snapshot,
        no_tools_attestation=runtime.no_tools_attestation,
        require_benchmark_sealable=True,
        expected_transport_provenance=provenance,
    )
    return summary


def execute_panel_soft_engineering_campaign_task(
    task_plan: ObjectBongardTaskPlan,
    *,
    campaign: PreparedPanelSoftEngineeringCampaign,
    underlying_transport: Callable[..., CodexStructuredResult] = run_codex_named_images_structured,
) -> PanelSoftEngineeringCampaignTaskRecord:
    """Execute one task; query release is reachable only through runner callback."""

    if not isinstance(campaign, PreparedPanelSoftEngineeringCampaign):
        raise TypeError("campaign must be PreparedPanelSoftEngineeringCampaign")
    campaign.__post_init__()
    prepared = campaign.release
    archive = campaign.archive
    runtime = campaign.runtime
    mode = campaign.selection_mode
    task = _prepared_task(prepared, task_plan)
    if not isinstance(archive, OfficialPanelArchive):
        raise TypeError("archive must be OfficialPanelArchive")
    if archive.record_digest != prepared.authorization.archive_record_digest:
        raise PanelSoftEngineeringCampaignError("official panel archive differs")
    store = prepared.store
    release_authority = campaign.release_authority
    authority_receipt = _persist_mapping(
        store, kind="panel-soft-release-authority", digest=release_authority.record_digest,
        data=release_authority.to_data(),
    )
    if authority_receipt.object_digest != release_authority.record_digest:
        raise PanelSoftEngineeringCampaignError("release authority was not durably reloaded")
    support_ids = (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
    released: list[ReleasedOfficialPanel] = []
    release_receipts: list[ObjectBongardWriteOnceReceipt] = []
    journals: list[
        ObjectBongardNamedImageTurnJournalTransport
        | ObjectBongardTextTurnJournalTransport
    ] = []
    rank_durable: dict[str, object] = {}

    support_rows = tuple(
        release_object_bongard_support_panel(
            prepared=prepared, archive=archive, panel_id=panel_id
        )
        for panel_id in support_ids
    )
    supports = tuple(item[0] for item in support_rows)
    released.extend(supports)
    release_receipts.extend(item[1] for item in support_rows)
    support_pngs = tuple(item.exact_png_bytes for item in supports)
    support_map = dict(zip(support_ids, support_pngs, strict=True))
    journal_root = store.root / "panel-soft-turn-journals" / task.task_id

    def journal(
        turn_kind: str,
        prompt: str,
        images: Sequence[tuple[str, bytes]],
        schema: Mapping[str, Any],
    ) -> ObjectBongardNamedImageTurnJournalTransport:
        result = ObjectBongardNamedImageTurnJournalTransport(
            journal_root / turn_kind,
            authorization_digest=release_authority.release_authorization_digest,
            execution_precommit_digest=release_authority.execution_precommit_digest,
            task_id=task.task_id,
            turn_kind=turn_kind,
            expected_prompt=prompt,
            expected_images=images,
            expected_output_schema=schema,
            runtime=runtime,
            underlying_transport=underlying_transport,
        )
        journals.append(result)
        return result

    proposer_transport = journal(
        "proposer", panel_soft_proposer_prompt(),
        tuple(zip(PANEL_SOFT_PROPOSER_PRESENTATION_NAMES, support_pngs, strict=True)),
        panel_soft_proposer_output_schema(),
    )
    proposer = propose_panel_soft_atoms(
        support_pngs,
        support_panel_ids=support_ids,
        expected_support_sha256=tuple(hashlib.sha256(item).hexdigest() for item in support_pngs),
        **_runtime_kwargs(runtime),
        transport=proposer_transport,
    )
    _persist_mapping(
        store, kind="panel-soft-proposer-artifact",
        digest="sha256:" + proposer.artifact_digest, data=proposer.to_data(),
    )
    support_artifacts: tuple[PanelSoftObserverArtifact, ...] = ()
    if proposer.vocabulary is not None:
        artifacts: list[PanelSoftObserverArtifact] = []
        prompt = panel_soft_observer_prompt(proposer.vocabulary)
        schema = panel_soft_observer_output_schema(proposer.vocabulary)
        for panel_index, (panel_id, panel) in enumerate(zip(support_ids, support_pngs, strict=True)):
            transports = tuple(
                journal(
                    f"support-{panel_index:03d}-repeat-{repeat}", prompt,
                    (("panel.png", panel),), schema,
                )
                for repeat in range(2)
            )
            artifact = observe_panel_soft_vocabulary(
                panel, panel_id=panel_id, vocabulary=proposer.vocabulary,
                expected_panel_sha256=hashlib.sha256(panel).hexdigest(),
                expected_vocabulary_digest=proposer.vocabulary.vocabulary_digest,
                **_runtime_kwargs(runtime), transport=_SequencedTransport(transports),
            )
            _persist_mapping(
                store, kind="panel-soft-observer-artifact",
                digest="sha256:" + artifact.artifact_digest, data=artifact.to_data(),
            )
            artifacts.append(artifact)
        support_artifacts = tuple(artifacts)

    def ranker_callback(
        version_space: Any,
    ) -> PanelSoftRankArtifact:
        policy_snapshot = runtime.cloud_policy_cache_snapshot
        if not isinstance(policy_snapshot, CloudPolicyCacheSnapshot):
            raise PanelSoftEngineeringCampaignError(
                "ranked task lacks the prepared policy-cache snapshot"
            )
        rank_input = PanelSoftRankInput.freeze(version_space)
        rank_prompt = panel_soft_ranker_prompt(rank_input)
        rank_schema = panel_soft_ranker_output_schema(rank_input)
        rank_journal = ObjectBongardTextTurnJournalTransport(
            journal_root / "support-rank",
            authorization_digest=release_authority.release_authorization_digest,
            execution_precommit_digest=release_authority.execution_precommit_digest,
            task_id=task.task_id,
            turn_kind="support-rank",
            expected_prompt=rank_prompt,
            expected_output_schema=rank_schema,
            runtime=runtime,
            underlying_transport=run_codex_text_structured,
        )
        journals.append(rank_journal)
        provenance = panel_soft_rank_transport_provenance(rank_journal)
        if provenance.benchmark_sealable is not True:
            raise PanelSoftEngineeringCampaignError(
                "rank journal provenance is not benchmark-sealable"
            )
        try:
            artifact = rank_panel_soft_version_space(
                version_space,
                model=runtime.model,
                reasoning_effort=runtime.reasoning_effort,
                minutes=runtime.minutes,
                verbose=runtime.verbose,
                executable=runtime.executable,
                expected_launcher_digest=runtime.expected_launcher_digest,
                cloud_policy_cache_snapshot=policy_snapshot,
                model_catalog_snapshot=runtime.model_catalog_snapshot,
                no_tools_attestation=runtime.no_tools_attestation,
                transport=rank_journal,
                allow_unverified_transport=False,
            )
        except Exception as exc:
            summary = rank_journal.verify()
            if (
                rank_journal.attempted_call_count,
                rank_journal.fresh_call_count,
                rank_journal.reused_call_count,
                rank_journal.refused_call_count,
            ) != (1, 1, 0, 0):
                raise PanelSoftEngineeringCampaignError(
                    "rank failure did not come from one fresh terminal attempt"
                ) from exc
            failure = PanelSoftRankFailureEvidence.seal(
                task_id=task.task_id,
                authorization_digest=(
                    release_authority.release_authorization_digest
                ),
                execution_precommit_digest=(
                    release_authority.execution_precommit_digest
                ),
                rank_input=rank_input,
                prompt=rank_prompt,
                output_schema=rank_schema,
                transport_provenance=provenance,
                journal=rank_journal,
                journal_summary=summary,
                source_exception=exc,
            )
            failure_receipt = _persist_mapping(
                store,
                kind="panel-soft-rank-failure-evidence",
                digest=failure.record_digest,
                data=failure.to_data(),
            )
            reloaded_failure = PanelSoftRankFailureEvidence.from_data(
                store.verify(
                    failure_receipt, expected_data=failure.to_data()
                )
            )
            if reloaded_failure != failure:
                raise PanelSoftEngineeringCampaignError(
                    "rank failure evidence durable reload differs"
                ) from exc
            rank_durable.update(
                rank_input=rank_input,
                prompt=rank_prompt,
                schema=rank_schema,
                journal=rank_journal,
                failure_evidence=reloaded_failure,
                failure_evidence_receipt=failure_receipt,
            )
            raise
        verified = verify_panel_soft_rank_artifact(
            artifact,
            version_space=version_space,
            expected_artifact_digest=artifact.artifact_digest,
            model=runtime.model,
            reasoning_effort=runtime.reasoning_effort,
            expected_launcher_digest=runtime.expected_launcher_digest,
            cloud_policy_cache_snapshot=policy_snapshot,
            model_catalog_snapshot=runtime.model_catalog_snapshot,
            no_tools_attestation=runtime.no_tools_attestation,
            require_benchmark_sealable=True,
            expected_transport_provenance=provenance,
        )
        artifact_receipt = _persist_mapping(
            store,
            kind="panel-soft-rank-artifact",
            digest="sha256:" + verified.artifact_digest,
            data=verified.to_data(),
        )
        reloaded = PanelSoftRankArtifact.from_data(
            store.verify(artifact_receipt, expected_data=verified.to_data())
        )
        reloaded = verify_panel_soft_rank_artifact(
            reloaded,
            version_space=version_space,
            expected_artifact_digest=verified.artifact_digest,
            model=runtime.model,
            reasoning_effort=runtime.reasoning_effort,
            expected_launcher_digest=runtime.expected_launcher_digest,
            cloud_policy_cache_snapshot=policy_snapshot,
            model_catalog_snapshot=runtime.model_catalog_snapshot,
            no_tools_attestation=runtime.no_tools_attestation,
            require_benchmark_sealable=True,
            expected_transport_provenance=provenance,
        )
        summary = rank_journal.verify()
        evidence = PanelSoftRankJournalEvidence.seal(
            task_id=task.task_id,
            authorization_digest=release_authority.release_authorization_digest,
            execution_precommit_digest=release_authority.execution_precommit_digest,
            rank_input=rank_input,
            prompt=rank_prompt,
            output_schema=rank_schema,
            artifact=reloaded,
            transport_provenance=provenance,
            journal_summary=summary,
            rank_artifact_store_receipt=artifact_receipt,
        )
        _verify_rank_journal_binding(
            journal=rank_journal,
            evidence=evidence,
            artifact=reloaded,
            rank_input=rank_input,
            prompt=rank_prompt,
            output_schema=rank_schema,
            campaign=campaign,
            require_one_fresh_attempt=True,
        )
        evidence_receipt = _persist_mapping(
            store,
            kind="panel-soft-rank-journal-evidence",
            digest=evidence.record_digest,
            data=evidence.to_data(),
        )
        reloaded_evidence = PanelSoftRankJournalEvidence.from_data(
            store.verify(evidence_receipt, expected_data=evidence.to_data())
        )
        if reloaded_evidence != evidence:
            raise PanelSoftEngineeringCampaignError(
                "rank journal evidence durable reload differs"
            )
        rank_durable.update(
            artifact=reloaded,
            artifact_receipt=artifact_receipt,
            evidence=reloaded_evidence,
            evidence_receipt=evidence_receipt,
            journal=rank_journal,
            rank_input=rank_input,
            prompt=rank_prompt,
            schema=rank_schema,
        )
        return reloaded

    durable: dict[str, object] = {}

    def commit_freeze(payload: bytes) -> PanelSoftEngineeringTaskFreezeCommit:
        freeze = PanelSoftEngineeringTaskFreeze.from_data(json.loads(payload))
        freeze_receipt = _persist_mapping(
            store, kind="panel-soft-task-freeze", digest=freeze.record_digest,
            data=freeze.to_data(),
        )
        commit = PanelSoftEngineeringTaskFreezeCommit.seal(
            freeze, payload, task_freeze_store_receipt_digest=freeze_receipt.record_digest,
        )
        commit_receipt = _persist_mapping(
            store, kind="panel-soft-task-freeze-commit", digest=commit.record_digest,
            data=commit.to_data(),
        )
        durable.update(
            payload=payload,
            freeze=freeze,
            freeze_receipt=freeze_receipt,
            commit=commit,
            commit_receipt=commit_receipt,
        )
        return commit

    def reload_freeze(commit_data: Mapping[str, Any]) -> bytes:
        if "commit" not in durable or dict(commit_data) != durable["commit"].to_data():  # type: ignore[union-attr]
            raise PanelSoftEngineeringCampaignError("freeze reload commit differs")
        freeze_receipt = durable["freeze_receipt"]
        if not isinstance(freeze_receipt, ObjectBongardWriteOnceReceipt):
            raise PanelSoftEngineeringCampaignError("freeze receipt differs")
        decoded = store.verify(
            freeze_receipt,
            expected_data=json.loads(durable["payload"]),  # type: ignore[arg-type]
        )
        reloaded = canonical_json(dict(decoded)) + b"\n"
        if reloaded != durable["payload"]:
            raise PanelSoftEngineeringCampaignError("exact freeze bytes differ")
        return reloaded

    def query_source(
        freeze_data: Mapping[str, Any], commit_data: Mapping[str, Any]
    ) -> Mapping[str, tuple[bytes, PanelSoftObserverArtifact]]:
        if (
            durable.get("payload") != canonical_json(dict(freeze_data)) + b"\n"
            or "freeze" not in durable
            or dict(freeze_data) != durable["freeze"].to_data()  # type: ignore[union-attr]
            or "commit" not in durable
            or dict(commit_data) != durable["commit"].to_data()  # type: ignore[union-attr]
            or proposer.vocabulary is None
        ):
            raise PanelSoftEngineeringCampaignError("query callback freeze binding differs")
        freeze_for_rank = durable["freeze"]
        if not isinstance(freeze_for_rank, PanelSoftEngineeringTaskFreeze):
            raise PanelSoftEngineeringCampaignError(
                "query callback lacks the typed freeze"
            )
        if mode == "support_only_codex_ranker":
            rank_artifact = rank_durable.get("artifact")
            rank_artifact_receipt = rank_durable.get("artifact_receipt")
            rank_evidence = rank_durable.get("evidence")
            rank_evidence_receipt = rank_durable.get("evidence_receipt")
            rank_journal = rank_durable.get("journal")
            rank_input = rank_durable.get("rank_input")
            rank_prompt = rank_durable.get("prompt")
            rank_schema = rank_durable.get("schema")
            if (
                not isinstance(rank_artifact, PanelSoftRankArtifact)
                or not isinstance(
                    rank_artifact_receipt, ObjectBongardWriteOnceReceipt
                )
                or not isinstance(rank_evidence, PanelSoftRankJournalEvidence)
                or not isinstance(
                    rank_evidence_receipt, ObjectBongardWriteOnceReceipt
                )
                or not isinstance(
                    rank_journal, ObjectBongardTextTurnJournalTransport
                )
                or not isinstance(rank_input, PanelSoftRankInput)
                or not isinstance(rank_prompt, str)
                or not isinstance(rank_schema, Mapping)
                or freeze_for_rank.rank_artifact_benchmark_sealable is not True
                or freeze_for_rank.allow_unverified_rank_artifact is not False
                or freeze_for_rank.rank_artifact_digest
                != rank_artifact.artifact_digest
                or freeze_for_rank.rank_input_digest
                != rank_input.rank_input_digest
                or freeze_for_rank.rank_receipt_digest
                != rank_artifact.receipt.receipt_digest
                or rank_artifact_receipt.object_kind
                != "panel-soft-rank-artifact"
                or rank_artifact_receipt.object_digest
                != "sha256:" + rank_artifact.artifact_digest
                or rank_evidence_receipt.object_kind
                != "panel-soft-rank-journal-evidence"
                or rank_evidence_receipt.object_digest
                != rank_evidence.record_digest
                or rank_evidence.rank_artifact_store_receipt_digest
                != rank_artifact_receipt.record_digest
                or rank_evidence.rank_artifact_digest
                != rank_artifact.artifact_digest
                or rank_evidence.rank_input_digest
                != rank_input.rank_input_digest
                or rank_evidence.rank_receipt_digest
                != rank_artifact.receipt.receipt_digest
                or PanelSoftRankArtifact.from_data(
                    store.verify(
                        rank_artifact_receipt,
                        expected_data=rank_artifact.to_data(),
                    )
                )
                != rank_artifact
                or PanelSoftRankJournalEvidence.from_data(
                    store.verify(
                        rank_evidence_receipt,
                        expected_data=rank_evidence.to_data(),
                    )
                )
                != rank_evidence
            ):
                raise PanelSoftEngineeringCampaignError(
                    "query release lacks the exact durable rank artifact custody"
                )
            _verify_rank_journal_binding(
                journal=rank_journal,
                evidence=rank_evidence,
                artifact=rank_artifact,
                rank_input=rank_input,
                prompt=rank_prompt,
                output_schema=rank_schema,
                campaign=campaign,
                require_one_fresh_attempt=False,
            )
        elif rank_durable:
            raise PanelSoftEngineeringCampaignError(
                "baseline query unexpectedly contains rank custody"
            )
        rows: dict[str, tuple[bytes, PanelSoftObserverArtifact]] = {}
        prompt = panel_soft_observer_prompt(proposer.vocabulary)
        schema = panel_soft_observer_output_schema(proposer.vocabulary)
        for side_index, (side, panel_id) in enumerate(
            (("side_0", task.side_0_query_panel_id), ("side_1", task.side_1_query_panel_id))
        ):
            freeze = durable["freeze"]
            commit = durable["commit"]
            freeze_receipt = durable["freeze_receipt"]
            commit_receipt = durable["commit_receipt"]
            payload = durable["payload"]
            if (
                not isinstance(freeze, PanelSoftEngineeringTaskFreeze)
                or not isinstance(commit, PanelSoftEngineeringTaskFreezeCommit)
                or not isinstance(freeze_receipt, ObjectBongardWriteOnceReceipt)
                or not isinstance(commit_receipt, ObjectBongardWriteOnceReceipt)
                or not isinstance(payload, bytes)
            ):
                raise PanelSoftEngineeringCampaignError(
                    "durable query-gate state differs"
                )
            official, release_receipt = _release_panel_soft_engineering_query_panel(
                prepared=prepared,
                archive=archive,
                panel_id=panel_id,
                freeze=freeze,
                commit=commit,
                exact_freeze_payload=payload,
                freeze_store_receipt=freeze_receipt,
                commit_store_receipt=commit_receipt,
            )
            released.append(official)
            release_receipts.append(release_receipt)
            panel = official.exact_png_bytes
            transports = tuple(
                journal(
                    f"query-{side_index}-repeat-{repeat}", prompt,
                    (("panel.png", panel),), schema,
                )
                for repeat in range(2)
            )
            artifact = observe_panel_soft_vocabulary(
                panel, panel_id=panel_id, vocabulary=proposer.vocabulary,
                expected_panel_sha256=hashlib.sha256(panel).hexdigest(),
                expected_vocabulary_digest=proposer.vocabulary.vocabulary_digest,
                **_runtime_kwargs(runtime), transport=_SequencedTransport(transports),
            )
            _persist_mapping(
                store, kind="panel-soft-observer-artifact",
                digest="sha256:" + artifact.artifact_digest, data=artifact.to_data(),
            )
            rows[side] = (panel, artifact)
        return rows

    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        PanelSoftEngineeringCampaignError("failed proposer reached a later phase")
    )
    try:
        runner = run_panel_soft_engineering_task(
            task, proposer, support_map, support_artifacts,
            execution_precommit_digest=release_authority.execution_precommit_digest,
            selection_mode=mode,
            ranker=(
                ranker_callback if mode == "support_only_codex_ranker" else None
            ),
            allow_unverified_rank_artifact=False,
            freeze_committer=commit_freeze if proposer.vocabulary is not None else forbidden,
            freeze_reloader=reload_freeze if proposer.vocabulary is not None else forbidden,
            query_source=query_source if proposer.vocabulary is not None else forbidden,
        )
    except PanelSoftEngineeringTaskRunnerError as exc:
        failure = rank_durable.get("failure_evidence")
        failure_receipt = rank_durable.get("failure_evidence_receipt")
        rank_input = rank_durable.get("rank_input")
        if (
            mode != "support_only_codex_ranker"
            or not isinstance(failure, PanelSoftRankFailureEvidence)
            or not isinstance(
                failure_receipt, ObjectBongardWriteOnceReceipt
            )
            or not isinstance(rank_input, PanelSoftRankInput)
        ):
            if rank_durable.get("artifact") is not None:
                raise PanelSoftEngineeringCampaignError(
                    "post-rank runner rejection is a campaign-integrity fatal; "
                    "it cannot be converted into denominator evidence"
                ) from exc
            raise
        runner = PanelSoftEngineeringRankTerminal.seal(
            task_plan=task,
            execution_precommit_digest=(
                release_authority.execution_precommit_digest
            ),
            proposer_artifact=proposer,
            support_png_by_panel_id=support_map,
            support_artifacts=support_artifacts,
            engineering_version_space=rank_input.engineering_version_space,
            rank_failure_evidence=failure,
            rank_failure_evidence_store_receipt=failure_receipt,
        )
    runner_kind = (
        "panel-soft-task-archive"
        if isinstance(runner, PanelSoftEngineeringTaskRunArchive)
        else (
            "panel-soft-rank-terminal"
            if isinstance(runner, PanelSoftEngineeringRankTerminal)
            else "panel-soft-proposer-terminal"
        )
    )
    _persist_mapping(
        store, kind=runner_kind, digest="sha256:" + runner.record_digest,
        data=runner.to_data(),
    )
    summaries = tuple(journal_item.verify() for journal_item in journals)
    rank_artifact = (
        runner.rank_artifact
        if isinstance(runner, PanelSoftEngineeringTaskRunArchive)
        else None
    )
    if (
        not isinstance(runner, PanelSoftEngineeringRankTerminal)
        and (rank_artifact is None) != (not rank_durable)
    ):
        raise PanelSoftEngineeringCampaignError(
            "runner rank artifact and pre-freeze durable custody differ"
        )
    record = PanelSoftEngineeringCampaignTaskRecord.seal(
        task_plan=task, selection_mode=mode,
        release_authority_digest=release_authority.record_digest,
        execution_precommit_digest=prepared.precommit.record_digest,
        exposure_successor_digest=prepared.successor.digest,
        selector_artifact_digest=(
            None if rank_artifact is None else rank_artifact.artifact_digest
        ),
        selector_call_identity=(
            None
            if rank_artifact is None
            else (
                rank_artifact.receipt.receipt_digest,
                rank_artifact.receipt.thread_id,
            )
        ),
        rank_artifact_store_receipt=(
            rank_durable.get("artifact_receipt")
        ),
        rank_journal_evidence=(
            rank_durable.get("evidence")
        ),
        rank_journal_evidence_store_receipt=(
            rank_durable.get("evidence_receipt")
        ),
        released_panels=released,
        release_store_receipts=release_receipts,
        runner_record=runner,
        turn_journal_summaries=summaries,
    )
    _persist_mapping(
        store, kind="panel-soft-campaign-task", digest="sha256:" + record.record_digest,
        data=record.to_data(),
    )
    return PanelSoftEngineeringCampaignTaskRecord.from_data(record.to_data())


def _cold_verify_stored_mapping(
    store: ObjectBongardReleaseStore,
    *,
    kind: str,
    digest: str,
    data: Mapping[str, Any],
) -> ObjectBongardWriteOnceReceipt:
    receipt = _expected_store_receipt(
        object_kind=kind, object_digest=digest, data=data
    )
    if dict(store.verify(receipt, expected_data=data)) != dict(data):
        raise PanelSoftEngineeringCampaignError(
            f"cold replay of {kind} differs"
        )
    return receipt


def _expected_task_journals(
    campaign: PreparedPanelSoftEngineeringCampaign,
    record: PanelSoftEngineeringCampaignTaskRecord,
) -> tuple[
    ObjectBongardNamedImageTurnJournalTransport
    | ObjectBongardTextTurnJournalTransport,
    ...,
]:
    task = record.task_plan
    runner = record.runner_record
    root = (
        campaign.release.store.root
        / "panel-soft-turn-journals"
        / task.task_id
    )
    support_panels = record.released_panels[:12]
    support_pngs = tuple(item.exact_png_bytes for item in support_panels)
    specifications: list[
        tuple[
            str, str, str, tuple[tuple[str, bytes], ...], Mapping[str, Any]
        ]
    ] = [
        (
            "named",
            "proposer",
            panel_soft_proposer_prompt(),
            tuple(
                zip(
                    PANEL_SOFT_PROPOSER_PRESENTATION_NAMES,
                    support_pngs,
                    strict=True,
                )
            ),
            panel_soft_proposer_output_schema(),
        )
    ]
    vocabulary = runner.proposer_artifact.vocabulary
    if vocabulary is not None:
        prompt = panel_soft_observer_prompt(vocabulary)
        schema = panel_soft_observer_output_schema(vocabulary)
        for panel_index, panel in enumerate(support_pngs):
            for repeat in range(2):
                specifications.append(
                    (
                        "named",
                        f"support-{panel_index:03d}-repeat-{repeat}",
                        prompt,
                        (("panel.png", panel),),
                        schema,
                    )
                )
        if (
            isinstance(runner, PanelSoftEngineeringTaskRunArchive)
            and runner.rank_artifact is not None
        ):
            rank_input = runner.rank_artifact.rank_input
            specifications.append(
                (
                    "text",
                    "support-rank",
                    panel_soft_ranker_prompt(rank_input),
                    (),
                    panel_soft_ranker_output_schema(rank_input),
                )
            )
        elif isinstance(runner, PanelSoftEngineeringRankTerminal):
            rank_input = PanelSoftRankInput.freeze(
                runner.engineering_version_space
            )
            specifications.append(
                (
                    "text",
                    "support-rank",
                    panel_soft_ranker_prompt(rank_input),
                    (),
                    panel_soft_ranker_output_schema(rank_input),
                )
            )
        if (
            isinstance(runner, PanelSoftEngineeringTaskRunArchive)
            and runner.status is PanelSoftEngineeringTaskRunStatus.COMPLETE
        ):
            for side_index, panel in enumerate(record.released_panels[12:]):
                for repeat in range(2):
                    specifications.append(
                        (
                            "named",
                            f"query-{side_index}-repeat-{repeat}",
                            prompt,
                            (("panel.png", panel.exact_png_bytes),),
                            schema,
                        )
                    )
    expected_names = {item[1] for item in specifications}
    if (
        not root.is_dir()
        or root.is_symlink()
        or {item.name for item in root.iterdir()} != expected_names
    ):
        raise PanelSoftEngineeringCampaignError(
            "turn journal directory inventory differs"
        )
    forbidden_transport = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        PanelSoftEngineeringCampaignError("cold replay attempted a model call")
    )
    journals: list[
        ObjectBongardNamedImageTurnJournalTransport
        | ObjectBongardTextTurnJournalTransport
    ] = []
    for modality, turn_kind, prompt, images, schema in specifications:
        directory = root / turn_kind
        if (
            not directory.is_dir()
            or directory.is_symlink()
            or {item.name for item in directory.iterdir()}
            != {"manifest.json", "claim.json", "result.json", "outcome.json"}
        ):
            raise PanelSoftEngineeringCampaignError(
                "terminal turn journal inventory differs"
            )
        if modality == "text":
            journals.append(
                ObjectBongardTextTurnJournalTransport(
                    directory,
                    authorization_digest=(
                        campaign.release_authority.release_authorization_digest
                    ),
                    execution_precommit_digest=record.execution_precommit_digest,
                    task_id=task.task_id,
                    turn_kind=turn_kind,
                    expected_prompt=prompt,
                    expected_output_schema=schema,
                    runtime=campaign.runtime,
                    underlying_transport=run_codex_text_structured,
                )
            )
        else:
            journals.append(ObjectBongardNamedImageTurnJournalTransport(
                directory,
                authorization_digest=(
                    campaign.release_authority.release_authorization_digest
                ),
                execution_precommit_digest=record.execution_precommit_digest,
                task_id=task.task_id,
                turn_kind=turn_kind,
                expected_prompt=prompt,
                expected_images=images,
                expected_output_schema=schema,
                runtime=campaign.runtime,
                underlying_transport=forbidden_transport,
            ))
    return tuple(journals)


def cold_replay_panel_soft_engineering_campaign_task(
    campaign: PreparedPanelSoftEngineeringCampaign,
    value: PanelSoftEngineeringCampaignTaskRecord,
    *,
    expected_record_digest: str,
) -> PanelSoftEngineeringCampaignTaskRecord:
    if not isinstance(campaign, PreparedPanelSoftEngineeringCampaign):
        raise TypeError("campaign must be PreparedPanelSoftEngineeringCampaign")
    campaign.__post_init__()
    restored = PanelSoftEngineeringCampaignTaskRecord.from_data(value.to_data())
    if restored.record_digest != _raw_digest(expected_record_digest, "expected campaign task digest"):
        raise PanelSoftEngineeringCampaignError("campaign task commitment differs")
    replayed = (
        PanelSoftEngineeringRankTerminal.from_data(
            restored.runner_record.to_data()
        )
        if isinstance(restored.runner_record, PanelSoftEngineeringRankTerminal)
        else cold_replay_panel_soft_engineering_task(
            restored.runner_record,
            expected_record_digest=restored.runner_record.record_digest,
        )
    )
    if replayed != restored.runner_record:
        raise PanelSoftEngineeringCampaignError("runner cold replay differs")
    if (
        restored.release_authority_digest
        != campaign.release_authority.record_digest
        or restored.task_plan
        not in campaign.plan.tasks
    ):
        raise PanelSoftEngineeringCampaignError(
            "campaign task replay custody differs"
        )
    store = campaign.release.store
    _cold_verify_stored_mapping(
        store,
        kind="panel-soft-campaign-task",
        digest="sha256:" + restored.record_digest,
        data=restored.to_data(),
    )
    runner_kind = (
        "panel-soft-task-archive"
        if isinstance(restored.runner_record, PanelSoftEngineeringTaskRunArchive)
        else (
            "panel-soft-rank-terminal"
            if isinstance(restored.runner_record, PanelSoftEngineeringRankTerminal)
            else "panel-soft-proposer-terminal"
        )
    )
    _cold_verify_stored_mapping(
        store,
        kind=runner_kind,
        digest="sha256:" + restored.runner_record.record_digest,
        data=restored.runner_record.to_data(),
    )
    artifacts: list[PanelSoftProposerArtifact | PanelSoftObserverArtifact] = [
        restored.runner_record.proposer_artifact
    ]
    if isinstance(
        restored.runner_record,
        (PanelSoftEngineeringTaskRunArchive, PanelSoftEngineeringRankTerminal),
    ):
        artifacts.extend(restored.runner_record.support_artifacts)
    if isinstance(restored.runner_record, PanelSoftEngineeringTaskRunArchive):
        artifacts.extend(restored.runner_record.query_artifacts)
        if restored.runner_record.freeze is not None:
            _cold_verify_stored_mapping(
                store,
                kind="panel-soft-task-freeze",
                digest=restored.runner_record.freeze.record_digest,
                data=restored.runner_record.freeze.to_data(),
            )
        if restored.runner_record.freeze_commit is not None:
            _cold_verify_stored_mapping(
                store,
                kind="panel-soft-task-freeze-commit",
                digest=restored.runner_record.freeze_commit.record_digest,
                data=restored.runner_record.freeze_commit.to_data(),
            )
    for artifact in artifacts:
        if isinstance(artifact, PanelSoftProposerArtifact):
            kind = "panel-soft-proposer-artifact"
        else:
            kind = "panel-soft-observer-artifact"
        _cold_verify_stored_mapping(
            store,
            kind=kind,
            digest="sha256:" + artifact.artifact_digest,
            data=artifact.to_data(),
        )
    for panel, receipt in zip(
        restored.released_panels,
        restored.release_store_receipts,
        strict=True,
    ):
        store.verify(receipt, expected_data=panel.to_data())
        panel.cold_verify(
            campaign.archive,
            expected_execution_precommit_digest=(
                campaign.release.precommit.record_digest
            ),
            expected_exposure_successor_digest=campaign.release.successor.digest,
        )
    expected_journals = _expected_task_journals(campaign, restored)
    verified_summaries = tuple(journal.verify() for journal in expected_journals)
    if verified_summaries != restored.turn_journal_summaries:
        raise PanelSoftEngineeringCampaignError(
            "turn journal summaries differ from durable terminal journals"
        )
    rank_artifact = (
        restored.runner_record.rank_artifact
        if isinstance(
            restored.runner_record, PanelSoftEngineeringTaskRunArchive
        )
        else None
    )
    if rank_artifact is not None:
        evidence = restored.rank_journal_evidence
        artifact_receipt = restored.rank_artifact_store_receipt
        evidence_receipt = restored.rank_journal_evidence_store_receipt
        rank_journals = tuple(
            item
            for item in expected_journals
            if isinstance(item, ObjectBongardTextTurnJournalTransport)
        )
        if (
            not isinstance(evidence, PanelSoftRankJournalEvidence)
            or not isinstance(artifact_receipt, ObjectBongardWriteOnceReceipt)
            or not isinstance(evidence_receipt, ObjectBongardWriteOnceReceipt)
            or len(rank_journals) != 1
            or PanelSoftRankArtifact.from_data(
                store.verify(
                    artifact_receipt, expected_data=rank_artifact.to_data()
                )
            )
            != rank_artifact
            or PanelSoftRankJournalEvidence.from_data(
                store.verify(evidence_receipt, expected_data=evidence.to_data())
            )
            != evidence
        ):
            raise PanelSoftEngineeringCampaignError(
                "cold rank artifact custody differs"
            )
        rank_input = rank_artifact.rank_input
        _verify_rank_journal_binding(
            journal=rank_journals[0],
            evidence=evidence,
            artifact=rank_artifact,
            rank_input=rank_input,
            prompt=panel_soft_ranker_prompt(rank_input),
            output_schema=panel_soft_ranker_output_schema(rank_input),
            campaign=campaign,
            require_one_fresh_attempt=False,
        )
    elif any(
        item is not None
        for item in (
            restored.rank_artifact_store_receipt,
            restored.rank_journal_evidence,
            restored.rank_journal_evidence_store_receipt,
        )
    ):
        raise PanelSoftEngineeringCampaignError(
            "rank custody exists without a rank artifact"
        )
    if isinstance(restored.runner_record, PanelSoftEngineeringRankTerminal):
        terminal = restored.runner_record
        failure = terminal.rank_failure_evidence
        rank_journals = tuple(
            item
            for item in expected_journals
            if isinstance(item, ObjectBongardTextTurnJournalTransport)
        )
        if (
            len(rank_journals) != 1
            or PanelSoftRankFailureEvidence.from_data(
                store.verify(
                    terminal.rank_failure_evidence_store_receipt,
                    expected_data=failure.to_data(),
                )
            )
            != failure
            or rank_journals[0].verify() != failure.journal_summary
        ):
            raise PanelSoftEngineeringCampaignError(
                "rank terminal journal custody differs"
            )
        rank_input = PanelSoftRankInput.freeze(
            terminal.engineering_version_space
        )
        rank_prompt = panel_soft_ranker_prompt(rank_input)
        rank_schema = panel_soft_ranker_output_schema(rank_input)
        provenance = panel_soft_rank_transport_provenance(rank_journals[0])
        manifest = _read_canonical_durable_mapping(
            rank_journals[0].manifest_path,
            "rank terminal journal manifest",
        )
        durable_successful_identity = _rank_terminal_journal_successful_identity(
            rank_journals[0]
        )
        if (
            provenance != failure.transport_provenance
            or provenance.benchmark_sealable is not True
            or manifest.get("record_digest")
            != failure.journal_summary.manifest_digest
            or manifest.get("authorization_digest")
            != failure.authorization_digest
            or manifest.get("execution_precommit_digest")
            != failure.execution_precommit_digest
            or manifest.get("task_id") != terminal.task_plan.task_id
            or manifest.get("turn_kind") != "support-rank"
            or manifest.get("prompt") != rank_prompt
            or manifest.get("prompt_sha256")
            != hashlib.sha256(rank_prompt.encode("utf-8", errors="strict")).hexdigest()
            or failure.prompt_sha256 != manifest.get("prompt_sha256")
            or manifest.get("output_schema") != rank_schema
            or manifest.get("output_schema_digest")
            != _content_address(rank_schema)
            or failure.output_schema_digest
            != manifest.get("output_schema_digest")
            or manifest.get("runtime_binding") != campaign.runtime.binding
            or failure.rank_input_digest != rank_input.rank_input_digest
            or failure.successful_call_identity != durable_successful_identity
        ):
            raise PanelSoftEngineeringCampaignError(
                "rank terminal manifest differs"
            )
    return restored


def _campaign_metrics(
    records: Sequence[PanelSoftEngineeringCampaignTaskRecord],
) -> tuple[int, int, int, int]:
    rows = tuple(_runner_metrics(item.runner_record) for item in records)
    return tuple(sum(row[index] for row in rows) for index in range(4))  # type: ignore[return-value]


def _campaign_call_identities(
    records: Sequence[PanelSoftEngineeringCampaignTaskRecord],
) -> tuple[tuple[str, str], ...]:
    rows = tuple(
        identity for record in records for identity in record.successful_call_identities
    )
    if (
        len({row[0] for row in rows}) != len(rows)
        or len({row[1] for row in rows}) != len(rows)
    ):
        raise PanelSoftEngineeringCampaignError(
            "campaign reuses a receipt digest or thread ID across tasks"
        )
    return rows


def _campaign_content(value: "PanelSoftEngineeringCampaignRecord") -> dict[str, object]:
    correct, determinate, abstain, errors = _campaign_metrics(value.task_records)
    selection_attempts = sum(
        _selection_model_attempt_count(item.runner_record)
        for item in value.task_records
    )
    successful_selection_calls = sum(
        _successful_selection_model_call_count(item.runner_record)
        for item in value.task_records
    )
    complete = sum(
        isinstance(item.runner_record, PanelSoftEngineeringTaskRunArchive)
        and item.runner_record.status is PanelSoftEngineeringTaskRunStatus.COMPLETE
        for item in value.task_records
    )
    return {
        "schema": PANEL_SOFT_CAMPAIGN_SCHEMA,
        "campaign_id": PANEL_SOFT_CAMPAIGN_ID,
        "campaign_source_digest": value.campaign_source_digest,
        "predicate_pair_selection_mode": value.selection_mode,
        "workers": value.workers,
        "task_execution_model": (
            "thread-pool-task-isolated-deterministic-plan-order"
        ),
        "task_records_in_frozen_plan_order": True,
        "batch_plan": value.plan.to_data(),
        "batch_plan_digest": value.plan.record_digest,
        "release_authority": value.release_authority.to_data(),
        "release_authority_digest": value.release_authority.record_digest,
        "task_records": [item.to_data() for item in value.task_records],
        "task_record_digests": [item.record_digest for item in value.task_records],
        "selected_task_count": len(value.task_records),
        "complete_task_count": complete,
        "correct_count": correct,
        "determinate_count": determinate,
        "abstain_count": abstain,
        "error_count": errors,
        "query_denominator": PANEL_SOFT_CAMPAIGN_QUERY_DENOMINATOR,
        "accuracy_ppm": (
            correct * 1_000_000 // PANEL_SOFT_CAMPAIGN_QUERY_DENOMINATOR
        ),
        "coverage_ppm": (
            determinate * 1_000_000 // PANEL_SOFT_CAMPAIGN_QUERY_DENOMINATOR
        ),
        "query_release_count": sum(
            len(item.released_panels) - 12 for item in value.task_records
        ),
        "selection_model_attempt_count": selection_attempts,
        "successful_selection_model_call_count": successful_selection_calls,
        "successful_model_call_count": len(value.global_call_identities),
        "successful_model_call_counts_by_task": [
            {
                "task_id": item.task_plan.task_id,
                "successful_model_call_count": len(
                    item.successful_call_identities
                ),
            }
            for item in value.task_records
        ],
        "terminal_turn_count": sum(
            len(item.turn_journal_summaries) for item in value.task_records
        ),
        "global_call_identities": [list(row) for row in value.global_call_identities],
        "receipt_digests_globally_unique": True,
        "thread_ids_globally_unique": True,
        "fixed_denominator_includes_abstain_and_error": True,
        "task_records_are_cold_replay_capable_without_model_calls": True,
        **_authority_data(value.selection_mode),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringCampaignRecord:
    campaign_source_digest: str
    selection_mode: str
    workers: int
    plan: ObjectBongardBatchPlan
    release_authority: PanelSoftCampaignReleaseAuthority
    task_records: tuple[PanelSoftEngineeringCampaignTaskRecord, ...]
    global_call_identities: tuple[tuple[str, str], ...]
    record_digest: str

    def __post_init__(self) -> None:
        _raw_digest(self.campaign_source_digest, "campaign source digest")
        mode = _selection_mode(self.selection_mode)
        _worker_count(self.workers)
        _raw_digest(self.record_digest, "campaign record digest")
        plan = ObjectBongardBatchPlan.from_data(self.plan.to_data())
        authority = PanelSoftCampaignReleaseAuthority.from_data(
            self.release_authority.to_data()
        )
        records = tuple(
            PanelSoftEngineeringCampaignTaskRecord.from_data(item.to_data())
            for item in self.task_records
        )
        correct, determinate, abstain, errors = _campaign_metrics(records)
        if (
            self.campaign_source_digest
            != panel_soft_engineering_campaign_source_digest()
            or plan != self.plan
            or authority != self.release_authority
            or authority.selection_mode != mode
            or authority.batch_plan_digest != plan.record_digest
            or records != self.task_records
            or len(records) != PANEL_SOFT_CAMPAIGN_TASK_COUNT
            or tuple(item.task_plan for item in records) != plan.tasks
            or any(item.selection_mode != mode for item in records)
            or any(
                item.release_authority_digest != authority.record_digest
                or item.execution_precommit_digest
                != authority.execution_precommit_digest
                or item.exposure_successor_digest
                != authority.exposure_successor_digest
                for item in records
            )
            or determinate + abstain + errors
            != PANEL_SOFT_CAMPAIGN_QUERY_DENOMINATOR
            or correct > determinate
            or self.global_call_identities != _campaign_call_identities(records)
            or self.record_digest != canonical_digest(_campaign_content(self))
        ):
            raise PanelSoftEngineeringCampaignError("campaign aggregate differs")

    @classmethod
    def seal(
        cls,
        *,
        plan: ObjectBongardBatchPlan,
        release_authority: PanelSoftCampaignReleaseAuthority,
        task_records: Sequence[PanelSoftEngineeringCampaignTaskRecord],
        workers: int,
    ) -> "PanelSoftEngineeringCampaignRecord":
        records = tuple(task_records)
        values = {
            "campaign_source_digest": panel_soft_engineering_campaign_source_digest(),
            "selection_mode": release_authority.selection_mode,
            "workers": _worker_count(workers),
            "plan": plan,
            "release_authority": release_authority,
            "task_records": records,
            "global_call_identities": _campaign_call_identities(records),
        }
        provisional = object.__new__(cls)
        for key, item in values.items():
            object.__setattr__(provisional, key, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=canonical_digest(_campaign_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_campaign_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "PanelSoftEngineeringCampaignRecord":
        raw = _fields(
            value,
            set(_campaign_content_fields()) | {"record_digest"},
            "panel-soft campaign",
        )
        mode = _selection_mode(raw["predicate_pair_selection_mode"])
        if (
            raw["schema"] != PANEL_SOFT_CAMPAIGN_SCHEMA
            or raw["campaign_id"] != PANEL_SOFT_CAMPAIGN_ID
            or _worker_count(raw["workers"]) != raw["workers"]
            or raw["task_execution_model"]
            != "thread-pool-task-isolated-deterministic-plan-order"
            or raw["task_records_in_frozen_plan_order"] is not True
            or raw["query_denominator"] != PANEL_SOFT_CAMPAIGN_QUERY_DENOMINATOR
            or raw["receipt_digests_globally_unique"] is not True
            or raw["thread_ids_globally_unique"] is not True
            or raw["fixed_denominator_includes_abstain_and_error"] is not True
            or raw[
                "task_records_are_cold_replay_capable_without_model_calls"
            ] is not True
            or any(
                type(raw[key]) is not type(item) or raw[key] != item
                for key, item in _authority_data(mode).items()
            )
            or not isinstance(raw["task_records"], list)
            or not isinstance(raw["global_call_identities"], list)
        ):
            raise PanelSoftEngineeringCampaignError("campaign policy differs")
        result = cls(
            raw["campaign_source_digest"],
            mode,
            raw["workers"],
            ObjectBongardBatchPlan.from_data(raw["batch_plan"]),
            PanelSoftCampaignReleaseAuthority.from_data(raw["release_authority"]),
            tuple(
                PanelSoftEngineeringCampaignTaskRecord.from_data(item)
                for item in raw["task_records"]
            ),
            tuple(tuple(item) for item in raw["global_call_identities"]),
            raw["record_digest"],
        )
        correct, determinate, abstain, errors = _campaign_metrics(result.task_records)
        selection_attempts = sum(
            _selection_model_attempt_count(item.runner_record)
            for item in result.task_records
        )
        successful_selection_calls = sum(
            _successful_selection_model_call_count(item.runner_record)
            for item in result.task_records
        )
        complete = sum(
            isinstance(item.runner_record, PanelSoftEngineeringTaskRunArchive)
            and item.runner_record.status is PanelSoftEngineeringTaskRunStatus.COMPLETE
            for item in result.task_records
        )
        if (
            raw["batch_plan_digest"] != result.plan.record_digest
            or raw["release_authority_digest"]
            != result.release_authority.record_digest
            or raw["task_record_digests"]
            != [item.record_digest for item in result.task_records]
            or raw["selected_task_count"] != len(result.task_records)
            or raw["complete_task_count"] != complete
            or (
                raw["correct_count"], raw["determinate_count"],
                raw["abstain_count"], raw["error_count"],
            )
            != (correct, determinate, abstain, errors)
            or raw["accuracy_ppm"]
            != correct * 1_000_000 // PANEL_SOFT_CAMPAIGN_QUERY_DENOMINATOR
            or raw["coverage_ppm"]
            != determinate * 1_000_000 // PANEL_SOFT_CAMPAIGN_QUERY_DENOMINATOR
            or raw["query_release_count"]
            != sum(len(item.released_panels) - 12 for item in result.task_records)
            or any(
                type(raw[key]) is not int
                for key in (
                    "selected_task_count", "complete_task_count",
                    "correct_count", "determinate_count", "abstain_count",
                    "error_count", "query_denominator", "accuracy_ppm",
                    "coverage_ppm", "query_release_count",
                    "selection_model_attempt_count",
                    "successful_selection_model_call_count",
                    "successful_model_call_count", "terminal_turn_count",
                )
            )
            or raw["selection_model_attempt_count"] != selection_attempts
            or raw["successful_selection_model_call_count"]
            != successful_selection_calls
            or raw["successful_model_call_count"]
            != len(result.global_call_identities)
            or raw["successful_model_call_counts_by_task"]
            != [
                {
                    "task_id": item.task_plan.task_id,
                    "successful_model_call_count": len(
                        item.successful_call_identities
                    ),
                }
                for item in result.task_records
            ]
            or raw["terminal_turn_count"]
            != sum(
                len(item.turn_journal_summaries)
                for item in result.task_records
            )
            or canonical_json(result.to_data()) != canonical_json(dict(raw))
        ):
            raise PanelSoftEngineeringCampaignError("campaign metrics differ")
        return result


def _campaign_content_fields() -> tuple[str, ...]:
    return (
        "schema", "campaign_id", "campaign_source_digest",
        "predicate_pair_selection_mode", "workers", "task_execution_model",
        "task_records_in_frozen_plan_order", "batch_plan", "batch_plan_digest",
        "release_authority", "release_authority_digest", "task_records",
        "task_record_digests", "selected_task_count", "complete_task_count",
        "correct_count", "determinate_count", "abstain_count", "error_count",
        "query_denominator", "accuracy_ppm", "coverage_ppm",
        "query_release_count", "selection_model_attempt_count",
        "successful_selection_model_call_count",
        "successful_model_call_count", "successful_model_call_counts_by_task",
        "terminal_turn_count",
        "global_call_identities", "receipt_digests_globally_unique",
        "thread_ids_globally_unique", "fixed_denominator_includes_abstain_and_error",
        "task_records_are_cold_replay_capable_without_model_calls",
        *_authority_data(),
    )


def _campaign_replay_content(
    value: "PanelSoftEngineeringCampaignReplayReceipt",
) -> dict[str, object]:
    return {
        "schema": PANEL_SOFT_CAMPAIGN_REPLAY_SCHEMA,
        "campaign_record_digest": value.campaign_record_digest,
        "externally_supplied_expected_campaign_digest": (
            value.expected_campaign_digest
        ),
        "batch_plan_digest": value.batch_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "release_authorization_digest": value.release_authorization_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "source_manifest_digest": value.source_manifest_digest,
        "runtime_evidence_digest": value.runtime_evidence_digest,
        "successor_mirror_evidence_digest": (
            value.successor_mirror_evidence_digest
        ),
        "task_record_digests": list(value.task_record_digests),
        "externally_anchored": True,
        "prepared_store_archive_runtime_mirror_release_and_journals_verified": True,
        "model_calls_made": 0,
        **_authority_data(value.selection_mode),
    }


@dataclass(frozen=True, slots=True)
class PanelSoftEngineeringCampaignReplayReceipt:
    selection_mode: str
    campaign_record_digest: str
    expected_campaign_digest: str
    batch_plan_digest: str
    execution_precommit_digest: str
    release_authorization_digest: str
    exposure_successor_digest: str
    source_manifest_digest: str
    runtime_evidence_digest: str
    successor_mirror_evidence_digest: str
    task_record_digests: tuple[str, ...]
    record_digest: str

    def __post_init__(self) -> None:
        _selection_mode(self.selection_mode)
        for name in ("campaign_record_digest", "expected_campaign_digest"):
            _raw_digest(getattr(self, name), name)
        for name in (
            "batch_plan_digest", "execution_precommit_digest",
            "release_authorization_digest", "exposure_successor_digest",
            "source_manifest_digest", "runtime_evidence_digest",
            "successor_mirror_evidence_digest", "record_digest",
        ):
            _address(getattr(self, name), name)
        if (
            self.campaign_record_digest != self.expected_campaign_digest
            or type(self.task_record_digests) is not tuple
            or len(self.task_record_digests) != PANEL_SOFT_CAMPAIGN_TASK_COUNT
            or any(
                _RAW_DIGEST.fullmatch(item) is None
                for item in self.task_record_digests
            )
            or self.record_digest != _content_address(
                _campaign_replay_content(self)
            )
        ):
            raise PanelSoftEngineeringCampaignError(
                "campaign replay receipt differs"
            )

    @classmethod
    def seal(
        cls,
        campaign: PreparedPanelSoftEngineeringCampaign,
        record: PanelSoftEngineeringCampaignRecord,
        *,
        expected_campaign_digest: str,
    ) -> "PanelSoftEngineeringCampaignReplayReceipt":
        values = {
            "selection_mode": campaign.selection_mode,
            "campaign_record_digest": record.record_digest,
            "expected_campaign_digest": expected_campaign_digest,
            "batch_plan_digest": campaign.plan.record_digest,
            "execution_precommit_digest": campaign.precommit.record_digest,
            "release_authorization_digest": (
                campaign.release.authorization.record_digest
            ),
            "exposure_successor_digest": campaign.release.successor.digest,
            "source_manifest_digest": (
                "sha256:" + campaign.source_manifest.manifest_digest
            ),
            "runtime_evidence_digest": campaign.runtime_evidence["record_digest"],
            "successor_mirror_evidence_digest": (
                campaign.research_exposure_successor_evidence["record_digest"]
            ),
            "task_record_digests": tuple(
                item.record_digest for item in record.task_records
            ),
        }
        provisional = object.__new__(cls)
        for key, item in values.items():
            object.__setattr__(provisional, key, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_content_address(_campaign_replay_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_campaign_replay_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "PanelSoftEngineeringCampaignReplayReceipt":
        raw = _fields(
            value,
            set(_campaign_replay_content_fields()) | {"record_digest"},
            "campaign replay receipt",
        )
        mode = _selection_mode(raw["predicate_pair_selection_mode"])
        if (
            raw["schema"] != PANEL_SOFT_CAMPAIGN_REPLAY_SCHEMA
            or raw["externally_anchored"] is not True
            or raw[
                "prepared_store_archive_runtime_mirror_release_and_journals_verified"
            ] is not True
            or type(raw["model_calls_made"]) is not int
            or raw["model_calls_made"] != 0
            or not isinstance(raw["task_record_digests"], list)
            or any(
                type(raw[key]) is not type(item) or raw[key] != item
                for key, item in _authority_data(mode).items()
            )
        ):
            raise PanelSoftEngineeringCampaignError(
                "campaign replay receipt policy differs"
            )
        result = cls(
            mode,
            raw["campaign_record_digest"],
            raw["externally_supplied_expected_campaign_digest"],
            raw["batch_plan_digest"],
            raw["execution_precommit_digest"],
            raw["release_authorization_digest"],
            raw["exposure_successor_digest"],
            raw["source_manifest_digest"],
            raw["runtime_evidence_digest"],
            raw["successor_mirror_evidence_digest"],
            tuple(raw["task_record_digests"]),
            raw["record_digest"],
        )
        if canonical_json(result.to_data()) != canonical_json(dict(raw)):
            raise PanelSoftEngineeringCampaignError(
                "campaign replay receipt is not canonical"
            )
        return result


def _campaign_replay_content_fields() -> tuple[str, ...]:
    return (
        "schema", "campaign_record_digest",
        "externally_supplied_expected_campaign_digest", "batch_plan_digest",
        "execution_precommit_digest", "release_authorization_digest",
        "exposure_successor_digest", "source_manifest_digest",
        "runtime_evidence_digest", "successor_mirror_evidence_digest",
        "task_record_digests", "externally_anchored",
        "prepared_store_archive_runtime_mirror_release_and_journals_verified",
        "model_calls_made", *_authority_data(),
    )


def execute_panel_soft_engineering_campaign(
    prepared_campaign: PreparedPanelSoftEngineeringCampaign,
    *,
    underlying_transport: Callable[..., CodexStructuredResult] = (
        run_codex_named_images_structured
    ),
) -> PanelSoftEngineeringCampaignRecord:
    """Execute all three sealed tasks and persist one denominator-six record."""

    if not isinstance(prepared_campaign, PreparedPanelSoftEngineeringCampaign):
        raise TypeError("prepared_campaign must be PreparedPanelSoftEngineeringCampaign")
    prepared_campaign.__post_init__()
    with ThreadPoolExecutor(
        max_workers=prepared_campaign.workers,
        thread_name_prefix="panel-soft-task",
    ) as executor:
        futures = tuple(
            executor.submit(
                execute_panel_soft_engineering_campaign_task,
                task,
                campaign=prepared_campaign,
                underlying_transport=underlying_transport,
            )
            for task in prepared_campaign.plan.tasks
        )
        records = tuple(future.result() for future in futures)
    record = PanelSoftEngineeringCampaignRecord.seal(
        plan=prepared_campaign.plan,
        release_authority=prepared_campaign.release_authority,
        task_records=records,
        workers=prepared_campaign.workers,
    )
    _persist_mapping(
        prepared_campaign.release.store,
        kind="panel-soft-campaign",
        digest="sha256:" + record.record_digest,
        data=record.to_data(),
    )
    return PanelSoftEngineeringCampaignRecord.from_data(record.to_data())


def cold_replay_panel_soft_engineering_campaign(
    prepared_campaign: PreparedPanelSoftEngineeringCampaign,
    value: PanelSoftEngineeringCampaignRecord,
    *,
    expected_record_digest: str,
) -> PanelSoftEngineeringCampaignReplayReceipt:
    if not isinstance(prepared_campaign, PreparedPanelSoftEngineeringCampaign):
        raise TypeError(
            "prepared_campaign must be PreparedPanelSoftEngineeringCampaign"
        )
    prepared_campaign.__post_init__()
    restored = PanelSoftEngineeringCampaignRecord.from_data(value.to_data())
    if restored.record_digest != _raw_digest(
        expected_record_digest, "expected campaign digest"
    ):
        raise PanelSoftEngineeringCampaignError("campaign commitment differs")
    if (
        restored.plan != prepared_campaign.plan
        or restored.release_authority != prepared_campaign.release_authority
        or restored.workers != prepared_campaign.workers
    ):
        raise PanelSoftEngineeringCampaignError(
            "campaign replay prepared custody differs"
        )
    _cold_verify_stored_mapping(
        prepared_campaign.release.store,
        kind="panel-soft-campaign",
        digest="sha256:" + restored.record_digest,
        data=restored.to_data(),
    )
    for task in restored.task_records:
        if cold_replay_panel_soft_engineering_campaign_task(
            prepared_campaign,
            task,
            expected_record_digest=task.record_digest,
        ) != task:
            raise PanelSoftEngineeringCampaignError("campaign task replay differs")
    receipt = PanelSoftEngineeringCampaignReplayReceipt.seal(
        prepared_campaign,
        restored,
        expected_campaign_digest=expected_record_digest,
    )
    persisted = _persist_mapping(
        prepared_campaign.release.store,
        kind="panel-soft-campaign-replay-receipt",
        digest=receipt.record_digest,
        data=receipt.to_data(),
    )
    if persisted.object_digest != receipt.record_digest:
        raise PanelSoftEngineeringCampaignError(
            "campaign replay receipt persistence differs"
        )
    return PanelSoftEngineeringCampaignReplayReceipt.from_data(receipt.to_data())


def _compact_campaign_completion_summary(
    record: PanelSoftEngineeringCampaignRecord,
) -> dict[str, object]:
    restored = PanelSoftEngineeringCampaignRecord.from_data(record.to_data())
    correct, determinate, abstain, errors = _campaign_metrics(
        restored.task_records
    )
    return {
        "schema": PANEL_SOFT_CAMPAIGN_COMPLETION_SUMMARY_SCHEMA,
        "campaign_id": PANEL_SOFT_CAMPAIGN_ID,
        "campaign_record_digest": restored.record_digest,
        "campaign_store_object_digest": "sha256:" + restored.record_digest,
        "predicate_pair_selection_mode": restored.selection_mode,
        "workers": restored.workers,
        "selected_task_count": len(restored.task_records),
        "complete_task_count": sum(
            isinstance(item.runner_record, PanelSoftEngineeringTaskRunArchive)
            and item.runner_record.status
            is PanelSoftEngineeringTaskRunStatus.COMPLETE
            for item in restored.task_records
        ),
        "correct_count": correct,
        "determinate_count": determinate,
        "abstain_count": abstain,
        "error_count": errors,
        "query_denominator": PANEL_SOFT_CAMPAIGN_QUERY_DENOMINATOR,
        "query_release_count": sum(
            len(item.released_panels) - 12 for item in restored.task_records
        ),
        "selection_model_attempt_count": sum(
            _selection_model_attempt_count(item.runner_record)
            for item in restored.task_records
        ),
        "successful_selection_model_call_count": sum(
            _successful_selection_model_call_count(item.runner_record)
            for item in restored.task_records
        ),
        "successful_model_call_count": len(restored.global_call_identities),
        "terminal_turn_count": sum(
            len(item.turn_journal_summaries) for item in restored.task_records
        ),
        "full_campaign_record_printed": False,
        "contains_panel_pixels_or_base64": False,
    }


def _compact_campaign_replay_summary(
    receipt: PanelSoftEngineeringCampaignReplayReceipt,
) -> dict[str, object]:
    restored = PanelSoftEngineeringCampaignReplayReceipt.from_data(
        receipt.to_data()
    )
    return {
        "schema": PANEL_SOFT_CAMPAIGN_REPLAY_SUMMARY_SCHEMA,
        "campaign_record_digest": restored.campaign_record_digest,
        "externally_supplied_expected_campaign_digest": (
            restored.expected_campaign_digest
        ),
        "replay_receipt_digest": restored.record_digest,
        "externally_anchored": True,
        "model_calls_made": 0,
        "full_campaign_record_printed": False,
        "contains_panel_pixels_or_base64": False,
    }


def _await_external_campaign_digest_and_replay(
    prepared: PreparedPanelSoftEngineeringCampaign,
    record: PanelSoftEngineeringCampaignRecord,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
) -> PanelSoftEngineeringCampaignReplayReceipt:
    """Emit the digest, then consume one externally supplied raw digest line."""

    print(
        canonical_json(_compact_campaign_completion_summary(record)).decode(
            "utf-8"
        ),
        file=output_stream,
        flush=True,
    )
    line = input_stream.readline()
    if (
        not isinstance(line, str)
        or len(line) != 65
        or not line.endswith("\n")
        or _RAW_DIGEST.fullmatch(line[:-1]) is None
    ):
        raise PanelSoftEngineeringCampaignError(
            "external campaign digest input must be exactly one raw SHA-256 line"
        )
    receipt = cold_replay_panel_soft_engineering_campaign(
        prepared,
        record,
        expected_record_digest=line[:-1],
    )
    print(
        canonical_json(_compact_campaign_replay_summary(receipt)).decode(
            "utf-8"
        ),
        file=output_stream,
        flush=True,
    )
    return receipt


def run_panel_soft_engineering_campaign_command(
    output_root: str | os.PathLike[str],
    *,
    selection_mode: str = "support_only_codex_ranker",
    descriptor_path: str | os.PathLike[str] = DEFAULT_DESCRIPTOR,
    archive_path: str | os.PathLike[str] = DEFAULT_ARCHIVE,
    split_path: str | os.PathLike[str] = DEFAULT_SPLIT,
    predecessor_path: str | os.PathLike[str] = DEFAULT_PREDECESSOR,
    historical_exposure_path: str | os.PathLike[str] = DEFAULT_HISTORICAL_EXPOSURE,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    minutes: int = DEFAULT_MINUTES,
    verbose: bool = False,
    executable: str = DEFAULT_CODEX_EXECUTABLE,
    expected_launcher_sha256: str = DEFAULT_CODEX_LAUNCHER_SHA256,
    workers: int = DEFAULT_WORKERS,
    expected_campaign_digest: str | None = None,
    await_external_campaign_digest: bool = False,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    underlying_transport: Callable[..., CodexStructuredResult] = (
        run_codex_named_images_structured
    ),
) -> PanelSoftEngineeringCampaignRecord:
    """Freshly attest, prepare, execute, and persist the campaign.

    Cold replay is performed only when a caller supplies an external campaign
    digest.  A freshly produced record is never accepted as its own trust root.
    """

    if type(await_external_campaign_digest) is not bool:
        raise TypeError("await_external_campaign_digest must be bool")
    if await_external_campaign_digest and expected_campaign_digest is not None:
        raise PanelSoftEngineeringCampaignError(
            "awaited and pre-supplied campaign digests are mutually exclusive"
        )
    prepared = prepare_panel_soft_engineering_campaign(
        output_root=output_root,
        selection_mode=selection_mode,
        descriptor_path=descriptor_path,
        archive_path=archive_path,
        split_path=split_path,
        predecessor_path=predecessor_path,
        historical_exposure_path=historical_exposure_path,
        selection_seed=selection_seed,
        model=model,
        reasoning_effort=reasoning_effort,
        minutes=minutes,
        verbose=verbose,
        executable=executable,
        expected_launcher_sha256=expected_launcher_sha256,
        workers=workers,
    )
    record = execute_panel_soft_engineering_campaign(
        prepared, underlying_transport=underlying_transport
    )
    if expected_campaign_digest is not None:
        cold_replay_panel_soft_engineering_campaign(
            prepared,
            record,
            expected_record_digest=expected_campaign_digest,
        )
    elif await_external_campaign_digest:
        _await_external_campaign_digest_and_replay(
            prepared,
            record,
            input_stream=sys.stdin if input_stream is None else input_stream,
            output_stream=sys.stdout if output_stream is None else output_stream,
        )
    return record


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact-unused TRAIN panel-soft engineering campaign; "
            "never an official benchmark."
        )
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--selection-mode",
        choices=PANEL_SOFT_SELECTION_MODES,
        default="support_only_codex_ranker",
    )
    parser.add_argument("--descriptor-path", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--archive-path", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--predecessor-path", type=Path, default=DEFAULT_PREDECESSOR)
    parser.add_argument(
        "--historical-exposure-path",
        type=Path,
        default=DEFAULT_HISTORICAL_EXPOSURE,
    )
    parser.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--executable", default=DEFAULT_CODEX_EXECUTABLE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--expected-launcher-sha256",
        default=DEFAULT_CODEX_LAUNCHER_SHA256,
    )
    digest_mode = parser.add_mutually_exclusive_group()
    digest_mode.add_argument(
        "--expected-campaign-digest",
        help=(
            "external raw SHA-256 commitment required to issue a cold-replay "
            "receipt; never inferred from the new result"
        ),
    )
    digest_mode.add_argument(
        "--await-external-campaign-digest",
        action="store_true",
        help=(
            "after persistence, print and flush a compact digest summary, then "
            "read exactly one externally supplied raw SHA-256 line from stdin"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    record = run_panel_soft_engineering_campaign_command(
        arguments.output_root,
        selection_mode=arguments.selection_mode,
        descriptor_path=arguments.descriptor_path,
        archive_path=arguments.archive_path,
        split_path=arguments.split_path,
        predecessor_path=arguments.predecessor_path,
        historical_exposure_path=arguments.historical_exposure_path,
        selection_seed=arguments.selection_seed,
        model=arguments.model,
        reasoning_effort=arguments.reasoning_effort,
        minutes=arguments.minutes,
        verbose=arguments.verbose,
        executable=arguments.executable,
        expected_launcher_sha256=arguments.expected_launcher_sha256,
        workers=arguments.workers,
        expected_campaign_digest=arguments.expected_campaign_digest,
        await_external_campaign_digest=(
            arguments.await_external_campaign_digest
        ),
    )
    if not arguments.await_external_campaign_digest:
        print(
            canonical_json(_compact_campaign_completion_summary(record)).decode(
                "utf-8"
            ),
            flush=True,
        )
    return 0


__all__ = (
    "DEFAULT_CODEX_LAUNCHER_SHA256", "DEFAULT_PLAN_DIGEST",
    "DEFAULT_PREDECESSOR_LEDGER_DIGEST",
    "DEFAULT_SELECTED_TASK_IDS", "DEFAULT_SELECTION_SEED",
    "PANEL_SOFT_CAMPAIGN_QUERY_DENOMINATOR", "PANEL_SOFT_SELECTION_MODES",
    "PanelSoftCampaignReleaseAuthority", "PanelSoftEngineeringCampaignError",
    "PanelSoftEngineeringCampaignRecord",
    "PanelSoftEngineeringCampaignReplayReceipt",
    "PanelSoftEngineeringCampaignTaskRecord",
    "PreparedPanelSoftEngineeringCampaign",
    "cold_replay_panel_soft_engineering_campaign",
    "cold_replay_panel_soft_engineering_campaign_task",
    "execute_panel_soft_engineering_campaign",
    "execute_panel_soft_engineering_campaign_task",
    "main", "panel_soft_engineering_campaign_source_bindings",
    "panel_soft_engineering_campaign_source_digest",
    "prepare_panel_soft_engineering_campaign",
    "run_panel_soft_engineering_campaign_command",
)


if __name__ == "__main__":  # pragma: no cover - exercised by the real command.
    raise SystemExit(main())
