"""Durable exposure and panel-release gate for broad object Bongard batches.

This module is deliberately below the task runner.  It accepts only the
metadata-only batch plan, content-addressed execution bindings, and the pinned
official archive.  All selected tasks are recorded in one durable exposure
successor before :class:`OfficialPanelArchive` is allowed to return PNG bytes.

Support panels are authorized by the frozen batch plan.  A sealed query panel
has an additional boundary: the caller must present a canonical task freeze
and a canonical decision commit, and both exact byte strings must already have
been persisted and reloaded by this module's write-once store.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass, field
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureEvent, ExposureLedger
from bongard.object_bongard_batch import (
    ObjectBongardBatchPlan,
    ObjectBongardTaskPlan,
    object_bongard_batch_algorithm_digest,
    object_bongard_batch_source_digest,
    object_bongard_task_inventory_digest,
)
from bongard.object_bongard_drill_batch import (
    ObjectBongardDrillBatchPlan,
    object_bongard_drill_batch_algorithm_digest,
    object_bongard_drill_batch_source_digest,
)
from bongard.official_panel_archive import OfficialPanelArchive, ReleasedOfficialPanel
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.release import OfficialReleaseDescriptor


PRECOMMIT_SCHEMA = "gkm.bongard-object-execution-precommit.v1"
AUTHORIZATION_SCHEMA = "gkm.bongard-object-release-authorization.v1"
STORE_RECEIPT_SCHEMA = "gkm.bongard-object-write-once-receipt.v1"
EXPOSURE_PHASE = "object-bongard-support-release"

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_CONFIG_KEY = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_FORBIDDEN_CONFIG_KEY = re.compile(r"(?:pixel|image|png|action|program|path|bytes)")
_MAX_OBJECT_BYTES = 64 * 1024 * 1024

_PRODUCTION_TASK_DECISION_TYPES = {
    (
        "bongard.object_scene_anchor_task_decision_custody",
        "ObjectSceneAnchorTaskDecisionFreeze",
        "ObjectSceneAnchorTaskDecisionCommit",
    ): "object-scene-anchor",
    (
        "bongard.panel_feature_task_runner",
        "PanelFeatureTaskFreeze",
        "PanelFeatureTaskFreezeCommit",
    ): "panel-feature",
    (
        "bongard.object_bongard_rubric_task_runner",
        "ObjectBongardRubricTaskFreeze",
        "ObjectBongardRubricTaskFreezeCommit",
    ): "object-bongard-rubric",
    (
        "bongard.panel_program_official_task",
        "PanelProgramOfficialTaskFreeze",
        "PanelProgramOfficialTaskCommit",
    ): "panel-program",
}


class ObjectBongardReleaseGateError(RuntimeError):
    """A precommit, durable record, exposure transition, or release is invalid."""


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _bytes_address(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardReleaseGateError(f"{label} must be a sha256: address")
    return value


def _require_raw_digest(value: object, label: str) -> str:
    if type(value) is not str or _RAW_DIGEST.fullmatch(value) is None:
        raise ObjectBongardReleaseGateError(f"{label} must be a raw SHA-256 digest")
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_release_or_replay": False,
    }


def object_bongard_release_gate_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _sorted_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ObjectBongardReleaseGateError(f"{label} must be a sequence")
    result = tuple(values)
    if (
        any(type(item) is not str or not item for item in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ObjectBongardReleaseGateError(f"{label} must be unique and sorted")
    return result


def _freeze_bindings(values: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if type(values) is not dict or not values:
        raise ObjectBongardReleaseGateError("runtime source bindings must be nonempty")
    result = tuple(sorted(values.items()))
    if any(type(key) is not str or _CONFIG_KEY.fullmatch(key) is None for key, _ in result):
        raise ObjectBongardReleaseGateError("runtime source binding name is invalid")
    for key, value in result:
        _require_address(value, f"runtime source binding {key}")
    return result


def _freeze_configuration(
    values: Mapping[str, str | int | bool],
) -> tuple[tuple[str, str | int | bool], ...]:
    if type(values) is not dict:
        raise ObjectBongardReleaseGateError("configuration must be a mapping")
    result = tuple(sorted(values.items()))
    for key, value in result:
        if (
            type(key) is not str
            or _CONFIG_KEY.fullmatch(key) is None
            or _FORBIDDEN_CONFIG_KEY.search(key) is not None
            or type(value) not in (str, int, bool)
            or (type(value) is str and (not value or "\x00" in value or len(value.encode()) > 512))
        ):
            raise ObjectBongardReleaseGateError(
                "configuration must contain bounded metadata scalars and no visual/action inputs"
            )
    return result


ObjectBongardReleasePlan = ObjectBongardBatchPlan | ObjectBongardDrillBatchPlan


def _plan_algorithm_digest(plan: ObjectBongardReleasePlan) -> str:
    if isinstance(plan, ObjectBongardDrillBatchPlan):
        return object_bongard_drill_batch_algorithm_digest()
    if isinstance(plan, ObjectBongardBatchPlan):
        return object_bongard_batch_algorithm_digest()
    raise TypeError("plan must be an object Bongard release plan")


def _plan_source_digest(plan: ObjectBongardReleasePlan) -> str:
    if isinstance(plan, ObjectBongardDrillBatchPlan):
        return object_bongard_drill_batch_source_digest()
    if isinstance(plan, ObjectBongardBatchPlan):
        return object_bongard_batch_source_digest()
    raise TypeError("plan must be an object Bongard release plan")


def _all_panels(plan: ObjectBongardReleasePlan) -> tuple[tuple[str, ...], tuple[str, ...]]:
    support = tuple(
        sorted(
            panel_id
            for task in plan.tasks
            for panel_id in (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids)
        )
    )
    query = tuple(
        sorted(
            panel_id
            for task in plan.tasks
            for panel_id in (task.side_0_query_panel_id, task.side_1_query_panel_id)
        )
    )
    return support, query


def _precommit_content(value: "ObjectBongardExecutionPrecommit") -> dict[str, object]:
    return {
        "schema": PRECOMMIT_SCHEMA,
        "batch_plan_digest": value.batch_plan_digest,
        "batch_algorithm_digest": value.batch_algorithm_digest,
        "batch_source_digest": value.batch_source_digest,
        "release_gate_source_digest": value.release_gate_source_digest,
        "release_descriptor_digest": value.release_descriptor_digest,
        "archive_record_digest": value.archive_record_digest,
        "archive_digest": value.archive_digest,
        "archive_central_directory_digest": value.archive_central_directory_digest,
        "corpus_digest": value.corpus_digest,
        "exposure_predecessor_digest": value.exposure_predecessor_digest,
        "task_inventory_digest": value.task_inventory_digest,
        "train_task_ids_digest": value.train_task_ids_digest,
        "exact_used_task_ids_digest": value.exact_used_task_ids_digest,
        "selected_task_ids": list(value.selected_task_ids),
        "authorized_support_panel_ids": list(value.authorized_support_panel_ids),
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "runtime_source_bindings": [list(row) for row in value.runtime_source_bindings],
        "configuration": [list(row) for row in value.configuration],
        "exposure_observed_at": value.exposure_observed_at,
        "exposure_actor": value.exposure_actor,
        "exposure_purpose": value.exposure_purpose,
        "exposure_source": value.exposure_source,
        "selection_inputs_include_pixels": False,
        "selection_inputs_include_action_programs": False,
        "official_test_authorized": False,
        "query_identities_sealed_before_support_pixels": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardExecutionPrecommit:
    batch_plan_digest: str
    batch_algorithm_digest: str
    batch_source_digest: str
    release_gate_source_digest: str
    release_descriptor_digest: str
    archive_record_digest: str
    archive_digest: str
    archive_central_directory_digest: str
    corpus_digest: str
    exposure_predecessor_digest: str
    task_inventory_digest: str
    train_task_ids_digest: str
    exact_used_task_ids_digest: str
    selected_task_ids: tuple[str, ...]
    authorized_support_panel_ids: tuple[str, ...]
    sealed_query_panel_ids: tuple[str, ...]
    runtime_source_bindings: tuple[tuple[str, str], ...]
    configuration: tuple[tuple[str, str | int | bool], ...]
    exposure_observed_at: str
    exposure_actor: str
    exposure_purpose: str
    exposure_source: str
    record_digest: str

    def __post_init__(self) -> None:
        for name in (
            "batch_plan_digest", "batch_algorithm_digest", "batch_source_digest",
            "release_gate_source_digest", "release_descriptor_digest",
            "archive_record_digest", "archive_digest", "archive_central_directory_digest",
            "corpus_digest", "exposure_predecessor_digest", "task_inventory_digest",
            "train_task_ids_digest", "exact_used_task_ids_digest", "record_digest",
        ):
            _require_address(getattr(self, name), name)
        for values, label in (
            (self.selected_task_ids, "selected task IDs"),
            (self.authorized_support_panel_ids, "support panel IDs"),
            (self.sealed_query_panel_ids, "query panel IDs"),
        ):
            if values != tuple(sorted(set(values))) or not values:
                raise ObjectBongardReleaseGateError(f"{label} differ")
        if set(self.authorized_support_panel_ids) & set(self.sealed_query_panel_ids):
            raise ObjectBongardReleaseGateError("support/query panels overlap")
        _freeze_bindings(dict(self.runtime_source_bindings))
        _freeze_configuration(dict(self.configuration))
        if any(type(item) is not str or not item for item in (
            self.exposure_observed_at, self.exposure_actor,
            self.exposure_purpose, self.exposure_source,
        )):
            raise ObjectBongardReleaseGateError("exposure metadata must be nonempty")
        if self.record_digest != _address(_precommit_content(self)):
            raise ObjectBongardReleaseGateError("execution precommit digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_precommit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, raw: Mapping[str, Any]) -> "ObjectBongardExecutionPrecommit":
        expected = {
            "schema", "batch_plan_digest", "batch_algorithm_digest", "batch_source_digest",
            "release_gate_source_digest", "release_descriptor_digest", "archive_record_digest",
            "archive_digest", "archive_central_directory_digest", "corpus_digest",
            "exposure_predecessor_digest", "task_inventory_digest", "train_task_ids_digest",
            "exact_used_task_ids_digest", "selected_task_ids", "authorized_support_panel_ids",
            "sealed_query_panel_ids", "runtime_source_bindings", "configuration",
            "exposure_observed_at", "exposure_actor", "exposure_purpose", "exposure_source",
            "selection_inputs_include_pixels", "selection_inputs_include_action_programs",
            "official_test_authorized", "query_identities_sealed_before_support_pixels",
            *_authority_data(), "record_digest",
        }
        if type(raw) is not dict or set(raw) != expected:
            raise ObjectBongardReleaseGateError("execution precommit fields differ")
        if (
            raw["schema"] != PRECOMMIT_SCHEMA
            or raw["selection_inputs_include_pixels"] is not False
            or raw["selection_inputs_include_action_programs"] is not False
            or raw["official_test_authorized"] is not False
            or raw["query_identities_sealed_before_support_pixels"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not all(type(raw[key]) is list for key in (
                "selected_task_ids", "authorized_support_panel_ids", "sealed_query_panel_ids",
                "runtime_source_bindings", "configuration",
            ))
        ):
            raise ObjectBongardReleaseGateError("execution precommit policy differs")
        try:
            result = cls(
                batch_plan_digest=raw["batch_plan_digest"],
                batch_algorithm_digest=raw["batch_algorithm_digest"],
                batch_source_digest=raw["batch_source_digest"],
                release_gate_source_digest=raw["release_gate_source_digest"],
                release_descriptor_digest=raw["release_descriptor_digest"],
                archive_record_digest=raw["archive_record_digest"],
                archive_digest=raw["archive_digest"],
                archive_central_directory_digest=raw["archive_central_directory_digest"],
                corpus_digest=raw["corpus_digest"],
                exposure_predecessor_digest=raw["exposure_predecessor_digest"],
                task_inventory_digest=raw["task_inventory_digest"],
                train_task_ids_digest=raw["train_task_ids_digest"],
                exact_used_task_ids_digest=raw["exact_used_task_ids_digest"],
                selected_task_ids=tuple(raw["selected_task_ids"]),
                authorized_support_panel_ids=tuple(raw["authorized_support_panel_ids"]),
                sealed_query_panel_ids=tuple(raw["sealed_query_panel_ids"]),
                runtime_source_bindings=tuple(tuple(row) for row in raw["runtime_source_bindings"]),
                configuration=tuple(tuple(row) for row in raw["configuration"]),
                exposure_observed_at=raw["exposure_observed_at"],
                exposure_actor=raw["exposure_actor"],
                exposure_purpose=raw["exposure_purpose"],
                exposure_source=raw["exposure_source"],
                record_digest=raw["record_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise ObjectBongardReleaseGateError("execution precommit is malformed") from exc
        if result.to_data() != dict(raw):
            raise ObjectBongardReleaseGateError("execution precommit is not canonical")
        return result


def create_object_bongard_execution_precommit(
    *,
    plan: ObjectBongardReleasePlan,
    predecessor: ExposureLedger,
    descriptor: OfficialReleaseDescriptor,
    archive: OfficialPanelArchive,
    task_ids: Sequence[str],
    train_task_ids: Sequence[str],
    exact_used_task_ids: Sequence[str],
    runtime_source_bindings: Mapping[str, str],
    configuration: Mapping[str, str | int | bool],
    exposure_observed_at: str,
    exposure_actor: str = "headless-codex-proposer",
    exposure_purpose: str = "broad-object-predicate-support-and-sealed-query",
    exposure_source: str = "official-shapebongard-v2-archive",
) -> ObjectBongardExecutionPrecommit:
    if type(plan) not in (ObjectBongardBatchPlan, ObjectBongardDrillBatchPlan):
        raise TypeError("plan must be an exact object Bongard release plan")
    if type(predecessor) is not ExposureLedger:
        raise TypeError("predecessor must be exact ExposureLedger")
    if type(descriptor) is not OfficialReleaseDescriptor:
        raise TypeError("descriptor must be exact OfficialReleaseDescriptor")
    if type(archive) is not OfficialPanelArchive:
        raise TypeError("archive must be exact OfficialPanelArchive")
    plan = type(plan).from_data(plan.to_data())
    predecessor = ExposureLedger.from_dict(predecessor.to_dict())
    descriptor = OfficialReleaseDescriptor.from_dict(descriptor.to_dict())
    inventory = _sorted_ids(task_ids, "official task inventory")
    train = _sorted_ids(train_task_ids, "TRAIN task inventory")
    used = _sorted_ids(exact_used_task_ids, "exact-used task inventory")
    selected = tuple(task.task_id for task in plan.tasks)
    support, query = _all_panels(plan)
    if (
        not set(train) <= set(inventory)
        or not set(used) <= set(inventory)
        or not set(selected) <= set(train)
        or set(selected) & (set(used) | set(predecessor.exposed_task_ids))
    ):
        raise ObjectBongardReleaseGateError("selected tasks are not exact-unused TRAIN tasks")
    expected_inventory = object_bongard_task_inventory_digest(inventory)
    if (
        plan.task_inventory_digest != expected_inventory
        or descriptor.task_ids_sha256 != expected_inventory
        or plan.train_task_ids_digest != _address(list(train))
        or plan.exact_used_task_ids_digest != _address(list(used))
        or plan.exposure_predecessor_digest != predecessor.digest
        or predecessor.corpus_digest != descriptor.corpus_manifest_sha256
        or plan.release_descriptor_digest != descriptor.digest
        or archive.release_descriptor_digest != descriptor.digest
        or archive.archive_digest != descriptor.archive_sha256
    ):
        raise ObjectBongardReleaseGateError("plan, inventory, exposure, and official release differ")
    archive_members = {name for name, _size, _crc in archive.members}
    expected_members = {
        f"ShapeBongard_V2/{panel.split('/', 1)[0]}/images/{panel.split('/', 1)[1]}"
        for panel in (*support, *query)
    }
    if not expected_members <= archive_members:
        raise ObjectBongardReleaseGateError("selected panel is absent from archive inventory")
    bindings = dict(runtime_source_bindings)
    automatic = {
        "batch_source": "sha256:" + _plan_source_digest(plan),
        "release_gate_source": "sha256:" + object_bongard_release_gate_source_digest(),
    }
    for key, value in automatic.items():
        if key in bindings and bindings[key] != value:
            raise ObjectBongardReleaseGateError(f"automatic source binding {key} differs")
        bindings[key] = value
    values: dict[str, object] = {
        "batch_plan_digest": plan.record_digest,
        "batch_algorithm_digest": _plan_algorithm_digest(plan),
        "batch_source_digest": automatic["batch_source"],
        "release_gate_source_digest": automatic["release_gate_source"],
        "release_descriptor_digest": descriptor.digest,
        "archive_record_digest": archive.record_digest,
        "archive_digest": archive.archive_digest,
        "archive_central_directory_digest": archive.central_directory_digest,
        "corpus_digest": predecessor.corpus_digest,
        "exposure_predecessor_digest": predecessor.digest,
        "task_inventory_digest": expected_inventory,
        "train_task_ids_digest": _address(list(train)),
        "exact_used_task_ids_digest": _address(list(used)),
        "selected_task_ids": selected,
        "authorized_support_panel_ids": support,
        "sealed_query_panel_ids": query,
        "runtime_source_bindings": _freeze_bindings(bindings),
        "configuration": _freeze_configuration(configuration),
        "exposure_observed_at": exposure_observed_at,
        "exposure_actor": exposure_actor,
        "exposure_purpose": exposure_purpose,
        "exposure_source": exposure_source,
    }
    provisional = object.__new__(ObjectBongardExecutionPrecommit)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    return ObjectBongardExecutionPrecommit(
        **values,  # type: ignore[arg-type]
        record_digest=_address(_precommit_content(provisional)),
    )


def _receipt_content(value: "ObjectBongardWriteOnceReceipt") -> dict[str, object]:
    return {
        "schema": STORE_RECEIPT_SCHEMA,
        "object_kind": value.object_kind,
        "object_digest": value.object_digest,
        "payload_digest": value.payload_digest,
        "size_bytes": value.size_bytes,
        "relative_path": value.relative_path,
        "persisted_and_reloaded": True,
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardWriteOnceReceipt:
    object_kind: str
    object_digest: str
    payload_digest: str
    size_bytes: int
    relative_path: str
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.object_kind) is not str or _NAME.fullmatch(self.object_kind) is None:
            raise ObjectBongardReleaseGateError("stored object kind is invalid")
        for name in ("object_digest", "payload_digest", "record_digest"):
            _require_address(getattr(self, name), name)
        expected_path = f"objects/{self.object_kind}/{self.object_digest[7:]}.json"
        if (
            type(self.size_bytes) is not int
            or not 0 < self.size_bytes <= _MAX_OBJECT_BYTES
            or type(self.relative_path) is not str
            or self.relative_path != expected_path
            or self.record_digest != _address(_receipt_content(self))
        ):
            raise ObjectBongardReleaseGateError("write-once receipt differs")

    def to_data(self) -> dict[str, object]:
        return {**_receipt_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, raw: Mapping[str, Any]) -> "ObjectBongardWriteOnceReceipt":
        expected = {
            "schema", "object_kind", "object_digest", "payload_digest",
            "size_bytes", "relative_path", "persisted_and_reloaded", "record_digest",
        }
        if (
            type(raw) is not dict
            or set(raw) != expected
            or raw["schema"] != STORE_RECEIPT_SCHEMA
            or raw["persisted_and_reloaded"] is not True
        ):
            raise ObjectBongardReleaseGateError("write-once receipt fields differ")
        result = cls(
            object_kind=raw["object_kind"], object_digest=raw["object_digest"],
            payload_digest=raw["payload_digest"], size_bytes=raw["size_bytes"],
            relative_path=raw["relative_path"], record_digest=raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardReleaseGateError("write-once receipt is not canonical")
        return result


def _stable_read(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.lstat(path)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ObjectBongardReleaseGateError("cannot open durable object") from exc
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or (
            opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns
        ) != identity or not 0 < opened.st_size <= _MAX_OBJECT_BYTES:
            raise ObjectBongardReleaseGateError("durable object is not a stable private file")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(1024 * 1024, _MAX_OBJECT_BYTES - total + 1)):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_OBJECT_BYTES:
                raise ObjectBongardReleaseGateError("durable object exceeds byte bound")
        after = os.fstat(descriptor)
        if total != opened.st_size or (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ) != identity:
            raise ObjectBongardReleaseGateError("durable object changed while reading")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise ObjectBongardReleaseGateError("short durable write")
            view = view[count:]
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
    if _stable_read(path) != payload:
        raise ObjectBongardReleaseGateError("content-addressed collision or tamper")


@dataclass(frozen=True, slots=True)
class ObjectBongardReleaseStore:
    root: Path

    def __post_init__(self) -> None:
        if type(self.root) is not type(Path()) or not self.root.is_absolute():
            raise ObjectBongardReleaseGateError("release store root must be absolute")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            actual = self.root.resolve(strict=True)
            metadata = os.lstat(self.root)
        except OSError as exc:
            raise ObjectBongardReleaseGateError("cannot prepare release store") from exc
        if actual != self.root or not stat.S_ISDIR(metadata.st_mode):
            raise ObjectBongardReleaseGateError("release store root must be a real directory")

    def persist(
        self,
        *,
        object_kind: str,
        object_digest: str,
        data: Mapping[str, Any],
    ) -> ObjectBongardWriteOnceReceipt:
        if type(object_kind) is not str or _NAME.fullmatch(object_kind) is None:
            raise ObjectBongardReleaseGateError("stored object kind is invalid")
        digest = _require_address(object_digest, "stored object digest")
        if type(data) is not dict:
            raise ObjectBongardReleaseGateError("stored object must be a mapping")
        payload = canonical_json(dict(data)) + b"\n"
        if not 0 < len(payload) <= _MAX_OBJECT_BYTES:
            raise ObjectBongardReleaseGateError("stored object exceeds byte bound")
        directory = self.root / "objects" / object_kind
        directory.mkdir(parents=True, exist_ok=True)
        if directory.resolve(strict=True) != directory or not stat.S_ISDIR(os.lstat(directory).st_mode):
            raise ObjectBongardReleaseGateError("stored object directory is unsafe")
        path = directory / f"{digest[7:]}.json"
        _write_once(path, payload)
        reloaded = _stable_read(path)
        try:
            decoded = json.loads(reloaded.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ObjectBongardReleaseGateError("durable object is not JSON") from exc
        if not isinstance(decoded, dict) or canonical_json(decoded) + b"\n" != payload:
            raise ObjectBongardReleaseGateError("durable object reload is not canonical")
        values: dict[str, object] = {
            "object_kind": object_kind,
            "object_digest": digest,
            "payload_digest": _bytes_address(payload),
            "size_bytes": len(payload),
            "relative_path": path.relative_to(self.root).as_posix(),
        }
        provisional = object.__new__(ObjectBongardWriteOnceReceipt)
        for key, value in values.items():
            object.__setattr__(provisional, key, value)
        return ObjectBongardWriteOnceReceipt(
            **values,  # type: ignore[arg-type]
            record_digest=_address(_receipt_content(provisional)),
        )

    def verify(
        self,
        receipt: ObjectBongardWriteOnceReceipt,
        *,
        expected_data: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if type(receipt) is not ObjectBongardWriteOnceReceipt:
            raise TypeError("receipt must be ObjectBongardWriteOnceReceipt")
        frozen_receipt = ObjectBongardWriteOnceReceipt.from_data(receipt.to_data())
        path = self.root / frozen_receipt.relative_path
        expected_parent = self.root / "objects" / frozen_receipt.object_kind
        if path.parent.resolve(strict=True) != expected_parent:
            raise ObjectBongardReleaseGateError("receipt escapes its release store")
        payload = _stable_read(path)
        if (
            len(payload) != frozen_receipt.size_bytes
            or _bytes_address(payload) != frozen_receipt.payload_digest
        ):
            raise ObjectBongardReleaseGateError("durable receipt payload differs")
        try:
            decoded = json.loads(payload.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ObjectBongardReleaseGateError("durable receipt payload is not JSON") from exc
        if not isinstance(decoded, dict) or canonical_json(decoded) + b"\n" != payload:
            raise ObjectBongardReleaseGateError(
                "durable receipt payload is not canonical"
            )
        if expected_data is not None:
            if type(expected_data) is not dict:
                raise TypeError("expected_data must be a mapping or None")
            expected = canonical_json(dict(expected_data)) + b"\n"
            if payload != expected:
                raise ObjectBongardReleaseGateError(
                    "durable receipt payload differs from expectation"
                )
        return decoded


def _authorization_content(value: "ObjectBongardReleaseAuthorization") -> dict[str, object]:
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "batch_plan_digest": value.batch_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "exposure_predecessor_digest": value.exposure_predecessor_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "exposure_event_digest": value.exposure_event_digest,
        "release_descriptor_digest": value.release_descriptor_digest,
        "archive_record_digest": value.archive_record_digest,
        "selected_task_ids": list(value.selected_task_ids),
        "authorized_support_panel_ids": list(value.authorized_support_panel_ids),
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "plan_store_receipt_digest": value.plan_store_receipt_digest,
        "precommit_store_receipt_digest": value.precommit_store_receipt_digest,
        "exposure_store_receipt_digest": value.exposure_store_receipt_digest,
        "exposure_successor_persisted_and_reloaded_before_authorization": True,
        "query_requires_durable_task_freeze_and_commit": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardReleaseAuthorization:
    batch_plan_digest: str
    execution_precommit_digest: str
    exposure_predecessor_digest: str
    exposure_successor_digest: str
    exposure_event_digest: str
    release_descriptor_digest: str
    archive_record_digest: str
    selected_task_ids: tuple[str, ...]
    authorized_support_panel_ids: tuple[str, ...]
    sealed_query_panel_ids: tuple[str, ...]
    plan_store_receipt_digest: str
    precommit_store_receipt_digest: str
    exposure_store_receipt_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        for name in (
            "batch_plan_digest", "execution_precommit_digest", "exposure_predecessor_digest",
            "exposure_successor_digest", "exposure_event_digest", "release_descriptor_digest",
            "archive_record_digest", "plan_store_receipt_digest",
            "precommit_store_receipt_digest", "exposure_store_receipt_digest", "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            self.selected_task_ids != tuple(sorted(set(self.selected_task_ids)))
            or not self.selected_task_ids
            or self.authorized_support_panel_ids
            != tuple(sorted(set(self.authorized_support_panel_ids)))
            or not self.authorized_support_panel_ids
            or self.sealed_query_panel_ids
            != tuple(sorted(set(self.sealed_query_panel_ids)))
            or not self.sealed_query_panel_ids
            or set(self.authorized_support_panel_ids) & set(self.sealed_query_panel_ids)
            or self.record_digest != _address(_authorization_content(self))
        ):
            raise ObjectBongardReleaseGateError("release authorization digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_authorization_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, raw: Mapping[str, Any]) -> "ObjectBongardReleaseAuthorization":
        expected = {
            "schema", "batch_plan_digest", "execution_precommit_digest",
            "exposure_predecessor_digest", "exposure_successor_digest", "exposure_event_digest",
            "release_descriptor_digest", "archive_record_digest", "selected_task_ids",
            "authorized_support_panel_ids", "sealed_query_panel_ids", "plan_store_receipt_digest",
            "precommit_store_receipt_digest", "exposure_store_receipt_digest",
            "exposure_successor_persisted_and_reloaded_before_authorization",
            "query_requires_durable_task_freeze_and_commit", *_authority_data(), "record_digest",
        }
        if type(raw) is not dict or set(raw) != expected:
            raise ObjectBongardReleaseGateError("release authorization fields differ")
        if (
            raw["schema"] != AUTHORIZATION_SCHEMA
            or raw["exposure_successor_persisted_and_reloaded_before_authorization"] is not True
            or raw["query_requires_durable_task_freeze_and_commit"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardReleaseGateError("release authorization policy differs")
        result = cls(
            batch_plan_digest=raw["batch_plan_digest"],
            execution_precommit_digest=raw["execution_precommit_digest"],
            exposure_predecessor_digest=raw["exposure_predecessor_digest"],
            exposure_successor_digest=raw["exposure_successor_digest"],
            exposure_event_digest=raw["exposure_event_digest"],
            release_descriptor_digest=raw["release_descriptor_digest"],
            archive_record_digest=raw["archive_record_digest"],
            selected_task_ids=tuple(raw["selected_task_ids"]),
            authorized_support_panel_ids=tuple(raw["authorized_support_panel_ids"]),
            sealed_query_panel_ids=tuple(raw["sealed_query_panel_ids"]),
            plan_store_receipt_digest=raw["plan_store_receipt_digest"],
            precommit_store_receipt_digest=raw["precommit_store_receipt_digest"],
            exposure_store_receipt_digest=raw["exposure_store_receipt_digest"],
            record_digest=raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardReleaseGateError("release authorization is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class PreparedObjectBongardRelease:
    store: ObjectBongardReleaseStore = field(compare=False, repr=False)
    plan: ObjectBongardReleasePlan
    precommit: ObjectBongardExecutionPrecommit
    predecessor: ExposureLedger
    successor: ExposureLedger
    authorization: ObjectBongardReleaseAuthorization
    plan_receipt: ObjectBongardWriteOnceReceipt
    precommit_receipt: ObjectBongardWriteOnceReceipt
    exposure_receipt: ObjectBongardWriteOnceReceipt
    authorization_receipt: ObjectBongardWriteOnceReceipt


def prepare_object_bongard_release(
    *,
    store: ObjectBongardReleaseStore,
    plan: ObjectBongardReleasePlan,
    precommit: ObjectBongardExecutionPrecommit,
    predecessor: ExposureLedger,
) -> PreparedObjectBongardRelease:
    if (
        precommit.batch_plan_digest != plan.record_digest
        or precommit.exposure_predecessor_digest != predecessor.digest
        or precommit.batch_algorithm_digest != _plan_algorithm_digest(plan)
        or precommit.batch_source_digest != "sha256:" + _plan_source_digest(plan)
        or precommit.release_gate_source_digest
        != "sha256:" + object_bongard_release_gate_source_digest()
    ):
        raise ObjectBongardReleaseGateError("precommit parents differ")
    if (
        precommit.selected_task_ids != tuple(task.task_id for task in plan.tasks)
        or _all_panels(plan) != (precommit.authorized_support_panel_ids, precommit.sealed_query_panel_ids)
        or set(precommit.selected_task_ids) & set(predecessor.exposed_task_ids)
    ):
        raise ObjectBongardReleaseGateError("precommit selection is no longer exact-unused")
    plan_receipt = store.persist(object_kind="batch-plan", object_digest=plan.record_digest, data=plan.to_data())
    store.verify(plan_receipt, expected_data=plan.to_data())
    precommit_receipt = store.persist(object_kind="execution-precommit", object_digest=precommit.record_digest, data=precommit.to_data())
    store.verify(precommit_receipt, expected_data=precommit.to_data())
    successor = predecessor.record(
        phase=EXPOSURE_PHASE,
        actor=precommit.exposure_actor,
        purpose=precommit.exposure_purpose,
        task_ids=precommit.selected_task_ids,
        source=precommit.exposure_source,
        observed_at=precommit.exposure_observed_at,
        known_task_ids=precommit.selected_task_ids,
        require_unseen=True,
    )
    if len(successor.events) != len(predecessor.events) + 1 or successor.events[-1].task_ids != precommit.selected_task_ids or successor.events[-1].panel_ids:
        raise ObjectBongardReleaseGateError("exposure successor is not the exact one-event transition")
    exposure_receipt = store.persist(object_kind="exposure-successor", object_digest=successor.digest, data=successor.to_dict())
    decoded_successor = ExposureLedger.from_dict(store.verify(exposure_receipt, expected_data=successor.to_dict()))
    if decoded_successor != successor:
        raise ObjectBongardReleaseGateError("exposure successor durable replay differs")
    values: dict[str, object] = {
        "batch_plan_digest": plan.record_digest,
        "execution_precommit_digest": precommit.record_digest,
        "exposure_predecessor_digest": predecessor.digest,
        "exposure_successor_digest": successor.digest,
        "exposure_event_digest": successor.events[-1].digest,
        "release_descriptor_digest": precommit.release_descriptor_digest,
        "archive_record_digest": precommit.archive_record_digest,
        "selected_task_ids": precommit.selected_task_ids,
        "authorized_support_panel_ids": precommit.authorized_support_panel_ids,
        "sealed_query_panel_ids": precommit.sealed_query_panel_ids,
        "plan_store_receipt_digest": plan_receipt.record_digest,
        "precommit_store_receipt_digest": precommit_receipt.record_digest,
        "exposure_store_receipt_digest": exposure_receipt.record_digest,
    }
    provisional = object.__new__(ObjectBongardReleaseAuthorization)
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    authorization = ObjectBongardReleaseAuthorization(
        **values,  # type: ignore[arg-type]
        record_digest=_address(_authorization_content(provisional)),
    )
    authorization_receipt = store.persist(object_kind="release-authorization", object_digest=authorization.record_digest, data=authorization.to_data())
    reloaded = ObjectBongardReleaseAuthorization.from_data(store.verify(authorization_receipt, expected_data=authorization.to_data()))
    if reloaded != authorization:
        raise ObjectBongardReleaseGateError("release authorization durable replay differs")
    return PreparedObjectBongardRelease(
        store, plan, precommit, predecessor, successor, authorization,
        plan_receipt, precommit_receipt, exposure_receipt, authorization_receipt,
    )


def verify_prepared_object_bongard_release(
    prepared: PreparedObjectBongardRelease,
) -> PreparedObjectBongardRelease:
    """Cold-reconstruct and return the exact authority-bearing release state."""
    if type(prepared) is not PreparedObjectBongardRelease:
        raise TypeError("prepared must be exact PreparedObjectBongardRelease")
    if type(prepared.store) is not ObjectBongardReleaseStore:
        raise TypeError("prepared store must be exact ObjectBongardReleaseStore")
    ObjectBongardReleaseStore.__post_init__(prepared.store)
    if type(prepared.plan) not in (
        ObjectBongardBatchPlan,
        ObjectBongardDrillBatchPlan,
    ):
        raise TypeError("prepared plan must be an exact admitted release plan")
    if any(type(task) is not ObjectBongardTaskPlan for task in prepared.plan.tasks):
        raise TypeError("prepared task plans must be exact ObjectBongardTaskPlan")
    if type(prepared.precommit) is not ObjectBongardExecutionPrecommit:
        raise TypeError("prepared precommit must be exact ObjectBongardExecutionPrecommit")
    if (
        type(prepared.predecessor) is not ExposureLedger
        or type(prepared.successor) is not ExposureLedger
    ):
        raise TypeError("prepared exposure ledgers must be exact ExposureLedger")
    if any(
        type(event) is not ExposureEvent
        for ledger in (prepared.predecessor, prepared.successor)
        for event in ledger.events
    ):
        raise TypeError("prepared exposure events must be exact ExposureEvent")
    if type(prepared.authorization) is not ObjectBongardReleaseAuthorization:
        raise TypeError(
            "prepared authorization must be exact ObjectBongardReleaseAuthorization"
        )
    receipts = (
        prepared.plan_receipt,
        prepared.precommit_receipt,
        prepared.exposure_receipt,
        prepared.authorization_receipt,
    )
    if any(type(receipt) is not ObjectBongardWriteOnceReceipt for receipt in receipts):
        raise TypeError("prepared receipts must be exact ObjectBongardWriteOnceReceipt")

    store = prepared.store
    plan_receipt, precommit_receipt, exposure_receipt, authorization_receipt = (
        receipts
    )
    plan_data = dict(store.verify(plan_receipt))
    precommit_data = dict(store.verify(precommit_receipt))
    successor_data = dict(store.verify(exposure_receipt))
    authorization_data = dict(store.verify(authorization_receipt))
    cold_plan = type(prepared.plan).from_data(plan_data)
    cold_precommit = ObjectBongardExecutionPrecommit.from_data(precommit_data)
    cold_successor = ExposureLedger.from_dict(successor_data)
    if not cold_successor.events:
        raise ObjectBongardReleaseGateError(
            "prepared exposure successor lacks its authorization event"
        )
    cold_predecessor = ExposureLedger(
        corpus_digest=cold_successor.corpus_digest,
        events=cold_successor.events[:-1],
    )
    cold_authorization = ObjectBongardReleaseAuthorization.from_data(
        authorization_data
    )
    cold_receipts = tuple(
        ObjectBongardWriteOnceReceipt.from_data(receipt.to_data())
        for receipt in receipts
    )
    if (
        cold_plan != prepared.plan
        or cold_precommit != prepared.precommit
        or cold_predecessor.digest != prepared.predecessor.digest
        or cold_successor != prepared.successor
        or cold_authorization != prepared.authorization
        or cold_receipts != receipts
    ):
        raise ObjectBongardReleaseGateError(
            "prepared release child canonical replay differs"
        )

    plan_receipt, precommit_receipt, exposure_receipt, authorization_receipt = cold_receipts
    if (
        plan_receipt.object_kind != "batch-plan"
        or plan_receipt.object_digest != cold_plan.record_digest
        or precommit_receipt.object_kind != "execution-precommit"
        or precommit_receipt.object_digest != cold_precommit.record_digest
        or exposure_receipt.object_kind != "exposure-successor"
        or exposure_receipt.object_digest != cold_successor.digest
        or authorization_receipt.object_kind != "release-authorization"
        or authorization_receipt.object_digest != cold_authorization.record_digest
    ):
        raise ObjectBongardReleaseGateError("prepared release receipt binding differs")

    if (
        cold_precommit.batch_plan_digest != cold_plan.record_digest
        or cold_precommit.batch_algorithm_digest
        != _plan_algorithm_digest(cold_plan)
        or cold_precommit.batch_source_digest
        != "sha256:" + _plan_source_digest(cold_plan)
        or cold_precommit.release_gate_source_digest
        != "sha256:" + object_bongard_release_gate_source_digest()
        or cold_precommit.release_descriptor_digest
        != cold_plan.release_descriptor_digest
        or cold_precommit.exposure_predecessor_digest != cold_predecessor.digest
        or cold_plan.exposure_predecessor_digest != cold_predecessor.digest
        or cold_precommit.corpus_digest != cold_predecessor.corpus_digest
        or cold_precommit.task_inventory_digest != cold_plan.task_inventory_digest
        or cold_precommit.train_task_ids_digest != cold_plan.train_task_ids_digest
        or cold_precommit.exact_used_task_ids_digest
        != cold_plan.exact_used_task_ids_digest
        or cold_precommit.selected_task_ids
        != tuple(task.task_id for task in cold_plan.tasks)
        or _all_panels(cold_plan)
        != (
            cold_precommit.authorized_support_panel_ids,
            cold_precommit.sealed_query_panel_ids,
        )
        or len(cold_successor.events) != len(cold_predecessor.events) + 1
        or cold_successor.corpus_digest != cold_predecessor.corpus_digest
        or cold_successor.events[:-1] != cold_predecessor.events
        or cold_successor.events[-1].task_ids != cold_precommit.selected_task_ids
        or cold_successor.events[-1].panel_ids
        or cold_authorization.batch_plan_digest != cold_plan.record_digest
        or cold_authorization.exposure_predecessor_digest != cold_predecessor.digest
        or cold_authorization.exposure_successor_digest != cold_successor.digest
        or cold_authorization.exposure_event_digest
        != cold_successor.events[-1].digest
        or cold_authorization.execution_precommit_digest != cold_precommit.record_digest
        or cold_authorization.release_descriptor_digest
        != cold_precommit.release_descriptor_digest
        or cold_authorization.archive_record_digest
        != cold_precommit.archive_record_digest
        or cold_authorization.selected_task_ids != cold_precommit.selected_task_ids
        or cold_authorization.authorized_support_panel_ids
        != cold_precommit.authorized_support_panel_ids
        or cold_authorization.sealed_query_panel_ids
        != cold_precommit.sealed_query_panel_ids
        or cold_authorization.plan_store_receipt_digest
        != plan_receipt.record_digest
        or cold_authorization.precommit_store_receipt_digest
        != precommit_receipt.record_digest
        or cold_authorization.exposure_store_receipt_digest
        != exposure_receipt.record_digest
    ):
        raise ObjectBongardReleaseGateError("prepared release cold replay differs")
    return PreparedObjectBongardRelease(
        store=store,
        plan=cold_plan,
        precommit=cold_precommit,
        predecessor=cold_predecessor,
        successor=cold_successor,
        authorization=cold_authorization,
        plan_receipt=plan_receipt,
        precommit_receipt=precommit_receipt,
        exposure_receipt=exposure_receipt,
        authorization_receipt=authorization_receipt,
    )


def _task_for_panel(plan: ObjectBongardReleasePlan, panel_id: str) -> ObjectBongardTaskPlan:
    matches = tuple(
        task for task in plan.tasks
        if panel_id in (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids,
                        task.side_0_query_panel_id, task.side_1_query_panel_id)
    )
    if len(matches) != 1:
        raise ObjectBongardReleaseGateError("panel is outside the frozen batch plan")
    return matches[0]


def _verify_release_archive(
    archive: OfficialPanelArchive,
    prepared: PreparedObjectBongardRelease,
) -> OfficialPanelArchive:
    """Revalidate the exact live archive binding before any panel read."""

    if type(archive) is not OfficialPanelArchive:
        raise TypeError("archive must be exact OfficialPanelArchive")
    OfficialPanelArchive.__post_init__(archive)
    if (
        archive.record_digest != prepared.authorization.archive_record_digest
        or archive.record_digest != prepared.precommit.archive_record_digest
        or archive.release_descriptor_digest
        != prepared.precommit.release_descriptor_digest
        or archive.archive_digest != prepared.precommit.archive_digest
        or archive.central_directory_digest
        != prepared.precommit.archive_central_directory_digest
    ):
        raise ObjectBongardReleaseGateError(
            "archive differs from the cold prepared release"
        )
    return archive


def release_object_bongard_support_panel(
    *, prepared: PreparedObjectBongardRelease, archive: OfficialPanelArchive, panel_id: str,
) -> tuple[ReleasedOfficialPanel, ObjectBongardWriteOnceReceipt]:
    if type(prepared) is not PreparedObjectBongardRelease:
        raise TypeError("prepared must be exact PreparedObjectBongardRelease")
    if type(prepared.store) is not ObjectBongardReleaseStore:
        raise TypeError("prepared store must be exact ObjectBongardReleaseStore")
    if type(archive) is not OfficialPanelArchive:
        raise TypeError("archive must be exact OfficialPanelArchive")
    prepared = verify_prepared_object_bongard_release(prepared)
    archive = _verify_release_archive(archive, prepared)
    if panel_id not in prepared.authorization.authorized_support_panel_ids:
        raise ObjectBongardReleaseGateError("support panel release is not authorized")
    released = ReleasedOfficialPanel.release(
        archive, panel_id,
        execution_precommit_digest=prepared.precommit.record_digest,
        exposure_successor_digest=prepared.successor.digest,
        expected_execution_precommit_digest=prepared.authorization.execution_precommit_digest,
        expected_exposure_successor_digest=prepared.authorization.exposure_successor_digest,
    )
    receipt = prepared.store.persist(object_kind="released-support-panel", object_digest=released.record_digest, data=released.to_data())
    if ReleasedOfficialPanel.from_data(prepared.store.verify(receipt, expected_data=released.to_data())) != released:
        raise ObjectBongardReleaseGateError("support panel durable replay differs")
    return released, receipt


@runtime_checkable
class ObjectBongardTaskFreezeProtocol(Protocol):
    task_id: str
    task_plan_digest: str
    execution_precommit_digest: str
    version_space_digest: str
    support_version_space_digest: str
    rank_response_digest: str
    selected_predicate_digest: str
    record_digest: str

    def to_data(self) -> Mapping[str, Any]: ...


@runtime_checkable
class ObjectBongardTaskCommitProtocol(ObjectBongardTaskFreezeProtocol, Protocol):
    task_freeze_digest: str
    exact_freeze_payload_digest: str
    task_freeze_store_receipt_digest: str


def _canonical_protocol_data(value: ObjectBongardTaskFreezeProtocol, label: str) -> dict[str, Any]:
    try:
        data = dict(value.to_data())
    except Exception as exc:
        raise ObjectBongardReleaseGateError(f"{label} does not expose canonical data") from exc
    if data.get("record_digest") != value.record_digest:
        raise ObjectBongardReleaseGateError(f"{label} record digest field differs")
    content = {key: item for key, item in data.items() if key != "record_digest"}
    if value.record_digest != _address(content) or json.loads(canonical_json(data)) != data:
        raise ObjectBongardReleaseGateError(f"{label} is not canonical")
    return data


def _production_task_decision_pair(
    freeze: ObjectBongardTaskFreezeProtocol,
    commit: ObjectBongardTaskCommitProtocol,
) -> tuple[type[Any], type[Any], str]:
    """Resolve an exact, production-owned freeze/commit pair lazily.

    The protocol types above are useful to task runners, but structural typing
    is not release authority.  Keep the imports at this release boundary so
    the production task modules can continue to import the protocol without a
    module-import cycle.
    """

    freeze_type = type(freeze)
    commit_type = type(commit)
    if freeze_type.__module__ != commit_type.__module__:
        raise ObjectBongardReleaseGateError(
            "task decision records are not an admitted exact production pair"
        )
    key = (freeze_type.__module__, freeze_type.__name__, commit_type.__name__)
    family = _PRODUCTION_TASK_DECISION_TYPES.get(key)
    if family is None:
        raise ObjectBongardReleaseGateError(
            "task decision records are not an admitted exact production pair"
        )
    try:
        owner = importlib.import_module(key[0])
        admitted_freeze_type = getattr(owner, key[1])
        admitted_commit_type = getattr(owner, key[2])
    except (AttributeError, ImportError) as exc:
        raise ObjectBongardReleaseGateError(
            "admitted production task decision implementation is unavailable"
        ) from exc
    if (
        freeze_type is not admitted_freeze_type
        or commit_type is not admitted_commit_type
    ):
        raise ObjectBongardReleaseGateError(
            "task decision records are not an admitted exact production pair"
        )
    return admitted_freeze_type, admitted_commit_type, family


def _canonical_production_task_decision_pair(
    freeze: ObjectBongardTaskFreezeProtocol,
    commit: ObjectBongardTaskCommitProtocol,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Deeply reconstruct an allowlisted pair from its canonical wire data."""

    freeze_type, commit_type, family = _production_task_decision_pair(
        freeze, commit
    )
    freeze_data = _canonical_protocol_data(freeze, "task freeze")
    commit_data = _canonical_protocol_data(commit, "task commit")
    try:
        cold_freeze = freeze_type.from_data(freeze_data)
        cold_commit = commit_type.from_data(commit_data)
    except Exception as exc:
        raise ObjectBongardReleaseGateError(
            "task decision records fail canonical production reconstruction"
        ) from exc
    if (
        type(cold_freeze) is not freeze_type
        or type(cold_commit) is not commit_type
        or cold_freeze != freeze
        or cold_commit != commit
        or cold_freeze.to_data() != freeze_data
        or cold_commit.to_data() != commit_data
    ):
        raise ObjectBongardReleaseGateError(
            "task decision canonical production replay differs"
        )
    return freeze_data, commit_data, family


def _cold_verify_production_task_decision_binding(
    *,
    freeze: ObjectBongardTaskFreezeProtocol,
    commit: ObjectBongardTaskCommitProtocol,
    freeze_data: Mapping[str, Any],
    freeze_receipt: ObjectBongardWriteOnceReceipt,
    family: str,
) -> None:
    """Recreate the production commit from the exact durable freeze bytes."""

    exact_payload = canonical_json(dict(freeze_data)) + b"\n"
    try:
        if family == "object-scene-anchor":
            commit.assert_matches(freeze, exact_payload)  # type: ignore[attr-defined]
        elif family == "panel-feature":
            expected = type(commit).seal(freeze, freeze_receipt)
            if expected != commit:
                raise ObjectBongardReleaseGateError(
                    "panel-feature task decision cold replay differs"
                )
        elif family == "object-bongard-rubric":
            commit.assert_matches(freeze, exact_payload)  # type: ignore[attr-defined]
        elif family == "panel-program":
            # The successor follows the same release-boundary convention: its
            # commit must expose an exact cold replay against freeze bytes and
            # the durable freeze receipt.
            commit.assert_matches(  # type: ignore[attr-defined]
                freeze,
                exact_payload,
                freeze_receipt,
            )
        else:  # pragma: no cover - closed by _production_task_decision_pair
            raise ObjectBongardReleaseGateError(
                "task decision production family is unknown"
            )
    except ObjectBongardReleaseGateError:
        raise
    except Exception as exc:
        raise ObjectBongardReleaseGateError(
            "task decision commit differs from production cold replay"
        ) from exc


def _validate_freeze_bindings(
    freeze: ObjectBongardTaskFreezeProtocol,
    *, task: ObjectBongardTaskPlan, prepared: PreparedObjectBongardRelease,
) -> dict[str, Any]:
    data = _canonical_protocol_data(freeze, "task freeze")
    _require_address(freeze.task_plan_digest, "task plan digest")
    _require_address(freeze.execution_precommit_digest, "execution precommit digest")
    for name in ("version_space_digest", "support_version_space_digest", "rank_response_digest", "selected_predicate_digest"):
        _require_raw_digest(getattr(freeze, name), name)
    if (
        freeze.task_id != task.task_id
        or freeze.task_plan_digest != task.record_digest
        or freeze.execution_precommit_digest != prepared.precommit.record_digest
        or freeze.support_version_space_digest != freeze.version_space_digest
    ):
        raise ObjectBongardReleaseGateError("task freeze bindings differ")
    return data


def persist_object_bongard_task_freeze(
    *, store: ObjectBongardReleaseStore, freeze: ObjectBongardTaskFreezeProtocol,
) -> ObjectBongardWriteOnceReceipt:
    data = _canonical_protocol_data(freeze, "task freeze")
    return store.persist(object_kind="task-freeze", object_digest=freeze.record_digest, data=data)


def persist_object_bongard_task_commit(
    *, store: ObjectBongardReleaseStore, commit: ObjectBongardTaskCommitProtocol,
) -> ObjectBongardWriteOnceReceipt:
    data = _canonical_protocol_data(commit, "task commit")
    return store.persist(object_kind="task-decision-commit", object_digest=commit.record_digest, data=data)


def release_object_bongard_query_panel(
    *,
    prepared: PreparedObjectBongardRelease,
    archive: OfficialPanelArchive,
    panel_id: str,
    task_freeze: ObjectBongardTaskFreezeProtocol,
    task_commit: ObjectBongardTaskCommitProtocol,
    task_freeze_receipt: ObjectBongardWriteOnceReceipt,
    task_commit_receipt: ObjectBongardWriteOnceReceipt,
) -> tuple[ReleasedOfficialPanel, ObjectBongardWriteOnceReceipt]:
    if type(prepared) is not PreparedObjectBongardRelease:
        raise TypeError("prepared must be exact PreparedObjectBongardRelease")
    if type(prepared.store) is not ObjectBongardReleaseStore:
        raise TypeError("prepared store must be exact ObjectBongardReleaseStore")
    if type(archive) is not OfficialPanelArchive:
        raise TypeError("archive must be exact OfficialPanelArchive")
    prepared = verify_prepared_object_bongard_release(prepared)
    archive = _verify_release_archive(archive, prepared)
    _freeze_data, _commit_data, production_family = (
        _canonical_production_task_decision_pair(task_freeze, task_commit)
    )
    freeze_raw = dict(prepared.store.verify(task_freeze_receipt))
    commit_raw = dict(prepared.store.verify(task_commit_receipt))
    cold_freeze = type(task_freeze).from_data(freeze_raw)
    cold_commit = type(task_commit).from_data(commit_raw)
    freeze_data, commit_data, cold_family = (
        _canonical_production_task_decision_pair(cold_freeze, cold_commit)
    )
    if cold_family != production_family:
        raise ObjectBongardReleaseGateError("task decision family changed on replay")
    task_freeze = cold_freeze
    task_commit = cold_commit
    task = _task_for_panel(prepared.plan, panel_id)
    if (
        panel_id not in prepared.authorization.sealed_query_panel_ids
        or panel_id not in (task.side_0_query_panel_id, task.side_1_query_panel_id)
    ):
        raise ObjectBongardReleaseGateError("query panel is not the task's sealed query")
    _validate_freeze_bindings(task_freeze, task=task, prepared=prepared)
    if (
        task_freeze_receipt.object_kind != "task-freeze"
        or task_freeze_receipt.object_digest != task_freeze.record_digest
        or task_commit_receipt.object_kind != "task-decision-commit"
        or task_commit_receipt.object_digest != task_commit.record_digest
        or task_commit.task_id != task_freeze.task_id
        or task_commit.task_plan_digest != task_freeze.task_plan_digest
        or task_commit.execution_precommit_digest != task_freeze.execution_precommit_digest
        or task_commit.version_space_digest != task_freeze.version_space_digest
        or task_commit.support_version_space_digest != task_freeze.support_version_space_digest
        or task_commit.rank_response_digest != task_freeze.rank_response_digest
        or task_commit.selected_predicate_digest != task_freeze.selected_predicate_digest
        or task_commit.task_freeze_digest != task_freeze.record_digest
        or task_commit.exact_freeze_payload_digest != task_freeze_receipt.payload_digest
        or task_commit.task_freeze_store_receipt_digest != task_freeze_receipt.record_digest
    ):
        raise ObjectBongardReleaseGateError("task decision commit does not bind the exact durable freeze")
    _cold_verify_production_task_decision_binding(
        freeze=task_freeze,
        commit=task_commit,
        freeze_data=freeze_data,
        freeze_receipt=task_freeze_receipt,
        family=production_family,
    )
    if production_family == "panel-program":
        try:
            from bongard.panel_program_official_task import (
                verify_panel_program_query_release_authority,
            )

            verify_panel_program_query_release_authority(
                prepared=prepared,
                freeze=task_freeze,  # exact type established above
            )
        except Exception as exc:
            raise ObjectBongardReleaseGateError(
                "panel-program support release authority differs"
            ) from exc
    released = ReleasedOfficialPanel.release(
        archive, panel_id,
        execution_precommit_digest=prepared.precommit.record_digest,
        exposure_successor_digest=prepared.successor.digest,
        expected_execution_precommit_digest=prepared.authorization.execution_precommit_digest,
        expected_exposure_successor_digest=prepared.authorization.exposure_successor_digest,
    )
    receipt = prepared.store.persist(object_kind="released-query-panel", object_digest=released.record_digest, data=released.to_data())
    if ReleasedOfficialPanel.from_data(prepared.store.verify(receipt, expected_data=released.to_data())) != released:
        raise ObjectBongardReleaseGateError("query panel durable replay differs")
    return released, receipt


__all__ = (
    "AUTHORIZATION_SCHEMA", "EXPOSURE_PHASE", "PRECOMMIT_SCHEMA",
    "ObjectBongardExecutionPrecommit", "ObjectBongardReleaseAuthorization",
    "ObjectBongardReleaseGateError", "ObjectBongardReleaseStore",
    "ObjectBongardTaskCommitProtocol", "ObjectBongardTaskFreezeProtocol",
    "ObjectBongardWriteOnceReceipt", "PreparedObjectBongardRelease",
    "create_object_bongard_execution_precommit", "object_bongard_release_gate_source_digest",
    "persist_object_bongard_task_commit", "persist_object_bongard_task_freeze",
    "prepare_object_bongard_release", "release_object_bongard_query_panel",
    "release_object_bongard_support_panel", "verify_prepared_object_bongard_release",
)
