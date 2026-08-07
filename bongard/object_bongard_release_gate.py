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
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger
from bongard.object_bongard_batch import (
    ObjectBongardBatchPlan,
    ObjectBongardTaskPlan,
    object_bongard_batch_algorithm_digest,
    object_bongard_batch_source_digest,
    object_bongard_task_inventory_digest,
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


class ObjectBongardReleaseGateError(RuntimeError):
    """A precommit, durable record, exposure transition, or release is invalid."""


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _bytes_address(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise ObjectBongardReleaseGateError(f"{label} must be a sha256: address")
    return value


def _require_raw_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _RAW_DIGEST.fullmatch(value) is None:
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
        any(not isinstance(item, str) or not item for item in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ObjectBongardReleaseGateError(f"{label} must be unique and sorted")
    return result


def _freeze_bindings(values: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, Mapping) or not values:
        raise ObjectBongardReleaseGateError("runtime source bindings must be nonempty")
    result = tuple(sorted(values.items()))
    if any(_CONFIG_KEY.fullmatch(key) is None for key, _ in result):
        raise ObjectBongardReleaseGateError("runtime source binding name is invalid")
    for key, value in result:
        _require_address(value, f"runtime source binding {key}")
    return result


def _freeze_configuration(
    values: Mapping[str, str | int | bool],
) -> tuple[tuple[str, str | int | bool], ...]:
    if not isinstance(values, Mapping):
        raise ObjectBongardReleaseGateError("configuration must be a mapping")
    result = tuple(sorted(values.items()))
    for key, value in result:
        if (
            not isinstance(key, str)
            or _CONFIG_KEY.fullmatch(key) is None
            or _FORBIDDEN_CONFIG_KEY.search(key) is not None
            or isinstance(value, float)
            or value is None
            or not isinstance(value, (str, int, bool))
            or (isinstance(value, str) and (not value or "\x00" in value or len(value.encode()) > 512))
        ):
            raise ObjectBongardReleaseGateError(
                "configuration must contain bounded metadata scalars and no visual/action inputs"
            )
    return result


def _all_panels(plan: ObjectBongardBatchPlan) -> tuple[tuple[str, ...], tuple[str, ...]]:
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
        if any(not isinstance(item, str) or not item for item in (
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
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ObjectBongardReleaseGateError("execution precommit fields differ")
        if (
            raw["schema"] != PRECOMMIT_SCHEMA
            or raw["selection_inputs_include_pixels"] is not False
            or raw["selection_inputs_include_action_programs"] is not False
            or raw["official_test_authorized"] is not False
            or raw["query_identities_sealed_before_support_pixels"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or not all(isinstance(raw[key], list) for key in (
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
    plan: ObjectBongardBatchPlan,
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
    if not isinstance(plan, ObjectBongardBatchPlan):
        raise TypeError("plan must be ObjectBongardBatchPlan")
    if not isinstance(predecessor, ExposureLedger):
        raise TypeError("predecessor must be ExposureLedger")
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
        "batch_source": "sha256:" + object_bongard_batch_source_digest(),
        "release_gate_source": "sha256:" + object_bongard_release_gate_source_digest(),
    }
    for key, value in automatic.items():
        if key in bindings and bindings[key] != value:
            raise ObjectBongardReleaseGateError(f"automatic source binding {key} differs")
        bindings[key] = value
    values: dict[str, object] = {
        "batch_plan_digest": plan.record_digest,
        "batch_algorithm_digest": object_bongard_batch_algorithm_digest(),
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
        if _NAME.fullmatch(self.object_kind) is None:
            raise ObjectBongardReleaseGateError("stored object kind is invalid")
        for name in ("object_digest", "payload_digest", "record_digest"):
            _require_address(getattr(self, name), name)
        expected_path = f"objects/{self.object_kind}/{self.object_digest[7:]}.json"
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or not 0 < self.size_bytes <= _MAX_OBJECT_BYTES
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
            not isinstance(raw, Mapping)
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
        if not isinstance(self.root, Path) or not self.root.is_absolute():
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
        if _NAME.fullmatch(object_kind) is None:
            raise ObjectBongardReleaseGateError("stored object kind is invalid")
        digest = _require_address(object_digest, "stored object digest")
        if not isinstance(data, Mapping):
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
        expected_data: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(receipt, ObjectBongardWriteOnceReceipt):
            raise TypeError("receipt must be ObjectBongardWriteOnceReceipt")
        path = self.root / receipt.relative_path
        if path.parent.resolve(strict=True) != (self.root / "objects" / receipt.object_kind):
            raise ObjectBongardReleaseGateError("receipt escapes its release store")
        payload = _stable_read(path)
        expected = canonical_json(dict(expected_data)) + b"\n"
        if (
            payload != expected
            or len(payload) != receipt.size_bytes
            or _bytes_address(payload) != receipt.payload_digest
        ):
            raise ObjectBongardReleaseGateError("durable receipt payload differs")
        return json.loads(payload)


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
        if not isinstance(raw, Mapping) or set(raw) != expected:
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
    plan: ObjectBongardBatchPlan
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
    plan: ObjectBongardBatchPlan,
    precommit: ObjectBongardExecutionPrecommit,
    predecessor: ExposureLedger,
) -> PreparedObjectBongardRelease:
    if (
        precommit.batch_plan_digest != plan.record_digest
        or precommit.exposure_predecessor_digest != predecessor.digest
        or precommit.batch_algorithm_digest != object_bongard_batch_algorithm_digest()
        or precommit.batch_source_digest != "sha256:" + object_bongard_batch_source_digest()
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


def verify_prepared_object_bongard_release(prepared: PreparedObjectBongardRelease) -> None:
    if not isinstance(prepared, PreparedObjectBongardRelease):
        raise TypeError("prepared must be PreparedObjectBongardRelease")
    store = prepared.store
    store.verify(prepared.plan_receipt, expected_data=prepared.plan.to_data())
    store.verify(prepared.precommit_receipt, expected_data=prepared.precommit.to_data())
    store.verify(prepared.exposure_receipt, expected_data=prepared.successor.to_dict())
    store.verify(prepared.authorization_receipt, expected_data=prepared.authorization.to_data())
    if (
        len(prepared.successor.events) != len(prepared.predecessor.events) + 1
        or prepared.successor.events[:-1] != prepared.predecessor.events
        or prepared.successor.events[-1].task_ids != prepared.precommit.selected_task_ids
        or prepared.successor.events[-1].panel_ids
        or prepared.authorization.exposure_successor_digest != prepared.successor.digest
        or prepared.authorization.execution_precommit_digest != prepared.precommit.record_digest
        or prepared.authorization.plan_store_receipt_digest
        != prepared.plan_receipt.record_digest
        or prepared.authorization.precommit_store_receipt_digest
        != prepared.precommit_receipt.record_digest
        or prepared.authorization.exposure_store_receipt_digest
        != prepared.exposure_receipt.record_digest
    ):
        raise ObjectBongardReleaseGateError("prepared release cold replay differs")


def _task_for_panel(plan: ObjectBongardBatchPlan, panel_id: str) -> ObjectBongardTaskPlan:
    matches = tuple(
        task for task in plan.tasks
        if panel_id in (*task.side_0_support_panel_ids, *task.side_1_support_panel_ids,
                        task.side_0_query_panel_id, task.side_1_query_panel_id)
    )
    if len(matches) != 1:
        raise ObjectBongardReleaseGateError("panel is outside the frozen batch plan")
    return matches[0]


def release_object_bongard_support_panel(
    *, prepared: PreparedObjectBongardRelease, archive: OfficialPanelArchive, panel_id: str,
) -> tuple[ReleasedOfficialPanel, ObjectBongardWriteOnceReceipt]:
    verify_prepared_object_bongard_release(prepared)
    if archive.record_digest != prepared.authorization.archive_record_digest or panel_id not in prepared.authorization.authorized_support_panel_ids:
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
    verify_prepared_object_bongard_release(prepared)
    task = _task_for_panel(prepared.plan, panel_id)
    if (
        archive.record_digest != prepared.authorization.archive_record_digest
        or panel_id not in prepared.authorization.sealed_query_panel_ids
        or panel_id not in (task.side_0_query_panel_id, task.side_1_query_panel_id)
    ):
        raise ObjectBongardReleaseGateError("query panel is not the task's sealed query")
    freeze_data = _validate_freeze_bindings(task_freeze, task=task, prepared=prepared)
    commit_data = _canonical_protocol_data(task_commit, "task commit")
    prepared.store.verify(task_freeze_receipt, expected_data=freeze_data)
    prepared.store.verify(task_commit_receipt, expected_data=commit_data)
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
