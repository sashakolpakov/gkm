"""Crash-safe publication of the skeleton calibration custody tombstone.

Only committed metadata records are opened.  The official archive, PNGs,
action programs, catalog sources, model, features, and predictions are never
accepted as inputs and are never read here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Final, Mapping

from bongard import canonical as canonical_module
from bongard import exposure as exposure_module
from bongard import runtime_source_snapshot as runtime_source_snapshot_module
from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureLedger
from bongard.runtime_source_snapshot import capture_loaded_source
from bongard import panel_action_count_skeleton_graph_calibration_runner as core
from bongard import panel_action_count_skeleton_graph_custody_incident as incident_api


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

RECEIPT_SCHEMA: Final = (
    "gkm.bongard-skeleton-graph-custody-incident-persistence-receipt.v1"
)
RECEIPT_FILENAME: Final = (
    "panel_action_count_skeleton_graph_custody_incident_persistence_v1.json"
)
INCIDENT_COMMIT: Final = "8c962aaf52a229206aef5497fed9ad777ead2937"
PINNED_INCIDENT_SOURCE_SHA256: Final = (
    "d66eac9ac00b306008cd35e307e42ec0aa8eb69c1c48ca8f42e6cfa685d75225"
)
PINNED_INCIDENT_FILE_SHA256: Final = (
    "sha256:0f076190b70cf320f999a959640c20aa2bd8fda89131a36b175a0c80d62dcd7b"
)
PINNED_INCIDENT_RECORD_DIGEST: Final = (
    "sha256:c647b0929a524a3fec64f74afbda1d1f469e6cf4ba1b8d6da1de788f0af2801f"
)
PINNED_CORE_SOURCE_SHA256: Final = (
    "b7f2ead679de658ae6d2389d8f186167f56a23325657fad9764c31b92e4e6265"
)
PINNED_CORE_COMMIT: Final = "a35cf269e418241da8db4fef6fb72ede20e5780f"
PINNED_CANONICAL_SOURCE_SHA256: Final = (
    "30bfaa4cb9bea1bd5176d84942d9e42b32b16512c9fcdc815b606012c6e89c1e"
)
PINNED_EXPOSURE_SOURCE_SHA256: Final = (
    "b46f424e843113eb113d8698831eba2a54ab228d8d1630ed062378506e490b59"
)
PINNED_RUNTIME_SOURCE_SNAPSHOT_SHA256: Final = (
    "67d37b28497e589f6766367a73a71bb3f6fe70510436123d5dac7730fc681ced"
)
PINNED_CALIBRATION_PREREG_SOURCE_SHA256: Final = (
    "9413f2f00a32fa38adcbab0d745a398881a20437f930f9d202ffff74e35b67a6"
)
PINNED_CORE_INTENT_SCHEMA: Final = (
    "gkm.bongard-skeleton-graph-calibration-campaign-attempt-intent.v2"
)
AUTHORITY_RELATIVE_DIRECTORY: Final = Path(
    "downloads/ShapeBongard_V2_full/"
    "panel_soft_exact_unused_train_20260809_ranked_v1/"
    "research-exposure-successors"
)
INCIDENT_RELATIVE_PATH: Final = incident_api.INCIDENT_RECORD_PATH
PREDECESSOR_FILENAME: Final = (
    "6995ea9cfda2f384cb0ba1b1cdc3611c965227c60fdb281d1e2e56fffa357b56."
    "exposure.json"
)
DEFAULT_REPOSITORY_ROOT: Final = Path(__file__).absolute().parents[1]

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OBSERVED_AT = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\Z"
)
_EXPOSURE_FILENAME = re.compile(r"[0-9a-f]{64}\.exposure\.json\Z")

_stage_hook: Callable[[str], None] = lambda _stage: None
_VERIFIED_ISSUANCE_TOKEN = object()


class SkeletonGraphCustodyIncidentPersistenceError(RuntimeError):
    pass


def source_sha256() -> str:
    current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current != _LOADED_SOURCE_SHA256:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "incident-persistence source changed after import"
        )
    return current


def _module_file_sha256(module: object, *, label: str) -> str:
    path_value = getattr(module, "__file__", None)
    if type(path_value) is not str:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            f"{label} source path differs"
        )
    return hashlib.sha256(Path(path_value).read_bytes()).hexdigest()


def _source_preflight() -> None:
    if (
        source_sha256() != _LOADED_SOURCE_SHA256
        or incident_api.source_sha256() != PINNED_INCIDENT_SOURCE_SHA256
        or core.source_sha256() != PINNED_CORE_SOURCE_SHA256
        or _module_file_sha256(canonical_module, label="canonical")
        != PINNED_CANONICAL_SOURCE_SHA256
        or _module_file_sha256(exposure_module, label="exposure")
        != PINNED_EXPOSURE_SOURCE_SHA256
        or _module_file_sha256(
            runtime_source_snapshot_module, label="runtime source snapshot"
        )
        != PINNED_RUNTIME_SOURCE_SNAPSHOT_SHA256
        or _module_file_sha256(core.prereg, label="calibration preregistration")
        != PINNED_CALIBRATION_PREREG_SOURCE_SHA256
        or canonical_digest is not canonical_module.canonical_digest
        or canonical_json is not canonical_module.canonical_json
        or ExposureLedger is not exposure_module.ExposureLedger
    ):
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "incident-persistence source dependency differs"
        )


def _production_repository_root() -> Path:
    module_paths = (
        getattr(canonical_module, "__file__", None),
        getattr(exposure_module, "__file__", None),
        getattr(runtime_source_snapshot_module, "__file__", None),
        getattr(core, "__file__", None),
        getattr(core.prereg, "__file__", None),
        getattr(incident_api, "__file__", None),
        __file__,
    )
    if any(type(path) is not str for path in module_paths):
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "production repository source path differs"
        )
    roots = tuple(Path(path).absolute().parents[1] for path in module_paths)
    if any(root != DEFAULT_REPOSITORY_ROOT for root in roots):
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "production repository roots differ"
        )
    return DEFAULT_REPOSITORY_ROOT


def _validate_observed_at(value: object, *, label: str) -> str:
    if type(value) is not str or _OBSERVED_AT.fullmatch(value) is None:
        raise SkeletonGraphCustodyIncidentPersistenceError(f"{label} differs")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            f"{label} is not a real UTC timestamp"
        ) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise SkeletonGraphCustodyIncidentPersistenceError(
            f"{label} is not UTC"
        )
    return value


def _address(value: object, *, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            f"{label} is not a SHA-256 address"
        )
    return value


def _file_address(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical_record_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json(value) + b"\n"


def _record_digest(content: Mapping[str, Any]) -> str:
    return "sha256:" + canonical_digest(content)


@dataclass(frozen=True, slots=True)
class SkeletonGraphCustodyIncidentPersistenceReceipt:
    repository_root_absolute_path: str
    repository_root_st_dev: int
    repository_root_st_ino: int
    repository_root_st_mode: int
    authority_directory_relative_path: str
    authority_directory_absolute_path: str
    authority_directory_st_dev: int
    authority_directory_st_ino: int
    authority_directory_st_mode: int
    incident_commit: str
    incident_source_sha256: str
    incident_file_sha256: str
    incident_record_digest: str
    predecessor_filename: str
    predecessor_file_sha256: str
    predecessor_ledger_digest: str
    predecessor_corpus_digest: str
    predecessor_event_count: int
    claim_filename: str
    claim_schema: str
    claim_file_sha256: str
    claim_record_digest: str
    incident_event_digest: str
    incident_event_observed_at: str
    incident_event_sequence: int
    successor_filename: str
    successor_file_sha256: str
    successor_ledger_digest: str
    successor_event_count: int
    core_commit: str
    core_source_sha256: str
    core_expected_intent_schema: str
    canonical_source_sha256: str
    exposure_source_sha256: str
    runtime_source_snapshot_sha256: str
    calibration_prereg_source_sha256: str
    core_claim_schema_rejected: bool
    persistence_completed: bool
    serialized_receipt_is_authority: bool
    fresh_store_verification_required: bool
    calibration_pixels_authorized: bool
    action_program_or_label_reads_authorized: bool
    target_query_support_test_pixels_authorized: bool
    benchmark_claim_authorized: bool
    persistence_source_sha256: str
    record_digest: str

    def __post_init__(self) -> None:
        _validate_receipt_shape(self)

    def content_data(self) -> dict[str, Any]:
        return {"schema": RECEIPT_SCHEMA, **{
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "record_digest"
        }}

    def to_data(self) -> dict[str, Any]:
        _validate_receipt_shape(self)
        return {**self.content_data(), "record_digest": self.record_digest}

    @property
    def file_sha256(self) -> str:
        return _file_address(_canonical_record_bytes(self.to_data()))

    @classmethod
    def from_data(
        cls, raw: object
    ) -> "SkeletonGraphCustodyIncidentPersistenceReceipt":
        if cls is not SkeletonGraphCustodyIncidentPersistenceReceipt:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "persistence parser requires the exact receipt class"
            )
        if type(raw) is not dict or any(type(key) is not str for key in raw):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "persistence receipt is not an exact object"
            )
        expected = set(cls.__dataclass_fields__) | {"schema"}
        if (
            set(raw) != expected
            or type(raw.get("schema")) is not str
            or raw.get("schema") != RECEIPT_SCHEMA
        ):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "persistence receipt fields differ"
            )
        result = cls(**{name: raw[name] for name in cls.__dataclass_fields__})
        return result


@dataclass(frozen=True, slots=True, init=False)
class SkeletonGraphVerifiedCustodyIncidentPersistence:
    """Fresh local capability; unlike the serialized receipt it is not authority data."""

    receipt: SkeletonGraphCustodyIncidentPersistenceReceipt
    _issuance_token: object

    @classmethod
    def _from_fresh_store(
        cls,
        *,
        issuance_token: object,
    ) -> "SkeletonGraphVerifiedCustodyIncidentPersistence":
        if issuance_token is not _VERIFIED_ISSUANCE_TOKEN:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "verified persistence issuance is private"
            )
        receipt = _verify_at_repository_root(_production_repository_root())
        result = object.__new__(cls)
        object.__setattr__(result, "receipt", receipt)
        object.__setattr__(result, "_issuance_token", _VERIFIED_ISSUANCE_TOKEN)
        _validate_verified_persistence(result)
        return result


def _validate_verified_persistence(
    value: SkeletonGraphVerifiedCustodyIncidentPersistence,
) -> None:
    _source_preflight()
    if (
        type(value) is not SkeletonGraphVerifiedCustodyIncidentPersistence
        or getattr(value, "_issuance_token", None) is not _VERIFIED_ISSUANCE_TOKEN
        or type(getattr(value, "receipt", None))
        is not SkeletonGraphCustodyIncidentPersistenceReceipt
        or value.receipt.repository_root_absolute_path
        != str(_production_repository_root())
        or value.receipt.authority_directory_absolute_path
        != str(_production_repository_root() / AUTHORITY_RELATIVE_DIRECTORY)
    ):
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "verified persistence capability differs"
        )
    _validate_receipt_shape(value.receipt)


def _validate_receipt_shape(
    value: SkeletonGraphCustodyIncidentPersistenceReceipt,
) -> None:
    if type(value) is not SkeletonGraphCustodyIncidentPersistenceReceipt:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "persistence receipt has the wrong exact type"
        )
    for name in (
        "repository_root_st_dev",
        "repository_root_st_ino",
        "repository_root_st_mode",
        "authority_directory_st_dev",
        "authority_directory_st_ino",
        "authority_directory_st_mode",
        "predecessor_event_count",
        "incident_event_sequence",
        "successor_event_count",
    ):
        if type(getattr(value, name)) is not int:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                f"receipt {name} is not an exact integer"
            )
    for name in (
        "core_claim_schema_rejected",
        "persistence_completed",
        "serialized_receipt_is_authority",
        "fresh_store_verification_required",
        "calibration_pixels_authorized",
        "action_program_or_label_reads_authorized",
        "target_query_support_test_pixels_authorized",
        "benchmark_claim_authorized",
    ):
        if type(getattr(value, name)) is not bool:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                f"receipt {name} is not an exact bool"
            )
    for name in (
        "repository_root_absolute_path",
        "authority_directory_relative_path",
        "authority_directory_absolute_path",
        "incident_commit",
        "incident_source_sha256",
        "predecessor_filename",
        "claim_filename",
        "claim_schema",
        "incident_event_observed_at",
        "successor_filename",
        "core_commit",
        "core_source_sha256",
        "core_expected_intent_schema",
        "canonical_source_sha256",
        "exposure_source_sha256",
        "runtime_source_snapshot_sha256",
        "calibration_prereg_source_sha256",
    ):
        if type(getattr(value, name)) is not str:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                f"receipt {name} is not an exact string"
            )
    for name in (
        "incident_file_sha256",
        "incident_record_digest",
        "predecessor_file_sha256",
        "predecessor_ledger_digest",
        "predecessor_corpus_digest",
        "claim_file_sha256",
        "claim_record_digest",
        "incident_event_digest",
        "successor_file_sha256",
        "successor_ledger_digest",
        "persistence_source_sha256",
        "record_digest",
    ):
        _address(getattr(value, name), label=f"receipt {name}")
    if (
        not Path(value.repository_root_absolute_path).is_absolute()
        or value.authority_directory_relative_path
        != AUTHORITY_RELATIVE_DIRECTORY.as_posix()
        or value.authority_directory_absolute_path
        != str(
            Path(value.repository_root_absolute_path)
            / AUTHORITY_RELATIVE_DIRECTORY
        )
        or value.repository_root_st_dev < 0
        or value.repository_root_st_ino <= 0
        or not stat.S_ISDIR(value.repository_root_st_mode)
        or value.authority_directory_st_dev < 0
        or value.authority_directory_st_ino <= 0
        or not stat.S_ISDIR(value.authority_directory_st_mode)
        or value.incident_commit != INCIDENT_COMMIT
        or value.incident_source_sha256 != PINNED_INCIDENT_SOURCE_SHA256
        or value.incident_file_sha256 != PINNED_INCIDENT_FILE_SHA256
        or value.incident_record_digest != PINNED_INCIDENT_RECORD_DIGEST
        or value.predecessor_filename != PREDECESSOR_FILENAME
        or value.predecessor_file_sha256
        != incident_api.EXPOSURE_PREDECESSOR_FILE_SHA256
        or value.predecessor_ledger_digest
        != incident_api.EXPOSURE_PREDECESSOR_LEDGER_DIGEST
        or value.predecessor_corpus_digest != incident_api.EXPOSURE_CORPUS_DIGEST
        or value.predecessor_event_count != 158
        or value.claim_filename != incident_api.CAMPAIGN_INTENT_FILENAME
        or value.claim_schema != incident_api.TOMBSTONE_CLAIM_SCHEMA
        or value.incident_event_sequence != 158
        or value.successor_event_count != 159
        or value.successor_filename
        != value.successor_ledger_digest.removeprefix("sha256:")
        + ".exposure.json"
        or value.core_commit != PINNED_CORE_COMMIT
        or value.core_source_sha256 != PINNED_CORE_SOURCE_SHA256
        or value.core_expected_intent_schema != PINNED_CORE_INTENT_SCHEMA
        or value.canonical_source_sha256 != PINNED_CANONICAL_SOURCE_SHA256
        or value.exposure_source_sha256 != PINNED_EXPOSURE_SOURCE_SHA256
        or value.runtime_source_snapshot_sha256
        != PINNED_RUNTIME_SOURCE_SNAPSHOT_SHA256
        or value.calibration_prereg_source_sha256
        != PINNED_CALIBRATION_PREREG_SOURCE_SHA256
        or value.core_claim_schema_rejected is not True
        or value.persistence_completed is not True
        or value.serialized_receipt_is_authority is not False
        or value.fresh_store_verification_required is not True
        or value.calibration_pixels_authorized is not False
        or value.action_program_or_label_reads_authorized is not False
        or value.target_query_support_test_pixels_authorized is not False
        or value.benchmark_claim_authorized is not False
        or value.persistence_source_sha256 != "sha256:" + source_sha256()
        or value.record_digest != _record_digest(value.content_data())
    ):
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "persistence receipt policy differs"
        )
    _validate_observed_at(
        value.incident_event_observed_at,
        label="receipt incident event observed_at",
    )


def _root_custody(repository_root: Path) -> core._OutputDirectoryCustody:
    root = Path(repository_root)
    if not root.is_absolute() or root != Path(os.path.abspath(root)):
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "repository root must be absolute and lexical-canonical"
        )
    try:
        return core._open_existing_directory(root, label="incident repository root")
    except Exception as exc:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "incident repository root custody failed"
        ) from exc


def _recheck_custody(custody: core._OutputDirectoryCustody) -> None:
    descriptor = custody._open()
    os.close(descriptor)


def _open_relative_directory(
    root: core._OutputDirectoryCustody,
    relative: Path,
    *,
    label: str,
) -> core._OutputDirectoryCustody:
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise SkeletonGraphCustodyIncidentPersistenceError(
            f"{label} relative path differs"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    check_descriptor: int | None = None
    root_check: int | None = None
    try:
        descriptor = root._open()
        for component in relative.parts:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        found = os.fstat(descriptor)
        absolute = root.path / relative
        check_descriptor, checked = core._open_directory_fd_no_symlink(absolute)
        root_check = root._open()
        identity = (found.st_dev, found.st_ino, found.st_mode)
        if (
            identity != (checked.st_dev, checked.st_ino, checked.st_mode)
            or not stat.S_ISDIR(found.st_mode)
        ):
            raise OSError(f"{label} identity differs")
        result = core._OutputDirectoryCustody(
            absolute,
            found.st_dev,
            found.st_ino,
            found.st_mode,
            descriptor,
        )
        descriptor = None
        return result
    except Exception as exc:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            f"{label} custody failed"
        ) from exc
    finally:
        for open_descriptor in (descriptor, check_descriptor, root_check):
            if open_descriptor is not None:
                os.close(open_descriptor)


def _authority_custody(
    root: core._OutputDirectoryCustody,
) -> core._OutputDirectoryCustody:
    return _open_relative_directory(
        root,
        AUTHORITY_RELATIVE_DIRECTORY,
        label="incident authority directory",
    )


def _stat_identity(found: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        found.st_dev,
        found.st_ino,
        found.st_mode,
        found.st_size,
        found.st_mtime_ns,
        found.st_ctime_ns,
    )


def _read_custody_file(
    custody: core._OutputDirectoryCustody,
    name: str,
    *,
    label: str,
    maximum: int = 16 << 20,
) -> bytes:
    descriptor = custody._open()
    file_descriptor: int | None = None
    try:
        if Path(name).name != name or name in ("", ".", ".."):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                f"{label} filename differs"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        file_descriptor = os.open(name, flags, dir_fd=descriptor)
        before = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                f"{label} is not a bounded regular file"
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
        named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        identity = _stat_identity(before)
        raw = b"".join(chunks)
        if (
            _stat_identity(after) != identity
            or _stat_identity(named) != identity
            or len(raw) != before.st_size
        ):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                f"{label} changed while reading"
            )
        os.close(file_descriptor)
        file_descriptor = None
        reloaded = core._read_dirfd_bytes(
            descriptor, name, label=label, maximum=maximum
        )
        final_named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if reloaded != raw or _stat_identity(final_named) != identity:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                f"{label} changed during fresh reload"
            )
        _recheck_custody(custody)
        return raw
    except SkeletonGraphCustodyIncidentPersistenceError:
        raise
    except Exception as exc:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            f"cannot read stable {label}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)


def _load_incident(
    root: core._OutputDirectoryCustody,
) -> tuple[incident_api.SkeletonGraphCustodyIncident, bytes]:
    parent = _open_relative_directory(
        root,
        INCIDENT_RELATIVE_PATH.parent,
        label="incident record parent",
    )
    try:
        raw = _read_custody_file(
            parent,
            INCIDENT_RELATIVE_PATH.name,
            label="incident record",
            maximum=256 << 10,
        )
    finally:
        parent.close()
    if _file_address(raw) != PINNED_INCIDENT_FILE_SHA256:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "incident record file address differs"
        )
    try:
        data = json.loads(raw)
        value = incident_api.SkeletonGraphCustodyIncident.from_data(data)
    except Exception as exc:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "incident record verification failed"
        ) from exc
    if (
        value.record_digest != PINNED_INCIDENT_RECORD_DIGEST
        or raw != _canonical_record_bytes(value.to_data())
        or incident_api.source_sha256() != PINNED_INCIDENT_SOURCE_SHA256
    ):
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "incident record/source binding differs"
        )
    return value, raw


def _load_predecessor(
    custody: core._OutputDirectoryCustody,
) -> tuple[ExposureLedger, bytes]:
    raw = _read_custody_file(
        custody,
        PREDECESSOR_FILENAME,
        label="incident exposure predecessor",
        maximum=32 << 20,
    )
    if _file_address(raw) != incident_api.EXPOSURE_PREDECESSOR_FILE_SHA256:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "incident predecessor file address differs"
        )
    try:
        data = json.loads(raw)
        ledger = ExposureLedger.from_dict(data)
    except Exception as exc:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "incident predecessor verification failed"
        ) from exc
    if (
        type(ledger) is not ExposureLedger
        or ledger.digest != incident_api.EXPOSURE_PREDECESSOR_LEDGER_DIGEST
        or ledger.corpus_digest != incident_api.EXPOSURE_CORPUS_DIGEST
        or len(ledger.events) != 158
    ):
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "incident predecessor differs"
        )
    return ledger, raw


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _read_optional(
    custody: core._OutputDirectoryCustody, name: str, *, label: str
) -> bytes | None:
    descriptor = custody._open()
    try:
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None
    finally:
        os.close(descriptor)
    return _read_custody_file(custody, name, label=label)


def _pending_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        re.escape(f".{name}.pending.")
        + r"[0-9]+\.[0-9]+\.[0-9a-f]{32}\Z"
    )


def _cleanup_private_temps(
    custody: core._OutputDirectoryCustody,
    names: tuple[str, ...],
) -> None:
    descriptor = custody._open()
    removed = False
    try:
        patterns = tuple(_pending_pattern(name) for name in names)
        for entry in os.listdir(descriptor):
            if type(entry) is not str or not any(
                pattern.fullmatch(entry) for pattern in patterns
            ):
                continue
            try:
                found = os.stat(entry, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(found.st_mode) or stat.S_IMODE(found.st_mode) != 0o600:
                raise SkeletonGraphCustodyIncidentPersistenceError(
                    "incident private temporary has an unexpected type"
                )
            try:
                os.unlink(entry, dir_fd=descriptor)
            except FileNotFoundError:
                continue
            removed = True
        if removed:
            os.fsync(descriptor)
    except SkeletonGraphCustodyIncidentPersistenceError:
        raise
    except Exception as exc:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "cannot clean incident private temporaries"
        ) from exc
    finally:
        os.close(descriptor)
    _recheck_custody(custody)


def _assert_no_private_temps(
    custody: core._OutputDirectoryCustody,
    names: tuple[str, ...],
) -> None:
    descriptor = custody._open()
    try:
        patterns = tuple(_pending_pattern(name) for name in names)
        if any(
            type(entry) is str
            and any(pattern.fullmatch(entry) for pattern in patterns)
            for entry in os.listdir(descriptor)
        ):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "incident private temporary inventory is not empty"
            )
    finally:
        os.close(descriptor)
    _recheck_custody(custody)


def _direct_successor_filenames(
    custody: core._OutputDirectoryCustody,
    predecessor: ExposureLedger,
) -> tuple[str, ...]:
    descriptor = custody._open()
    try:
        before = tuple(
            sorted(
                entry
                for entry in os.listdir(descriptor)
                if type(entry) is str and _EXPOSURE_FILENAME.fullmatch(entry)
            )
        )
    finally:
        os.close(descriptor)
    if len(before) > 1024:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "exposure-ledger inventory exceeds the fixed cap"
        )
    direct: list[str] = []
    for filename in before:
        raw = _read_custody_file(
            custody,
            filename,
            label=f"exposure ledger inventory {filename}",
            maximum=32 << 20,
        )
        try:
            ledger = ExposureLedger.from_dict(json.loads(raw))
        except Exception as exc:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "exposure-ledger inventory contains an invalid record"
            ) from exc
        if (
            type(ledger) is not ExposureLedger
            or raw != ledger.to_json().encode("utf-8")
            or filename
            != ledger.digest.removeprefix("sha256:") + ".exposure.json"
        ):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "exposure-ledger inventory binding differs"
            )
        if (
            ledger.corpus_digest == predecessor.corpus_digest
            and len(ledger.events) == len(predecessor.events) + 1
            and ledger.events[:-1] == predecessor.events
        ):
            direct.append(filename)
    descriptor = custody._open()
    try:
        after = tuple(
            sorted(
                entry
                for entry in os.listdir(descriptor)
                if type(entry) is str and _EXPOSURE_FILENAME.fullmatch(entry)
            )
        )
    finally:
        os.close(descriptor)
    if after != before:
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "exposure-ledger inventory changed while reading"
        )
    _recheck_custody(custody)
    return tuple(direct)


def _core_rejects_claim(raw: bytes) -> None:
    if (
        core.source_sha256() != PINNED_CORE_SOURCE_SHA256
        or core.CAMPAIGN_ATTEMPT_AUTHORITY_SCHEMA != PINNED_CORE_INTENT_SCHEMA
    ):
        raise SkeletonGraphCustodyIncidentPersistenceError(
            "calibration core binding differs"
        )
    try:
        core._record_from_bytes(
            raw,
            schema=core.CAMPAIGN_ATTEMPT_AUTHORITY_SCHEMA,
            label="incident tombstone as campaign intent",
        )
    except core.SkeletonGraphCalibrationRunnerError:
        return
    raise SkeletonGraphCustodyIncidentPersistenceError(
        "calibration core accepted the incident tombstone as an authorization"
    )


def _build_receipt(
    *,
    root: core._OutputDirectoryCustody,
    authority: core._OutputDirectoryCustody,
    incident: incident_api.SkeletonGraphCustodyIncident,
    incident_raw: bytes,
    predecessor: ExposureLedger,
    claim: incident_api.SkeletonGraphIncidentTombstoneClaim,
    claim_raw: bytes,
    successor: ExposureLedger,
    successor_raw: bytes,
) -> SkeletonGraphCustodyIncidentPersistenceReceipt:
    event = successor.events[-1]
    values: dict[str, Any] = {
        "repository_root_absolute_path": str(root.path),
        "repository_root_st_dev": root.device,
        "repository_root_st_ino": root.inode,
        "repository_root_st_mode": root.mode,
        "authority_directory_relative_path": (
            AUTHORITY_RELATIVE_DIRECTORY.as_posix()
        ),
        "authority_directory_absolute_path": str(authority.path),
        "authority_directory_st_dev": authority.device,
        "authority_directory_st_ino": authority.inode,
        "authority_directory_st_mode": authority.mode,
        "incident_commit": INCIDENT_COMMIT,
        "incident_source_sha256": PINNED_INCIDENT_SOURCE_SHA256,
        "incident_file_sha256": _file_address(incident_raw),
        "incident_record_digest": incident.record_digest,
        "predecessor_filename": PREDECESSOR_FILENAME,
        "predecessor_file_sha256": incident_api.EXPOSURE_PREDECESSOR_FILE_SHA256,
        "predecessor_ledger_digest": predecessor.digest,
        "predecessor_corpus_digest": predecessor.corpus_digest,
        "predecessor_event_count": len(predecessor.events),
        "claim_filename": incident_api.CAMPAIGN_INTENT_FILENAME,
        "claim_schema": incident_api.TOMBSTONE_CLAIM_SCHEMA,
        "claim_file_sha256": _file_address(claim_raw),
        "claim_record_digest": claim.record_digest,
        "incident_event_digest": event.digest,
        "incident_event_observed_at": event.observed_at,
        "incident_event_sequence": event.sequence,
        "successor_filename": claim.successor_filename,
        "successor_file_sha256": _file_address(successor_raw),
        "successor_ledger_digest": successor.digest,
        "successor_event_count": len(successor.events),
        "core_commit": PINNED_CORE_COMMIT,
        "core_source_sha256": PINNED_CORE_SOURCE_SHA256,
        "core_expected_intent_schema": PINNED_CORE_INTENT_SCHEMA,
        "canonical_source_sha256": PINNED_CANONICAL_SOURCE_SHA256,
        "exposure_source_sha256": PINNED_EXPOSURE_SOURCE_SHA256,
        "runtime_source_snapshot_sha256": (
            PINNED_RUNTIME_SOURCE_SNAPSHOT_SHA256
        ),
        "calibration_prereg_source_sha256": (
            PINNED_CALIBRATION_PREREG_SOURCE_SHA256
        ),
        "core_claim_schema_rejected": True,
        "persistence_completed": True,
        "serialized_receipt_is_authority": False,
        "fresh_store_verification_required": True,
        "calibration_pixels_authorized": False,
        "action_program_or_label_reads_authorized": False,
        "target_query_support_test_pixels_authorized": False,
        "benchmark_claim_authorized": False,
        "persistence_source_sha256": "sha256:" + source_sha256(),
    }
    provisional = object.__new__(SkeletonGraphCustodyIncidentPersistenceReceipt)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    receipt = SkeletonGraphCustodyIncidentPersistenceReceipt(
        **values, record_digest=_record_digest(provisional.content_data())
    )
    _validate_receipt_shape(receipt)
    return receipt


def _persist_at_repository_root(
    repository_root: Path,
    *,
    observed_at: str | None = None,
) -> SkeletonGraphCustodyIncidentPersistenceReceipt:
    _source_preflight()
    if observed_at is not None:
        _validate_observed_at(observed_at, label="incident observed_at")
    root = _root_custody(repository_root)
    authority: core._OutputDirectoryCustody | None = None
    try:
        incident, incident_raw = _load_incident(root)
        authority = _authority_custody(root)
        predecessor, _predecessor_raw = _load_predecessor(authority)
        initial_direct_successors = _direct_successor_filenames(
            authority, predecessor
        )
        existing_claim_raw = _read_optional(
            authority,
            incident_api.CAMPAIGN_INTENT_FILENAME,
            label="incident tombstone claim",
        )
        existing_receipt_raw = _read_optional(
            authority,
            RECEIPT_FILENAME,
            label="incident persistence receipt",
        )
        claim_created = existing_claim_raw is None
        if claim_created:
            if existing_receipt_raw is not None or initial_direct_successors:
                raise SkeletonGraphCustodyIncidentPersistenceError(
                    "incident successor or receipt predates the fixed claim"
                )
            chosen = observed_at or _now()
            _validate_observed_at(chosen, label="chosen incident observed_at")
            successor, claim = incident_api.build_incident_exposure_tombstone(
                predecessor, incident=incident, observed_at=chosen
            )
            claim_raw = incident_api.tombstone_claim_bytes(claim)
            successor_raw = successor.to_json().encode("utf-8")
            _cleanup_private_temps(
                authority,
                (
                    incident_api.CAMPAIGN_INTENT_FILENAME,
                    claim.successor_filename,
                    RECEIPT_FILENAME,
                ),
            )
            if _read_optional(
                authority,
                claim.successor_filename,
                label="incident exposure successor",
            ) is not None:
                raise SkeletonGraphCustodyIncidentPersistenceError(
                    "incident successor predates the fixed claim"
                )
        else:
            try:
                claim = incident_api.SkeletonGraphIncidentTombstoneClaim.from_data(
                    json.loads(existing_claim_raw)
                )
            except Exception as exc:
                raise SkeletonGraphCustodyIncidentPersistenceError(
                    "existing campaign claim is not this incident tombstone"
                ) from exc
            _validate_observed_at(
                claim.incident_event_observed_at,
                label="existing incident observed_at",
            )
            if observed_at is not None and observed_at != claim.incident_event_observed_at:
                raise SkeletonGraphCustodyIncidentPersistenceError(
                    "alternate incident timestamp would fork the campaign"
                )
            successor, expected_claim = incident_api.build_incident_exposure_tombstone(
                predecessor,
                incident=incident,
                observed_at=claim.incident_event_observed_at,
            )
            if claim != expected_claim:
                raise SkeletonGraphCustodyIncidentPersistenceError(
                    "existing incident tombstone claim differs"
                )
            claim_raw = incident_api.tombstone_claim_bytes(claim)
            successor_raw = successor.to_json().encode("utf-8")
            if initial_direct_successors not in (
                (),
                (claim.successor_filename,),
            ):
                raise SkeletonGraphCustodyIncidentPersistenceError(
                    "incident exposure ledger has an alternate direct child"
                )
            if existing_claim_raw != claim_raw:
                raise SkeletonGraphCustodyIncidentPersistenceError(
                    "existing incident tombstone bytes differ"
                )
            existing_successor_raw = _read_optional(
                authority,
                claim.successor_filename,
                label="incident exposure successor",
            )
            if existing_receipt_raw is not None and existing_successor_raw is None:
                raise SkeletonGraphCustodyIncidentPersistenceError(
                    "terminal incident receipt lacks its predecessor successor"
                )
            if (
                existing_successor_raw is not None
                and existing_successor_raw != successor_raw
            ):
                raise SkeletonGraphCustodyIncidentPersistenceError(
                    "existing incident successor differs"
                )
        _core_rejects_claim(claim_raw)
        try:
            core._atomic_write_once_bytes(
                authority,
                incident_api.CAMPAIGN_INTENT_FILENAME,
                claim_raw,
                label="incident tombstone claim",
                allow_identical_existing=not claim_created,
            )
            _cleanup_private_temps(
                authority, (incident_api.CAMPAIGN_INTENT_FILENAME,)
            )
        except Exception as exc:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "cannot persist incident tombstone claim"
            ) from exc
        _stage_hook("claim")
        try:
            core._atomic_write_once_bytes(
                authority,
                claim.successor_filename,
                successor_raw,
                label="incident exposure successor",
                allow_identical_existing=not claim_created,
            )
            _cleanup_private_temps(authority, (claim.successor_filename,))
        except Exception as exc:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "cannot persist incident exposure successor"
            ) from exc
        _stage_hook("successor")
        if _direct_successor_filenames(authority, predecessor) != (
            claim.successor_filename,
        ):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "incident successor inventory differs after publication"
            )
        receipt = _build_receipt(
            root=root,
            authority=authority,
            incident=incident,
            incident_raw=incident_raw,
            predecessor=predecessor,
            claim=claim,
            claim_raw=claim_raw,
            successor=successor,
            successor_raw=successor_raw,
        )
        receipt_raw = _canonical_record_bytes(receipt.to_data())
        try:
            core._atomic_write_once_bytes(
                authority,
                RECEIPT_FILENAME,
                receipt_raw,
                label="incident persistence receipt",
                allow_identical_existing=not claim_created,
            )
            _cleanup_private_temps(authority, (RECEIPT_FILENAME,))
        except Exception as exc:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "cannot persist incident receipt"
            ) from exc
        _stage_hook("receipt")
        _recheck_custody(root)
        _recheck_custody(authority)
    finally:
        if authority is not None:
            authority.close()
        root.close()
    return _verify_at_repository_root(repository_root)


def _verify_at_repository_root(
    repository_root: Path,
) -> SkeletonGraphCustodyIncidentPersistenceReceipt:
    """Fresh, zero-write verification of the fixed store and all chain records."""

    _source_preflight()
    root = _root_custody(repository_root)
    authority: core._OutputDirectoryCustody | None = None
    try:
        incident, incident_raw = _load_incident(root)
        authority = _authority_custody(root)
        predecessor, predecessor_raw = _load_predecessor(authority)
        claim_raw = _read_custody_file(
            authority,
            incident_api.CAMPAIGN_INTENT_FILENAME,
            label="incident tombstone claim",
        )
        try:
            claim = incident_api.SkeletonGraphIncidentTombstoneClaim.from_data(
                json.loads(claim_raw)
            )
        except Exception as exc:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "persisted incident claim is invalid"
            ) from exc
        _validate_observed_at(
            claim.incident_event_observed_at,
            label="persisted incident observed_at",
        )
        if _direct_successor_filenames(authority, predecessor) != (
            claim.successor_filename,
        ):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "persisted incident direct-successor inventory differs"
            )
        successor_raw = _read_custody_file(
            authority,
            claim.successor_filename,
            label="incident exposure successor",
            maximum=32 << 20,
        )
        try:
            successor = ExposureLedger.from_dict(json.loads(successor_raw))
        except Exception as exc:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "persisted incident successor is invalid"
            ) from exc
        incident_api.verify_incident_tombstone_claim(
            predecessor,
            successor=successor,
            claim=claim,
            incident=incident,
        )
        if (
            claim_raw != incident_api.tombstone_claim_bytes(claim)
            or successor_raw != successor.to_json().encode("utf-8")
        ):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "persisted incident bytes are not exact"
            )
        _core_rejects_claim(claim_raw)
        expected = _build_receipt(
            root=root,
            authority=authority,
            incident=incident,
            incident_raw=incident_raw,
            predecessor=predecessor,
            claim=claim,
            claim_raw=claim_raw,
            successor=successor,
            successor_raw=successor_raw,
        )
        receipt_raw = _read_custody_file(
            authority,
            RECEIPT_FILENAME,
            label="incident persistence receipt",
        )
        try:
            receipt = SkeletonGraphCustodyIncidentPersistenceReceipt.from_data(
                json.loads(receipt_raw)
            )
        except Exception as exc:
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "persisted incident receipt is invalid"
            ) from exc
        if (
            receipt != expected
            or receipt_raw != _canonical_record_bytes(receipt.to_data())
            or receipt.file_sha256 != _file_address(receipt_raw)
        ):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "persisted incident receipt differs"
            )
        _assert_no_private_temps(
            authority,
            (
                incident_api.CAMPAIGN_INTENT_FILENAME,
                claim.successor_filename,
                RECEIPT_FILENAME,
            ),
        )
        incident_again, incident_raw_again = _load_incident(root)
        predecessor_again, predecessor_raw_again = _load_predecessor(authority)
        claim_raw_again = _read_custody_file(
            authority,
            incident_api.CAMPAIGN_INTENT_FILENAME,
            label="incident tombstone claim final reread",
        )
        successor_raw_again = _read_custody_file(
            authority,
            claim.successor_filename,
            label="incident exposure successor final reread",
            maximum=32 << 20,
        )
        receipt_raw_again = _read_custody_file(
            authority,
            RECEIPT_FILENAME,
            label="incident persistence receipt final reread",
        )
        if _direct_successor_filenames(authority, predecessor) != (
            claim.successor_filename,
        ):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "incident direct-successor inventory changed during verification"
            )
        _assert_no_private_temps(
            authority,
            (
                incident_api.CAMPAIGN_INTENT_FILENAME,
                claim.successor_filename,
                RECEIPT_FILENAME,
            ),
        )
        if (
            incident_again != incident
            or incident_raw_again != incident_raw
            or predecessor_again != predecessor
            or predecessor_raw_again != predecessor_raw
            or claim_raw_again != claim_raw
            or successor_raw_again != successor_raw
            or receipt_raw_again != receipt_raw
        ):
            raise SkeletonGraphCustodyIncidentPersistenceError(
                "incident persistence chain changed during final verification"
            )
        _recheck_custody(root)
        _recheck_custody(authority)
        return receipt
    finally:
        if authority is not None:
            authority.close()
        root.close()


def persist_custody_incident_tombstone(
) -> SkeletonGraphVerifiedCustodyIncidentPersistence:
    """Persist only in the repository that owns this exact loaded source."""

    root = _production_repository_root()
    _persist_at_repository_root(root)
    return SkeletonGraphVerifiedCustodyIncidentPersistence._from_fresh_store(
        issuance_token=_VERIFIED_ISSUANCE_TOKEN,
    )


def verify_persisted_custody_incident_tombstone(
) -> SkeletonGraphVerifiedCustodyIncidentPersistence:
    """Freshly verify only the repository that owns this exact loaded source."""

    return SkeletonGraphVerifiedCustodyIncidentPersistence._from_fresh_store(
        issuance_token=_VERIFIED_ISSUANCE_TOKEN,
    )


__all__ = (
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA",
    "SkeletonGraphCustodyIncidentPersistenceError",
    "SkeletonGraphCustodyIncidentPersistenceReceipt",
    "SkeletonGraphVerifiedCustodyIncidentPersistence",
    "persist_custody_incident_tombstone",
    "source_sha256",
    "verify_persisted_custody_incident_tombstone",
)
