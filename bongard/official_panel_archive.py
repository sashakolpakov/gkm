"""Authenticated, bounded reads from the pinned official Bongard release ZIP.

This module is intentionally independent of every historical benchmark runner.
It exposes exact PNG bytes only after a caller has supplied the already-durable
execution and exposure commitments.  Those commitments are recorded in each
released-panel record; they are not treated as ambient process state.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass, field
import base64
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
import zipfile

from bongard.canonical import canonical_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.release import OfficialReleaseDescriptor


ARCHIVE_SCHEMA = "gkm.bongard-official-panel-archive.v1"
RECEIPT_SCHEMA = "gkm.bongard-official-panel-receipt.v1"
RELEASED_PANEL_SCHEMA = "gkm.bongard-released-panel.v1"

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(
    r"(?P<family>bd|ff|hd)/(?P=family)_[A-Za-z0-9_-]+/[01]/[0-6]\.png\Z"
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PNG_BYTES = 4_000_000


class OfficialPanelArchiveError(ValueError):
    """The official archive, a panel release, or a replay differed."""


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise OfficialPanelArchiveError(f"{label} must be a sha256: address")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != expected:
        raise OfficialPanelArchiveError(f"{label} fields differ from schema")


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def _descriptor_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_regular_no_follow(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OfficialPanelArchiveError(f"cannot open official archive {path}") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
        os.close(descriptor)
        raise OfficialPanelArchiveError("official archive is not a private regular file")
    return descriptor


def _hash_open_file(descriptor: int) -> tuple[str, int, tuple[int, ...]]:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    after = os.fstat(descriptor)
    if _descriptor_identity(before) != _descriptor_identity(after):
        raise OfficialPanelArchiveError("official archive changed while hashing")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), before.st_size, _descriptor_identity(before)


def _panel_member(panel_id: str) -> str:
    if type(panel_id) is not str or _PANEL_ID.fullmatch(panel_id) is None:
        raise OfficialPanelArchiveError(
            "panel_id is outside the official bd/ff/hd PNG namespace"
        )
    family, task_and_panel = panel_id.split("/", 1)
    return f"ShapeBongard_V2/{family}/images/{task_and_panel}"


@dataclass(frozen=True, slots=True)
class OfficialPanelReceipt:
    panel_id: str
    archive_member: str
    sha256: str
    size_bytes: int
    zip_crc32: int
    release_descriptor_digest: str
    archive_digest: str
    central_directory_digest: str
    record_digest: str

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": RECEIPT_SCHEMA,
            "panel_id": self.panel_id,
            "archive_member": self.archive_member,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "zip_crc32": self.zip_crc32,
            "release_descriptor_digest": self.release_descriptor_digest,
            "archive_digest": self.archive_digest,
            "central_directory_digest": self.central_directory_digest,
            **_authority_data(),
        }

    def __post_init__(self) -> None:
        expected_member = _panel_member(self.panel_id)
        if type(self.archive_member) is not str or self.archive_member != expected_member:
            raise OfficialPanelArchiveError("receipt member differs from panel_id")
        for name in (
            "sha256",
            "release_descriptor_digest",
            "archive_digest",
            "central_directory_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            type(self.size_bytes) is not int
            or not 0 < self.size_bytes <= _MAX_PNG_BYTES
            or type(self.zip_crc32) is not int
            or not 0 <= self.zip_crc32 <= 0xFFFFFFFF
            or self.record_digest != _address(self.content_dict())
        ):
            raise OfficialPanelArchiveError("panel receipt identity differs")

    @classmethod
    def seal(
        cls,
        *,
        panel_id: str,
        payload: bytes,
        archive_member: str,
        zip_crc32: int,
        release_descriptor_digest: str,
        archive_digest: str,
        central_directory_digest: str,
    ) -> "OfficialPanelReceipt":
        values: dict[str, object] = {
            "panel_id": panel_id,
            "archive_member": archive_member,
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "zip_crc32": zip_crc32,
            "release_descriptor_digest": release_descriptor_digest,
            "archive_digest": archive_digest,
            "central_directory_digest": central_directory_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(provisional.content_dict()),
        )

    def to_data(self) -> dict[str, object]:
        return {**self.content_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "OfficialPanelReceipt":
        expected = {
            "schema",
            "panel_id",
            "archive_member",
            "sha256",
            "size_bytes",
            "zip_crc32",
            "release_descriptor_digest",
            "archive_digest",
            "central_directory_digest",
            *_authority_data(),
            "record_digest",
        }
        _exact_fields(value, expected, "official panel receipt")
        result = cls(
            panel_id=value["panel_id"],
            archive_member=value["archive_member"],
            sha256=value["sha256"],
            size_bytes=value["size_bytes"],
            zip_crc32=value["zip_crc32"],
            release_descriptor_digest=value["release_descriptor_digest"],
            archive_digest=value["archive_digest"],
            central_directory_digest=value["central_directory_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise OfficialPanelArchiveError("panel receipt is not canonical")
        return result


@dataclass(frozen=True, slots=True, init=False)
class OfficialPanelArchive:
    """Pinned ZIP identity and immutable central-directory inventory."""

    release_descriptor_digest: str
    archive_digest: str
    archive_size_bytes: int
    central_directory_digest: str
    archive_path: Path = field(repr=False, compare=False)
    archive_identity: tuple[int, ...] = field(repr=False, compare=False)
    members: tuple[tuple[str, int, int], ...] = field(repr=False, compare=False)
    record_digest: str

    def __post_init__(self) -> None:
        for name in (
            "release_descriptor_digest",
            "archive_digest",
            "central_directory_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            type(self.archive_size_bytes) is not int
            or self.archive_size_bytes <= 0
            or not isinstance(self.archive_path, Path)
            or not self.archive_path.is_absolute()
            or type(self.archive_identity) is not tuple
            or len(self.archive_identity) != 5
            or any(type(item) is not int for item in self.archive_identity)
            or type(self.members) is not tuple
            or any(type(row) is not tuple or len(row) != 3 for row in self.members)
            or self.members != tuple(sorted(self.members))
            or len({item[0] for item in self.members}) != len(self.members)
            or any(
                type(name) is not str
                or not name
                or type(size) is not int
                or size < 0
                or type(crc) is not int
                or not 0 <= crc <= 0xFFFFFFFF
                for name, size, crc in self.members
            )
        ):
            raise OfficialPanelArchiveError("official archive inventory is invalid")
        expected_central = _address(
            {
                "schema": ARCHIVE_SCHEMA,
                "members": [
                    {"name": name, "size_bytes": size, "zip_crc32": crc}
                    for name, size, crc in self.members
                ],
            }
        )
        if (
            self.central_directory_digest != expected_central
            or self.record_digest != _address(self.identity_data())
        ):
            raise OfficialPanelArchiveError("official archive binding differs")

    @classmethod
    def load(
        cls,
        descriptor: OfficialReleaseDescriptor,
        archive_path: str | Path,
        *,
        expected_release_descriptor_digest: str,
    ) -> "OfficialPanelArchive":
        if type(descriptor) is not OfficialReleaseDescriptor:
            raise TypeError("descriptor must be exact OfficialReleaseDescriptor")
        cold_descriptor = OfficialReleaseDescriptor.from_dict(descriptor.to_dict())
        if cold_descriptor != descriptor:
            raise OfficialPanelArchiveError(
                "release descriptor canonical replay differs"
            )
        descriptor = cold_descriptor
        expected = _require_address(
            expected_release_descriptor_digest, "release descriptor digest"
        )
        if descriptor.digest != expected:
            raise OfficialPanelArchiveError("release descriptor differs from commitment")
        archive = Path(os.path.abspath(os.path.expanduser(str(archive_path))))
        if archive.name != descriptor.archive_filename:
            raise OfficialPanelArchiveError("official archive filename differs")
        descriptor_fd = _open_regular_no_follow(archive)
        try:
            digest, size, identity = _hash_open_file(descriptor_fd)
            if (
                "sha256:" + digest != descriptor.archive_sha256
                or size != descriptor.archive_size_bytes
            ):
                raise OfficialPanelArchiveError("official archive identity differs")
            with os.fdopen(os.dup(descriptor_fd), "rb") as handle:
                with zipfile.ZipFile(handle) as bundle:
                    infos = bundle.infolist()
        except (OSError, zipfile.BadZipFile) as exc:
            if isinstance(exc, OfficialPanelArchiveError):
                raise
            raise OfficialPanelArchiveError("official release ZIP is invalid") from exc
        finally:
            os.close(descriptor_fd)
        names = tuple(info.filename for info in infos)
        if len(names) != len(set(names)):
            raise OfficialPanelArchiveError("official release ZIP repeats members")
        members = tuple(
            sorted((item.filename, item.file_size, item.CRC) for item in infos)
        )
        central = _address(
            {
                "schema": ARCHIVE_SCHEMA,
                "members": [
                    {"name": name, "size_bytes": item_size, "zip_crc32": crc}
                    for name, item_size, crc in members
                ],
            }
        )
        values: dict[str, object] = {
            "release_descriptor_digest": expected,
            "archive_digest": descriptor.archive_sha256,
            "archive_size_bytes": descriptor.archive_size_bytes,
            "central_directory_digest": central,
            "archive_path": archive,
            "archive_identity": identity,
            "members": members,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        object.__setattr__(
            provisional,
            "record_digest",
            _address(provisional.identity_data()),
        )
        provisional.__post_init__()
        return provisional

    def identity_data(self) -> dict[str, object]:
        return {
            "schema": ARCHIVE_SCHEMA,
            "release_descriptor_digest": self.release_descriptor_digest,
            "archive_digest": self.archive_digest,
            "archive_size_bytes": self.archive_size_bytes,
            "central_directory_digest": self.central_directory_digest,
            "layout": "ShapeBongard_V2/<family>/images/<task>/<side>/<index>.png",
            "read_policy": "exact-official-zip-member-after-durable-exposure",
            **_authority_data(),
        }

    def read_panel(self, panel_id: str) -> tuple[bytes, OfficialPanelReceipt]:
        member = _panel_member(panel_id)
        inventory = {name: (size, crc) for name, size, crc in self.members}
        if member not in inventory:
            raise OfficialPanelArchiveError("panel is absent from official archive")
        advertised_size, _ = inventory[member]
        if not 0 < advertised_size <= _MAX_PNG_BYTES:
            raise OfficialPanelArchiveError(
                "official panel exceeds the pre-decompression byte bound"
            )
        descriptor = _open_regular_no_follow(self.archive_path)
        try:
            before = os.fstat(descriptor)
            if _descriptor_identity(before) != self.archive_identity:
                raise OfficialPanelArchiveError("official archive changed after pinning")
            digest, archive_size, live_identity = _hash_open_file(descriptor)
            if (
                "sha256:" + digest != self.archive_digest
                or archive_size != self.archive_size_bytes
                or live_identity != self.archive_identity
            ):
                raise OfficialPanelArchiveError(
                    "official archive bytes differ from the pinned release"
                )
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                with zipfile.ZipFile(handle) as bundle:
                    infos = bundle.infolist()
                    names = tuple(item.filename for item in infos)
                    live_members = tuple(
                        sorted(
                            (item.filename, item.file_size, item.CRC)
                            for item in infos
                        )
                    )
                    if (
                        len(names) != len(set(names))
                        or live_members != self.members
                    ):
                        raise OfficialPanelArchiveError(
                            "official archive directory differs from the pin"
                        )
                    payload = bundle.read(member)
            after = os.fstat(descriptor)
            if _descriptor_identity(after) != self.archive_identity:
                raise OfficialPanelArchiveError("official archive changed while reading")
        except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
            if isinstance(exc, OfficialPanelArchiveError):
                raise
            raise OfficialPanelArchiveError("cannot read authenticated panel") from exc
        finally:
            os.close(descriptor)
        size, crc = inventory[member]
        if (
            len(payload) != size
            or not 0 < len(payload) <= _MAX_PNG_BYTES
            or not payload.startswith(_PNG_SIGNATURE)
        ):
            raise OfficialPanelArchiveError("official panel is not a bounded PNG")
        receipt = OfficialPanelReceipt.seal(
            panel_id=panel_id,
            payload=payload,
            archive_member=member,
            zip_crc32=crc,
            release_descriptor_digest=self.release_descriptor_digest,
            archive_digest=self.archive_digest,
            central_directory_digest=self.central_directory_digest,
        )
        return payload, receipt

    def verify_panel(
        self,
        payload: bytes,
        receipt: OfficialPanelReceipt | Mapping[str, Any],
    ) -> OfficialPanelReceipt:
        archived = (
            receipt
            if isinstance(receipt, OfficialPanelReceipt)
            else OfficialPanelReceipt.from_data(receipt)
        )
        released, replay = self.read_panel(archived.panel_id)
        if released != payload or replay != archived:
            raise OfficialPanelArchiveError("panel cold replay differs")
        return archived


def _released_panel_content(value: "ReleasedOfficialPanel") -> dict[str, object]:
    return {
        "schema": RELEASED_PANEL_SCHEMA,
        "panel_id": value.panel_id,
        "exact_png_base64": base64.b64encode(value.exact_png_bytes).decode("ascii"),
        "exact_png_digest": value.exact_png_digest,
        "release_receipt": value.release_receipt.to_data(),
        "execution_precommit_digest": value.execution_precommit_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "released_after_durable_exposure": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ReleasedOfficialPanel:
    """Exact panel bytes whose two prerequisite commitments are explicit."""

    panel_id: str
    exact_png_bytes: bytes = field(repr=False)
    exact_png_digest: str
    release_receipt: OfficialPanelReceipt
    execution_precommit_digest: str
    exposure_successor_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.release_receipt) is not OfficialPanelReceipt:
            raise TypeError("release_receipt must be exact OfficialPanelReceipt")
        OfficialPanelReceipt.__post_init__(self.release_receipt)
        if _panel_member(self.panel_id) != self.release_receipt.archive_member:
            raise OfficialPanelArchiveError("released panel identity differs")
        for name in (
            "exact_png_digest",
            "execution_precommit_digest",
            "exposure_successor_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            type(self.exact_png_bytes) is not bytes
            or not self.exact_png_bytes.startswith(_PNG_SIGNATURE)
            or not 0 < len(self.exact_png_bytes) <= _MAX_PNG_BYTES
            or self.exact_png_digest
            != "sha256:" + hashlib.sha256(self.exact_png_bytes).hexdigest()
            or self.exact_png_digest != self.release_receipt.sha256
            or self.panel_id != self.release_receipt.panel_id
            or self.record_digest != _address(_released_panel_content(self))
        ):
            raise OfficialPanelArchiveError("released panel parents differ")

    @classmethod
    def release(
        cls,
        archive: OfficialPanelArchive,
        panel_id: str,
        *,
        execution_precommit_digest: str,
        exposure_successor_digest: str,
        expected_execution_precommit_digest: str,
        expected_exposure_successor_digest: str,
    ) -> "ReleasedOfficialPanel":
        precommit = _require_address(
            execution_precommit_digest, "execution precommit digest"
        )
        exposure = _require_address(
            exposure_successor_digest, "exposure successor digest"
        )
        if precommit != _require_address(
            expected_execution_precommit_digest,
            "expected execution precommit digest",
        ):
            raise OfficialPanelArchiveError(
                "execution precommit differs from external commitment"
            )
        if exposure != _require_address(
            expected_exposure_successor_digest,
            "expected exposure successor digest",
        ):
            raise OfficialPanelArchiveError(
                "exposure successor differs from durable commitment"
            )
        payload, receipt = archive.read_panel(panel_id)
        values: dict[str, object] = {
            "panel_id": panel_id,
            "exact_png_bytes": payload,
            "exact_png_digest": receipt.sha256,
            "release_receipt": receipt,
            "execution_precommit_digest": precommit,
            "exposure_successor_digest": exposure,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(_released_panel_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_released_panel_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: Mapping[str, Any]) -> "ReleasedOfficialPanel":
        expected = {
            "schema",
            "panel_id",
            "exact_png_base64",
            "exact_png_digest",
            "release_receipt",
            "execution_precommit_digest",
            "exposure_successor_digest",
            "released_after_durable_exposure",
            *_authority_data(),
            "record_digest",
        }
        _exact_fields(value, expected, "released official panel")
        if type(value["release_receipt"]) is not dict:
            raise OfficialPanelArchiveError("released panel receipt is malformed")
        try:
            payload = base64.b64decode(value["exact_png_base64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise OfficialPanelArchiveError("released panel PNG is malformed") from exc
        result = cls(
            panel_id=value["panel_id"],
            exact_png_bytes=payload,
            exact_png_digest=value["exact_png_digest"],
            release_receipt=OfficialPanelReceipt.from_data(value["release_receipt"]),
            execution_precommit_digest=value["execution_precommit_digest"],
            exposure_successor_digest=value["exposure_successor_digest"],
            record_digest=value["record_digest"],
        )
        if result.to_data() != dict(value):
            raise OfficialPanelArchiveError("released panel is not canonical")
        return result

    def cold_verify(
        self,
        archive: OfficialPanelArchive,
        *,
        expected_execution_precommit_digest: str,
        expected_exposure_successor_digest: str,
    ) -> None:
        if self.execution_precommit_digest != _require_address(
            expected_execution_precommit_digest,
            "expected execution precommit digest",
        ):
            raise OfficialPanelArchiveError(
                "released panel precommit differs from external commitment"
            )
        if self.exposure_successor_digest != _require_address(
            expected_exposure_successor_digest,
            "expected exposure successor digest",
        ):
            raise OfficialPanelArchiveError(
                "released panel exposure differs from durable commitment"
            )
        archive.verify_panel(self.exact_png_bytes, self.release_receipt)


__all__ = [
    "ARCHIVE_SCHEMA",
    "RECEIPT_SCHEMA",
    "RELEASED_PANEL_SCHEMA",
    "OfficialPanelArchive",
    "OfficialPanelArchiveError",
    "OfficialPanelReceipt",
    "ReleasedOfficialPanel",
]
