"""Authenticated panel reads from a verified extracted ShapeBongard corpus.

The official release descriptor pins both the original ZIP and a canonical
manifest of the extracted tree.  Some installations retain only that exact
tree.  This module makes the latter an explicit release authority: loading an
archive freshly rebuilds and verifies the complete corpus manifest, and every
subsequent panel read is checked again against its manifest row.

The resulting records are deliberately distinct from the ZIP-backed records
in :mod:`bongard.official_panel_archive`.  They share the small attribute
surface needed by task runners (panel ID, exact bytes/digest, precommit and
exposure digests), but never claim ZIP-member or central-directory custody.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.corpus import CorpusManifest, PanelManifest, ShapeBongardCorpus, TaskManifest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.release import OfficialReleaseDescriptor


EXTRACTED_ARCHIVE_SCHEMA = "gkm.bongard-official-extracted-panel-archive.v1"
EXTRACTED_RECEIPT_SCHEMA = "gkm.bongard-official-extracted-panel-receipt.v1"
RELEASED_EXTRACTED_PANEL_SCHEMA = (
    "gkm.bongard-released-official-extracted-panel.v1"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(
    r"(?P<family>bd|ff|hd)/(?P<task>(?P=family)_[A-Za-z0-9_-]+)/"
    r"(?P<side>[01])/(?P<index>[0-6])\.png\Z"
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PNG_BYTES = 4_000_000


class OfficialExtractedPanelArchiveError(RuntimeError):
    """The verified corpus, a panel read, or a release record differed."""


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _require_address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise OfficialExtractedPanelArchiveError(
            f"{label} must be a sha256: address"
        )
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise OfficialExtractedPanelArchiveError(f"{label} fields differ")
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


def _panel_components(panel_id: str) -> tuple[str, str, str, str]:
    if type(panel_id) is not str:
        raise OfficialExtractedPanelArchiveError("panel ID must be an exact string")
    match = _PANEL_ID.fullmatch(panel_id)
    if match is None:
        raise OfficialExtractedPanelArchiveError(
            "panel ID is outside the official bd/ff/hd PNG namespace"
        )
    return (
        match.group("family"),
        match.group("task"),
        match.group("side"),
        match.group("index"),
    )


def _stable_panel_read(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.lstat(path)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OfficialExtractedPanelArchiveError(
            "cannot open extracted official panel safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            != identity
            or not 0 < opened.st_size <= _MAX_PNG_BYTES
        ):
            raise OfficialExtractedPanelArchiveError(
                "extracted official panel is not a stable bounded regular file"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_PNG_BYTES - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_PNG_BYTES:
                raise OfficialExtractedPanelArchiveError(
                    "extracted official panel exceeds the byte bound"
                )
        after = os.fstat(descriptor)
        if (
            total != opened.st_size
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != identity
        ):
            raise OfficialExtractedPanelArchiveError(
                "extracted official panel changed while reading"
            )
    finally:
        os.close(descriptor)
    try:
        final = os.lstat(path)
    except OSError as exc:
        raise OfficialExtractedPanelArchiveError(
            "extracted official panel changed after reading"
        ) from exc
    if (
        not stat.S_ISREG(final.st_mode)
        or (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        )
        != identity
    ):
        raise OfficialExtractedPanelArchiveError(
            "extracted official panel path changed while reading"
        )
    payload = b"".join(chunks)
    if not payload.startswith(_PNG_SIGNATURE):
        raise OfficialExtractedPanelArchiveError(
            "extracted official panel lacks the PNG signature"
        )
    return payload


def _task_digest(task: TaskManifest) -> str:
    return _address(task.content_dict())


def _validate_manifest(
    manifest: CorpusManifest,
    *,
    descriptor: OfficialReleaseDescriptor,
) -> tuple[dict[str, PanelManifest], dict[str, str]]:
    if type(manifest) is not CorpusManifest:
        raise TypeError("verified manifest must be exact CorpusManifest")
    if manifest.digest != _address(manifest.content_dict()):
        raise OfficialExtractedPanelArchiveError("corpus manifest self-digest differs")
    if manifest.digest != descriptor.corpus_manifest_sha256:
        raise OfficialExtractedPanelArchiveError(
            "corpus manifest differs from the official release descriptor"
        )
    if dict(manifest.family_counts) != dict(descriptor.family_counts):
        raise OfficialExtractedPanelArchiveError("corpus family counts differ")
    by_id: dict[str, PanelManifest] = {}
    task_digests: dict[str, str] = {}
    for task in manifest.tasks:
        if type(task) is not TaskManifest or task.digest != _task_digest(task):
            raise OfficialExtractedPanelArchiveError("task manifest digest differs")
        if task.task_id in task_digests:
            raise OfficialExtractedPanelArchiveError(
                "corpus manifest repeats a task ID"
            )
        task_digests[task.task_id] = task.digest
        for panel in task.panels:
            if type(panel) is not PanelManifest:
                raise OfficialExtractedPanelArchiveError(
                    "task manifest contains a non-panel row"
                )
            family, task_id, side, index = _panel_components(panel.panel_id)
            expected_polarity = "positive" if side == "1" else "negative"
            if (
                panel.task_id != task.task_id
                or panel.family != task.family
                or (family, task_id) != (panel.family, panel.task_id)
                or panel.polarity != expected_polarity
                or panel.index != int(index)
                or panel.filename != f"{index}.png"
                or not isinstance(panel.path, Path)
                or not panel.path.is_absolute()
                or type(panel.size_bytes) is not int
                or not 0 < panel.size_bytes <= _MAX_PNG_BYTES
            ):
                raise OfficialExtractedPanelArchiveError(
                    "panel manifest identity differs"
                )
            _require_address(panel.sha256, "panel manifest digest")
            if panel.panel_id in by_id:
                raise OfficialExtractedPanelArchiveError(
                    "corpus manifest repeats a panel ID"
                )
            by_id[panel.panel_id] = panel
    return by_id, task_digests


def _archive_content(value: "OfficialExtractedPanelArchive") -> dict[str, object]:
    return {
        "schema": EXTRACTED_ARCHIVE_SCHEMA,
        "release_descriptor_digest": value.release_descriptor_digest,
        "corpus_manifest_digest": value.corpus_manifest_digest,
        "layout": value.layout,
        "panel_count": len(value.panel_by_id),
        "read_policy": (
            "fresh-full-manifest-verification-then-stable-panel-reread"
        ),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class OfficialExtractedPanelArchive:
    """In-memory authority produced by one fresh full-corpus verification."""

    release_descriptor_digest: str
    corpus_manifest_digest: str
    layout: str
    corpus_root: Path = field(repr=False, compare=False)
    panel_by_id: Mapping[str, PanelManifest] = field(repr=False, compare=False)
    task_digest_by_task_id: Mapping[str, str] = field(repr=False, compare=False)
    record_digest: str

    def __post_init__(self) -> None:
        _require_address(self.release_descriptor_digest, "release descriptor digest")
        _require_address(self.corpus_manifest_digest, "corpus manifest digest")
        _require_address(self.record_digest, "extracted archive digest")
        if (
            self.layout not in {"archive", "generator"}
            or not isinstance(self.corpus_root, Path)
            or not self.corpus_root.is_absolute()
            or not isinstance(self.panel_by_id, Mapping)
            or not self.panel_by_id
            or tuple(self.panel_by_id) != tuple(sorted(self.panel_by_id))
            or not isinstance(self.task_digest_by_task_id, Mapping)
            or tuple(self.task_digest_by_task_id)
            != tuple(sorted(self.task_digest_by_task_id))
            or any(
                _ADDRESS.fullmatch(item) is None
                for item in self.task_digest_by_task_id.values()
            )
            or self.record_digest != _address(_archive_content(self))
        ):
            raise OfficialExtractedPanelArchiveError(
                "extracted archive identity differs"
            )

    @classmethod
    def load(
        cls,
        descriptor: OfficialReleaseDescriptor,
        corpus: ShapeBongardCorpus,
        *,
        expected_release_descriptor_digest: str,
    ) -> "OfficialExtractedPanelArchive":
        """Freshly hash and authenticate the complete extracted release."""

        if type(descriptor) is not OfficialReleaseDescriptor:
            raise TypeError("descriptor must be exact OfficialReleaseDescriptor")
        if type(corpus) is not ShapeBongardCorpus:
            raise TypeError("corpus must be exact ShapeBongardCorpus")
        expected = _require_address(
            expected_release_descriptor_digest, "expected release descriptor digest"
        )
        if descriptor.digest != expected:
            raise OfficialExtractedPanelArchiveError(
                "release descriptor differs from its external commitment"
            )
        manifest = descriptor.verify_corpus(corpus)
        return cls._from_verified_manifest(descriptor, corpus.root, manifest)

    @classmethod
    def _from_verified_manifest(
        cls,
        descriptor: OfficialReleaseDescriptor,
        corpus_root: Path,
        manifest: CorpusManifest,
    ) -> "OfficialExtractedPanelArchive":
        """Construct from a manifest already verified by ``descriptor``.

        This helper exists for focused tests. Production callers must use
        :meth:`load`, which performs the expensive full-tree verification.
        """

        if type(descriptor) is not OfficialReleaseDescriptor:
            raise TypeError("descriptor must be exact OfficialReleaseDescriptor")
        root = Path(corpus_root)
        if not root.is_absolute():
            raise OfficialExtractedPanelArchiveError(
                "extracted corpus root must be absolute"
            )
        by_id, task_digests = _validate_manifest(manifest, descriptor=descriptor)
        frozen = MappingProxyType({key: by_id[key] for key in sorted(by_id)})
        frozen_task_digests = MappingProxyType(
            {key: task_digests[key] for key in sorted(task_digests)}
        )
        values: dict[str, object] = {
            "release_descriptor_digest": descriptor.digest,
            "corpus_manifest_digest": manifest.digest,
            "layout": manifest.layout,
            "corpus_root": root,
            "panel_by_id": frozen,
            "task_digest_by_task_id": frozen_task_digests,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(_archive_content(provisional)),
        )

    def read_panel(
        self, panel_id: str
    ) -> tuple[bytes, "OfficialExtractedPanelReceipt"]:
        family, task_id, side, index = _panel_components(panel_id)
        try:
            row = self.panel_by_id[panel_id]
        except KeyError as exc:
            raise OfficialExtractedPanelArchiveError(
                "panel is absent from the verified corpus manifest"
            ) from exc
        component = "images" if self.layout == "archive" else "png"
        relative = f"{family}/{component}/{task_id}/{side}/{index}.png"
        expected_path = self.corpus_root / relative
        if row.path != expected_path:
            raise OfficialExtractedPanelArchiveError(
                "panel path differs from the verified corpus layout"
            )
        payload = _stable_panel_read(expected_path)
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if digest != row.sha256 or len(payload) != row.size_bytes:
            raise OfficialExtractedPanelArchiveError(
                "panel bytes differ from the verified corpus manifest"
            )
        receipt = OfficialExtractedPanelReceipt.seal(
            panel_id=panel_id,
            relative_path=relative,
            payload=payload,
            task_manifest_digest=self.task_digest_by_task_id[row.task_id],
            corpus_manifest_digest=self.corpus_manifest_digest,
            release_descriptor_digest=self.release_descriptor_digest,
            extracted_archive_digest=self.record_digest,
        )
        return payload, receipt

    def verify_panel(
        self,
        payload: bytes,
        receipt: "OfficialExtractedPanelReceipt" | Mapping[str, Any],
    ) -> "OfficialExtractedPanelReceipt":
        archived = (
            receipt
            if type(receipt) is OfficialExtractedPanelReceipt
            else OfficialExtractedPanelReceipt.from_data(receipt)
        )
        released, replay = self.read_panel(archived.panel_id)
        if released != payload or replay != archived:
            raise OfficialExtractedPanelArchiveError("panel cold replay differs")
        return archived


def _receipt_content(value: "OfficialExtractedPanelReceipt") -> dict[str, object]:
    return {
        "schema": EXTRACTED_RECEIPT_SCHEMA,
        "panel_id": value.panel_id,
        "relative_path": value.relative_path,
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "task_manifest_digest": value.task_manifest_digest,
        "corpus_manifest_digest": value.corpus_manifest_digest,
        "release_descriptor_digest": value.release_descriptor_digest,
        "extracted_archive_digest": value.extracted_archive_digest,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class OfficialExtractedPanelReceipt:
    panel_id: str
    relative_path: str
    sha256: str
    size_bytes: int
    task_manifest_digest: str
    corpus_manifest_digest: str
    release_descriptor_digest: str
    extracted_archive_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        family, task, side, index = _panel_components(self.panel_id)
        if self.relative_path not in {
            f"{family}/images/{task}/{side}/{index}.png",
            f"{family}/png/{task}/{side}/{index}.png",
        }:
            raise OfficialExtractedPanelArchiveError(
                "extracted receipt path differs from panel ID"
            )
        for name in (
            "sha256",
            "task_manifest_digest",
            "corpus_manifest_digest",
            "release_descriptor_digest",
            "extracted_archive_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            type(self.size_bytes) is not int
            or not 0 < self.size_bytes <= _MAX_PNG_BYTES
            or self.record_digest != _address(_receipt_content(self))
        ):
            raise OfficialExtractedPanelArchiveError(
                "extracted panel receipt identity differs"
            )

    @classmethod
    def seal(
        cls,
        *,
        panel_id: str,
        relative_path: str,
        payload: bytes,
        task_manifest_digest: str,
        corpus_manifest_digest: str,
        release_descriptor_digest: str,
        extracted_archive_digest: str,
    ) -> "OfficialExtractedPanelReceipt":
        values: dict[str, object] = {
            "panel_id": panel_id,
            "relative_path": relative_path,
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "task_manifest_digest": task_manifest_digest,
            "corpus_manifest_digest": corpus_manifest_digest,
            "release_descriptor_digest": release_descriptor_digest,
            "extracted_archive_digest": extracted_archive_digest,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,  # type: ignore[arg-type]
            record_digest=_address(_receipt_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_receipt_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "OfficialExtractedPanelReceipt":
        raw = _fields(
            value,
            {
                "schema",
                "panel_id",
                "relative_path",
                "sha256",
                "size_bytes",
                "task_manifest_digest",
                "corpus_manifest_digest",
                "release_descriptor_digest",
                "extracted_archive_digest",
                *_authority_data(),
                "record_digest",
            },
            "official extracted panel receipt",
        )
        if (
            raw["schema"] != EXTRACTED_RECEIPT_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise OfficialExtractedPanelArchiveError(
                "extracted panel receipt policy differs"
            )
        result = cls(
            raw["panel_id"],
            raw["relative_path"],
            raw["sha256"],
            raw["size_bytes"],
            raw["task_manifest_digest"],
            raw["corpus_manifest_digest"],
            raw["release_descriptor_digest"],
            raw["extracted_archive_digest"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise OfficialExtractedPanelArchiveError(
                "extracted panel receipt is not canonical"
            )
        return result


def _released_content(
    value: "ReleasedOfficialExtractedPanel",
) -> dict[str, object]:
    return {
        "schema": RELEASED_EXTRACTED_PANEL_SCHEMA,
        "panel_id": value.panel_id,
        "exact_png_base64": base64.b64encode(value.exact_png_bytes).decode("ascii"),
        "exact_png_digest": value.exact_png_digest,
        "release_receipt": value.release_receipt.to_data(),
        "execution_precommit_digest": value.execution_precommit_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "released_after_durable_exposure": True,
        "source_authority": "verified-official-extracted-corpus-manifest",
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ReleasedOfficialExtractedPanel:
    """Exact panel bytes released from the authenticated extracted tree."""

    panel_id: str
    exact_png_bytes: bytes = field(repr=False)
    exact_png_digest: str
    release_receipt: OfficialExtractedPanelReceipt
    execution_precommit_digest: str
    exposure_successor_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        _panel_components(self.panel_id)
        for name in (
            "exact_png_digest",
            "execution_precommit_digest",
            "exposure_successor_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            type(self.release_receipt) is not OfficialExtractedPanelReceipt
            or type(self.exact_png_bytes) is not bytes
            or not self.exact_png_bytes.startswith(_PNG_SIGNATURE)
            or not 0 < len(self.exact_png_bytes) <= _MAX_PNG_BYTES
            or self.panel_id != self.release_receipt.panel_id
            or self.exact_png_digest != self.release_receipt.sha256
            or self.exact_png_digest
            != "sha256:" + hashlib.sha256(self.exact_png_bytes).hexdigest()
            or self.record_digest != _address(_released_content(self))
        ):
            raise OfficialExtractedPanelArchiveError(
                "released extracted panel parents differ"
            )

    @classmethod
    def release(
        cls,
        archive: OfficialExtractedPanelArchive,
        panel_id: str,
        *,
        execution_precommit_digest: str,
        exposure_successor_digest: str,
        expected_execution_precommit_digest: str,
        expected_exposure_successor_digest: str,
    ) -> "ReleasedOfficialExtractedPanel":
        if type(archive) is not OfficialExtractedPanelArchive:
            raise TypeError("archive must be exact OfficialExtractedPanelArchive")
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
            raise OfficialExtractedPanelArchiveError(
                "execution precommit differs from its external commitment"
            )
        if exposure != _require_address(
            expected_exposure_successor_digest,
            "expected exposure successor digest",
        ):
            raise OfficialExtractedPanelArchiveError(
                "exposure successor differs from its durable commitment"
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
            record_digest=_address(_released_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_released_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, value: object) -> "ReleasedOfficialExtractedPanel":
        raw = _fields(
            value,
            {
                "schema",
                "panel_id",
                "exact_png_base64",
                "exact_png_digest",
                "release_receipt",
                "execution_precommit_digest",
                "exposure_successor_digest",
                "released_after_durable_exposure",
                "source_authority",
                *_authority_data(),
                "record_digest",
            },
            "released official extracted panel",
        )
        if (
            raw["schema"] != RELEASED_EXTRACTED_PANEL_SCHEMA
            or raw["released_after_durable_exposure"] is not True
            or raw["source_authority"]
            != "verified-official-extracted-corpus-manifest"
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise OfficialExtractedPanelArchiveError(
                "released extracted panel policy differs"
            )
        try:
            payload = base64.b64decode(raw["exact_png_base64"], validate=True)
        except (TypeError, ValueError) as exc:
            raise OfficialExtractedPanelArchiveError(
                "released extracted panel PNG is malformed"
            ) from exc
        result = cls(
            raw["panel_id"],
            payload,
            raw["exact_png_digest"],
            OfficialExtractedPanelReceipt.from_data(raw["release_receipt"]),
            raw["execution_precommit_digest"],
            raw["exposure_successor_digest"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise OfficialExtractedPanelArchiveError(
                "released extracted panel is not canonical"
            )
        return result

    def cold_verify(
        self,
        archive: OfficialExtractedPanelArchive,
        *,
        expected_execution_precommit_digest: str,
        expected_exposure_successor_digest: str,
    ) -> None:
        if self.execution_precommit_digest != _require_address(
            expected_execution_precommit_digest,
            "expected execution precommit digest",
        ):
            raise OfficialExtractedPanelArchiveError(
                "released panel precommit differs from external commitment"
            )
        if self.exposure_successor_digest != _require_address(
            expected_exposure_successor_digest,
            "expected exposure successor digest",
        ):
            raise OfficialExtractedPanelArchiveError(
                "released panel exposure differs from durable commitment"
            )
        archive.verify_panel(self.exact_png_bytes, self.release_receipt)


__all__ = (
    "EXTRACTED_ARCHIVE_SCHEMA",
    "EXTRACTED_RECEIPT_SCHEMA",
    "RELEASED_EXTRACTED_PANEL_SCHEMA",
    "OfficialExtractedPanelArchive",
    "OfficialExtractedPanelArchiveError",
    "OfficialExtractedPanelReceipt",
    "ReleasedOfficialExtractedPanel",
)
