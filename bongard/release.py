"""Exact identity checks for the official ShapeBongard_V2 release.

Structural corpus validation answers "does this directory have the expected
shape?".  It cannot answer "are these the released bytes?".  This module keeps
those claims separate: a checked-in descriptor pins the archive, split file,
task inventory, and extracted corpus manifest by SHA-256.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .corpus import CorpusManifest, ShapeBongardCorpus


RELEASE_SCHEMA = "gkm.shape-bongard-official-release.v1"
DEFAULT_RELEASE_PATH = (
    Path(__file__).with_name("data") / "shape_bongard_v2_release_v1.json"
)
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class ReleaseIdentityError(RuntimeError):
    """The descriptor or candidate release bytes do not match exactly."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _address_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _line_address(values: tuple[str, ...]) -> str:
    return _address_bytes("".join(f"{value}\n" for value in values).encode("utf-8"))


def _stable_file_identity(path: Path) -> tuple[str, int]:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise ReleaseIdentityError(f"cannot hash release file {path}: {exc}") from exc
    if not path.is_file():
        raise ReleaseIdentityError(f"release path is not a regular file: {path}")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ReleaseIdentityError(f"release file changed while hashing: {path}")
    return "sha256:" + digest.hexdigest(), after.st_size


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ReleaseIdentityError(f"{label} must be a JSON object")
    return value


def _strict_counts(value: object, label: str) -> tuple[tuple[str, int], ...]:
    data = _object(value, label)
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in data.items()
    ):
        raise ReleaseIdentityError(f"{label} must map names to non-negative integers")
    return tuple(sorted(data.items()))


@dataclass(frozen=True, slots=True)
class OfficialReleaseDescriptor:
    release_id: str
    archive_filename: str
    archive_sha256: str
    archive_size_bytes: int
    split_filename: str
    split_sha256: str
    split_size_bytes: int
    upstream_repository: str
    upstream_commit: str
    family_counts: tuple[tuple[str, int], ...]
    primary_split_counts: tuple[tuple[str, int], ...]
    regime_counts: tuple[tuple[str, int], ...]
    task_ids_sha256: str
    corpus_manifest_sha256: str
    schema: str = RELEASE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RELEASE_SCHEMA:
            raise ReleaseIdentityError(f"unsupported release schema: {self.schema!r}")
        for label, value in (
            ("release_id", self.release_id),
            ("archive_filename", self.archive_filename),
            ("split_filename", self.split_filename),
            ("upstream_repository", self.upstream_repository),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ReleaseIdentityError(f"{label} must be a non-empty string")
        for label, value in (
            ("archive_sha256", self.archive_sha256),
            ("split_sha256", self.split_sha256),
            ("task_ids_sha256", self.task_ids_sha256),
            ("corpus_manifest_sha256", self.corpus_manifest_sha256),
        ):
            if _ADDRESS.fullmatch(value) is None:
                raise ReleaseIdentityError(f"{label} must be a sha256: content address")
        for label, value in (
            ("archive_size_bytes", self.archive_size_bytes),
            ("split_size_bytes", self.split_size_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ReleaseIdentityError(f"{label} must be a positive integer")
        if _COMMIT.fullmatch(self.upstream_commit) is None:
            raise ReleaseIdentityError("upstream_commit must be an exact 40-hex commit")
        for label, counts in (
            ("family_counts", self.family_counts),
            ("primary_split_counts", self.primary_split_counts),
            ("regime_counts", self.regime_counts),
        ):
            if tuple(sorted(counts)) != counts or len(dict(counts)) != len(counts):
                raise ReleaseIdentityError(f"{label} must be uniquely keyed and sorted")
            if any(
                not name
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for name, count in counts
            ):
                raise ReleaseIdentityError(f"{label} contains an invalid count")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "release_id": self.release_id,
            "archive": {
                "filename": self.archive_filename,
                "sha256": self.archive_sha256,
                "size_bytes": self.archive_size_bytes,
            },
            "split": {
                "filename": self.split_filename,
                "sha256": self.split_sha256,
                "size_bytes": self.split_size_bytes,
            },
            "upstream": {
                "repository": self.upstream_repository,
                "commit": self.upstream_commit,
            },
            "family_counts": dict(self.family_counts),
            "primary_split_counts": dict(self.primary_split_counts),
            "regime_counts": dict(self.regime_counts),
            "task_ids_sha256": self.task_ids_sha256,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
        }

    @property
    def digest(self) -> str:
        return _address_bytes(_canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialReleaseDescriptor":
        expected = {
            "schema",
            "release_id",
            "archive",
            "split",
            "upstream",
            "family_counts",
            "primary_split_counts",
            "regime_counts",
            "task_ids_sha256",
            "corpus_manifest_sha256",
        }
        if set(raw) != expected:
            raise ReleaseIdentityError(
                "release descriptor fields differ: "
                f"missing={sorted(expected - set(raw))}, extra={sorted(set(raw) - expected)}"
            )
        archive = _object(raw["archive"], "archive")
        split = _object(raw["split"], "split")
        upstream = _object(raw["upstream"], "upstream")
        if set(archive) != {"filename", "sha256", "size_bytes"}:
            raise ReleaseIdentityError("archive descriptor fields differ")
        if set(split) != {"filename", "sha256", "size_bytes"}:
            raise ReleaseIdentityError("split descriptor fields differ")
        if set(upstream) != {"repository", "commit"}:
            raise ReleaseIdentityError("upstream descriptor fields differ")
        try:
            return cls(
                schema=str(raw["schema"]),
                release_id=str(raw["release_id"]),
                archive_filename=str(archive["filename"]),
                archive_sha256=str(archive["sha256"]),
                archive_size_bytes=archive["size_bytes"],
                split_filename=str(split["filename"]),
                split_sha256=str(split["sha256"]),
                split_size_bytes=split["size_bytes"],
                upstream_repository=str(upstream["repository"]),
                upstream_commit=str(upstream["commit"]),
                family_counts=_strict_counts(raw["family_counts"], "family_counts"),
                primary_split_counts=_strict_counts(
                    raw["primary_split_counts"], "primary_split_counts"
                ),
                regime_counts=_strict_counts(raw["regime_counts"], "regime_counts"),
                task_ids_sha256=str(raw["task_ids_sha256"]),
                corpus_manifest_sha256=str(raw["corpus_manifest_sha256"]),
            )
        except (TypeError, ValueError) as exc:
            raise ReleaseIdentityError(f"invalid release descriptor: {exc}") from exc

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_RELEASE_PATH
    ) -> "OfficialReleaseDescriptor":
        source = Path(path)
        try:
            payload = source.read_bytes()
            raw = json.loads(payload)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReleaseIdentityError(f"cannot read release descriptor {source}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ReleaseIdentityError("release descriptor root must be an object")
        # Repository JSON is one canonical JSON value followed by the normal
        # POSIX text-file newline.  No alternate whitespace is accepted.
        if _canonical_json_bytes(raw) + b"\n" != payload:
            raise ReleaseIdentityError("release descriptor must use canonical JSON bytes")
        return cls.from_dict(raw)

    def verify_archive(self, path: str | Path) -> None:
        source = Path(path)
        digest, size = _stable_file_identity(source)
        if source.name != self.archive_filename:
            raise ReleaseIdentityError(
                f"archive filename is {source.name!r}, expected {self.archive_filename!r}"
            )
        if size != self.archive_size_bytes or digest != self.archive_sha256:
            raise ReleaseIdentityError(
                f"archive identity is ({size}, {digest}), expected "
                f"({self.archive_size_bytes}, {self.archive_sha256})"
            )

    def verify_split(self, path: str | Path) -> None:
        source = Path(path)
        digest, size = _stable_file_identity(source)
        if source.name != self.split_filename:
            raise ReleaseIdentityError(
                f"split filename is {source.name!r}, expected {self.split_filename!r}"
            )
        if size != self.split_size_bytes or digest != self.split_sha256:
            raise ReleaseIdentityError(
                f"split identity is ({size}, {digest}), expected "
                f"({self.split_size_bytes}, {self.split_sha256})"
            )

    def verify_corpus(
        self,
        corpus: ShapeBongardCorpus,
        *,
        manifest: CorpusManifest | None = None,
    ) -> CorpusManifest:
        """Prove that an extracted tree matches the pinned official bytes.

        This is intentionally expensive: it hashes every panel through a fresh
        canonical corpus manifest.  A supplied manifest is treated only as an
        additional equality assertion; it never replaces rebuilding from the
        directory passed as ``corpus``.
        """

        corpus.validate_complete(require_split=True)
        if corpus.split.source_path is None:
            raise ReleaseIdentityError("complete corpus has no split source path")
        self.verify_split(corpus.split.source_path)
        if dict(corpus.family_counts) != dict(self.family_counts):
            raise ReleaseIdentityError("corpus family counts differ from release descriptor")
        groups = corpus.split.canonical_groups
        actual_primary = {name: len(groups[name]) for name in ("train", "val", "test")}
        actual_regimes = {name: len(groups[name]) for name in ("FF", "BA", "CM", "NV")}
        if actual_primary != dict(self.primary_split_counts):
            raise ReleaseIdentityError("primary split counts differ from release descriptor")
        if actual_regimes != dict(self.regime_counts):
            raise ReleaseIdentityError("test regime counts differ from release descriptor")
        task_ids_digest = _line_address(tuple(sorted(corpus.task_ids)))
        if task_ids_digest != self.task_ids_sha256:
            raise ReleaseIdentityError("task inventory differs from release descriptor")
        rebuilt = corpus.build_manifest()
        if not isinstance(rebuilt, CorpusManifest):
            raise ReleaseIdentityError(
                "corpus builder returned an unexpected manifest representation"
            )
        if manifest is not None:
            if not isinstance(manifest, CorpusManifest):
                raise ReleaseIdentityError(
                    "supplied manifest has an unexpected representation"
                )
            if manifest.to_dict() != rebuilt.to_dict():
                raise ReleaseIdentityError(
                    "supplied corpus manifest differs from freshly rebuilt corpus bytes"
                )
        if rebuilt.digest != self.corpus_manifest_sha256:
            raise ReleaseIdentityError(
                f"corpus manifest is {rebuilt.digest}, expected {self.corpus_manifest_sha256}"
            )
        return rebuilt


def load_official_release(
    path: str | Path = DEFAULT_RELEASE_PATH,
) -> OfficialReleaseDescriptor:
    """Load the checked-in exact official release descriptor."""

    return OfficialReleaseDescriptor.load(path)
