"""Deterministic, fail-closed image audit for ShapeBongard corpora.

The corpus inventory proves that fourteen ``.png`` paths exist per task.  This
module establishes the stronger image boundary needed by a benchmark:

* every source is a non-symlink regular file containing exactly one PNG
  container (no trailing bytes);
* the bytes are read from the corpus once into a bounded-memory spool, hashed,
  verified by Pillow, and every frame is decoded from that frozen snapshot;
* a supplied :class:`~bongard.corpus.CorpusManifest` is checked recursively and
  its already-recorded panel hashes are compared during that same source read;
* file identities and timestamps are checked before, during, and after the
  complete audit so ordinary concurrent replacement or mutation fails closed.

Image mode, dimensions, metadata keys, and frame count are observations, not
guesses baked into this module.  A caller may perform an exploratory pass,
inspect the distributions, and only then pass explicit
:class:`ImageExpectations` for a strict confirmation pass.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping, Sequence
import warnings

from PIL import Image

from bongard.corpus import (
    BongardTask,
    CorpusManifest,
    PanelManifest,
    ShapeBongardCorpus,
    TaskManifest,
)


AUDIT_SCHEMA = "gkm.shape-bongard-image-audit.v1"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SHA256_PREFIXED = "sha256:"
_READ_CHUNK_BYTES = 1024 * 1024


class ImageAuditError(RuntimeError):
    """A panel, manifest, or filesystem invariant failed during the audit."""


class ImageExpectationError(ImageAuditError):
    """Observed image properties violated explicitly supplied expectations."""

    def __init__(self, report: "ImageAuditReport") -> None:
        self.report = report
        super().__init__(
            f"{report.anomaly_count} image property mismatches; "
            "the complete report is available as exception.report"
        )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ImageAuditError(f"audit value is not canonical JSON: {exc}") from exc


def _address(value: object) -> str:
    return _SHA256_PREFIXED + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_address(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith(
        _SHA256_PREFIXED
    ):
        raise ImageAuditError(f"{label} is not a canonical sha256 address")
    suffix = value.removeprefix(_SHA256_PREFIXED)
    if any(character not in "0123456789abcdef" for character in suffix):
        raise ImageAuditError(f"{label} is not a canonical sha256 address")
    return value


def _feed(hasher: Any, value: object) -> None:
    """Unambiguously append one canonical record to a streaming summary."""

    payload = _canonical_json_bytes(value)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


@dataclass(frozen=True)
class ImageExpectations:
    """Properties observed elsewhere and explicitly required by the caller."""

    mode: str
    width: int
    height: int
    info_keys: tuple[str, ...]
    frame_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("expected image mode must be non-empty")
        for label, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"expected image {label} must be positive")
        if (
            not isinstance(self.info_keys, tuple)
            or any(not isinstance(key, str) or not key for key in self.info_keys)
            or tuple(sorted(set(self.info_keys))) != self.info_keys
        ):
            raise ValueError("expected info_keys must be unique non-empty strings in order")
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count <= 0
        ):
            raise ValueError("expected frame_count must be positive")

    def to_data(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "width": self.width,
            "height": self.height,
            "info_keys": list(self.info_keys),
            "frame_count": self.frame_count,
        }


@dataclass(frozen=True)
class ImageAnomaly:
    panel_id: str
    property_name: str
    expected: object
    observed: object

    def to_data(self) -> dict[str, object]:
        return {
            "panel_id": self.panel_id,
            "property": self.property_name,
            "expected": self.expected,
            "observed": self.observed,
        }


@dataclass(frozen=True)
class ImageAuditReport:
    """Compact data-only result; no source paths or decoded pixels are retained."""

    task_count: int
    panel_count: int
    byte_count_total: int
    family_task_counts: tuple[tuple[str, int], ...]
    family_panel_counts: tuple[tuple[str, int], ...]
    format_counts: tuple[tuple[str, int], ...]
    mode_counts: tuple[tuple[str, int], ...]
    size_counts: tuple[tuple[int, int, int], ...]
    info_key_set_counts: tuple[tuple[tuple[str, ...], int], ...]
    frame_count_counts: tuple[tuple[int, int], ...]
    content_summary_digest: str
    property_summary_digest: str
    corpus_manifest_digest: str | None
    expectations: ImageExpectations | None
    require_expected_properties: bool
    anomaly_count: int
    anomalies: tuple[ImageAnomaly, ...]
    anomalies_truncated: bool
    digest: str
    schema: str = AUDIT_SCHEMA

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "task_count": self.task_count,
            "panel_count": self.panel_count,
            "byte_count_total": self.byte_count_total,
            "family_task_counts": dict(self.family_task_counts),
            "family_panel_counts": dict(self.family_panel_counts),
            "format_counts": dict(self.format_counts),
            "mode_counts": dict(self.mode_counts),
            "size_counts": [
                {"width": width, "height": height, "count": count}
                for width, height, count in self.size_counts
            ],
            "info_key_set_counts": [
                {"info_keys": list(keys), "count": count}
                for keys, count in self.info_key_set_counts
            ],
            "frame_count_counts": [
                {"frame_count": frame_count, "count": count}
                for frame_count, count in self.frame_count_counts
            ],
            "content_summary_digest": self.content_summary_digest,
            "property_summary_digest": self.property_summary_digest,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "expectations": self.expectations.to_data() if self.expectations else None,
            "require_expected_properties": self.require_expected_properties,
            "anomaly_count": self.anomaly_count,
            "anomalies": [anomaly.to_data() for anomaly in self.anomalies],
            "anomalies_truncated": self.anomalies_truncated,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_dict(), "digest": self.digest}


@dataclass(frozen=True)
class _PanelSpec:
    panel_id: str
    task_id: str
    family: str
    polarity: str
    index: int
    filename: str
    path: Path
    manifest_panel: PanelManifest | None


@dataclass(frozen=True)
class _DecodedProperties:
    format: str
    mode: str
    width: int
    height: int
    info_keys: tuple[str, ...]
    frame_count: int
    frame_properties: tuple[tuple[str, int, int, tuple[str, ...]], ...]


def _lexical_absolute(path: Path) -> Path:
    """Make a path absolute without following a final symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _file_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _lstat_regular(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ImageAuditError(f"cannot stat panel {path}: {exc}") from exc
    if stat.S_ISLNK(value.st_mode):
        raise ImageAuditError(f"panel path is a symlink: {path}")
    if not stat.S_ISREG(value.st_mode):
        raise ImageAuditError(f"panel path is not a regular file: {path}")
    return value


def _lstat_directory(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ImageAuditError(f"cannot stat corpus directory {path}: {exc}") from exc
    if stat.S_ISLNK(value.st_mode):
        raise ImageAuditError(f"corpus directory is a symlink: {path}")
    if not stat.S_ISDIR(value.st_mode):
        raise ImageAuditError(f"corpus path is not a directory: {path}")
    return value


def _validate_manifest(
    corpus: ShapeBongardCorpus, manifest: CorpusManifest | None
) -> tuple[TaskManifest | None, ...]:
    tasks = tuple(sorted(corpus.tasks, key=lambda task: (task.task_id, task.family)))
    if manifest is None:
        return (None,) * len(tasks)

    if _address(manifest.content_dict()) != _validate_address(
        manifest.digest, "corpus manifest digest"
    ):
        raise ImageAuditError("corpus manifest content does not match its digest")
    if manifest.layout != corpus.layout:
        raise ImageAuditError("corpus manifest layout differs from the audited corpus")
    if dict(manifest.family_counts) != dict(corpus.family_counts):
        raise ImageAuditError("corpus manifest family counts differ from the audited corpus")
    if manifest.split.to_manifest_dict() != corpus.split.to_manifest_dict():
        raise ImageAuditError("corpus manifest split differs from the audited corpus")
    if len(manifest.tasks) != len(tasks):
        raise ImageAuditError("corpus manifest task count differs from the audited corpus")

    expected_tasks: list[TaskManifest] = []
    for task, task_manifest in zip(tasks, manifest.tasks, strict=True):
        if (task_manifest.task_id, task_manifest.family) != (task.task_id, task.family):
            raise ImageAuditError("corpus manifest task order or identity is non-canonical")
        if _address(task_manifest.content_dict()) != _validate_address(
            task_manifest.digest, f"task manifest digest for {task.task_id}"
        ):
            raise ImageAuditError(
                f"task manifest content does not match its digest: {task.task_id}"
            )
        if len(task_manifest.panels) != len(task.panels):
            raise ImageAuditError(f"manifest panel count differs for {task.task_id}")
        expected_tasks.append(task_manifest)
    return tuple(expected_tasks)


def _png_directory_entries(directory: Path) -> tuple[Path, ...]:
    _lstat_directory(directory)
    try:
        entries = tuple(sorted(directory.iterdir(), key=lambda path: path.name))
    except OSError as exc:
        raise ImageAuditError(f"cannot enumerate corpus directory {directory}: {exc}") from exc

    pngs: list[Path] = []
    for entry in entries:
        if entry.suffix.lower() != ".png":
            continue
        # Check with lstat rather than is_file(): is_file() follows symlinks.
        _lstat_regular(entry)
        pngs.append(_lexical_absolute(entry))
    return tuple(pngs)


def _task_specs(
    task: BongardTask, task_manifest: TaskManifest | None
) -> tuple[_PanelSpec, ...]:
    root = _lexical_absolute(task.root)
    _lstat_directory(root)
    specs: list[_PanelSpec] = []
    manifest_panels: Sequence[PanelManifest | None]
    if task_manifest is None:
        manifest_panels = (None,) * len(task.panels)
    else:
        manifest_panels = task_manifest.panels
    manifest_index = 0

    for polarity, label, paths in (
        ("positive", "1", task.positive),
        ("negative", "0", task.negative),
    ):
        directory = root / label
        actual_pngs = _png_directory_entries(directory)
        supplied = tuple(_lexical_absolute(Path(path)) for path in paths)
        canonical = tuple(directory / path.name for path in supplied)
        if supplied != canonical:
            raise ImageAuditError(
                f"{task.task_id}/{label}: corpus panel path escapes its label directory"
            )
        if tuple(path.name for path in actual_pngs) != tuple(path.name for path in supplied):
            raise ImageAuditError(
                f"{task.task_id}/{label}: live PNG inventory differs from the corpus inventory"
            )
        if len(set(path.name for path in supplied)) != len(supplied):
            raise ImageAuditError(f"{task.task_id}/{label}: duplicate panel filename")

        for index, path in enumerate(supplied):
            panel_id = f"{task.family}/{task.task_id}/{label}/{path.name}"
            manifest_panel = manifest_panels[manifest_index]
            manifest_index += 1
            if manifest_panel is not None:
                expected_fields = (
                    panel_id,
                    task.task_id,
                    task.family,
                    polarity,
                    index,
                    path.name,
                )
                observed_fields = (
                    manifest_panel.panel_id,
                    manifest_panel.task_id,
                    manifest_panel.family,
                    manifest_panel.polarity,
                    manifest_panel.index,
                    manifest_panel.filename,
                )
                if observed_fields != expected_fields:
                    raise ImageAuditError(f"manifest panel identity differs for {panel_id}")
                _validate_address(manifest_panel.sha256, f"panel digest for {panel_id}")
                if (
                    isinstance(manifest_panel.size_bytes, bool)
                    or not isinstance(manifest_panel.size_bytes, int)
                    or manifest_panel.size_bytes <= 0
                ):
                    raise ImageAuditError(f"invalid manifest byte count for {panel_id}")
            specs.append(
                _PanelSpec(
                    panel_id=panel_id,
                    task_id=task.task_id,
                    family=task.family,
                    polarity=polarity,
                    index=index,
                    filename=path.name,
                    path=path,
                    manifest_panel=manifest_panel,
                )
            )
    return tuple(specs)


def _iter_specs(
    corpus: ShapeBongardCorpus,
    manifest_tasks: Sequence[TaskManifest | None],
) -> Iterator[_PanelSpec]:
    tasks = tuple(sorted(corpus.tasks, key=lambda task: (task.task_id, task.family)))
    if len(tasks) != len(manifest_tasks):
        raise ImageAuditError("internal task/manifest alignment failed")
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for task, task_manifest in zip(tasks, manifest_tasks, strict=True):
        corpus_root = _lexical_absolute(corpus.root)
        _lstat_directory(corpus_root)
        family_root = corpus_root / task.family
        _lstat_directory(family_root)
        candidates: list[Path] = []
        for component in ("images", "png"):
            component_root = family_root / component
            try:
                component_stat = component_root.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ImageAuditError(
                    f"cannot stat corpus directory {component_root}: {exc}"
                ) from exc
            if stat.S_ISLNK(component_stat.st_mode):
                raise ImageAuditError(f"corpus directory is a symlink: {component_root}")
            if not stat.S_ISDIR(component_stat.st_mode):
                raise ImageAuditError(f"corpus path is not a directory: {component_root}")
            candidate = component_root / task.task_id
            try:
                candidate.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ImageAuditError(f"cannot stat task path {candidate}: {exc}") from exc
            _lstat_directory(candidate)
            candidates.append(candidate)
        if len(candidates) != 1:
            raise ImageAuditError(
                f"task {task.task_id} has {len(candidates)} live archive/generator roots"
            )
        if _lexical_absolute(task.root) != candidates[0]:
            raise ImageAuditError(
                f"task {task.task_id} resolves outside its canonical corpus location"
            )
        for spec in _task_specs(task, task_manifest):
            if spec.panel_id in seen_ids:
                raise ImageAuditError(f"duplicate panel id: {spec.panel_id}")
            if spec.path in seen_paths:
                raise ImageAuditError(f"one source path is reused by multiple panels: {spec.path}")
            seen_ids.add(spec.panel_id)
            seen_paths.add(spec.path)
            yield spec


def _validate_png_container(snapshot: BinaryIO) -> None:
    """Reject malformed framing and data after the mandatory final IEND."""

    snapshot.seek(0)
    if snapshot.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
        raise ImageAuditError("source bytes do not begin with the PNG signature")
    saw_ihdr = False
    while True:
        header = snapshot.read(8)
        if len(header) != 8:
            raise ImageAuditError("PNG ended before a complete IEND chunk")
        length = int.from_bytes(header[:4], "big")
        chunk_type = header[4:]
        if any(not (65 <= byte <= 90 or 97 <= byte <= 122) for byte in chunk_type):
            raise ImageAuditError("PNG contains an invalid chunk type")
        if not saw_ihdr:
            if chunk_type != b"IHDR" or length != 13:
                raise ImageAuditError("PNG does not begin with a 13-byte IHDR chunk")
            saw_ihdr = True
        try:
            snapshot.seek(length, os.SEEK_CUR)
        except (OSError, ValueError) as exc:
            raise ImageAuditError(f"cannot traverse PNG chunks: {exc}") from exc
        if len(snapshot.read(4)) != 4:
            raise ImageAuditError("PNG chunk is truncated before its CRC")
        if chunk_type == b"IEND":
            if length != 0:
                raise ImageAuditError("PNG IEND chunk must be empty")
            if snapshot.read(1):
                raise ImageAuditError("PNG contains trailing bytes after IEND")
            return


def _info_keys(image: Image.Image) -> tuple[str, ...]:
    keys = tuple(image.info.keys())
    if any(not isinstance(key, str) or not key for key in keys):
        raise ImageAuditError("Pillow returned a non-string or empty image info key")
    return tuple(sorted(set(keys)))


def _check_dimensions(width: int, height: int, max_pixels: int) -> None:
    if width <= 0 or height <= 0:
        raise ImageAuditError(f"image dimensions must be positive, observed {width}x{height}")
    if width * height > max_pixels:
        raise ImageAuditError(
            f"image dimensions {width}x{height} exceed the {max_pixels}-pixel safety limit"
        )


def _decode_png_snapshot(
    snapshot: BinaryIO, *, max_pixels: int, max_frames: int
) -> _DecodedProperties:
    """Run Pillow verification and decode every frame from the frozen bytes."""

    _validate_png_container(snapshot)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            snapshot.seek(0)
            with Image.open(snapshot) as image:
                if image.format != "PNG":
                    raise ImageAuditError(
                        f"Pillow identified source as {image.format!r}, expected 'PNG'"
                    )
                width, height = image.size
                _check_dimensions(width, height, max_pixels)
                mode = image.mode
                info_keys = _info_keys(image)
                frame_count = getattr(image, "n_frames", 1)
                if (
                    isinstance(frame_count, bool)
                    or not isinstance(frame_count, int)
                    or frame_count <= 0
                    or frame_count > max_frames
                ):
                    raise ImageAuditError(
                        f"image frame count {frame_count!r} is outside 1..{max_frames}"
                    )
                image.verify()

            snapshot.seek(0)
            with Image.open(snapshot) as image:
                if image.format != "PNG":
                    raise ImageAuditError("image format changed between verify and load")
                if (image.mode, image.size, _info_keys(image)) != (
                    mode,
                    (width, height),
                    info_keys,
                ):
                    raise ImageAuditError("image properties changed between verify and load")
                loaded_frames = getattr(image, "n_frames", 1)
                if loaded_frames != frame_count:
                    raise ImageAuditError("frame count changed between verify and load")
                frame_properties: list[tuple[str, int, int, tuple[str, ...]]] = []
                for frame_index in range(frame_count):
                    image.seek(frame_index)
                    frame_width, frame_height = image.size
                    _check_dimensions(frame_width, frame_height, max_pixels)
                    image.load()
                    frame_properties.append(
                        (
                            image.mode,
                            frame_width,
                            frame_height,
                            _info_keys(image),
                        )
                    )
    except ImageAuditError:
        raise
    except (Exception, Warning) as exc:
        # Pillow uses several exception classes (including OSError) for damaged
        # streams.  At this trust boundary every decode failure has one result.
        raise ImageAuditError(f"Pillow could not verify and load PNG: {exc}") from exc

    return _DecodedProperties(
        format="PNG",
        mode=mode,
        width=width,
        height=height,
        info_keys=info_keys,
        frame_count=frame_count,
        frame_properties=tuple(frame_properties),
    )


def _audit_panel(
    spec: _PanelSpec,
    *,
    max_panel_bytes: int,
    max_pixels: int,
    max_frames: int,
    spool_memory_limit: int,
) -> tuple[str, int, _DecodedProperties, tuple[int, ...]]:
    before_path = _lstat_regular(spec.path)
    if before_path.st_size <= 0:
        raise ImageAuditError(f"panel is empty: {spec.panel_id}")
    if before_path.st_size > max_panel_bytes:
        raise ImageAuditError(
            f"panel {spec.panel_id} exceeds the {max_panel_bytes}-byte safety limit"
        )

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(spec.path, flags)
    except OSError as exc:
        raise ImageAuditError(f"cannot open panel {spec.panel_id}: {exc}") from exc

    digest = hashlib.sha256()
    byte_count = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as source, tempfile.SpooledTemporaryFile(
            max_size=spool_memory_limit, mode="w+b"
        ) as snapshot:
            opened = os.fstat(source.fileno())
            if _file_fingerprint(opened) != _file_fingerprint(before_path):
                raise ImageAuditError(f"panel path changed while opening: {spec.panel_id}")
            while True:
                chunk = source.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > max_panel_bytes:
                    raise ImageAuditError(
                        f"panel {spec.panel_id} exceeds the byte safety limit while reading"
                    )
                digest.update(chunk)
                snapshot.write(chunk)
            after_read = os.fstat(source.fileno())
            if _file_fingerprint(after_read) != _file_fingerprint(opened):
                raise ImageAuditError(f"panel changed while reading: {spec.panel_id}")
            if byte_count != opened.st_size:
                raise ImageAuditError(f"panel byte count changed while reading: {spec.panel_id}")

            raw_digest = _SHA256_PREFIXED + digest.hexdigest()
            if spec.manifest_panel is not None:
                if byte_count != spec.manifest_panel.size_bytes:
                    raise ImageAuditError(
                        f"panel byte count differs from manifest: {spec.panel_id}"
                    )
                if raw_digest != spec.manifest_panel.sha256:
                    raise ImageAuditError(
                        f"panel content differs from manifest: {spec.panel_id}"
                    )
            snapshot.flush()
            properties = _decode_png_snapshot(
                snapshot, max_pixels=max_pixels, max_frames=max_frames
            )
            after_decode = os.fstat(source.fileno())
            if _file_fingerprint(after_decode) != _file_fingerprint(opened):
                raise ImageAuditError(f"panel changed during decode: {spec.panel_id}")

        after_path = _lstat_regular(spec.path)
        if _file_fingerprint(after_path) != _file_fingerprint(before_path):
            raise ImageAuditError(f"panel path changed during audit: {spec.panel_id}")
    except Exception:
        # os.fdopen owns and closes descriptor after it succeeds.  If it failed
        # before taking ownership, close the descriptor without masking cause.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise

    return raw_digest, byte_count, properties, _file_fingerprint(after_path)


def _current_source_state(
    corpus: ShapeBongardCorpus,
    manifest_tasks: Sequence[TaskManifest | None],
) -> str:
    hasher = hashlib.sha256()
    for spec in _iter_specs(corpus, manifest_tasks):
        value = _lstat_regular(spec.path)
        _feed(hasher, {"panel_id": spec.panel_id, "stat": _file_fingerprint(value)})
    return hasher.hexdigest()


def _sorted_counter(counter: Mapping[Any, int]) -> tuple[tuple[Any, int], ...]:
    return tuple(sorted(counter.items(), key=lambda item: item[0]))


def audit_corpus_images(
    corpus: ShapeBongardCorpus,
    *,
    corpus_manifest: CorpusManifest | None = None,
    expected_properties: ImageExpectations | None = None,
    require_expected_properties: bool = False,
    max_anomalies: int = 1_000,
    max_panel_bytes: int = 64 * 1024 * 1024,
    max_pixels: int = 16_777_216,
    max_frames: int = 128,
    spool_memory_limit: int = 1024 * 1024,
) -> ImageAuditReport:
    """Audit every inventoried panel and return a canonical compact report.

    ``expected_properties`` only compares observations and records anomalies.
    Set ``require_expected_properties=True`` to fail after the complete pass if
    any panel differs.  Strict mode deliberately requires an explicit
    ``ImageExpectations`` object; this module never guesses an official mode,
    size, metadata set, or frame count.

    Source bytes are read exactly once per panel.  Pillow operates on the
    bounded-memory (and disk-spilling) snapshot, so supplying a manifest does
    not trigger a second source hash or a hidden call to ``build_manifest``.
    """

    if require_expected_properties and expected_properties is None:
        raise ValueError(
            "strict property validation requires explicit expected_properties"
        )
    for label, value in (
        ("max_anomalies", max_anomalies),
        ("max_panel_bytes", max_panel_bytes),
        ("max_pixels", max_pixels),
        ("max_frames", max_frames),
        ("spool_memory_limit", spool_memory_limit),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")

    manifest_tasks = _validate_manifest(corpus, corpus_manifest)
    tasks = tuple(sorted(corpus.tasks, key=lambda task: (task.task_id, task.family)))
    family_task_counts: Counter[str] = Counter(task.family for task in tasks)
    family_panel_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    size_counts: Counter[tuple[int, int]] = Counter()
    info_key_counts: Counter[tuple[str, ...]] = Counter()
    frame_counts: Counter[int] = Counter()
    content_hasher = hashlib.sha256()
    property_hasher = hashlib.sha256()
    initial_state_hasher = hashlib.sha256()
    byte_count_total = 0
    panel_count = 0
    anomaly_count = 0
    anomalies: list[ImageAnomaly] = []

    def anomaly(spec: _PanelSpec, field: str, expected: object, observed: object) -> None:
        nonlocal anomaly_count
        anomaly_count += 1
        if len(anomalies) < max_anomalies:
            anomalies.append(ImageAnomaly(spec.panel_id, field, expected, observed))

    for spec in _iter_specs(corpus, manifest_tasks):
        raw_digest, byte_count, properties, fingerprint = _audit_panel(
            spec,
            max_panel_bytes=max_panel_bytes,
            max_pixels=max_pixels,
            max_frames=max_frames,
            spool_memory_limit=spool_memory_limit,
        )
        panel_count += 1
        byte_count_total += byte_count
        family_panel_counts[spec.family] += 1
        format_counts[properties.format] += 1
        mode_counts[properties.mode] += 1
        size_counts[(properties.width, properties.height)] += 1
        info_key_counts[properties.info_keys] += 1
        frame_counts[properties.frame_count] += 1
        _feed(
            content_hasher,
            {
                "panel_id": spec.panel_id,
                "sha256": raw_digest,
                "byte_count": byte_count,
            },
        )
        _feed(
            property_hasher,
            {
                "panel_id": spec.panel_id,
                "format": properties.format,
                "mode": properties.mode,
                "width": properties.width,
                "height": properties.height,
                "info_keys": properties.info_keys,
                "frame_count": properties.frame_count,
                "frame_properties": properties.frame_properties,
            },
        )
        _feed(initial_state_hasher, {"panel_id": spec.panel_id, "stat": fingerprint})

        if expected_properties is not None:
            if properties.mode != expected_properties.mode:
                anomaly(spec, "mode", expected_properties.mode, properties.mode)
            if (properties.width, properties.height) != (
                expected_properties.width,
                expected_properties.height,
            ):
                anomaly(
                    spec,
                    "size",
                    [expected_properties.width, expected_properties.height],
                    [properties.width, properties.height],
                )
            if properties.info_keys != expected_properties.info_keys:
                anomaly(
                    spec,
                    "info_keys",
                    list(expected_properties.info_keys),
                    list(properties.info_keys),
                )
            if properties.frame_count != expected_properties.frame_count:
                anomaly(
                    spec,
                    "frame_count",
                    expected_properties.frame_count,
                    properties.frame_count,
                )

    if panel_count == 0:
        raise ImageAuditError("cannot audit an empty corpus")
    if _current_source_state(corpus, manifest_tasks) != initial_state_hasher.hexdigest():
        raise ImageAuditError("one or more panel paths changed during the complete audit")

    size_distribution = tuple(
        (width, height, count)
        for (width, height), count in sorted(size_counts.items())
    )
    info_distribution = tuple(sorted(info_key_counts.items(), key=lambda item: item[0]))
    report_fields: dict[str, object] = {
        "task_count": len(tasks),
        "panel_count": panel_count,
        "byte_count_total": byte_count_total,
        "family_task_counts": _sorted_counter(family_task_counts),
        "family_panel_counts": _sorted_counter(family_panel_counts),
        "format_counts": _sorted_counter(format_counts),
        "mode_counts": _sorted_counter(mode_counts),
        "size_counts": size_distribution,
        "info_key_set_counts": info_distribution,
        "frame_count_counts": _sorted_counter(frame_counts),
        "content_summary_digest": _SHA256_PREFIXED + content_hasher.hexdigest(),
        "property_summary_digest": _SHA256_PREFIXED + property_hasher.hexdigest(),
        "corpus_manifest_digest": corpus_manifest.digest if corpus_manifest else None,
        "expectations": expected_properties,
        "require_expected_properties": require_expected_properties,
        "anomaly_count": anomaly_count,
        "anomalies": tuple(anomalies),
        "anomalies_truncated": anomaly_count > len(anomalies),
    }
    provisional = ImageAuditReport(**report_fields, digest="")
    report = ImageAuditReport(**report_fields, digest=_address(provisional.content_dict()))
    if require_expected_properties and anomaly_count:
        raise ImageExpectationError(report)
    return report


__all__ = [
    "AUDIT_SCHEMA",
    "ImageAnomaly",
    "ImageAuditError",
    "ImageAuditReport",
    "ImageExpectationError",
    "ImageExpectations",
    "audit_corpus_images",
]
