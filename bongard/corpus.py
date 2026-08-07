"""Canonical, content-addressed access to the ShapeBongard_V2 corpus.

The official archive and the upstream generator use almost the same directory
tree, but disagree on one component::

    archive:   <root>/{ff,bd,hd}/images/<task>/{1,0}/{0..6}.png
    generator: <root>/{ff,bd,hd}/png/<task>/{1,0}/{0..6}.png

This module accepts both layouts.  It deliberately does not render programs or
guess labels from pixels: the corpus boundary is a read-only inventory of the
fourteen source PNGs belonging to each task.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "gkm.shape-bongard-corpus.v1"
TASK_MANIFEST_SCHEMA = "gkm.shape-bongard-task.v1"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

FAMILIES = ("ff", "bd", "hd")
EXPECTED_FAMILY_COUNTS: Mapping[str, int] = MappingProxyType(
    {"ff": 3_600, "bd": 4_000, "hd": 4_400}
)
EXPECTED_SPLIT_COUNTS: Mapping[str, int] = MappingProxyType(
    {"train": 9_300, "val": 900, "test": 1_800}
)
EXPECTED_REGIME_COUNTS: Mapping[str, int] = MappingProxyType(
    {"FF": 600, "BA": 480, "CM": 400, "NV": 320}
)


class CorpusError(RuntimeError):
    """Base class for corpus discovery and integrity errors."""


class CorpusLayoutError(CorpusError):
    """The supplied path does not identify one unambiguous corpus root."""


class CorpusValidationError(CorpusError):
    """The corpus or its official split fails a structural invariant."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _address(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_address(path: Path, *, require_png: bool = False) -> tuple[str, int]:
    """Hash one no-follow file descriptor and re-bind it to *path*.

    A path-level ``stat/read/stat`` sequence can be redirected between the
    first check and ``open``.  The corpus manifest is an authentication
    boundary, so bind the read to one regular-file inode and require the
    pathname to name that same inode before and after hashing.
    """

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = os.lstat(path)
        if not stat.S_ISREG(before_path.st_mode):
            raise CorpusValidationError(
                f"panel is not a regular no-follow file: {path}"
            )
        descriptor = os.open(path, flags)
    except CorpusValidationError:
        raise
    except OSError as exc:
        raise CorpusValidationError(f"cannot open panel safely: {path}") from exc

    identity = (
        before_path.st_dev,
        before_path.st_ino,
        before_path.st_size,
        before_path.st_mtime_ns,
        before_path.st_ctime_ns,
    )
    digest = hashlib.sha256()
    prefix = b""
    total = 0
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or opened_identity != identity:
            raise CorpusValidationError(
                f"panel path changed while being opened: {path}"
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            if len(prefix) < len(PNG_SIGNATURE):
                prefix += chunk[: len(PNG_SIGNATURE) - len(prefix)]
            digest.update(chunk)
            total += len(chunk)
        after_open = os.fstat(descriptor)
        after_identity = (
            after_open.st_dev,
            after_open.st_ino,
            after_open.st_size,
            after_open.st_mtime_ns,
            after_open.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(after_open.st_mode)
            or after_identity != identity
            or total != opened.st_size
        ):
            raise CorpusValidationError(f"file changed while hashing: {path}")
    except OSError as exc:
        raise CorpusValidationError(f"cannot hash panel safely: {path}") from exc
    finally:
        os.close(descriptor)

    try:
        after_path = os.lstat(path)
    except OSError as exc:
        raise CorpusValidationError(
            f"panel path changed after hashing: {path}"
        ) from exc
    if (
        not stat.S_ISREG(after_path.st_mode)
        or (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_size,
            after_path.st_mtime_ns,
            after_path.st_ctime_ns,
        )
        != identity
    ):
        raise CorpusValidationError(f"panel path changed while hashing: {path}")
    if require_png and prefix != PNG_SIGNATURE:
        raise CorpusValidationError(f"file has .png suffix but no PNG signature: {path}")
    return "sha256:" + digest.hexdigest(), total


def _normalise_group_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


_PRIMARY_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}

_REGIME_ALIASES = {
    "test_ff": "FF",
    "tval_ff": "FF",
    "ff_test": "FF",
    "test_bd": "BA",
    "tval_bd": "BA",
    "bd_test": "BA",
    "test_ba": "BA",
    "test_hd_comb": "CM",
    "tval_hd_comb": "CM",
    "hd_comb_test": "CM",
    "test_cm": "CM",
    "test_hd_novel": "NV",
    "tval_hd_novel": "NV",
    "hd_novel_test": "NV",
    "test_nv": "NV",
}


@dataclass(frozen=True)
class SplitAssignment:
    """Canonical official split metadata for one task."""

    split: str | None
    regime: str | None


@dataclass(frozen=True)
class SplitIndex:
    """Immutable normalisation of ``ShapeBongard_V2_split.json``."""

    groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    source_path: Path | None = None
    source_digest: str | None = None

    @classmethod
    def empty(cls) -> "SplitIndex":
        return cls()

    @classmethod
    def load(cls, path: str | Path) -> "SplitIndex":
        source = Path(path).expanduser().resolve()
        try:
            raw_bytes = source.read_bytes()
            raw = json.loads(raw_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusValidationError(f"cannot read split file {source}: {exc}") from exc
        if not isinstance(raw, dict):
            raise CorpusValidationError("split JSON must be an object mapping names to task lists")

        groups: list[tuple[str, tuple[str, ...]]] = []
        for key, values in raw.items():
            if not isinstance(key, str) or not isinstance(values, list):
                raise CorpusValidationError("each split entry must be a string key and JSON list")
            if not all(isinstance(value, str) and value for value in values):
                raise CorpusValidationError(f"split {key!r} contains a non-string or empty task id")
            if len(values) != len(set(values)):
                raise CorpusValidationError(f"split {key!r} contains duplicate task ids")
            groups.append((key, tuple(sorted(values))))
        return cls(
            groups=tuple(sorted(groups)),
            source_path=source,
            source_digest="sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
        )

    @property
    def raw_groups(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(dict(self.groups))

    def _canonical_sets(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        primary: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
        regimes: dict[str, set[str]] = {name: set() for name in ("FF", "BA", "CM", "NV")}
        for raw_name, task_ids in self.groups:
            name = _normalise_group_name(raw_name)
            if name in _PRIMARY_ALIASES:
                primary[_PRIMARY_ALIASES[name]].update(task_ids)
            if name in _REGIME_ALIASES:
                regimes[_REGIME_ALIASES[name]].update(task_ids)

        # The released file contains the four test_* groups but no aggregate
        # test key.  Their union is the official test split.
        regime_union = set().union(*regimes.values())
        if primary["test"]:
            if regime_union and regime_union != primary["test"]:
                raise CorpusValidationError(
                    "explicit test split differs from the union of FF/BA/CM/NV"
                )
        else:
            primary["test"] = regime_union
        return primary, regimes

    @property
    def canonical_groups(self) -> Mapping[str, tuple[str, ...]]:
        primary, regimes = self._canonical_sets()
        result = {name: tuple(sorted(values)) for name, values in primary.items()}
        result.update({name: tuple(sorted(values)) for name, values in regimes.items()})
        return MappingProxyType(result)

    def assignment(self, task_id: str) -> SplitAssignment:
        primary, regimes = self._canonical_sets()
        split_hits = [name for name, values in primary.items() if task_id in values]
        regime_hits = [name for name, values in regimes.items() if task_id in values]
        if len(split_hits) > 1:
            raise CorpusValidationError(
                f"task {task_id!r} belongs to multiple primary splits: {split_hits}"
            )
        if len(regime_hits) > 1:
            raise CorpusValidationError(
                f"task {task_id!r} belongs to multiple test regimes: {regime_hits}"
            )
        return SplitAssignment(
            split=split_hits[0] if split_hits else None,
            regime=regime_hits[0] if regime_hits else None,
        )

    def validate(self, task_ids: Iterable[str], *, official_counts: bool = False) -> None:
        known = set(task_ids)
        primary, regimes = self._canonical_sets()
        referenced = set().union(*primary.values(), *regimes.values())
        unknown = referenced - known
        if unknown:
            sample = ", ".join(sorted(unknown)[:5])
            raise CorpusValidationError(
                f"split file references {len(unknown)} absent tasks (first: {sample})"
            )

        for names, groups in ((tuple(primary), primary), (tuple(regimes), regimes)):
            for index, left in enumerate(names):
                for right in names[index + 1 :]:
                    overlap = groups[left] & groups[right]
                    if overlap:
                        sample = ", ".join(sorted(overlap)[:5])
                        raise CorpusValidationError(
                            f"split groups {left} and {right} overlap (first: {sample})"
                        )

        for regime, values in regimes.items():
            if not values <= primary["test"]:
                raise CorpusValidationError(f"regime {regime} is not a subset of test")

        classified = set().union(*primary.values())
        if classified != known:
            missing = known - classified
            sample = ", ".join(sorted(missing)[:5])
            raise CorpusValidationError(
                f"split file leaves {len(missing)} tasks unclassified (first: {sample})"
            )

        if official_counts:
            actual_primary = {name: len(values) for name, values in primary.items()}
            actual_regimes = {name: len(values) for name, values in regimes.items()}
            if actual_primary != dict(EXPECTED_SPLIT_COUNTS):
                raise CorpusValidationError(
                    f"official split counts are {actual_primary}, expected {dict(EXPECTED_SPLIT_COUNTS)}"
                )
            if actual_regimes != dict(EXPECTED_REGIME_COUNTS):
                raise CorpusValidationError(
                    f"official regime counts are {actual_regimes}, expected {dict(EXPECTED_REGIME_COUNTS)}"
                )

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_digest,
            "groups": {name: list(values) for name, values in self.canonical_groups.items()},
        }


@dataclass(frozen=True)
class PanelManifest:
    """A content-addressed source panel; ``path`` is intentionally not hashed."""

    panel_id: str
    task_id: str
    family: str
    polarity: str
    index: int
    filename: str
    path: Path
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "task_id": self.task_id,
            "family": self.family,
            "polarity": self.polarity,
            "index": self.index,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class TaskManifest:
    task_id: str
    family: str
    panels: tuple[PanelManifest, ...]
    digest: str

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_MANIFEST_SCHEMA,
            "task_id": self.task_id,
            "family": self.family,
            "panels": [panel.to_dict() for panel in self.panels],
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        result["digest"] = self.digest
        return result


@dataclass(frozen=True)
class BongardTask:
    """The fourteen immutable source paths for one Bongard problem."""

    task_id: str
    family: str
    root: Path
    positive: tuple[Path, ...]
    negative: tuple[Path, ...]

    @property
    def panels(self) -> tuple[Path, ...]:
        return self.positive + self.negative

    @property
    def positive_paths(self) -> tuple[Path, ...]:
        return self.positive

    @property
    def negative_paths(self) -> tuple[Path, ...]:
        return self.negative

    def build_manifest(self) -> TaskManifest:
        panels: list[PanelManifest] = []
        for polarity, paths in (("positive", self.positive), ("negative", self.negative)):
            label = "1" if polarity == "positive" else "0"
            for index, path in enumerate(paths):
                digest, size = _file_address(path, require_png=True)
                panels.append(
                    PanelManifest(
                        panel_id=f"{self.family}/{self.task_id}/{label}/{path.name}",
                        task_id=self.task_id,
                        family=self.family,
                        polarity=polarity,
                        index=index,
                        filename=path.name,
                        path=path,
                        sha256=digest,
                        size_bytes=size,
                    )
                )
        content = {
            "schema": TASK_MANIFEST_SCHEMA,
            "task_id": self.task_id,
            "family": self.family,
            "panels": [panel.to_dict() for panel in panels],
        }
        return TaskManifest(
            task_id=self.task_id,
            family=self.family,
            panels=tuple(panels),
            digest=_address(content),
        )


@dataclass(frozen=True)
class CorpusManifest:
    layout: str
    family_counts: tuple[tuple[str, int], ...]
    tasks: tuple[TaskManifest, ...]
    split: SplitIndex
    digest: str

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "layout": self.layout,
            "family_counts": dict(self.family_counts),
            "tasks": [
                {"task_id": task.task_id, "family": task.family, "digest": task.digest}
                for task in self.tasks
            ],
            "split": self.split.to_manifest_dict(),
        }

    def to_dict(self, *, include_task_manifests: bool = False) -> dict[str, Any]:
        result = self.content_dict()
        if include_task_manifests:
            result["task_manifests"] = [task.to_dict() for task in self.tasks]
        result["digest"] = self.digest
        return result


def _png_paths(label_dir: Path) -> tuple[Path, ...]:
    if not label_dir.is_dir():
        raise CorpusValidationError(f"missing label directory: {label_dir}")
    candidates = tuple(
        sorted(
            (path for path in label_dir.iterdir() if path.suffix.lower() == ".png"),
            key=lambda path: path.name,
        )
    )
    expected_names = tuple(f"{index}.png" for index in range(7))
    if tuple(path.name for path in candidates) != expected_names:
        raise CorpusValidationError(
            f"expected 7 PNGs canonically named 0.png..6.png in {label_dir}, "
            f"found {[path.name for path in candidates]}"
        )
    paths: list[Path] = []
    for path in candidates:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise CorpusValidationError(
                f"cannot inspect source PNG without following links: {path}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise CorpusValidationError(
                f"source PNG is not a regular no-follow file: {path}"
            )
        paths.append(path.resolve(strict=True))
    return tuple(paths)


def _image_component(root: Path, family: str) -> tuple[str, Path] | None:
    family_dir = root / family
    candidates = [(name, family_dir / name) for name in ("images", "png")]
    present = [(name, path) for name, path in candidates if path.is_dir()]
    if len(present) > 1:
        nonempty = [(name, path) for name, path in present if any(path.iterdir())]
        if len(nonempty) > 1:
            raise CorpusLayoutError(
                f"both archive and generator task trees are populated for {family}: {family_dir}"
            )
        return nonempty[0] if nonempty else present[0]
    return present[0] if present else None


def _looks_like_root(path: Path) -> bool:
    return any((path / family / component).is_dir() for family in FAMILIES for component in ("images", "png"))


def _candidate_roots(path: Path, *, max_depth: int = 4) -> tuple[Path, ...]:
    if _looks_like_root(path):
        return (path,)
    preferred = (
        path / "ShapeBongard_V2",
        path / "materials" / "ShapeBongard_V2",
    )
    direct = tuple(candidate for candidate in preferred if _looks_like_root(candidate))
    if direct:
        return direct

    found: set[Path] = set()
    queue: deque[tuple[Path, int]] = deque([(path, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        try:
            children = tuple(current.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir() or child.name.startswith("."):
                continue
            if _looks_like_root(child):
                found.add(child.resolve())
            else:
                queue.append((child, depth + 1))
    return tuple(sorted(found, key=str))


class ShapeBongardCorpus:
    """Validated inventory of official or generator-layout Bongard tasks."""

    def __init__(
        self,
        root: Path,
        tasks: Sequence[BongardTask],
        *,
        layout: str,
        split: SplitIndex,
    ) -> None:
        self.root = root
        self.tasks = tuple(sorted(tasks, key=lambda task: task.task_id))
        self.layout = layout
        self.split = split
        self._by_id = {task.task_id: task for task in self.tasks}
        if len(self._by_id) != len(self.tasks):
            duplicates = [name for name, count in Counter(task.task_id for task in self.tasks).items() if count > 1]
            raise CorpusValidationError(f"duplicate task ids across families: {duplicates[:5]}")

    @classmethod
    def discover(
        cls,
        path: str | Path,
        *,
        split_file: str | Path | None = None,
        require_complete: bool = False,
        require_split: bool = True,
        max_depth: int = 4,
    ) -> "ShapeBongardCorpus":
        requested = Path(path).expanduser().resolve()
        if not requested.is_dir():
            raise CorpusLayoutError(f"corpus search path is not a directory: {requested}")
        candidates = _candidate_roots(requested, max_depth=max_depth)
        if not candidates:
            raise CorpusLayoutError(
                f"no ShapeBongard root below {requested}; expected ff/bd/hd with images/ or png/"
            )
        if len(candidates) > 1:
            named = [candidate for candidate in candidates if candidate.name == "ShapeBongard_V2"]
            if len(named) == 1:
                root = named[0]
            else:
                rendered = ", ".join(str(candidate) for candidate in candidates)
                raise CorpusLayoutError(f"ambiguous corpus roots: {rendered}")
        else:
            root = candidates[0]
        corpus = cls.from_root(root, split_file=split_file)
        if require_complete:
            corpus.validate_complete(require_split=require_split)
        return corpus

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        split_file: str | Path | None = None,
    ) -> "ShapeBongardCorpus":
        resolved = Path(root).expanduser().resolve()
        if not _looks_like_root(resolved):
            raise CorpusLayoutError(f"not a ShapeBongard root: {resolved}")

        tasks: list[BongardTask] = []
        components: set[str] = set()
        for family in FAMILIES:
            located = _image_component(resolved, family)
            if located is None:
                continue
            component, task_root = located
            components.add(component)
            for task_dir in sorted(task_root.iterdir(), key=lambda path: path.name):
                if not task_dir.is_dir() or task_dir.name.startswith("."):
                    continue
                task_id = task_dir.name
                if not task_id:
                    raise CorpusValidationError(f"empty task id under {task_root}")
                expected_prefix = family + "_"
                if not task_id.startswith(expected_prefix):
                    raise CorpusValidationError(
                        f"task {task_id!r} in {family} does not start with {expected_prefix!r}"
                    )
                positive = _png_paths(task_dir / "1")
                negative = _png_paths(task_dir / "0")
                tasks.append(
                    BongardTask(
                        task_id=task_id,
                        family=family,
                        root=task_dir.resolve(),
                        positive=positive,
                        negative=negative,
                    )
                )

        if not tasks:
            raise CorpusValidationError(f"no Bongard tasks found in {resolved}")
        layout = "archive" if components == {"images"} else "generator" if components == {"png"} else "mixed"

        if split_file is None:
            default_split = resolved / "ShapeBongard_V2_split.json"
            split = SplitIndex.load(default_split) if default_split.is_file() else SplitIndex.empty()
        else:
            split = SplitIndex.load(split_file)
        if split.groups:
            split.validate((task.task_id for task in tasks), official_counts=False)
        return cls(resolved, tasks, layout=layout, split=split)

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[BongardTask]:
        return iter(self.tasks)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    @property
    def family_counts(self) -> Mapping[str, int]:
        counts = Counter(task.family for task in self.tasks)
        return MappingProxyType({family: counts.get(family, 0) for family in FAMILIES})

    def task(self, task_id: str) -> BongardTask:
        try:
            return self._by_id[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown Bongard task: {task_id}") from exc

    def tasks_in_split(self, split: str) -> tuple[BongardTask, ...]:
        name = split.upper() if split.upper() in EXPECTED_REGIME_COUNTS else split.lower()
        try:
            members = set(self.split.canonical_groups[name])
        except KeyError as exc:
            raise KeyError(f"unknown split or regime: {split}") from exc
        return tuple(task for task in self.tasks if task.task_id in members)

    def assignment(self, task_id: str) -> SplitAssignment:
        self.task(task_id)
        return self.split.assignment(task_id)

    def validate_complete(self, *, require_split: bool = True) -> None:
        counts = dict(self.family_counts)
        if counts != dict(EXPECTED_FAMILY_COUNTS):
            raise CorpusValidationError(
                f"family counts are {counts}, expected {dict(EXPECTED_FAMILY_COUNTS)}"
            )
        if len(self.tasks) != sum(EXPECTED_FAMILY_COUNTS.values()):
            raise CorpusValidationError(f"expected 12,000 tasks, found {len(self.tasks)}")
        if require_split and not self.split.groups:
            raise CorpusValidationError("complete corpus has no ShapeBongard_V2_split.json")
        if self.split.groups:
            self.split.validate(self.task_ids, official_counts=True)

    def build_manifest(self) -> CorpusManifest:
        task_manifests = tuple(task.build_manifest() for task in self.tasks)
        content = {
            "schema": SCHEMA_VERSION,
            "layout": self.layout,
            "family_counts": dict(self.family_counts),
            "tasks": [
                {"task_id": task.task_id, "family": task.family, "digest": task.digest}
                for task in task_manifests
            ],
            "split": self.split.to_manifest_dict(),
        }
        return CorpusManifest(
            layout=self.layout,
            family_counts=tuple(self.family_counts.items()),
            tasks=task_manifests,
            split=self.split,
            digest=_address(content),
        )


def discover_corpus(path: str | Path, **kwargs: Any) -> ShapeBongardCorpus:
    """Convenience alias for :meth:`ShapeBongardCorpus.discover`."""

    return ShapeBongardCorpus.discover(path, **kwargs)
