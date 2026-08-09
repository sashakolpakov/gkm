"""Durable support/query release gate for an authenticated extracted corpus.

This boundary is intentionally specific to
:class:`OfficialExtractedPanelArchive`.  It binds the official descriptor,
exact task inventory and split, freshly authenticated extracted-tree manifest,
and one preregistered :class:`PanelFeatureTargetedDrillPlan`.  It never opens or
claims custody over ZIP bytes or a ZIP central directory.

Preparation persists and reloads the plan, execution precommit, and the exact
one-event exposure-ledger successor before support pixels can be read.  Support
release is restricted to the twelve preregistered support identities per task.
The two preregistered query identities remain closed until an exact canonical
Python predicate freeze and its decision commit have both been durably stored
and reloaded.
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
import stat
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest, canonical_json
from bongard.corpus import (
    SCHEMA_VERSION as CORPUS_MANIFEST_SCHEMA,
    TASK_MANIFEST_SCHEMA,
    CorpusValidationError,
    PanelManifest,
    SplitIndex,
)
from bongard.exposure import ExposureLedger
from bongard.object_bongard_batch import (
    ObjectBongardTaskPlan,
    object_bongard_task_inventory_digest,
)
from bongard.object_bongard_release_gate import (
    ObjectBongardReleaseStore,
    ObjectBongardTaskCommitProtocol,
    ObjectBongardTaskFreezeProtocol,
    ObjectBongardWriteOnceReceipt,
    persist_object_bongard_task_commit,
    persist_object_bongard_task_freeze,
)
from bongard.official_extracted_panel_archive import (
    EXTRACTED_ARCHIVE_SCHEMA,
    OfficialExtractedPanelArchive,
    ReleasedOfficialExtractedPanel,
)
from bongard.panel_feature_targeted_drill_plan import (
    PanelFeatureTargetedDrillPlan,
    panel_feature_targeted_drill_plan_source_digest,
    verify_panel_feature_targeted_drill_plan,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.release import OfficialReleaseDescriptor


PANEL_FEATURE_EXTRACTED_PRECOMMIT_SCHEMA = (
    "gkm.bongard-panel-feature-extracted-execution-precommit.v1"
)
PANEL_FEATURE_EXTRACTED_AUTHORIZATION_SCHEMA = (
    "gkm.bongard-panel-feature-extracted-release-authorization.v1"
)
PANEL_FEATURE_EXTRACTED_EXPOSURE_PHASE = (
    "panel-feature-targeted-extracted-support-release"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RAW_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_KEY = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_FORBIDDEN_CONFIG_KEY = re.compile(r"(?:pixel|image|png|action|program|path|bytes)")
_MAX_SPLIT_BYTES = 64 * 1024 * 1024


class PanelFeatureExtractedReleaseGateError(RuntimeError):
    """A tree, plan, ledger, durable record, freeze, or release differs."""


def panel_feature_extracted_release_gate_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _bytes_address(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_address(value: object, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise PanelFeatureExtractedReleaseGateError(
            f"{label} must be a sha256: address"
        )
    return value


def _require_raw_digest(value: object, label: str) -> str:
    if type(value) is not str or _RAW_DIGEST.fullmatch(value) is None:
        raise PanelFeatureExtractedReleaseGateError(
            f"{label} must be a raw SHA-256"
        )
    return value


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(type(key) is not str for key in value)
        or set(value) != expected
    ):
        raise PanelFeatureExtractedReleaseGateError(f"{label} fields differ")
    return value


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_release_or_replay": False,
        "release_source_authority": "authenticated-extracted-tree-manifest",
        "zip_archive_opened_or_required": False,
        "zip_archive_digest_used_as_release_custody": False,
        "zip_central_directory_custody_claimed": False,
    }


def _sorted_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise PanelFeatureExtractedReleaseGateError(f"{label} must be a sequence")
    result = tuple(values)
    if (
        not result
        or result != tuple(sorted(set(result)))
        or any(type(item) is not str or not item for item in result)
    ):
        raise PanelFeatureExtractedReleaseGateError(
            f"{label} must be nonempty, unique, and sorted"
        )
    return result


def _freeze_bindings(values: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, Mapping) or not values:
        raise PanelFeatureExtractedReleaseGateError(
            "runtime source bindings must be nonempty"
        )
    result = tuple(sorted(values.items()))
    if any(
        type(key) is not str
        or _CONFIG_KEY.fullmatch(key) is None
        or type(value) is not str
        for key, value in result
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "runtime source binding differs"
        )
    for key, value in result:
        _require_address(value, f"runtime source binding {key}")
    return result


def _freeze_configuration(
    values: Mapping[str, str | int | bool],
) -> tuple[tuple[str, str | int | bool], ...]:
    if not isinstance(values, Mapping):
        raise PanelFeatureExtractedReleaseGateError("configuration must be a mapping")
    result = tuple(sorted(values.items()))
    for key, value in result:
        if (
            type(key) is not str
            or _CONFIG_KEY.fullmatch(key) is None
            or _FORBIDDEN_CONFIG_KEY.search(key) is not None
            or isinstance(value, float)
            or value is None
            or type(value) not in (str, int, bool)
            or (
                type(value) is str
                and (
                    not value
                    or "\x00" in value
                    or len(value.encode("utf-8")) > 512
                )
            )
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "configuration must be bounded metadata with no visual/action inputs"
            )
    return result


def _stable_split_read(path: Path) -> bytes:
    if not path.is_absolute():
        raise PanelFeatureExtractedReleaseGateError(
            "split source path must be absolute"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.lstat(path)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PanelFeatureExtractedReleaseGateError(
            "cannot open split source safely"
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
            or not 0 < opened.st_size <= _MAX_SPLIT_BYTES
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "split source is not a stable private regular file"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor, min(1024 * 1024, _MAX_SPLIT_BYTES - total + 1)
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_SPLIT_BYTES:
                raise PanelFeatureExtractedReleaseGateError(
                    "split source exceeds byte bound"
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
            raise PanelFeatureExtractedReleaseGateError(
                "split source changed while reading"
            )
    finally:
        os.close(descriptor)
    if path.resolve(strict=True) != path:
        raise PanelFeatureExtractedReleaseGateError(
            "split source traverses a symlinked path"
        )
    return b"".join(chunks)


def _canonical_split(
    split: SplitIndex,
    *,
    descriptor: OfficialReleaseDescriptor,
    task_ids: tuple[str, ...],
) -> SplitIndex:
    if type(split) is not SplitIndex:
        raise TypeError("split must be exact SplitIndex")
    if split.source_path is None or split.source_digest is None:
        raise PanelFeatureExtractedReleaseGateError(
            "release split needs an exact source file"
        )
    path = split.source_path
    payload = _stable_split_read(path)
    if (
        path.name != descriptor.split_filename
        or len(payload) != descriptor.split_size_bytes
        or _bytes_address(payload) != descriptor.split_sha256
        or split.source_digest != descriptor.split_sha256
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "split source differs from the official descriptor"
        )
    try:
        decoded = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PanelFeatureExtractedReleaseGateError(
            "split source is not valid JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise PanelFeatureExtractedReleaseGateError("split source must be an object")
    groups: list[tuple[str, tuple[str, ...]]] = []
    for key, values in decoded.items():
        if (
            type(key) is not str
            or type(values) is not list
            or any(type(item) is not str or not item for item in values)
            or len(values) != len(set(values))
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "split source groups differ"
            )
        groups.append((key, tuple(sorted(values))))
    replayed = SplitIndex(
        tuple(sorted(groups)),
        path,
        _bytes_address(payload),
    )
    try:
        replayed.validate(task_ids, official_counts=False)
    except (CorpusValidationError, TypeError, ValueError) as exc:
        raise PanelFeatureExtractedReleaseGateError(
            "split partition differs from the task inventory"
        ) from exc
    if replayed != split:
        raise PanelFeatureExtractedReleaseGateError(
            "split index differs from exact source bytes"
        )
    canonical = replayed.canonical_groups
    primary = {name: len(canonical[name]) for name in ("train", "val", "test")}
    regimes = {name: len(canonical[name]) for name in ("FF", "BA", "CM", "NV")}
    if (
        primary != dict(descriptor.primary_split_counts)
        or regimes != dict(descriptor.regime_counts)
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "split counts differ from the official descriptor"
        )
    return replayed


def _all_plan_panels(
    plan: PanelFeatureTargetedDrillPlan,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    support = tuple(
        sorted(
            panel_id
            for task in plan.tasks
            for panel_id in (
                *task.side_0_support_panel_ids,
                *task.side_1_support_panel_ids,
            )
        )
    )
    query = tuple(
        sorted(
            panel_id
            for task in plan.tasks
            for panel_id in (
                task.side_0_query_panel_id,
                task.side_1_query_panel_id,
            )
        )
    )
    if (
        not support
        or not query
        or len(support) != 12 * len(plan.tasks)
        or len(query) != 2 * len(plan.tasks)
        or len(set(support)) != len(support)
        or len(set(query)) != len(query)
        or set(support) & set(query)
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "targeted plan support/query inventory differs"
        )
    return support, query


def _canonical_plan(value: object) -> PanelFeatureTargetedDrillPlan:
    if type(value) is not PanelFeatureTargetedDrillPlan:
        raise TypeError("plan must be exact PanelFeatureTargetedDrillPlan")
    restored = PanelFeatureTargetedDrillPlan.from_data(value.to_data())
    if restored != value:
        raise PanelFeatureExtractedReleaseGateError(
            "targeted plan canonical reload differs"
        )
    return restored


def _canonical_descriptor(value: object) -> OfficialReleaseDescriptor:
    if type(value) is not OfficialReleaseDescriptor:
        raise TypeError("descriptor must be exact OfficialReleaseDescriptor")
    restored = OfficialReleaseDescriptor.from_dict(value.to_dict())
    if restored != value:
        raise PanelFeatureExtractedReleaseGateError(
            "release descriptor canonical reload differs"
        )
    return restored


def _canonical_ledger(value: object, label: str) -> ExposureLedger:
    if type(value) is not ExposureLedger:
        raise TypeError(f"{label} must be exact ExposureLedger")
    restored = ExposureLedger.from_dict(value.to_dict())
    if restored != value:
        raise PanelFeatureExtractedReleaseGateError(
            f"{label} canonical reload differs"
        )
    return restored


def _verify_extracted_tree_inventory(
    archive: OfficialExtractedPanelArchive,
    *,
    descriptor: OfficialReleaseDescriptor,
    task_ids: tuple[str, ...],
    split: SplitIndex,
) -> None:
    if type(archive) is not OfficialExtractedPanelArchive:
        raise TypeError("archive must be exact OfficialExtractedPanelArchive")
    try:
        root_metadata = os.lstat(archive.corpus_root)
        resolved_root = archive.corpus_root.resolve(strict=True)
    except OSError as exc:
        raise PanelFeatureExtractedReleaseGateError(
            "extracted corpus root is unavailable"
        ) from exc
    expected_panels = tuple(
        sorted(
            f"{task_id.split('_', 1)[0]}/{task_id}/{side}/{index}.png"
            for task_id in task_ids
            for side in ("0", "1")
            for index in range(7)
        )
    )
    family_counts: dict[str, int] = {
        family: 0 for family, _count in descriptor.family_counts
    }
    for task_id in task_ids:
        family = task_id.split("_", 1)[0]
        if family not in family_counts:
            raise PanelFeatureExtractedReleaseGateError(
                "task inventory family is absent from the release descriptor"
            )
        family_counts[family] = family_counts.get(family, 0) + 1
    archive_record_content = {
        "schema": EXTRACTED_ARCHIVE_SCHEMA,
        "release_descriptor_digest": archive.release_descriptor_digest,
        "corpus_manifest_digest": archive.corpus_manifest_digest,
        "layout": archive.layout,
        "panel_count": len(archive.panel_by_id),
        "read_policy": (
            "fresh-full-manifest-verification-then-stable-panel-reread"
        ),
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_release_or_replay": False,
    }
    if (
        archive.release_descriptor_digest != descriptor.digest
        or archive.corpus_manifest_digest != descriptor.corpus_manifest_sha256
        or archive.record_digest != _address(archive_record_content)
        or archive.corpus_root != resolved_root
        or not stat.S_ISDIR(root_metadata.st_mode)
        or tuple(archive.task_digest_by_task_id) != task_ids
        or tuple(archive.panel_by_id) != expected_panels
        or family_counts != dict(descriptor.family_counts)
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "extracted tree, manifest, descriptor, and task inventory differ"
        )
    component = "images" if archive.layout == "archive" else "png"
    panels_by_task: dict[str, list[PanelManifest]] = {
        task_id: [] for task_id in task_ids
    }
    for panel_id, row in archive.panel_by_id.items():
        family, task_tail = panel_id.split("/", 1)
        task_id, side, filename = task_tail.split("/")
        expected_path = (
            archive.corpus_root / family / component / task_id / side / filename
        )
        try:
            panel_metadata = os.lstat(expected_path)
            resolved_path = expected_path.resolve(strict=True)
        except OSError as exc:
            raise PanelFeatureExtractedReleaseGateError(
                "extracted tree manifest path is unavailable"
            ) from exc
        if (
            type(row) is not PanelManifest
            or row.panel_id != panel_id
            or row.task_id != task_id
            or row.family != family
            or row.path != expected_path
            or resolved_path != expected_path
            or not stat.S_ISREG(panel_metadata.st_mode)
            or panel_metadata.st_nlink != 1
            or panel_metadata.st_size != row.size_bytes
            or type(row.size_bytes) is not int
            or row.size_bytes <= 0
            or type(row.sha256) is not str
            or _ADDRESS.fullmatch(row.sha256) is None
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "extracted tree manifest panel row differs"
            )
        panels_by_task[task_id].append(row)

    task_rows: list[dict[str, str]] = []
    for task_id in task_ids:
        family = task_id.split("_", 1)[0]
        panels = tuple(
            sorted(
                panels_by_task[task_id],
                key=lambda row: (
                    0 if row.polarity == "positive" else 1,
                    row.index,
                ),
            )
        )
        task_content = {
            "schema": TASK_MANIFEST_SCHEMA,
            "task_id": task_id,
            "family": family,
            "panels": [row.to_dict() for row in panels],
        }
        task_digest = _address(task_content)
        if (
            len(panels) != 14
            or task_digest != archive.task_digest_by_task_id[task_id]
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "extracted task manifest chain differs"
            )
        task_rows.append(
            {"task_id": task_id, "family": family, "digest": task_digest}
        )
    corpus_content = {
        "schema": CORPUS_MANIFEST_SCHEMA,
        "layout": archive.layout,
        "family_counts": dict(sorted(family_counts.items())),
        "tasks": task_rows,
        "split": split.to_manifest_dict(),
    }
    if _address(corpus_content) != descriptor.corpus_manifest_sha256:
        raise PanelFeatureExtractedReleaseGateError(
            "extracted corpus manifest chain differs"
        )


def _precommit_content(
    value: "PanelFeatureExtractedExecutionPrecommit",
) -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_EXTRACTED_PRECOMMIT_SCHEMA,
        "targeted_drill_plan_digest": value.targeted_drill_plan_digest,
        "targeted_drill_algorithm_digest": value.targeted_drill_algorithm_digest,
        "targeted_drill_source_digest": value.targeted_drill_source_digest,
        "release_gate_source_digest": value.release_gate_source_digest,
        "release_descriptor_digest": value.release_descriptor_digest,
        "split_source_digest": value.split_source_digest,
        "split_manifest_digest": value.split_manifest_digest,
        "task_inventory_digest": value.task_inventory_digest,
        "train_task_ids_digest": value.train_task_ids_digest,
        "extracted_corpus_manifest_digest": (
            value.extracted_corpus_manifest_digest
        ),
        "extracted_archive_record_digest": value.extracted_archive_record_digest,
        "extracted_layout": value.extracted_layout,
        "exposure_predecessor_digest": value.exposure_predecessor_digest,
        "selected_task_ids": list(value.selected_task_ids),
        "authorized_support_panel_ids": list(value.authorized_support_panel_ids),
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "runtime_source_bindings": [list(item) for item in value.runtime_source_bindings],
        "configuration": [list(item) for item in value.configuration],
        "exposure_observed_at": value.exposure_observed_at,
        "exposure_actor": value.exposure_actor,
        "exposure_purpose": value.exposure_purpose,
        "exposure_source": value.exposure_source,
        "preregistered_plan_digest_supplied_externally": True,
        "release_descriptor_digest_supplied_externally": True,
        "split_source_read_without_following_symlinks": True,
        "query_identities_sealed_before_support_pixels": True,
        "panel_bytes_opened_during_precommit": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureExtractedExecutionPrecommit:
    targeted_drill_plan_digest: str
    targeted_drill_algorithm_digest: str
    targeted_drill_source_digest: str
    release_gate_source_digest: str
    release_descriptor_digest: str
    split_source_digest: str
    split_manifest_digest: str
    task_inventory_digest: str
    train_task_ids_digest: str
    extracted_corpus_manifest_digest: str
    extracted_archive_record_digest: str
    extracted_layout: str
    exposure_predecessor_digest: str
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
            "targeted_drill_plan_digest",
            "targeted_drill_algorithm_digest",
            "targeted_drill_source_digest",
            "release_gate_source_digest",
            "release_descriptor_digest",
            "split_source_digest",
            "split_manifest_digest",
            "task_inventory_digest",
            "train_task_ids_digest",
            "extracted_corpus_manifest_digest",
            "extracted_archive_record_digest",
            "exposure_predecessor_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if self.extracted_layout not in {"archive", "generator"}:
            raise PanelFeatureExtractedReleaseGateError(
                "extracted corpus layout differs"
            )
        for values, label in (
            (self.selected_task_ids, "selected task IDs"),
            (self.authorized_support_panel_ids, "support panel IDs"),
            (self.sealed_query_panel_ids, "query panel IDs"),
        ):
            if not values or values != tuple(sorted(set(values))):
                raise PanelFeatureExtractedReleaseGateError(f"{label} differ")
        if (
            set(self.authorized_support_panel_ids) & set(self.sealed_query_panel_ids)
            or len(self.authorized_support_panel_ids) != 12 * len(self.selected_task_ids)
            or len(self.sealed_query_panel_ids) != 2 * len(self.selected_task_ids)
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "support/query release inventory differs"
            )
        if (
            self.runtime_source_bindings
            != _freeze_bindings(dict(self.runtime_source_bindings))
            or self.configuration
            != _freeze_configuration(dict(self.configuration))
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "runtime bindings or configuration are not canonical"
            )
        if any(
            type(item) is not str or not item
            for item in (
                self.exposure_observed_at,
                self.exposure_actor,
                self.exposure_purpose,
                self.exposure_source,
            )
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "exposure metadata must be nonempty"
            )
        if self.record_digest != _address(_precommit_content(self)):
            raise PanelFeatureExtractedReleaseGateError(
                "extracted execution precommit digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_precommit_content(self), "record_digest": self.record_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "PanelFeatureExtractedExecutionPrecommit":
        raw = _fields(
            value,
            {
                "schema",
                "targeted_drill_plan_digest",
                "targeted_drill_algorithm_digest",
                "targeted_drill_source_digest",
                "release_gate_source_digest",
                "release_descriptor_digest",
                "split_source_digest",
                "split_manifest_digest",
                "task_inventory_digest",
                "train_task_ids_digest",
                "extracted_corpus_manifest_digest",
                "extracted_archive_record_digest",
                "extracted_layout",
                "exposure_predecessor_digest",
                "selected_task_ids",
                "authorized_support_panel_ids",
                "sealed_query_panel_ids",
                "runtime_source_bindings",
                "configuration",
                "exposure_observed_at",
                "exposure_actor",
                "exposure_purpose",
                "exposure_source",
                "preregistered_plan_digest_supplied_externally",
                "release_descriptor_digest_supplied_externally",
                "split_source_read_without_following_symlinks",
                "query_identities_sealed_before_support_pixels",
                "panel_bytes_opened_during_precommit",
                *_authority_data(),
                "record_digest",
            },
            "panel-feature extracted execution precommit",
        )
        if (
            raw["schema"] != PANEL_FEATURE_EXTRACTED_PRECOMMIT_SCHEMA
            or raw["preregistered_plan_digest_supplied_externally"] is not True
            or raw["release_descriptor_digest_supplied_externally"] is not True
            or raw["split_source_read_without_following_symlinks"] is not True
            or raw["query_identities_sealed_before_support_pixels"] is not True
            or raw["panel_bytes_opened_during_precommit"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(
                type(raw[name]) is not list
                for name in (
                    "selected_task_ids",
                    "authorized_support_panel_ids",
                    "sealed_query_panel_ids",
                    "runtime_source_bindings",
                    "configuration",
                )
            )
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "extracted execution precommit policy differs"
            )
        try:
            result = cls(
                raw["targeted_drill_plan_digest"],
                raw["targeted_drill_algorithm_digest"],
                raw["targeted_drill_source_digest"],
                raw["release_gate_source_digest"],
                raw["release_descriptor_digest"],
                raw["split_source_digest"],
                raw["split_manifest_digest"],
                raw["task_inventory_digest"],
                raw["train_task_ids_digest"],
                raw["extracted_corpus_manifest_digest"],
                raw["extracted_archive_record_digest"],
                raw["extracted_layout"],
                raw["exposure_predecessor_digest"],
                tuple(raw["selected_task_ids"]),
                tuple(raw["authorized_support_panel_ids"]),
                tuple(raw["sealed_query_panel_ids"]),
                tuple(tuple(item) for item in raw["runtime_source_bindings"]),
                tuple(tuple(item) for item in raw["configuration"]),
                raw["exposure_observed_at"],
                raw["exposure_actor"],
                raw["exposure_purpose"],
                raw["exposure_source"],
                raw["record_digest"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PanelFeatureExtractedReleaseGateError):
                raise
            raise PanelFeatureExtractedReleaseGateError(
                "extracted execution precommit is malformed"
            ) from exc
        if result.to_data() != dict(raw):
            raise PanelFeatureExtractedReleaseGateError(
                "extracted execution precommit is not canonical"
            )
        return result


def create_panel_feature_extracted_execution_precommit(
    *,
    plan: PanelFeatureTargetedDrillPlan,
    expected_plan_digest: str,
    selection_seed: str,
    predecessor: ExposureLedger,
    descriptor: OfficialReleaseDescriptor,
    expected_release_descriptor_digest: str,
    archive: OfficialExtractedPanelArchive,
    split: SplitIndex,
    task_ids: Sequence[str],
    runtime_source_bindings: Mapping[str, str],
    configuration: Mapping[str, str | int | bool],
    exposure_observed_at: str,
    exposure_actor: str = "headless-codex-proposer",
    exposure_purpose: str = "targeted-panel-feature-support-and-sealed-query",
    exposure_source: str = "official-authenticated-extracted-shapebongard-tree",
) -> PanelFeatureExtractedExecutionPrecommit:
    """Bind all metadata and the extracted manifest without opening panel bytes."""

    canonical_plan = _canonical_plan(plan)
    expected_plan = _require_address(
        expected_plan_digest, "expected preregistered plan digest"
    )
    if canonical_plan.record_digest != expected_plan:
        raise PanelFeatureExtractedReleaseGateError(
            "targeted plan differs from its preregistered commitment"
        )
    canonical_descriptor = _canonical_descriptor(descriptor)
    expected_descriptor = _require_address(
        expected_release_descriptor_digest,
        "expected release descriptor digest",
    )
    if canonical_descriptor.digest != expected_descriptor:
        raise PanelFeatureExtractedReleaseGateError(
            "release descriptor differs from its external commitment"
        )
    canonical_predecessor = _canonical_ledger(predecessor, "exposure predecessor")
    inventory = _sorted_ids(task_ids, "official task inventory")
    canonical_split = _canonical_split(
        split,
        descriptor=canonical_descriptor,
        task_ids=inventory,
    )
    train = tuple(canonical_split.canonical_groups["train"])
    try:
        verify_panel_feature_targeted_drill_plan(
            canonical_plan,
            task_ids=inventory,
            train_task_ids=train,
            predecessor=canonical_predecessor,
            selection_seed=selection_seed,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, PanelFeatureExtractedReleaseGateError):
            raise
        raise PanelFeatureExtractedReleaseGateError(
            "targeted drill plan metadata replay differs"
        ) from exc
    _verify_extracted_tree_inventory(
        archive,
        descriptor=canonical_descriptor,
        task_ids=inventory,
        split=canonical_split,
    )
    inventory_digest = object_bongard_task_inventory_digest(inventory)
    train_digest = _address(list(train))
    selected = tuple(task.task_id for task in canonical_plan.tasks)
    support, query = _all_plan_panels(canonical_plan)
    if (
        canonical_descriptor.task_ids_sha256 != inventory_digest
        or canonical_plan.task_inventory_digest != inventory_digest
        or canonical_plan.train_task_ids_digest != train_digest
        or canonical_plan.split_source_digest != canonical_descriptor.split_sha256
        or canonical_plan.release_descriptor_digest != canonical_descriptor.digest
        or canonical_plan.exposure_predecessor_digest
        != canonical_predecessor.digest
        or canonical_plan.exposed_task_ids_digest
        != _address(list(sorted(canonical_predecessor.exposed_task_ids)))
        or canonical_predecessor.corpus_digest
        != canonical_descriptor.corpus_manifest_sha256
        or not set(selected) <= set(train)
        or set(selected) & set(canonical_predecessor.exposed_task_ids)
        or any(panel_id not in archive.panel_by_id for panel_id in (*support, *query))
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "plan, inventory, split, ledger, and extracted manifest differ"
        )
    bindings = dict(runtime_source_bindings)
    automatic = {
        "targeted_plan_source": (
            "sha256:" + panel_feature_targeted_drill_plan_source_digest()
        ),
        "release_gate_source": (
            "sha256:" + panel_feature_extracted_release_gate_source_digest()
        ),
    }
    for key, value in automatic.items():
        if key in bindings and bindings[key] != value:
            raise PanelFeatureExtractedReleaseGateError(
                f"automatic source binding {key} differs"
            )
        bindings[key] = value
    values: dict[str, object] = {
        "targeted_drill_plan_digest": canonical_plan.record_digest,
        "targeted_drill_algorithm_digest": canonical_plan.algorithm_digest,
        "targeted_drill_source_digest": automatic["targeted_plan_source"],
        "release_gate_source_digest": automatic["release_gate_source"],
        "release_descriptor_digest": canonical_descriptor.digest,
        "split_source_digest": canonical_descriptor.split_sha256,
        "split_manifest_digest": _address(canonical_split.to_manifest_dict()),
        "task_inventory_digest": inventory_digest,
        "train_task_ids_digest": train_digest,
        "extracted_corpus_manifest_digest": archive.corpus_manifest_digest,
        "extracted_archive_record_digest": archive.record_digest,
        "extracted_layout": archive.layout,
        "exposure_predecessor_digest": canonical_predecessor.digest,
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
    provisional = object.__new__(PanelFeatureExtractedExecutionPrecommit)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return PanelFeatureExtractedExecutionPrecommit(
        **values,  # type: ignore[arg-type]
        record_digest=_address(_precommit_content(provisional)),
    )


def _authorization_content(
    value: "PanelFeatureExtractedReleaseAuthorization",
) -> dict[str, object]:
    return {
        "schema": PANEL_FEATURE_EXTRACTED_AUTHORIZATION_SCHEMA,
        "targeted_drill_plan_digest": value.targeted_drill_plan_digest,
        "execution_precommit_digest": value.execution_precommit_digest,
        "exposure_predecessor_digest": value.exposure_predecessor_digest,
        "exposure_successor_digest": value.exposure_successor_digest,
        "exposure_event_digest": value.exposure_event_digest,
        "release_descriptor_digest": value.release_descriptor_digest,
        "split_source_digest": value.split_source_digest,
        "task_inventory_digest": value.task_inventory_digest,
        "extracted_corpus_manifest_digest": (
            value.extracted_corpus_manifest_digest
        ),
        "extracted_archive_record_digest": value.extracted_archive_record_digest,
        "selected_task_ids": list(value.selected_task_ids),
        "authorized_support_panel_ids": list(value.authorized_support_panel_ids),
        "sealed_query_panel_ids": list(value.sealed_query_panel_ids),
        "plan_store_receipt_digest": value.plan_store_receipt_digest,
        "precommit_store_receipt_digest": value.precommit_store_receipt_digest,
        "exposure_store_receipt_digest": value.exposure_store_receipt_digest,
        "exposure_successor_persisted_and_reloaded_before_authorization": True,
        "support_release_requires_durable_exposure_successor": True,
        "query_release_requires_exact_durable_python_freeze_and_commit": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class PanelFeatureExtractedReleaseAuthorization:
    targeted_drill_plan_digest: str
    execution_precommit_digest: str
    exposure_predecessor_digest: str
    exposure_successor_digest: str
    exposure_event_digest: str
    release_descriptor_digest: str
    split_source_digest: str
    task_inventory_digest: str
    extracted_corpus_manifest_digest: str
    extracted_archive_record_digest: str
    selected_task_ids: tuple[str, ...]
    authorized_support_panel_ids: tuple[str, ...]
    sealed_query_panel_ids: tuple[str, ...]
    plan_store_receipt_digest: str
    precommit_store_receipt_digest: str
    exposure_store_receipt_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        for name in (
            "targeted_drill_plan_digest",
            "execution_precommit_digest",
            "exposure_predecessor_digest",
            "exposure_successor_digest",
            "exposure_event_digest",
            "release_descriptor_digest",
            "split_source_digest",
            "task_inventory_digest",
            "extracted_corpus_manifest_digest",
            "extracted_archive_record_digest",
            "plan_store_receipt_digest",
            "precommit_store_receipt_digest",
            "exposure_store_receipt_digest",
            "record_digest",
        ):
            _require_address(getattr(self, name), name)
        if (
            not self.selected_task_ids
            or self.selected_task_ids != tuple(sorted(set(self.selected_task_ids)))
            or self.authorized_support_panel_ids
            != tuple(sorted(set(self.authorized_support_panel_ids)))
            or self.sealed_query_panel_ids
            != tuple(sorted(set(self.sealed_query_panel_ids)))
            or not self.authorized_support_panel_ids
            or not self.sealed_query_panel_ids
            or set(self.authorized_support_panel_ids) & set(self.sealed_query_panel_ids)
            or self.record_digest != _address(_authorization_content(self))
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "extracted release authorization differs"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_authorization_content(self),
            "record_digest": self.record_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "PanelFeatureExtractedReleaseAuthorization":
        raw = _fields(
            value,
            {
                "schema",
                "targeted_drill_plan_digest",
                "execution_precommit_digest",
                "exposure_predecessor_digest",
                "exposure_successor_digest",
                "exposure_event_digest",
                "release_descriptor_digest",
                "split_source_digest",
                "task_inventory_digest",
                "extracted_corpus_manifest_digest",
                "extracted_archive_record_digest",
                "selected_task_ids",
                "authorized_support_panel_ids",
                "sealed_query_panel_ids",
                "plan_store_receipt_digest",
                "precommit_store_receipt_digest",
                "exposure_store_receipt_digest",
                "exposure_successor_persisted_and_reloaded_before_authorization",
                "support_release_requires_durable_exposure_successor",
                "query_release_requires_exact_durable_python_freeze_and_commit",
                *_authority_data(),
                "record_digest",
            },
            "panel-feature extracted release authorization",
        )
        if (
            raw["schema"] != PANEL_FEATURE_EXTRACTED_AUTHORIZATION_SCHEMA
            or raw[
                "exposure_successor_persisted_and_reloaded_before_authorization"
            ]
            is not True
            or raw["support_release_requires_durable_exposure_successor"] is not True
            or raw[
                "query_release_requires_exact_durable_python_freeze_and_commit"
            ]
            is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(
                type(raw[name]) is not list
                for name in (
                    "selected_task_ids",
                    "authorized_support_panel_ids",
                    "sealed_query_panel_ids",
                )
            )
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "extracted release authorization policy differs"
            )
        result = cls(
            raw["targeted_drill_plan_digest"],
            raw["execution_precommit_digest"],
            raw["exposure_predecessor_digest"],
            raw["exposure_successor_digest"],
            raw["exposure_event_digest"],
            raw["release_descriptor_digest"],
            raw["split_source_digest"],
            raw["task_inventory_digest"],
            raw["extracted_corpus_manifest_digest"],
            raw["extracted_archive_record_digest"],
            tuple(raw["selected_task_ids"]),
            tuple(raw["authorized_support_panel_ids"]),
            tuple(raw["sealed_query_panel_ids"]),
            raw["plan_store_receipt_digest"],
            raw["precommit_store_receipt_digest"],
            raw["exposure_store_receipt_digest"],
            raw["record_digest"],
        )
        if result.to_data() != dict(raw):
            raise PanelFeatureExtractedReleaseGateError(
                "extracted release authorization is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class PreparedPanelFeatureExtractedRelease:
    store: ObjectBongardReleaseStore = field(compare=False, repr=False)
    plan: PanelFeatureTargetedDrillPlan
    precommit: PanelFeatureExtractedExecutionPrecommit
    predecessor: ExposureLedger
    successor: ExposureLedger
    authorization: PanelFeatureExtractedReleaseAuthorization
    plan_receipt: ObjectBongardWriteOnceReceipt
    precommit_receipt: ObjectBongardWriteOnceReceipt
    exposure_receipt: ObjectBongardWriteOnceReceipt
    authorization_receipt: ObjectBongardWriteOnceReceipt


def prepare_panel_feature_extracted_release(
    *,
    store: ObjectBongardReleaseStore,
    plan: PanelFeatureTargetedDrillPlan,
    precommit: PanelFeatureExtractedExecutionPrecommit,
    predecessor: ExposureLedger,
) -> PreparedPanelFeatureExtractedRelease:
    if type(store) is not ObjectBongardReleaseStore:
        raise TypeError("store must be exact ObjectBongardReleaseStore")
    canonical_plan = _canonical_plan(plan)
    if type(precommit) is not PanelFeatureExtractedExecutionPrecommit:
        raise TypeError(
            "precommit must be exact PanelFeatureExtractedExecutionPrecommit"
        )
    canonical_precommit = PanelFeatureExtractedExecutionPrecommit.from_data(
        precommit.to_data()
    )
    canonical_predecessor = _canonical_ledger(predecessor, "exposure predecessor")
    support, query = _all_plan_panels(canonical_plan)
    selected = tuple(task.task_id for task in canonical_plan.tasks)
    if (
        canonical_precommit.targeted_drill_plan_digest
        != canonical_plan.record_digest
        or canonical_precommit.targeted_drill_algorithm_digest
        != canonical_plan.algorithm_digest
        or canonical_precommit.targeted_drill_source_digest
        != "sha256:" + panel_feature_targeted_drill_plan_source_digest()
        or canonical_precommit.release_gate_source_digest
        != "sha256:" + panel_feature_extracted_release_gate_source_digest()
        or canonical_precommit.exposure_predecessor_digest
        != canonical_predecessor.digest
        or canonical_precommit.selected_task_ids != selected
        or canonical_precommit.authorized_support_panel_ids != support
        or canonical_precommit.sealed_query_panel_ids != query
        or set(selected) & set(canonical_predecessor.exposed_task_ids)
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "precommit, plan, and exposure predecessor differ"
        )
    plan_receipt = store.persist(
        object_kind="targeted-drill-plan",
        object_digest=canonical_plan.record_digest,
        data=canonical_plan.to_data(),
    )
    decoded_plan = PanelFeatureTargetedDrillPlan.from_data(
        store.verify(plan_receipt, expected_data=canonical_plan.to_data())
    )
    if decoded_plan != canonical_plan:
        raise PanelFeatureExtractedReleaseGateError(
            "targeted drill plan durable replay differs"
        )
    precommit_receipt = store.persist(
        object_kind="extracted-execution-precommit",
        object_digest=canonical_precommit.record_digest,
        data=canonical_precommit.to_data(),
    )
    decoded_precommit = PanelFeatureExtractedExecutionPrecommit.from_data(
        store.verify(
            precommit_receipt,
            expected_data=canonical_precommit.to_data(),
        )
    )
    if decoded_precommit != canonical_precommit:
        raise PanelFeatureExtractedReleaseGateError(
            "extracted execution precommit durable replay differs"
        )
    successor = canonical_predecessor.record(
        phase=PANEL_FEATURE_EXTRACTED_EXPOSURE_PHASE,
        actor=canonical_precommit.exposure_actor,
        purpose=canonical_precommit.exposure_purpose,
        task_ids=canonical_precommit.selected_task_ids,
        source=canonical_precommit.exposure_source,
        observed_at=canonical_precommit.exposure_observed_at,
        known_task_ids=canonical_precommit.selected_task_ids,
        require_unseen=True,
    )
    if (
        len(successor.events) != len(canonical_predecessor.events) + 1
        or successor.events[:-1] != canonical_predecessor.events
        or successor.events[-1].task_ids != canonical_precommit.selected_task_ids
        or successor.events[-1].panel_ids
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "exposure successor is not the exact one-event transition"
        )
    exposure_receipt = store.persist(
        object_kind="extracted-exposure-successor",
        object_digest=successor.digest,
        data=successor.to_dict(),
    )
    decoded_successor = ExposureLedger.from_dict(
        store.verify(exposure_receipt, expected_data=successor.to_dict())
    )
    if decoded_successor != successor:
        raise PanelFeatureExtractedReleaseGateError(
            "exposure successor durable replay differs"
        )
    values: dict[str, object] = {
        "targeted_drill_plan_digest": canonical_plan.record_digest,
        "execution_precommit_digest": canonical_precommit.record_digest,
        "exposure_predecessor_digest": canonical_predecessor.digest,
        "exposure_successor_digest": successor.digest,
        "exposure_event_digest": successor.events[-1].digest,
        "release_descriptor_digest": canonical_precommit.release_descriptor_digest,
        "split_source_digest": canonical_precommit.split_source_digest,
        "task_inventory_digest": canonical_precommit.task_inventory_digest,
        "extracted_corpus_manifest_digest": (
            canonical_precommit.extracted_corpus_manifest_digest
        ),
        "extracted_archive_record_digest": (
            canonical_precommit.extracted_archive_record_digest
        ),
        "selected_task_ids": canonical_precommit.selected_task_ids,
        "authorized_support_panel_ids": (
            canonical_precommit.authorized_support_panel_ids
        ),
        "sealed_query_panel_ids": canonical_precommit.sealed_query_panel_ids,
        "plan_store_receipt_digest": plan_receipt.record_digest,
        "precommit_store_receipt_digest": precommit_receipt.record_digest,
        "exposure_store_receipt_digest": exposure_receipt.record_digest,
    }
    provisional = object.__new__(PanelFeatureExtractedReleaseAuthorization)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    authorization = PanelFeatureExtractedReleaseAuthorization(
        **values,  # type: ignore[arg-type]
        record_digest=_address(_authorization_content(provisional)),
    )
    authorization_receipt = store.persist(
        object_kind="extracted-release-authorization",
        object_digest=authorization.record_digest,
        data=authorization.to_data(),
    )
    reloaded_authorization = PanelFeatureExtractedReleaseAuthorization.from_data(
        store.verify(
            authorization_receipt,
            expected_data=authorization.to_data(),
        )
    )
    if reloaded_authorization != authorization:
        raise PanelFeatureExtractedReleaseGateError(
            "release authorization durable replay differs"
        )
    return PreparedPanelFeatureExtractedRelease(
        store,
        canonical_plan,
        canonical_precommit,
        canonical_predecessor,
        successor,
        authorization,
        plan_receipt,
        precommit_receipt,
        exposure_receipt,
        authorization_receipt,
    )


def verify_prepared_panel_feature_extracted_release(
    prepared: PreparedPanelFeatureExtractedRelease,
) -> None:
    if type(prepared) is not PreparedPanelFeatureExtractedRelease:
        raise TypeError(
            "prepared must be exact PreparedPanelFeatureExtractedRelease"
        )
    canonical_plan = _canonical_plan(prepared.plan)
    canonical_precommit = PanelFeatureExtractedExecutionPrecommit.from_data(
        prepared.precommit.to_data()
    )
    predecessor = _canonical_ledger(prepared.predecessor, "exposure predecessor")
    successor = _canonical_ledger(prepared.successor, "exposure successor")
    authorization = PanelFeatureExtractedReleaseAuthorization.from_data(
        prepared.authorization.to_data()
    )
    store = prepared.store
    store.verify(prepared.plan_receipt, expected_data=canonical_plan.to_data())
    store.verify(
        prepared.precommit_receipt,
        expected_data=canonical_precommit.to_data(),
    )
    store.verify(prepared.exposure_receipt, expected_data=successor.to_dict())
    store.verify(
        prepared.authorization_receipt,
        expected_data=authorization.to_data(),
    )
    if (
        len(successor.events) != len(predecessor.events) + 1
        or successor.events[:-1] != predecessor.events
        or successor.events[-1].task_ids != canonical_precommit.selected_task_ids
        or successor.events[-1].panel_ids
        or canonical_precommit.targeted_drill_plan_digest
        != canonical_plan.record_digest
        or canonical_precommit.exposure_predecessor_digest != predecessor.digest
        or authorization.execution_precommit_digest
        != canonical_precommit.record_digest
        or authorization.exposure_predecessor_digest != predecessor.digest
        or authorization.exposure_successor_digest != successor.digest
        or authorization.exposure_event_digest != successor.events[-1].digest
        or authorization.plan_store_receipt_digest
        != prepared.plan_receipt.record_digest
        or authorization.precommit_store_receipt_digest
        != prepared.precommit_receipt.record_digest
        or authorization.exposure_store_receipt_digest
        != prepared.exposure_receipt.record_digest
        or prepared.plan_receipt.object_kind != "targeted-drill-plan"
        or prepared.precommit_receipt.object_kind
        != "extracted-execution-precommit"
        or prepared.exposure_receipt.object_kind
        != "extracted-exposure-successor"
        or prepared.authorization_receipt.object_kind
        != "extracted-release-authorization"
        or prepared.plan_receipt.object_digest != canonical_plan.record_digest
        or prepared.precommit_receipt.object_digest
        != canonical_precommit.record_digest
        or prepared.exposure_receipt.object_digest != successor.digest
        or prepared.authorization_receipt.object_digest
        != authorization.record_digest
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "prepared extracted release cold replay differs"
        )


def _verify_release_archive(
    prepared: PreparedPanelFeatureExtractedRelease,
    archive: OfficialExtractedPanelArchive,
) -> None:
    if type(archive) is not OfficialExtractedPanelArchive:
        raise TypeError("archive must be exact OfficialExtractedPanelArchive")
    if (
        archive.record_digest
        != prepared.authorization.extracted_archive_record_digest
        or archive.record_digest
        != prepared.precommit.extracted_archive_record_digest
        or archive.release_descriptor_digest
        != prepared.authorization.release_descriptor_digest
        or archive.corpus_manifest_digest
        != prepared.authorization.extracted_corpus_manifest_digest
        or archive.layout != prepared.precommit.extracted_layout
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "release archive differs from extracted-tree authorization"
        )


def _persist_released_panel(
    *,
    prepared: PreparedPanelFeatureExtractedRelease,
    released: ReleasedOfficialExtractedPanel,
    object_kind: str,
) -> tuple[ReleasedOfficialExtractedPanel, ObjectBongardWriteOnceReceipt]:
    receipt = prepared.store.persist(
        object_kind=object_kind,
        object_digest=released.record_digest,
        data=released.to_data(),
    )
    restored = ReleasedOfficialExtractedPanel.from_data(
        prepared.store.verify(receipt, expected_data=released.to_data())
    )
    if restored != released:
        raise PanelFeatureExtractedReleaseGateError(
            "released extracted panel durable replay differs"
        )
    return released, receipt


def release_panel_feature_extracted_support_panel(
    *,
    prepared: PreparedPanelFeatureExtractedRelease,
    archive: OfficialExtractedPanelArchive,
    panel_id: str,
) -> tuple[ReleasedOfficialExtractedPanel, ObjectBongardWriteOnceReceipt]:
    verify_prepared_panel_feature_extracted_release(prepared)
    _verify_release_archive(prepared, archive)
    if (
        type(panel_id) is not str
        or panel_id not in prepared.authorization.authorized_support_panel_ids
        or panel_id in prepared.authorization.sealed_query_panel_ids
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "support panel release is not authorized"
        )
    released = ReleasedOfficialExtractedPanel.release(
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
    return _persist_released_panel(
        prepared=prepared,
        released=released,
        object_kind="released-extracted-support-panel",
    )


def _task_for_panel(
    plan: PanelFeatureTargetedDrillPlan,
    panel_id: str,
) -> ObjectBongardTaskPlan:
    matches = tuple(
        task
        for task in plan.tasks
        if panel_id
        in (
            *task.side_0_support_panel_ids,
            *task.side_1_support_panel_ids,
            task.side_0_query_panel_id,
            task.side_1_query_panel_id,
        )
    )
    if len(matches) != 1:
        raise PanelFeatureExtractedReleaseGateError(
            "panel is outside the targeted drill plan"
        )
    return matches[0]


def _canonical_protocol_data(value: object, label: str) -> dict[str, Any]:
    try:
        data = dict(value.to_data())  # type: ignore[union-attr]
        record_digest = value.record_digest  # type: ignore[union-attr]
    except Exception as exc:
        raise PanelFeatureExtractedReleaseGateError(
            f"{label} does not expose canonical data"
        ) from exc
    if data.get("record_digest") != record_digest:
        raise PanelFeatureExtractedReleaseGateError(
            f"{label} record digest field differs"
        )
    content = {key: item for key, item in data.items() if key != "record_digest"}
    try:
        canonical = json.loads(canonical_json(data))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PanelFeatureExtractedReleaseGateError(
            f"{label} is not canonical JSON"
        ) from exc
    if record_digest != _address(content) or canonical != data:
        raise PanelFeatureExtractedReleaseGateError(f"{label} is not canonical")
    return data


def _python_predicate_policy(data: Mapping[str, Any], label: str) -> None:
    if (
        data.get("predicate_authority_id") != PYTHON_PREDICATE_AUTHORITY_ID
        or data.get("implementation_language") != "python"
        or data.get("engineering_only") is not True
        or data.get("uncalibrated") is not True
        or data.get("scientific_evidence") is not False
        or data.get("benchmark_authoritative") is not False
        or data.get("positive_only") is not True
        or data.get("negation_allowed") is not False
        or data.get("polarity_flip_allowed") is not False
        or data.get("arbitrary_predicate_code_allowed") is not False
        or data.get("lean_present") is not False
        or data.get("lean_required") is not False
        or data.get("lean_affects_identity_selection_decision_or_replay")
        is not False
    ):
        raise PanelFeatureExtractedReleaseGateError(
            f"{label} is not an exact closed Python predicate record"
        )


def _one_positive_python_predicate_policy(
    data: Mapping[str, Any], label: str
) -> None:
    """Require the successor's asymmetric, single-formula authority contract."""

    _python_predicate_policy(data, label)
    if (
        data.get("one_positive_formula_only") is not True
        or data.get("negative_formula_present") is not False
        or data.get("complement_allowed") is not False
        or data.get("primary_version_space_only_gate") is not True
        or data.get("opposite_version_space_diagnostic_only") is not True
    ):
        raise PanelFeatureExtractedReleaseGateError(
            f"{label} is not an exact one-positive Python predicate record"
        )


def _validate_successor_support_custody(
    bound_inventory: object,
    *,
    task: ObjectBongardTaskPlan,
    archive: OfficialExtractedPanelArchive,
) -> object:
    """Join successor evidence bytes to the authenticated extracted manifest."""

    # Local imports keep the release boundary below the successor runner and
    # avoid turning its gate import into a cycle.
    from bongard.panel_feature_evidence_bundle import PanelFeatureEvidencePhase
    from bongard.panel_feature_task_bound_inventory import (
        TaskBoundClosedCatalogInventory,
        cold_replay_task_bound_closed_catalog_inventory,
    )

    if type(bound_inventory) is not TaskBoundClosedCatalogInventory:
        raise TypeError(
            "successor freeze must contain exact "
            "TaskBoundClosedCatalogInventory"
        )
    try:
        bound = cold_replay_task_bound_closed_catalog_inventory(
            bound_inventory,
            expected_artifact_address=bound_inventory.artifact_address,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PanelFeatureExtractedReleaseGateError(
            "successor support evidence/inventory replay differs"
        ) from exc
    support = bound.evidence_bundle.panels_for_phase(
        PanelFeatureEvidencePhase.SUPPORT
    )
    expected_ids = task.side_0_support_panel_ids + task.side_1_support_panel_ids
    if (
        bound.task_plan != task
        or tuple(item.panel_id for item in support) != expected_ids
        or bound.sealed_query_panel_ids
        != (task.side_0_query_panel_id, task.side_1_query_panel_id)
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "successor support evidence does not bind the released task roles"
        )
    for panel in support:
        manifest_row = archive.panel_by_id.get(panel.panel_id)
        if (
            manifest_row is None
            or manifest_row.sha256 != "sha256:" + panel.panel_png_digest
            or manifest_row.size_bytes != len(panel.panel_png)
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "successor support evidence differs from the authenticated "
                "extracted manifest"
            )
    return bound


def _validate_legacy_freeze_bindings(
    freeze: ObjectBongardTaskFreezeProtocol,
    *,
    task: ObjectBongardTaskPlan,
    prepared: PreparedPanelFeatureExtractedRelease,
) -> dict[str, Any]:
    # Imported locally so the low-level release boundary does not create a
    # module cycle.  Query release is deliberately narrower than the generic
    # object protocol: only this pipeline's fully replayable Python IR is an
    # admissible decision freeze.
    from bongard.panel_feature_task_runner import PanelFeatureTaskFreeze

    if type(freeze) is not PanelFeatureTaskFreeze:
        raise TypeError("task freeze must be exact PanelFeatureTaskFreeze")
    try:
        replayed = PanelFeatureTaskFreeze.from_data(freeze.to_data())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PanelFeatureExtractedReleaseGateError(
            "task freeze Python IR replay differs"
        ) from exc
    if replayed != freeze:
        raise PanelFeatureExtractedReleaseGateError(
            "task freeze Python IR replay differs"
        )
    data = _canonical_protocol_data(freeze, "task freeze")
    _python_predicate_policy(data, "task freeze")
    _require_address(freeze.task_plan_digest, "task plan digest")
    _require_address(freeze.execution_precommit_digest, "execution precommit digest")
    for name in (
        "version_space_digest",
        "support_version_space_digest",
        "rank_response_digest",
        "selected_predicate_digest",
    ):
        _require_raw_digest(getattr(freeze, name), name)
    if (
        freeze.task_id != task.task_id
        or freeze.task_plan_digest != task.record_digest
        or freeze.execution_precommit_digest != prepared.precommit.record_digest
        or freeze.support_version_space_digest != freeze.version_space_digest
        or data.get("sealed_query_panel_ids")
        != [task.side_0_query_panel_id, task.side_1_query_panel_id]
        or data.get("query_bytes_included") is not False
        or data.get("query_observations_included") is not False
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "task freeze bindings differ"
        )
    return data


def _validate_successor_freeze_bindings(
    freeze: object,
    *,
    task: ObjectBongardTaskPlan,
    prepared: PreparedPanelFeatureExtractedRelease,
    archive: OfficialExtractedPanelArchive,
) -> dict[str, Any]:
    from bongard.panel_feature_primary_task_runner import (
        PrimaryFormulaRankJournalTerminal,
        PrimaryFormulaSupportStatus,
        PrimaryFormulaTaskFreeze,
        verify_primary_formula_task_freeze,
    )
    from bongard.panel_feature_predicate import AllOf
    from bongard.panel_soft_ontology import NativeOrientation

    if type(freeze) is not PrimaryFormulaTaskFreeze:
        raise TypeError("task freeze must be exact PrimaryFormulaTaskFreeze")
    try:
        replayed = verify_primary_formula_task_freeze(
            freeze,
            expected_record_digest=freeze.record_digest,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PanelFeatureExtractedReleaseGateError(
            "successor task freeze replay differs"
        ) from exc
    data = _canonical_protocol_data(replayed, "successor task freeze")
    _one_positive_python_predicate_policy(data, "successor task freeze")
    bound = _validate_successor_support_custody(
        replayed.support_phase.task_bound_inventory,
        task=task,
        archive=archive,
    )
    try:
        formula = replayed.resolve_selected_all_of()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PanelFeatureExtractedReleaseGateError(
            "successor selected formula replay differs"
        ) from exc
    space = bound.inventory.primary_version_space  # type: ignore[union-attr]
    phase = replayed.support_phase
    query_ids = (task.side_0_query_panel_id, task.side_1_query_panel_id)
    if (
        type(replayed.execution_precommit)
        is not PanelFeatureExtractedExecutionPrecommit
        or replayed.execution_precommit_kind
        != "panel_feature_extracted_execution_precommit_v1"
        or replayed.execution_precommit != prepared.precommit
        or replayed.execution_precommit_digest
        != prepared.precommit.record_digest
        or replayed.task_id != task.task_id
        or replayed.task_plan_digest != task.record_digest
        or replayed.version_space_digest != space.version_space_digest
        or replayed.support_version_space_digest
        != replayed.version_space_digest
        or type(formula) is not AllOf
        or formula != replayed.selected_formula
        or formula.formula_digest != replayed.selected_predicate_digest
        or formula.formula_digest not in space.survivor_formula_digests
        or formula.native_orientation is not NativeOrientation.SIDE0_POSITIVE
        or replayed.sealed_query_panel_ids != query_ids
        or data.get("sealed_query_panel_ids") != list(query_ids)
        or data.get("query_bytes_included") is not False
        or data.get("query_observations_included") is not False
        or data.get(
            "query_release_authorized_only_after_exact_durable_commit"
        )
        is not True
        or phase.gap is not None
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "successor task/precommit/query/formula bindings differ"
        )
    survivor_count = len(space.survivor_formulas)
    if phase.status is PrimaryFormulaSupportStatus.UNIQUE_PRIMARY_SURVIVOR:
        if (
            survivor_count != 1
            or replayed.selection_mode != "unique_primary_support_survivor"
            or replayed.rank_artifact is not None
            or replayed.rank_journal_terminal is not None
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "unique successor freeze contains rank authority"
            )
    elif phase.status is PrimaryFormulaSupportStatus.RANK_REQUIRED:
        terminal = replayed.rank_journal_terminal
        if (
            survivor_count <= 1
            or replayed.selection_mode
            != "verified_rank_with_durable_journal_terminal"
            or terminal is None
            or type(terminal) is not PrimaryFormulaRankJournalTerminal
            or PrimaryFormulaRankJournalTerminal.from_data(terminal.to_data())
            != terminal
            or replayed.rank_artifact is None
            or terminal.rank_artifact != replayed.rank_artifact
            or terminal.authorization_digest
            != prepared.authorization.record_digest
            or terminal.execution_precommit_digest
            != prepared.precommit.record_digest
            or terminal.task_id != task.task_id
        ):
            raise PanelFeatureExtractedReleaseGateError(
                "successor rank terminal does not bind this release"
            )
    else:
        raise PanelFeatureExtractedReleaseGateError(
            "typed primary support gap cannot authorize query release"
        )
    return data


def _validate_freeze_bindings(
    freeze: ObjectBongardTaskFreezeProtocol,
    *,
    task: ObjectBongardTaskPlan,
    prepared: PreparedPanelFeatureExtractedRelease,
    archive: OfficialExtractedPanelArchive,
) -> dict[str, Any]:
    # Both imports are local: the successor runner imports this gate for its
    # exact extracted-precommit branch.
    from bongard.panel_feature_primary_task_runner import PrimaryFormulaTaskFreeze
    from bongard.panel_feature_task_runner import PanelFeatureTaskFreeze

    if type(freeze) is PanelFeatureTaskFreeze:
        return _validate_legacy_freeze_bindings(
            freeze,
            task=task,
            prepared=prepared,
        )
    if type(freeze) is PrimaryFormulaTaskFreeze:
        return _validate_successor_freeze_bindings(
            freeze,
            task=task,
            prepared=prepared,
            archive=archive,
        )
    raise TypeError(
        "task freeze must be exact PanelFeatureTaskFreeze or "
        "PrimaryFormulaTaskFreeze"
    )


def _validate_legacy_commit_bindings(
    commit: ObjectBongardTaskCommitProtocol,
    *,
    freeze: ObjectBongardTaskFreezeProtocol,
    freeze_receipt: ObjectBongardWriteOnceReceipt,
) -> dict[str, Any]:
    from bongard.panel_feature_task_runner import PanelFeatureTaskFreezeCommit

    if type(commit) is not PanelFeatureTaskFreezeCommit:
        raise TypeError("task commit must be exact PanelFeatureTaskFreezeCommit")
    try:
        replayed = PanelFeatureTaskFreezeCommit.from_data(commit.to_data())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PanelFeatureExtractedReleaseGateError(
            "task commit Python IR replay differs"
        ) from exc
    if replayed != commit:
        raise PanelFeatureExtractedReleaseGateError(
            "task commit Python IR replay differs"
        )
    data = _canonical_protocol_data(commit, "task commit")
    _python_predicate_policy(data, "task commit")
    if (
        data.get("durably_persisted_and_reloaded_before_query_release") is not True
        or commit.task_id != freeze.task_id
        or commit.task_plan_digest != freeze.task_plan_digest
        or commit.execution_precommit_digest != freeze.execution_precommit_digest
        or commit.version_space_digest != freeze.version_space_digest
        or commit.support_version_space_digest
        != freeze.support_version_space_digest
        or commit.rank_response_digest != freeze.rank_response_digest
        or commit.selected_predicate_digest != freeze.selected_predicate_digest
        or commit.task_freeze_digest != freeze.record_digest
        or commit.exact_freeze_payload_digest != freeze_receipt.payload_digest
        or commit.task_freeze_store_receipt_digest != freeze_receipt.record_digest
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "task decision commit does not bind the exact durable Python freeze"
        )
    return data


def _validate_successor_commit_bindings(
    commit: object,
    *,
    freeze: object,
    freeze_receipt: ObjectBongardWriteOnceReceipt,
) -> dict[str, Any]:
    from bongard.panel_feature_primary_task_runner import (
        PrimaryFormulaTaskFreeze,
        PrimaryFormulaTaskFreezeCommit,
        verify_primary_formula_task_commit,
    )

    if type(freeze) is not PrimaryFormulaTaskFreeze:
        raise TypeError("successor commit needs exact PrimaryFormulaTaskFreeze")
    if type(commit) is not PrimaryFormulaTaskFreezeCommit:
        raise TypeError("task commit must be exact PrimaryFormulaTaskFreezeCommit")
    try:
        replayed = verify_primary_formula_task_commit(
            commit,
            expected_record_digest=commit.record_digest,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise PanelFeatureExtractedReleaseGateError(
            "successor task commit replay differs"
        ) from exc
    data = _canonical_protocol_data(replayed, "successor task commit")
    _one_positive_python_predicate_policy(data, "successor task commit")
    if (
        replayed.task_freeze != freeze
        or replayed.task_freeze_store_receipt != freeze_receipt
        or data.get("durably_persisted_and_reloaded_before_query_release")
        is not True
        or data.get("exact_canonical_freeze_bytes_bound") is not True
        or replayed.task_id != freeze.task_id
        or replayed.task_plan_digest != freeze.task_plan_digest
        or replayed.execution_precommit_digest
        != freeze.execution_precommit_digest
        or replayed.version_space_digest != freeze.version_space_digest
        or replayed.support_version_space_digest
        != freeze.support_version_space_digest
        or replayed.rank_response_digest != freeze.rank_response_digest
        or replayed.selected_predicate_digest
        != freeze.selected_predicate_digest
        or replayed.task_freeze_digest != freeze.record_digest
        or replayed.exact_freeze_payload_digest != freeze_receipt.payload_digest
        or replayed.task_freeze_store_receipt_digest
        != freeze_receipt.record_digest
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "successor decision commit does not bind the exact durable freeze"
        )
    return data


def _validate_commit_bindings(
    commit: ObjectBongardTaskCommitProtocol,
    *,
    freeze: ObjectBongardTaskFreezeProtocol,
    freeze_receipt: ObjectBongardWriteOnceReceipt,
) -> dict[str, Any]:
    from bongard.panel_feature_primary_task_runner import (
        PrimaryFormulaTaskFreeze,
        PrimaryFormulaTaskFreezeCommit,
    )
    from bongard.panel_feature_task_runner import (
        PanelFeatureTaskFreeze,
        PanelFeatureTaskFreezeCommit,
    )

    if type(freeze) is PanelFeatureTaskFreeze:
        if type(commit) is not PanelFeatureTaskFreezeCommit:
            raise TypeError(
                "legacy freeze requires exact PanelFeatureTaskFreezeCommit"
            )
        return _validate_legacy_commit_bindings(
            commit,
            freeze=freeze,
            freeze_receipt=freeze_receipt,
        )
    if type(freeze) is PrimaryFormulaTaskFreeze:
        if type(commit) is not PrimaryFormulaTaskFreezeCommit:
            raise TypeError(
                "successor freeze requires exact PrimaryFormulaTaskFreezeCommit"
            )
        return _validate_successor_commit_bindings(
            commit,
            freeze=freeze,
            freeze_receipt=freeze_receipt,
        )
    raise TypeError(
        "commit binding needs an exact legacy or successor task freeze"
    )


def release_panel_feature_extracted_query_panel(
    *,
    prepared: PreparedPanelFeatureExtractedRelease,
    archive: OfficialExtractedPanelArchive,
    panel_id: str,
    task_freeze: ObjectBongardTaskFreezeProtocol,
    task_commit: ObjectBongardTaskCommitProtocol,
    task_freeze_receipt: ObjectBongardWriteOnceReceipt,
    task_commit_receipt: ObjectBongardWriteOnceReceipt,
) -> tuple[ReleasedOfficialExtractedPanel, ObjectBongardWriteOnceReceipt]:
    verify_prepared_panel_feature_extracted_release(prepared)
    _verify_release_archive(prepared, archive)
    task = _task_for_panel(prepared.plan, panel_id)
    if (
        panel_id not in prepared.authorization.sealed_query_panel_ids
        or panel_id
        not in (task.side_0_query_panel_id, task.side_1_query_panel_id)
        or panel_id in prepared.authorization.authorized_support_panel_ids
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "query panel is not the task's preregistered sealed query"
        )
    freeze_data = _validate_freeze_bindings(
        task_freeze,
        task=task,
        prepared=prepared,
        archive=archive,
    )
    commit_data = _validate_commit_bindings(
        task_commit,
        freeze=task_freeze,
        freeze_receipt=task_freeze_receipt,
    )
    prepared.store.verify(task_freeze_receipt, expected_data=freeze_data)
    prepared.store.verify(task_commit_receipt, expected_data=commit_data)
    if (
        task_freeze_receipt.object_kind != "task-freeze"
        or task_freeze_receipt.object_digest != task_freeze.record_digest
        or task_commit_receipt.object_kind != "task-decision-commit"
        or task_commit_receipt.object_digest != task_commit.record_digest
    ):
        raise PanelFeatureExtractedReleaseGateError(
            "task freeze/commit store receipts differ"
        )
    released = ReleasedOfficialExtractedPanel.release(
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
    return _persist_released_panel(
        prepared=prepared,
        released=released,
        object_kind="released-extracted-query-panel",
    )


__all__ = (
    "PANEL_FEATURE_EXTRACTED_AUTHORIZATION_SCHEMA",
    "PANEL_FEATURE_EXTRACTED_EXPOSURE_PHASE",
    "PANEL_FEATURE_EXTRACTED_PRECOMMIT_SCHEMA",
    "PanelFeatureExtractedExecutionPrecommit",
    "PanelFeatureExtractedReleaseAuthorization",
    "PanelFeatureExtractedReleaseGateError",
    "PreparedPanelFeatureExtractedRelease",
    "create_panel_feature_extracted_execution_precommit",
    "panel_feature_extracted_release_gate_source_digest",
    "persist_object_bongard_task_commit",
    "persist_object_bongard_task_freeze",
    "prepare_panel_feature_extracted_release",
    "release_panel_feature_extracted_query_panel",
    "release_panel_feature_extracted_support_panel",
    "verify_prepared_panel_feature_extracted_release",
)
