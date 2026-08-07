"""Support-only closed-language gate for the already exposed A3 task.

The command deliberately has no proposal, query, held-out, test, exposure, or
Lean input.  It accepts exactly the checked-in attempt-three forensics record,
freezes the complete positive Python predicate union, and only then reads the
twelve support PNGs named by that record.  The output is a compact,
content-addressed coverage result rather than a benchmark score.

Run, when explicitly authorized, with::

    python -m bongard.a3_closed_language_gate \
      --corpus-root downloads/ShapeBongard_V2_full/ShapeBongard_V2 \
      --output-dir downloads/a3_closed_language_gate_v1

Importing this module and constructing the argument parser do not enumerate
the library or read any PNG.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from bongard.artifacts import canonical_digest, canonical_json
from bongard.closed_visual_predicates import (
    ClosedPredicateKind,
    FrozenClosedPredicateLibrary,
    SupportExpressibilityResult,
    closed_visual_predicate_evaluator_digest,
    closed_visual_predicate_source_digest,
    freeze_complete_closed_predicate_library,
    support_only_expressibility_oracle,
)
from bongard.composite_visual_packet import (
    ExactPanelWitnessPacket,
    composite_visual_packet_source_digest,
    exact_panel_witness_extractor_digest,
    extract_exact_panel_witness_packet,
    verify_exact_panel_witness_packet,
)
from bongard.corpus import PNG_SIGNATURE


GATE_SCHEMA = "gkm.bongard-a3-closed-language-gate-result.v2"
GATE_ALGORITHM_ID = "bongard.a3-closed-language-gate/support-only-v2"
SUPPORT_MAPPING_SCHEMA = "gkm.bongard-a3-canonical-support-mapping.v1"
CANONICAL_FORENSICS_SCHEMA = (
    "gkm.bongard-atomic-smoke-attempt3-relational-forensics.v1"
)
CANONICAL_FORENSICS_RECORD_DIGEST = (
    "0487edf805fda6de40ecfc42add1d8bf95e435e0f6912f6e2fd8d2a25e89eb2a"
)
CANONICAL_FORENSICS_FILE_SHA256 = (
    "a674869ce98575733b86c9a6ba9e2c32a6f5784a7134d59d8b1cd3db651ab46c"
)
CANONICAL_SUPPORT_MAPPING_DIGEST = (
    "d91190a336e7eb0b3725ba51b309dcedd6cb5f9daee2d523788fd8b9cae81834"
)
CANONICAL_HELDOUT_AUTHORITY_DIGEST = (
    "bc0b281d9060c5a303868ce66f347d5118b360cca7c7184d33026aaeb6f2baa7"
)
CANONICAL_TASK_ID = "bd_mismatch_triangle_rec6_0000"
CANONICAL_SPLIT = "train"
CANONICAL_CORPUS_MANIFEST_DIGEST = (
    "sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138"
)
CANONICAL_SPLIT_SOURCE_DIGEST = (
    "sha256:ebb9cd474478e0776dff539951070db2c96b9b312c4b0b073689d20792ed7230"
)
CANONICAL_RECORD_PATH = (
    Path(__file__).with_name("data")
    / "atomic_smoke_attempt3_relational_forensics_v1.json"
)
COMPLETE_LIBRARY_MEMBER_COUNT = 65_678
COMPLETE_LIBRARY_KIND_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        ClosedPredicateKind.DIRECT_COUNTS.value: 64_400,
        ClosedPredicateKind.RELATIONAL.value: 1_260,
        ClosedPredicateKind.SYMMETRY.value: 18,
    }
)
_EXPECTED_POSITIVE_INDICES = (0, 1, 2, 3, 5, 6)
_EXPECTED_NEGATIVE_INDICES = (0, 1, 2, 3, 4, 6)
_EXPECTED_HELDOUT = ((False, 5), (True, 4))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_CONTROL_BYTES = 256 * 1024
_MAX_PNG_BYTES = 32 * 1024 * 1024
_MAX_RESULT_BYTES = 64 * 1024
_GATE_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class A3ClosedLanguageGateError(RuntimeError):
    """The support-only gate could not establish its fail-closed boundary."""


def _require_exact_fields(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise A3ClosedLanguageGateError(
            f"{label} fields differ from the canonical schema"
        )
    if any(not isinstance(key, str) for key in value):
        raise A3ClosedLanguageGateError(f"{label} keys must be strings")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise A3ClosedLanguageGateError(f"{label} must be a lowercase SHA-256")
    return value


def _require_address(value: object, label: str) -> str:
    if not isinstance(value, str) or _ADDRESS.fullmatch(value) is None:
        raise A3ClosedLanguageGateError(
            f"{label} must be a lowercase sha256: content address"
        )
    return value


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise A3ClosedLanguageGateError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _absolute_lexical(path: str | Path) -> Path:
    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts[1:]
    ):
        raise A3ClosedLanguageGateError("path is not canonical absolute lexical form")
    return candidate


def _open_absolute_no_symlinks(path: str | Path, *, directory: bool) -> int:
    """Open every component with ``O_NOFOLLOW`` using directory descriptors."""

    candidate = _absolute_lexical(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise A3ClosedLanguageGateError("platform lacks O_NOFOLLOW")
    descriptor = os.open("/", flags | directory_flag)
    try:
        components = candidate.parts[1:]
        if not components:
            if not directory:
                raise A3ClosedLanguageGateError("root is not an input file")
            return descriptor
        for index, component in enumerate(components):
            final = index == len(components) - 1
            component_flags = flags | nofollow
            if not final or directory:
                component_flags |= directory_flag
            next_descriptor = os.open(
                component, component_flags, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except A3ClosedLanguageGateError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise A3ClosedLanguageGateError(
            f"cannot open no-follow path {candidate}"
        ) from exc


def _descriptor_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )


def _stable_read_no_symlinks(path: str | Path, *, maximum: int) -> bytes:
    descriptor = _open_absolute_no_symlinks(path, directory=False)
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise A3ClosedLanguageGateError(
                "input must be a nonempty bounded regular file"
            )
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise A3ClosedLanguageGateError("input ended before its recorded size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise A3ClosedLanguageGateError("input grew while being read")
        after = os.fstat(descriptor)
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise A3ClosedLanguageGateError("input changed while being read")
    except OSError as exc:
        raise A3ClosedLanguageGateError(f"cannot read {path}") from exc
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise A3ClosedLanguageGateError("input byte count changed while being read")
    return payload


def _canonical_json_file(path: str | Path) -> tuple[dict[str, Any], bytes]:
    payload = _stable_read_no_symlinks(path, maximum=_MAX_CONTROL_BYTES)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_json_object_no_duplicates,
        )
    except A3ClosedLanguageGateError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise A3ClosedLanguageGateError(
            "forensics authority is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise A3ClosedLanguageGateError("forensics authority must be a JSON object")
    if canonical_json(value) + b"\n" != payload:
        raise A3ClosedLanguageGateError(
            "forensics authority bytes are not the sole canonical encoding"
        )
    return value, payload


def _safe_relative_png(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise A3ClosedLanguageGateError(f"{label} is not a POSIX relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.suffix.lower() != ".png"
    ):
        raise A3ClosedLanguageGateError(f"{label} is unsafe or noncanonical")
    return value


@dataclass(frozen=True, slots=True)
class SupportPanelAuthority:
    label: bool
    source_index: int
    relative_path: str
    png_sha256: str
    historical_loop_packet_digest: str

    def __post_init__(self) -> None:
        if type(self.label) is not bool:
            raise TypeError("support label must be Boolean")
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("support source index must be non-negative")
        _safe_relative_png(self.relative_path, "support relative_path")
        _require_sha256(self.png_sha256, "support PNG digest")
        _require_sha256(
            self.historical_loop_packet_digest,
            "historical loop packet digest",
        )

    def to_data(self) -> dict[str, object]:
        return {
            "label": self.label,
            "source_index": self.source_index,
            "relative_path": self.relative_path,
            "png_sha256": self.png_sha256,
            "historical_loop_packet_digest": self.historical_loop_packet_digest,
        }


@dataclass(frozen=True, slots=True)
class HeldoutPanelAuthority:
    label: bool
    source_index: int
    relative_path: str
    png_sha256: str

    def __post_init__(self) -> None:
        if type(self.label) is not bool:
            raise TypeError("held-out label must be Boolean")
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("held-out source index must be non-negative")
        _safe_relative_png(self.relative_path, "held-out relative_path")
        _require_sha256(self.png_sha256, "held-out PNG digest")

    def to_data(self) -> dict[str, object]:
        return {
            "label": self.label,
            "source_index": self.source_index,
            "relative_path": self.relative_path,
            "png_sha256": self.png_sha256,
        }


@dataclass(frozen=True, slots=True)
class A3SupportMapping:
    authority_record_digest: str
    authority_file_sha256: str
    task_id: str
    split: str
    corpus_manifest_digest: str
    split_source_digest: str
    supports: tuple[SupportPanelAuthority, ...]
    heldouts: tuple[HeldoutPanelAuthority, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.authority_record_digest, "authority record digest")
        _require_sha256(self.authority_file_sha256, "authority file digest")
        if not self.task_id or not self.split:
            raise ValueError("mapping task and split must be nonempty")
        _require_address(self.corpus_manifest_digest, "corpus manifest digest")
        _require_address(self.split_source_digest, "split source digest")
        if not isinstance(self.supports, tuple) or len(self.supports) != 12:
            raise ValueError("mapping must contain exactly twelve supports")
        if not isinstance(self.heldouts, tuple) or len(self.heldouts) != 2:
            raise ValueError("mapping must retain exactly two forbidden held-outs")
        if any(not isinstance(item, SupportPanelAuthority) for item in self.supports):
            raise TypeError("mapping supports must be typed authorities")
        if any(not isinstance(item, HeldoutPanelAuthority) for item in self.heldouts):
            raise TypeError("mapping held-outs must be typed authorities")
        if sum(item.label for item in self.supports) != 6:
            raise ValueError("mapping must contain six supports from each class")
        support_paths = tuple(item.relative_path for item in self.supports)
        support_digests = tuple(item.png_sha256 for item in self.supports)
        heldout_paths = tuple(item.relative_path for item in self.heldouts)
        heldout_digests = tuple(item.png_sha256 for item in self.heldouts)
        if len(set(support_paths)) != 12 or len(set(support_digests)) != 12:
            raise ValueError("support paths and PNG digests must be unique")
        if len(set(heldout_paths)) != 2 or len(set(heldout_digests)) != 2:
            raise ValueError("held-out paths and PNG digests must be unique")
        if set(support_paths) & set(heldout_paths) or set(support_digests) & set(
            heldout_digests
        ):
            raise ValueError("support and held-out authorities overlap")
        expected_order = tuple(
            sorted(
                self.supports,
                key=lambda item: (not item.label, item.source_index),
            )
        )
        if self.supports != expected_order:
            raise ValueError("supports must be positive-then-negative source order")

    @property
    def support_mapping_digest(self) -> str:
        return canonical_digest(
            {
                "schema": SUPPORT_MAPPING_SCHEMA,
                "authority_record_digest": self.authority_record_digest,
                "task_id": self.task_id,
                "split": self.split,
                "supports": [item.to_data() for item in self.supports],
            }
        )

    @property
    def heldout_authority_digest(self) -> str:
        return canonical_digest(
            {
                "schema": "gkm.bongard-a3-forbidden-heldout-authority.v1",
                "authority_record_digest": self.authority_record_digest,
                "heldouts": [item.to_data() for item in self.heldouts],
            }
        )


def _require_canonical_mapping(mapping: A3SupportMapping) -> None:
    if (
        mapping.authority_record_digest != CANONICAL_FORENSICS_RECORD_DIGEST
        or mapping.authority_file_sha256 != CANONICAL_FORENSICS_FILE_SHA256
        or mapping.task_id != CANONICAL_TASK_ID
        or mapping.split != CANONICAL_SPLIT
        or mapping.corpus_manifest_digest != CANONICAL_CORPUS_MANIFEST_DIGEST
        or mapping.split_source_digest != CANONICAL_SPLIT_SOURCE_DIGEST
        or mapping.support_mapping_digest != CANONICAL_SUPPORT_MAPPING_DIGEST
        or mapping.heldout_authority_digest
        != CANONICAL_HELDOUT_AUTHORITY_DIGEST
    ):
        raise A3ClosedLanguageGateError(
            "only the pinned canonical A3 support mapping is authorized"
        )


def load_canonical_a3_support_mapping(
    record_path: str | Path = CANONICAL_RECORD_PATH,
) -> A3SupportMapping:
    """Load the one pinned canonical A3 record without touching corpus PNGs."""

    record, payload = _canonical_json_file(record_path)
    expected_top = frozenset(
        {
            "algorithms",
            "base_query",
            "base_query_digest",
            "claim_boundary",
            "interpretation",
            "journal_support_panel_sequence",
            "library",
            "panels",
            "record_digest",
            "schema",
            "source_binding",
            "summary",
            "support_selection",
        }
    )
    _require_exact_fields(record, expected_top, "forensics record")
    if record["schema"] != CANONICAL_FORENSICS_SCHEMA:
        raise A3ClosedLanguageGateError("wrong forensics authority schema")
    content = dict(record)
    declared = _require_sha256(
        content.pop("record_digest"), "forensics record digest"
    )
    if (
        declared != CANONICAL_FORENSICS_RECORD_DIGEST
        or canonical_digest(content) != declared
    ):
        raise A3ClosedLanguageGateError("forensics authority digest is not the pin")

    source = _require_exact_fields(
        record["source_binding"],
        frozenset(
            {
                "attempt_run_content_address",
                "corpus_manifest_digest",
                "label_to_archive_side",
                "split",
                "split_source_digest",
                "task_id",
            }
        ),
        "forensics source binding",
    )
    if (
        source["task_id"] != CANONICAL_TASK_ID
        or source["split"] != CANONICAL_SPLIT
        or source["corpus_manifest_digest"] != CANONICAL_CORPUS_MANIFEST_DIGEST
        or source["split_source_digest"] != CANONICAL_SPLIT_SOURCE_DIGEST
        or source["label_to_archive_side"] != {"false": "0", "true": "1"}
    ):
        raise A3ClosedLanguageGateError("forensics source binding drifted")
    _require_address(source["attempt_run_content_address"], "attempt run address")
    if record["claim_boundary"] != {
        "benchmark_claim_authorized": False,
        "new_pixels_opened_for_this_record": False,
        "official_test_authorized": False,
        "purpose": (
            "post-hoc relational forensics over the already exposed attempt-three "
            "train task"
        ),
        "replaces_historical_soft_atom_record": False,
    }:
        raise A3ClosedLanguageGateError("forensics claim boundary drifted")

    selection = _require_exact_fields(
        record["support_selection"],
        frozenset(
            {
                "heldout_negative_source_index",
                "heldout_positive_source_index",
                "negative_source_indices",
                "positive_source_indices",
                "proof",
            }
        ),
        "support selection",
    )
    if (
        selection["positive_source_indices"] != list(_EXPECTED_POSITIVE_INDICES)
        or selection["negative_source_indices"]
        != list(_EXPECTED_NEGATIVE_INDICES)
        or selection["heldout_positive_source_index"] != 4
        or selection["heldout_negative_source_index"] != 5
    ):
        raise A3ClosedLanguageGateError("canonical A3 support selection drifted")

    panels = record["panels"]
    if not isinstance(panels, list) or len(panels) != 14:
        raise A3ClosedLanguageGateError("forensics record must contain 14 panels")
    panel_fields = frozenset(
        {
            "base_disposition",
            "base_result_digest",
            "label",
            "packet_digest",
            "png_sha256",
            "relative_path",
            "role",
            "source_index",
        }
    )
    support_values: list[SupportPanelAuthority] = []
    heldout_values: list[HeldoutPanelAuthority] = []
    panel_by_digest: dict[str, Mapping[str, Any]] = {}
    seen_slots: set[tuple[bool, int]] = set()
    for raw in panels:
        panel = _require_exact_fields(raw, panel_fields, "forensics panel")
        label = panel["label"]
        index = panel["source_index"]
        if (
            type(label) is not bool
            or isinstance(index, bool)
            or not isinstance(index, int)
        ):
            raise A3ClosedLanguageGateError("panel label/index types drifted")
        if index not in range(7) or (label, index) in seen_slots:
            raise A3ClosedLanguageGateError("panel slot is repeated or out of range")
        seen_slots.add((label, index))
        side = "1" if label else "0"
        expected_path = f"bd/images/{CANONICAL_TASK_ID}/{side}/{index}.png"
        relative_path = _safe_relative_png(panel["relative_path"], "panel path")
        if relative_path != expected_path:
            raise A3ClosedLanguageGateError("panel path differs from canonical slot")
        png_digest = _require_sha256(panel["png_sha256"], "panel PNG digest")
        packet_digest = _require_sha256(
            panel["packet_digest"], "historical loop packet digest"
        )
        _require_sha256(panel["base_result_digest"], "base result digest")
        if panel["role"] not in {"support", "heldout"}:
            raise A3ClosedLanguageGateError("panel role is not support/heldout")
        if png_digest in panel_by_digest:
            raise A3ClosedLanguageGateError("panel PNG digest repeats")
        panel_by_digest[png_digest] = panel
        if panel["role"] == "support":
            support_values.append(
                SupportPanelAuthority(
                    label=label,
                    source_index=index,
                    relative_path=relative_path,
                    png_sha256=png_digest,
                    historical_loop_packet_digest=packet_digest,
                )
            )
        else:
            heldout_values.append(
                HeldoutPanelAuthority(
                    label=label,
                    source_index=index,
                    relative_path=relative_path,
                    png_sha256=png_digest,
                )
            )

    if seen_slots != {(label, index) for label in (False, True) for index in range(7)}:
        raise A3ClosedLanguageGateError("forensics panel inventory is incomplete")
    if tuple(sorted((item.label, item.source_index) for item in heldout_values)) != (
        _EXPECTED_HELDOUT
    ):
        raise A3ClosedLanguageGateError("held-out slots differ from the authority")
    support_values.sort(key=lambda item: (not item.label, item.source_index))
    if tuple(item.source_index for item in support_values if item.label) != (
        _EXPECTED_POSITIVE_INDICES
    ) or tuple(item.source_index for item in support_values if not item.label) != (
        _EXPECTED_NEGATIVE_INDICES
    ):
        raise A3ClosedLanguageGateError("support slots differ from the authority")

    journal = record["journal_support_panel_sequence"]
    if not isinstance(journal, list) or len(journal) != 12:
        raise A3ClosedLanguageGateError("journal support sequence must contain 12 rows")
    journal_digests: list[str] = []
    for offset, raw in enumerate(journal):
        item = _require_exact_fields(
            raw,
            frozenset({"label", "panel_id", "png_sha256", "source_index"}),
            "journal support row",
        )
        digest = _require_sha256(item["png_sha256"], "journal PNG digest")
        panel = panel_by_digest.get(digest)
        if (
            item["panel_id"] != f"support-panel-{offset:02d}"
            or panel is None
            or panel["role"] != "support"
            or item["label"] is not panel["label"]
            or item["source_index"] != panel["source_index"]
        ):
            raise A3ClosedLanguageGateError("journal support mapping drifted")
        journal_digests.append(digest)
    if len(set(journal_digests)) != 12 or set(journal_digests) != {
        item.png_sha256 for item in support_values
    }:
        raise A3ClosedLanguageGateError("journal does not name exactly the supports")

    mapping = A3SupportMapping(
        authority_record_digest=declared,
        authority_file_sha256=hashlib.sha256(payload).hexdigest(),
        task_id=source["task_id"],
        split=source["split"],
        corpus_manifest_digest=source["corpus_manifest_digest"],
        split_source_digest=source["split_source_digest"],
        supports=tuple(support_values),
        heldouts=tuple(
            sorted(
                heldout_values,
                key=lambda item: (item.label, item.source_index),
            )
        ),
    )
    _require_canonical_mapping(mapping)
    return mapping


@dataclass(frozen=True, slots=True)
class FrozenLibraryBinding:
    library: FrozenClosedPredicateLibrary
    library_digest: str
    member_count: int
    member_counts_by_kind: Mapping[str, int]
    member_kind_by_digest: Mapping[str, ClosedPredicateKind]


def _bind_frozen_library(
    library: FrozenClosedPredicateLibrary,
    *,
    expected_member_count: int,
    require_complete: bool,
    expected_kind_counts: Mapping[str, int] | None,
) -> FrozenLibraryBinding:
    if not isinstance(library, FrozenClosedPredicateLibrary):
        raise A3ClosedLanguageGateError("library freeze returned the wrong type")
    if require_complete and library.construction_id != (
        "complete-proposer-reachable-closed-union/v2"
    ):
        raise A3ClosedLanguageGateError("library is not the complete closed union")
    if len(library.members) != expected_member_count:
        raise A3ClosedLanguageGateError(
            f"closed library has {len(library.members)} members, expected "
            f"{expected_member_count}"
        )
    member_kind = {item.digest: item.kind for item in library.members}
    if len(member_kind) != expected_member_count:
        raise A3ClosedLanguageGateError("closed library member digests repeat")
    kind_counts = Counter(item.value for item in member_kind.values())
    canonical_counts = {
        kind.value: kind_counts.get(kind.value, 0) for kind in ClosedPredicateKind
    }
    if expected_kind_counts is not None and canonical_counts != dict(
        expected_kind_counts
    ):
        raise A3ClosedLanguageGateError(
            "complete closed library tagged-kind inventory drifted"
        )
    return FrozenLibraryBinding(
        library=library,
        library_digest=library.digest,
        member_count=expected_member_count,
        member_counts_by_kind=MappingProxyType(canonical_counts),
        member_kind_by_digest=MappingProxyType(member_kind),
    )


def _freeze_complete_library_before_pixels() -> FrozenLibraryBinding:
    """The sole production freeze edge; this function accepts no panel input."""

    library = freeze_complete_closed_predicate_library()
    return _bind_frozen_library(
        library,
        expected_member_count=COMPLETE_LIBRARY_MEMBER_COUNT,
        require_complete=True,
        expected_kind_counts=COMPLETE_LIBRARY_KIND_COUNTS,
    )


def _support_candidate(corpus_root: str | Path, relative_path: str) -> Path:
    root = _absolute_lexical(corpus_root)
    safe = _safe_relative_png(relative_path, "support path")
    candidate = root.joinpath(*PurePosixPath(safe).parts)
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # defensive: PurePosix validation already rejects this
        raise A3ClosedLanguageGateError("support path escapes corpus root") from exc
    return candidate


def _read_authenticated_support_png(
    *, corpus_root: str | Path, authority: SupportPanelAuthority
) -> bytes:
    candidate = _support_candidate(corpus_root, authority.relative_path)
    payload = _stable_read_no_symlinks(candidate, maximum=_MAX_PNG_BYTES)
    if not payload.startswith(PNG_SIGNATURE):
        raise A3ClosedLanguageGateError(
            f"support is not an exact PNG: {authority.relative_path}"
        )
    observed = hashlib.sha256(payload).hexdigest()
    if observed != authority.png_sha256:
        raise A3ClosedLanguageGateError(
            f"support SHA-256 differs from authority: {authority.relative_path}"
        )
    return payload


def _separator_counts_by_kind(
    separator_digests: Sequence[str],
    member_kind_by_digest: Mapping[str, ClosedPredicateKind],
) -> dict[str, int]:
    counts = {kind.value: 0 for kind in ClosedPredicateKind}
    seen: set[str] = set()
    for digest in separator_digests:
        _require_sha256(digest, "separator digest")
        if digest in seen:
            raise A3ClosedLanguageGateError("oracle separator digest repeats")
        seen.add(digest)
        kind = member_kind_by_digest.get(digest)
        if kind is None:
            raise A3ClosedLanguageGateError(
                "oracle separator is not in the frozen library"
            )
        counts[kind.value] += 1
    return counts


def _write_once_durable(path: str | Path, payload: bytes) -> Path:
    if not payload or len(payload) > _MAX_RESULT_BYTES:
        raise A3ClosedLanguageGateError("gate result is empty or not compact")
    destination = _absolute_lexical(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_descriptor = _open_absolute_no_symlinks(
        destination.parent, directory=True
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(
                destination.name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            if _stable_read_no_symlinks(
                destination, maximum=_MAX_RESULT_BYTES
            ) != payload:
                raise A3ClosedLanguageGateError(
                    f"refusing to overwrite different artifact at {destination}"
                )
        except OSError as exc:
            raise A3ClosedLanguageGateError(
                f"cannot create gate result {destination}"
            ) from exc
        else:
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise A3ClosedLanguageGateError("short gate-result write")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    if _stable_read_no_symlinks(destination, maximum=_MAX_RESULT_BYTES) != payload:
        raise A3ClosedLanguageGateError("durable gate-result reload differs")
    return destination


@dataclass(frozen=True, slots=True)
class A3ClosedLanguageGateResult:
    report: Mapping[str, Any]
    report_path: Path


PacketExtractor = Callable[[bytes], ExactPanelWitnessPacket]
PacketVerifier = Callable[..., ExactPanelWitnessPacket]
SupportOracle = Callable[..., SupportExpressibilityResult]
SupportReader = Callable[..., bytes]


def _execute_frozen_support_gate(
    *,
    mapping: A3SupportMapping,
    frozen: FrozenLibraryBinding,
    corpus_root: str | Path,
    output_dir: str | Path,
    packet_extractor: PacketExtractor = extract_exact_panel_witness_packet,
    packet_verifier: PacketVerifier = verify_exact_panel_witness_packet,
    oracle: SupportOracle = support_only_expressibility_oracle,
    support_reader: SupportReader = _read_authenticated_support_png,
) -> A3ClosedLanguageGateResult:
    """Read supports and evaluate an already-frozen library.

    This internal seam permits small synthetic unit libraries.  The public
    runner below is the only command path and always supplies the complete
    65,678-member proposer-reachable freeze.
    """

    if not isinstance(mapping, A3SupportMapping):
        raise TypeError("mapping must be an A3SupportMapping")
    _require_canonical_mapping(mapping)
    if not isinstance(frozen, FrozenLibraryBinding):
        raise TypeError("frozen must be a FrozenLibraryBinding")
    if set(item.relative_path for item in mapping.supports) & set(
        item.relative_path for item in mapping.heldouts
    ):
        raise A3ClosedLanguageGateError("held-out path entered support inventory")

    packets: list[ExactPanelWitnessPacket] = []
    receipts: list[dict[str, object]] = []
    read_paths: list[str] = []
    for authority in mapping.supports:
        payload = support_reader(corpus_root=corpus_root, authority=authority)
        read_paths.append(authority.relative_path)
        if hashlib.sha256(payload).hexdigest() != authority.png_sha256:
            raise A3ClosedLanguageGateError(
                "support reader returned bytes outside the recorded SHA-256"
            )
        packet = packet_extractor(payload)
        if not isinstance(packet, ExactPanelWitnessPacket):
            raise A3ClosedLanguageGateError(
                "composite extractor returned the wrong packet type"
            )
        if packet.panel_digest != authority.png_sha256:
            raise A3ClosedLanguageGateError(
                "composite packet does not bind the exact support PNG"
            )
        try:
            verified = packet_verifier(packet, expected_png_bytes=payload)
        except (TypeError, ValueError) as exc:
            raise A3ClosedLanguageGateError(
                "composite packet failed exact cold verification"
            ) from exc
        if verified != packet:
            raise A3ClosedLanguageGateError(
                "composite verifier returned a different packet"
            )
        packets.append(packet)
        receipts.append(
            {
                "label": authority.label,
                "source_index": authority.source_index,
                "relative_path": authority.relative_path,
                "png_sha256": authority.png_sha256,
                "historical_loop_packet_digest": (
                    authority.historical_loop_packet_digest
                ),
                "exact_composite_packet_digest": packet.digest(),
            }
        )
    expected_paths = [item.relative_path for item in mapping.supports]
    if read_paths != expected_paths or len(read_paths) != 12:
        raise A3ClosedLanguageGateError("support read inventory drifted")
    if set(read_paths) & {item.relative_path for item in mapping.heldouts}:
        raise A3ClosedLanguageGateError("a held-out path was read")

    positives = tuple(
        packet for packet, authority in zip(packets, mapping.supports, strict=True)
        if authority.label
    )
    negatives = tuple(
        packet for packet, authority in zip(packets, mapping.supports, strict=True)
        if not authority.label
    )
    try:
        oracle_result = oracle(
            frozen.library,
            positive_support_packets=positives,
            negative_support_packets=negatives,
            model_predicate=None,
        )
    except (TypeError, ValueError) as exc:
        raise A3ClosedLanguageGateError("support-only oracle failed closed") from exc
    if not isinstance(oracle_result, SupportExpressibilityResult):
        raise A3ClosedLanguageGateError("oracle returned the wrong result type")
    if (
        oracle_result.library_digest != frozen.library_digest
        or oracle_result.evaluator_digest
        != closed_visual_predicate_evaluator_digest()
        or oracle_result.model_predicate_digest is not None
        or oracle_result.model_is_exact_separator is not None
    ):
        raise A3ClosedLanguageGateError(
            "oracle result crossed the frozen/no-model edge"
        )
    expected_positive = tuple(item.digest() for item in positives)
    expected_negative = tuple(item.digest() for item in negatives)
    if (
        oracle_result.positive_packet_digests != expected_positive
        or oracle_result.negative_packet_digests != expected_negative
    ):
        raise A3ClosedLanguageGateError("oracle support packet order drifted")

    separator_counts = _separator_counts_by_kind(
        oracle_result.separator_digests, frozen.member_kind_by_digest
    )
    if sum(separator_counts.values()) != len(oracle_result.separator_digests):
        raise A3ClosedLanguageGateError("separator tagged-kind counts do not close")

    receipt_digest = canonical_digest(
        {
            "schema": "gkm.bongard-a3-exact-support-receipts.v1",
            "support_mapping_digest": mapping.support_mapping_digest,
            "receipts": receipts,
        }
    )
    content: dict[str, object] = {
        "schema": GATE_SCHEMA,
        "algorithm_id": GATE_ALGORITHM_ID,
        "algorithm_identities": {
            "gate_python_source_digest": _GATE_SOURCE_DIGEST,
            "closed_predicate_source_digest": (
                closed_visual_predicate_source_digest()
            ),
            "closed_predicate_evaluator_digest": (
                closed_visual_predicate_evaluator_digest()
            ),
            "composite_packet_source_digest": (
                composite_visual_packet_source_digest()
            ),
            "exact_composite_extractor_digest": (
                exact_panel_witness_extractor_digest()
            ),
            "oracle_algorithm_id": (
                "bongard.support-only-expressibility-oracle/v1"
            ),
            "python_is_canonical": True,
            "lean_required": False,
        },
        "source": {
            "forensics_schema": CANONICAL_FORENSICS_SCHEMA,
            "forensics_record_digest": mapping.authority_record_digest,
            "forensics_file_sha256": mapping.authority_file_sha256,
            "support_mapping_digest": mapping.support_mapping_digest,
            "forbidden_heldout_authority_digest": (
                mapping.heldout_authority_digest
            ),
            "task_id": mapping.task_id,
            "split": mapping.split,
            "corpus_manifest_digest": mapping.corpus_manifest_digest,
            "split_source_digest": mapping.split_source_digest,
        },
        "frozen_library": {
            "construction_id": frozen.library.construction_id,
            "library_digest": frozen.library_digest,
            "member_count": frozen.member_count,
            "member_counts_by_tagged_kind": dict(frozen.member_counts_by_kind),
            "freeze_preceded_any_png_read": True,
        },
        "support": {
            "count": len(receipts),
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "exact_receipts_digest": receipt_digest,
            "panels": receipts,
        },
        "oracle": {
            "result_digest": oracle_result.digest,
            "evaluation_matrix_digest": oracle_result.evaluation_matrix_digest,
            "diagnosis": oracle_result.diagnosis.value,
            "exact_forward_separator_count": len(
                oracle_result.separator_digests
            ),
            "separator_counts_by_tagged_kind": separator_counts,
            "separator_inventory_digest": canonical_digest(
                list(oracle_result.separator_digests)
            ),
            "model_predicate_digest": None,
            "model_is_exact_separator": None,
        },
        "claim_boundary": {
            "evaluation_kind": (
                "already-exposed-support-only-closed-language-coverage"
            ),
            "benchmark_or_generalization_claim_authorized": False,
            "new_exposure_event_created": False,
            "new_pixels_opened": False,
            "model_or_proposer_called": False,
            "query_pixels_read": False,
            "heldout_pixels_read": False,
            "official_test_pixels_read": False,
            "action_program_json_authorized": False,
            "negation_rescue_authorized": False,
            "polarity_flip_authorized": False,
            "canonical_attempt3_support_mapping_only": True,
        },
    }
    report = dict(content)
    report["record_digest"] = "sha256:" + canonical_digest(content)
    payload = canonical_json(report) + b"\n"
    if len(payload) > _MAX_RESULT_BYTES:
        raise A3ClosedLanguageGateError("canonical gate result is not compact")
    output_name = (
        report["record_digest"].removeprefix("sha256:")
        + ".a3-closed-language-gate.json"
    )
    report_path = _write_once_durable(
        _absolute_lexical(output_dir) / output_name,
        payload,
    )
    cold, cold_payload = _canonical_json_file(report_path)
    cold_content = dict(cold)
    cold_digest = cold_content.pop("record_digest", None)
    if (
        cold_payload != payload
        or cold != report
        or cold_digest != report["record_digest"]
        or cold_digest != "sha256:" + canonical_digest(cold_content)
    ):
        raise A3ClosedLanguageGateError("cold durable gate-result replay failed")
    return A3ClosedLanguageGateResult(
        report=MappingProxyType(report), report_path=report_path
    )


def run_a3_closed_language_gate(
    *,
    corpus_root: str | Path,
    output_dir: str | Path,
    forensics_record_path: str | Path = CANONICAL_RECORD_PATH,
) -> A3ClosedLanguageGateResult:
    """Run the pinned A3 support gate, freezing all 65,678 members first."""

    # Metadata is authenticated first.  This reads only the checked-in JSON.
    mapping = load_canonical_a3_support_mapping(forensics_record_path)

    # This complete freeze and inventory check must finish before control can
    # enter the only function in this module that reads support PNG bytes.
    frozen = _freeze_complete_library_before_pixels()

    return _execute_frozen_support_gate(
        mapping=mapping,
        frozen=frozen,
        corpus_root=corpus_root,
        output_dir=output_dir,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the already-exposed A3 support-only closed-language gate. "
            "This does not read held-out/query/test pixels or call a model."
        )
    )
    parser.add_argument(
        "--corpus-root",
        required=True,
        help="extracted ShapeBongard_V2 root containing bd/images",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="write-once destination for the compact content-addressed result",
    )
    parser.add_argument(
        "--forensics-record",
        default=str(CANONICAL_RECORD_PATH),
        help=(
            "path to an exact byte-identical copy of the pinned canonical A3 "
            "forensics record"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_a3_closed_language_gate(
            corpus_root=arguments.corpus_root,
            output_dir=arguments.output_dir,
            forensics_record_path=arguments.forensics_record,
        )
    except (A3ClosedLanguageGateError, OSError, TypeError, ValueError) as exc:
        print(f"a3-closed-language-gate: {exc}", file=sys.stderr)
        return 2
    print(result.report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main tests
    raise SystemExit(main())


__all__ = [
    "A3ClosedLanguageGateError",
    "A3ClosedLanguageGateResult",
    "A3SupportMapping",
    "CANONICAL_FORENSICS_RECORD_DIGEST",
    "CANONICAL_RECORD_PATH",
    "COMPLETE_LIBRARY_MEMBER_COUNT",
    "HeldoutPanelAuthority",
    "SupportPanelAuthority",
    "load_canonical_a3_support_mapping",
    "main",
    "run_a3_closed_language_gate",
]
