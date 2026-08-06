"""Read-only audit of the privileged ShapeBongard action-program metadata.

The three ``*_action_programs.json`` files are render recipes, not observations
available to a solver.  They expose latent geometric construction information
that is absent from the PNG interface.  Consequently they are privileged
post-hoc diagnostic/oracle metadata: they may be used to audit the release or
to state an explicitly labelled oracle upper bound, but MUST NEVER be included
in proposer, support-model, query-model, synthesis, or prediction inputs.

This module deliberately returns only aggregate counts and content addresses.
It never returns parsed programs.  Upstream used ordinary ``json.dump`` rather
than this repository's canonical byte encoding, so the audit binds both the
unchanged raw bytes and the canonical encoding of the parsed value.  A mismatch
between those encodings is an observation, not permission to rewrite upstream.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from bongard.corpus import FAMILIES, ShapeBongardCorpus
from bongard.release import OfficialReleaseDescriptor


AUDIT_SCHEMA = "gkm.shape-bongard-action-program-audit.v1"
PRIVILEGE_CLASS = "privileged_post_hoc_diagnostic_oracle_metadata"

# These limits cover the complete official release while making every recursive
# collection finite before callers inspect untrusted JSON.
MAX_ACTION_PROGRAM_FILE_BYTES = 32 * 1024 * 1024
MAX_TASKS_PER_FAMILY = 5_000
MAX_JSON_CONTAINER_DEPTH = 5
SIDES_PER_TASK = 2
PANELS_PER_SIDE = 7
MIN_SHAPE_PROGRAMS_PER_PANEL = 1
MAX_SHAPE_PROGRAMS_PER_PANEL = 2
MIN_ACTIONS_PER_SHAPE_PROGRAM = 1
MAX_ACTIONS_PER_SHAPE_PROGRAM = 9
MAX_ACTION_UTF8_BYTES = 64

ACTION_KINDS = ("arc", "line")
STROKE_STYLES = ("circle", "normal", "square", "triangle", "zigzag")
_NORMALIZED_DECIMAL = r"(?:0\.[0-9]{3}|1\.000)"
_STYLE = "(?:" + "|".join(STROKE_STYLES) + ")"
_LINE_ACTION = re.compile(
    rf"line_(?P<style>{_STYLE})_(?P<length>{_NORMALIZED_DECIMAL})-"
    rf"(?P<turn>{_NORMALIZED_DECIMAL})\Z"
)
_ARC_ACTION = re.compile(
    rf"arc_(?P<style>{_STYLE})_(?P<radius>{_NORMALIZED_DECIMAL})_"
    rf"(?P<arc>{_NORMALIZED_DECIMAL})-(?P<turn>{_NORMALIZED_DECIMAL})\Z"
)
_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_READ_CHUNK_BYTES = 1024 * 1024

USAGE_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "classification": PRIVILEGE_CLASS,
        "allowed_uses": (
            "post_hoc_release_diagnostics",
            "explicitly_labelled_oracle_upper_bound",
        ),
        "forbidden_inputs": (
            "proposer",
            "support_model",
            "query_model",
            "predicate_synthesis",
            "prediction",
        ),
        "must_never_enter_proposer_or_query_inputs": True,
        "audit_output": "aggregate_counts_and_content_addresses_only",
        "reason": (
            "render recipes expose latent geometric construction information "
            "that is not present in the benchmark PNG interface"
        ),
    }
)


class ActionProgramAuditError(RuntimeError):
    """The action-program source could not be safely read or parsed."""


class ActionProgramValidationError(ActionProgramAuditError):
    """One or more release invariants failed after a complete audit."""

    def __init__(self, report: "ActionProgramAuditReport") -> None:
        self.report = report
        super().__init__(
            f"{report.anomaly_count} action-program metadata anomalies; "
            "the complete bounded report is available as exception.report"
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
        raise ActionProgramAuditError(
            f"parsed action programs are not canonical-JSON encodable: {exc}"
        ) from exc


def _address_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _address_value(value: object) -> str:
    return _address_bytes(_canonical_json_bytes(value))


def _line_address(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return _address_bytes(payload)


def _validate_address(value: str | None, label: str) -> None:
    if value is not None and _ADDRESS.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical sha256 address")


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
        raise ActionProgramAuditError(f"cannot stat action-program file {path}: {exc}") from exc
    if stat.S_ISLNK(value.st_mode):
        raise ActionProgramAuditError(f"action-program file is a symlink: {path}")
    if not stat.S_ISREG(value.st_mode):
        raise ActionProgramAuditError(
            f"action-program path is not a regular file: {path}"
        )
    return value


def _lstat_directory(path: Path) -> None:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ActionProgramAuditError(f"cannot stat action-program directory {path}: {exc}") from exc
    if stat.S_ISLNK(value.st_mode):
        raise ActionProgramAuditError(f"action-program directory is a symlink: {path}")
    if not stat.S_ISDIR(value.st_mode):
        raise ActionProgramAuditError(f"action-program parent is not a directory: {path}")


def _read_stable_snapshot(path: Path, *, max_file_bytes: int) -> tuple[bytes, tuple[int, ...]]:
    before = _lstat_regular(path)
    if before.st_size <= 0:
        raise ActionProgramAuditError(f"action-program file is empty: {path}")
    if before.st_size > max_file_bytes:
        raise ActionProgramAuditError(
            f"action-program file exceeds the {max_file_bytes}-byte safety limit: {path}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ActionProgramAuditError(f"cannot open action-program file {path}: {exc}") from exc

    try:
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            opened = os.fstat(source.fileno())
            if _file_fingerprint(opened) != _file_fingerprint(before):
                raise ActionProgramAuditError(
                    f"action-program path changed while opening: {path}"
                )
            chunks: list[bytes] = []
            byte_count = 0
            while True:
                chunk = source.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > max_file_bytes:
                    raise ActionProgramAuditError(
                        f"action-program file exceeded its byte limit while reading: {path}"
                    )
                chunks.append(chunk)
            after_read = os.fstat(source.fileno())
            if _file_fingerprint(after_read) != _file_fingerprint(opened):
                raise ActionProgramAuditError(
                    f"action-program file changed while reading: {path}"
                )
            if byte_count != opened.st_size:
                raise ActionProgramAuditError(
                    f"action-program byte count changed while reading: {path}"
                )
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise

    after = _lstat_regular(path)
    if _file_fingerprint(after) != _file_fingerprint(before):
        raise ActionProgramAuditError(f"action-program path changed during read: {path}")
    return b"".join(chunks), _file_fingerprint(after)


def _maximum_json_container_depth(text: str) -> int:
    """Compute container depth without being fooled by brackets in strings."""

    depth = 0
    maximum = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            maximum = max(maximum, depth)
            if maximum > MAX_JSON_CONTAINER_DEPTH:
                return maximum
        elif character in "]}":
            depth -= 1
    return maximum


class _DuplicateJSONKey(ValueError):
    pass


def _strict_json(payload: bytes, label: str) -> object:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ActionProgramAuditError(f"{label} is not strict UTF-8: {exc}") from exc
    maximum_depth = _maximum_json_container_depth(text)
    if maximum_depth > MAX_JSON_CONTAINER_DEPTH:
        raise ActionProgramAuditError(
            f"{label} exceeds JSON container depth {MAX_JSON_CONTAINER_DEPTH}"
        )

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJSONKey(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError) as exc:
        raise ActionProgramAuditError(f"cannot parse {label} as strict JSON: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ActionProgramAnomaly:
    family: str
    code: str
    location: str
    detail: str
    occurrences: int = 1

    def to_data(self) -> dict[str, object]:
        return {
            "family": self.family,
            "code": self.code,
            "location": self.location,
            "detail": self.detail,
            "occurrences": self.occurrences,
        }


class _Anomalies:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.count = 0
        self.items: list[ActionProgramAnomaly] = []

    def add(
        self,
        family: str,
        code: str,
        location: str,
        detail: str,
        *,
        occurrences: int = 1,
    ) -> None:
        self.count += occurrences
        if len(self.items) < self.maximum:
            self.items.append(
                ActionProgramAnomaly(family, code, location, detail, occurrences)
            )


def _distribution_data(values: Sequence[tuple[int, int]], name: str) -> list[dict[str, int]]:
    return [{name: value, "count": count} for value, count in values]


@dataclass(frozen=True, slots=True)
class ActionProgramFamilyAudit:
    family: str
    filename: str
    raw_sha256: str
    size_bytes: int
    canonical_json_sha256: str
    canonical_size_bytes: int
    canonical_bytes_equal: bool
    root_key_order_sorted: bool
    inventory_task_count: int
    metadata_task_count: int
    inventory_task_ids_sha256: str
    metadata_task_ids_sha256: str
    task_keys_exact: bool
    side_count: int
    panel_count: int
    shape_program_count: int
    action_count: int
    unique_action_count: int
    action_kind_counts: tuple[tuple[str, int], ...]
    stroke_style_counts: tuple[tuple[str, int], ...]
    shape_programs_per_panel_counts: tuple[tuple[int, int], ...]
    actions_per_shape_program_counts: tuple[tuple[int, int], ...]
    max_action_utf8_bytes_observed: int
    structure_valid: bool
    anomaly_count: int

    def to_data(self) -> dict[str, object]:
        return {
            "family": self.family,
            "filename": self.filename,
            "raw_sha256": self.raw_sha256,
            "size_bytes": self.size_bytes,
            "canonical_json_sha256": self.canonical_json_sha256,
            "canonical_size_bytes": self.canonical_size_bytes,
            "canonical_bytes_equal": self.canonical_bytes_equal,
            "root_key_order_sorted": self.root_key_order_sorted,
            "inventory_task_count": self.inventory_task_count,
            "metadata_task_count": self.metadata_task_count,
            "inventory_task_ids_sha256": self.inventory_task_ids_sha256,
            "metadata_task_ids_sha256": self.metadata_task_ids_sha256,
            "task_keys_exact": self.task_keys_exact,
            "side_count": self.side_count,
            "panel_count": self.panel_count,
            "shape_program_count": self.shape_program_count,
            "action_count": self.action_count,
            "unique_action_count": self.unique_action_count,
            "action_kind_counts": dict(self.action_kind_counts),
            "stroke_style_counts": dict(self.stroke_style_counts),
            "shape_programs_per_panel_counts": _distribution_data(
                self.shape_programs_per_panel_counts, "shape_programs"
            ),
            "actions_per_shape_program_counts": _distribution_data(
                self.actions_per_shape_program_counts, "actions"
            ),
            "max_action_utf8_bytes_observed": self.max_action_utf8_bytes_observed,
            "structure_valid": self.structure_valid,
            "anomaly_count": self.anomaly_count,
        }


@dataclass(frozen=True, slots=True)
class ActionProgramAuditReport:
    release_descriptor_digest: str | None
    release_corpus_manifest_digest_reference: str | None
    inventory_task_ids_sha256: str
    families: tuple[ActionProgramFamilyAudit, ...]
    source_bundle_digest: str
    max_file_bytes: int
    task_count: int
    side_count: int
    panel_count: int
    shape_program_count: int
    action_count: int
    unique_action_count: int
    raw_byte_count_total: int
    canonical_byte_count_total: int
    action_kind_counts: tuple[tuple[str, int], ...]
    stroke_style_counts: tuple[tuple[str, int], ...]
    anomaly_count: int
    anomalies: tuple[ActionProgramAnomaly, ...]
    anomalies_truncated: bool
    digest: str
    schema: str = AUDIT_SCHEMA

    def content_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "usage_policy": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in USAGE_POLICY.items()
            },
            "bounds": {
                "max_file_bytes": self.max_file_bytes,
                "max_tasks_per_family": MAX_TASKS_PER_FAMILY,
                "max_json_container_depth": MAX_JSON_CONTAINER_DEPTH,
                "sides_per_task": SIDES_PER_TASK,
                "panels_per_side": PANELS_PER_SIDE,
                "shape_programs_per_panel": [
                    MIN_SHAPE_PROGRAMS_PER_PANEL,
                    MAX_SHAPE_PROGRAMS_PER_PANEL,
                ],
                "side_order": ["positive", "negative"],
                "actions_per_shape_program": [
                    MIN_ACTIONS_PER_SHAPE_PROGRAM,
                    MAX_ACTIONS_PER_SHAPE_PROGRAM,
                ],
                "max_action_utf8_bytes": MAX_ACTION_UTF8_BYTES,
                "action_kinds": list(ACTION_KINDS),
                "stroke_styles": list(STROKE_STYLES),
                "numeric_encoding": "normalized_0_to_1_exactly_three_decimals",
            },
            "serialization_policy": {
                "raw_upstream_files_rewritten": False,
                "raw_bytes_verified_independently": True,
                "parsed_values_canonically_hashed": True,
                "canonical_encoding": "UTF-8/sorted-keys/no-insignificant-whitespace",
            },
            "release_descriptor_digest": self.release_descriptor_digest,
            "release_corpus_manifest_digest_reference": (
                self.release_corpus_manifest_digest_reference
            ),
            "inventory_task_ids_sha256": self.inventory_task_ids_sha256,
            "families": {
                family.family: family.to_data() for family in self.families
            },
            "source_bundle_digest": self.source_bundle_digest,
            "totals": {
                "tasks": self.task_count,
                "sides": self.side_count,
                "panels": self.panel_count,
                "shape_programs": self.shape_program_count,
                "actions": self.action_count,
                "unique_actions": self.unique_action_count,
                "raw_bytes": self.raw_byte_count_total,
                "canonical_bytes": self.canonical_byte_count_total,
                "action_kind_counts": dict(self.action_kind_counts),
                "stroke_style_counts": dict(self.stroke_style_counts),
            },
            "anomaly_count": self.anomaly_count,
            "anomalies": [item.to_data() for item in self.anomalies],
            "anomalies_truncated": self.anomalies_truncated,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_dict(), "digest": self.digest}


@dataclass(frozen=True, slots=True)
class _FamilyOutcome:
    report: ActionProgramFamilyAudit
    unique_actions: frozenset[str]
    fingerprint: tuple[int, ...]
    path: Path


def _audit_family(
    corpus: ShapeBongardCorpus,
    family: str,
    inventory: tuple[str, ...],
    anomalies: _Anomalies,
    *,
    max_file_bytes: int,
) -> _FamilyOutcome:
    family_start = anomalies.count
    family_directory = corpus.root / family
    _lstat_directory(family_directory)
    path = family_directory / f"{family}_action_programs.json"
    payload, fingerprint = _read_stable_snapshot(path, max_file_bytes=max_file_bytes)
    raw_sha256 = _address_bytes(payload)
    raw = _strict_json(payload, path.name)
    canonical = _canonical_json_bytes(raw)

    metadata: Mapping[str, object]
    root_structure_valid = isinstance(raw, dict)
    if not root_structure_valid:
        anomalies.add(family, "root_type", "$", "root must be a JSON object")
        metadata = {}
    else:
        metadata = raw

    metadata_keys = tuple(metadata)
    inventory_set = set(inventory)
    metadata_set = set(metadata_keys)
    missing = tuple(sorted(inventory_set - metadata_set))
    extra = tuple(sorted(metadata_set - inventory_set))
    if missing:
        anomalies.add(
            family,
            "missing_task_key",
            "$",
            f"{len(missing)} inventory task keys are absent; first={list(missing[:5])}",
            occurrences=len(missing),
        )
    if extra:
        anomalies.add(
            family,
            "extra_task_key",
            "$",
            f"{len(extra)} metadata task keys are not inventoried; first={list(extra[:5])}",
            occurrences=len(extra),
        )
    if len(metadata_keys) > MAX_TASKS_PER_FAMILY:
        anomalies.add(
            family,
            "task_count_bound",
            "$",
            f"{len(metadata_keys)} tasks exceeds {MAX_TASKS_PER_FAMILY}",
            occurrences=len(metadata_keys) - MAX_TASKS_PER_FAMILY,
        )
        root_structure_valid = False

    side_count = 0
    panel_count = 0
    shape_program_count = 0
    action_count = 0
    action_kinds: Counter[str] = Counter()
    stroke_styles: Counter[str] = Counter()
    shapes_per_panel: Counter[int] = Counter()
    actions_per_shape: Counter[int] = Counter()
    unique_actions: set[str] = set()
    max_action_bytes = 0
    structural_start = anomalies.count

    for task_id in sorted(metadata_keys):
        task = metadata[task_id]
        task_location = f"$.{task_id}"
        if not isinstance(task, list):
            anomalies.add(family, "task_type", task_location, "task must be a list")
            continue
        side_count += len(task)
        if len(task) != SIDES_PER_TASK:
            anomalies.add(
                family,
                "side_count",
                task_location,
                f"expected {SIDES_PER_TASK} sides, observed {len(task)}",
                occurrences=abs(len(task) - SIDES_PER_TASK) or 1,
            )
        for side_index, side in enumerate(task):
            side_location = f"{task_location}[{side_index}]"
            if not isinstance(side, list):
                anomalies.add(family, "side_type", side_location, "side must be a list")
                continue
            panel_count += len(side)
            if len(side) != PANELS_PER_SIDE:
                anomalies.add(
                    family,
                    "panel_count",
                    side_location,
                    f"expected {PANELS_PER_SIDE} panels, observed {len(side)}",
                    occurrences=abs(len(side) - PANELS_PER_SIDE) or 1,
                )
            for panel_index, panel in enumerate(side):
                panel_location = f"{side_location}[{panel_index}]"
                if not isinstance(panel, list):
                    anomalies.add(
                        family, "panel_type", panel_location, "panel must be a list"
                    )
                    continue
                shape_count = len(panel)
                shapes_per_panel[shape_count] += 1
                shape_program_count += shape_count
                if not (
                    MIN_SHAPE_PROGRAMS_PER_PANEL
                    <= shape_count
                    <= MAX_SHAPE_PROGRAMS_PER_PANEL
                ):
                    anomalies.add(
                        family,
                        "shape_program_count",
                        panel_location,
                        f"observed {shape_count}, expected "
                        f"{MIN_SHAPE_PROGRAMS_PER_PANEL}..{MAX_SHAPE_PROGRAMS_PER_PANEL}",
                    )
                for shape_index, shape_program in enumerate(panel):
                    shape_location = f"{panel_location}[{shape_index}]"
                    if not isinstance(shape_program, list):
                        anomalies.add(
                            family,
                            "shape_program_type",
                            shape_location,
                            "shape program must be a list",
                        )
                        continue
                    shape_actions = len(shape_program)
                    actions_per_shape[shape_actions] += 1
                    action_count += shape_actions
                    if not (
                        MIN_ACTIONS_PER_SHAPE_PROGRAM
                        <= shape_actions
                        <= MAX_ACTIONS_PER_SHAPE_PROGRAM
                    ):
                        anomalies.add(
                            family,
                            "action_count",
                            shape_location,
                            f"observed {shape_actions}, expected "
                            f"{MIN_ACTIONS_PER_SHAPE_PROGRAM}.."
                            f"{MAX_ACTIONS_PER_SHAPE_PROGRAM}",
                        )
                    for action_index, action in enumerate(shape_program):
                        action_location = f"{shape_location}[{action_index}]"
                        if not isinstance(action, str):
                            anomalies.add(
                                family,
                                "action_type",
                                action_location,
                                "action must be a string",
                            )
                            continue
                        encoded_bytes = len(action.encode("utf-8"))
                        max_action_bytes = max(max_action_bytes, encoded_bytes)
                        if encoded_bytes > MAX_ACTION_UTF8_BYTES:
                            anomalies.add(
                                family,
                                "action_byte_bound",
                                action_location,
                                f"action uses {encoded_bytes} UTF-8 bytes, maximum is "
                                f"{MAX_ACTION_UTF8_BYTES}",
                            )
                        line_match = _LINE_ACTION.fullmatch(action)
                        arc_match = _ARC_ACTION.fullmatch(action)
                        match = line_match or arc_match
                        if match is None:
                            anomalies.add(
                                family,
                                "action_grammar",
                                action_location,
                                f"non-canonical action token {action!r}",
                            )
                            continue
                        kind = "line" if line_match is not None else "arc"
                        action_kinds[kind] += 1
                        stroke_styles[match.group("style")] += 1
                        unique_actions.add(action)

    structure_valid = root_structure_valid and anomalies.count == structural_start
    family_anomaly_count = anomalies.count - family_start
    family_report = ActionProgramFamilyAudit(
        family=family,
        filename=path.name,
        raw_sha256=raw_sha256,
        size_bytes=len(payload),
        canonical_json_sha256=_address_bytes(canonical),
        canonical_size_bytes=len(canonical),
        canonical_bytes_equal=payload == canonical,
        root_key_order_sorted=metadata_keys == tuple(sorted(metadata_keys)),
        inventory_task_count=len(inventory),
        metadata_task_count=len(metadata_keys),
        inventory_task_ids_sha256=_line_address(inventory),
        metadata_task_ids_sha256=_line_address(metadata_keys),
        task_keys_exact=not missing and not extra,
        side_count=side_count,
        panel_count=panel_count,
        shape_program_count=shape_program_count,
        action_count=action_count,
        unique_action_count=len(unique_actions),
        action_kind_counts=tuple(sorted(action_kinds.items())),
        stroke_style_counts=tuple(sorted(stroke_styles.items())),
        shape_programs_per_panel_counts=tuple(sorted(shapes_per_panel.items())),
        actions_per_shape_program_counts=tuple(sorted(actions_per_shape.items())),
        max_action_utf8_bytes_observed=max_action_bytes,
        structure_valid=structure_valid,
        anomaly_count=family_anomaly_count,
    )
    return _FamilyOutcome(
        family_report,
        frozenset(unique_actions),
        fingerprint,
        path,
    )


def audit_action_program_metadata(
    corpus: ShapeBongardCorpus,
    *,
    official_release: OfficialReleaseDescriptor | None = None,
    require_valid: bool = True,
    max_anomalies: int = 1_000,
    max_file_bytes: int = MAX_ACTION_PROGRAM_FILE_BYTES,
) -> ActionProgramAuditReport:
    """Audit all three metadata files without exposing their parsed contents.

    Supplying ``official_release`` additionally binds the corpus inventory to
    the checked-in complete-release task-id digest and family counts.  It does
    not read action programs into any benchmark execution or model pathway.
    ``require_valid=True`` fails after the complete pass and attaches the
    bounded aggregate report to :class:`ActionProgramValidationError`.
    """

    if not isinstance(corpus, ShapeBongardCorpus):
        raise TypeError("corpus must be a ShapeBongardCorpus")
    for label, value in (
        ("max_anomalies", max_anomalies),
        ("max_file_bytes", max_file_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")

    _lstat_directory(corpus.root)
    inventories = {
        family: tuple(sorted(task.task_id for task in corpus.tasks if task.family == family))
        for family in FAMILIES
    }
    all_inventory = tuple(sorted(task_id for values in inventories.values() for task_id in values))
    inventory_digest = _line_address(all_inventory)
    anomalies = _Anomalies(max_anomalies)

    release_descriptor_digest: str | None = None
    release_corpus_manifest_digest_reference: str | None = None
    if official_release is not None:
        if not isinstance(official_release, OfficialReleaseDescriptor):
            raise TypeError("official_release must be an OfficialReleaseDescriptor")
        release_descriptor_digest = official_release.digest
        release_corpus_manifest_digest_reference = (
            official_release.corpus_manifest_sha256
        )
        _validate_address(release_descriptor_digest, "release descriptor digest")
        _validate_address(
            release_corpus_manifest_digest_reference,
            "release corpus manifest digest reference",
        )
        if dict(corpus.family_counts) != dict(official_release.family_counts):
            anomalies.add(
                "all",
                "official_family_counts",
                "$inventory",
                f"observed {dict(corpus.family_counts)}, expected "
                f"{dict(official_release.family_counts)}",
            )
        if inventory_digest != official_release.task_ids_sha256:
            anomalies.add(
                "all",
                "official_task_inventory",
                "$inventory",
                f"observed {inventory_digest}, expected {official_release.task_ids_sha256}",
            )

    outcomes = tuple(
        _audit_family(
            corpus,
            family,
            inventories[family],
            anomalies,
            max_file_bytes=max_file_bytes,
        )
        for family in FAMILIES
    )

    # Detect replacement of an earlier source while later families were being
    # audited.  Parsed bytes never survive beyond this aggregate report.
    for outcome in outcomes:
        if _file_fingerprint(_lstat_regular(outcome.path)) != outcome.fingerprint:
            raise ActionProgramAuditError(
                f"action-program path changed during complete audit: {outcome.path}"
            )

    families = tuple(outcome.report for outcome in outcomes)
    action_kinds: Counter[str] = Counter()
    stroke_styles: Counter[str] = Counter()
    for family in families:
        action_kinds.update(dict(family.action_kind_counts))
        stroke_styles.update(dict(family.stroke_style_counts))
    source_bundle = [
        {
            "family": family.family,
            "filename": family.filename,
            "raw_sha256": family.raw_sha256,
            "size_bytes": family.size_bytes,
        }
        for family in families
    ]
    report_fields: dict[str, Any] = {
        "release_descriptor_digest": release_descriptor_digest,
        "release_corpus_manifest_digest_reference": (
            release_corpus_manifest_digest_reference
        ),
        "inventory_task_ids_sha256": inventory_digest,
        "families": families,
        "source_bundle_digest": _address_value(source_bundle),
        "max_file_bytes": max_file_bytes,
        "task_count": sum(family.metadata_task_count for family in families),
        "side_count": sum(family.side_count for family in families),
        "panel_count": sum(family.panel_count for family in families),
        "shape_program_count": sum(family.shape_program_count for family in families),
        "action_count": sum(family.action_count for family in families),
        "unique_action_count": len(
            set().union(*(set(outcome.unique_actions) for outcome in outcomes))
        ),
        "raw_byte_count_total": sum(family.size_bytes for family in families),
        "canonical_byte_count_total": sum(
            family.canonical_size_bytes for family in families
        ),
        "action_kind_counts": tuple(sorted(action_kinds.items())),
        "stroke_style_counts": tuple(sorted(stroke_styles.items())),
        "anomaly_count": anomalies.count,
        "anomalies": tuple(anomalies.items),
        "anomalies_truncated": anomalies.count > sum(
            item.occurrences for item in anomalies.items
        ),
    }
    provisional = ActionProgramAuditReport(**report_fields, digest="")
    report = ActionProgramAuditReport(
        **report_fields,
        digest=_address_value(provisional.content_dict()),
    )
    if require_valid and report.anomaly_count:
        raise ActionProgramValidationError(report)
    return report


__all__ = [
    "ACTION_KINDS",
    "AUDIT_SCHEMA",
    "ActionProgramAnomaly",
    "ActionProgramAuditError",
    "ActionProgramAuditReport",
    "ActionProgramFamilyAudit",
    "ActionProgramValidationError",
    "MAX_ACTION_PROGRAM_FILE_BYTES",
    "MAX_ACTION_UTF8_BYTES",
    "MAX_ACTIONS_PER_SHAPE_PROGRAM",
    "MAX_JSON_CONTAINER_DEPTH",
    "MAX_SHAPE_PROGRAMS_PER_PANEL",
    "MAX_TASKS_PER_FAMILY",
    "PANELS_PER_SIDE",
    "PRIVILEGE_CLASS",
    "SIDES_PER_TASK",
    "STROKE_STYLES",
    "USAGE_POLICY",
    "audit_action_program_metadata",
]
