"""Fail-closed record for the 2026-08-10 official-program exposure.

This module is deliberately metadata-only.  It never opens the official
archive, PNGs, action-program file, catalog sources, model, or predictions.
It records the already-observed custody failure and deterministically builds
the one exposure-ledger child and fixed campaign-intent tombstone that must be
persisted before this calibration campaign can be considered retired.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Final, Mapping

from bongard.canonical import canonical_digest, canonical_json
from bongard.exposure import ExposureEvent, ExposureLedger
from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

INCIDENT_SCHEMA: Final = "gkm.bongard-skeleton-graph-custody-incident.v1"
TOMBSTONE_CLAIM_SCHEMA: Final = (
    "gkm.bongard-skeleton-graph-custody-incident-tombstone-claim.v1"
)
INCIDENT_RECORD_PATH: Final = Path(
    "bongard/data/panel_action_count_skeleton_graph_custody_incident_20260810_v1.json"
)
CAMPAIGN_INTENT_FILENAME: Final = (
    "panel_action_count_skeleton_graph_campaign_attempt_v2.json"
)
EXPOSURE_PREDECESSOR_LEDGER_DIGEST: Final = (
    "sha256:6995ea9cfda2f384cb0ba1b1cdc3611c965227c60fdb281d1e2e56fffa357b56"
)
EXPOSURE_PREDECESSOR_FILE_SHA256: Final = (
    "sha256:8c5034e77f769a67b1bc16b41881e14887592e070e730d062049ea33e1467ff8"
)
EXPOSURE_CORPUS_DIGEST: Final = (
    "sha256:6fa51548520190a412812ba8f872dc3c7a7a2b2c47c0e42a4d9f6df351dce138"
)
OFFICIAL_HD_ACTION_PROGRAM_PATH: Final = (
    "downloads/ShapeBongard_V2_full/ShapeBongard_V2/hd/hd_action_programs.json"
)
OFFICIAL_HD_ACTION_PROGRAM_FILE_SHA256: Final = (
    "sha256:190f3f850d98fa9df0f85cbbafa05fbbaf6d8845586c186ce062af8812ba7e7c"
)
OFFICIAL_HD_ACTION_PROGRAM_SIZE_BYTES: Final = 10_311_475
OFFICIAL_HD_ACTION_PROGRAM_TASK_COUNT: Final = 4_400
ABANDONED_WRAPPER_SOURCE_SHA256: Final = (
    "sha256:5665e8f14800009feccb1bcd16f670a5e2dda55acb61fc1bbaff1ff91d7c5328"
)
ABANDONED_WRAPPER_TEST_SHA256: Final = (
    "sha256:8afdcb4172ea00e0dc302c37f84782ef1b44a2335e805eb7af1f2fafea90335b"
)

TARGET_FAMILY_TASK_IDS: Final = tuple(
    f"hd_convex-has_four_straight_lines_{index:04d}" for index in range(20)
)
SAME_FAMILY_CALIBRATION_TASK_IDS: Final = TARGET_FAMILY_TASK_IDS[2:18]

RG_PATTERN: Final = (
    "hd_convex-has_four_straight_lines_0000|"
    "convex-has_four_straight_lines_0000|TARGET_TASK_ID"
)
RG_ARGV: Final = (
    "rg",
    "-n",
    RG_PATTERN,
    "bongard",
    "downloads/ShapeBongard_V2_full",
    "--glob",
    "*.py",
    "--glob",
    "*.json",
    "--glob",
    "*.md",
    "--glob",
    "*.txt",
)
HEAD_ARGV: Final = ("head", "-n", "400")
SHELL_COMMAND: Final = (
    "rg -n 'hd_convex-has_four_straight_lines_0000|"
    "convex-has_four_straight_lines_0000|TARGET_TASK_ID' "
    "bongard downloads/ShapeBongard_V2_full --glob '*.py' --glob '*.json' "
    "--glob '*.md' --glob '*.txt' 2>/dev/null | head -n 400"
)

_ADDRESS = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OBSERVED_AT = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\Z"
)


class SkeletonGraphCustodyIncidentError(RuntimeError):
    """The incident or tombstone evidence is malformed or inconsistent."""


def source_sha256() -> str:
    current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current != _LOADED_SOURCE_SHA256:
        raise SkeletonGraphCustodyIncidentError(
            "custody-incident source changed after import"
        )
    return current


def _address(value: object, *, label: str) -> str:
    if type(value) is not str or _ADDRESS.fullmatch(value) is None:
        raise SkeletonGraphCustodyIncidentError(f"{label} is not a SHA-256 address")
    return value


def _record_digest(content: Mapping[str, Any]) -> str:
    return "sha256:" + canonical_digest(content)


def _canonical_record_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json(value) + b"\n"


def _validate_exact_ledger(value: ExposureLedger, *, label: str) -> None:
    if (
        type(value) is not ExposureLedger
        or type(value.corpus_digest) is not str
        or type(value.events) is not tuple
    ):
        raise SkeletonGraphCustodyIncidentError(f"{label} ledger types differ")
    _address(value.corpus_digest, label=f"{label} corpus")
    for event in value.events:
        if (
            type(event) is not ExposureEvent
            or type(event.sequence) is not int
            or type(event.observed_at) is not str
            or type(event.phase) is not str
            or type(event.actor) is not str
            or type(event.purpose) is not str
            or type(event.task_ids) is not tuple
            or type(event.panel_ids) is not tuple
            or any(type(item) is not str for item in event.task_ids)
            or any(type(item) is not str for item in event.panel_ids)
            or (event.source is not None and type(event.source) is not str)
            or (
                event.previous_digest is not None
                and type(event.previous_digest) is not str
            )
            or type(event.digest) is not str
        ):
            raise SkeletonGraphCustodyIncidentError(
                f"{label} exposure event types differ"
            )
        _address(event.digest, label=f"{label} event")
        if event.source is not None and event.source.startswith("sha256:"):
            _address(event.source, label=f"{label} event source")
        if event.previous_digest is not None:
            _address(event.previous_digest, label=f"{label} previous event")


@dataclass(frozen=True, slots=True)
class SkeletonGraphCustodyIncident:
    incident_id: str
    incident_date: str
    shell_command: str
    rg_argv: tuple[str, ...]
    head_argv: tuple[str, ...]
    stderr_redirect: str
    search_roots: tuple[str, ...]
    search_globs: tuple[str, ...]
    search_patterns: tuple[str, ...]
    official_program_path: str
    official_program_file_sha256: str
    official_program_size_bytes: int
    official_program_task_count: int
    target_family_task_ids: tuple[str, ...]
    same_family_calibration_task_ids: tuple[str, ...]
    tool_output_line_count: int
    outer_original_token_count: int
    nested_original_token_count: int
    renderer_omitted_byte_count: int
    evidence_statements: tuple[str, ...]
    reason_codes: tuple[str, ...]
    no_png_bytes_opened: bool
    no_zip_or_archive_bytes_opened: bool
    no_model_or_prediction_execution: bool
    no_files_written_by_incident_command: bool
    generic_hd_programs_conservatively_preexposed: bool
    official_test_programs_preexposed_if_resident: bool
    calibration_live_launch_authorized: bool
    target_query_authorized: bool
    benchmark_claim_authorized: bool
    required_terminal_outcome: str
    abandoned_wrapper_source_sha256: str
    abandoned_wrapper_test_sha256: str
    abandoned_wrapper_audit_finding_count: int
    abandoned_wrapper_deleted_uncommitted: bool
    incident_recorder_source_sha256: str
    record_digest: str

    def content_data(self) -> dict[str, Any]:
        return {
            "schema": INCIDENT_SCHEMA,
            "incident_id": self.incident_id,
            "incident_date": self.incident_date,
            "shell_command": self.shell_command,
            "rg_argv": list(self.rg_argv),
            "head_argv": list(self.head_argv),
            "stderr_redirect": self.stderr_redirect,
            "search_roots": list(self.search_roots),
            "search_globs": list(self.search_globs),
            "search_patterns": list(self.search_patterns),
            "official_program_path": self.official_program_path,
            "official_program_file_sha256": self.official_program_file_sha256,
            "official_program_size_bytes": self.official_program_size_bytes,
            "official_program_task_count": self.official_program_task_count,
            "target_family_task_ids": list(self.target_family_task_ids),
            "same_family_calibration_task_ids": list(
                self.same_family_calibration_task_ids
            ),
            "tool_output_line_count": self.tool_output_line_count,
            "outer_original_token_count": self.outer_original_token_count,
            "nested_original_token_count": self.nested_original_token_count,
            "renderer_omitted_byte_count": self.renderer_omitted_byte_count,
            "evidence_statements": list(self.evidence_statements),
            "reason_codes": list(self.reason_codes),
            "no_png_bytes_opened": self.no_png_bytes_opened,
            "no_zip_or_archive_bytes_opened": self.no_zip_or_archive_bytes_opened,
            "no_model_or_prediction_execution": self.no_model_or_prediction_execution,
            "no_files_written_by_incident_command": (
                self.no_files_written_by_incident_command
            ),
            "generic_hd_programs_conservatively_preexposed": (
                self.generic_hd_programs_conservatively_preexposed
            ),
            "official_test_programs_preexposed_if_resident": (
                self.official_test_programs_preexposed_if_resident
            ),
            "calibration_live_launch_authorized": (
                self.calibration_live_launch_authorized
            ),
            "target_query_authorized": self.target_query_authorized,
            "benchmark_claim_authorized": self.benchmark_claim_authorized,
            "required_terminal_outcome": self.required_terminal_outcome,
            "abandoned_wrapper_source_sha256": self.abandoned_wrapper_source_sha256,
            "abandoned_wrapper_test_sha256": self.abandoned_wrapper_test_sha256,
            "abandoned_wrapper_audit_finding_count": (
                self.abandoned_wrapper_audit_finding_count
            ),
            "abandoned_wrapper_deleted_uncommitted": (
                self.abandoned_wrapper_deleted_uncommitted
            ),
            "incident_recorder_source_sha256": self.incident_recorder_source_sha256,
        }

    def to_data(self) -> dict[str, Any]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, raw: object) -> "SkeletonGraphCustodyIncident":
        if cls is not SkeletonGraphCustodyIncident:
            raise SkeletonGraphCustodyIncidentError(
                "incident parser requires the exact record class"
            )
        if type(raw) is not dict:
            raise SkeletonGraphCustodyIncidentError("incident record is not an object")
        list_fields = {
            "rg_argv",
            "head_argv",
            "search_roots",
            "search_globs",
            "search_patterns",
            "target_family_task_ids",
            "same_family_calibration_task_ids",
            "evidence_statements",
            "reason_codes",
        }
        expected = set(cls.__dataclass_fields__) | {"schema"}
        if (
            any(type(key) is not str for key in raw)
            or set(raw) != expected
            or type(raw.get("schema")) is not str
            or raw.get("schema") != INCIDENT_SCHEMA
        ):
            raise SkeletonGraphCustodyIncidentError("incident record fields differ")
        if any(type(raw[name]) is not list for name in list_fields):
            raise SkeletonGraphCustodyIncidentError("incident sequence fields differ")
        values = {name: raw[name] for name in cls.__dataclass_fields__}
        for name in list_fields:
            values[name] = tuple(raw[name])
        result = cls(**values)
        _validate_frozen_incident(result)
        expected_record = build_overbroad_rg_incident()
        if result.to_data() != expected_record.to_data():
            raise SkeletonGraphCustodyIncidentError(
                "incident record differs from the frozen incident"
            )
        return result


def build_overbroad_rg_incident() -> SkeletonGraphCustodyIncident:
    values = _frozen_incident_values()
    provisional = object.__new__(SkeletonGraphCustodyIncident)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    result = SkeletonGraphCustodyIncident(
        **values, record_digest=_record_digest(provisional.content_data())
    )
    _validate_frozen_incident(result)
    return result


def _frozen_incident_values() -> dict[str, Any]:
    return {
        "incident_id": "overbroad-rg-official-program-exposure-20260810-v1",
        "incident_date": "2026-08-10",
        "shell_command": SHELL_COMMAND,
        "rg_argv": RG_ARGV,
        "head_argv": HEAD_ARGV,
        "stderr_redirect": "/dev/null",
        "search_roots": ("bongard", "downloads/ShapeBongard_V2_full"),
        "search_globs": ("*.py", "*.json", "*.md", "*.txt"),
        "search_patterns": tuple(RG_PATTERN.split("|")),
        "official_program_path": OFFICIAL_HD_ACTION_PROGRAM_PATH,
        "official_program_file_sha256": OFFICIAL_HD_ACTION_PROGRAM_FILE_SHA256,
        "official_program_size_bytes": OFFICIAL_HD_ACTION_PROGRAM_SIZE_BYTES,
        "official_program_task_count": OFFICIAL_HD_ACTION_PROGRAM_TASK_COUNT,
        "target_family_task_ids": TARGET_FAMILY_TASK_IDS,
        "same_family_calibration_task_ids": SAME_FAMILY_CALIBRATION_TASK_IDS,
        "tool_output_line_count": 85,
        "outer_original_token_count": 40_030,
        "nested_original_token_count": 2_705_621,
        "renderer_omitted_byte_count": 9_773_907,
        "evidence_statements": (
            "the target _0000 key or path was visibly materialized",
            "same-family _0002 through _0017 identifiers and cohort metadata were visible",
            "substantial literal _0019 action arrays were visible",
            "the single-line HD action-program authority streamed upstream before truncation",
            "no separate official TEST label file was visibly identified",
        ),
        "reason_codes": (
            "official_program_authority_read_before_prediction_barrier",
            "target_identifier_and_program_line_materialized",
            "same_family_calibration_programs_preexposed",
            "causal_prediction_label_barrier_broken",
        ),
        "no_png_bytes_opened": True,
        "no_zip_or_archive_bytes_opened": True,
        "no_model_or_prediction_execution": True,
        "no_files_written_by_incident_command": True,
        "generic_hd_programs_conservatively_preexposed": True,
        "official_test_programs_preexposed_if_resident": True,
        "calibration_live_launch_authorized": False,
        "target_query_authorized": False,
        "benchmark_claim_authorized": False,
        "required_terminal_outcome": "typed_custody_gap",
        "abandoned_wrapper_source_sha256": ABANDONED_WRAPPER_SOURCE_SHA256,
        "abandoned_wrapper_test_sha256": ABANDONED_WRAPPER_TEST_SHA256,
        "abandoned_wrapper_audit_finding_count": 7,
        "abandoned_wrapper_deleted_uncommitted": True,
        "incident_recorder_source_sha256": "sha256:" + source_sha256(),
    }


def _frozen_incident_record_digest() -> str:
    values = _frozen_incident_values()
    provisional = object.__new__(SkeletonGraphCustodyIncident)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return _record_digest(provisional.content_data())


def _validate_frozen_incident(value: SkeletonGraphCustodyIncident) -> None:
    if type(value) is not SkeletonGraphCustodyIncident:
        raise SkeletonGraphCustodyIncidentError(
            "incident has the wrong exact record type"
        )
    expected_values = _frozen_incident_values()
    exact_int_fields = (
        "official_program_size_bytes",
        "official_program_task_count",
        "tool_output_line_count",
        "outer_original_token_count",
        "nested_original_token_count",
        "renderer_omitted_byte_count",
        "abandoned_wrapper_audit_finding_count",
    )
    exact_bool_fields = (
        "no_png_bytes_opened",
        "no_zip_or_archive_bytes_opened",
        "no_model_or_prediction_execution",
        "no_files_written_by_incident_command",
        "generic_hd_programs_conservatively_preexposed",
        "official_test_programs_preexposed_if_resident",
        "calibration_live_launch_authorized",
        "target_query_authorized",
        "benchmark_claim_authorized",
        "abandoned_wrapper_deleted_uncommitted",
    )
    if (
        any(
            getattr(value, name) != expected
            for name, expected in expected_values.items()
        )
        or any(
            type(getattr(value, name)) is not type(expected)
            for name, expected in expected_values.items()
        )
        or any(type(getattr(value, name)) is not int for name in exact_int_fields)
        or any(type(getattr(value, name)) is not bool for name in exact_bool_fields)
        or any(
            type(item) is not str
            for name in (
                "rg_argv",
                "head_argv",
                "search_roots",
                "search_globs",
                "search_patterns",
                "target_family_task_ids",
                "same_family_calibration_task_ids",
                "evidence_statements",
                "reason_codes",
            )
            for item in getattr(value, name)
        )
        or value.target_family_task_ids != TARGET_FAMILY_TASK_IDS
        or value.same_family_calibration_task_ids
        != SAME_FAMILY_CALIBRATION_TASK_IDS
        or value.incident_recorder_source_sha256 != "sha256:" + source_sha256()
        or value.record_digest != _record_digest(value.content_data())
    ):
        raise SkeletonGraphCustodyIncidentError("frozen incident policy differs")
    _address(value.official_program_file_sha256, label="official program")
    _address(value.abandoned_wrapper_source_sha256, label="wrapper source")
    _address(value.abandoned_wrapper_test_sha256, label="wrapper test")
    _address(value.incident_recorder_source_sha256, label="incident source")
    _address(value.record_digest, label="incident record")


def load_incident_record(path: Path = INCIDENT_RECORD_PATH) -> SkeletonGraphCustodyIncident:
    raw = Path(path).read_bytes()
    if len(raw) > 256 << 10:
        raise SkeletonGraphCustodyIncidentError("incident record exceeds byte cap")
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkeletonGraphCustodyIncidentError("incident record is invalid JSON") from exc
    result = SkeletonGraphCustodyIncident.from_data(decoded)
    if raw != _canonical_record_bytes(result.to_data()):
        raise SkeletonGraphCustodyIncidentError("incident record is not canonical")
    return result


@dataclass(frozen=True, slots=True)
class SkeletonGraphIncidentTombstoneClaim:
    incident_record_digest: str
    predecessor_ledger_digest: str
    predecessor_file_sha256: str
    successor_ledger_digest: str
    successor_file_sha256: str
    successor_filename: str
    incident_event_digest: str
    incident_event_observed_at: str
    affected_task_ids: tuple[str, ...]
    fixed_campaign_intent_filename: str
    calibration_pixels_authorized: bool
    action_program_or_label_reads_authorized: bool
    target_query_support_pixels_authorized: bool
    benchmark_claim_authorized: bool
    required_terminal_outcome: str
    write_once_persistence_required: bool
    incident_recorder_source_sha256: str
    record_digest: str

    def content_data(self) -> dict[str, Any]:
        return {
            "schema": TOMBSTONE_CLAIM_SCHEMA,
            "incident_record_digest": self.incident_record_digest,
            "predecessor_ledger_digest": self.predecessor_ledger_digest,
            "predecessor_file_sha256": self.predecessor_file_sha256,
            "successor_ledger_digest": self.successor_ledger_digest,
            "successor_file_sha256": self.successor_file_sha256,
            "successor_filename": self.successor_filename,
            "incident_event_digest": self.incident_event_digest,
            "incident_event_observed_at": self.incident_event_observed_at,
            "affected_task_ids": list(self.affected_task_ids),
            "fixed_campaign_intent_filename": self.fixed_campaign_intent_filename,
            "calibration_pixels_authorized": self.calibration_pixels_authorized,
            "action_program_or_label_reads_authorized": (
                self.action_program_or_label_reads_authorized
            ),
            "target_query_support_pixels_authorized": (
                self.target_query_support_pixels_authorized
            ),
            "benchmark_claim_authorized": self.benchmark_claim_authorized,
            "required_terminal_outcome": self.required_terminal_outcome,
            "write_once_persistence_required": self.write_once_persistence_required,
            "incident_recorder_source_sha256": self.incident_recorder_source_sha256,
        }

    def to_data(self) -> dict[str, Any]:
        return {**self.content_data(), "record_digest": self.record_digest}

    @classmethod
    def from_data(cls, raw: object) -> "SkeletonGraphIncidentTombstoneClaim":
        if cls is not SkeletonGraphIncidentTombstoneClaim:
            raise SkeletonGraphCustodyIncidentError(
                "tombstone parser requires the exact claim class"
            )
        if type(raw) is not dict:
            raise SkeletonGraphCustodyIncidentError("tombstone claim is not an object")
        expected = set(cls.__dataclass_fields__) | {"schema"}
        if (
            any(type(key) is not str for key in raw)
            or type(raw.get("schema")) is not str
            or set(raw) != expected
            or raw.get("schema") != TOMBSTONE_CLAIM_SCHEMA
            or type(raw.get("affected_task_ids")) is not list
        ):
            raise SkeletonGraphCustodyIncidentError("tombstone claim fields differ")
        values = {name: raw[name] for name in cls.__dataclass_fields__}
        values["affected_task_ids"] = tuple(raw["affected_task_ids"])
        result = cls(**values)
        _validate_claim_shape(result)
        return result


def build_incident_exposure_tombstone(
    predecessor: ExposureLedger,
    *,
    incident: SkeletonGraphCustodyIncident,
    observed_at: str,
) -> tuple[ExposureLedger, SkeletonGraphIncidentTombstoneClaim]:
    _validate_frozen_incident(incident)
    _validate_exact_ledger(predecessor, label="incident predecessor")
    if (
        type(predecessor) is not ExposureLedger
        or predecessor.digest != EXPOSURE_PREDECESSOR_LEDGER_DIGEST
        or predecessor.corpus_digest != EXPOSURE_CORPUS_DIGEST
        or type(observed_at) is not str
        or _OBSERVED_AT.fullmatch(observed_at) is None
    ):
        raise SkeletonGraphCustodyIncidentError("incident predecessor or time differs")
    successor = predecessor.record(
        phase="custody-incident",
        actor="gkm-codex-custody-incident-recorder-v1",
        purpose="official-programs-preexposed-no-launch",
        task_ids=incident.target_family_task_ids,
        source=incident.record_digest,
        observed_at=observed_at,
        allow_sealed=True,
        require_unseen=False,
    )
    _validate_exact_ledger(successor, label="incident successor")
    event = successor.events[-1]
    successor_raw = successor.to_json().encode("utf-8")
    values: dict[str, Any] = {
        "incident_record_digest": incident.record_digest,
        "predecessor_ledger_digest": predecessor.digest,
        "predecessor_file_sha256": EXPOSURE_PREDECESSOR_FILE_SHA256,
        "successor_ledger_digest": successor.digest,
        "successor_file_sha256": "sha256:" + hashlib.sha256(successor_raw).hexdigest(),
        "successor_filename": successor.digest.removeprefix("sha256:")
        + ".exposure.json",
        "incident_event_digest": event.digest,
        "incident_event_observed_at": observed_at,
        "affected_task_ids": incident.target_family_task_ids,
        "fixed_campaign_intent_filename": CAMPAIGN_INTENT_FILENAME,
        "calibration_pixels_authorized": False,
        "action_program_or_label_reads_authorized": False,
        "target_query_support_pixels_authorized": False,
        "benchmark_claim_authorized": False,
        "required_terminal_outcome": "typed_custody_gap",
        "write_once_persistence_required": True,
        "incident_recorder_source_sha256": "sha256:" + source_sha256(),
    }
    provisional = object.__new__(SkeletonGraphIncidentTombstoneClaim)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    claim = SkeletonGraphIncidentTombstoneClaim(
        **values, record_digest=_record_digest(provisional.content_data())
    )
    verify_incident_tombstone_claim(
        predecessor,
        successor=successor,
        claim=claim,
        incident=incident,
    )
    return successor, claim


def verify_incident_tombstone_claim(
    predecessor: ExposureLedger,
    *,
    successor: ExposureLedger,
    claim: SkeletonGraphIncidentTombstoneClaim,
    incident: SkeletonGraphCustodyIncident,
) -> None:
    if type(incident) is not SkeletonGraphCustodyIncident:
        raise SkeletonGraphCustodyIncidentError("incident has the wrong exact type")
    _validate_exact_ledger(predecessor, label="incident predecessor")
    _validate_exact_ledger(successor, label="incident successor")
    _validate_frozen_incident(incident)
    if (
        type(claim) is not SkeletonGraphIncidentTombstoneClaim
        or predecessor.digest != EXPOSURE_PREDECESSOR_LEDGER_DIGEST
        or predecessor.corpus_digest != EXPOSURE_CORPUS_DIGEST
        or claim.predecessor_ledger_digest != predecessor.digest
        or claim.predecessor_file_sha256 != EXPOSURE_PREDECESSOR_FILE_SHA256
        or claim.incident_record_digest != _frozen_incident_record_digest()
        or claim.record_digest != _record_digest(claim.content_data())
        or claim.affected_task_ids != TARGET_FAMILY_TASK_IDS
        or type(claim.write_once_persistence_required) is not bool
        or claim.write_once_persistence_required is not True
        or claim.calibration_pixels_authorized is not False
        or claim.action_program_or_label_reads_authorized is not False
        or claim.target_query_support_pixels_authorized is not False
        or claim.benchmark_claim_authorized is not False
        or successor.events[:-1] != predecessor.events
        or len(successor.events) != len(predecessor.events) + 1
        or successor.corpus_digest != predecessor.corpus_digest
        or successor.digest != claim.successor_ledger_digest
        or claim.successor_filename
        != successor.digest.removeprefix("sha256:") + ".exposure.json"
        or claim.incident_record_digest != incident.record_digest
        or successor.events[-1].source != incident.record_digest
        or successor.events[-1].task_ids != TARGET_FAMILY_TASK_IDS
        or successor.events[-1].panel_ids != ()
        or successor.events[-1].phase != "custody-incident"
        or successor.events[-1].actor
        != "gkm-codex-custody-incident-recorder-v1"
        or successor.events[-1].purpose
        != "official-programs-preexposed-no-launch"
        or successor.events[-1].digest != claim.incident_event_digest
        or successor.events[-1].observed_at != claim.incident_event_observed_at
        or claim.fixed_campaign_intent_filename != CAMPAIGN_INTENT_FILENAME
        or claim.required_terminal_outcome != "typed_custody_gap"
        or claim.incident_recorder_source_sha256 != "sha256:" + source_sha256()
    ):
        raise SkeletonGraphCustodyIncidentError("incident tombstone claim differs")
    successor_raw = successor.to_json().encode("utf-8")
    if claim.successor_file_sha256 != "sha256:" + hashlib.sha256(
        successor_raw
    ).hexdigest():
        raise SkeletonGraphCustodyIncidentError("incident successor bytes differ")
    _validate_claim_shape(claim)


def _validate_claim_shape(claim: SkeletonGraphIncidentTombstoneClaim) -> None:
    if type(claim) is not SkeletonGraphIncidentTombstoneClaim:
        raise SkeletonGraphCustodyIncidentError(
            "tombstone claim has the wrong exact type"
        )
    for name in (
        "calibration_pixels_authorized",
        "action_program_or_label_reads_authorized",
        "target_query_support_pixels_authorized",
        "benchmark_claim_authorized",
        "write_once_persistence_required",
    ):
        if type(getattr(claim, name)) is not bool:
            raise SkeletonGraphCustodyIncidentError(
                f"tombstone claim {name} is not an exact bool"
            )
    for name in (
        "successor_filename",
        "incident_event_observed_at",
        "fixed_campaign_intent_filename",
        "required_terminal_outcome",
    ):
        if type(getattr(claim, name)) is not str:
            raise SkeletonGraphCustodyIncidentError(
                f"tombstone claim {name} is not an exact string"
            )
    for value, label in (
        (claim.incident_record_digest, "claim incident"),
        (claim.predecessor_ledger_digest, "claim predecessor"),
        (claim.predecessor_file_sha256, "claim predecessor file"),
        (claim.successor_ledger_digest, "claim successor"),
        (claim.successor_file_sha256, "claim successor file"),
        (claim.incident_event_digest, "claim event"),
        (claim.incident_recorder_source_sha256, "claim source"),
        (claim.record_digest, "claim record"),
    ):
        _address(value, label=label)
    if (
        type(claim.affected_task_ids) is not tuple
        or any(type(item) is not str for item in claim.affected_task_ids)
        or claim.affected_task_ids != TARGET_FAMILY_TASK_IDS
        or _OBSERVED_AT.fullmatch(claim.incident_event_observed_at) is None
        or claim.predecessor_ledger_digest != EXPOSURE_PREDECESSOR_LEDGER_DIGEST
        or claim.predecessor_file_sha256 != EXPOSURE_PREDECESSOR_FILE_SHA256
        or claim.incident_record_digest != _frozen_incident_record_digest()
        or claim.successor_filename
        != claim.successor_ledger_digest.removeprefix("sha256:")
        + ".exposure.json"
        or claim.fixed_campaign_intent_filename != CAMPAIGN_INTENT_FILENAME
        or claim.calibration_pixels_authorized is not False
        or claim.action_program_or_label_reads_authorized is not False
        or claim.target_query_support_pixels_authorized is not False
        or claim.benchmark_claim_authorized is not False
        or claim.write_once_persistence_required is not True
        or claim.required_terminal_outcome != "typed_custody_gap"
        or claim.incident_recorder_source_sha256 != "sha256:" + source_sha256()
        or claim.record_digest != _record_digest(claim.content_data())
    ):
        raise SkeletonGraphCustodyIncidentError("tombstone claim types differ")


def tombstone_claim_bytes(claim: SkeletonGraphIncidentTombstoneClaim) -> bytes:
    _validate_claim_shape(claim)
    verify_address = _address(claim.record_digest, label="tombstone claim")
    if verify_address != _record_digest(claim.content_data()):
        raise SkeletonGraphCustodyIncidentError("tombstone claim digest differs")
    return _canonical_record_bytes(claim.to_data())


__all__ = (
    "CAMPAIGN_INTENT_FILENAME",
    "EXPOSURE_PREDECESSOR_FILE_SHA256",
    "EXPOSURE_PREDECESSOR_LEDGER_DIGEST",
    "INCIDENT_RECORD_PATH",
    "INCIDENT_SCHEMA",
    "SkeletonGraphCustodyIncident",
    "SkeletonGraphCustodyIncidentError",
    "SkeletonGraphIncidentTombstoneClaim",
    "TARGET_FAMILY_TASK_IDS",
    "TOMBSTONE_CLAIM_SCHEMA",
    "build_incident_exposure_tombstone",
    "build_overbroad_rg_incident",
    "load_incident_record",
    "source_sha256",
    "tombstone_claim_bytes",
    "verify_incident_tombstone_claim",
)
