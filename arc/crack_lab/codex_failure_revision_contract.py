#!/usr/bin/env python3
"""Authenticate the bounded multi-turn Codex failure-revision envelope."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any


DEFAULT_ROUNDS = 1
TREATMENT_ROUNDS = 4
MAX_ROUNDS = TREATMENT_ROUNDS
TREATMENT_MINUTES_LIMIT = 300
WINDOW_ALLOCATION_SECONDS = TREATMENT_MINUTES_LIMIT * 60.0
SETTLEMENT_RESERVE_CAP_SECONDS = 30 * 60.0
SETTLEMENT_RESERVE_FRACTION = 0.10
SETTLEMENT_RESERVE_SECONDS = min(
    SETTLEMENT_RESERVE_CAP_SECONDS,
    WINDOW_ALLOCATION_SECONDS * SETTLEMENT_RESERVE_FRACTION,
)
SLICE_BUDGET_SECONDS = (
    WINDOW_ALLOCATION_SECONDS - SETTLEMENT_RESERVE_SECONDS
)
PROTOCOL_SHA256 = (
    "d44cbe80ef8b228da223942653e8f796f120124484c2c3c37fc9de35c733b3a2"
)
PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER = (
    b"GKM_PUBLIC_ACTION_PROTOCOL_VIOLATION"
)
BOUNDARY_BINDING = {
    "filesystem_boundary_policy_schema": 1,
    "filesystem_boundary_policy_sha256": (
        "7ab5447704c83f607c3f61d7fb69f9df4690b71cf5abf10e381d6e875d6be202"
    ),
    "compatibility_arena_module_sha256": (
        "9174a6ec78abea5b6c7cdc1afd49b47725cc3cf49a9e9c3c390b66f9aefd6b43"
    ),
    "compatibility_boundary_authority": "behavioral_defense_in_depth",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9_.-]+")
USAGE_FIELDS = (
    "input_tokens", "cached_input_tokens", "output_tokens",
    "reasoning_output_tokens", "observed_tokens",
)
VERIFIER_FIELDS = (
    "verifier_classification", "verification_mode",
    "verifier_error_present", "reached_before", "reached_after",
    "solved_target",
)
OUTCOME_TERMINAL_CAUSES = frozenset({
    ("infrastructure", "revision_window_exhausted"),
    ("credit_out", "campaign_guard"),
    ("infrastructure", "between_round_infrastructure"),
})
NESTED_TERMINAL_FAILURES = frozenset({
    ("taint", "public_action_protocol_violation"),
    ("taint", "host_process_introspection"),
    ("taint", "external_web_or_network"),
    ("taint", "private_runtime_introspection"),
    ("taint", "filesystem_boundary_violation"),
    ("infrastructure", "interrupted"),
    ("containment", "hard_wall_time"),
    ("evidence", "invalid_or_reused_turn_lifecycle"),
    ("credit_out", "provider_credit_out"),
    ("infrastructure", "known_transient"),
    ("infrastructure", "unknown_cli"),
    ("infrastructure", "launch_error"),
})
ROUND_KEYS = frozenset({
    "round_index", "turn_kind", "thread_id", "target_level",
    "termination_kind", "allocation_policy", "allocation_basis_seconds",
    "rounds_left_at_launch", "allocation_seconds", "minutes_limit",
    "duration_seconds", "allocation_expired", "timed_out", "returncode",
    "launch_error", "interrupted", "process_group_stop_attempted",
    "process_group_quiesced", "surviving_process_group",
    "protected_transcript_status", "protected_diagnostics_status",
    "protected_transcript_error", "round_transcript_offset",
    "round_transcript_size", "round_transcript_sha256",
    "round_diagnostics_offset", "round_diagnostics_size",
    "round_diagnostics_sha256", "task_feedback_sha256",
    "failure_revision_protocol_sha256", *BOUNDARY_BINDING,
    "failure_class", "failure_detail_class",
    "public_action_protocol_violation", "filesystem_boundary_violation",
    "filesystem_boundary_violation_reason", "taint_verdict",
    *VERIFIER_FIELDS, "thread_started_events", "turn_completed_events",
    "usage_reported", *USAGE_FIELDS,
})
OUTCOME_ROUND_KEYS = frozenset({
    "target_level", "round_index", "turn_kind", "termination_kind",
    *VERIFIER_FIELDS,
})
OUTCOME_RECORD_BINDING_FIELDS = (
    "run_label", "model", "reasoning_effort", "game", "target_level",
    "frontier_binding_schema", "parent_checkpoint_sha256",
    "parent_source_tree_sha256", "frontier_sha256", "reached",
    "parent_action_count",
)
AGGREGATE_MARKERS = frozenset({
    "failure_revision_protocol_sha256", "rounds_used", "rounds_max",
    "terminal_round_index", "rounds_evaluated", "completed_round_count",
    "timeout_round_count", "timeout_round_indices", "rounds",
    "aggregate_terminal_status", "thread_id_authority",
    "window_allocation_seconds", "slice_budget_seconds",
    "settlement_reserve_seconds", "returncode_authority",
})
REQUIRED_TOP_KEYS = AGGREGATE_MARKERS | frozenset({
    "target_level", "reached", "transcript", "diagnostics",
    "rounds_left_at_launch",
    "allocation_policy", "termination_kind", "returncode",
    "duration_seconds", "minutes_limit", "allocation_expired", "timed_out",
    "process_group_stop_attempted", "process_group_quiesced",
    "surviving_process_group", "public_action_protocol_violation",
    "filesystem_boundary_violation", "filesystem_boundary_violation_reason",
    "taint_verdict", "failure_class", "failure_detail_class",
    "interrupted", "thread_id", "task_feedback_sha256",
    "protected_transcript_status", "protected_transcript_size",
    "protected_transcript_sha256", "protected_transcript_error",
    "protected_diagnostics_status",
    "protected_diagnostics_size", "protected_diagnostics_sha256",
    "thread_started_events", "turn_completed_events", "usage_reported",
    *VERIFIER_FIELDS, *USAGE_FIELDS, *BOUNDARY_BINDING,
})


class ContractError(ValueError):
    """One treatment record does not match its authenticated contract."""


@dataclass(frozen=True)
class Aggregate:
    rounds_used: int
    rounds_max: int
    rounds_evaluated: int
    rounds: tuple[dict[str, Any], ...]
    terminal_failure: bool
    timed_out: bool


def _fail(message: str) -> None:
    raise ContractError(message)


def _int(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _number(value: object, *, positive: bool = False) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _sha(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _boundary_taint_detail(reason: str) -> str:
    normalized = reason.lower()
    if (
        "host_process_introspection" in normalized
        or "host process introspection" in normalized
    ):
        return "host_process_introspection"
    if "external web/network" in normalized or "external_network" in normalized:
        return "external_web_or_network"
    if (
        "private game/runtime introspection" in normalized
        or "runtime_introspection" in normalized
    ):
        return "private_runtime_introspection"
    if "public action protocol violation" in normalized:
        return "public_action_protocol_violation"
    return "filesystem_boundary_violation"


def _terminal_top_failure(
    nested_failure: tuple[object, object],
) -> tuple[str, str] | None:
    failure_class, detail = nested_failure
    if failure_class == "taint":
        return "taint", "terminal_taint"
    if failure_class == "evidence":
        return "evidence", "terminal_evidence"
    if failure_class == "credit_out":
        return "credit_out", "campaign_guard"
    if failure_class == "containment":
        return "containment", "hard_wall_time"
    if nested_failure == ("infrastructure", "interrupted"):
        return "infrastructure", "interrupted"
    if failure_class == "infrastructure":
        return "infrastructure", "terminal_infrastructure"
    return None


def has_aggregate_markers(record: dict[str, Any]) -> bool:
    return bool(
        AGGREGATE_MARKERS & set(record)
        or any(
            key.startswith("failure_revision_")
            or key.startswith("round_")
            or key.startswith("rounds_")
            for key in record
        )
    )


def _usage_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    usage: dict[str, Any] = {field: 0 for field in USAGE_FIELDS}
    usage["usage_reported"] = False
    for event in events:
        if event.get("type") != "turn.completed" or not isinstance(
            event.get("usage"), dict
        ):
            continue
        for field in USAGE_FIELDS[:-1]:
            value = event["usage"].get(field)
            if _int(value):
                usage[field] = value
        usage["usage_reported"] = True
    usage["observed_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _slice_events(payload: bytes) -> list[dict[str, Any]]:
    if payload and not payload.endswith(b"\n"):
        _fail("aggregate transcript slice lacks a line boundary")
    events = []
    for line in payload.splitlines():
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(
                "aggregate transcript slice is not strict JSONL"
            ) from exc
        if not isinstance(event, dict):
            _fail("aggregate transcript slice contains a non-object")
        events.append(event)
    return events


def _validate_verifier(
    row: dict[str, Any], *, target_level: int | None,
    reached_before: int | None,
) -> bool:
    values = [row.get(field) for field in VERIFIER_FIELDS]
    if all(value is None for value in values):
        return False
    if any(value is None for value in values):
        _fail("aggregate verifier metadata is partial")
    classification, mode, error_present, before, after, solved = values
    target = target_level
    if target is None:
        target = row.get("target_level")
    if any((
        classification not in {
            "target_reached", "verifier_error", "partial_progress",
            "no_progress",
        },
        not isinstance(mode, str) or not mode,
        type(error_present) is not bool,
        not _int(before), not _int(after), not _int(target, minimum=1),
        type(solved) is not bool,
        reached_before is not None and before != reached_before,
        after < before if _int(before) and _int(after) else True,
        solved != (after >= target)
        if _int(after) and _int(target, minimum=1) else True,
    )):
        _fail("aggregate verifier metadata is malformed")
    expected = (
        "target_reached" if solved else
        "verifier_error" if error_present else
        "partial_progress" if after > before else "no_progress"
    )
    if classification != expected:
        _fail("aggregate verifier classification is incoherent")
    return True


def validate_exec(
    record: dict[str, Any], *,
    expected_rounds_max: int | None = None,
    expected_protocol_sha256: str = PROTOCOL_SHA256,
    expected_boundary_binding: dict[str, Any] = BOUNDARY_BINDING,
    target_level: int | None = None,
    reached_before: int | None = None,
    transcript_payload: bytes | None = None,
    diagnostics_payload: bytes | None = None,
    require_evidence: bool = False,
) -> Aggregate | None:
    """Validate one aggregate and, when supplied, its exact sealed bytes."""

    if not has_aggregate_markers(record):
        return None
    if not REQUIRED_TOP_KEYS.issubset(record):
        _fail("aggregate is missing required top-level fields")
    if any(
        key.startswith(("failure_revision_", "round_", "rounds_"))
        and key not in REQUIRED_TOP_KEYS
        for key in record
    ):
        _fail("aggregate carries an unknown revision-control field")
    if any((
        not _int(record.get("target_level"), minimum=1),
        target_level is not None and record.get("target_level") != target_level,
        not _int(record.get("reached")),
        reached_before is not None and record.get("reached") != reached_before,
        not isinstance(record.get("transcript"), str),
        SAFE_COMPONENT_RE.fullmatch(record.get("transcript") or "") is None,
        not isinstance(record.get("diagnostics"), str),
        SAFE_COMPONENT_RE.fullmatch(record.get("diagnostics") or "") is None,
        record.get("diagnostics") == record.get("transcript"),
        record.get("protected_transcript_error") is not None,
        not _sha(record.get("task_feedback_sha256")),
    )):
        _fail("aggregate top-level binding is malformed")
    used, maximum, evaluated = (
        record.get("rounds_used"), record.get("rounds_max"),
        record.get("rounds_evaluated"),
    )
    if (
        expected_rounds_max is not None
        and expected_rounds_max != TREATMENT_ROUNDS
    ):
        _fail("aggregate validator was configured for a noncanonical maximum")
    canonical_maximum = TREATMENT_ROUNDS
    rounds = record.get("rounds")
    if any((
        not _int(used, minimum=1), not _int(maximum, minimum=2),
        maximum > MAX_ROUNDS if _int(maximum, minimum=2) else True,
        maximum != canonical_maximum,
        used > maximum if _int(used, minimum=1) and _int(maximum) else True,
        record.get("terminal_round_index") != used,
        not _int(evaluated), evaluated > used if _int(evaluated) and _int(used) else True,
        not isinstance(rounds, list) or len(rounds) != used,
        record.get("failure_revision_protocol_sha256")
        != expected_protocol_sha256,
        record.get("allocation_policy") != "hard",
    )):
        _fail("aggregate count/protocol schema is malformed")
    for field, expected in expected_boundary_binding.items():
        if record.get(field) != expected:
            _fail("aggregate boundary binding changed")

    whole_payloads = {
        "transcript": transcript_payload,
        "diagnostics": diagnostics_payload,
    }
    for kind, payload in whole_payloads.items():
        if any((
            record.get(f"protected_{kind}_status") != "sealed",
            not _int(record.get(f"protected_{kind}_size")),
            not _sha(record.get(f"protected_{kind}_sha256")),
            require_evidence and payload is None,
        )):
            _fail("aggregate protected evidence metadata is malformed")
        if payload is not None and any((
            len(payload) != record[f"protected_{kind}_size"],
            hashlib.sha256(payload).hexdigest()
            != record[f"protected_{kind}_sha256"],
        )):
            _fail("aggregate protected evidence changed")

    offsets = {"transcript": 0, "diagnostics": 0}
    seen_threads: set[str] = set()
    evaluated_prefix = 0
    timeout_indices = []
    completed_count = 0
    normalized: list[dict[str, Any]] = []
    nested_duration = 0.0
    prior_allocation_basis: float | None = None
    for index, raw in enumerate(rounds, 1):
        if not isinstance(raw, dict) or set(raw) != ROUND_KEYS:
            _fail("aggregate round schema is ambiguous")
        if raw.get("round_index") != index or raw.get("turn_kind") != (
            "proposal" if index == 1 else "revision"
        ):
            _fail("aggregate round ordering is malformed")
        if raw.get("failure_revision_protocol_sha256") != expected_protocol_sha256:
            _fail("aggregate round protocol binding changed")
        for field, expected in expected_boundary_binding.items():
            if raw.get(field) != expected:
                _fail("aggregate round boundary binding changed")
        if not _sha(raw.get("task_feedback_sha256")):
            _fail("aggregate round feedback binding is malformed")
        allocation_basis = raw.get("allocation_basis_seconds")
        allocation_seconds = raw.get("allocation_seconds")
        rounds_left = raw.get("rounds_left_at_launch")
        basis_milliseconds = (
            round(allocation_basis * 1000)
            if _number(allocation_basis, positive=True) else None
        )
        if any((
            not _int(raw.get("target_level"), minimum=1),
            target_level is not None and raw.get("target_level") != target_level,
            raw.get("allocation_policy") != "hard",
            not _number(allocation_basis, positive=True),
            not _int(rounds_left, minimum=1),
            rounds_left != maximum - index + 1,
            basis_milliseconds is None,
            abs(allocation_basis * 1000 - basis_milliseconds) > 1e-6
            if basis_milliseconds is not None else True,
            not _number(allocation_seconds, positive=True),
            allocation_seconds != basis_milliseconds // rounds_left / 1000
            if basis_milliseconds is not None and _int(rounds_left, minimum=1)
            else True,
            not _number(raw.get("minutes_limit"), positive=True),
            round(raw.get("minutes_limit") * 60, 3) != allocation_seconds
            if _number(raw.get("minutes_limit"), positive=True)
            and _number(allocation_seconds, positive=True) else True,
            allocation_basis > record.get("slice_budget_seconds")
            if _number(allocation_basis, positive=True)
            and _number(record.get("slice_budget_seconds"), positive=True)
            else True,
            prior_allocation_basis is not None
            and allocation_basis > prior_allocation_basis,
            not _number(raw.get("duration_seconds")),
            type(raw.get("allocation_expired")) is not bool,
            type(raw.get("timed_out")) is not bool,
            raw.get("allocation_expired") is not raw.get("timed_out"),
            type(raw.get("interrupted")) is not bool,
            type(raw.get("process_group_stop_attempted")) is not bool,
            type(raw.get("process_group_quiesced")) is not bool,
            type(raw.get("surviving_process_group")) is not bool,
            type(raw.get("public_action_protocol_violation")) is not bool,
            type(raw.get("filesystem_boundary_violation")) is not bool,
            raw.get("taint_verdict") not in {"clean", "tainted"},
            raw.get("returncode") is not None
            and type(raw.get("returncode")) is not int,
            raw.get("launch_error") is not None
            and not isinstance(raw.get("launch_error"), str),
            raw.get("filesystem_boundary_violation") is False
            and raw.get("filesystem_boundary_violation_reason") is not None,
            raw.get("filesystem_boundary_violation") is True
            and not isinstance(
                raw.get("filesystem_boundary_violation_reason"), str
            ),
            raw.get("process_group_quiesced") is not True,
            raw.get("surviving_process_group") is not False,
            raw.get("protected_transcript_status") != "sealed",
            raw.get("protected_diagnostics_status") != "sealed",
            raw.get("protected_transcript_error") is not None,
            raw.get("taint_verdict") != (
                "tainted"
                if (
                    raw.get("public_action_protocol_violation") is True
                    or raw.get("filesystem_boundary_violation") is True
                )
                else "clean"
            ),
            (
                raw.get("public_action_protocol_violation") is True
                or raw.get("filesystem_boundary_violation") is True
            ) and raw.get("taint_verdict") != "tainted",
        )):
            _fail("aggregate round control metadata is malformed")
        prior_allocation_basis = float(allocation_basis)
        nested_duration += float(raw["duration_seconds"])
        slice_protocol_violation = False
        for kind, payload in whole_payloads.items():
            offset, size, digest = (
                raw.get(f"round_{kind}_offset"),
                raw.get(f"round_{kind}_size"),
                raw.get(f"round_{kind}_sha256"),
            )
            if offset != offsets[kind] or not _int(size) or not _sha(digest):
                _fail("aggregate evidence slice schema is malformed")
            if payload is not None:
                selected = payload[offset:offset + size]
                if len(selected) != size or hashlib.sha256(selected).hexdigest() != digest:
                    _fail("aggregate evidence slice changed")
                slice_protocol_violation = bool(
                    slice_protocol_violation
                    or PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER in selected
                )
            offsets[kind] += size
        if any(
            offsets[kind] > record[f"protected_{kind}_size"]
            for kind in offsets
        ):
            _fail("aggregate evidence slices exceed the sealed pair")
        if (
            all(payload is not None for payload in whole_payloads.values())
            and raw.get("public_action_protocol_violation")
            is not slice_protocol_violation
        ):
            _fail("aggregate evidence protocol-violation projection changed")

        started = raw.get("thread_started_events")
        completed = raw.get("turn_completed_events")
        thread_id = raw.get("thread_id")
        if any((
            not _int(started), not _int(completed), started not in {0, 1},
            completed not in {0, 1}, completed > started,
            thread_id is not None and (
                not isinstance(thread_id, str)
                or SAFE_COMPONENT_RE.fullmatch(thread_id) is None
            ),
            started == 1 and thread_id is None,
            started == 0 and thread_id is not None,
        )):
            _fail("aggregate round lifecycle is malformed")
        events: list[dict[str, Any]] | None = None
        if transcript_payload is not None:
            start = raw["round_transcript_offset"]
            events = _slice_events(
                transcript_payload[start:start + raw["round_transcript_size"]]
            )
            actual_ids = [
                event.get("thread_id") for event in events
                if event.get("type") == "thread.started"
            ]
            if any(
                not isinstance(value, str)
                or SAFE_COMPONENT_RE.fullmatch(value) is None
                for value in actual_ids
            ) or any((
                len(actual_ids) != started,
                sum(event.get("type") == "turn.completed" for event in events)
                != completed,
                (actual_ids[-1] if actual_ids else None) != thread_id,
            )):
                _fail("aggregate transcript lifecycle changed")
            expected_usage = _usage_from_events(events)
            if any(raw.get(field) != expected_usage[field] for field in (
                "usage_reported", *USAGE_FIELDS,
            )):
                _fail("aggregate transcript usage changed")
        if thread_id is not None:
            if thread_id in seen_threads:
                _fail("aggregate transcript reused a thread id")
            seen_threads.add(thread_id)
        for field in USAGE_FIELDS:
            if not _int(raw.get(field)):
                _fail("aggregate round usage is malformed")
        if raw.get("observed_tokens") != (
            raw.get("input_tokens") + raw.get("output_tokens")
        ) or type(raw.get("usage_reported")) is not bool:
            _fail("aggregate round usage is incoherent")

        termination = raw.get("termination_kind")
        failure_class, failure_detail = (
            raw.get("failure_class"), raw.get("failure_detail_class")
        )
        if termination == "completed":
            completed_count += 1
            if any((
                failure_class is not None, failure_detail is not None,
                started != 1, completed != 1,
                raw.get("allocation_expired") is not False,
                raw.get("timed_out") is not False,
                raw.get("returncode") != 0,
                raw.get("launch_error") is not None,
                raw.get("interrupted") is not False,
                raw.get("process_group_stop_attempted") is not False,
                raw.get("process_group_quiesced") is not True,
                raw.get("surviving_process_group") is not False,
                raw.get("protected_transcript_status") != "sealed",
                raw.get("protected_diagnostics_status") != "sealed",
                raw.get("protected_transcript_error") is not None,
                raw.get("public_action_protocol_violation") is not False,
                raw.get("filesystem_boundary_violation") is not False,
                raw.get("filesystem_boundary_violation_reason") is not None,
                raw.get("taint_verdict") != "clean",
            )):
                _fail("completed aggregate round is not clean")
        elif termination == "terminal_failure":
            if index != used or not isinstance(failure_class, str) or not failure_class:
                _fail("aggregate terminal failure is misplaced")
            if failure_detail is not None and (
                not isinstance(failure_detail, str) or not failure_detail
            ):
                _fail("aggregate terminal failure detail is malformed")
            nested_failure = (failure_class, failure_detail)
            if nested_failure not in NESTED_TERMINAL_FAILURES:
                _fail("aggregate terminal failure class is not canonical")
            protocol_taint = (
                raw.get("public_action_protocol_violation") is True
            )
            boundary_taint = (
                raw.get("filesystem_boundary_violation") is True
            )
            boundary_reason = raw.get(
                "filesystem_boundary_violation_reason"
            )
            if failure_class == "taint":
                if not (protocol_taint or boundary_taint):
                    _fail("aggregate taint failure lacks a taint marker")
                if failure_detail == "public_action_protocol_violation":
                    if not (
                        protocol_taint
                        or (
                            boundary_taint
                            and isinstance(boundary_reason, str)
                            and _boundary_taint_detail(boundary_reason)
                            == failure_detail
                        )
                    ):
                        _fail("aggregate protocol taint detail disagrees")
                elif any((
                    protocol_taint,
                    not boundary_taint,
                    not isinstance(boundary_reason, str),
                    _boundary_taint_detail(boundary_reason)
                    != failure_detail
                    if isinstance(boundary_reason, str) else True,
                )):
                    _fail("aggregate boundary taint detail disagrees")
            elif protocol_taint or boundary_taint:
                _fail("aggregate non-taint failure carries taint markers")
            if raw.get("interrupted") is not (
                nested_failure == ("infrastructure", "interrupted")
            ):
                _fail("aggregate interruption classification disagrees")
            if (
                (nested_failure == ("infrastructure", "launch_error"))
                is not isinstance(raw.get("launch_error"), str)
            ):
                _fail("aggregate launch-error classification disagrees")
            if any((
                nested_failure == ("containment", "hard_wall_time")
                and raw.get("timed_out") is not True,
                nested_failure in {
                    ("credit_out", "provider_credit_out"),
                    ("infrastructure", "known_transient"),
                    ("infrastructure", "unknown_cli"),
                }
                and (
                    type(raw.get("returncode")) is not int
                    or raw.get("returncode") == 0
                ),
                nested_failure == ("infrastructure", "launch_error")
                and raw.get("returncode") is not None,
                nested_failure
                == ("evidence", "invalid_or_reused_turn_lifecycle")
                and (
                    started == completed == 1
                    or raw.get("returncode") != 0
                ),
                nested_failure in {
                    ("evidence", "invalid_or_reused_turn_lifecycle"),
                    ("credit_out", "provider_credit_out"),
                    ("infrastructure", "known_transient"),
                    ("infrastructure", "unknown_cli"),
                    ("infrastructure", "launch_error"),
                }
                and raw.get("process_group_stop_attempted") is not False,
            )):
                _fail("aggregate terminal failure controls disagree")
        else:
            _fail("aggregate round termination kind is unknown")
        if raw.get("timed_out") is True:
            timeout_indices.append(index)
            if any((
                termination != "terminal_failure", index != used,
                started not in {0, 1}, completed not in {0, 1},
                started == 0 and completed != 0,
                raw.get("allocation_expired") is not True,
                failure_class != "containment",
                failure_detail != "hard_wall_time",
                raw.get("process_group_stop_attempted") is not True,
                raw.get("process_group_quiesced") is not True,
                raw.get("surviving_process_group") is not False,
                raw.get("protected_transcript_status") != "sealed",
                raw.get("protected_diagnostics_status") != "sealed",
                raw.get("protected_transcript_error") is not None,
                raw.get("public_action_protocol_violation") is not False,
                raw.get("filesystem_boundary_violation") is not False,
                raw.get("filesystem_boundary_violation_reason") is not None,
                raw.get("taint_verdict") != "clean",
            )):
                _fail("aggregate hard timeout is not terminal and sealed")
        is_evaluated = _validate_verifier(
            raw, target_level=target_level, reached_before=reached_before
        )
        if is_evaluated:
            if evaluated_prefix != index - 1 or termination != "completed":
                _fail("aggregate verifier metadata is not a completed prefix")
            evaluated_prefix += 1
        normalized.append(raw)

    if any(offsets[kind] != record[f"protected_{kind}_size"] for kind in offsets):
        _fail("aggregate evidence slices do not exhaust the sealed pair")
    if all(payload is not None for payload in whole_payloads.values()):
        whole_protocol_violation = any(
            PUBLIC_ACTION_PROTOCOL_VIOLATION_MARKER in payload
            for payload in whole_payloads.values()
            if payload is not None
        )
        if record.get("public_action_protocol_violation") is not (
            whole_protocol_violation
        ):
            _fail("aggregate whole-evidence protocol projection changed")
    if evaluated_prefix != evaluated:
        _fail("aggregate evaluated-round count disagrees")
    terminal = normalized[-1]
    if record.get("aggregate_terminal_status") not in {
        "clean", "terminal_failure",
    }:
        _fail("aggregate terminal status is unknown")
    terminal_failure = record.get("aggregate_terminal_status") == "terminal_failure"
    if record.get("termination_kind") != (
        "terminal_failure" if terminal_failure else "completed"
    ) or ((record.get("failure_class") is None) == terminal_failure):
        _fail("aggregate terminal status is incoherent")
    if not terminal_failure and evaluated != used:
        _fail("clean aggregate has an unevaluated round")
    if terminal_failure and evaluated not in {used, used - 1}:
        _fail("terminal aggregate verifier prefix is malformed")
    if terminal_failure and any((
        not isinstance(record.get("failure_class"), str),
        not record.get("failure_class"),
        not isinstance(record.get("failure_detail_class"), str),
        not record.get("failure_detail_class"),
    )):
        _fail("aggregate terminal failure classification is malformed")
    if terminal_failure:
        top_failure = (
            record.get("failure_class"),
            record.get("failure_detail_class"),
        )
        if terminal.get("termination_kind") == "terminal_failure":
            expected_top_failure = _terminal_top_failure((
                terminal.get("failure_class"),
                terminal.get("failure_detail_class"),
            ))
            if top_failure != expected_top_failure:
                _fail("aggregate terminal failure projection disagrees")
        elif any((
            top_failure not in OUTCOME_TERMINAL_CAUSES,
            used >= maximum,
            evaluated != used,
            completed_count != used,
        )):
            _fail("aggregate completed-prefix terminal cause is not canonical")
    if timeout_indices and any((
        timeout_indices != [used], not terminal_failure,
        evaluated != used - 1, record.get("timed_out") is not True,
        record.get("failure_class") != "containment",
        record.get("failure_detail_class") != "hard_wall_time",
    )):
        _fail("aggregate timeout continuation is forbidden")
    if record.get("completed_round_count") != completed_count or any((
        record.get("timeout_round_count") != len(timeout_indices),
        record.get("timeout_round_indices") != timeout_indices,
        not _number(record.get("duration_seconds")),
        round(nested_duration, 3) != record.get("duration_seconds"),
        record.get("minutes_limit") != TREATMENT_MINUTES_LIMIT,
        record.get("window_allocation_seconds")
        != WINDOW_ALLOCATION_SECONDS,
        record.get("slice_budget_seconds") != SLICE_BUDGET_SECONDS,
        record.get("settlement_reserve_seconds")
        != SETTLEMENT_RESERVE_SECONDS,
        record.get("returncode") is not None,
        record.get("returncode_authority") != "host_aggregate",
    )):
        _fail("aggregate campaign totals are incoherent")
    for field in ("thread_started_events", "turn_completed_events", *USAGE_FIELDS):
        if record.get(field) != sum(row[field] for row in normalized):
            _fail("aggregate lifecycle/usage totals disagree")
    if record.get("usage_reported") is not all(
        row["usage_reported"] is True for row in normalized
    ):
        _fail("aggregate usage-reporting total disagrees")
    if record.get("rounds_left_at_launch") != terminal.get(
        "rounds_left_at_launch"
    ):
        _fail("aggregate terminal fair-share projection disagrees")
    for field in ("task_feedback_sha256", *VERIFIER_FIELDS):
        if record.get(field) != terminal.get(field):
            _fail("aggregate terminal projection disagrees")
    nested_allocation_expired = any(
        row["allocation_expired"] for row in normalized
    )
    between_round_window_exhausted = bool(
        terminal_failure
        and record.get("failure_class") == "infrastructure"
        and record.get("failure_detail_class")
        == "revision_window_exhausted"
        and evaluated == completed_count == used
        and not timeout_indices
        and record.get("taint_verdict") == "clean"
        and record.get("interrupted") is False
    )
    if any((
        record.get("allocation_expired") is not (
            nested_allocation_expired or between_round_window_exhausted
        ),
        record.get("timed_out") is not bool(timeout_indices),
        record.get("process_group_stop_attempted") is not any(
            row["process_group_stop_attempted"] for row in normalized
        ),
        record.get("process_group_quiesced") is not all(
            row["process_group_quiesced"] for row in normalized
        ),
        record.get("surviving_process_group") is not any(
            row["surviving_process_group"] for row in normalized
        ),
        record.get("public_action_protocol_violation") is not any(
            row["public_action_protocol_violation"] for row in normalized
        ),
        record.get("filesystem_boundary_violation") is not any(
            row["filesystem_boundary_violation"] for row in normalized
        ),
        record.get("filesystem_boundary_violation_reason") != next((
            row["filesystem_boundary_violation_reason"]
            for row in normalized
            if row["filesystem_boundary_violation_reason"] is not None
        ), None),
        record.get("taint_verdict") != (
            "tainted" if any(
                row["taint_verdict"] == "tainted" for row in normalized
            ) else "clean"
        ),
        record.get("interrupted") is not any(
            row["interrupted"] for row in normalized
        ),
    )):
        _fail("aggregate control projection disagrees")
    top_thread = record.get("thread_id")
    if not isinstance(top_thread, str) or SAFE_COMPONENT_RE.fullmatch(top_thread) is None:
        _fail("aggregate thread id is unsafe")
    authority = record.get("thread_id_authority")
    if authority == "terminal_provider_thread":
        if top_thread != terminal.get("thread_id") or terminal.get("thread_started_events") != 1:
            _fail("aggregate provider-thread authority disagrees")
    elif authority == "host_aggregate_fallback":
        if terminal.get("thread_started_events") == 1:
            _fail("aggregate fallback-thread authority is unnecessary")
        binding = "\0".join((
            str(record.get("transcript") or ""),
            str(record.get("protected_transcript_sha256") or ""),
        )).encode("utf-8")
        if top_thread != "failure-revision-" + hashlib.sha256(binding).hexdigest():
            _fail("aggregate fallback-thread authority disagrees")
    else:
        _fail("aggregate thread authority is unknown")
    return Aggregate(
        rounds_used=used, rounds_max=maximum, rounds_evaluated=evaluated,
        rounds=tuple(normalized), terminal_failure=terminal_failure,
        timed_out=bool(timeout_indices),
    )


def validate_outcome(
    record: dict[str, Any], outcome: dict[str, Any], aggregate: Aggregate,
    *, target_level: int, reached_before: int,
) -> None:
    """Authenticate the evaluated-prefix projection in one bound outcome."""

    metadata = outcome.get("failure_revision_rounds")
    if aggregate.timed_out:
        _fail("a hard-timeout aggregate cannot append a level outcome")
    if any((
        outcome.get("thread_id") != record.get("thread_id"),
        outcome.get("codex_exec_transcript") != record.get("transcript"),
        outcome.get("target_level") != target_level,
        outcome.get("reached_before") != reached_before,
        outcome.get("taint_verdict") != "clean",
        any(
            outcome.get(field) != record.get(field)
            for field in OUTCOME_RECORD_BINDING_FIELDS
            if field in record
        ),
    )):
        _fail("aggregate outcome top-level binding changed")
    if not isinstance(metadata, list) or len(metadata) != aggregate.rounds_evaluated:
        _fail("aggregate outcome does not cover its evaluated prefix")
    solved_metadata = False
    for index, raw in enumerate(metadata, 1):
        if not isinstance(raw, dict) or set(raw) != OUTCOME_ROUND_KEYS:
            _fail("aggregate outcome round schema is ambiguous")
        if raw.get("target_level") != target_level:
            _fail("aggregate outcome target changed")
        bound = aggregate.rounds[index - 1]
        if any(raw.get(field) != bound.get(field) for field in (
            "round_index", "turn_kind", "termination_kind", *VERIFIER_FIELDS,
        )) or raw.get("reached_before") != reached_before:
            _fail("aggregate outcome round binding changed")
        if solved_metadata or (raw["solved_target"] and index != len(metadata)):
            _fail("aggregate outcome continued after a solved round")
        solved_metadata = raw["solved_target"]
    if type(outcome.get("solved_target")) is not bool:
        _fail("aggregate outcome solved verdict is malformed")
    solved = outcome["solved_target"]
    terminal_evaluated = aggregate.rounds[aggregate.rounds_evaluated - 1]
    if any((
        outcome.get("reached_after")
        != terminal_evaluated.get("reached_after"),
        solved is not terminal_evaluated.get("solved_target"),
        type(outcome.get("winning_path_present")) is not bool,
        solved and outcome.get("winning_path_present") is not True,
        (
            not _int(outcome.get("winning_marginal_C"))
            if solved else outcome.get("winning_marginal_C") is not None
        ),
    )):
        _fail("aggregate outcome terminal projection changed")
    if aggregate.terminal_failure and any((
        solved,
        aggregate.rounds_used >= aggregate.rounds_max,
        aggregate.rounds_evaluated != aggregate.rounds_used,
        record.get("completed_round_count") != aggregate.rounds_used,
        (record.get("failure_class"), record.get("failure_detail_class"))
        not in OUTCOME_TERMINAL_CAUSES,
        any(
            row.get("termination_kind") != "completed"
            or row.get("thread_started_events") != 1
            or row.get("turn_completed_events") != 1
            or row.get("protected_transcript_status") != "sealed"
            or row.get("protected_diagnostics_status") != "sealed"
            or row.get("process_group_quiesced") is not True
            or row.get("failure_class") is not None
            or row.get("taint_verdict") != "clean"
            or row.get("solved_target") is not False
            for row in aggregate.rounds
        ),
    )):
        _fail("terminal aggregate is not eligible for a clean false outcome")
    if solved_metadata and not solved:
        _fail("aggregate solved prefix has an unsolved outcome")
    if solved and not solved_metadata and not aggregate.terminal_failure:
        _fail("clean aggregate solve lacks verifier authority")
    if not solved and not aggregate.terminal_failure and (
        aggregate.rounds_used != aggregate.rounds_max
    ):
        _fail("clean no-progress aggregate stopped before exhausting rounds")


def promotion_authority(record: dict[str, Any], aggregate: Aggregate) -> dict[str, Any]:
    """Project exactly the authority that a solved manifest must contain."""

    if any((
        aggregate.terminal_failure,
        aggregate.timed_out,
        aggregate.rounds_evaluated != aggregate.rounds_used,
        record.get("completed_round_count") != aggregate.rounds_used,
        record.get("timeout_round_count") != 0,
        record.get("timeout_round_indices") != [],
        record.get("solved_target") is not True,
        any(
            row.get("termination_kind") != "completed"
            or row.get("solved_target") is not (index == aggregate.rounds_used)
            for index, row in enumerate(aggregate.rounds, 1)
        ),
    )):
        _fail("winning aggregate lacks clean completed-round authority")

    return {
        "protocol_sha256": record["failure_revision_protocol_sha256"],
        "allocation_policy": "hard",
        "window_allocation_seconds": record["window_allocation_seconds"],
        "slice_budget_seconds": record["slice_budget_seconds"],
        "settlement_reserve_seconds": record["settlement_reserve_seconds"],
        "rounds_used": aggregate.rounds_used,
        "rounds_max": aggregate.rounds_max,
        "rounds_evaluated": aggregate.rounds_used,
        "completed_round_count": aggregate.rounds_used,
        "timeout_round_count": 0,
        "timeout_round_indices": [],
        "terminal_round_index": aggregate.rounds_used,
        "terminal_thread_id": record["thread_id"],
        "thread_id_authority": record["thread_id_authority"],
        "transcript_size": record["protected_transcript_size"],
        "transcript_sha256": record["protected_transcript_sha256"],
        "diagnostics_size": record["protected_diagnostics_size"],
        "diagnostics_sha256": record["protected_diagnostics_sha256"],
        "rounds": [{
            "round_index": row["round_index"],
            "thread_id": row["thread_id"],
            "task_feedback_sha256": row["task_feedback_sha256"],
            "termination_kind": "completed",
            "allocation_basis_seconds": row["allocation_basis_seconds"],
            "rounds_left_at_launch": row["rounds_left_at_launch"],
            "allocation_seconds": row["allocation_seconds"],
            "minutes_limit": row["minutes_limit"],
            "duration_seconds": row["duration_seconds"],
            "verifier_classification": row["verifier_classification"],
            "verification_mode": row["verification_mode"],
            "reached_before": row["reached_before"],
            "reached_after": row["reached_after"],
            "solved_target": row["solved_target"],
            "thread_started_events": 1,
            "turn_completed_events": 1,
            **{
                f"{kind}_{field}": row[f"round_{kind}_{field}"]
                for kind in ("transcript", "diagnostics")
                for field in ("offset", "size", "sha256")
            },
        } for row in aggregate.rounds],
    }
