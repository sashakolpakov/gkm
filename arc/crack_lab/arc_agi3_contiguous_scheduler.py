#!/usr/bin/env python3
"""Authoritative scheduler and independent audit for an ARC-AGI-3 campaign.

The production runner is an executor.  This module is the only authority for
frontier ordering, effort/allocation escalation, WIP selection, terminal lane
classification, supervision-cycle policy, and finite budget reservation.  A
runner must append the returned ``SCHEDULER_DECISION`` before creating a
generation directory, process, container, or network request.  The following
``ATTEMPT_RESERVED`` event consumes that exact decision once.

The audit command does not accept test names, outcomes, or a caller-supplied
PASS.  It reopens the immutable hash-chained journal, control files, source
evidence, decisions, reservations, settlements, and promotion transitions.
Its PASS is deliberately policy/accounting scoped: it never authorizes WIP
reuse, calls a boundary solved, or authorizes launch/release by itself.  Those
claims require the unified read-only runner, promotion, replay, taint, and
release audit.  Legacy journals without scheduler decisions correctly fail
this audit.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import stat
import sys
import uuid
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import arc_agi3_source_schema as SourceSchema
import arc_agi3_arena_rpc as ArenaRpc


SCHEDULER_SCHEMA = 1
AUDIT_SCHEMA = 1
JOURNAL_SCHEMA = 1
COST_SCALE = 1_000_000_000
SUCCESS_SCALE = 1_000_000
FREE_ENERGY_SCALE = 1_000_000
FREE_ENERGY_COMPLEXITY_WEIGHT = 20_000  # 0.02 at FREE_ENERGY_SCALE.
UNKNOWN_CONDITIONAL_NOVELTY = 128
MAX_LANES = 6
AUXILIARY_ANALYSIS_START_NO_PROGRESS = 5
AUXILIARY_ANALYSIS_EXPAND_NO_PROGRESS = 7
MAX_AUXILIARY_ANALYSES_PER_FRONTIER = 2
SUPERVISORY_PROPOSER_ROLE = "supervisory_proposer"
SUPERVISORY_HANDOFF_KIND = "SUPERVISORY_HANDOFF"
SIDE_EXPERT_ALLOWED_INPUT_CLASSES = (
    "verified_parent_identity_and_budget",
    "admitted_clean_same_frontier_wip",
    "public_observation_transcripts",
    "generic_solver_and_evidence_contract",
    "scheduler_assignment",
)
SUPERVISORY_ALLOWED_INPUT_CLASSES = (
    "verified_parent_identity_and_budget",
    "immutable_authenticated_exact_parent_solver_source_snapshot",
    "admitted_clean_same_frontier_wip_solver_source_snapshot",
    "admitted_clean_same_frontier_native_wip_summaries",
    "public_observation_transcripts",
    "authenticated_same_frontier_side_expert_reports",
    "side_expert_admission_or_rejection_receipts",
    "generic_solver_and_evidence_contract",
    "scheduler_assignment",
)
SUPERVISORY_FORBIDDEN_INPUT_CLASSES = (
    "game_or_environment_implementation",
    "other_game_or_lineage",
    "broader_canonical_solution_archive_or_other_boundary",
    "campaign_plan",
    "raw_supervisor_session",
    "quarantine_archive_except_selected_receipts",
    "manuscript",
    "comparator",
    "benchmark",
    "post_hoc_label",
    "credential",
    "parent_repo_git_metadata",
    "informal_session_note",
    "interactive_operator_or_user_channel",
)
MAX_JOURNAL_PREFIX_BYTES = 24 * 1024 * 1024
MIN_DISPATCH_HEADROOM_BYTES = 1 * 1024 * 1024
MAX_JOURNAL_EVENT_BYTES = 256 * 1024
JOURNAL_SEGMENT_EVENT_LIMIT = 256
MAX_JOURNALED_OBSERVATIONS_PER_ATTEMPT = 72
MIN_JOURNALED_OBSERVATION_INTERVAL_SECONDS = 5 * 60
MAX_RETAINED_TREE_FILES = 8192
MAX_RETAINED_TREE_BYTES = 512 * 1024 * 1024
MAX_RETAINED_FILE_BYTES = 64 * 1024 * 1024
OPERATION_RETRY_BACKOFF_SECONDS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
# Controller-substrate recovery is deliberately slower than an ordinary
# idempotent operation acknowledgement retry.  A failed pre-turn substrate
# latches every not-yet-started lane; only one controller-only health probe is
# admitted at each durable deadline.
SUBSTRATE_HEALTH_REPROBE_BACKOFF_SECONDS = (
    30.0,
    60.0,
    120.0,
    240.0,
    480.0,
    960.0,
)
META_SUBSTRATE_RECOVERY_RECOMMENDATION = (
    "REMATERIALIZE_AND_REPROBE_CONTROLLER_SUBSTRATE"
)
FAILURE_CIRCUIT_THRESHOLD = 6
FAILURE_FAULT_DOMAINS = (
    "operation_error",
    "provider_auth",
    "provider_availability",
    "provider_failure",
    "terminal_infrastructure",
    "containment_infrastructure",
    "controller_substrate",
)
EXPECTED_GAMES = 25
EXPECTED_LEVELS = 183
POLICY_NAME = "arc_agi3_contiguous_scheduler_v2"
SELECTION_METRIC = "positive_unmatched_normalized_ast_zlib_v1"
UNKNOWN_SELECTION_METRIC = "fixed_ignorance_prior_v1"
EVENT_FILE_RE = re.compile(r"\d{20}-[A-Za-z0-9_.:-]+\.json")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_.:-]{1,200}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GAME_RE = re.compile(r"[a-z0-9]{4}")


PROPOSER_POLICY_TEXT = """\
ARC-AGI-3 contiguous acquisition policy v1
Solve only the next level from the verified parent and the observations made
through the public Arena interface. Reuse an existing solver leg when observed
dynamics repeat. Add the smallest general leg needed when observations establish
genuine novelty. Prefer lower conditional description growth among candidates
that replay successfully; replay success always dominates compactness. The
accounting proxy is F=-R+0.02C. Do not inspect game source, environment files,
other campaigns, comparator artifacts, post-hoc labels, host state, or sibling
lineages.
"""

AUXILIARY_ANALYSIS_POLICY_TEXT = """\
ARC-AGI-3 observation-only auxiliary-analysis policy v1
Persistent clean no-progress, rather than game identity, activates this
escalation. Keep the long-coherence proposer lineage running at max effort.
Use an independent agent sidecar to identify the unresolved complexity class,
then assign private-copy specialists drawn only from mechanism induction, state
representation, exact planning, and prefix compression. The sidecar's model and
reasoning effort are pinned separately in the campaign launch manifest; the
role is not an alias for a provider effort setting. Specialists may read the
verified parent, admitted clean same-frontier WIP, and observations made through
the public Arena interface. They may not inspect game source, environment
files, hidden descriptions, other campaigns, comparator artifacts, post-hoc
labels, host state, or sibling lineages. They cannot mutate live WIP or promote.
Each assignment must be deliberately orthogonal to the live proposer and must
end with a Socratic attempt to falsify its own conclusions. Their outputs remain
quarantined until provenance, taint, and fresh replay gates admit them against
the exact parent and frontier. A public frame hash is not a hidden-state
identity: when a fixed public action basis gives different one-step public
response signatures, preserve every split representative and describe the
result only as bounded behavioral equivalence.
"""


Specialization = Literal[
    "complexity_diagnosis",
    "mechanism_induction",
    "state_representation",
    "exact_planning",
    "prefix_compression",
    "supervisory_synthesis",
]
SUPERVISORY_SPECIALIZATION: Specialization = "supervisory_synthesis"
SPECIALIST_PRIORITY_DOMAIN: tuple[Specialization, ...] = (
    "mechanism_induction",
    "state_representation",
    "exact_planning",
    "prefix_compression",
)
ALL_AUXILIARY_SPECIALIZATIONS = frozenset(
    {
        "complexity_diagnosis",
        *SPECIALIST_PRIORITY_DOMAIN,
        SUPERVISORY_SPECIALIZATION,
    }
)
SUPPORTED_AUXILIARY_REASONING_EFFORTS = frozenset(
    {"medium", "high", "xhigh", "max"}
)
AUXILIARY_ACTIVE_PHASES = frozenset(
    {"RESERVED", "INPUT_PREPARED", "RUNNING"}
)
TERMINAL_RESULT_KINDS = (
    "clean_no_progress",
    "tainted",
    "protocol_invalid",
    "infrastructure",
    "candidate",
    "blocker",
)
PROMOTION_FAILURE_CODES = (
    "promotion_gate_rejected",
    "promotion_commit_invalid",
)
# ``BLOCKED`` is deliberately much narrower than an arbitrary proposer claim.
# This finite set contains only host-observable, game-agnostic conditions that
# make the immutable K -> K+1 frontier structurally unavailable.  New codes
# require a schema/conformance change; model prose can never add one.
HOST_BLOCKER_CODES = (
    "arena_parent_terminal_before_target",
)
HOST_BLOCKER_RECEIPT_KIND = "contiguous_host_blocker"
HOST_BLOCKER_RECEIPT_NAME = "host_blocker_receipt.json"
HOST_BLOCKER_REASON_PREFIX = "host_blocker:"
HOST_BLOCKER_AUTHORITY = "host_arena_rpc_parent_snapshot_v1"
HOST_BLOCKER_RECEIPT_FIELDS = frozenset({
    "schema",
    "kind",
    "campaign_id",
    "generation_id",
    "attempt_id",
    "attempt_spec_sha256",
    "authority",
    "code",
    "game",
    "frontier_sha256",
    "parent_checkpoint_sha256",
    "parent_level",
    "target_level",
    "arena_session_binding_receipt_path",
    "arena_session_binding_receipt_sha256",
    "arena_binding_sha256",
    "parent_path_sha256",
    "parent_snapshot_sha256",
    "parent_terminal",
    "arena_host_result",
    "arena_host_result_sha256",
    "host_authentication_sha256",
})
HOST_BLOCKER_ARENA_RESULT_FIELDS = frozenset({
    "binding_sha256",
    "game",
    "exploration_mode",
    "parent_level",
    "levels_completed",
    "parent_path",
    "path",
    "parent_replay_steps",
    "exploration_steps",
    "resets",
    "total_steps",
    "parent_terminal",
    "parent_snapshot_sha256",
})
NONCOUNTING_RUNTIME_OUTCOMES = (
    "capacity",
    "rate_limit",
    "provider_failure",
    "containment_fault",
)
SUPERVISION_CYCLE_STAGES = (
    "consume_durable_decisions",
    "poll_live_attempts",
    "collect_terminal_evidence",
    "prove_teardown",
    "classify_and_settle",
    "commit_exact_promotions",
    "invalidate_or_admit_auxiliary_outputs",
    "recover_reserved_auxiliary_work",
    "recover_or_dispatch_distinct_primary_frontiers",
    "dispatch_eligible_auxiliary_analysis",
)
SUPERVISION_DECISION_INPUTS = (
    "hash_chained_journal",
    "authoritative_inventory",
    "exact_parent_and_frontier_hashes",
    "authenticated_terminal_and_usage_receipts",
    "taint_replay_provenance_and_manifest_receipts",
    "exact_frontier_clean_no_progress_count",
    "durable_last_dispatch_sequence",
    "capacity_and_budget_state",
)
SUPERVISION_FORBIDDEN_INPUTS = (
    "game_semantics",
    "operator_hint",
    "interactive_operator_or_user_channel",
    "model_final_prose",
    "remembered_solution",
    "post_hoc_label",
    "comparator_result",
    "process_list_order",
    "wall_clock_race",
)


class SchedulerError(RuntimeError):
    """A fail-closed policy, decision, budget, or audit violation."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _meta_substrate_recovery_authentication_sha256(
    receipt_body: Mapping[str, object],
    *,
    operator_configuration_sha256: str,
) -> str:
    if (
        "authorization_authentication_sha256" in receipt_body
        or not _is_sha256(operator_configuration_sha256)
        or receipt_body.get("operator_configuration_sha256")
        != operator_configuration_sha256
    ):
        raise SchedulerError(
            "meta substrate authorization body is malformed"
        )
    return hashlib.sha256(
        b"arc-agi3-meta-substrate-recovery-auth-v1\0"
        + operator_configuration_sha256.encode("ascii")
        + b"\0"
        + canonical_json(dict(receipt_body))
    ).hexdigest()


def _substrate_incident_identity_sha256(
    *,
    campaign_id: str,
    attempt_id: str,
    game: str,
    frontier_sha256: str,
    substrate_identity_sha256: str,
    failure_receipt_sha256: str,
    failure_class: str,
    failure_code: str,
) -> str:
    body = {
        "schema": 1,
        "kind": "contiguous_controller_substrate_incident",
        "campaign_id": campaign_id,
        "attempt_id": attempt_id,
        "game": game,
        "frontier_sha256": frontier_sha256,
        "substrate_identity_sha256": substrate_identity_sha256,
        "failure_receipt_sha256": failure_receipt_sha256,
        "failure_class": failure_class,
        "failure_code": failure_code,
    }
    if (
        not _is_identifier(campaign_id)
        or not _is_canonical_uuid(attempt_id)
        or not _is_identifier(game)
        or any(
            not _is_sha256(body[name])
            for name in (
                "frontier_sha256",
                "substrate_identity_sha256",
                "failure_receipt_sha256",
            )
        )
        or failure_class not in {
            "DETERMINISTIC_CONFIGURATION",
            "TRANSIENT_INFRASTRUCTURE",
        }
        or not _is_identifier(failure_code)
    ):
        raise SchedulerError(
            "substrate incident identity input is malformed"
        )
    return hashlib.sha256(
        b"arc-agi3-controller-substrate-incident-v1\0"
        + canonical_json(body)
    ).hexdigest()


def _meta_substrate_resume_authentication_sha256(
    receipt_body: Mapping[str, object],
    *,
    operator_configuration_sha256: str,
) -> str:
    if (
        "resume_authentication_sha256" in receipt_body
        or not _is_sha256(operator_configuration_sha256)
        or receipt_body.get("operator_configuration_sha256")
        != operator_configuration_sha256
    ):
        raise SchedulerError(
            "meta substrate resume body is malformed"
        )
    return hashlib.sha256(
        b"arc-agi3-meta-substrate-resume-auth-v1\0"
        + operator_configuration_sha256.encode("ascii")
        + b"\0"
        + canonical_json(dict(receipt_body))
    ).hexdigest()


def auxiliary_input_allowlist_sha256(
    *,
    role: Literal["side_expert", "supervisory_proposer"],
) -> str:
    allowed = (
        SIDE_EXPERT_ALLOWED_INPUT_CLASSES
        if role == "side_expert"
        else SUPERVISORY_ALLOWED_INPUT_CLASSES
    )
    return sha256_json(
        {
            "schema": 1,
            "role": role,
            "allowed_input_classes": list(allowed),
            "forbidden_input_classes": list(
                SUPERVISORY_FORBIDDEN_INPUT_CLASSES
            ),
            "sealed_input_required": True,
            "symlinks_allowed": False,
            "hardlinks_allowed": False,
            "path_escapes_allowed": False,
        }
    )


def public_observation_ledger_sha256(
    *,
    game: str,
    frontier_sha256: str,
    parent_checkpoint_sha256: str,
    receipt_sha256s: Sequence[str],
) -> str:
    receipts = tuple(sorted(set(receipt_sha256s)))
    if (
        GAME_RE.fullmatch(game) is None
        or not _is_sha256(frontier_sha256)
        or not _is_sha256(parent_checkpoint_sha256)
        or any(not _is_sha256(item) for item in receipts)
    ):
        raise SchedulerError(
            "public observation ledger identity is malformed"
        )
    return sha256_json(
        {
            "schema": 1,
            "kind": "exact_frontier_public_observation_ledger",
            "game": game,
            "frontier_sha256": frontier_sha256,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "public_observation_receipt_sha256s": list(receipts),
        }
    )


PUBLIC_OBSERVATION_AUTHORITY_RESULT_KINDS = frozenset(
    {"clean_no_progress", "candidate"}
)
PUBLIC_OBSERVATION_TRANSITION_KIND = (
    "exact_attempt_public_observation_transition"
)


def public_observation_transition(
    *,
    attempt_id: str,
    generation_id: str,
    game: str,
    frontier_sha256: str,
    parent_checkpoint_sha256: str,
    host_transcript_path: str,
    result_kind: str,
    receipt_sha256s: Sequence[str],
) -> dict[str, object]:
    """Build the canonical write-ahead observation-authority transition.

    Every accepted collection records all native public-observation receipts,
    including forensic-only receipts from non-authoritative outcomes.  Only a
    clean no-progress or candidate result grants those receipts same-frontier
    lineage authority.
    """

    receipts = tuple(receipt_sha256s)
    host_path = Path(host_transcript_path)
    if (
        not _is_identifier(attempt_id)
        or not _is_identifier(generation_id)
        or GAME_RE.fullmatch(game) is None
        or not _is_sha256(frontier_sha256)
        or not _is_sha256(parent_checkpoint_sha256)
        or result_kind not in TERMINAL_RESULT_KINDS
        or not host_path.is_absolute()
        or host_path.name != "backend.jsonl"
        or host_path.parent.name != "host"
        or tuple(sorted(set(receipts))) != receipts
        or any(not _is_sha256(item) for item in receipts)
    ):
        raise SchedulerError(
            "public observation transition identity is malformed"
        )
    authority = (
        "same_frontier_lineage"
        if result_kind in PUBLIC_OBSERVATION_AUTHORITY_RESULT_KINDS
        else "forensic_only_no_lineage_authority"
    )
    if (
        result_kind in PUBLIC_OBSERVATION_AUTHORITY_RESULT_KINDS
        and not receipts
    ):
        raise SchedulerError(
            "authoritative public observation transition is empty"
        )
    return {
        "schema": 1,
        "kind": PUBLIC_OBSERVATION_TRANSITION_KIND,
        "attempt_id": attempt_id,
        "generation_id": generation_id,
        "game": game,
        "frontier_sha256": frontier_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "result_kind": result_kind,
        "authority": authority,
        "receipt_root_path": str(
            host_path.parent / "public_observations"
        ),
        "receipt_sha256s": list(receipts),
    }


def validate_public_observation_transition(
    value: object,
    *,
    attempt_id: str,
    generation_id: str,
    game: str,
    frontier_sha256: str,
    parent_checkpoint_sha256: str,
    host_transcript_path: str,
    result_kind: str | None = None,
    receipt_sha256s: Sequence[str] | None = None,
    reopen_receipts: bool,
) -> tuple[str, ...]:
    """Validate one transition and return only lineage-authoritative hashes."""

    transition = _strict_keys(
        value,
        {
            "schema",
            "kind",
            "attempt_id",
            "generation_id",
            "game",
            "frontier_sha256",
            "parent_checkpoint_sha256",
            "result_kind",
            "authority",
            "receipt_root_path",
            "receipt_sha256s",
        },
        "public observation transition",
    )
    raw_receipts = transition["receipt_sha256s"]
    if not isinstance(raw_receipts, list):
        raise SchedulerError(
            "public observation transition receipts are not a list"
        )
    expected = public_observation_transition(
        attempt_id=attempt_id,
        generation_id=generation_id,
        game=game,
        frontier_sha256=frontier_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        host_transcript_path=host_transcript_path,
        result_kind=(
            str(transition["result_kind"])
            if result_kind is None
            else result_kind
        ),
        receipt_sha256s=tuple(raw_receipts),
    )
    if (
        transition != expected
        or (
            result_kind is not None
            and transition["result_kind"] != result_kind
        )
        or (
            receipt_sha256s is not None
            and tuple(raw_receipts) != tuple(receipt_sha256s)
        )
    ):
        raise SchedulerError(
            "public observation transition crosses collection identity"
        )
    if reopen_receipts:
        root = Path(str(transition["receipt_root_path"]))
        try:
            metadata = root.stat(follow_symlinks=False)
            observed_names = {
                path.name for path in root.iterdir()
            }
        except OSError as exc:
            raise SchedulerError(
                "public observation receipt root cannot be reopened"
            ) from exc
        expected_names = {
            f"{digest}.json" for digest in raw_receipts
        }
        if (
            root.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or observed_names != expected_names
        ):
            raise SchedulerError(
                "public observation receipt inventory is substituted"
            )
        for digest in raw_receipts:
            path = root / f"{digest}.json"
            receipt = _reopen_json_receipt(
                str(path),
                digest,
                label="native public observation receipt",
            )
            try:
                semantic_digest = (
                    ArenaRpc.validate_public_observation_receipt(
                        receipt,
                        game=game,
                        frontier_sha256=frontier_sha256,
                        parent_checkpoint_sha256=(
                            parent_checkpoint_sha256
                        ),
                    )
                )
            except Exception as exc:
                raise SchedulerError(
                    "native public observation receipt is malformed"
                ) from exc
            if semantic_digest != digest:
                raise SchedulerError(
                    "native public observation receipt changed identity"
                )
    receipts = tuple(raw_receipts)
    return (
        receipts
        if transition["authority"] == "same_frontier_lineage"
        else ()
    )


def _is_int(value: object, *, minimum: int = 0) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
    )


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _is_identifier(value: object) -> bool:
    return isinstance(value, str) and IDENTIFIER_RE.fullmatch(value) is not None


def _is_canonical_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return parsed.variant == uuid.RFC_4122 and str(parsed) == value


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise SchedulerError(f"{label} is not a SHA-256 digest")
    return str(value)


def _require_identifier(value: object, label: str) -> str:
    if not _is_identifier(value):
        raise SchedulerError(f"{label} is not a safe identifier")
    return str(value)


def _event_digest(body: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(body)).hexdigest()


def inventory_sha256(inventory: Mapping[str, int]) -> str:
    normalized = validate_inventory(inventory)
    return sha256_json(normalized)


def validate_inventory(inventory: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(inventory, Mapping):
        raise SchedulerError("inventory must be a mapping")
    normalized: dict[str, int] = {}
    for game, target in inventory.items():
        if (
            not isinstance(game, str)
            or GAME_RE.fullmatch(game) is None
            or not _is_int(target, minimum=1)
        ):
            raise SchedulerError("inventory contains an invalid game/target")
        normalized[game] = int(target)
    normalized = dict(sorted(normalized.items()))
    if (
        len(normalized) != EXPECTED_GAMES
        or sum(normalized.values()) != EXPECTED_LEVELS
    ):
        raise SchedulerError(
            "inventory must contain exactly 25 games / 183 levels"
        )
    # Count/total equality is insufficient (for example, inventing re86 L9
    # while removing a real level elsewhere preserves 183).  Reopen the same
    # toolkit-derived per-game authority used by checkpoint admission.
    try:
        import arc_agi3_contiguous_supervisor as Supervisor

        authoritative = dict(
            sorted(Supervisor.authoritative_inventory().items())
        )
    except Exception as exc:
        raise SchedulerError(
            "cannot derive the authoritative per-game inventory"
        ) from exc
    if normalized != authoritative:
        raise SchedulerError(
            "inventory does not exactly match authoritative game targets"
        )
    return normalized


@dataclass(frozen=True)
class RetryPolicy:
    schema: Literal[1]
    no_progress: int
    phase: Literal[
        "ordinary", "hard_frontier", "long_coherence"
    ]
    effort: Literal["medium", "high", "xhigh", "max"]
    soft_allocation_seconds: int
    requested_wip_mode: Literal[
        "exclude", "restore_clean_same_frontier"
    ]
    coherence_reset: bool


@dataclass(frozen=True)
class TerminalPolicyTransition:
    """Canonical lane transition after host-authenticated terminal evidence."""

    schema: Literal[2]
    result_kind: Literal[
        "clean_no_progress",
        "tainted",
        "protocol_invalid",
        "infrastructure",
        "candidate",
        "blocker",
    ]
    next_lane_phase: Literal["READY", "PROMOTING", "BLOCKED"]
    retry_coordinate_delta: Literal[0, 1]
    current_attempt_wip_disposition: Literal[
        "admit_clean_same_frontier_replacement",
        "discard",
    ]
    prior_wip_disposition: Literal[
        "clear_before_optional_replacement",
        "revoke_same_thread_frontier_context",
        "retain_authenticated_if_no_exposure",
        "retain_authenticated_pending_promotion",
        "clear_terminal_frontier",
    ]
    candidate_disposition: Literal[
        "none", "quarantine_until_exact_promotion_gates"
    ]


@dataclass(frozen=True)
class PromotionFailurePolicyTransition:
    """Canonical recovery after an exact candidate fails promotion gates."""

    schema: Literal[1]
    event_kind: Literal["PROMOTION_FAILED"]
    next_lane_phase: Literal["READY"]
    retry_coordinate_delta: Literal[0]
    wip_disposition: Literal[
        "retain_preexisting_clean_same_frontier"
    ]
    candidate_disposition: Literal["discard_rejected_candidate"]
    blocker_authority: Literal[False]


@dataclass(frozen=True)
class ComplexityProfile:
    """Untrusted search-routing advice bound to clean public observations.

    The profile chooses where auxiliary search should spend compute.  It is
    never evidence that a mechanic is true and can never authorize promotion.
    The host admits the referenced observation receipt and taint result before
    the scheduler may use the priority ordering.
    """

    schema: Literal[1]
    profile_id: str
    round_index: int
    frontier_sha256: str
    observation_receipt_sha256: str
    taint_scan_receipt_sha256: str
    priorities: tuple[
        Literal[
            "mechanism_induction",
            "state_representation",
            "exact_planning",
            "prefix_compression",
        ],
        ...,
    ]


@dataclass(frozen=True)
class AuxiliaryAnalysisPolicy:
    """Deterministic max-to-independent-expert escalation."""

    schema: Literal[1]
    no_progress: int
    phase: Literal["disabled", "diagnose", "specialize"]
    role: Literal["independent_side_expert"] | None
    model_effort_source: Literal["campaign_launch_manifest"] | None
    profile_id: str | None
    round_index: int | None
    max_parallel: int
    specializations: tuple[Specialization, ...]
    assignment_mode: Literal["orthogonal_complexity_obligation"]
    must_differ_from_active_lanes: Literal[True]
    minimum_socratic_passes: int
    workspace_mode: Literal["immutable_private_copy"]
    input_mode: Literal[
        "verified_parent_clean_wip_public_observations_only"
    ]
    output_mode: Literal["quarantine_only"]
    admission_mode: Literal[
        "exact_parent_fresh_replay_taint_and_provenance"
    ]
    mutates_live_lineage: Literal[False]


@dataclass(frozen=True)
class FrontierComplexitySchedule:
    """Both escalation axes at one exact-frontier retry coordinate."""

    schema: Literal[1]
    coordinate: Literal["exact_frontier_clean_no_progress_retries"]
    no_progress: int
    primary: RetryPolicy
    auxiliary: AuxiliaryAnalysisPolicy


@dataclass(frozen=True)
class SupervisoryProposerLaunchConfiguration:
    """Role-specific pins for the quarantine-only tactical synthesizer."""

    schema: Literal[1]
    role: Literal["supervisory_proposer"]
    automatic_dispatch_enabled: bool
    model: str
    reasoning_effort: Literal["medium", "high", "xhigh", "max"]
    context_limit_tokens: int
    max_concurrency: Literal[1]


@dataclass(frozen=True)
class AuxiliaryLaunchConfiguration:
    """Durable host contract for a sidecar backend.

    The production manifest deliberately leaves all three launch authorities
    false until a real private-bundle builder, isolated backend, and host-only
    admission gate are attested.  A model effort is a transport choice, never
    the identity of the independent role.
    """

    schema: Literal[1]
    automatic_dispatch_enabled: bool
    backend_attested: bool
    input_bundle_attested: bool
    admission_gate_attested: bool
    model: str
    reasoning_effort: Literal["medium", "high", "xhigh", "max"]
    backend_contract_sha256: str | None
    input_bundle_contract_sha256: str | None
    admission_contract_sha256: str | None
    supervisory_proposer: SupervisoryProposerLaunchConfiguration


@dataclass(frozen=True)
class CleanProposerSettlement:
    """One audited clean no-progress transition on an exact frontier."""

    schema: Literal[1]
    game: str
    frontier_sha256: str
    parent_checkpoint_sha256: str
    attempt_id: str
    scheduler_decision_id: str
    no_progress_before: int
    effort: Literal["medium", "high", "xhigh", "max"]
    soft_allocation_seconds: int
    requested_wip_mode: Literal[
        "exclude", "restore_clean_same_frontier"
    ]
    supervisory_handoff_sha256: str | None
    result_sequence: int
    result_digest: str


@dataclass(frozen=True)
class SocraticChallengeEvidence:
    """Structured minimum evidence for the required self-challenge pass."""

    schema: Literal[1]
    hypothesis: str
    counter_hypothesis: str
    falsification_attempt: str
    observation_receipt_sha256s: tuple[str, ...]
    rejected_conclusions: tuple[str, ...]
    surviving_conclusions: tuple[str, ...]


@dataclass(frozen=True)
class SupervisoryHypothesisEvidence:
    """One cited, falsifiable tactical claim; never a fact authority."""

    schema: Literal[1]
    claim_id: str
    statement: str
    observation_receipt_sha256s: tuple[str, ...]
    falsifiers: tuple[str, ...]
    bounded_next_tests: tuple[str, ...]


@dataclass(frozen=True)
class SupervisoryHandoffEvidence:
    """Quarantine-only LLM synthesis for one already-selected frontier."""

    schema: Literal[1]
    kind: Literal["SUPERVISORY_HANDOFF"]
    role: Literal["supervisory_proposer"]
    handoff_id: str
    assignment_id: str
    frontier_sha256: str
    parent_checkpoint_sha256: str
    input_manifest_sha256: str
    model: str
    reasoning_effort: Literal["medium", "high", "xhigh", "max"]
    unresolved_obligation: str
    relied_on_observation_receipt_sha256s: tuple[str, ...]
    claims: tuple[SupervisoryHypothesisEvidence, ...]
    rejected_alternatives: tuple[str, ...]
    confidence_and_caveats: str
    socratic_challenge_sha256: str
    result_authority: Literal["quarantine_only"]
    native_reproduction_required: Literal[True]
    raw_context_included: Literal[False]
    scheduler_authority: Literal[False]
    mutation_authority: Literal[False]
    promotion_authority: Literal[False]


@dataclass(frozen=True)
class NativeObservationReproduction:
    """One cited sidecar observation freshly reproduced by a native role."""

    schema: Literal[1]
    source_observation_receipt_sha256: str
    native_observation_receipt_sha256: str
    public_action_basis_sha256: str
    public_response_signature_sha256: str
    status: Literal["REPRODUCED"]


@dataclass(frozen=True)
class SupervisoryNativeReproductionReceipt:
    """Gate before handoff-derived evidence can gain lineage authority.

    The handoff itself may enter the native prompt only as an explicitly
    unverified hypothesis.  This receipt is required before any resulting
    WIP, candidate, or promotion can be admitted.
    """

    schema: Literal[1]
    kind: Literal["SUPERVISORY_NATIVE_REPRODUCTION"]
    authority: Literal["host_only"]
    assignment_id: str
    frontier_sha256: str
    parent_checkpoint_sha256: str
    input_manifest_sha256: str
    supervisory_handoff_sha256: str
    native_attempt_id: str
    native_attempt_spec_sha256: str
    native_arena_session_binding_receipt_sha256: str
    native_host_transcript_sha256: str
    reproductions: tuple[NativeObservationReproduction, ...]
    fresh_native_session: Literal[True]
    public_observation_interface_only: Literal[True]
    supervisory_workspace_mounted: Literal[False]
    scheduler_fields_overridden: Literal[False]
    live_lineage_mutated: Literal[False]
    promotion_authorized: Literal[False]
    status: Literal["PASS"]


@dataclass(frozen=True)
class NativeSidecarRequestDraft:
    """A native proposer's non-authoritative request before host admission."""

    schema: Literal[1]
    kind: Literal["NATIVE_SIDECAR_REQUEST_DRAFT"]
    request_id: str
    game: str
    frontier_sha256: str
    parent_checkpoint_sha256: str
    native_attempt_id: str
    semantic_brief: str
    cited_public_observation_receipt_sha256s: tuple[str, ...]
    scheduler_authored: Literal[False]
    live_lineage_mutation_authority: Literal[False]
    promotion_authority: Literal[False]
    draft_sha256: str


@dataclass(frozen=True)
class SidecarRequestEvidence:
    """A game-specific brief authored outside the deterministic scheduler."""

    schema: Literal[1]
    kind: Literal[
        "NATIVE_SIDECAR_REQUEST",
        "SUPERVISORY_SIDECAR_REQUEST",
    ]
    request_id: str
    game: str
    frontier_sha256: str
    parent_checkpoint_sha256: str
    authority: Literal[
        "native_proposer",
        "admitted_supervisory_proposer",
    ]
    semantic_brief: str
    cited_public_observation_receipt_sha256s: tuple[str, ...]
    native_attempt_id: str | None
    supervisory_assignment_id: str | None
    supervisory_handoff_sha256: str | None
    origin_admission_receipt_sha256: str
    scheduler_authored: Literal[False]
    live_lineage_mutation_authority: Literal[False]
    promotion_authority: Literal[False]
    request_sha256: str


@dataclass(frozen=True)
class AuxiliaryOutputEvidence:
    """Quarantined analysis only; this type has no candidate/WIP authority."""

    schema: Literal[1]
    assignment_id: str
    expert_id: str
    thread_id: str
    specialization: Specialization
    frontier_sha256: str
    parent_checkpoint_sha256: str
    input_manifest_sha256: str
    output_manifest_sha256: str
    public_observation_receipt_sha256s: tuple[str, ...]
    challenge: SocraticChallengeEvidence
    quarantined_artifact_sha256s: tuple[str, ...]
    result_authority: Literal["quarantine_only"]
    mutates_live_lineage: Literal[False]
    supervisory_handoff: SupervisoryHandoffEvidence | None = None


@dataclass(frozen=True)
class AuxiliaryInputManifestCommitment:
    """Side-effect-free projection later materialized after reservation."""

    schema: Literal[1]
    kind: Literal["planned_auxiliary_private_input"]
    game: str
    frontier_sha256: str
    parent_checkpoint_sha256: str
    parent_source_tree_sha256: str
    wip_snapshot_id: str | None
    wip_tree_sha256: str | None
    wip_solver_source_tree_sha256: str | None
    observation_ledger_sha256: str
    profile_id: str | None
    round_index: int
    specialization: Specialization
    input_bundle_contract_sha256: str
    immutable_inputs: Literal[True]
    live_lineage_mounted: Literal[False]
    public_observations_only: Literal[True]
    input_role: Literal["side_expert", "supervisory_proposer"]
    allowed_input_classes: tuple[str, ...]
    forbidden_input_classes: tuple[str, ...]
    input_allowlist_sha256: str
    authenticated_public_observation_receipt_sha256s: tuple[str, ...]
    native_solver_source_tree_sha256s: tuple[str, ...]
    authenticated_side_expert_evidence_sha256s: tuple[str, ...]
    authenticated_evidence_set_sha256: str
    sidecar_request: SidecarRequestEvidence
    sidecar_request_sha256: str
    sealed_input_required: Literal[True]
    symlinks_allowed: Literal[False]
    hardlinks_allowed: Literal[False]
    path_escapes_allowed: Literal[False]


@dataclass(frozen=True)
class ComplexityRoundState:
    """One admitted, receipt-bound routing profile for an exact frontier."""

    schema: Literal[1]
    game: str
    frontier_sha256: str
    parent_checkpoint_sha256: str
    parent_source_tree_sha256: str
    round_index: int
    profile: ComplexityProfile
    diagnosis_assignment_id: str
    trigger_no_progress: int
    trigger_history_sha256: str
    input_manifest_sha256: str
    observation_ledger_sha256: str
    admission_receipt_path: str
    admission_receipt_sha256: str
    admitted_sequence: int
    admitted_event_digest: str
    invalidated: bool = False


@dataclass(frozen=True)
class AuxiliaryAssignmentState:
    """Journal-reconstructed non-writer occupancy, separate from a lane."""

    schema: Literal[1]
    assignment_id: str
    decision_id: str
    reservation_id: str
    game: str
    frontier_sha256: str
    parent_checkpoint_sha256: str
    trigger_no_progress: int
    trigger_history_sha256: str
    profile_id: str | None
    round_index: int
    specialization: Specialization
    expert_id: str
    thread_id: str
    active_proposer_attempt_id: str
    input_manifest: AuxiliaryInputManifestCommitment
    input_manifest_sha256: str
    observation_ledger_sha256: str
    model: str
    reasoning_effort: Literal["medium", "high", "xhigh", "max"]
    role: Literal["side_expert", "supervisory_proposer"]
    context_limit_tokens: int | None
    role_max_concurrency: Literal[1] | None
    supervisory_launch_configuration_sha256: str | None
    sidecar_request: SidecarRequestEvidence
    sidecar_request_sha256: str
    phase: Literal[
        "RESERVED",
        "INPUT_PREPARED",
        "RUNNING",
        "QUARANTINED",
        "ADMITTED",
        "REJECTED",
        "ABORTED",
    ]
    output: AuxiliaryOutputEvidence | None = None
    invalidated: bool = False
    admission_receipt_path: str | None = None
    admission_receipt_sha256: str | None = None
    admitted_sequence: int | None = None
    admitted_event_digest: str | None = None


@dataclass(frozen=True)
class AuxiliaryDecision:
    """Hash-bound decision for one otherwise-idle, non-writer sidecar."""

    schema: Literal[1]
    policy_name: str
    policy_sha256: str
    decision_id: str
    campaign_id: str
    assignment_id: str
    reservation_id: str
    journal_head_sequence: int
    journal_head_digest: str
    game: str
    frontier_sha256: str
    parent_checkpoint_sha256: str
    parent_source_tree_sha256: str
    no_progress: int
    trigger_history_sha256: str
    active_proposer_attempt_id: str
    active_attempt_ids: tuple[str, ...]
    active_auxiliary_assignment_ids: tuple[str, ...]
    max_lanes: int
    profile_id: str | None
    round_index: int
    specialization: Specialization
    expert_id: str
    thread_id: str
    model: str
    reasoning_effort: Literal["medium", "high", "xhigh", "max"]
    role: Literal["side_expert", "supervisory_proposer"]
    context_limit_tokens: int | None
    role_max_concurrency: Literal[1] | None
    supervisory_launch_configuration: (
        SupervisoryProposerLaunchConfiguration | None
    )
    supervisory_launch_configuration_sha256: str | None
    input_manifest: AuxiliaryInputManifestCommitment
    input_manifest_sha256: str
    sidecar_request: SidecarRequestEvidence
    sidecar_request_sha256: str
    observation_ledger_sha256: str
    backend_contract_sha256: str
    input_bundle_contract_sha256: str
    admission_contract_sha256: str
    cost_window_id: str
    limit_units: int | None
    settled_units: int
    live_reservation_units: int
    reservation_units: int | None
    decision_sha256: str


_RETRY_PREFIX: tuple[
    tuple[str, int, Literal["exclude", "restore_clean_same_frontier"]],
    ...,
] = (
    ("medium", 15 * 60, "exclude"),
    ("high", 20 * 60, "restore_clean_same_frontier"),
    ("xhigh", 25 * 60, "restore_clean_same_frontier"),
    ("xhigh", 40 * 60, "restore_clean_same_frontier"),
    ("max", 60 * 60, "restore_clean_same_frontier"),
    ("max", 90 * 60, "exclude"),
    ("max", 120 * 60, "restore_clean_same_frontier"),
    ("max", 180 * 60, "exclude"),
    ("max", 180 * 60, "restore_clean_same_frontier"),
)
_EFFORT_RANK = {"medium": 0, "high": 1, "xhigh": 2, "max": 3}


def retry_policy(no_progress: int) -> RetryPolicy:
    """Return the only permitted effort/allocation/WIP row for a frontier."""

    if not _is_int(no_progress):
        raise SchedulerError("no_progress must be a nonnegative integer")
    if no_progress < len(_RETRY_PREFIX):
        effort, seconds, wip_mode = _RETRY_PREFIX[no_progress]
        phase: Literal[
            "ordinary", "hard_frontier", "long_coherence"
        ] = "ordinary" if no_progress <= 2 else "hard_frontier"
    else:
        effort = "max"
        seconds = 300 * 60
        wip_mode = (
            "exclude"
            if no_progress % 2 == 1
            else "restore_clean_same_frontier"
        )
        phase = "long_coherence"
    return RetryPolicy(
        schema=1,
        no_progress=no_progress,
        phase=phase,
        effort=effort,  # type: ignore[arg-type]
        soft_allocation_seconds=seconds,
        requested_wip_mode=wip_mode,
        coherence_reset=wip_mode == "exclude",
    )


def terminal_policy_transition(
    result_kind: object,
) -> TerminalPolicyTransition:
    """Map one authenticated result to the only permitted lane transition.

    This function deliberately has no game, operator, model-prose, or
    comparator input.  Game reasoning ends at candidate/WIP publication; the
    host supervisor reduces admitted receipts through this fixed table.
    """

    if (
        not isinstance(result_kind, str)
        or result_kind not in TERMINAL_RESULT_KINDS
    ):
        raise SchedulerError("terminal result kind is outside policy")
    table: dict[str, TerminalPolicyTransition] = {
        "clean_no_progress": TerminalPolicyTransition(
            schema=2,
            result_kind="clean_no_progress",
            next_lane_phase="READY",
            retry_coordinate_delta=1,
            current_attempt_wip_disposition=(
                "admit_clean_same_frontier_replacement"
            ),
            prior_wip_disposition=(
                "clear_before_optional_replacement"
            ),
            candidate_disposition="none",
        ),
        "tainted": TerminalPolicyTransition(
            schema=2,
            result_kind="tainted",
            next_lane_phase="READY",
            retry_coordinate_delta=0,
            current_attempt_wip_disposition="discard",
            prior_wip_disposition=(
                "revoke_same_thread_frontier_context"
            ),
            candidate_disposition="none",
        ),
        "protocol_invalid": TerminalPolicyTransition(
            schema=2,
            result_kind="protocol_invalid",
            next_lane_phase="READY",
            retry_coordinate_delta=0,
            current_attempt_wip_disposition="discard",
            prior_wip_disposition=(
                "revoke_same_thread_frontier_context"
            ),
            candidate_disposition="none",
        ),
        "infrastructure": TerminalPolicyTransition(
            schema=2,
            result_kind="infrastructure",
            next_lane_phase="READY",
            retry_coordinate_delta=0,
            current_attempt_wip_disposition="discard",
            prior_wip_disposition=(
                "retain_authenticated_if_no_exposure"
            ),
            candidate_disposition="none",
        ),
        "candidate": TerminalPolicyTransition(
            schema=2,
            result_kind="candidate",
            next_lane_phase="PROMOTING",
            retry_coordinate_delta=0,
            current_attempt_wip_disposition="discard",
            prior_wip_disposition=(
                "retain_authenticated_pending_promotion"
            ),
            candidate_disposition=(
                "quarantine_until_exact_promotion_gates"
            ),
        ),
        "blocker": TerminalPolicyTransition(
            schema=2,
            result_kind="blocker",
            next_lane_phase="BLOCKED",
            retry_coordinate_delta=0,
            current_attempt_wip_disposition="discard",
            prior_wip_disposition="clear_terminal_frontier",
            candidate_disposition="none",
        ),
    }
    return table[str(result_kind)]


def promotion_failure_policy_transition(
) -> PromotionFailurePolicyTransition:
    """Return the only transition for a rejected promotion candidate.

    Promotion-integrity failure cannot manufacture ``BLOCKED`` authority.
    Only the finite, host-authenticated blocker receipt can stop a frontier.
    """

    return PromotionFailurePolicyTransition(
        schema=1,
        event_kind="PROMOTION_FAILED",
        next_lane_phase="READY",
        retry_coordinate_delta=0,
        wip_disposition="retain_preexisting_clean_same_frontier",
        candidate_disposition="discard_rejected_candidate",
        blocker_authority=False,
    )


def reduce_terminal_wip(
    *,
    transition: TerminalPolicyTransition,
    prior_wip: object,
    current_attempt_wip: object,
    exposure_detected: bool,
) -> object:
    """Apply the versioned two-dimensional WIP policy exactly once."""

    if (
        not isinstance(transition, TerminalPolicyTransition)
        or transition.schema != 2
        or not isinstance(exposure_detected, bool)
    ):
        raise SchedulerError("terminal WIP transition is malformed")
    if (
        transition.current_attempt_wip_disposition == "discard"
        and current_attempt_wip is not None
    ):
        raise SchedulerError(
            "terminal result carries WIP that policy must discard"
        )
    if (
        transition.current_attempt_wip_disposition
        == "admit_clean_same_frontier_replacement"
    ):
        return current_attempt_wip
    if transition.prior_wip_disposition in {
        "revoke_same_thread_frontier_context",
        "clear_terminal_frontier",
        "clear_before_optional_replacement",
    }:
        return None
    if (
        transition.prior_wip_disposition
        == "retain_authenticated_if_no_exposure"
    ):
        return None if exposure_detected else prior_wip
    if (
        transition.prior_wip_disposition
        == "retain_authenticated_pending_promotion"
    ):
        return prior_wip
    raise SchedulerError("terminal WIP disposition is outside policy")


def advance_retry_coordinate(no_progress: object, outcome: object) -> int:
    """Advance the complexity coordinate only for clean settled failures."""

    if not _is_int(no_progress):
        raise SchedulerError(
            "exact-frontier retry coordinate must be nonnegative"
        )
    if outcome in TERMINAL_RESULT_KINDS:
        delta = terminal_policy_transition(outcome).retry_coordinate_delta
    elif outcome in NONCOUNTING_RUNTIME_OUTCOMES:
        delta = 0
    else:
        raise SchedulerError(
            "exact-frontier retry outcome is outside policy"
        )
    return int(no_progress) + delta


def validate_complexity_profile(
    profile: ComplexityProfile,
    *,
    frontier_sha256: str,
) -> ComplexityProfile:
    """Reject stale, unscanned, ambiguous specialist-routing advice."""

    _require_sha256(frontier_sha256, "frontier")
    if (
        profile.schema != 1
        or not _is_identifier(profile.profile_id)
        or not _is_int(profile.round_index)
        or profile.frontier_sha256 != frontier_sha256
        or not _is_sha256(profile.observation_receipt_sha256)
        or not _is_sha256(profile.taint_scan_receipt_sha256)
        or not profile.priorities
        or len(profile.priorities) > len(SPECIALIST_PRIORITY_DOMAIN)
        or len(set(profile.priorities)) != len(profile.priorities)
        or any(
            priority not in SPECIALIST_PRIORITY_DOMAIN
            for priority in profile.priorities
        )
    ):
        raise SchedulerError(
            "complexity profile is stale, ambiguous, or outside the "
            "game-independent specialist domain"
        )
    return profile


def disabled_auxiliary_launch_configuration(
    *,
    model: str = "gpt-5.6-sol",
    reasoning_effort: Literal[
        "medium", "high", "xhigh", "max"
    ] = "max",
) -> AuxiliaryLaunchConfiguration:
    return AuxiliaryLaunchConfiguration(
        schema=1,
        automatic_dispatch_enabled=False,
        backend_attested=False,
        input_bundle_attested=False,
        admission_gate_attested=False,
        model=model,
        reasoning_effort=reasoning_effort,
        backend_contract_sha256=None,
        input_bundle_contract_sha256=None,
        admission_contract_sha256=None,
        supervisory_proposer=SupervisoryProposerLaunchConfiguration(
            schema=1,
            role=SUPERVISORY_PROPOSER_ROLE,
            automatic_dispatch_enabled=False,
            model=model,
            reasoning_effort=reasoning_effort,
            context_limit_tokens=200_000,
            max_concurrency=1,
        ),
    )


def validate_supervisory_proposer_launch_configuration(
    value: SupervisoryProposerLaunchConfiguration,
) -> SupervisoryProposerLaunchConfiguration:
    if (
        not isinstance(value, SupervisoryProposerLaunchConfiguration)
        or value.schema != 1
        or value.role != SUPERVISORY_PROPOSER_ROLE
        or not isinstance(value.automatic_dispatch_enabled, bool)
        or not _is_identifier(value.model)
        or value.reasoning_effort
        not in SUPPORTED_AUXILIARY_REASONING_EFFORTS
        or not _is_int(value.context_limit_tokens, minimum=1)
        or value.context_limit_tokens > 2_000_000
        or value.max_concurrency != 1
        or isinstance(value.max_concurrency, bool)
    ):
        raise SchedulerError(
            "supervisory proposer launch configuration is malformed"
        )
    return value


def validate_auxiliary_launch_configuration(
    value: AuxiliaryLaunchConfiguration,
) -> AuxiliaryLaunchConfiguration:
    if (
        not isinstance(value, AuxiliaryLaunchConfiguration)
        or value.schema != 1
        or not _is_identifier(value.model)
        or value.reasoning_effort
        not in SUPPORTED_AUXILIARY_REASONING_EFFORTS
        or any(
            not isinstance(flag, bool)
            for flag in (
                value.automatic_dispatch_enabled,
                value.backend_attested,
                value.input_bundle_attested,
                value.admission_gate_attested,
            )
        )
    ):
        raise SchedulerError("auxiliary launch configuration is malformed")
    validate_supervisory_proposer_launch_configuration(
        value.supervisory_proposer
    )
    flags = (
        value.backend_attested,
        value.input_bundle_attested,
        value.admission_gate_attested,
    )
    digests = (
        value.backend_contract_sha256,
        value.input_bundle_contract_sha256,
        value.admission_contract_sha256,
    )
    if value.automatic_dispatch_enabled:
        if not all(flags) or not all(_is_sha256(item) for item in digests):
            raise SchedulerError(
                "automatic auxiliary dispatch lacks an attested backend, "
                "private bundle, or admission gate"
            )
    elif any(flags) or any(item is not None for item in digests):
        raise SchedulerError(
            "disabled auxiliary dispatch must not claim launch attestation"
        )
    if (
        value.supervisory_proposer.automatic_dispatch_enabled
        and not value.automatic_dispatch_enabled
    ):
        raise SchedulerError(
            "supervisory proposer cannot outlive disabled auxiliary dispatch"
        )
    return value


def auxiliary_launch_configuration_to_dict(
    value: AuxiliaryLaunchConfiguration,
) -> dict[str, object]:
    return asdict(validate_auxiliary_launch_configuration(value))


def auxiliary_launch_configuration_from_dict(
    value: object,
) -> AuxiliaryLaunchConfiguration:
    raw = _strict_keys(
        value,
        set(AuxiliaryLaunchConfiguration.__dataclass_fields__),
        "auxiliary launch configuration",
    )
    supervisory_raw = _strict_keys(
        raw["supervisory_proposer"],
        set(SupervisoryProposerLaunchConfiguration.__dataclass_fields__),
        "supervisory proposer launch configuration",
    )
    try:
        supervisory = SupervisoryProposerLaunchConfiguration(
            **supervisory_raw
        )
        parsed = AuxiliaryLaunchConfiguration(
            **{**raw, "supervisory_proposer": supervisory}
        )
    except TypeError as exc:
        raise SchedulerError(
            "auxiliary launch configuration schema mismatch"
        ) from exc
    return validate_auxiliary_launch_configuration(parsed)


def auxiliary_analysis_policy(
    no_progress: int,
    *,
    frontier_sha256: str,
    profile: ComplexityProfile | None = None,
    active_specializations: Sequence[Specialization] = (),
    completed_specializations: Sequence[Specialization] = (),
) -> AuxiliaryAnalysisPolicy:
    """Choose independent side experts from complexity evidence, not game ID.

    The ordinary ladder and first 60-minute max attempt at ``n=4`` must all end
    as clean no-progress before this path opens at ``n=5``.  A diagnosis
    produces a receipt-bound priority ordering while the primary lane runs its
    90-minute max reset.  If the 90-minute reset and 120-minute cumulative max
    attempt also fail, ``n=7`` may use two otherwise-idle worker slots, but
    never two copies of the same specialization and never a second writable
    lineage.

    ``active_specializations`` and ``completed_specializations`` are the exact
    journal-reconstructed state for the current profile.  A later diagnostic
    profile may deliberately schedule a new round; silently repeating a
    completed specialization under one profile is forbidden.
    """

    if not _is_int(no_progress):
        raise SchedulerError("no_progress must be a nonnegative integer")
    _require_sha256(frontier_sha256, "frontier")
    active = tuple(active_specializations)
    completed = tuple(completed_specializations)
    for label, values in (("active", active), ("completed", completed)):
        if len(set(values)) != len(values):
            raise SchedulerError(
                f"{label} auxiliary specializations are invalid"
            )
    if set(active) & set(completed):
        raise SchedulerError(
            "an auxiliary specialization cannot be active and completed"
        )
    common = {
        "schema": 1,
        "no_progress": no_progress,
        "assignment_mode": "orthogonal_complexity_obligation",
        "must_differ_from_active_lanes": True,
        "minimum_socratic_passes": 1,
        "workspace_mode": "immutable_private_copy",
        "input_mode":
            "verified_parent_clean_wip_public_observations_only",
        "output_mode": "quarantine_only",
        "admission_mode":
            "exact_parent_fresh_replay_taint_and_provenance",
        "mutates_live_lineage": False,
    }
    if no_progress < AUXILIARY_ANALYSIS_START_NO_PROGRESS:
        if profile is not None or active or completed:
            raise SchedulerError(
                "auxiliary analysis state exists before its complexity "
                "threshold"
            )
        return AuxiliaryAnalysisPolicy(
            phase="disabled",
            role=None,
            model_effort_source=None,
            profile_id=None,
            round_index=None,
            max_parallel=0,
            specializations=(),
            **common,  # type: ignore[arg-type]
        )
    if profile is None:
        if (
            any(value != "complexity_diagnosis" for value in active)
            or completed
        ):
            raise SchedulerError(
                "specialist state exists without an admitted complexity "
                "profile"
            )
        return AuxiliaryAnalysisPolicy(
            phase="diagnose",
            role="independent_side_expert",
            model_effort_source="campaign_launch_manifest",
            profile_id=None,
            round_index=None,
            max_parallel=1,
            specializations=(
                ()
                if "complexity_diagnosis" in active
                else ("complexity_diagnosis",)
            ),
            **common,  # type: ignore[arg-type]
        )
    validate_complexity_profile(
        profile, frontier_sha256=frontier_sha256
    )
    for label, values in (("active", active), ("completed", completed)):
        if (
            any(value not in SPECIALIST_PRIORITY_DOMAIN for value in values)
            or any(value not in profile.priorities for value in values)
        ):
            raise SchedulerError(
                f"{label} auxiliary specializations are not members of "
                "the exact complexity profile"
            )
    selected = tuple(
        priority
        for priority in profile.priorities
        if priority not in active and priority not in completed
    )
    cap = (
        MAX_AUXILIARY_ANALYSES_PER_FRONTIER
        if no_progress >= AUXILIARY_ANALYSIS_EXPAND_NO_PROGRESS
        else 1
    )
    if len(active) > cap:
        raise SchedulerError(
            "active auxiliary specialists exceed the complexity-phase cap"
        )
    selected = selected[: max(0, cap - len(active))]
    return AuxiliaryAnalysisPolicy(
        phase="specialize",
        role="independent_side_expert",
        model_effort_source="campaign_launch_manifest",
        profile_id=profile.profile_id,
        round_index=profile.round_index,
        max_parallel=cap,
        specializations=selected,
        **common,  # type: ignore[arg-type]
    )


def frontier_complexity_schedule(
    no_progress: int,
    *,
    frontier_sha256: str,
    profile: ComplexityProfile | None = None,
    active_specializations: Sequence[Specialization] = (),
    completed_specializations: Sequence[Specialization] = (),
) -> FrontierComplexitySchedule:
    """Project primary effort and sidecar roles from the same retry count."""

    return FrontierComplexitySchedule(
        schema=1,
        coordinate="exact_frontier_clean_no_progress_retries",
        no_progress=no_progress,
        primary=retry_policy(no_progress),
        auxiliary=auxiliary_analysis_policy(
            no_progress,
            frontier_sha256=frontier_sha256,
            profile=profile,
            active_specializations=active_specializations,
            completed_specializations=completed_specializations,
        ),
    )


def policy_projection(max_no_progress: int = 32) -> dict[str, object]:
    if not _is_int(max_no_progress):
        raise SchedulerError("max_no_progress must be nonnegative")
    return {
        "schema": SCHEDULER_SCHEMA,
        "name": POLICY_NAME,
        "cost_scale": COST_SCALE,
        "success_scale": SUCCESS_SCALE,
        "free_energy_scale": FREE_ENERGY_SCALE,
        "free_energy_complexity_weight":
            FREE_ENERGY_COMPLEXITY_WEIGHT,
        "unknown_conditional_novelty": UNKNOWN_CONDITIONAL_NOVELTY,
        "selection_metric": SELECTION_METRIC,
        "operational_complexity_coordinate":
            "exact_frontier_clean_no_progress_retries",
        "complexity_coordinate_reset": "exact_promotion",
        "complexity_coordinate_nonincrements": [
            "infrastructure",
            "capacity_or_rate_limit",
            "taint",
            "blocker",
            "containment",
        ],
        "supervision_loop": {
            "authority": "receipt_reducer_only",
            "cycle_stages": list(SUPERVISION_CYCLE_STAGES),
            "decision_inputs": list(SUPERVISION_DECISION_INPUTS),
            "forbidden_inputs": list(SUPERVISION_FORBIDDEN_INPUTS),
            "terminal_transitions": {
                kind: asdict(terminal_policy_transition(kind))
                for kind in TERMINAL_RESULT_KINDS
            },
            "promotion_failure": {
                "codes": list(PROMOTION_FAILURE_CODES),
                "transition": asdict(
                    promotion_failure_policy_transition()
                ),
            },
            "blocker_authority": {
                "codes": list(HOST_BLOCKER_CODES),
                "receipt_kind": HOST_BLOCKER_RECEIPT_KIND,
                "authority": HOST_BLOCKER_AUTHORITY,
                "canonical_reason_prefix":
                    HOST_BLOCKER_REASON_PREFIX,
                "invalid_claim_transition": "infrastructure",
                "closed_receipt_revalidation": True,
            },
            "policy_change_protocol": (
                "version_hash_tests_then_prospective_dispatch"
            ),
            "silent_live_operator_steering": False,
            "failure_circuits": {
                "threshold": FAILURE_CIRCUIT_THRESHOLD,
                "fault_domains": list(FAILURE_FAULT_DOMAINS),
                "backoff_seconds": list(
                    OPERATION_RETRY_BACKOFF_SECONDS
                ),
                "scopes": [
                    "exact_operation_and_fault_domain",
                    "campaign_wide_fault_domain",
                ],
                "success_reset":
                    "authenticated_matching_external_transition_only",
                "clean_no_progress_counts_as_failure": False,
                "unlimited_budget_caps_search": False,
                "teardown_and_abort_independent": True,
                "exhaustion_transition": "OPERATOR_INCIDENT",
            },
        },
        "ordering": [
            "last_dispatch_sequence",
            "estimated_free_energy_micro",
            "negative_reused_definition_calls",
            "game",
        ],
        "proposer_policy_sha256":
            hashlib.sha256(PROPOSER_POLICY_TEXT.encode("utf-8")).hexdigest(),
        "rows": [
            asdict(retry_policy(index))
            for index in range(max_no_progress + 1)
        ],
        "tail": {
            "starts_at": 9,
            "effort": "max",
            "soft_allocation_seconds": 300 * 60,
            "odd_wip_mode": "exclude",
            "even_wip_mode": "restore_clean_same_frontier",
        },
        "auxiliary_analysis": {
            "policy_sha256": hashlib.sha256(
                AUXILIARY_ANALYSIS_POLICY_TEXT.encode("utf-8")
            ).hexdigest(),
            "trigger": (
                "ordinary_ladder_plus_first_clean_max_no_progress"
            ),
            "starts_at_no_progress":
                AUXILIARY_ANALYSIS_START_NO_PROGRESS,
            "expands_at_no_progress":
                AUXILIARY_ANALYSIS_EXPAND_NO_PROGRESS,
            "role": "independent_side_expert",
            "model_effort_source": "campaign_launch_manifest",
            "provider_effort_is_not_the_role": True,
            "maximum_parallel_per_frontier":
                MAX_AUXILIARY_ANALYSES_PER_FRONTIER,
            "specialist_priority_domain":
                list(SPECIALIST_PRIORITY_DOMAIN),
            "assignment_mode": "orthogonal_complexity_obligation",
            "must_differ_from_active_lanes": True,
            "minimum_socratic_passes": 1,
            "workspace_mode": "immutable_private_copy",
            "input_mode":
                "verified_parent_clean_wip_public_observations_only",
            "output_mode": "quarantine_only",
            "admission_mode":
                "exact_parent_fresh_replay_taint_and_provenance",
            "mutates_live_lineage": False,
            "game_specific_rules": False,
            "state_equivalence_policy": (
                "public_frame_plus_bounded_public_response_signatures"
            ),
            "preserve_public_response_splits": True,
            "hidden_state_completeness_claim": False,
        },
    }


SCHEDULER_POLICY_SHA256 = sha256_json(policy_projection())
PROPOSER_POLICY_SHA256 = hashlib.sha256(
    PROPOSER_POLICY_TEXT.encode("utf-8")
).hexdigest()


def _normalize_policy_row(value: object) -> dict[str, object]:
    fields = (
        "schema",
        "no_progress",
        "phase",
        "effort",
        "soft_allocation_seconds",
        "requested_wip_mode",
        "coherence_reset",
    )
    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw = {
            field: getattr(value, field, None)
            for field in fields
        }
        # Compatibility with the pre-scheduler runner dataclass.
        if raw["requested_wip_mode"] is None:
            raw["requested_wip_mode"] = getattr(value, "wip_mode", None)
        if raw["phase"] is None and _is_int(raw["no_progress"]):
            raw["phase"] = retry_policy(int(raw["no_progress"])).phase
    return {field: raw.get(field) for field in fields}


def verify_runner_policy(
    runner_policy: Callable[[int], object],
    *,
    declared_policy_sha256: str,
) -> str:
    """Exercise a runner projection and reject any policy drift.

    Rows 0..32 cover the complete prefix and both parity arms of the periodic
    long-coherence tail.  The runner should normally delegate directly to
    :func:`retry_policy`; this check exists to make accidental reimplementation
    or stale compatibility wrappers fail closed.
    """

    if declared_policy_sha256 != SCHEDULER_POLICY_SHA256:
        raise SchedulerError("runner declared the wrong scheduler policy hash")
    previous_rank = -1
    for index in range(33):
        expected = asdict(retry_policy(index))
        actual = _normalize_policy_row(runner_policy(index))
        if actual != expected:
            raise SchedulerError(
                f"runner retry policy differs at no_progress={index}"
            )
        rank = _EFFORT_RANK[str(actual["effort"])]
        if rank < previous_rank:
            raise SchedulerError(
                f"runner effort de-escalates at no_progress={index}"
            )
        previous_rank = rank
    return SCHEDULER_POLICY_SHA256


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise SchedulerError(f"{label} must be a finite nonnegative number")
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise SchedulerError(f"{label} is invalid") from exc
    if not selected.is_finite() or selected < 0:
        raise SchedulerError(f"{label} must be a finite nonnegative number")
    return selected


def limit_to_units(value: object | None) -> int | None:
    """Normalize a finite campaign limit downward; ``None`` stays unlimited."""

    if value is None:
        return None
    return int(
        (_decimal(value, "limit") * COST_SCALE).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )


def charge_to_units(value: object) -> int:
    """Normalize authenticated usage upward so accounting never understates."""

    return int(
        (_decimal(value, "charge") * COST_SCALE).to_integral_value(
            rounding=ROUND_CEILING
        )
    )


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    attempt_id: str
    units: int


@dataclass(frozen=True)
class BudgetState:
    cost_window_id: str
    limit_units: int | None
    settled_units: int
    live_reservations: tuple[BudgetReservation, ...] = ()


def validate_budget_state(state: BudgetState) -> BudgetState:
    _require_identifier(state.cost_window_id, "cost_window_id")
    if (
        (state.limit_units is not None and not _is_int(state.limit_units))
        or not _is_int(state.settled_units)
    ):
        raise SchedulerError("budget state contains invalid units")
    seen_reservations: set[str] = set()
    seen_attempts: set[str] = set()
    live = 0
    for reservation in state.live_reservations:
        _require_identifier(reservation.reservation_id, "reservation_id")
        _require_identifier(reservation.attempt_id, "attempt_id")
        if (
            not _is_int(reservation.units, minimum=1)
            or reservation.reservation_id in seen_reservations
            or reservation.attempt_id in seen_attempts
        ):
            raise SchedulerError("budget reservations are invalid or duplicated")
        seen_reservations.add(reservation.reservation_id)
        seen_attempts.add(reservation.attempt_id)
        live += reservation.units
    if (
        state.limit_units is not None
        and state.settled_units + live > state.limit_units
    ):
        raise SchedulerError("finite budget is overbooked")
    return state


def reservation_allowance(
    state: BudgetState, *, slots_to_fill: int
) -> int | None:
    """Return one atomic reservation share without global overbooking."""

    validate_budget_state(state)
    if not _is_int(slots_to_fill, minimum=1):
        raise SchedulerError("slots_to_fill must be positive")
    if state.limit_units is None:
        return None
    reserved = sum(item.units for item in state.live_reservations)
    available = state.limit_units - state.settled_units - reserved
    if available <= 0:
        return 0
    # Sequential ceiling shares are safe: after this reservation is installed,
    # the next call sees the reduced available balance and one fewer slot.
    return (available + slots_to_fill - 1) // slots_to_fill


def reserve_budget(
    state: BudgetState,
    *,
    reservation_id: str,
    attempt_id: str,
    units: int | None,
) -> BudgetState:
    """Install one reservation.  Unlimited mode records no artificial cap."""

    validate_budget_state(state)
    _require_identifier(reservation_id, "reservation_id")
    _require_identifier(attempt_id, "attempt_id")
    if any(
        item.reservation_id == reservation_id
        or item.attempt_id == attempt_id
        for item in state.live_reservations
    ):
        raise SchedulerError("budget reservation identity was already consumed")
    if state.limit_units is None:
        if units is not None:
            raise SchedulerError("unlimited mode must not install a cost cutoff")
        return state
    if not _is_int(units, minimum=1):
        raise SchedulerError("finite mode requires a positive reservation")
    selected = BudgetReservation(reservation_id, attempt_id, int(units))
    updated = BudgetState(
        cost_window_id=state.cost_window_id,
        limit_units=state.limit_units,
        settled_units=state.settled_units,
        live_reservations=tuple(
            sorted(
                (*state.live_reservations, selected),
                key=lambda item: item.reservation_id,
            )
        ),
    )
    return validate_budget_state(updated)


def settle_budget(
    state: BudgetState,
    *,
    reservation_id: str,
    attempt_id: str,
    charged_units: int,
) -> BudgetState:
    """Settle/release exactly once; authenticated charge cannot exceed reserve."""

    validate_budget_state(state)
    if not _is_int(charged_units):
        raise SchedulerError("charged_units must be nonnegative")
    matches = [
        item for item in state.live_reservations
        if item.reservation_id == reservation_id
        and item.attempt_id == attempt_id
    ]
    if state.limit_units is None:
        if matches:
            raise SchedulerError("unlimited budget unexpectedly has reservations")
        return BudgetState(
            cost_window_id=state.cost_window_id,
            limit_units=None,
            settled_units=state.settled_units + charged_units,
            live_reservations=(),
        )
    if len(matches) != 1:
        raise SchedulerError("budget settlement is missing or duplicated")
    selected = matches[0]
    if charged_units > selected.units:
        raise SchedulerError("authenticated charge exceeds reserved allowance")
    updated = BudgetState(
        cost_window_id=state.cost_window_id,
        limit_units=state.limit_units,
        settled_units=state.settled_units + charged_units,
        live_reservations=tuple(
            item for item in state.live_reservations
            if item.reservation_id != reservation_id
        ),
    )
    return validate_budget_state(updated)


def release_budget(
    state: BudgetState, *, reservation_id: str, attempt_id: str
) -> BudgetState:
    return settle_budget(
        state,
        reservation_id=reservation_id,
        attempt_id=attempt_id,
        charged_units=0,
    )


def _read_regular(path: Path, *, maximum: int) -> bytes:
    if path.is_symlink():
        raise SchedulerError(f"source file is a symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SchedulerError(f"cannot open source file: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > maximum
        ):
            raise SchedulerError(f"source file is not an admissible regular file: {path}")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise SchedulerError(f"source file changed while reading: {path}")
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        if (
            after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
            or after.st_size != metadata.st_size
            or after.st_mtime_ns != metadata.st_mtime_ns
            or after.st_ctime_ns != metadata.st_ctime_ns
        ):
            raise SchedulerError(f"source file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _regular_tree_hash(
    root: Path, expected_sha256: str, *, label: str
) -> str:
    _require_sha256(expected_sha256, label)
    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise SchedulerError(f"{label} root is not an absolute regular directory")
    entries = sorted(root.rglob("*"))
    if len(entries) > MAX_RETAINED_TREE_FILES:
        raise SchedulerError(f"{label} exceeds the retained file-count bound")
    directories = [root]
    files: list[Path] = []
    for entry in entries:
        if entry.is_symlink():
            raise SchedulerError(f"{label} contains a symbolic link")
        if entry.is_dir():
            directories.append(entry)
        elif entry.is_file():
            files.append(entry)
        else:
            raise SchedulerError(f"{label} contains a non-regular entry")
    before: dict[Path, tuple[int, int, int, int]] = {}
    for directory in directories:
        metadata = directory.stat(follow_symlinks=False)
        before[directory] = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
    digest = hashlib.sha256()
    total = 0
    for path in files:
        raw = _read_regular(path, maximum=MAX_RETAINED_FILE_BYTES)
        total += len(raw)
        if total > MAX_RETAINED_TREE_BYTES:
            raise SchedulerError(f"{label} exceeds the retained byte bound")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(raw).hexdigest().encode("ascii"))
        digest.update(b"\n")
    after_entries = sorted(root.rglob("*"))
    if after_entries != entries:
        raise SchedulerError(f"{label} changed while it was inspected")
    for directory, expected in before.items():
        metadata = directory.stat(follow_symlinks=False)
        observed = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        if observed != expected:
            raise SchedulerError(f"{label} changed while it was inspected")
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != expected_sha256:
        raise SchedulerError(f"{label} tree bytes differ from their hash")
    return observed_sha256


def _source_tree(root: Path, expected_sha256: str) -> dict[str, bytes]:
    _require_sha256(expected_sha256, "source tree")
    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise SchedulerError("source root must be an absolute regular directory")
    payloads: dict[str, bytes] = {}
    entries = list(root.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise SchedulerError("solver source must be a flat regular-file tree")
    for entry in entries:
        payloads[entry.name] = _read_regular(
            entry, maximum=SourceSchema.MAX_FILE_BYTES
        )
    try:
        SourceSchema.validate_source_payloads(payloads)
    except SourceSchema.SourceSchemaError as exc:
        raise SchedulerError("solver source violates the shared schema") from exc
    digest = hashlib.sha256()
    for name, raw in sorted(payloads.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(raw).hexdigest().encode("ascii"))
        digest.update(b"\n")
    if digest.hexdigest() != expected_sha256:
        raise SchedulerError("solver source tree hash changed")
    return payloads


def _source_tree_pointer(
    root: Path,
) -> tuple[tuple[object, ...], ...]:
    """Return a cheap, exact OS pointer signature for one flat source tree.

    A cached hash/evidence value is reusable only while every inode, mode,
    link-count, size, mtime, and ctime remains identical.  In particular,
    restoring an old mtime after an in-place edit cannot preserve the ctime,
    and atomic replacement cannot preserve the inode.
    """

    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise SchedulerError("source root must be an absolute regular directory")
    root_metadata = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise SchedulerError("source root must be an absolute regular directory")
    entries = sorted(root.iterdir(), key=lambda item: item.name)
    if len(entries) > SourceSchema.MAX_FILES:
        raise SchedulerError("solver source exceeds its file-count bound")
    signature: list[tuple[object, ...]] = [(
        ".",
        root_metadata.st_dev,
        root_metadata.st_ino,
        root_metadata.st_mode,
        root_metadata.st_nlink,
        root_metadata.st_size,
        root_metadata.st_mtime_ns,
        root_metadata.st_ctime_ns,
    )]
    for entry in entries:
        metadata = entry.stat(follow_symlinks=False)
        if (
            entry.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > SourceSchema.MAX_FILE_BYTES
        ):
            raise SchedulerError(
                "solver source must be a flat regular-file tree"
            )
        signature.append((
            entry.name,
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ))
    return tuple(signature)


def _normalized_units(
    payloads: Mapping[str, bytes],
) -> tuple[list[bytes], dict[str, str], set[str]]:
    units: list[bytes] = []
    definition_versions: dict[str, list[str]] = {}
    call_versions: dict[str, list[set[str]]] = {}

    class _ExecutableNamedCalls(ast.NodeVisitor):
        """Conservative static calls in one function's executable body.

        A nested definition is not executed when its enclosing function is
        called, and a branch whose condition is a literal constant can be
        pruned without guessing about runtime state.  This remains a static
        reachability proxy; it is not a dynamic winning-trace witness.
        """

        def __init__(self) -> None:
            self.calls: set[str] = set()

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if isinstance(node.func, ast.Name):
                self.calls.add(node.func.id)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            return

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            return

        def visit_If(self, node: ast.If) -> None:  # noqa: N802
            try:
                condition = ast.literal_eval(node.test)
            except (ValueError, TypeError):
                self.visit(node.test)
                for child in (*node.body, *node.orelse):
                    self.visit(child)
                return
            selected = node.body if bool(condition) else node.orelse
            for child in selected:
                self.visit(child)

        def visit_While(self, node: ast.While) -> None:  # noqa: N802
            try:
                condition = ast.literal_eval(node.test)
            except (ValueError, TypeError):
                self.visit(node.test)
                for child in (*node.body, *node.orelse):
                    self.visit(child)
                return
            if bool(condition):
                for child in node.body:
                    self.visit(child)
            else:
                for child in node.orelse:
                    self.visit(child)

    for name, raw in sorted(payloads.items()):
        text = raw.decode("utf-8")
        if name.endswith(".py"):
            try:
                tree = ast.parse(text, filename=name)
            except SyntaxError as exc:
                raise SchedulerError(f"source has invalid Python AST: {name}") from exc
            body = tree.body or [ast.Pass()]
            for node in body:
                dumped = ast.dump(
                    node, annotate_fields=True, include_attributes=False
                )
                units.append(name.encode() + b"\0" + dumped.encode())
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    definition_versions.setdefault(node.name, []).append(
                        dumped
                    )
                    visitor = _ExecutableNamedCalls()
                    for child in node.body:
                        visitor.visit(child)
                    call_versions.setdefault(node.name, []).append(
                        visitor.calls
                    )
        elif name.endswith(".json"):
            try:
                normalized = canonical_json(json.loads(text))
            except json.JSONDecodeError as exc:
                raise SchedulerError(f"source has invalid JSON: {name}") from exc
            units.append(name.encode() + b"\0" + normalized)
        else:
            normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
            units.append(name.encode() + b"\0" + normalized_text.encode("utf-8"))
    # This is deliberately a conservative *static* reachability proxy rooted
    # only at the unique solve() entry.  Merely defining play_level_* does not
    # make it executable, and ambiguous duplicate definitions cannot carry a
    # call-graph edge.  Final "hard reuse" claims require separate dynamic
    # winning-trace evidence.
    calls_by_definition = {
        name: versions[0]
        for name, versions in call_versions.items()
        if len(versions) == 1
    }
    reachable_calls: set[str] = set()
    pending = ["solve"] if "solve" in calls_by_definition else []
    visited: set[str] = set()
    while pending:
        function = pending.pop()
        if function in visited:
            continue
        visited.add(function)
        for called in calls_by_definition.get(function, set()):
            reachable_calls.add(called)
            if called in calls_by_definition and called not in visited:
                pending.append(called)
    definitions = {
        name: versions[0]
        for name, versions in definition_versions.items()
        if len(versions) == 1
    }
    return units, definitions, reachable_calls


@dataclass(frozen=True)
class SelectionEvidence:
    schema: Literal[1]
    metric: Literal[
        "positive_unmatched_normalized_ast_zlib_v1",
        "fixed_ignorance_prior_v1",
    ]
    parent_source_path: str
    parent_source_tree_sha256: str
    candidate_source_path: str | None
    candidate_source_tree_sha256: str | None
    conditional_novelty: int
    retained_normalized_units: int
    reused_definition_calls: tuple[str, ...]
    evidence_sha256: str


_MAX_SELECTION_EVIDENCE_CACHE_ENTRIES = 1024
_selection_evidence_cache: dict[
    tuple[object, ...], SelectionEvidence
] = {}


def _selection_evidence_body(
    *,
    metric: str,
    parent_source_path: str,
    parent_source_tree_sha256: str,
    candidate_source_path: str | None,
    candidate_source_tree_sha256: str | None,
    conditional_novelty: int,
    retained_normalized_units: int,
    reused_definition_calls: Sequence[str],
) -> dict[str, object]:
    return {
        "schema": 1,
        "metric": metric,
        "parent_source_path": parent_source_path,
        "parent_source_tree_sha256": parent_source_tree_sha256,
        "candidate_source_path": candidate_source_path,
        "candidate_source_tree_sha256": candidate_source_tree_sha256,
        "conditional_novelty": conditional_novelty,
        "retained_normalized_units": retained_normalized_units,
        "reused_definition_calls": list(reused_definition_calls),
    }


def selection_evidence(
    *,
    parent_source_path: str,
    parent_source_tree_sha256: str,
    candidate_source_path: str | None = None,
    candidate_source_tree_sha256: str | None = None,
) -> SelectionEvidence:
    """Measure conditional normalized-AST novelty and hard reuse witnesses."""

    parent_path = Path(parent_source_path)
    parent_pointer = _source_tree_pointer(parent_path)
    candidate_path = (
        Path(candidate_source_path)
        if candidate_source_path is not None
        else None
    )
    candidate_pointer = (
        _source_tree_pointer(candidate_path)
        if candidate_path is not None
        else None
    )
    cache_key = (
        str(parent_path),
        parent_source_tree_sha256,
        parent_pointer,
        str(candidate_path) if candidate_path is not None else None,
        candidate_source_tree_sha256,
        candidate_pointer,
    )
    cached = _selection_evidence_cache.get(cache_key)
    if cached is not None:
        if (
            _source_tree_pointer(parent_path) != parent_pointer
            or (
                candidate_path is not None
                and _source_tree_pointer(candidate_path)
                != candidate_pointer
            )
        ):
            raise SchedulerError(
                "solver source changed while cached evidence was checked"
            )
        return cached

    parent = _source_tree(parent_path, parent_source_tree_sha256)
    if candidate_source_path is None:
        if candidate_source_tree_sha256 is not None:
            raise SchedulerError("unknown-prior evidence has a candidate hash")
        body = _selection_evidence_body(
            metric=UNKNOWN_SELECTION_METRIC,
            parent_source_path=str(parent_path),
            parent_source_tree_sha256=parent_source_tree_sha256,
            candidate_source_path=None,
            candidate_source_tree_sha256=None,
            conditional_novelty=UNKNOWN_CONDITIONAL_NOVELTY,
            retained_normalized_units=0,
            reused_definition_calls=(),
        )
    else:
        if candidate_source_tree_sha256 is None:
            raise SchedulerError("candidate source lacks its tree hash")
        assert candidate_path is not None
        candidate = _source_tree(
            candidate_path, candidate_source_tree_sha256
        )
        parent_units, parent_definitions, _ = _normalized_units(parent)
        candidate_units, candidate_definitions, candidate_calls = (
            _normalized_units(candidate)
        )
        remaining = Counter(parent_units)
        novel: list[bytes] = []
        retained = 0
        for unit in candidate_units:
            if remaining[unit] > 0:
                remaining[unit] -= 1
                retained += 1
            else:
                novel.append(unit)
        novelty = len(zlib.compress(b"\n".join(novel), 9)) if novel else 0
        unchanged_definitions = {
            name
            for name, dumped in candidate_definitions.items()
            if parent_definitions.get(name) == dumped
        }
        reuse = tuple(sorted(candidate_calls & unchanged_definitions))
        body = _selection_evidence_body(
            metric=SELECTION_METRIC,
            parent_source_path=str(parent_path),
            parent_source_tree_sha256=parent_source_tree_sha256,
            candidate_source_path=str(candidate_path),
            candidate_source_tree_sha256=candidate_source_tree_sha256,
            conditional_novelty=novelty,
            retained_normalized_units=retained,
            reused_definition_calls=reuse,
        )
    evidence = SelectionEvidence(
        schema=1,
        metric=body["metric"],  # type: ignore[arg-type]
        parent_source_path=str(body["parent_source_path"]),
        parent_source_tree_sha256=str(
            body["parent_source_tree_sha256"]
        ),
        candidate_source_path=body["candidate_source_path"],  # type: ignore[arg-type]
        candidate_source_tree_sha256=body[
            "candidate_source_tree_sha256"
        ],  # type: ignore[arg-type]
        conditional_novelty=int(body["conditional_novelty"]),
        retained_normalized_units=int(body["retained_normalized_units"]),
        reused_definition_calls=tuple(
            str(item) for item in body["reused_definition_calls"]
        ),
        evidence_sha256=sha256_json(body),
    )
    # Never publish evidence computed across a concurrent pointer change.
    if (
        _source_tree_pointer(parent_path) != parent_pointer
        or (
            candidate_path is not None
            and _source_tree_pointer(candidate_path)
            != candidate_pointer
        )
    ):
        raise SchedulerError(
            "solver source changed while selection evidence was computed"
        )
    if len(_selection_evidence_cache) >= (
        _MAX_SELECTION_EVIDENCE_CACHE_ENTRIES
    ):
        _selection_evidence_cache.pop(
            next(iter(_selection_evidence_cache))
        )
    _selection_evidence_cache[cache_key] = evidence
    return evidence


def verify_selection_evidence(evidence: SelectionEvidence) -> None:
    expected = selection_evidence(
        parent_source_path=evidence.parent_source_path,
        parent_source_tree_sha256=evidence.parent_source_tree_sha256,
        candidate_source_path=evidence.candidate_source_path,
        candidate_source_tree_sha256=evidence.candidate_source_tree_sha256,
    )
    if expected != evidence:
        raise SchedulerError("selection evidence is stale or forged")


@dataclass(frozen=True)
class WipBinding:
    snapshot_id: str
    wip_root_path: str
    wip_tree_sha256: str
    solver_source_path: str
    solver_source_tree_sha256: str
    game: str
    target_level: int
    parent_checkpoint_sha256: str
    frontier_sha256: str
    codex_thread_id: str
    final_thread_binding_path: str
    final_thread_binding_sha256: str
    wip_export_receipt_path: str
    wip_export_receipt_sha256: str
    final_transcript_chain_receipt_path: str
    final_transcript_chain_receipt_sha256: str
    transcript_chain_sha256: str
    controller_state_scan_receipt_path: str
    controller_state_scan_receipt_sha256: str
    retained_canary_scan_receipt_path: str
    retained_canary_scan_receipt_sha256: str
    taint_scan_receipt_path: str
    taint_scan_receipt_sha256: str
    token_usage_receipt_path: str
    token_usage_receipt_sha256: str
    provider_usage_receipt_path: str
    provider_usage_receipt_sha256: str
    app_server_state_dir: str
    app_server_state_tree_sha256: str
    wip_publication_receipt_path: str
    wip_publication_receipt_sha256: str
    supervisory_handoff_sha256: str | None
    supervisory_native_reproduction_receipt_path: str | None
    supervisory_native_reproduction_receipt_sha256: str | None
    taint_verdict: Literal["clean"] = "clean"


@dataclass(frozen=True)
class Frontier:
    game: str
    target: int
    reached: int
    no_progress: int
    last_dispatch_sequence: int
    parent_checkpoint_sha256: str
    parent_source_path: str
    parent_source_tree_sha256: str
    frontier_sha256: str
    active_attempt_id: str | None
    draining: bool
    blocked_reason: str | None
    wip: WipBinding | None
    evidence: SelectionEvidence
    public_observation_receipt_sha256s: tuple[str, ...]
    observation_ledger_sha256: str


@dataclass(frozen=True)
class CampaignSnapshot:
    campaign_id: str
    journal_head_sequence: int
    journal_head_digest: str
    inventory: tuple[tuple[str, int], ...]
    max_lanes: int
    frontiers: tuple[Frontier, ...]
    budget: BudgetState
    clean_proposer_settlements: tuple[CleanProposerSettlement, ...] = ()
    complexity_rounds: tuple[ComplexityRoundState, ...] = ()
    auxiliary_assignments: tuple[AuxiliaryAssignmentState, ...] = ()
    sidecar_requests: tuple[SidecarRequestEvidence, ...] = ()


@dataclass(frozen=True)
class SupervisoryHandoffBinding:
    """Exact admitted handoff exposed only as a labeled hypothesis."""

    schema: Literal[1]
    assignment_id: str
    frontier_sha256: str
    parent_checkpoint_sha256: str
    output_manifest_sha256: str
    supervisory_handoff_sha256: str
    output: AuxiliaryOutputEvidence
    admission_receipt_path: str
    admission_receipt_sha256: str
    admitted_sequence: int
    admitted_event_digest: str
    prompt_authority: Literal["unverified_hypothesis_only"]
    derived_evidence_requires_native_reproduction: Literal[True]
    scheduler_authority: Literal[False]
    mutation_authority: Literal[False]
    promotion_authority: Literal[False]


@dataclass(frozen=True)
class DispatchChoice:
    game: str
    target_level: int
    authoritative_target: int
    no_progress: int
    effort: Literal["medium", "high", "xhigh", "max"]
    soft_allocation_seconds: int
    requested_wip_mode: Literal[
        "exclude", "restore_clean_same_frontier"
    ]
    effective_wip_mode: Literal[
        "exclude", "restore_clean_same_frontier"
    ]
    thread_mode: Literal["new", "resume"]
    selected_wip: WipBinding | None
    success_prior_micro: int
    conditional_novelty: int
    estimated_free_energy_micro: int
    reused_definition_calls: tuple[str, ...]
    ranking_key: tuple[int, int, int, str]
    slots_to_fill: int
    reservation_units: int | None
    selected_supervisory_handoff: SupervisoryHandoffBinding | None


def _require_bounded_text(
    value: object, label: str, *, maximum: int = 4096
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        raise SchedulerError(f"{label} must be nonempty bounded text")
    return value


def validate_socratic_challenge(
    value: SocraticChallengeEvidence,
) -> SocraticChallengeEvidence:
    if not isinstance(value, SocraticChallengeEvidence) or value.schema != 1:
        raise SchedulerError("Socratic challenge schema is invalid")
    for label, text in (
        ("hypothesis", value.hypothesis),
        ("counter hypothesis", value.counter_hypothesis),
        ("falsification attempt", value.falsification_attempt),
    ):
        _require_bounded_text(text, label)
    if (
        not value.observation_receipt_sha256s
        or len(set(value.observation_receipt_sha256s))
        != len(value.observation_receipt_sha256s)
        or any(
            not _is_sha256(item)
            for item in value.observation_receipt_sha256s
        )
        or (
            not value.rejected_conclusions
            and not value.surviving_conclusions
        )
    ):
        raise SchedulerError(
            "Socratic challenge lacks distinct public-observation evidence "
            "or conclusions"
        )
    for label, conclusions in (
        ("rejected conclusion", value.rejected_conclusions),
        ("surviving conclusion", value.surviving_conclusions),
    ):
        if len(set(conclusions)) != len(conclusions):
            raise SchedulerError(f"{label}s are duplicated")
        for conclusion in conclusions:
            _require_bounded_text(conclusion, label, maximum=2048)
    return value


def supervisory_handoff_sha256(
    value: SupervisoryHandoffEvidence,
) -> str:
    return sha256_json(asdict(value))


def validate_supervisory_handoff(
    value: SupervisoryHandoffEvidence,
    *,
    challenge: SocraticChallengeEvidence,
    output: AuxiliaryOutputEvidence | None = None,
) -> SupervisoryHandoffEvidence:
    """Reject any free-form, stale, or authoritative tactical handoff."""

    validate_socratic_challenge(challenge)
    if (
        not isinstance(value, SupervisoryHandoffEvidence)
        or value.schema != 1
        or value.kind != SUPERVISORY_HANDOFF_KIND
        or value.role != SUPERVISORY_PROPOSER_ROLE
        or not _is_identifier(value.handoff_id)
        or not _is_identifier(value.assignment_id)
        or not _is_sha256(value.frontier_sha256)
        or not _is_sha256(value.parent_checkpoint_sha256)
        or not _is_sha256(value.input_manifest_sha256)
        or not _is_identifier(value.model)
        or value.reasoning_effort
        not in SUPPORTED_AUXILIARY_REASONING_EFFORTS
        or not value.relied_on_observation_receipt_sha256s
        or len(set(value.relied_on_observation_receipt_sha256s))
        != len(value.relied_on_observation_receipt_sha256s)
        or any(
            not _is_sha256(item)
            for item in value.relied_on_observation_receipt_sha256s
        )
        or value.socratic_challenge_sha256
        != sha256_json(asdict(challenge))
        or value.result_authority != "quarantine_only"
        or value.native_reproduction_required is not True
        or value.raw_context_included is not False
        or value.scheduler_authority is not False
        or value.mutation_authority is not False
        or value.promotion_authority is not False
    ):
        raise SchedulerError(
            "SUPERVISORY_HANDOFF identity or authority is malformed"
        )
    _require_bounded_text(
        value.unresolved_obligation,
        "supervisory unresolved obligation",
    )
    _require_bounded_text(
        value.confidence_and_caveats,
        "supervisory confidence and caveats",
    )
    for label, rows in (
        ("rejected alternative", value.rejected_alternatives),
    ):
        if (
            not rows
            or len(rows) > 32
            or len(set(rows)) != len(rows)
        ):
            raise SchedulerError(
                f"SUPERVISORY_HANDOFF {label}s are absent or duplicated"
            )
        for row in rows:
            _require_bounded_text(row, label, maximum=2048)
    cited = set(value.relied_on_observation_receipt_sha256s)
    claim_ids: set[str] = set()
    if not value.claims or len(value.claims) > 32:
        raise SchedulerError(
            "SUPERVISORY_HANDOFF has no bounded typed claims"
        )
    for claim in value.claims:
        if (
            not isinstance(claim, SupervisoryHypothesisEvidence)
            or claim.schema != 1
            or not _is_identifier(claim.claim_id)
            or claim.claim_id in claim_ids
            or not claim.observation_receipt_sha256s
            or len(set(claim.observation_receipt_sha256s))
            != len(claim.observation_receipt_sha256s)
            or not set(claim.observation_receipt_sha256s) <= cited
            or not claim.falsifiers
            or len(set(claim.falsifiers)) != len(claim.falsifiers)
            or not claim.bounded_next_tests
            or len(set(claim.bounded_next_tests))
            != len(claim.bounded_next_tests)
        ):
            raise SchedulerError(
                "SUPERVISORY_HANDOFF claim is uncited or unfalsifiable"
            )
        claim_ids.add(claim.claim_id)
        _require_bounded_text(
            claim.statement, "supervisory hypothesis", maximum=2048
        )
        for label, rows in (
            ("supervisory falsifier", claim.falsifiers),
            ("supervisory bounded next test", claim.bounded_next_tests),
        ):
            for row in rows:
                _require_bounded_text(row, label, maximum=2048)
    if not set(challenge.observation_receipt_sha256s) <= cited:
        raise SchedulerError(
            "SUPERVISORY_HANDOFF self-challenge cites unrelied evidence"
        )
    if output is not None and (
        value.assignment_id != output.assignment_id
        or value.frontier_sha256 != output.frontier_sha256
        or value.parent_checkpoint_sha256
        != output.parent_checkpoint_sha256
        or value.input_manifest_sha256
        != output.input_manifest_sha256
        or not cited
        <= set(output.public_observation_receipt_sha256s)
    ):
        raise SchedulerError(
            "SUPERVISORY_HANDOFF is stale or cites outside its output"
        )
    return value


def validate_supervisory_native_reproduction(
    value: SupervisoryNativeReproductionReceipt,
    *,
    handoff: SupervisoryHandoffEvidence,
) -> SupervisoryNativeReproductionReceipt:
    """Require fresh native public evidence for every relied-on citation."""

    if (
        not isinstance(value, SupervisoryNativeReproductionReceipt)
        or value.schema != 1
        or value.kind != "SUPERVISORY_NATIVE_REPRODUCTION"
        or value.authority != "host_only"
        or value.assignment_id != handoff.assignment_id
        or value.frontier_sha256 != handoff.frontier_sha256
        or value.parent_checkpoint_sha256
        != handoff.parent_checkpoint_sha256
        or value.input_manifest_sha256 != handoff.input_manifest_sha256
        or value.supervisory_handoff_sha256
        != supervisory_handoff_sha256(handoff)
        or not _is_identifier(value.native_attempt_id)
        or not _is_sha256(value.native_attempt_spec_sha256)
        or not _is_sha256(
            value.native_arena_session_binding_receipt_sha256
        )
        or not _is_sha256(value.native_host_transcript_sha256)
        or not value.reproductions
        or value.fresh_native_session is not True
        or value.public_observation_interface_only is not True
        or value.supervisory_workspace_mounted is not False
        or value.scheduler_fields_overridden is not False
        or value.live_lineage_mutated is not False
        or value.promotion_authorized is not False
        or value.status != "PASS"
    ):
        raise SchedulerError(
            "supervisory native reproduction receipt is malformed"
        )
    source_rows: list[str] = []
    native_rows: list[str] = []
    for row in value.reproductions:
        if (
            not isinstance(row, NativeObservationReproduction)
            or row.schema != 1
            or row.status != "REPRODUCED"
            or any(
                not _is_sha256(item)
                for item in (
                    row.source_observation_receipt_sha256,
                    row.native_observation_receipt_sha256,
                    row.public_action_basis_sha256,
                    row.public_response_signature_sha256,
                )
            )
        ):
            raise SchedulerError(
                "supervisory native reproduction row is malformed"
            )
        source_rows.append(row.source_observation_receipt_sha256)
        native_rows.append(row.native_observation_receipt_sha256)
    if (
        len(source_rows) != len(set(source_rows))
        or len(native_rows) != len(set(native_rows))
        or set(source_rows)
        != set(handoff.relied_on_observation_receipt_sha256s)
    ):
        raise SchedulerError(
            "native reproduction does not cover every handoff citation once"
        )
    return value


def validate_supervisory_handoff_binding(
    value: SupervisoryHandoffBinding,
) -> SupervisoryHandoffBinding:
    if not isinstance(value, SupervisoryHandoffBinding):
        raise SchedulerError("supervisory handoff binding has wrong type")
    output = validate_auxiliary_output(value.output)
    handoff = output.supervisory_handoff
    if (
        value.schema != 1
        or handoff is None
        or not _is_identifier(value.assignment_id)
        or value.assignment_id != output.assignment_id
        or value.assignment_id != handoff.assignment_id
        or value.frontier_sha256 != output.frontier_sha256
        or value.frontier_sha256 != handoff.frontier_sha256
        or value.parent_checkpoint_sha256
        != output.parent_checkpoint_sha256
        or value.parent_checkpoint_sha256
        != handoff.parent_checkpoint_sha256
        or value.output_manifest_sha256
        != output.output_manifest_sha256
        or value.supervisory_handoff_sha256
        != supervisory_handoff_sha256(handoff)
        or not isinstance(value.admission_receipt_path, str)
        or not Path(value.admission_receipt_path).is_absolute()
        or not _is_sha256(value.admission_receipt_sha256)
        or not _is_int(value.admitted_sequence, minimum=1)
        or not _is_sha256(value.admitted_event_digest)
        or value.prompt_authority
        != "unverified_hypothesis_only"
        or value.derived_evidence_requires_native_reproduction is not True
        or value.scheduler_authority is not False
        or value.mutation_authority is not False
        or value.promotion_authority is not False
    ):
        raise SchedulerError(
            "supervisory handoff binding is stale or authoritative"
        )
    return value


def validate_auxiliary_output(
    value: AuxiliaryOutputEvidence,
    *,
    assignment: AuxiliaryAssignmentState | None = None,
) -> AuxiliaryOutputEvidence:
    if (
        not isinstance(value, AuxiliaryOutputEvidence)
        or value.schema != 1
        or not _is_identifier(value.assignment_id)
        or not _is_identifier(value.expert_id)
        or not _is_canonical_uuid(value.thread_id)
        or value.specialization not in ALL_AUXILIARY_SPECIALIZATIONS
        or not _is_sha256(value.frontier_sha256)
        or not _is_sha256(value.parent_checkpoint_sha256)
        or not _is_sha256(value.input_manifest_sha256)
        or not _is_sha256(value.output_manifest_sha256)
        or not value.public_observation_receipt_sha256s
        or len(set(value.public_observation_receipt_sha256s))
        != len(value.public_observation_receipt_sha256s)
        or any(
            not _is_sha256(item)
            for item in value.public_observation_receipt_sha256s
        )
        or len(set(value.quarantined_artifact_sha256s))
        != len(value.quarantined_artifact_sha256s)
        or any(
            not _is_sha256(item)
            for item in value.quarantined_artifact_sha256s
        )
        or value.result_authority != "quarantine_only"
        or value.mutates_live_lineage is not False
    ):
        raise SchedulerError("auxiliary output is not quarantine-only evidence")
    validate_socratic_challenge(value.challenge)
    if not set(value.challenge.observation_receipt_sha256s) <= set(
        value.public_observation_receipt_sha256s
    ):
        raise SchedulerError(
            "Socratic challenge references observations outside the "
            "quarantined output"
        )
    if value.supervisory_handoff is not None:
        validate_supervisory_handoff(
            value.supervisory_handoff,
            challenge=value.challenge,
            output=value,
        )
    if assignment is not None and (
        value.assignment_id != assignment.assignment_id
        or value.expert_id != assignment.expert_id
        or value.thread_id != assignment.thread_id
        or value.specialization != assignment.specialization
        or value.frontier_sha256 != assignment.frontier_sha256
        or value.parent_checkpoint_sha256
        != assignment.parent_checkpoint_sha256
        or value.input_manifest_sha256
        != assignment.input_manifest_sha256
        or not set(
            value.public_observation_receipt_sha256s
        )
        <= set(
            assignment.input_manifest
            .authenticated_public_observation_receipt_sha256s
        )
        or (
            value.supervisory_handoff is not None
            and assignment.input_manifest.input_role
            != SUPERVISORY_PROPOSER_ROLE
        )
        or (
            value.supervisory_handoff is not None
            and (
                assignment.role != SUPERVISORY_PROPOSER_ROLE
                or value.supervisory_handoff.model
                != assignment.model
                or value.supervisory_handoff.reasoning_effort
                != assignment.reasoning_effort
                or not _is_sha256(
                    assignment
                    .supervisory_launch_configuration_sha256
                )
            )
        )
        or (
            value.supervisory_handoff is None
            and assignment.input_manifest.input_role
            == SUPERVISORY_PROPOSER_ROLE
        )
    ):
        raise SchedulerError(
            "auxiliary output is substituted across an assignment boundary"
        )
    return value


def _normalized_semantic_text(value: str) -> str:
    """Normalize report prose without importing assignment-time metadata."""

    return " ".join(value.split())


def auxiliary_output_semantic_evidence_sha256(
    value: AuxiliaryOutputEvidence,
) -> str:
    """Content identity for an admitted non-supervisory expert report.

    Output manifests, assignment/thread identities, timestamps, and admission
    receipts remain provenance only.  Reissuing byte-identical findings under
    a fresh assignment therefore cannot create a new supervisory evidence
    epoch.
    """

    output = validate_auxiliary_output(value)
    if (
        output.specialization == SUPERVISORY_SPECIALIZATION
        or output.supervisory_handoff is not None
    ):
        raise SchedulerError(
            "supervisory output cannot masquerade as side-evidence content"
        )
    challenge = output.challenge
    return sha256_json(
        {
            "schema": 1,
            "kind": "auxiliary_side_evidence_content",
            "specialization": output.specialization,
            "public_observation_receipt_sha256s": sorted(
                output.public_observation_receipt_sha256s
            ),
            "socratic_findings": {
                "hypothesis": _normalized_semantic_text(
                    challenge.hypothesis
                ),
                "counter_hypothesis": _normalized_semantic_text(
                    challenge.counter_hypothesis
                ),
                "falsification_attempt": _normalized_semantic_text(
                    challenge.falsification_attempt
                ),
                "observation_receipt_sha256s": sorted(
                    challenge.observation_receipt_sha256s
                ),
                "rejected_conclusions": sorted(
                    _normalized_semantic_text(item)
                    for item in challenge.rejected_conclusions
                ),
                "surviving_conclusions": sorted(
                    _normalized_semantic_text(item)
                    for item in challenge.surviving_conclusions
                ),
            },
            "report_artifact_content_sha256s": sorted(
                output.quarantined_artifact_sha256s
            ),
        }
    )


def supervisory_handoff_semantic_sha256(
    value: AuxiliaryOutputEvidence,
) -> str:
    """Content identity excluding assignment, thread, and envelope metadata."""

    output = validate_auxiliary_output(value)
    handoff = output.supervisory_handoff
    if (
        output.specialization != SUPERVISORY_SPECIALIZATION
        or handoff is None
    ):
        raise SchedulerError(
            "semantic supervisory handoff digest requires a handoff output"
        )
    challenge = output.challenge
    claims = sorted(
        (
            {
                "statement": _normalized_semantic_text(
                    claim.statement
                ),
                "observation_receipt_sha256s": sorted(
                    claim.observation_receipt_sha256s
                ),
                "falsifiers": sorted(
                    _normalized_semantic_text(item)
                    for item in claim.falsifiers
                ),
                "bounded_next_tests": sorted(
                    _normalized_semantic_text(item)
                    for item in claim.bounded_next_tests
                ),
            }
            for claim in handoff.claims
        ),
        key=sha256_json,
    )
    return sha256_json(
        {
            "schema": 1,
            "kind": "supervisory_handoff_semantic_content",
            "relied_on_observation_receipt_sha256s": sorted(
                handoff.relied_on_observation_receipt_sha256s
            ),
            "claims": claims,
            "rejected_alternatives": sorted(
                _normalized_semantic_text(item)
                for item in handoff.rejected_alternatives
            ),
            "unresolved_obligation": _normalized_semantic_text(
                handoff.unresolved_obligation
            ),
            "confidence_and_caveats": _normalized_semantic_text(
                handoff.confidence_and_caveats
            ),
            "socratic_challenge": {
                "hypothesis": _normalized_semantic_text(
                    challenge.hypothesis
                ),
                "counter_hypothesis": _normalized_semantic_text(
                    challenge.counter_hypothesis
                ),
                "falsification_attempt": _normalized_semantic_text(
                    challenge.falsification_attempt
                ),
                "observation_receipt_sha256s": sorted(
                    challenge.observation_receipt_sha256s
                ),
                "rejected_conclusions": sorted(
                    _normalized_semantic_text(item)
                    for item in challenge.rejected_conclusions
                ),
                "surviving_conclusions": sorted(
                    _normalized_semantic_text(item)
                    for item in challenge.surviving_conclusions
                ),
            },
        }
    )


def _sidecar_request_body(
    value: SidecarRequestEvidence,
) -> dict[str, object]:
    body = asdict(value)
    body.pop("request_sha256", None)
    body["cited_public_observation_receipt_sha256s"] = list(
        value.cited_public_observation_receipt_sha256s
    )
    return body


def _native_sidecar_request_draft_body(
    value: NativeSidecarRequestDraft,
) -> dict[str, object]:
    body = asdict(value)
    body.pop("draft_sha256", None)
    body["cited_public_observation_receipt_sha256s"] = list(
        value.cited_public_observation_receipt_sha256s
    )
    return body


def validate_native_sidecar_request_draft(
    value: NativeSidecarRequestDraft,
) -> NativeSidecarRequestDraft:
    if (
        not isinstance(value, NativeSidecarRequestDraft)
        or value.schema != 1
        or value.kind != "NATIVE_SIDECAR_REQUEST_DRAFT"
        or not _is_identifier(value.request_id)
        or GAME_RE.fullmatch(value.game) is None
        or not _is_sha256(value.frontier_sha256)
        or not _is_sha256(value.parent_checkpoint_sha256)
        or not _is_identifier(value.native_attempt_id)
        or not isinstance(value.semantic_brief, str)
        or not value.semantic_brief.strip()
        or value.semantic_brief != value.semantic_brief.strip()
        or len(value.semantic_brief.encode("utf-8")) > 16_384
        or not value.cited_public_observation_receipt_sha256s
        or tuple(
            sorted(
                set(
                    value
                    .cited_public_observation_receipt_sha256s
                )
            )
        )
        != value.cited_public_observation_receipt_sha256s
        or any(
            not _is_sha256(item)
            for item in (
                value
                .cited_public_observation_receipt_sha256s
            )
        )
        or value.scheduler_authored is not False
        or value.live_lineage_mutation_authority is not False
        or value.promotion_authority is not False
        or value.draft_sha256
        != sha256_json(_native_sidecar_request_draft_body(value))
    ):
        raise SchedulerError(
            "native sidecar request draft is malformed or authoritative"
        )
    return value


def native_sidecar_request_from_draft(
    draft: NativeSidecarRequestDraft,
    *,
    settlement: CleanProposerSettlement,
) -> SidecarRequestEvidence:
    """Admit only the exact draft carried by its clean terminal result."""

    validate_native_sidecar_request_draft(draft)
    validate_clean_proposer_settlement(settlement)
    if (
        draft.game != settlement.game
        or draft.frontier_sha256 != settlement.frontier_sha256
        or draft.parent_checkpoint_sha256
        != settlement.parent_checkpoint_sha256
        or draft.native_attempt_id != settlement.attempt_id
    ):
        raise SchedulerError(
            "native sidecar request draft is stale or cross-attempt"
        )
    request = SidecarRequestEvidence(
        schema=1,
        kind="NATIVE_SIDECAR_REQUEST",
        request_id=draft.request_id,
        game=draft.game,
        frontier_sha256=draft.frontier_sha256,
        parent_checkpoint_sha256=draft.parent_checkpoint_sha256,
        authority="native_proposer",
        semantic_brief=draft.semantic_brief,
        cited_public_observation_receipt_sha256s=(
            draft.cited_public_observation_receipt_sha256s
        ),
        native_attempt_id=draft.native_attempt_id,
        supervisory_assignment_id=None,
        supervisory_handoff_sha256=None,
        origin_admission_receipt_sha256=settlement.result_digest,
        scheduler_authored=False,
        live_lineage_mutation_authority=False,
        promotion_authority=False,
        request_sha256="",
    )
    return validate_sidecar_request(
        replace(
            request,
            request_sha256=sha256_json(
                _sidecar_request_body(request)
            ),
        )
    )


def supervisory_sidecar_request_from_assignment(
    assignment: AuxiliaryAssignmentState,
) -> SidecarRequestEvidence:
    """Project one request verbatim from an admitted supervisory handoff."""

    validate_auxiliary_assignment(assignment)
    output = assignment.output
    handoff = (
        None if output is None else output.supervisory_handoff
    )
    if (
        assignment.phase != "ADMITTED"
        or assignment.role != SUPERVISORY_PROPOSER_ROLE
        or output is None
        or handoff is None
        or assignment.admission_receipt_sha256 is None
    ):
        raise SchedulerError(
            "supervisory sidecar request lacks admitted handoff origin"
        )
    request = SidecarRequestEvidence(
        schema=1,
        kind="SUPERVISORY_SIDECAR_REQUEST",
        request_id=f"{handoff.handoff_id}:sidecar",
        game=assignment.game,
        frontier_sha256=assignment.frontier_sha256,
        parent_checkpoint_sha256=(
            assignment.parent_checkpoint_sha256
        ),
        authority="admitted_supervisory_proposer",
        # The deterministic host does not paraphrase or extend tactics.
        semantic_brief=handoff.unresolved_obligation,
        cited_public_observation_receipt_sha256s=tuple(
            sorted(
                set(
                    handoff
                    .relied_on_observation_receipt_sha256s
                )
            )
        ),
        native_attempt_id=None,
        supervisory_assignment_id=assignment.assignment_id,
        supervisory_handoff_sha256=(
            supervisory_handoff_sha256(handoff)
        ),
        origin_admission_receipt_sha256=(
            assignment.admission_receipt_sha256
        ),
        scheduler_authored=False,
        live_lineage_mutation_authority=False,
        promotion_authority=False,
        request_sha256="",
    )
    return validate_sidecar_request(
        replace(
            request,
            request_sha256=sha256_json(
                _sidecar_request_body(request)
            ),
        )
    )


def validate_sidecar_request(
    value: SidecarRequestEvidence,
) -> SidecarRequestEvidence:
    native = (
        isinstance(value, SidecarRequestEvidence)
        and value.kind == "NATIVE_SIDECAR_REQUEST"
    )
    supervisory = (
        isinstance(value, SidecarRequestEvidence)
        and value.kind == "SUPERVISORY_SIDECAR_REQUEST"
    )
    if (
        not isinstance(value, SidecarRequestEvidence)
        or value.schema != 1
        or not (native or supervisory)
        or not _is_identifier(value.request_id)
        or GAME_RE.fullmatch(value.game) is None
        or not _is_sha256(value.frontier_sha256)
        or not _is_sha256(value.parent_checkpoint_sha256)
        or value.authority
        != (
            "native_proposer"
            if native
            else "admitted_supervisory_proposer"
        )
        or not isinstance(value.semantic_brief, str)
        or not value.semantic_brief.strip()
        or value.semantic_brief != value.semantic_brief.strip()
        or len(value.semantic_brief.encode("utf-8")) > 16_384
        or not value.cited_public_observation_receipt_sha256s
        or tuple(
            sorted(
                set(
                    value
                    .cited_public_observation_receipt_sha256s
                )
            )
        )
        != value.cited_public_observation_receipt_sha256s
        or any(
            not _is_sha256(item)
            for item in (
                value
                .cited_public_observation_receipt_sha256s
            )
        )
        or (
            native
            and (
                not _is_identifier(value.native_attempt_id)
                or value.supervisory_assignment_id is not None
                or value.supervisory_handoff_sha256 is not None
            )
        )
        or (
            supervisory
            and (
                value.native_attempt_id is not None
                or not _is_identifier(
                    value.supervisory_assignment_id
                )
                or not _is_sha256(
                    value.supervisory_handoff_sha256
                )
            )
        )
        or not _is_sha256(
            value.origin_admission_receipt_sha256
        )
        or value.scheduler_authored is not False
        or value.live_lineage_mutation_authority is not False
        or value.promotion_authority is not False
        or value.request_sha256
        != sha256_json(_sidecar_request_body(value))
    ):
        raise SchedulerError(
            "sidecar request provenance is malformed"
        )
    return value


def validate_auxiliary_input_manifest(
    value: AuxiliaryInputManifestCommitment,
) -> AuxiliaryInputManifestCommitment:
    if (
        not isinstance(value, AuxiliaryInputManifestCommitment)
        or value.schema != 1
        or value.kind != "planned_auxiliary_private_input"
        or not isinstance(value.game, str)
        or GAME_RE.fullmatch(value.game) is None
        or not _is_sha256(value.frontier_sha256)
        or not _is_sha256(value.parent_checkpoint_sha256)
        or not _is_sha256(value.parent_source_tree_sha256)
        or (
            value.wip_snapshot_id is not None
            and not _is_identifier(value.wip_snapshot_id)
        )
        or (
            value.wip_tree_sha256 is not None
            and not _is_sha256(value.wip_tree_sha256)
        )
        or (
            value.wip_solver_source_tree_sha256 is not None
            and not _is_sha256(
                value.wip_solver_source_tree_sha256
            )
        )
        or (
            (value.wip_snapshot_id is None)
            != (value.wip_tree_sha256 is None)
        )
        or (
            (value.wip_snapshot_id is None)
            != (value.wip_solver_source_tree_sha256 is None)
        )
        or not _is_sha256(value.observation_ledger_sha256)
        or (
            value.profile_id is not None
            and not _is_identifier(value.profile_id)
        )
        or not _is_int(value.round_index)
        or value.specialization not in ALL_AUXILIARY_SPECIALIZATIONS
        or (
            value.specialization == "complexity_diagnosis"
            and value.profile_id is not None
        )
        or (
            value.specialization != "complexity_diagnosis"
            and value.profile_id is None
        )
        or not _is_sha256(value.input_bundle_contract_sha256)
        or value.immutable_inputs is not True
        or value.live_lineage_mounted is not False
        or value.public_observations_only is not True
        or value.input_role
        not in {"side_expert", SUPERVISORY_PROPOSER_ROLE}
        or value.allowed_input_classes
        != (
            SIDE_EXPERT_ALLOWED_INPUT_CLASSES
            if value.input_role == "side_expert"
            else SUPERVISORY_ALLOWED_INPUT_CLASSES
        )
        or value.forbidden_input_classes
        != SUPERVISORY_FORBIDDEN_INPUT_CLASSES
        or value.input_allowlist_sha256
        != auxiliary_input_allowlist_sha256(role=value.input_role)
        or any(
            not _is_sha256(item)
            for item in (
                *value
                .authenticated_public_observation_receipt_sha256s,
                *value.native_solver_source_tree_sha256s,
                *value.authenticated_side_expert_evidence_sha256s,
            )
        )
        or tuple(
            sorted(
                value
                .authenticated_public_observation_receipt_sha256s
            )
        )
        != value.authenticated_public_observation_receipt_sha256s
        or tuple(sorted(value.native_solver_source_tree_sha256s))
        != value.native_solver_source_tree_sha256s
        or tuple(
            sorted(
                value.authenticated_side_expert_evidence_sha256s
            )
        )
        != value.authenticated_side_expert_evidence_sha256s
        or len(
            set(
                value
                .authenticated_public_observation_receipt_sha256s
            )
        )
        != len(
            value
            .authenticated_public_observation_receipt_sha256s
        )
        or len(set(value.native_solver_source_tree_sha256s))
        != len(value.native_solver_source_tree_sha256s)
        or not value.authenticated_public_observation_receipt_sha256s
        or value.native_solver_source_tree_sha256s
        != (
            ()
            if value.wip_solver_source_tree_sha256 is None
            else (value.wip_solver_source_tree_sha256,)
        )
        or len(
            set(value.authenticated_side_expert_evidence_sha256s)
        )
        != len(value.authenticated_side_expert_evidence_sha256s)
        or (
            value.input_role == "side_expert"
            and value.authenticated_side_expert_evidence_sha256s
        )
        or (
            value.input_role == SUPERVISORY_PROPOSER_ROLE
            and not (
                value
                .authenticated_public_observation_receipt_sha256s
                or value.native_solver_source_tree_sha256s
                or value.authenticated_side_expert_evidence_sha256s
            )
        )
        or value.authenticated_evidence_set_sha256
        != sha256_json(
            {
                "public_observation_receipts": list(
                    value
                    .authenticated_public_observation_receipt_sha256s
                ),
                "native_solver_source_trees": list(
                    value.native_solver_source_tree_sha256s
                ),
                "side_expert": list(
                    value.authenticated_side_expert_evidence_sha256s
                ),
            }
        )
        or validate_sidecar_request(value.sidecar_request)
        != value.sidecar_request
        or value.sidecar_request_sha256
        != value.sidecar_request.request_sha256
        or value.sidecar_request.game != value.game
        or value.sidecar_request.frontier_sha256
        != value.frontier_sha256
        or value.sidecar_request.parent_checkpoint_sha256
        != value.parent_checkpoint_sha256
        or not set(
            value.sidecar_request
            .cited_public_observation_receipt_sha256s
        ).issubset(
            value
            .authenticated_public_observation_receipt_sha256s
        )
        or value.sealed_input_required is not True
        or value.symlinks_allowed is not False
        or value.hardlinks_allowed is not False
        or value.path_escapes_allowed is not False
    ):
        raise SchedulerError(
            "planned auxiliary input manifest is malformed"
        )
    return value


def validate_clean_proposer_settlement(
    value: CleanProposerSettlement,
) -> CleanProposerSettlement:
    if (
        not isinstance(value, CleanProposerSettlement)
        or value.schema != 1
        or not isinstance(value.game, str)
        or GAME_RE.fullmatch(value.game) is None
        or not _is_sha256(value.frontier_sha256)
        or not _is_sha256(value.parent_checkpoint_sha256)
        or not _is_identifier(value.attempt_id)
        or not _is_identifier(value.scheduler_decision_id)
        or not _is_int(value.no_progress_before)
        or not _is_int(value.soft_allocation_seconds, minimum=1)
        or (
            value.supervisory_handoff_sha256 is not None
            and not _is_sha256(value.supervisory_handoff_sha256)
        )
        or not _is_int(value.result_sequence, minimum=1)
        or not _is_sha256(value.result_digest)
    ):
        raise SchedulerError("clean proposer settlement is malformed")
    policy = retry_policy(value.no_progress_before)
    if (
        value.effort != policy.effort
        or value.soft_allocation_seconds
        != policy.soft_allocation_seconds
        or value.requested_wip_mode != policy.requested_wip_mode
    ):
        raise SchedulerError(
            "clean proposer settlement does not follow the retry ladder"
        )
    return value


def validate_complexity_round(
    value: ComplexityRoundState,
) -> ComplexityRoundState:
    if (
        not isinstance(value, ComplexityRoundState)
        or value.schema != 1
        or not isinstance(value.game, str)
        or GAME_RE.fullmatch(value.game) is None
        or not _is_sha256(value.frontier_sha256)
        or not _is_sha256(value.parent_checkpoint_sha256)
        or not _is_sha256(value.parent_source_tree_sha256)
        or not _is_int(value.round_index)
        or not _is_identifier(value.diagnosis_assignment_id)
        or not _is_int(
            value.trigger_no_progress,
            minimum=AUXILIARY_ANALYSIS_START_NO_PROGRESS,
        )
        or not _is_sha256(value.trigger_history_sha256)
        or not _is_sha256(value.input_manifest_sha256)
        or not _is_sha256(value.observation_ledger_sha256)
        or not isinstance(value.admission_receipt_path, str)
        or not Path(value.admission_receipt_path).is_absolute()
        or not _is_sha256(value.admission_receipt_sha256)
        or not _is_int(value.admitted_sequence, minimum=1)
        or not _is_sha256(value.admitted_event_digest)
        or not isinstance(value.invalidated, bool)
        or value.profile.round_index != value.round_index
        or value.profile.frontier_sha256 != value.frontier_sha256
    ):
        raise SchedulerError("complexity round is not exactly bound")
    validate_complexity_profile(
        value.profile, frontier_sha256=value.frontier_sha256
    )
    return value


def validate_auxiliary_assignment(
    value: AuxiliaryAssignmentState,
) -> AuxiliaryAssignmentState:
    if (
        not isinstance(value, AuxiliaryAssignmentState)
        or value.schema != 1
        or any(
            not _is_identifier(item)
            for item in (
                value.assignment_id,
                value.decision_id,
                value.reservation_id,
                value.expert_id,
                value.active_proposer_attempt_id,
            )
        )
        or not isinstance(value.game, str)
        or GAME_RE.fullmatch(value.game) is None
        or not _is_sha256(value.frontier_sha256)
        or not _is_sha256(value.parent_checkpoint_sha256)
        or not _is_int(
            value.trigger_no_progress,
            minimum=AUXILIARY_ANALYSIS_START_NO_PROGRESS,
        )
        or not _is_sha256(value.trigger_history_sha256)
        or (
            value.profile_id is not None
            and not _is_identifier(value.profile_id)
        )
        or not _is_int(value.round_index)
        or value.specialization not in ALL_AUXILIARY_SPECIALIZATIONS
        or not _is_canonical_uuid(value.thread_id)
        or not _is_sha256(value.input_manifest_sha256)
        or value.input_manifest_sha256
        != sha256_json(
            asdict(
                validate_auxiliary_input_manifest(
                    value.input_manifest
                )
            )
        )
        or (
            value.specialization == SUPERVISORY_SPECIALIZATION
        )
        != (
            value.input_manifest.input_role
            == SUPERVISORY_PROPOSER_ROLE
        )
        or validate_sidecar_request(value.sidecar_request)
        != value.sidecar_request
        or value.sidecar_request_sha256
        != value.sidecar_request.request_sha256
        or value.sidecar_request
        != value.input_manifest.sidecar_request
        or value.sidecar_request_sha256
        != value.input_manifest.sidecar_request_sha256
        or not _is_sha256(value.observation_ledger_sha256)
        or not _is_identifier(value.model)
        or value.reasoning_effort
        not in SUPPORTED_AUXILIARY_REASONING_EFFORTS
        or value.role
        not in {"side_expert", SUPERVISORY_PROPOSER_ROLE}
        or (
            value.specialization == SUPERVISORY_SPECIALIZATION
        )
        != (value.role == SUPERVISORY_PROPOSER_ROLE)
        or (
            value.role == SUPERVISORY_PROPOSER_ROLE
            and (
                not _is_int(
                    value.context_limit_tokens, minimum=1
                )
                or value.context_limit_tokens > 2_000_000
                or value.role_max_concurrency != 1
                or isinstance(value.role_max_concurrency, bool)
                or not _is_sha256(
                    value.supervisory_launch_configuration_sha256
                )
            )
        )
        or (
            value.role == "side_expert"
            and any(
                item is not None
                for item in (
                    value.context_limit_tokens,
                    value.role_max_concurrency,
                    value.supervisory_launch_configuration_sha256,
                )
            )
        )
        or value.phase
        not in {
            "RESERVED",
            "INPUT_PREPARED",
            "RUNNING",
            "QUARANTINED",
            "ADMITTED",
            "REJECTED",
            "ABORTED",
        }
        or not isinstance(value.invalidated, bool)
        or (
            value.specialization == "complexity_diagnosis"
            and value.profile_id is not None
        )
        or (
            value.specialization != "complexity_diagnosis"
            and value.profile_id is None
        )
        or (
            value.phase
            in {"QUARANTINED", "ADMITTED", "REJECTED"}
        )
        != (value.output is not None)
        or (value.phase == "ABORTED" and value.output is not None)
        or (
            value.phase == "ADMITTED"
            and (
                not isinstance(value.admission_receipt_path, str)
                or not Path(value.admission_receipt_path).is_absolute()
                or not _is_sha256(value.admission_receipt_sha256)
                or not _is_int(value.admitted_sequence, minimum=1)
                or not _is_sha256(value.admitted_event_digest)
            )
        )
        or (
            value.phase != "ADMITTED"
            and any(
                item is not None
                for item in (
                    value.admission_receipt_path,
                    value.admission_receipt_sha256,
                    value.admitted_sequence,
                    value.admitted_event_digest,
                )
            )
        )
    ):
        raise SchedulerError("auxiliary assignment state is malformed")
    if value.output is not None:
        validate_auxiliary_output(value.output, assignment=value)
    return value


def _active_auxiliary_assignments(
    snapshot: CampaignSnapshot,
) -> tuple[AuxiliaryAssignmentState, ...]:
    return tuple(
        item
        for item in snapshot.auxiliary_assignments
        if item.phase in AUXILIARY_ACTIVE_PHASES
    )


def _validate_wip(frontier: Frontier) -> None:
    wip = frontier.wip
    if wip is None:
        return
    for label, value in (
        ("WIP tree", wip.wip_tree_sha256),
        ("WIP solver source", wip.solver_source_tree_sha256),
        ("WIP parent", wip.parent_checkpoint_sha256),
        ("WIP frontier", wip.frontier_sha256),
        ("WIP thread", wip.final_thread_binding_sha256),
        ("WIP export", wip.wip_export_receipt_sha256),
        (
            "WIP transcript receipt",
            wip.final_transcript_chain_receipt_sha256,
        ),
        ("WIP transcript", wip.transcript_chain_sha256),
        ("WIP state scan", wip.controller_state_scan_receipt_sha256),
        (
            "WIP retained canary scan",
            wip.retained_canary_scan_receipt_sha256,
        ),
        ("WIP taint scan", wip.taint_scan_receipt_sha256),
        ("WIP token usage", wip.token_usage_receipt_sha256),
        ("WIP provider usage", wip.provider_usage_receipt_sha256),
        ("WIP state", wip.app_server_state_tree_sha256),
        ("WIP publication", wip.wip_publication_receipt_sha256),
    ):
        _require_sha256(value, label)
    if wip.supervisory_handoff_sha256 is not None:
        _require_sha256(
            wip.supervisory_handoff_sha256,
            "WIP supervisory handoff",
        )
        _require_sha256(
            wip.supervisory_native_reproduction_receipt_sha256,
            "WIP supervisory native reproduction",
        )
    if (
        any(
            not isinstance(value, str)
            or not Path(value).is_absolute()
            for value in (
                wip.wip_root_path,
                wip.solver_source_path,
                wip.final_thread_binding_path,
                wip.wip_export_receipt_path,
                wip.final_transcript_chain_receipt_path,
                wip.controller_state_scan_receipt_path,
                wip.retained_canary_scan_receipt_path,
                wip.taint_scan_receipt_path,
                wip.token_usage_receipt_path,
                wip.provider_usage_receipt_path,
                wip.app_server_state_dir,
                wip.wip_publication_receipt_path,
                *(
                    (
                        str(
                            wip
                            .supervisory_native_reproduction_receipt_path
                        ),
                    )
                    if wip.supervisory_handoff_sha256 is not None
                    else ()
                ),
            )
        )
        or (
            wip.supervisory_handoff_sha256 is None
            and any(
                item is not None
                for item in (
                    wip
                    .supervisory_native_reproduction_receipt_path,
                    wip
                    .supervisory_native_reproduction_receipt_sha256,
                )
            )
        )
        or Path(wip.solver_source_path).parent
        != Path(wip.wip_root_path)
        or wip.game != frontier.game
        or wip.target_level != frontier.reached + 1
        or wip.parent_checkpoint_sha256
        != frontier.parent_checkpoint_sha256
        or wip.frontier_sha256 != frontier.frontier_sha256
        or wip.taint_verdict != "clean"
        or not _is_canonical_uuid(wip.codex_thread_id)
    ):
        raise SchedulerError("WIP is not bound to the exact clean frontier")


def validate_snapshot(snapshot: CampaignSnapshot) -> CampaignSnapshot:
    _require_identifier(snapshot.campaign_id, "campaign_id")
    if not _is_int(snapshot.journal_head_sequence):
        raise SchedulerError("journal head sequence is invalid")
    _require_sha256(snapshot.journal_head_digest, "journal head")
    inventory = validate_inventory(dict(snapshot.inventory))
    if tuple(inventory.items()) != snapshot.inventory:
        raise SchedulerError("snapshot inventory is not canonical")
    if (
        not _is_int(snapshot.max_lanes, minimum=1)
        or snapshot.max_lanes > MAX_LANES
    ):
        raise SchedulerError("lane capacity is outside 1..6")
    if tuple(sorted(item.game for item in snapshot.frontiers)) != tuple(
        inventory
    ):
        raise SchedulerError("frontier set does not match inventory")
    active_games: set[str] = set()
    for frontier in snapshot.frontiers:
        if (
            frontier.target != inventory[frontier.game]
            or not _is_int(frontier.reached)
            or frontier.reached > frontier.target
            or not _is_int(frontier.no_progress)
            or not _is_int(frontier.last_dispatch_sequence)
        ):
            raise SchedulerError("frontier target/progress is invalid")
        for label, value in (
            ("parent checkpoint", frontier.parent_checkpoint_sha256),
            ("parent source", frontier.parent_source_tree_sha256),
            ("frontier", frontier.frontier_sha256),
        ):
            _require_sha256(value, label)
        if frontier.active_attempt_id is not None:
            _require_identifier(frontier.active_attempt_id, "active attempt")
            if frontier.game in active_games:
                raise SchedulerError("same game has overlapping active attempts")
            active_games.add(frontier.game)
        if frontier.draining and frontier.active_attempt_id is None:
            raise SchedulerError("draining frontier has no active attempt")
        _validate_wip(frontier)
        verify_selection_evidence(frontier.evidence)
        expected_candidate = (
            frontier.wip.solver_source_path
            if frontier.wip is not None else None
        )
        expected_candidate_sha = (
            frontier.wip.solver_source_tree_sha256
            if frontier.wip is not None else None
        )
        if (
            frontier.evidence.parent_source_path
            != frontier.parent_source_path
            or frontier.evidence.parent_source_tree_sha256
            != frontier.parent_source_tree_sha256
            or frontier.evidence.candidate_source_path
            != expected_candidate
            or frontier.evidence.candidate_source_tree_sha256
            != expected_candidate_sha
        ):
            raise SchedulerError(
                "selection evidence is not bound to the frontier/WIP"
            )
        if (
            tuple(
                sorted(set(
                    frontier.public_observation_receipt_sha256s
                ))
            )
            != frontier.public_observation_receipt_sha256s
            or any(
                not _is_sha256(item)
                for item in (
                    frontier.public_observation_receipt_sha256s
                )
            )
            or frontier.observation_ledger_sha256
            != public_observation_ledger_sha256(
                game=frontier.game,
                frontier_sha256=frontier.frontier_sha256,
                parent_checkpoint_sha256=(
                    frontier.parent_checkpoint_sha256
                ),
                receipt_sha256s=(
                    frontier.public_observation_receipt_sha256s
                ),
            )
        ):
            raise SchedulerError(
                "frontier public-observation ledger is not exact"
            )
    settlements_by_identity: set[tuple[str, int]] = set()
    for settlement in snapshot.clean_proposer_settlements:
        validate_clean_proposer_settlement(settlement)
        identity = (
            settlement.frontier_sha256,
            settlement.no_progress_before,
        )
        if identity in settlements_by_identity:
            raise SchedulerError(
                "clean proposer retry coordinate is duplicated"
            )
        settlements_by_identity.add(identity)
    request_ids: set[str] = set()
    request_sha256s: set[str] = set()
    request_origins: set[tuple[str, str]] = set()
    frontier_by_coordinate = {
        (
            item.game,
            item.frontier_sha256,
            item.parent_checkpoint_sha256,
        ): item
        for item in snapshot.frontiers
    }
    for request in snapshot.sidecar_requests:
        validate_sidecar_request(request)
        frontier = frontier_by_coordinate.get(
            (
                request.game,
                request.frontier_sha256,
                request.parent_checkpoint_sha256,
            )
        )
        if (
            request.request_id in request_ids
            or request.request_sha256 in request_sha256s
            or (
                request.authority,
                str(
                    request.native_attempt_id
                    if request.authority == "native_proposer"
                    else request.supervisory_assignment_id
                ),
            )
            in request_origins
            or frontier is None
            or not set(
                request
                .cited_public_observation_receipt_sha256s
            ).issubset(
                frontier.public_observation_receipt_sha256s
            )
        ):
            raise SchedulerError(
                "sidecar request is duplicated, stale, or cross-frontier"
            )
        request_ids.add(request.request_id)
        request_sha256s.add(request.request_sha256)
        request_origins.add(
            (
                request.authority,
                str(
                    request.native_attempt_id
                    if request.authority == "native_proposer"
                    else request.supervisory_assignment_id
                ),
            )
        )
    rounds_by_frontier: dict[
        tuple[str, str], list[ComplexityRoundState]
    ] = {}
    profile_ids: set[str] = set()
    for round_state in snapshot.complexity_rounds:
        validate_complexity_round(round_state)
        if round_state.profile.profile_id in profile_ids:
            raise SchedulerError("complexity profile identity is reused")
        profile_ids.add(round_state.profile.profile_id)
        rounds_by_frontier.setdefault(
            (round_state.game, round_state.frontier_sha256), []
        ).append(round_state)
    for rows in rounds_by_frontier.values():
        indexes = sorted(item.round_index for item in rows)
        if indexes != list(range(len(indexes))):
            raise SchedulerError(
                "complexity round indexes are not contiguous from zero"
            )
    assignment_ids: set[str] = set()
    decision_ids: set[str] = set()
    reservation_ids: set[str] = set()
    expert_ids: set[str] = set()
    thread_ids: set[str] = set()
    live_round_obligations: set[tuple[str, str, int, str]] = set()
    per_profile_specializations: set[tuple[str, str]] = set()
    round_lookup = {
        (item.game, item.frontier_sha256, item.profile.profile_id): item
        for item in snapshot.complexity_rounds
    }
    frontier_lookup = {item.game: item for item in snapshot.frontiers}
    active_per_frontier: Counter[str] = Counter()
    active_supervisory_per_frontier: Counter[str] = Counter()
    active_supervisory_total = 0
    admitted_supervisory_semantics: set[
        tuple[str, str, str, str]
    ] = set()
    consumed_sidecar_request_sha256s: set[str] = set()
    for assignment in snapshot.auxiliary_assignments:
        validate_auxiliary_assignment(assignment)
        if (
            (
                not assignment.invalidated
                and (
                    assignment.sidecar_request_sha256
                    not in request_sha256s
                    or assignment.sidecar_request
                    not in snapshot.sidecar_requests
                )
            )
            or (
                not assignment.invalidated
                and assignment.sidecar_request_sha256
                in consumed_sidecar_request_sha256s
            )
        ):
            raise SchedulerError(
                "auxiliary assignment repeats or uses an absent sidecar "
                "request"
            )
        if not assignment.invalidated:
            consumed_sidecar_request_sha256s.add(
                assignment.sidecar_request_sha256
            )
        if (
            assignment.phase == "ADMITTED"
            and not assignment.invalidated
            and assignment.role == SUPERVISORY_PROPOSER_ROLE
            and assignment.output is not None
        ):
            semantic_identity = (
                assignment.game,
                assignment.frontier_sha256,
                assignment.parent_checkpoint_sha256,
                supervisory_handoff_semantic_sha256(
                    assignment.output
                ),
            )
            if semantic_identity in admitted_supervisory_semantics:
                raise SchedulerError(
                    "supervisory handoff repeats admitted semantic content"
                )
            admitted_supervisory_semantics.add(semantic_identity)
        for value, seen, label in (
            (assignment.assignment_id, assignment_ids, "assignment"),
            (assignment.decision_id, decision_ids, "auxiliary decision"),
            (
                assignment.reservation_id,
                reservation_ids,
                "auxiliary reservation",
            ),
            (assignment.expert_id, expert_ids, "expert"),
            (assignment.thread_id, thread_ids, "expert thread"),
        ):
            if value in seen:
                raise SchedulerError(f"{label} identity is reused")
            seen.add(value)
        if assignment.profile_id is not None:
            round_state = round_lookup.get(
                (
                    assignment.game,
                    assignment.frontier_sha256,
                    assignment.profile_id,
                )
            )
            if (
                round_state is None
                or round_state.round_index != assignment.round_index
                or (
                    assignment.specialization
                    != SUPERVISORY_SPECIALIZATION
                    and assignment.specialization
                    not in round_state.profile.priorities
                )
            ):
                raise SchedulerError(
                    "auxiliary assignment is outside its exact profile/round"
                )
            if assignment.phase != "ABORTED":
                profile_specialization = (
                    assignment.profile_id,
                    assignment.specialization
                    if assignment.specialization
                    != SUPERVISORY_SPECIALIZATION
                    else (
                        SUPERVISORY_SPECIALIZATION
                        + ":"
                        + assignment.input_manifest
                        .authenticated_evidence_set_sha256
                    ),
                )
                if profile_specialization in per_profile_specializations:
                    raise SchedulerError(
                        "complexity profile repeats an auxiliary obligation"
                    )
                per_profile_specializations.add(profile_specialization)
        if (
            not assignment.invalidated
            and assignment.phase not in {"ABORTED", "REJECTED"}
        ):
            obligation = (
                assignment.game,
                assignment.frontier_sha256,
                assignment.round_index,
                assignment.specialization,
            )
            if obligation in live_round_obligations:
                raise SchedulerError(
                    "frontier round repeats an auxiliary obligation"
                )
            live_round_obligations.add(obligation)
        if assignment.phase in AUXILIARY_ACTIVE_PHASES:
            frontier = frontier_lookup.get(assignment.game)
            if frontier is None or (
                not assignment.invalidated
                and (
                    frontier.frontier_sha256
                    != assignment.frontier_sha256
                    or frontier.parent_checkpoint_sha256
                    != assignment.parent_checkpoint_sha256
                )
            ):
                raise SchedulerError(
                    "active auxiliary assignment is stale or invalidated"
                )
            if not assignment.invalidated:
                active_per_frontier[assignment.frontier_sha256] += 1
                if (
                    assignment.specialization
                    == SUPERVISORY_SPECIALIZATION
                ):
                    active_supervisory_total += 1
                    active_supervisory_per_frontier[
                        assignment.frontier_sha256
                    ] += 1
                    if (
                        active_supervisory_per_frontier[
                            assignment.frontier_sha256
                        ]
                        > 1
                    ):
                        raise SchedulerError(
                            "more than one supervisory proposer is active "
                            "on one frontier"
                        )
                if (
                    active_per_frontier[assignment.frontier_sha256]
                    > (
                        MAX_AUXILIARY_ANALYSES_PER_FRONTIER
                        if frontier.no_progress
                        >= AUXILIARY_ANALYSIS_EXPAND_NO_PROGRESS
                        else 1
                    )
                ):
                    raise SchedulerError(
                        "active auxiliary assignments exceed the retry cap"
                    )
    if active_supervisory_total > 1:
        raise SchedulerError(
            "active supervisory proposers exceed global role concurrency"
        )
    settlements_by_attempt_id = {
        item.attempt_id: item
        for item in snapshot.clean_proposer_settlements
    }
    admitted_supervisors = {
        item.assignment_id: item
        for item in snapshot.auxiliary_assignments
        if item.phase == "ADMITTED"
        and item.role == SUPERVISORY_PROPOSER_ROLE
        and item.output is not None
        and item.output.supervisory_handoff is not None
    }
    for request in snapshot.sidecar_requests:
        if request.authority == "native_proposer":
            origin = settlements_by_attempt_id.get(
                str(request.native_attempt_id)
            )
            if (
                origin is None
                or origin.result_digest
                != request.origin_admission_receipt_sha256
            ):
                raise SchedulerError(
                    "native sidecar request lacks its exact clean settlement"
                )
        else:
            origin = admitted_supervisors.get(
                str(request.supervisory_assignment_id)
            )
            if (
                origin is None
                or origin.admission_receipt_sha256
                != request.origin_admission_receipt_sha256
                or sha256_json(
                    asdict(origin.output.supervisory_handoff)
                )
                != request.supervisory_handoff_sha256
            ):
                raise SchedulerError(
                    "supervisory sidecar request lacks exact admitted origin"
                )
    active_auxiliary = len(_active_auxiliary_assignments(snapshot))
    if len(active_games) + active_auxiliary > snapshot.max_lanes:
        raise SchedulerError(
            "proposer plus auxiliary occupancy exceeds capacity"
        )
    validate_budget_state(snapshot.budget)
    return snapshot


def _frontier_score(frontier: Frontier) -> tuple[int, int]:
    policy = retry_policy(frontier.no_progress)
    evidence = (
        frontier.evidence
        if (
            policy.requested_wip_mode == "restore_clean_same_frontier"
            and frontier.wip is not None
        )
        else selection_evidence(
            parent_source_path=frontier.parent_source_path,
            parent_source_tree_sha256=frontier.parent_source_tree_sha256,
        )
    )
    probability = SUCCESS_SCALE // (frontier.no_progress + 2)
    free_energy = (
        -probability
        + FREE_ENERGY_COMPLEXITY_WEIGHT
        * evidence.conditional_novelty
    )
    return probability, free_energy


def eligible_frontiers(snapshot: CampaignSnapshot) -> tuple[Frontier, ...]:
    validate_snapshot(snapshot)
    return tuple(
        frontier
        for frontier in snapshot.frontiers
        if (
            frontier.reached < frontier.target
            and frontier.active_attempt_id is None
            and not frontier.draining
            and frontier.blocked_reason is None
        )
    )


def _selected_supervisory_handoff(
    snapshot: CampaignSnapshot,
    frontier: Frontier,
    *,
    required_handoff_sha256: str | None = None,
    include_consumed: bool = False,
) -> SupervisoryHandoffBinding | None:
    if (
        required_handoff_sha256 is not None
        and not _is_sha256(required_handoff_sha256)
    ):
        raise SchedulerError(
            "required supervisory handoff identity is malformed"
        )
    consumed = {
        item.supervisory_handoff_sha256
        for item in snapshot.clean_proposer_settlements
        if item.game == frontier.game
        and item.frontier_sha256 == frontier.frontier_sha256
        and item.parent_checkpoint_sha256
        == frontier.parent_checkpoint_sha256
        and item.supervisory_handoff_sha256 is not None
    }
    candidates = tuple(
        item
        for item in snapshot.auxiliary_assignments
        if item.game == frontier.game
        and item.frontier_sha256 == frontier.frontier_sha256
        and item.parent_checkpoint_sha256
        == frontier.parent_checkpoint_sha256
        and item.specialization == SUPERVISORY_SPECIALIZATION
        and item.phase == "ADMITTED"
        and not item.invalidated
        and item.output is not None
        and item.output.supervisory_handoff is not None
        and (
            required_handoff_sha256 is None
            or supervisory_handoff_sha256(
                item.output.supervisory_handoff
            )
            == required_handoff_sha256
        )
        and (
            include_consumed
            or supervisory_handoff_sha256(
                item.output.supervisory_handoff
            )
            not in consumed
        )
    )
    if not candidates:
        return None
    selected = max(
        candidates,
        key=lambda item: (
            int(item.admitted_sequence or 0),
            item.assignment_id,
        ),
    )
    if any(
        value is None
        for value in (
            selected.admission_receipt_path,
            selected.admission_receipt_sha256,
            selected.admitted_sequence,
            selected.admitted_event_digest,
        )
    ):
        raise SchedulerError(
            "admitted supervisory handoff lacks durable admission identity"
        )
    output = selected.output
    assert output is not None
    handoff = output.supervisory_handoff
    assert handoff is not None
    return validate_supervisory_handoff_binding(
        SupervisoryHandoffBinding(
            schema=1,
            assignment_id=selected.assignment_id,
            frontier_sha256=frontier.frontier_sha256,
            parent_checkpoint_sha256=(
                frontier.parent_checkpoint_sha256
            ),
            output_manifest_sha256=output.output_manifest_sha256,
            supervisory_handoff_sha256=(
                supervisory_handoff_sha256(handoff)
            ),
            output=output,
            admission_receipt_path=str(
                selected.admission_receipt_path
            ),
            admission_receipt_sha256=str(
                selected.admission_receipt_sha256
            ),
            admitted_sequence=int(selected.admitted_sequence),
            admitted_event_digest=str(
                selected.admitted_event_digest
            ),
            prompt_authority="unverified_hypothesis_only",
            derived_evidence_requires_native_reproduction=True,
            scheduler_authority=False,
            mutation_authority=False,
            promotion_authority=False,
        )
    )


def _selected_native_context(
    snapshot: CampaignSnapshot,
    frontier: Frontier,
    policy: RetryPolicy,
) -> tuple[
    WipBinding | None,
    SupervisoryHandoffBinding | None,
]:
    """Select one provenance-compatible native context and tactical handoff."""

    fresh_handoff = _selected_supervisory_handoff(
        snapshot, frontier
    )
    selected_wip = (
        frontier.wip
        if (
            policy.requested_wip_mode
            == "restore_clean_same_frontier"
            and frontier.wip is not None
        )
        else None
    )
    if (
        selected_wip is None
        or selected_wip.supervisory_handoff_sha256 is None
    ):
        return selected_wip, fresh_handoff
    if (
        selected_wip
        .supervisory_native_reproduction_receipt_sha256
        is None
    ):
        # An exposed WIP without its native reproduction provenance can never
        # re-enter a clean proposer context.
        return None, fresh_handoff
    matching_handoff = _selected_supervisory_handoff(
        snapshot,
        frontier,
        required_handoff_sha256=(
            selected_wip.supervisory_handoff_sha256
        ),
        include_consumed=True,
    )
    if matching_handoff is None:
        return None, fresh_handoff
    if (
        fresh_handoff is not None
        and fresh_handoff.supervisory_handoff_sha256
        != matching_handoff.supervisory_handoff_sha256
    ):
        # A newer unconsumed handoff starts a fresh context.  It cannot be
        # silently combined with WIP already exposed to another hypothesis.
        return None, fresh_handoff
    return selected_wip, matching_handoff


def choose_frontier(snapshot: CampaignSnapshot) -> DispatchChoice | None:
    """Select one distinct game; capacity is a ceiling, never a quota."""

    validate_snapshot(snapshot)
    active_count = sum(
        frontier.active_attempt_id is not None
        for frontier in snapshot.frontiers
    ) + len(_active_auxiliary_assignments(snapshot))
    capacity = snapshot.max_lanes - active_count
    if capacity <= 0:
        return None
    candidates = eligible_frontiers(snapshot)
    if not candidates:
        return None
    ranked: list[
        tuple[
            tuple[int, int, int, str],
            Frontier,
            RetryPolicy,
            SelectionEvidence,
            int,
            WipBinding | None,
            SupervisoryHandoffBinding | None,
        ]
    ] = []
    for frontier in candidates:
        policy = retry_policy(frontier.no_progress)
        selected_wip, selected_handoff = _selected_native_context(
            snapshot, frontier, policy
        )
        effective_evidence = (
            frontier.evidence
            if selected_wip is not None
            else selection_evidence(
                parent_source_path=frontier.parent_source_path,
                parent_source_tree_sha256=(
                    frontier.parent_source_tree_sha256
                ),
            )
        )
        probability = SUCCESS_SCALE // (frontier.no_progress + 2)
        free_energy = (
            -probability
            + FREE_ENERGY_COMPLEXITY_WEIGHT
            * effective_evidence.conditional_novelty
        )
        key = (
            frontier.last_dispatch_sequence,
            free_energy,
            -len(effective_evidence.reused_definition_calls),
            frontier.game,
        )
        ranked.append(
            (
                key,
                frontier,
                policy,
                effective_evidence,
                probability,
                selected_wip,
                selected_handoff,
            )
        )
    ranked.sort(key=lambda item: item[0])
    (
        key,
        frontier,
        policy,
        effective_evidence,
        probability,
        selected_wip,
        selected_handoff,
    ) = ranked[0]
    effective_wip_mode: Literal[
        "exclude", "restore_clean_same_frontier"
    ] = (
        "restore_clean_same_frontier"
        if selected_wip is not None
        else "exclude"
    )
    slots = min(capacity, len(candidates))
    allowance = reservation_allowance(
        snapshot.budget, slots_to_fill=slots
    )
    if snapshot.budget.limit_units is not None and allowance == 0:
        return None
    return DispatchChoice(
        game=frontier.game,
        target_level=frontier.reached + 1,
        authoritative_target=frontier.target,
        no_progress=frontier.no_progress,
        effort=policy.effort,
        soft_allocation_seconds=policy.soft_allocation_seconds,
        requested_wip_mode=policy.requested_wip_mode,
        effective_wip_mode=effective_wip_mode,
        thread_mode="resume" if selected_wip is not None else "new",
        selected_wip=selected_wip,
        success_prior_micro=probability,
        conditional_novelty=effective_evidence.conditional_novelty,
        estimated_free_energy_micro=key[1],
        reused_definition_calls=effective_evidence.reused_definition_calls,
        ranking_key=key,
        slots_to_fill=slots,
        reservation_units=allowance,
        selected_supervisory_handoff=selected_handoff,
    )


def _frontier_projection(frontier: Frontier) -> dict[str, object]:
    value = asdict(frontier)
    return value


def _eligible_projection(snapshot: CampaignSnapshot) -> list[dict[str, object]]:
    return [
        _frontier_projection(frontier)
        for frontier in eligible_frontiers(snapshot)
    ]


@dataclass(frozen=True)
class SchedulerDecision:
    schema: Literal[1]
    policy_name: str
    policy_sha256: str
    proposer_policy_sha256: str
    decision_id: str
    campaign_id: str
    attempt_id: str
    generation_id: str
    reservation_id: str
    journal_head_sequence: int
    journal_head_digest: str
    inventory_sha256: str
    eligible_frontiers: tuple[dict[str, object], ...]
    eligible_frontiers_sha256: str
    active_attempt_ids: tuple[str, ...]
    active_auxiliary_assignment_ids: tuple[str, ...]
    max_lanes: int
    cost_window_id: str
    limit_units: int | None
    settled_units: int
    live_reservation_units: int
    choice: DispatchChoice
    decision_sha256: str


def _decision_body(
    *,
    snapshot: CampaignSnapshot,
    choice: DispatchChoice,
    decision_id: str,
    attempt_id: str,
    generation_id: str,
    reservation_id: str,
) -> dict[str, object]:
    eligible = _eligible_projection(snapshot)
    active = tuple(
        sorted(
            frontier.active_attempt_id
            for frontier in snapshot.frontiers
            if frontier.active_attempt_id is not None
        )
    )
    active_auxiliary = tuple(
        sorted(
            assignment.assignment_id
            for assignment in _active_auxiliary_assignments(snapshot)
        )
    )
    return {
        "schema": 1,
        "policy_name": POLICY_NAME,
        "policy_sha256": SCHEDULER_POLICY_SHA256,
        "proposer_policy_sha256": PROPOSER_POLICY_SHA256,
        "decision_id": decision_id,
        "campaign_id": snapshot.campaign_id,
        "attempt_id": attempt_id,
        "generation_id": generation_id,
        "reservation_id": reservation_id,
        "journal_head_sequence": snapshot.journal_head_sequence,
        "journal_head_digest": snapshot.journal_head_digest,
        "inventory_sha256": inventory_sha256(dict(snapshot.inventory)),
        "eligible_frontiers": eligible,
        "eligible_frontiers_sha256": sha256_json(eligible),
        "active_attempt_ids": list(active),
        "active_auxiliary_assignment_ids": list(active_auxiliary),
        "max_lanes": snapshot.max_lanes,
        "cost_window_id": snapshot.budget.cost_window_id,
        "limit_units": snapshot.budget.limit_units,
        "settled_units": snapshot.budget.settled_units,
        "live_reservation_units": sum(
            item.units for item in snapshot.budget.live_reservations
        ),
        "choice": asdict(choice),
    }


def build_decision(
    snapshot: CampaignSnapshot,
    *,
    decision_id: str,
    attempt_id: str,
    generation_id: str,
    reservation_id: str,
) -> SchedulerDecision | None:
    """Bind a recomputable dispatch to identities before any external effect."""

    for label, value in (
        ("decision_id", decision_id),
        ("attempt_id", attempt_id),
        ("generation_id", generation_id),
        ("reservation_id", reservation_id),
    ):
        _require_identifier(value, label)
    if len({decision_id, attempt_id, generation_id, reservation_id}) != 4:
        raise SchedulerError("scheduler identities must be distinct")
    choice = choose_frontier(snapshot)
    if choice is None:
        return None
    body = _decision_body(
        snapshot=snapshot,
        choice=choice,
        decision_id=decision_id,
        attempt_id=attempt_id,
        generation_id=generation_id,
        reservation_id=reservation_id,
    )
    return SchedulerDecision(
        schema=1,
        policy_name=POLICY_NAME,
        policy_sha256=SCHEDULER_POLICY_SHA256,
        proposer_policy_sha256=PROPOSER_POLICY_SHA256,
        decision_id=decision_id,
        campaign_id=snapshot.campaign_id,
        attempt_id=attempt_id,
        generation_id=generation_id,
        reservation_id=reservation_id,
        journal_head_sequence=snapshot.journal_head_sequence,
        journal_head_digest=snapshot.journal_head_digest,
        inventory_sha256=str(body["inventory_sha256"]),
        eligible_frontiers=tuple(body["eligible_frontiers"]),  # type: ignore[arg-type]
        eligible_frontiers_sha256=str(
            body["eligible_frontiers_sha256"]
        ),
        active_attempt_ids=tuple(body["active_attempt_ids"]),  # type: ignore[arg-type]
        active_auxiliary_assignment_ids=tuple(
            body["active_auxiliary_assignment_ids"]  # type: ignore[arg-type]
        ),
        max_lanes=snapshot.max_lanes,
        cost_window_id=snapshot.budget.cost_window_id,
        limit_units=snapshot.budget.limit_units,
        settled_units=snapshot.budget.settled_units,
        live_reservation_units=int(body["live_reservation_units"]),
        choice=choice,
        decision_sha256=sha256_json(body),
    )


def decision_to_dict(decision: SchedulerDecision) -> dict[str, object]:
    value = asdict(decision)
    return value


def reservation_binding(decision: SchedulerDecision) -> dict[str, object]:
    """Fields that the immediately following reservation must echo exactly."""

    return {
        "scheduler_decision_id": decision.decision_id,
        "scheduler_decision_sha256": decision.decision_sha256,
        "scheduler_policy_sha256": decision.policy_sha256,
        "budget_reservation_id": decision.reservation_id,
        "budget_reservation_units": decision.choice.reservation_units,
        "cost_window_id": decision.cost_window_id,
        "attempt_id": decision.attempt_id,
        "generation_id": decision.generation_id,
    }


def verify_decision(
    snapshot: CampaignSnapshot, decision: SchedulerDecision
) -> None:
    rebuilt = build_decision(
        snapshot,
        decision_id=decision.decision_id,
        attempt_id=decision.attempt_id,
        generation_id=decision.generation_id,
        reservation_id=decision.reservation_id,
    )
    if rebuilt is None or rebuilt != decision:
        raise SchedulerError("scheduler decision is stale, mutated, or forged")


def _exact_clean_history(
    snapshot: CampaignSnapshot,
    frontier: Frontier,
) -> tuple[CleanProposerSettlement, ...]:
    rows = tuple(
        sorted(
            (
                validate_clean_proposer_settlement(item)
                for item in snapshot.clean_proposer_settlements
                if item.game == frontier.game
                and item.frontier_sha256 == frontier.frontier_sha256
            ),
            key=lambda item: item.no_progress_before,
        )
    )
    if (
        len(rows) != frontier.no_progress
        or [item.no_progress_before for item in rows]
        != list(range(frontier.no_progress))
        or any(
            item.parent_checkpoint_sha256
            != frontier.parent_checkpoint_sha256
            for item in rows
        )
    ):
        raise SchedulerError(
            "auxiliary trigger is not derived from the exact clean "
            "same-frontier proposer history"
        )
    return rows


def clean_history_sha256(
    rows: Sequence[CleanProposerSettlement],
) -> str:
    normalized = [
        asdict(validate_clean_proposer_settlement(item))
        for item in rows
    ]
    return sha256_json(normalized)


def planned_auxiliary_input_manifest(
    *,
    frontier: Frontier,
    sidecar_request: SidecarRequestEvidence,
    observation_ledger_sha256: str,
    profile_id: str | None,
    round_index: int,
    specialization: Specialization,
    input_bundle_contract_sha256: str,
) -> AuxiliaryInputManifestCommitment:
    """Commit to the only private input bundle a reservation may materialize."""

    validate_sidecar_request(sidecar_request)
    if (
        sidecar_request.game != frontier.game
        or sidecar_request.frontier_sha256
        != frontier.frontier_sha256
        or sidecar_request.parent_checkpoint_sha256
        != frontier.parent_checkpoint_sha256
        or not set(
            sidecar_request
            .cited_public_observation_receipt_sha256s
        ).issubset(
            frontier.public_observation_receipt_sha256s
        )
    ):
        raise SchedulerError(
            "sidecar request targets a stale/cross-frontier input"
        )
    _require_sha256(
        observation_ledger_sha256, "auxiliary observation ledger"
    )
    _require_sha256(
        input_bundle_contract_sha256, "auxiliary input bundle contract"
    )
    manifest = AuxiliaryInputManifestCommitment(
        schema=1,
        kind="planned_auxiliary_private_input",
        game=frontier.game,
        frontier_sha256=frontier.frontier_sha256,
        parent_checkpoint_sha256=frontier.parent_checkpoint_sha256,
        parent_source_tree_sha256=frontier.parent_source_tree_sha256,
        wip_snapshot_id=(
            frontier.wip.snapshot_id if frontier.wip is not None else None
        ),
        wip_tree_sha256=(
            frontier.wip.wip_tree_sha256 if frontier.wip is not None else None
        ),
        wip_solver_source_tree_sha256=(
            frontier.wip.solver_source_tree_sha256
            if frontier.wip is not None
            else None
        ),
        observation_ledger_sha256=observation_ledger_sha256,
        profile_id=profile_id,
        round_index=round_index,
        specialization=specialization,
        input_bundle_contract_sha256=input_bundle_contract_sha256,
        immutable_inputs=True,
        live_lineage_mounted=False,
        public_observations_only=True,
        input_role="side_expert",
        allowed_input_classes=SIDE_EXPERT_ALLOWED_INPUT_CLASSES,
        forbidden_input_classes=SUPERVISORY_FORBIDDEN_INPUT_CLASSES,
        input_allowlist_sha256=auxiliary_input_allowlist_sha256(
            role="side_expert"
        ),
        authenticated_public_observation_receipt_sha256s=tuple(
            sorted(frontier.public_observation_receipt_sha256s)
        ),
        native_solver_source_tree_sha256s=(
            ()
            if frontier.wip is None
            else (frontier.wip.solver_source_tree_sha256,)
        ),
        authenticated_side_expert_evidence_sha256s=(),
        authenticated_evidence_set_sha256=sha256_json(
            {
                "public_observation_receipts": sorted(
                    frontier.public_observation_receipt_sha256s
                ),
                "native_solver_source_trees": (
                    []
                    if frontier.wip is None
                    else [frontier.wip.solver_source_tree_sha256]
                ),
                "side_expert": [],
            }
        ),
        sidecar_request=sidecar_request,
        sidecar_request_sha256=sidecar_request.request_sha256,
        sealed_input_required=True,
        symlinks_allowed=False,
        hardlinks_allowed=False,
        path_escapes_allowed=False,
    )
    return validate_auxiliary_input_manifest(manifest)


def planned_supervisory_proposer_input_manifest(
    *,
    frontier: Frontier,
    sidecar_request: SidecarRequestEvidence,
    observation_ledger_sha256: str,
    profile_id: str,
    round_index: int,
    input_bundle_contract_sha256: str,
    authenticated_public_observation_receipt_sha256s: Sequence[str],
    native_solver_source_tree_sha256s: Sequence[str],
    authenticated_side_expert_evidence_sha256s: Sequence[str],
) -> AuxiliaryInputManifestCommitment:
    """Commit the exact sealed evidence set for a tactical LLM synthesis."""

    public = tuple(
        sorted(authenticated_public_observation_receipt_sha256s)
    )
    solver = tuple(sorted(native_solver_source_tree_sha256s))
    side = tuple(sorted(authenticated_side_expert_evidence_sha256s))
    manifest = planned_auxiliary_input_manifest(
        frontier=frontier,
        sidecar_request=sidecar_request,
        observation_ledger_sha256=observation_ledger_sha256,
        profile_id=profile_id,
        round_index=round_index,
        specialization=SUPERVISORY_SPECIALIZATION,
        input_bundle_contract_sha256=input_bundle_contract_sha256,
    )
    return validate_auxiliary_input_manifest(
        replace(
            manifest,
            input_role=SUPERVISORY_PROPOSER_ROLE,
            allowed_input_classes=SUPERVISORY_ALLOWED_INPUT_CLASSES,
            input_allowlist_sha256=auxiliary_input_allowlist_sha256(
                role=SUPERVISORY_PROPOSER_ROLE
            ),
            authenticated_public_observation_receipt_sha256s=public,
            native_solver_source_tree_sha256s=solver,
            authenticated_side_expert_evidence_sha256s=side,
            authenticated_evidence_set_sha256=sha256_json(
                {
                    "public_observation_receipts": list(public),
                    "native_solver_source_trees": list(solver),
                    "side_expert": list(side),
                }
            ),
        )
    )


def _current_complexity_round(
    snapshot: CampaignSnapshot,
    frontier: Frontier,
) -> ComplexityRoundState | None:
    rows = [
        validate_complexity_round(item)
        for item in snapshot.complexity_rounds
        if item.game == frontier.game
        and item.frontier_sha256 == frontier.frontier_sha256
        and not item.invalidated
    ]
    if not rows:
        return None
    return max(rows, key=lambda item: item.round_index)


def _supervisory_authenticated_evidence(
    *,
    frontier: Frontier,
    assignments: Sequence[AuxiliaryAssignmentState],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    public_observations = tuple(
        sorted(set(frontier.public_observation_receipt_sha256s))
    )
    solver_sources = tuple(
        sorted(
            (
                (frontier.wip.solver_source_tree_sha256,)
                if frontier.wip is not None
                else ()
            )
        )
    )
    side = tuple(
        sorted(
            auxiliary_output_semantic_evidence_sha256(item.output)
            for item in assignments
            if item.game == frontier.game
            and item.frontier_sha256 == frontier.frontier_sha256
            and item.parent_checkpoint_sha256
            == frontier.parent_checkpoint_sha256
            if item.specialization != SUPERVISORY_SPECIALIZATION
            and item.phase == "ADMITTED"
            and not item.invalidated
            and item.output is not None
        )
    )
    return public_observations, solver_sources, side


def _has_complete_post_handoff_reset_continuation_pair(
    history: Sequence[CleanProposerSettlement],
    *,
    prior_no_progress: int,
) -> bool:
    later = tuple(
        sorted(
            (
                item
                for item in history
                if item.no_progress_before >= prior_no_progress
            ),
            key=lambda item: item.no_progress_before,
        )
    )
    return any(
        right.no_progress_before == left.no_progress_before + 1
        and {
            left.requested_wip_mode,
            right.requested_wip_mode,
        }
        == {"exclude", "restore_clean_same_frontier"}
        for left, right in zip(later, later[1:])
    )


def _supervisory_round_eligible(
    *,
    frontier: Frontier,
    history: Sequence[CleanProposerSettlement],
    assignments: Sequence[AuxiliaryAssignmentState],
) -> bool:
    supervisory = tuple(
        item
        for item in assignments
        if item.specialization == SUPERVISORY_SPECIALIZATION
    )
    if any(
        item.phase in AUXILIARY_ACTIVE_PHASES for item in supervisory
    ):
        return False
    public, solver, side = _supervisory_authenticated_evidence(
        frontier=frontier, assignments=assignments
    )
    if not public and not solver and not side:
        return False
    prior = tuple(
        item
        for item in supervisory
        if item.phase in {"QUARANTINED", "ADMITTED", "REJECTED"}
    )
    if not prior:
        return True
    latest = max(
        prior,
        key=lambda item: (
            item.trigger_no_progress,
            item.assignment_id,
        ),
    )
    prior_public = set(
        latest.input_manifest
        .authenticated_public_observation_receipt_sha256s
    )
    prior_solver = set(
        latest.input_manifest.native_solver_source_tree_sha256s
    )
    prior_side = set(
        latest.input_manifest
        .authenticated_side_expert_evidence_sha256s
    )
    admitted_evidence_epoch_changed = (
        set(public) != prior_public
        or set(solver) != prior_solver
        or set(side) != prior_side
    )
    completed_pair = (
        _has_complete_post_handoff_reset_continuation_pair(
            history,
            prior_no_progress=latest.trigger_no_progress,
        )
    )
    return admitted_evidence_epoch_changed or completed_pair


def _eligible_sidecar_requests(
    snapshot: CampaignSnapshot,
    frontier: Frontier,
) -> tuple[SidecarRequestEvidence, ...]:
    """Return only exact-frontier, unconsumed proposer-authored briefs."""

    consumed = {
        item.sidecar_request_sha256
        for item in snapshot.auxiliary_assignments
    }
    settlements = {
        item.attempt_id: item
        for item in snapshot.clean_proposer_settlements
        if item.game == frontier.game
        and item.frontier_sha256 == frontier.frontier_sha256
        and item.parent_checkpoint_sha256
        == frontier.parent_checkpoint_sha256
    }
    admitted_supervisors = {
        item.assignment_id: item
        for item in snapshot.auxiliary_assignments
        if item.phase == "ADMITTED"
        and item.game == frontier.game
        and item.frontier_sha256 == frontier.frontier_sha256
        and item.parent_checkpoint_sha256
        == frontier.parent_checkpoint_sha256
        and item.output is not None
        and item.output.supervisory_handoff is not None
    }
    eligible: list[SidecarRequestEvidence] = []
    for raw in snapshot.sidecar_requests:
        request = validate_sidecar_request(raw)
        if (
            request.request_sha256 in consumed
            or request.game != frontier.game
            or request.frontier_sha256 != frontier.frontier_sha256
            or request.parent_checkpoint_sha256
            != frontier.parent_checkpoint_sha256
            or not set(
                request
                .cited_public_observation_receipt_sha256s
            ).issubset(
                frontier.public_observation_receipt_sha256s
            )
        ):
            continue
        if request.authority == "native_proposer":
            settlement = settlements.get(
                str(request.native_attempt_id)
            )
            if (
                settlement is None
                or settlement.result_digest
                != request.origin_admission_receipt_sha256
            ):
                continue
        else:
            assignment = admitted_supervisors.get(
                str(request.supervisory_assignment_id)
            )
            if (
                assignment is None
                or assignment.admission_receipt_sha256
                != request.origin_admission_receipt_sha256
                or sha256_json(
                    asdict(
                        assignment.output.supervisory_handoff
                    )
                )
                != request.supervisory_handoff_sha256
            ):
                continue
        eligible.append(request)
    return tuple(
        sorted(
            eligible,
            key=lambda item: (
                item.request_id,
                item.request_sha256,
            ),
        )
    )


def _auxiliary_assignment_plan(
    snapshot: CampaignSnapshot,
    frontier: Frontier,
    *,
    supervisory_enabled: bool = True,
    supervisory_capacity_available: bool = True,
) -> tuple[
    Specialization,
    ComplexityRoundState | None,
    int,
    tuple[CleanProposerSettlement, ...],
] | None:
    if (
        frontier.reached >= frontier.target
        or frontier.active_attempt_id is None
        or frontier.draining
        or frontier.blocked_reason is not None
        or frontier.no_progress
        < AUXILIARY_ANALYSIS_START_NO_PROGRESS
        or retry_policy(frontier.no_progress).effort != "max"
    ):
        return None
    history = _exact_clean_history(snapshot, frontier)
    if not _eligible_sidecar_requests(snapshot, frontier):
        return None
    current_round = _current_complexity_round(snapshot, frontier)
    assignments = tuple(
        item
        for item in snapshot.auxiliary_assignments
        if item.game == frontier.game
        and item.frontier_sha256 == frontier.frontier_sha256
        and not item.invalidated
    )
    if current_round is None:
        if any(
            item.specialization == "complexity_diagnosis"
            and item.phase not in {"ABORTED", "REJECTED"}
            for item in assignments
        ):
            return None
        return (
            "complexity_diagnosis",
            None,
            0,
            history,
        )
    profile_assignments = tuple(
        item
        for item in assignments
        if item.profile_id == current_round.profile.profile_id
    )
    active = tuple(
        item.specialization
        for item in profile_assignments
        if item.specialization != SUPERVISORY_SPECIALIZATION
        if item.phase in AUXILIARY_ACTIVE_PHASES
    )
    completed = tuple(
        item.specialization
        for item in profile_assignments
        if item.specialization != SUPERVISORY_SPECIALIZATION
        if item.phase in {"QUARANTINED", "ADMITTED", "REJECTED"}
    )
    if (
        supervisory_enabled
        and supervisory_capacity_available
        and _supervisory_round_eligible(
            frontier=frontier,
            history=history,
            assignments=assignments,
        )
    ):
        return (
            SUPERVISORY_SPECIALIZATION,
            current_round,
            current_round.round_index,
            history,
        )
    policy = auxiliary_analysis_policy(
        frontier.no_progress,
        frontier_sha256=frontier.frontier_sha256,
        profile=current_round.profile,
        active_specializations=active,
        completed_specializations=completed,
    )
    if policy.specializations:
        return (
            policy.specializations[0],
            current_round,
            current_round.round_index,
            history,
        )
    if (
        set(completed) == set(current_round.profile.priorities)
        and not active
        and frontier.no_progress
        >= current_round.trigger_no_progress + 2
    ):
        next_round_index = current_round.round_index + 1
        if any(
            item.specialization == "complexity_diagnosis"
            and item.round_index == next_round_index
            and item.phase not in {"ABORTED", "REJECTED"}
            for item in assignments
        ):
            return None
        return (
            "complexity_diagnosis",
            None,
            next_round_index,
            history,
        )
    return None


def _ordered_auxiliary_candidates(
    snapshot: CampaignSnapshot,
    *,
    supervisory_enabled: bool = True,
    supervisory_max_concurrency: int = 1,
) -> list[
    tuple[
        tuple[int, int, str],
        Frontier,
        Specialization,
        ComplexityRoundState | None,
        int,
        tuple[CleanProposerSettlement, ...],
    ]
]:
    candidates: list[
        tuple[
            tuple[int, int, str],
            Frontier,
            Specialization,
            ComplexityRoundState | None,
            int,
            tuple[CleanProposerSettlement, ...],
        ]
    ] = []
    active_supervisors = sum(
        item.role == SUPERVISORY_PROPOSER_ROLE
        and item.phase in AUXILIARY_ACTIVE_PHASES
        and not item.invalidated
        for item in snapshot.auxiliary_assignments
    )
    supervisory_capacity_available = (
        active_supervisors < supervisory_max_concurrency
    )
    for frontier in snapshot.frontiers:
        plan = _auxiliary_assignment_plan(
            snapshot,
            frontier,
            supervisory_enabled=supervisory_enabled,
            supervisory_capacity_available=(
                supervisory_capacity_available
            ),
        )
        if plan is None:
            continue
        specialization, current_round, round_index, history = plan
        candidates.append(
            (
                (
                    frontier.last_dispatch_sequence,
                    -frontier.no_progress,
                    frontier.game,
                ),
                frontier,
                specialization,
                current_round,
                round_index,
                history,
            )
        )
    candidates.sort(key=lambda item: item[0])
    return candidates


def choose_auxiliary_frontier(
    snapshot: CampaignSnapshot,
    *,
    supervisory_enabled: bool = True,
    supervisory_max_concurrency: int = 1,
) -> Frontier | None:
    """Return the exact frontier a subsequent sidecar decision would use."""

    validate_snapshot(snapshot)
    if choose_frontier(snapshot) is not None:
        return None
    active_total = sum(
        item.active_attempt_id is not None
        for item in snapshot.frontiers
    ) + len(_active_auxiliary_assignments(snapshot))
    if active_total >= snapshot.max_lanes:
        return None
    candidates = _ordered_auxiliary_candidates(
        snapshot,
        supervisory_enabled=supervisory_enabled,
        supervisory_max_concurrency=supervisory_max_concurrency,
    )
    return candidates[0][1] if candidates else None


def _auxiliary_decision_body(
    *,
    snapshot: CampaignSnapshot,
    frontier: Frontier,
    history: Sequence[CleanProposerSettlement],
    current_round: ComplexityRoundState | None,
    round_index: int,
    specialization: Specialization,
    decision_id: str,
    assignment_id: str,
    reservation_id: str,
    expert_id: str,
    thread_id: str,
    sidecar_request: SidecarRequestEvidence,
    input_manifest: AuxiliaryInputManifestCommitment,
    observation_ledger_sha256: str,
    launch_configuration: AuxiliaryLaunchConfiguration,
    reservation_units: int | None,
) -> dict[str, object]:
    active_attempts = tuple(
        sorted(
            item.active_attempt_id
            for item in snapshot.frontiers
            if item.active_attempt_id is not None
        )
    )
    active_auxiliary = tuple(
        sorted(
            item.assignment_id
            for item in _active_auxiliary_assignments(snapshot)
        )
    )
    supervisory_launch = launch_configuration.supervisory_proposer
    is_supervisory = specialization == SUPERVISORY_SPECIALIZATION
    selected_model = (
        supervisory_launch.model
        if is_supervisory
        else launch_configuration.model
    )
    selected_effort = (
        supervisory_launch.reasoning_effort
        if is_supervisory
        else launch_configuration.reasoning_effort
    )
    return {
        "schema": 1,
        "policy_name": POLICY_NAME,
        "policy_sha256": SCHEDULER_POLICY_SHA256,
        "decision_id": decision_id,
        "campaign_id": snapshot.campaign_id,
        "assignment_id": assignment_id,
        "reservation_id": reservation_id,
        "journal_head_sequence": snapshot.journal_head_sequence,
        "journal_head_digest": snapshot.journal_head_digest,
        "game": frontier.game,
        "frontier_sha256": frontier.frontier_sha256,
        "parent_checkpoint_sha256":
            frontier.parent_checkpoint_sha256,
        "parent_source_tree_sha256":
            frontier.parent_source_tree_sha256,
        "no_progress": frontier.no_progress,
        "trigger_history_sha256": clean_history_sha256(history),
        "active_proposer_attempt_id": frontier.active_attempt_id,
        "active_attempt_ids": list(active_attempts),
        "active_auxiliary_assignment_ids": list(active_auxiliary),
        "max_lanes": snapshot.max_lanes,
        "profile_id": (
            current_round.profile.profile_id
            if current_round is not None
            and specialization != "complexity_diagnosis"
            else None
        ),
        "round_index": round_index,
        "specialization": specialization,
        "expert_id": expert_id,
        "thread_id": thread_id,
        "model": selected_model,
        "reasoning_effort": selected_effort,
        "role": (
            SUPERVISORY_PROPOSER_ROLE
            if is_supervisory
            else "side_expert"
        ),
        "context_limit_tokens": (
            supervisory_launch.context_limit_tokens
            if is_supervisory
            else None
        ),
        "role_max_concurrency": (
            supervisory_launch.max_concurrency
            if is_supervisory
            else None
        ),
        "supervisory_launch_configuration": (
            asdict(supervisory_launch)
            if is_supervisory
            else None
        ),
        "supervisory_launch_configuration_sha256": (
            sha256_json(asdict(supervisory_launch))
            if is_supervisory
            else None
        ),
        "input_manifest": asdict(input_manifest),
        "input_manifest_sha256": sha256_json(asdict(input_manifest)),
        "sidecar_request": asdict(sidecar_request),
        "sidecar_request_sha256":
            sidecar_request.request_sha256,
        "observation_ledger_sha256": observation_ledger_sha256,
        "backend_contract_sha256":
            launch_configuration.backend_contract_sha256,
        "input_bundle_contract_sha256":
            launch_configuration.input_bundle_contract_sha256,
        "admission_contract_sha256":
            launch_configuration.admission_contract_sha256,
        "cost_window_id": snapshot.budget.cost_window_id,
        "limit_units": snapshot.budget.limit_units,
        "settled_units": snapshot.budget.settled_units,
        "live_reservation_units": sum(
            item.units for item in snapshot.budget.live_reservations
        ),
        "reservation_units": reservation_units,
    }


def build_auxiliary_decision(
    snapshot: CampaignSnapshot,
    *,
    decision_id: str,
    assignment_id: str,
    reservation_id: str,
    expert_id: str,
    thread_id: str,
    observation_ledger_sha256: str,
    launch_configuration: AuxiliaryLaunchConfiguration,
) -> AuxiliaryDecision | None:
    """Select one sidecar only after all primary reservations are filled."""

    validate_snapshot(snapshot)
    launch_configuration = validate_auxiliary_launch_configuration(
        launch_configuration
    )
    if not launch_configuration.automatic_dispatch_enabled:
        return None
    for label, value in (
        ("decision_id", decision_id),
        ("assignment_id", assignment_id),
        ("reservation_id", reservation_id),
        ("expert_id", expert_id),
    ):
        _require_identifier(value, label)
    if len({decision_id, assignment_id, reservation_id, expert_id}) != 4:
        raise SchedulerError("auxiliary identities must be distinct")
    if not _is_canonical_uuid(thread_id):
        raise SchedulerError("auxiliary thread_id is not a canonical UUID")
    _require_sha256(
        observation_ledger_sha256, "auxiliary observation ledger"
    )
    used_identifiers = {
        item
        for assignment in snapshot.auxiliary_assignments
        for item in (
            assignment.assignment_id,
            assignment.decision_id,
            assignment.reservation_id,
            assignment.expert_id,
        )
    }
    used_identifiers.update(
        item.active_attempt_id
        for item in snapshot.frontiers
        if item.active_attempt_id is not None
    )
    if {decision_id, assignment_id, reservation_id, expert_id} & (
        used_identifiers
    ):
        raise SchedulerError("auxiliary identity was already used")
    if any(
        assignment.thread_id == thread_id
        or assignment.expert_id == expert_id
        for assignment in snapshot.auxiliary_assignments
    ) or any(
        frontier.wip is not None
        and frontier.wip.codex_thread_id == thread_id
        for frontier in snapshot.frontiers
    ):
        raise SchedulerError("auxiliary expert/thread identity is not fresh")
    # A proposer reservation always wins the next free slot.
    if choose_frontier(snapshot) is not None:
        return None
    active_total = sum(
        item.active_attempt_id is not None
        for item in snapshot.frontiers
    ) + len(_active_auxiliary_assignments(snapshot))
    capacity = snapshot.max_lanes - active_total
    if capacity <= 0:
        return None
    candidates = _ordered_auxiliary_candidates(
        snapshot,
        supervisory_enabled=(
            launch_configuration.supervisory_proposer
            .automatic_dispatch_enabled
        ),
        supervisory_max_concurrency=(
            launch_configuration.supervisory_proposer
            .max_concurrency
        ),
    )
    if not candidates:
        return None
    (
        _,
        frontier,
        specialization,
        current_round,
        round_index,
        history,
    ) = candidates[0]
    eligible_requests = _eligible_sidecar_requests(
        snapshot, frontier
    )
    if not eligible_requests:
        raise SchedulerError(
            "auxiliary candidate lacks proposer-authored request"
        )
    sidecar_request = eligible_requests[0]
    allowance = reservation_allowance(
        snapshot.budget, slots_to_fill=capacity
    )
    if snapshot.budget.limit_units is not None and allowance == 0:
        return None
    profile_id = (
        current_round.profile.profile_id
        if current_round is not None
        and specialization != "complexity_diagnosis"
        else None
    )
    input_bundle_contract_sha256 = str(
        launch_configuration.input_bundle_contract_sha256
    )
    if specialization == SUPERVISORY_SPECIALIZATION:
        (
            public_observation_evidence,
            native_solver_sources,
            side_evidence,
        ) = (
            _supervisory_authenticated_evidence(
                frontier=frontier,
                assignments=tuple(
                    item
                    for item in snapshot.auxiliary_assignments
                    if item.game == frontier.game
                    and item.frontier_sha256
                    == frontier.frontier_sha256
                    and not item.invalidated
                ),
            )
        )
        if profile_id is None:
            raise SchedulerError(
                "supervisory proposer lacks an admitted complexity round"
            )
        input_manifest = planned_supervisory_proposer_input_manifest(
            frontier=frontier,
            sidecar_request=sidecar_request,
            observation_ledger_sha256=observation_ledger_sha256,
            profile_id=profile_id,
            round_index=round_index,
            input_bundle_contract_sha256=(
                input_bundle_contract_sha256
            ),
            authenticated_public_observation_receipt_sha256s=(
                public_observation_evidence
            ),
            native_solver_source_tree_sha256s=(
                native_solver_sources
            ),
            authenticated_side_expert_evidence_sha256s=side_evidence,
        )
    else:
        input_manifest = planned_auxiliary_input_manifest(
            frontier=frontier,
            sidecar_request=sidecar_request,
            observation_ledger_sha256=observation_ledger_sha256,
            profile_id=profile_id,
            round_index=round_index,
            specialization=specialization,
            input_bundle_contract_sha256=input_bundle_contract_sha256,
        )
    body = _auxiliary_decision_body(
        snapshot=snapshot,
        frontier=frontier,
        history=history,
        current_round=current_round,
        round_index=round_index,
        specialization=specialization,
        decision_id=decision_id,
        assignment_id=assignment_id,
        reservation_id=reservation_id,
        expert_id=expert_id,
        thread_id=thread_id,
        sidecar_request=sidecar_request,
        input_manifest=input_manifest,
        observation_ledger_sha256=observation_ledger_sha256,
        launch_configuration=launch_configuration,
        reservation_units=allowance,
    )
    return AuxiliaryDecision(
        **{
            **body,
            "active_attempt_ids": tuple(body["active_attempt_ids"]),
            "active_auxiliary_assignment_ids": tuple(
                body["active_auxiliary_assignment_ids"]
            ),
            "input_manifest": input_manifest,
            "sidecar_request": sidecar_request,
        },  # type: ignore[arg-type]
        decision_sha256=sha256_json(body),
    )


def auxiliary_decision_to_dict(
    decision: AuxiliaryDecision,
) -> dict[str, object]:
    return asdict(decision)


def auxiliary_reservation_binding(
    decision: AuxiliaryDecision,
) -> dict[str, object]:
    return {
        "auxiliary_decision_id": decision.decision_id,
        "auxiliary_decision_sha256": decision.decision_sha256,
        "scheduler_policy_sha256": decision.policy_sha256,
        "budget_reservation_id": decision.reservation_id,
        "budget_reservation_units": decision.reservation_units,
        "cost_window_id": decision.cost_window_id,
        "assignment_id": decision.assignment_id,
    }


def auxiliary_reservation_projection(
    decision: AuxiliaryDecision,
) -> dict[str, object]:
    return {
        **auxiliary_reservation_binding(decision),
        "campaign_id": decision.campaign_id,
        "game": decision.game,
        "frontier_sha256": decision.frontier_sha256,
        "parent_checkpoint_sha256":
            decision.parent_checkpoint_sha256,
        "parent_source_tree_sha256":
            decision.parent_source_tree_sha256,
        "no_progress": decision.no_progress,
        "trigger_history_sha256": decision.trigger_history_sha256,
        "active_proposer_attempt_id":
            decision.active_proposer_attempt_id,
        "profile_id": decision.profile_id,
        "round_index": decision.round_index,
        "specialization": decision.specialization,
        "expert_id": decision.expert_id,
        "thread_id": decision.thread_id,
        "model": decision.model,
        "reasoning_effort": decision.reasoning_effort,
        "input_manifest": asdict(decision.input_manifest),
        "input_manifest_sha256": decision.input_manifest_sha256,
        "sidecar_request": asdict(decision.sidecar_request),
        "sidecar_request_sha256":
            decision.sidecar_request_sha256,
        "observation_ledger_sha256":
            decision.observation_ledger_sha256,
        "backend_contract_sha256":
            decision.backend_contract_sha256,
        "input_bundle_contract_sha256":
            decision.input_bundle_contract_sha256,
        "admission_contract_sha256":
            decision.admission_contract_sha256,
    }


def verify_auxiliary_decision(
    snapshot: CampaignSnapshot,
    decision: AuxiliaryDecision,
    *,
    launch_configuration: AuxiliaryLaunchConfiguration,
) -> None:
    rebuilt = build_auxiliary_decision(
        snapshot,
        decision_id=decision.decision_id,
        assignment_id=decision.assignment_id,
        reservation_id=decision.reservation_id,
        expert_id=decision.expert_id,
        thread_id=decision.thread_id,
        observation_ledger_sha256=decision.observation_ledger_sha256,
        launch_configuration=launch_configuration,
    )
    if rebuilt is None or rebuilt != decision:
        raise SchedulerError(
            "auxiliary decision is stale, manually triggered, or forged"
        )


def _strict_keys(
    value: object, expected: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise SchedulerError(f"{label} schema mismatch")
    return value


def _complexity_profile_from_dict(value: object) -> ComplexityProfile:
    raw = _strict_keys(
        value,
        set(ComplexityProfile.__dataclass_fields__),
        "complexity profile",
    )
    priorities = raw.get("priorities")
    if not isinstance(priorities, list):
        raise SchedulerError("complexity profile priorities are not a list")
    try:
        profile = ComplexityProfile(
            **{**raw, "priorities": tuple(priorities)}
        )
    except TypeError as exc:
        raise SchedulerError("complexity profile schema mismatch") from exc
    return validate_complexity_profile(
        profile, frontier_sha256=profile.frontier_sha256
    )


def complexity_profile_from_dict(value: object) -> ComplexityProfile:
    return _complexity_profile_from_dict(value)


def _socratic_challenge_from_dict(
    value: object,
) -> SocraticChallengeEvidence:
    raw = _strict_keys(
        value,
        set(SocraticChallengeEvidence.__dataclass_fields__),
        "Socratic challenge",
    )
    for name in (
        "observation_receipt_sha256s",
        "rejected_conclusions",
        "surviving_conclusions",
    ):
        if not isinstance(raw.get(name), list):
            raise SchedulerError(f"Socratic challenge {name} is not a list")
    try:
        challenge = SocraticChallengeEvidence(
            **{
                **raw,
                "observation_receipt_sha256s": tuple(
                    raw["observation_receipt_sha256s"]
                ),
                "rejected_conclusions": tuple(
                    raw["rejected_conclusions"]
                ),
                "surviving_conclusions": tuple(
                    raw["surviving_conclusions"]
                ),
            }
        )
    except TypeError as exc:
        raise SchedulerError("Socratic challenge schema mismatch") from exc
    return validate_socratic_challenge(challenge)


def native_sidecar_request_draft_from_dict(
    value: object,
) -> NativeSidecarRequestDraft:
    raw = _strict_keys(
        value,
        set(NativeSidecarRequestDraft.__dataclass_fields__),
        "native sidecar request draft",
    )
    citations = raw.get(
        "cited_public_observation_receipt_sha256s"
    )
    if not isinstance(citations, list):
        raise SchedulerError(
            "native sidecar request draft citations are not a list"
        )
    try:
        draft = NativeSidecarRequestDraft(
            **{
                **raw,
                "cited_public_observation_receipt_sha256s":
                    tuple(citations),
            }
        )
    except TypeError as exc:
        raise SchedulerError(
            "native sidecar request draft schema mismatch"
        ) from exc
    return validate_native_sidecar_request_draft(draft)


def sidecar_request_from_dict(
    value: object,
) -> SidecarRequestEvidence:
    raw = _strict_keys(
        value,
        set(SidecarRequestEvidence.__dataclass_fields__),
        "sidecar request",
    )
    citations = raw.get(
        "cited_public_observation_receipt_sha256s"
    )
    if not isinstance(citations, list):
        raise SchedulerError(
            "sidecar request citations are not a list"
        )
    try:
        request = SidecarRequestEvidence(
            **{
                **raw,
                "cited_public_observation_receipt_sha256s":
                    tuple(citations),
            }
        )
    except TypeError as exc:
        raise SchedulerError(
            "sidecar request schema mismatch"
        ) from exc
    return validate_sidecar_request(request)


def _auxiliary_input_manifest_from_dict(
    value: object,
) -> AuxiliaryInputManifestCommitment:
    raw = _strict_keys(
        value,
        set(AuxiliaryInputManifestCommitment.__dataclass_fields__),
        "planned auxiliary input manifest",
    )
    for name in (
        "allowed_input_classes",
        "forbidden_input_classes",
        "authenticated_public_observation_receipt_sha256s",
        "native_solver_source_tree_sha256s",
        "authenticated_side_expert_evidence_sha256s",
    ):
        if not isinstance(raw.get(name), list):
            raise SchedulerError(
                f"planned auxiliary input manifest {name} is not a list"
            )
    try:
        sidecar_request = sidecar_request_from_dict(
            raw["sidecar_request"]
        )
        manifest = AuxiliaryInputManifestCommitment(
            **{
                **raw,
                "allowed_input_classes": tuple(
                    raw["allowed_input_classes"]
                ),
                "forbidden_input_classes": tuple(
                    raw["forbidden_input_classes"]
                ),
                "authenticated_public_observation_receipt_sha256s": tuple(
                    raw[
                        "authenticated_public_observation_receipt_sha256s"
                    ]
                ),
                "native_solver_source_tree_sha256s": tuple(
                    raw["native_solver_source_tree_sha256s"]
                ),
                "authenticated_side_expert_evidence_sha256s": tuple(
                    raw[
                        "authenticated_side_expert_evidence_sha256s"
                    ]
                ),
                "sidecar_request": sidecar_request,
            }
        )
    except TypeError as exc:
        raise SchedulerError(
            "planned auxiliary input manifest schema mismatch"
        ) from exc
    return validate_auxiliary_input_manifest(manifest)


def _supervisory_handoff_from_dict(
    value: object,
    *,
    challenge: SocraticChallengeEvidence,
) -> SupervisoryHandoffEvidence:
    raw = _strict_keys(
        value,
        set(SupervisoryHandoffEvidence.__dataclass_fields__),
        "SUPERVISORY_HANDOFF",
    )
    for name in (
        "relied_on_observation_receipt_sha256s",
        "claims",
        "rejected_alternatives",
    ):
        if not isinstance(raw.get(name), list):
            raise SchedulerError(f"SUPERVISORY_HANDOFF {name} is not a list")
    typed_claims: list[SupervisoryHypothesisEvidence] = []
    for value in raw["claims"]:
        claim = _strict_keys(
            value,
            set(SupervisoryHypothesisEvidence.__dataclass_fields__),
            "SUPERVISORY_HANDOFF claim",
        )
        for name in (
            "observation_receipt_sha256s",
            "falsifiers",
            "bounded_next_tests",
        ):
            if not isinstance(claim.get(name), list):
                raise SchedulerError(
                    f"SUPERVISORY_HANDOFF claim {name} is not a list"
                )
        try:
            typed_claims.append(
                SupervisoryHypothesisEvidence(
                    **{
                        **claim,
                        "observation_receipt_sha256s": tuple(
                            claim["observation_receipt_sha256s"]
                        ),
                        "falsifiers": tuple(claim["falsifiers"]),
                        "bounded_next_tests": tuple(
                            claim["bounded_next_tests"]
                        ),
                    }
                )
            )
        except TypeError as exc:
            raise SchedulerError(
                "SUPERVISORY_HANDOFF claim schema mismatch"
            ) from exc
    try:
        handoff = SupervisoryHandoffEvidence(
            **{
                **raw,
                "relied_on_observation_receipt_sha256s": tuple(
                    raw["relied_on_observation_receipt_sha256s"]
                ),
                "claims": tuple(typed_claims),
                "rejected_alternatives": tuple(
                    raw["rejected_alternatives"]
                ),
            }
        )
    except TypeError as exc:
        raise SchedulerError(
            "SUPERVISORY_HANDOFF schema mismatch"
        ) from exc
    return validate_supervisory_handoff(
        handoff, challenge=challenge
    )


def supervisory_native_reproduction_from_dict(
    value: object,
    *,
    handoff: SupervisoryHandoffEvidence,
) -> SupervisoryNativeReproductionReceipt:
    raw = _strict_keys(
        value,
        set(SupervisoryNativeReproductionReceipt.__dataclass_fields__),
        "supervisory native reproduction",
    )
    rows = raw.get("reproductions")
    if not isinstance(rows, list):
        raise SchedulerError(
            "supervisory native reproductions are not a list"
        )
    typed_rows: list[NativeObservationReproduction] = []
    for row in rows:
        item = _strict_keys(
            row,
            set(NativeObservationReproduction.__dataclass_fields__),
            "native observation reproduction",
        )
        try:
            typed_rows.append(NativeObservationReproduction(**item))
        except TypeError as exc:
            raise SchedulerError(
                "native observation reproduction schema mismatch"
            ) from exc
    try:
        receipt = SupervisoryNativeReproductionReceipt(
            **{**raw, "reproductions": tuple(typed_rows)}
        )
    except TypeError as exc:
        raise SchedulerError(
            "supervisory native reproduction schema mismatch"
        ) from exc
    return validate_supervisory_native_reproduction(
        receipt, handoff=handoff
    )


def auxiliary_output_from_dict(
    value: object,
    *,
    assignment: AuxiliaryAssignmentState | None = None,
) -> AuxiliaryOutputEvidence:
    raw = _strict_keys(
        value,
        set(AuxiliaryOutputEvidence.__dataclass_fields__),
        "auxiliary output",
    )
    for name in (
        "public_observation_receipt_sha256s",
        "quarantined_artifact_sha256s",
    ):
        if not isinstance(raw.get(name), list):
            raise SchedulerError(f"auxiliary output {name} is not a list")
    challenge = _socratic_challenge_from_dict(raw["challenge"])
    handoff = (
        None
        if raw.get("supervisory_handoff") is None
        else _supervisory_handoff_from_dict(
            raw["supervisory_handoff"],
            challenge=challenge,
        )
    )
    try:
        output = AuxiliaryOutputEvidence(
            **{
                **raw,
                "public_observation_receipt_sha256s": tuple(
                    raw["public_observation_receipt_sha256s"]
                ),
                "quarantined_artifact_sha256s": tuple(
                    raw["quarantined_artifact_sha256s"]
                ),
                "challenge": challenge,
                "supervisory_handoff": handoff,
            }
        )
    except TypeError as exc:
        raise SchedulerError("auxiliary output schema mismatch") from exc
    return validate_auxiliary_output(output, assignment=assignment)


def auxiliary_decision_from_dict(value: object) -> AuxiliaryDecision:
    raw = _strict_keys(
        value,
        set(AuxiliaryDecision.__dataclass_fields__),
        "auxiliary decision",
    )
    body = {key: raw[key] for key in raw if key != "decision_sha256"}
    input_manifest = _auxiliary_input_manifest_from_dict(
        raw["input_manifest"]
    )
    sidecar_request = sidecar_request_from_dict(
        raw["sidecar_request"]
    )
    supervisory_launch = (
        None
        if raw["supervisory_launch_configuration"] is None
        else validate_supervisory_proposer_launch_configuration(
            SupervisoryProposerLaunchConfiguration(
                **_strict_keys(
                    raw["supervisory_launch_configuration"],
                    set(
                        SupervisoryProposerLaunchConfiguration
                        .__dataclass_fields__
                    ),
                    "supervisory proposer launch decision pin",
                )
            )
        )
    )
    for name in (
        "active_attempt_ids",
        "active_auxiliary_assignment_ids",
    ):
        if (
            not isinstance(raw.get(name), list)
            or raw[name] != sorted(set(raw[name]))
            or any(not _is_identifier(item) for item in raw[name])
        ):
            raise SchedulerError(f"auxiliary decision {name} is invalid")
    if (
        raw["schema"] != 1
        or raw["policy_name"] != POLICY_NAME
        or raw["policy_sha256"] != SCHEDULER_POLICY_SHA256
        or any(
            not _is_identifier(raw.get(name))
            for name in (
                "decision_id",
                "campaign_id",
                "assignment_id",
                "reservation_id",
                "expert_id",
                "active_proposer_attempt_id",
                "model",
                "cost_window_id",
            )
        )
        or not _is_int(raw["journal_head_sequence"])
        or not _is_sha256(raw["journal_head_digest"])
        or not isinstance(raw["game"], str)
        or GAME_RE.fullmatch(raw["game"]) is None
        or any(
            not _is_sha256(raw.get(name))
            for name in (
                "frontier_sha256",
                "parent_checkpoint_sha256",
                "parent_source_tree_sha256",
                "trigger_history_sha256",
                "input_manifest_sha256",
                "sidecar_request_sha256",
                "observation_ledger_sha256",
                "backend_contract_sha256",
                "input_bundle_contract_sha256",
                "admission_contract_sha256",
                "decision_sha256",
            )
        )
        or not _is_int(
            raw["no_progress"],
            minimum=AUXILIARY_ANALYSIS_START_NO_PROGRESS,
        )
        or raw["active_proposer_attempt_id"]
        not in raw["active_attempt_ids"]
        or not _is_int(raw["max_lanes"], minimum=1)
        or raw["max_lanes"] > MAX_LANES
        or len(raw["active_attempt_ids"])
        + len(raw["active_auxiliary_assignment_ids"])
        >= raw["max_lanes"]
        or (
            raw["profile_id"] is not None
            and not _is_identifier(raw["profile_id"])
        )
        or not _is_int(raw["round_index"])
        or raw["specialization"]
        not in ALL_AUXILIARY_SPECIALIZATIONS
        or (
            raw["specialization"] == "complexity_diagnosis"
            and raw["profile_id"] is not None
        )
        or (
            raw["specialization"] != "complexity_diagnosis"
            and raw["profile_id"] is None
        )
        or not _is_canonical_uuid(raw["thread_id"])
        or raw["reasoning_effort"]
        not in SUPPORTED_AUXILIARY_REASONING_EFFORTS
        or raw["role"]
        not in {"side_expert", SUPERVISORY_PROPOSER_ROLE}
        or (
            raw["specialization"] == SUPERVISORY_SPECIALIZATION
        )
        != (raw["role"] == SUPERVISORY_PROPOSER_ROLE)
        or (
            raw["role"] == SUPERVISORY_PROPOSER_ROLE
            and (
                not _is_int(
                    raw["context_limit_tokens"], minimum=1
                )
                or raw["context_limit_tokens"] > 2_000_000
                or raw["role_max_concurrency"] != 1
                or isinstance(raw["role_max_concurrency"], bool)
                or not _is_sha256(
                    raw[
                        "supervisory_launch_configuration_sha256"
                    ]
                )
                or supervisory_launch is None
                or raw["model"] != supervisory_launch.model
                or raw["reasoning_effort"]
                != supervisory_launch.reasoning_effort
                or raw["context_limit_tokens"]
                != supervisory_launch.context_limit_tokens
                or raw["role_max_concurrency"]
                != supervisory_launch.max_concurrency
                or raw[
                    "supervisory_launch_configuration_sha256"
                ]
                != sha256_json(asdict(supervisory_launch))
                or supervisory_launch.automatic_dispatch_enabled
                is not True
                or input_manifest.input_role
                != SUPERVISORY_PROPOSER_ROLE
            )
        )
        or (
            raw["role"] == "side_expert"
            and any(
                item is not None
                for item in (
                    raw["context_limit_tokens"],
                    raw["role_max_concurrency"],
                    supervisory_launch,
                    raw[
                        "supervisory_launch_configuration_sha256"
                    ],
                )
            )
        )
        or raw["input_manifest_sha256"]
        != sha256_json(asdict(input_manifest))
        or raw["sidecar_request_sha256"]
        != sidecar_request.request_sha256
        or sidecar_request != input_manifest.sidecar_request
        or raw["sidecar_request_sha256"]
        != input_manifest.sidecar_request_sha256
        or raw["observation_ledger_sha256"]
        != input_manifest.observation_ledger_sha256
        or raw["game"] != input_manifest.game
        or raw["frontier_sha256"] != input_manifest.frontier_sha256
        or raw["parent_checkpoint_sha256"]
        != input_manifest.parent_checkpoint_sha256
        or raw["parent_source_tree_sha256"]
        != input_manifest.parent_source_tree_sha256
        or raw["profile_id"] != input_manifest.profile_id
        or raw["round_index"] != input_manifest.round_index
        or raw["specialization"] != input_manifest.specialization
        or raw["input_bundle_contract_sha256"]
        != input_manifest.input_bundle_contract_sha256
        or (
            raw["limit_units"] is not None
            and not _is_int(raw["limit_units"])
        )
        or not _is_int(raw["settled_units"])
        or not _is_int(raw["live_reservation_units"])
        or (
            raw["reservation_units"] is not None
            and not _is_int(raw["reservation_units"], minimum=1)
        )
        or raw["decision_sha256"] != sha256_json(body)
    ):
        raise SchedulerError("auxiliary decision has invalid values")
    try:
        return AuxiliaryDecision(
            **{
                **raw,
                "active_attempt_ids": tuple(raw["active_attempt_ids"]),
                "active_auxiliary_assignment_ids": tuple(
                    raw["active_auxiliary_assignment_ids"]
                ),
                "input_manifest": input_manifest,
                "sidecar_request": sidecar_request,
                "supervisory_launch_configuration":
                    supervisory_launch,
            }
        )
    except TypeError as exc:
        raise SchedulerError("auxiliary decision schema mismatch") from exc


def _evidence_from_dict(value: object) -> SelectionEvidence:
    raw = _strict_keys(
        value,
        {
            "schema",
            "metric",
            "parent_source_path",
            "parent_source_tree_sha256",
            "candidate_source_path",
            "candidate_source_tree_sha256",
            "conditional_novelty",
            "retained_normalized_units",
            "reused_definition_calls",
            "evidence_sha256",
        },
        "selection evidence",
    )
    if (
        raw["schema"] != 1
        or raw["metric"]
        not in {SELECTION_METRIC, UNKNOWN_SELECTION_METRIC}
        or not isinstance(raw["parent_source_path"], str)
        or not _is_sha256(raw["parent_source_tree_sha256"])
        or (
            raw["candidate_source_path"] is not None
            and not isinstance(raw["candidate_source_path"], str)
        )
        or (
            raw["candidate_source_tree_sha256"] is not None
            and not _is_sha256(raw["candidate_source_tree_sha256"])
        )
        or not _is_int(raw["conditional_novelty"])
        or not _is_int(raw["retained_normalized_units"])
        or not isinstance(raw["reused_definition_calls"], list)
        or any(
            not isinstance(item, str)
            for item in raw["reused_definition_calls"]
        )
        or raw["reused_definition_calls"]
        != sorted(set(raw["reused_definition_calls"]))
        or not _is_sha256(raw["evidence_sha256"])
    ):
        raise SchedulerError("selection evidence has invalid values")
    return SelectionEvidence(
        schema=1,
        metric=raw["metric"],
        parent_source_path=raw["parent_source_path"],
        parent_source_tree_sha256=raw[
            "parent_source_tree_sha256"
        ],
        candidate_source_path=raw["candidate_source_path"],
        candidate_source_tree_sha256=raw[
            "candidate_source_tree_sha256"
        ],
        conditional_novelty=raw["conditional_novelty"],
        retained_normalized_units=raw["retained_normalized_units"],
        reused_definition_calls=tuple(raw["reused_definition_calls"]),
        evidence_sha256=raw["evidence_sha256"],
    )


def _wip_from_dict(value: object) -> WipBinding:
    raw = _strict_keys(
        value,
        {
            "snapshot_id",
            "wip_root_path",
            "wip_tree_sha256",
            "solver_source_path",
            "solver_source_tree_sha256",
            "game",
            "target_level",
            "parent_checkpoint_sha256",
            "frontier_sha256",
            "codex_thread_id",
            "final_thread_binding_path",
            "final_thread_binding_sha256",
            "wip_export_receipt_path",
            "wip_export_receipt_sha256",
            "final_transcript_chain_receipt_path",
            "final_transcript_chain_receipt_sha256",
            "transcript_chain_sha256",
            "controller_state_scan_receipt_path",
            "controller_state_scan_receipt_sha256",
            "retained_canary_scan_receipt_path",
            "retained_canary_scan_receipt_sha256",
            "taint_scan_receipt_path",
            "taint_scan_receipt_sha256",
            "token_usage_receipt_path",
            "token_usage_receipt_sha256",
            "provider_usage_receipt_path",
            "provider_usage_receipt_sha256",
            "app_server_state_dir",
            "app_server_state_tree_sha256",
            "wip_publication_receipt_path",
            "wip_publication_receipt_sha256",
            "supervisory_handoff_sha256",
            "supervisory_native_reproduction_receipt_path",
            "supervisory_native_reproduction_receipt_sha256",
            "taint_verdict",
        },
        "WIP binding",
    )
    if (
        not _is_identifier(raw["snapshot_id"])
        or any(
            not isinstance(raw[name], str)
            or not Path(raw[name]).is_absolute()
            for name in (
                "wip_root_path",
                "solver_source_path",
                "final_thread_binding_path",
                "wip_export_receipt_path",
                "final_transcript_chain_receipt_path",
                "controller_state_scan_receipt_path",
                "retained_canary_scan_receipt_path",
                "taint_scan_receipt_path",
                "token_usage_receipt_path",
                "provider_usage_receipt_path",
                "app_server_state_dir",
                "wip_publication_receipt_path",
            )
        )
        or any(
            not _is_sha256(raw[name])
            for name in (
                "wip_tree_sha256",
                "solver_source_tree_sha256",
                "parent_checkpoint_sha256",
                "frontier_sha256",
                "final_thread_binding_sha256",
                "wip_export_receipt_sha256",
                "final_transcript_chain_receipt_sha256",
                "transcript_chain_sha256",
                "controller_state_scan_receipt_sha256",
                "retained_canary_scan_receipt_sha256",
                "taint_scan_receipt_sha256",
                "token_usage_receipt_sha256",
                "provider_usage_receipt_sha256",
                "app_server_state_tree_sha256",
                "wip_publication_receipt_sha256",
            )
        )
        or not isinstance(raw["game"], str)
        or GAME_RE.fullmatch(raw["game"]) is None
        or not _is_int(raw["target_level"], minimum=1)
        or not _is_canonical_uuid(raw["codex_thread_id"])
        or Path(raw["solver_source_path"]).parent
        != Path(raw["wip_root_path"])
        or raw["taint_verdict"] != "clean"
        or (
            raw["supervisory_handoff_sha256"] is None
            and any(
                item is not None
                for item in (
                    raw[
                        "supervisory_native_reproduction_receipt_path"
                    ],
                    raw[
                        "supervisory_native_reproduction_receipt_sha256"
                    ],
                )
            )
        )
        or (
            raw["supervisory_handoff_sha256"] is not None
            and (
                not _is_sha256(
                    raw["supervisory_handoff_sha256"]
                )
                or not isinstance(
                    raw[
                        "supervisory_native_reproduction_receipt_path"
                    ],
                    str,
                )
                or not Path(
                    raw[
                        "supervisory_native_reproduction_receipt_path"
                    ]
                ).is_absolute()
                or not _is_sha256(
                    raw[
                        "supervisory_native_reproduction_receipt_sha256"
                    ]
                )
            )
        )
    ):
        raise SchedulerError("WIP binding has invalid values")
    return WipBinding(**raw)


def wip_binding_from_dict(value: object) -> WipBinding:
    """Strict shared parser used by the scheduler and production runner."""

    return _wip_from_dict(value)


def wip_binding_to_dict(wip: WipBinding) -> dict[str, object]:
    """Return the one canonical durable WIP projection."""

    return asdict(_wip_from_dict(asdict(wip)))


def _supervisory_handoff_binding_from_dict(
    value: object,
) -> SupervisoryHandoffBinding:
    raw = _strict_keys(
        value,
        set(SupervisoryHandoffBinding.__dataclass_fields__),
        "supervisory handoff binding",
    )
    output = auxiliary_output_from_dict(raw["output"])
    try:
        binding = SupervisoryHandoffBinding(
            **{**raw, "output": output}
        )
    except TypeError as exc:
        raise SchedulerError(
            "supervisory handoff binding schema mismatch"
        ) from exc
    return validate_supervisory_handoff_binding(binding)


def supervisory_handoff_binding_from_dict(
    value: object,
) -> SupervisoryHandoffBinding:
    """Strict shared parser for one admitted unverified handoff binding."""

    return _supervisory_handoff_binding_from_dict(value)


def supervisory_handoff_binding_to_dict(
    value: SupervisoryHandoffBinding,
) -> dict[str, object]:
    """Serialize the full host-only binding, including audit paths.

    This projection is for the scheduler journal and trusted host state only.
    Proposer-visible bundles must use :func:`supervisory_prompt_projection`.
    """

    return asdict(validate_supervisory_handoff_binding(value))


def supervisory_prompt_projection(
    value: SupervisoryHandoffBinding,
) -> dict[str, object]:
    """Minimal path-free bytes allowed to reach a native prompt."""

    binding = validate_supervisory_handoff_binding(value)
    handoff = binding.output.supervisory_handoff
    assert handoff is not None
    return {
        "schema": 1,
        "kind": "supervisory_unverified_hypothesis",
        "label": "UNVERIFIED_HYPOTHESIS",
        "assignment_id": binding.assignment_id,
        "frontier_sha256": binding.frontier_sha256,
        "parent_checkpoint_sha256":
            binding.parent_checkpoint_sha256,
        "output_manifest_sha256":
            binding.output_manifest_sha256,
        "supervisory_handoff_sha256":
            binding.supervisory_handoff_sha256,
        "admission_receipt_sha256":
            binding.admission_receipt_sha256,
        "prompt_authority": "unverified_hypothesis_only",
        "native_reproduction_required_before_wip_candidate_or_promotion":
            True,
        "scheduler_authority": False,
        "mutation_authority": False,
        "promotion_authority": False,
        "handoff": asdict(handoff),
        "socratic_challenge": asdict(binding.output.challenge),
    }


def _choice_from_dict(value: object) -> DispatchChoice:
    raw = _strict_keys(
        value,
        {
            "game",
            "target_level",
            "authoritative_target",
            "no_progress",
            "effort",
            "soft_allocation_seconds",
            "requested_wip_mode",
            "effective_wip_mode",
            "thread_mode",
            "selected_wip",
            "success_prior_micro",
            "conditional_novelty",
            "estimated_free_energy_micro",
            "reused_definition_calls",
            "ranking_key",
            "slots_to_fill",
            "reservation_units",
            "selected_supervisory_handoff",
        },
        "dispatch choice",
    )
    selected_wip = (
        None
        if raw["selected_wip"] is None
        else _wip_from_dict(raw["selected_wip"])
    )
    selected_supervisory_handoff = (
        None
        if raw["selected_supervisory_handoff"] is None
        else _supervisory_handoff_binding_from_dict(
            raw["selected_supervisory_handoff"]
        )
    )
    if (
        not isinstance(raw["game"], str)
        or GAME_RE.fullmatch(raw["game"]) is None
        or not _is_int(raw["target_level"], minimum=1)
        or not _is_int(raw["authoritative_target"], minimum=1)
        or not _is_int(raw["no_progress"])
        or raw["effort"] not in _EFFORT_RANK
        or not _is_int(raw["soft_allocation_seconds"], minimum=1)
        or raw["requested_wip_mode"]
        not in {"exclude", "restore_clean_same_frontier"}
        or raw["effective_wip_mode"]
        not in {"exclude", "restore_clean_same_frontier"}
        or raw["thread_mode"] not in {"new", "resume"}
        or not _is_int(raw["success_prior_micro"])
        or not _is_int(raw["conditional_novelty"])
        or not isinstance(raw["estimated_free_energy_micro"], int)
        or isinstance(raw["estimated_free_energy_micro"], bool)
        or not isinstance(raw["reused_definition_calls"], list)
        or any(
            not isinstance(item, str)
            for item in raw["reused_definition_calls"]
        )
        or not isinstance(raw["ranking_key"], list)
        or len(raw["ranking_key"]) != 4
        or not isinstance(raw["ranking_key"][0], int)
        or not isinstance(raw["ranking_key"][1], int)
        or not isinstance(raw["ranking_key"][2], int)
        or not isinstance(raw["ranking_key"][3], str)
        or not _is_int(raw["slots_to_fill"], minimum=1)
        or (
            raw["reservation_units"] is not None
            and not _is_int(raw["reservation_units"], minimum=1)
        )
    ):
        raise SchedulerError("dispatch choice has invalid values")
    return DispatchChoice(
        game=raw["game"],
        target_level=raw["target_level"],
        authoritative_target=raw["authoritative_target"],
        no_progress=raw["no_progress"],
        effort=raw["effort"],
        soft_allocation_seconds=raw["soft_allocation_seconds"],
        requested_wip_mode=raw["requested_wip_mode"],
        effective_wip_mode=raw["effective_wip_mode"],
        thread_mode=raw["thread_mode"],
        selected_wip=selected_wip,
        success_prior_micro=raw["success_prior_micro"],
        conditional_novelty=raw["conditional_novelty"],
        estimated_free_energy_micro=raw[
            "estimated_free_energy_micro"
        ],
        reused_definition_calls=tuple(
            raw["reused_definition_calls"]
        ),
        ranking_key=tuple(raw["ranking_key"]),  # type: ignore[arg-type]
        slots_to_fill=raw["slots_to_fill"],
        reservation_units=raw["reservation_units"],
        selected_supervisory_handoff=selected_supervisory_handoff,
    )


def decision_from_dict(value: object) -> SchedulerDecision:
    raw = _strict_keys(
        value,
        {
            "schema",
            "policy_name",
            "policy_sha256",
            "proposer_policy_sha256",
            "decision_id",
            "campaign_id",
            "attempt_id",
            "generation_id",
            "reservation_id",
            "journal_head_sequence",
            "journal_head_digest",
            "inventory_sha256",
            "eligible_frontiers",
            "eligible_frontiers_sha256",
            "active_attempt_ids",
            "active_auxiliary_assignment_ids",
            "max_lanes",
            "cost_window_id",
            "limit_units",
            "settled_units",
            "live_reservation_units",
            "choice",
            "decision_sha256",
        },
        "scheduler decision",
    )
    body = {key: raw[key] for key in raw if key != "decision_sha256"}
    if (
        raw["schema"] != 1
        or raw["policy_name"] != POLICY_NAME
        or raw["policy_sha256"] != SCHEDULER_POLICY_SHA256
        or raw["proposer_policy_sha256"] != PROPOSER_POLICY_SHA256
        or not _is_identifier(raw["decision_id"])
        or not _is_identifier(raw["campaign_id"])
        or not _is_identifier(raw["attempt_id"])
        or not _is_identifier(raw["generation_id"])
        or not _is_identifier(raw["reservation_id"])
        or not _is_int(raw["journal_head_sequence"])
        or not _is_sha256(raw["journal_head_digest"])
        or not _is_sha256(raw["inventory_sha256"])
        or not isinstance(raw["eligible_frontiers"], list)
        or not _is_sha256(raw["eligible_frontiers_sha256"])
        or raw["eligible_frontiers_sha256"]
        != sha256_json(raw["eligible_frontiers"])
        or not isinstance(raw["active_attempt_ids"], list)
        or any(
            not _is_identifier(item)
            for item in raw["active_attempt_ids"]
        )
        or raw["active_attempt_ids"]
        != sorted(set(raw["active_attempt_ids"]))
        or not isinstance(
            raw["active_auxiliary_assignment_ids"], list
        )
        or any(
            not _is_identifier(item)
            for item in raw["active_auxiliary_assignment_ids"]
        )
        or raw["active_auxiliary_assignment_ids"]
        != sorted(set(raw["active_auxiliary_assignment_ids"]))
        or not _is_int(raw["max_lanes"], minimum=1)
        or raw["max_lanes"] > MAX_LANES
        or not _is_identifier(raw["cost_window_id"])
        or (
            raw["limit_units"] is not None
            and not _is_int(raw["limit_units"])
        )
        or not _is_int(raw["settled_units"])
        or not _is_int(raw["live_reservation_units"])
        or not _is_sha256(raw["decision_sha256"])
        or raw["decision_sha256"] != sha256_json(body)
    ):
        raise SchedulerError("scheduler decision has invalid values")
    normalized_frontiers: list[dict[str, object]] = []
    for item in raw["eligible_frontiers"]:
        if not isinstance(item, dict):
            raise SchedulerError(
                "scheduler decision frontier is not an object"
            )
        normalized = dict(item)
        evidence = normalized.get("evidence")
        if not isinstance(evidence, dict):
            raise SchedulerError(
                "scheduler decision frontier lacks evidence"
            )
        normalized_evidence = dict(evidence)
        reuse = normalized_evidence.get("reused_definition_calls")
        if not isinstance(reuse, (list, tuple)) or any(
            not isinstance(name, str) for name in reuse
        ):
            raise SchedulerError(
                "scheduler decision frontier reuse is invalid"
            )
        normalized_evidence["reused_definition_calls"] = tuple(reuse)
        normalized["evidence"] = normalized_evidence
        public_receipts = normalized.get(
            "public_observation_receipt_sha256s"
        )
        if not isinstance(public_receipts, (list, tuple)) or any(
            not _is_sha256(item) for item in public_receipts
        ):
            raise SchedulerError(
                "scheduler decision frontier observations are invalid"
            )
        normalized[
            "public_observation_receipt_sha256s"
        ] = tuple(public_receipts)
        normalized_frontiers.append(normalized)
    choice = _choice_from_dict(raw["choice"])
    return SchedulerDecision(
        schema=1,
        policy_name=raw["policy_name"],
        policy_sha256=raw["policy_sha256"],
        proposer_policy_sha256=raw["proposer_policy_sha256"],
        decision_id=raw["decision_id"],
        campaign_id=raw["campaign_id"],
        attempt_id=raw["attempt_id"],
        generation_id=raw["generation_id"],
        reservation_id=raw["reservation_id"],
        journal_head_sequence=raw["journal_head_sequence"],
        journal_head_digest=raw["journal_head_digest"],
        inventory_sha256=raw["inventory_sha256"],
        eligible_frontiers=tuple(normalized_frontiers),
        eligible_frontiers_sha256=raw["eligible_frontiers_sha256"],
        active_attempt_ids=tuple(raw["active_attempt_ids"]),
        active_auxiliary_assignment_ids=tuple(
            raw["active_auxiliary_assignment_ids"]
        ),
        max_lanes=raw["max_lanes"],
        cost_window_id=raw["cost_window_id"],
        limit_units=raw["limit_units"],
        settled_units=raw["settled_units"],
        live_reservation_units=raw["live_reservation_units"],
        choice=choice,
        decision_sha256=raw["decision_sha256"],
    )


def _frontier_digest(
    game: str, reached: int, parent_checkpoint_sha256: str
) -> str:
    return sha256_json(
        {
            "game": game,
            "reached": reached,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
        }
    )


def _read_json_regular(path: Path, *, maximum: int) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_regular(path, maximum=maximum).decode("utf-8")
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SchedulerError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SchedulerError(f"expected JSON object: {path}")
    return value


def _verify_regular_sha256(
    path_value: object,
    digest_value: object,
    *,
    label: str,
    maximum: int,
) -> Path:
    if not isinstance(path_value, str) or not _is_sha256(digest_value):
        raise SchedulerError(f"{label} path/hash binding is invalid")
    path = Path(path_value)
    if not path.is_absolute():
        raise SchedulerError(f"{label} path is not absolute")
    try:
        raw = _read_regular(path, maximum=maximum)
    except SchedulerError as exc:
        raise SchedulerError(f"{label} cannot be reopened: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != digest_value:
        raise SchedulerError(f"{label} bytes differ from their hash")
    return path


def _reopen_json_receipt(
    path_value: object,
    digest_value: object,
    *,
    label: str,
) -> dict[str, Any]:
    path = _verify_regular_sha256(
        path_value,
        digest_value,
        label=label,
        maximum=16 * 1024 * 1024,
    )
    return _read_json_regular(path, maximum=16 * 1024 * 1024)


def _verify_host_blocker_result(
    payload: Mapping[str, Any],
    *,
    attempt: Mapping[str, object],
    campaign_id: str,
) -> str:
    """Verify the public half of a host-authenticated blocker settlement.

    The runner verifies the live-only HMAC.  This independent auditor reopens
    both receipts and proves their exact attempt/frontier/Arena binding; it
    never accepts a free-form model or operator reason as blocker authority.
    """

    blocker = payload.get("blocker")
    if not isinstance(blocker, Mapping) or set(blocker) != {
        "code",
        "receipt_path",
        "receipt_sha256",
    }:
        raise SchedulerError(
            "blocker result lacks exact host evidence"
        )
    code = blocker.get("code")
    if (
        code not in HOST_BLOCKER_CODES
        or payload.get("reason") != HOST_BLOCKER_REASON_PREFIX + str(code)
    ):
        raise SchedulerError(
            "blocker result uses a noncanonical host code"
        )
    receipt_path = _verify_regular_sha256(
        blocker.get("receipt_path"),
        blocker.get("receipt_sha256"),
        label="host blocker receipt",
        maximum=16 * 1024 * 1024,
    )
    if receipt_path.name != HOST_BLOCKER_RECEIPT_NAME:
        raise SchedulerError("host blocker receipt has the wrong name")
    receipt = _read_json_regular(
        receipt_path, maximum=16 * 1024 * 1024
    )
    arena_result = receipt.get("arena_host_result")
    expected_parent_level = int(attempt["target_level"]) - 1
    if (
        set(receipt) != HOST_BLOCKER_RECEIPT_FIELDS
        or receipt.get("schema") != 1
        or receipt.get("kind") != HOST_BLOCKER_RECEIPT_KIND
        or receipt.get("campaign_id") != campaign_id
        or receipt.get("generation_id") != attempt["generation_id"]
        or receipt.get("attempt_id") != attempt["attempt_id"]
        or not _is_sha256(receipt.get("attempt_spec_sha256"))
        or receipt.get("authority") != HOST_BLOCKER_AUTHORITY
        or receipt.get("code") != code
        or receipt.get("game") != attempt["game"]
        or receipt.get("frontier_sha256")
        != attempt["frontier_sha256"]
        or receipt.get("parent_checkpoint_sha256")
        != attempt["parent_checkpoint_sha256"]
        or receipt.get("parent_level") != expected_parent_level
        or receipt.get("target_level") != attempt["target_level"]
        or receipt.get("parent_terminal") is not True
        or not _is_sha256(receipt.get("arena_binding_sha256"))
        or not _is_sha256(receipt.get("parent_path_sha256"))
        or not _is_sha256(receipt.get("parent_snapshot_sha256"))
        or not _is_sha256(receipt.get("arena_host_result_sha256"))
        or not _is_sha256(receipt.get("host_authentication_sha256"))
        or not isinstance(arena_result, Mapping)
        or set(arena_result) != HOST_BLOCKER_ARENA_RESULT_FIELDS
        or sha256_json(dict(arena_result))
        != receipt.get("arena_host_result_sha256")
        or arena_result.get("binding_sha256")
        != receipt.get("arena_binding_sha256")
        or arena_result.get("game") != attempt["game"]
        or arena_result.get("parent_level") != expected_parent_level
        or arena_result.get("levels_completed") != expected_parent_level
        or arena_result.get("parent_terminal") is not True
        or arena_result.get("parent_snapshot_sha256")
        != receipt.get("parent_snapshot_sha256")
        or not isinstance(arena_result.get("parent_path"), list)
        or sha256_json(arena_result["parent_path"])
        != receipt.get("parent_path_sha256")
    ):
        raise SchedulerError(
            "host blocker receipt is malformed or bound to another frontier"
        )
    arena_path = _verify_regular_sha256(
        receipt.get("arena_session_binding_receipt_path"),
        receipt.get("arena_session_binding_receipt_sha256"),
        label="blocker Arena binding receipt",
        maximum=16 * 1024 * 1024,
    )
    if (
        arena_path.parent != receipt_path.parent
        or arena_path.name != "arena_session_binding_receipt.json"
    ):
        raise SchedulerError(
            "blocker Arena binding receipt left the attempt host root"
        )
    arena_receipt = _read_json_regular(
        arena_path, maximum=16 * 1024 * 1024
    )
    binding_event = arena_receipt.get("binding_event")
    if (
        set(arena_receipt) != {
            "schema",
            "kind",
            "campaign_id",
            "generation_id",
            "attempt_id",
            "attempt_spec_sha256",
            "binding_event",
        }
        or arena_receipt.get("schema") != 1
        or arena_receipt.get("kind")
        != "contiguous_arena_session_binding"
        or arena_receipt.get("campaign_id") != campaign_id
        or arena_receipt.get("generation_id")
        != attempt["generation_id"]
        or arena_receipt.get("attempt_id") != attempt["attempt_id"]
        or arena_receipt.get("attempt_spec_sha256")
        != receipt.get("attempt_spec_sha256")
        or not isinstance(binding_event, Mapping)
        or binding_event.get("binding_sha256")
        != receipt.get("arena_binding_sha256")
        or binding_event.get("seed_snapshot_sha256")
        != receipt.get("parent_snapshot_sha256")
        or binding_event.get("parent_path_sha256")
        != receipt.get("parent_path_sha256")
        or binding_event.get("game") != attempt["game"]
        or binding_event.get("frontier_sha256")
        != attempt["frontier_sha256"]
        or binding_event.get("parent_checkpoint_sha256")
        != attempt["parent_checkpoint_sha256"]
        or binding_event.get("parent_level") != expected_parent_level
        or binding_event.get("target_level") != attempt["target_level"]
    ):
        raise SchedulerError(
            "blocker Arena binding receipt is stale or substituted"
        )
    return str(code)


def _require_clean_scan(
    receipt: Mapping[str, Any],
    *,
    nested_name: str | None,
    label: str,
) -> None:
    scan: object = (
        receipt.get(nested_name)
        if nested_name is not None
        else receipt
    )
    if not isinstance(scan, Mapping):
        raise SchedulerError(f"{label} lacks its scan body")
    hits = scan.get("hits")
    canaries = scan.get("canary_occurrences", ())
    clean_canaries = (
        canaries == 0
        or canaries == []
        or canaries == {}
        or canaries == ()
    )
    if (
        scan.get("status") != "CLEAN"
        or hits not in ([], ())
        or not clean_canaries
    ):
        raise SchedulerError(f"{label} does not prove a clean result")


def _validate_terminal_wip(
    *,
    value: object,
    lane: Mapping[str, object],
    attempt: Mapping[str, object],
    campaign_id: str,
    cost_used: object,
) -> dict[str, object] | None:
    projected = _wip_projection_from_result(value)
    if projected is None:
        return None
    wip = _wip_from_dict(projected)
    if (
        wip.game != attempt["game"]
        or wip.target_level != attempt["target_level"]
        or wip.parent_checkpoint_sha256
        != attempt["parent_checkpoint_sha256"]
        or wip.frontier_sha256 != lane["frontier_sha256"]
        or wip.taint_verdict != "clean"
    ):
        raise SchedulerError(
            "terminal WIP does not match the exact clean attempt frontier"
        )
    _regular_tree_hash(
        Path(wip.wip_root_path),
        wip.wip_tree_sha256,
        label="terminal WIP",
    )
    _source_tree(
        Path(wip.solver_source_path),
        wip.solver_source_tree_sha256,
    )
    _regular_tree_hash(
        Path(wip.app_server_state_dir),
        wip.app_server_state_tree_sha256,
        label="terminal app-server state",
    )
    receipts = {
        "final": _reopen_json_receipt(
            wip.final_thread_binding_path,
            wip.final_thread_binding_sha256,
            label="final thread binding",
        ),
        "export": _reopen_json_receipt(
            wip.wip_export_receipt_path,
            wip.wip_export_receipt_sha256,
            label="WIP export receipt",
        ),
        "transcript": _reopen_json_receipt(
            wip.final_transcript_chain_receipt_path,
            wip.final_transcript_chain_receipt_sha256,
            label="final transcript-chain receipt",
        ),
        "controller": _reopen_json_receipt(
            wip.controller_state_scan_receipt_path,
            wip.controller_state_scan_receipt_sha256,
            label="controller-state scan receipt",
        ),
        "retained": _reopen_json_receipt(
            wip.retained_canary_scan_receipt_path,
            wip.retained_canary_scan_receipt_sha256,
            label="retained-canary scan receipt",
        ),
        "taint": _reopen_json_receipt(
            wip.taint_scan_receipt_path,
            wip.taint_scan_receipt_sha256,
            label="taint scan receipt",
        ),
        "token": _reopen_json_receipt(
            wip.token_usage_receipt_path,
            wip.token_usage_receipt_sha256,
            label="token-usage receipt",
        ),
        "provider": _reopen_json_receipt(
            wip.provider_usage_receipt_path,
            wip.provider_usage_receipt_sha256,
            label="provider-usage receipt",
        ),
        "publication": _reopen_json_receipt(
            wip.wip_publication_receipt_path,
            wip.wip_publication_receipt_sha256,
            label="WIP publication receipt",
        ),
    }
    attempt_id = str(attempt["attempt_id"])
    generation_id = str(attempt["generation_id"])
    publication = receipts["publication"]
    if (
        publication.get("kind") != "contiguous_wip_publication"
        or publication.get("schema") != 1
        or publication.get("campaign_id") != campaign_id
        or publication.get("generation_id") != generation_id
        or publication.get("attempt_id") != attempt_id
        or not _is_sha256(publication.get("attempt_spec_sha256"))
    ):
        raise SchedulerError("terminal WIP publication identity is invalid")
    receipt_identity = {
        "schema": 1,
        "campaign_id": campaign_id,
        "generation_id": generation_id,
        "attempt_id": attempt_id,
        "attempt_spec_sha256": publication["attempt_spec_sha256"],
    }
    receipt_paths = [
        Path(value)
        for value in (
            wip.final_thread_binding_path,
            wip.wip_export_receipt_path,
            wip.final_transcript_chain_receipt_path,
            wip.controller_state_scan_receipt_path,
            wip.retained_canary_scan_receipt_path,
            wip.taint_scan_receipt_path,
            wip.token_usage_receipt_path,
            wip.provider_usage_receipt_path,
            wip.wip_publication_receipt_path,
        )
    ]
    if len(set(receipt_paths)) != len(receipt_paths):
        raise SchedulerError("terminal WIP aliases distinct evidence receipts")
    for label, receipt in receipts.items():
        if any(
            receipt.get(key) != expected
            for key, expected in receipt_identity.items()
        ):
            raise SchedulerError(
                f"{label} WIP receipt is not bound to this attempt"
            )
    final = receipts["final"]
    expected_final = {
        "kind": "contiguous_final_thread_binding",
        "thread_id": wip.codex_thread_id,
        "transcript_chain_sha256": wip.transcript_chain_sha256,
        "token_usage_receipt_sha256": wip.token_usage_receipt_sha256,
        "provider_usage_receipt_sha256":
            wip.provider_usage_receipt_sha256,
        "app_server_state_tree_sha256":
            wip.app_server_state_tree_sha256,
        "controller_state_scan_receipt_sha256":
            wip.controller_state_scan_receipt_sha256,
        "retained_canary_scan_receipt_sha256":
            wip.retained_canary_scan_receipt_sha256,
        "taint_scan_receipt_sha256": wip.taint_scan_receipt_sha256,
        "wip_export_receipt_sha256": wip.wip_export_receipt_sha256,
    }
    if any(final.get(key) != expected for key, expected in expected_final.items()):
        raise SchedulerError(
            "final thread binding omits terminal WIP evidence"
        )
    transcript = receipts["transcript"]
    if (
        transcript.get("kind")
        != "contiguous_final_transcript_chain"
        or transcript.get("thread_id") != wip.codex_thread_id
        or transcript.get("chain_head_sha256")
        != wip.transcript_chain_sha256
    ):
        raise SchedulerError("terminal WIP transcript receipt is mismatched")
    token = receipts["token"]
    if (
        token.get("kind") != "contiguous_token_usage"
        or token.get("thread_id") != wip.codex_thread_id
        or token.get("final_event_observed") is not True
        or not isinstance(token.get("observations"), list)
        or not token["observations"]
    ):
        raise SchedulerError("terminal WIP token receipt is incomplete")
    provider = receipts["provider"]
    provider_keys = {
        "schema",
        "kind",
        "campaign_id",
        "generation_id",
        "attempt_id",
        "attempt_spec_sha256",
        "thread_id",
        "turn_id",
        "token_usage_observations",
        "pre_provider_usage_window",
        "post_provider_usage_window",
        "provider_usage_settlement",
    }
    try:
        import arc_agi3_codex_app_server_transport as Transport

        observations = provider.get("token_usage_observations")
        pre = Transport.provider_usage_window_from_dict(
            provider.get("pre_provider_usage_window")
        )
        post = Transport.provider_usage_window_from_dict(
            provider.get("post_provider_usage_window")
        )
        settlement = Transport.provider_usage_settlement_from_dict(
            provider.get("provider_usage_settlement"),
            pre=pre,
            post=post,
            token_usage_observations=observations,
        )
        provider_valid = (
            set(provider) == provider_keys
            and provider.get("kind") == "contiguous_provider_usage"
            and provider.get("generation_id")
            == attempt["generation_id"]
            and _is_sha256(provider.get("attempt_spec_sha256"))
            and provider.get("thread_id") == wip.codex_thread_id
            and _is_identifier(provider.get("turn_id"))
            and isinstance(observations, list)
            and observations
            and charge_to_units(settlement.charge)
            == charge_to_units(cost_used)
        )
    except Exception:
        provider_valid = False
    if not provider_valid:
        raise SchedulerError(
            "terminal WIP provider usage is not independently settled"
        )
    export = receipts["export"]
    if (
        export.get("kind") != "contiguous_wip_export"
        or export.get("game") != wip.game
        or export.get("target_level") != wip.target_level
        or export.get("parent_checkpoint_sha256")
        != wip.parent_checkpoint_sha256
        or export.get("frontier_sha256") != wip.frontier_sha256
        or export.get("wip_tree_sha256") != wip.wip_tree_sha256
        or export.get("solver_source_tree_sha256")
        != wip.solver_source_tree_sha256
    ):
        raise SchedulerError("terminal WIP export receipt is mismatched")
    _require_clean_scan(
        receipts["controller"],
        nested_name="controller_state_scan",
        label="controller-state scan",
    )
    if (
        receipts["controller"].get("kind")
        != "contiguous_controller_state_scan"
    ):
        raise SchedulerError("controller-state scan kind is invalid")
    _require_clean_scan(
        receipts["retained"],
        nested_name="retained_canary_scan",
        label="retained-canary scan",
    )
    if (
        receipts["retained"].get("kind")
        != "contiguous_retained_canary_scan"
    ):
        raise SchedulerError("retained-canary scan kind is invalid")
    _require_clean_scan(
        receipts["taint"],
        nested_name=None,
        label="taint scan",
    )
    if receipts["taint"].get("kind") != "contiguous_taint_scan":
        raise SchedulerError("taint scan kind is invalid")
    expected_publication = asdict(wip)
    expected_publication.pop("wip_publication_receipt_path")
    expected_publication.pop("wip_publication_receipt_sha256")
    if (
        set(publication)
        != {
            "schema",
            "kind",
            "campaign_id",
            "generation_id",
            "attempt_id",
            "attempt_spec_sha256",
            *expected_publication,
        }
        or any(
            publication.get(key) != expected
            for key, expected in expected_publication.items()
        )
    ):
        raise SchedulerError(
            "terminal WIP publication omits or changes its WIP binding"
        )
    return projected


def _validate_promoted_artifacts(
    *,
    game: str,
    authoritative_target: int,
    to_level: int,
    checkpoint_path: object,
    checkpoint_sha256: object,
    source_path: object,
    source_tree_sha256: object,
) -> None:
    checkpoint = _verify_regular_sha256(
        checkpoint_path,
        checkpoint_sha256,
        label="promoted checkpoint",
        maximum=64 * 1024 * 1024,
    )
    if not isinstance(source_path, str) or not _is_sha256(
        source_tree_sha256
    ):
        raise SchedulerError("promoted source path/hash binding is invalid")
    _source_tree(Path(source_path), str(source_tree_sha256))
    try:
        import arc_agi3_contiguous_supervisor as Supervisor

        parsed = Supervisor.load_trusted_checkpoint(
            checkpoint,
            expected_game=game,
            authoritative_target=authoritative_target,
        )
    except Exception as exc:
        raise SchedulerError("promoted checkpoint schema is invalid") from exc
    if parsed.reached != to_level or not parsed.validated:
        raise SchedulerError(
            "promoted checkpoint does not prove the exact target level"
        )


def _journal_segment_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    segment_directories: list[Path] = []
    for entry in root.iterdir():
        if entry.name.startswith("."):
            if (
                entry.name
                in {
                    ".journal.lock",
                    ".storage-emergency-reserve",
                    ".storage-quiescence-reserve",
                }
                or re.fullmatch(
                    r"\.pending-[A-Za-z0-9_.:-]+",
                    entry.name,
                )
                or re.fullmatch(
                    r"\.segment-\d{8}-closure\.json",
                    entry.name,
                )
            ):
                continue
            raise SchedulerError(
                "journal contains an unexpected hidden entry: "
                f"{entry.name}"
            )
        if entry.is_symlink():
            raise SchedulerError(
                f"journal contains a symlink entry: {entry.name}"
            )
        if entry.is_dir():
            if re.fullmatch(r"segment-\d{8}", entry.name) is None:
                raise SchedulerError(
                    "journal contains an unexpected directory: "
                    f"{entry.name}"
                )
            segment_directories.append(entry)
            continue
        if (
            not entry.is_file()
            or EVENT_FILE_RE.fullmatch(entry.name) is None
        ):
            raise SchedulerError(
                f"journal contains an unexpected entry: {entry.name}"
            )
        paths.append(entry)
    for directory in sorted(segment_directories):
        segment_number = int(directory.name.split("-")[1])
        for entry in directory.iterdir():
            if entry.name.startswith("."):
                if (
                    entry.name
                    in {
                        ".checkpoint.json",
                        ".closure.json",
                    }
                    or re.fullmatch(
                        r"\.pending-[A-Za-z0-9_.:-]+",
                        entry.name,
                    )
                ):
                    continue
                raise SchedulerError(
                    "journal segment contains an unexpected hidden "
                    f"entry: {entry.name}"
                )
            if (
                entry.is_symlink()
                or not entry.is_file()
                or EVENT_FILE_RE.fullmatch(entry.name) is None
                or (
                    (int(entry.name[:20]) - 1)
                    // JOURNAL_SEGMENT_EVENT_LIMIT
                    + 1
                )
                != segment_number
            ):
                raise SchedulerError(
                    "journal segment contains an invalid event"
                )
            paths.append(entry)
    return sorted(paths, key=lambda path: path.name)


def _expected_journal_segment_closure(
    events: Sequence[Mapping[str, Any]],
    segment_number: int,
) -> dict[str, Any]:
    start = (
        (segment_number - 1)
        * JOURNAL_SEGMENT_EVENT_LIMIT
        + 1
    )
    end = segment_number * JOURNAL_SEGMENT_EVENT_LIMIT
    selected = [
        event
        for event in events
        if start <= int(event["sequence"]) <= end
    ]
    if len(selected) != JOURNAL_SEGMENT_EVENT_LIMIT:
        raise SchedulerError(
            "journal segment closure lacks its full event range"
        )
    inventory = [
        {
            "sequence": event["sequence"],
            "event_id": event["event_id"],
            "digest": event["digest"],
        }
        for event in selected
    ]
    return {
        "schema": 1,
        "kind": "contiguous_journal_checkpoint_segment",
        "segment_number": segment_number,
        "start_sequence": start,
        "end_sequence": end,
        "event_count": len(selected),
        "first_event_digest": selected[0]["digest"],
        "last_event_digest": selected[-1]["digest"],
        "event_inventory_sha256": hashlib.sha256(
            canonical_json(inventory)
        ).hexdigest(),
        "status": "CLOSED",
    }


def _read_journal_segment_control(
    path: Path, *, label: str
) -> dict[str, Any]:
    metadata = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o400
    ):
        raise SchedulerError(
            f"{label} is not one immutable regular file"
        )
    return _read_json_regular(path, maximum=64 * 1024)


def _validate_journal_segment_chain(
    root: Path,
    events: Sequence[Mapping[str, Any]],
) -> None:
    directories: dict[int, Path] = {}
    closures: dict[int, Path] = {}
    for entry in root.iterdir():
        match = re.fullmatch(r"segment-(\d{8})", entry.name)
        if match and entry.is_dir() and not entry.is_symlink():
            directories[int(match.group(1))] = entry
            continue
        match = re.fullmatch(
            r"\.segment-(\d{8})-closure\.json",
            entry.name,
        )
        if match:
            closures[int(match.group(1))] = entry
    if directories:
        highest = max(directories)
        if (
            min(directories) != 2
            or set(directories) != set(range(2, highest + 1))
        ):
            raise SchedulerError(
                "journal segment directory chain has a gap"
            )
    if set(closures) - {1}:
        raise SchedulerError(
            "journal root has a misplaced segment closure"
        )
    for number, directory in directories.items():
        path = directory / ".closure.json"
        if path.exists() or path.is_symlink():
            closures[number] = path
    event_count = len(events)
    full_segments = event_count // JOURNAL_SEGMENT_EVENT_LIMIT
    last_event_segment = (
        full_segments
        if event_count > 0
        and event_count % JOURNAL_SEGMENT_EVENT_LIMIT == 0
        else full_segments + (1 if event_count else 0)
    )
    allowed_highest = (
        last_event_segment + 1
        if event_count > 0
        and event_count % JOURNAL_SEGMENT_EVENT_LIMIT == 0
        else last_event_segment
    )
    if directories and max(directories) > allowed_highest:
        raise SchedulerError(
            "journal segment directory is ahead of history"
        )
    for number, path in sorted(closures.items()):
        if number > full_segments:
            raise SchedulerError(
                "journal closes an incomplete segment"
            )
        expected = _expected_journal_segment_closure(
            events, number
        )
        observed = _read_journal_segment_control(
            path, label="journal segment closure"
        )
        if set(observed) != set(expected) or observed != expected:
            raise SchedulerError(
                "journal segment closure changed"
            )
    for number, directory in sorted(directories.items()):
        prior_number = number - 1
        prior_path = (
            root / ".segment-00000001-closure.json"
            if prior_number == 1
            else (
                root
                / f"segment-{prior_number:08d}"
                / ".closure.json"
            )
        )
        if closures.get(prior_number) != prior_path:
            raise SchedulerError(
                "journal segment lacks its prior closure"
            )
        checkpoint_path = directory / ".checkpoint.json"
        selected_events = [
            event
            for event in events
            if (
                (int(event["sequence"]) - 1)
                // JOURNAL_SEGMENT_EVENT_LIMIT
                + 1
            )
            == number
        ]
        if not (
            checkpoint_path.exists()
            or checkpoint_path.is_symlink()
        ):
            if selected_events or (
                directory / ".closure.json"
            ).exists():
                raise SchedulerError(
                    "journal segment lacks its checkpoint"
                )
            continue
        expected_checkpoint = {
            "schema": 1,
            "kind": "contiguous_journal_segment_genesis",
            "segment_number": number,
            "start_sequence": (
                (number - 1)
                * JOURNAL_SEGMENT_EVENT_LIMIT
                + 1
            ),
            "previous_segment_closure_path": str(prior_path),
            "previous_segment_closure_sha256": hashlib.sha256(
                _read_regular(
                    prior_path, maximum=64 * 1024
                )
            ).hexdigest(),
            "previous_event_digest": events[
                (number - 1)
                * JOURNAL_SEGMENT_EVENT_LIMIT
                - 1
            ]["digest"],
            "status": "OPEN",
        }
        observed_checkpoint = _read_journal_segment_control(
            checkpoint_path,
            label="journal segment checkpoint",
        )
        if (
            set(observed_checkpoint) != set(expected_checkpoint)
            or observed_checkpoint != expected_checkpoint
        ):
            raise SchedulerError(
                "journal segment checkpoint changed"
            )


def read_journal(campaign_root: Path) -> list[dict[str, Any]]:
    root = Path(campaign_root).resolve() / "attempt_journal"
    if root.is_symlink() or not root.is_dir():
        raise SchedulerError("campaign has no regular attempt_journal")
    def directory_signature() -> tuple[
        tuple[str, int, int], ...
    ]:
        selected = [root, *sorted(
            path
            for path in root.iterdir()
            if (
                not path.is_symlink()
                and path.is_dir()
                and re.fullmatch(
                    r"segment-\d{8}", path.name
                )
                is not None
            )
        )]
        return tuple(
            (
                str(path.relative_to(root)) or ".",
                path.stat(follow_symlinks=False).st_mtime_ns,
                path.stat(follow_symlinks=False).st_ctime_ns,
            )
            for path in selected
        )

    before = directory_signature()
    paths = _journal_segment_paths(root)
    events: list[dict[str, Any]] = []
    prior: str | None = None
    identifiers: set[str] = set()
    for expected_sequence, path in enumerate(paths, 1):
        event = _strict_keys(
            _read_json_regular(path, maximum=64 * 1024 * 1024),
            {
                "schema",
                "sequence",
                "event_id",
                "kind",
                "recorded_at",
                "previous_digest",
                "payload",
                "digest",
            },
            "journal event",
        )
        body = {key: event[key] for key in event if key != "digest"}
        if (
            event["schema"] != JOURNAL_SCHEMA
            or event["sequence"] != expected_sequence
            or isinstance(event["sequence"], bool)
            or not _is_identifier(event["event_id"])
            or not _is_identifier(event["kind"])
            or not isinstance(event["recorded_at"], (int, float))
            or isinstance(event["recorded_at"], bool)
            or not math.isfinite(float(event["recorded_at"]))
            or event["previous_digest"] != prior
            or not isinstance(event["payload"], dict)
            or event["event_id"] in identifiers
            or not _is_sha256(event["digest"])
            or event["digest"] != _event_digest(body)
            or not path.name.startswith(f"{expected_sequence:020d}-")
        ):
            raise SchedulerError(
                f"invalid journal event at sequence {expected_sequence}"
            )
        events.append(event)
        identifiers.add(event["event_id"])
        prior = event["digest"]
    _validate_journal_segment_chain(root, events)
    after = directory_signature()
    if (
        before != after
        or [
            str(path.relative_to(root))
            for path in _journal_segment_paths(root)
        ]
        != [str(path.relative_to(root)) for path in paths]
    ):
        raise SchedulerError("journal changed while it was being audited")
    if not events:
        raise SchedulerError("campaign journal is empty")
    return events


def journal_prefix_status(campaign_root: Path) -> dict[str, int]:
    """Return trusted full-prefix consumption/headroom for live status."""

    root = Path(campaign_root).resolve() / "attempt_journal"
    if root.is_symlink() or not root.is_dir():
        raise SchedulerError("campaign has no regular attempt_journal")
    used = 0
    events = 0
    for entry in _journal_segment_paths(root):
        metadata = entry.stat(follow_symlinks=False)
        if (
            metadata.st_nlink != 1
            or metadata.st_size > MAX_JOURNAL_EVENT_BYTES
        ):
            raise SchedulerError(
                "journal prefix contains a hard-linked or oversized event"
            )
        used += metadata.st_size
        events += 1
    # Each event file is ``canonical(event) + "\\n"``.  The retained canonical
    # JSON tuple is ``[event,...]`` and is therefore exactly one byte larger
    # than the sum of those newline-terminated files (including for N=1).
    retained_prefix_bytes = used + 1
    if retained_prefix_bytes > MAX_JOURNAL_PREFIX_BYTES:
        raise SchedulerError(
            "journal full-prefix evidence exceeds its 24 MiB bound"
        )
    return {
        "events": events,
        "used_bytes": retained_prefix_bytes,
        "limit_bytes": MAX_JOURNAL_PREFIX_BYTES,
        "headroom_bytes":
            MAX_JOURNAL_PREFIX_BYTES - retained_prefix_bytes,
    }


def require_dispatch_headroom(campaign_root: Path) -> dict[str, int]:
    """Fail new dispatch early while preserving terminal-evidence capacity."""

    status = journal_prefix_status(campaign_root)
    if status["headroom_bytes"] < MIN_DISPATCH_HEADROOM_BYTES:
        raise SchedulerError(
            "journal lacks the 1 MiB terminal-evidence dispatch reserve"
        )
    return status


def _zero_entry(value: object, label: str) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise SchedulerError(f"{label} entry is invalid")
    path = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path, str) or not _is_sha256(digest):
        raise SchedulerError(f"{label} entry is invalid")
    return path, digest


def _wip_projection_from_result(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    projected = _strict_keys(
        value,
        set(WipBinding.__dataclass_fields__),
        "result WIP",
    )
    return asdict(_wip_from_dict(projected))


def _reservation_scheduler_binding(
    payload: Mapping[str, object],
) -> dict[str, object]:
    candidates: list[Mapping[str, object]] = [payload]
    for key in ("scheduler", "reservation"):
        selected = payload.get(key)
        if isinstance(selected, Mapping):
            candidates.append(selected)
    required = set(reservation_binding.__annotations__)  # always empty/use below
    del required
    names = {
        "scheduler_decision_id",
        "scheduler_decision_sha256",
        "scheduler_policy_sha256",
        "budget_reservation_id",
        "budget_reservation_units",
        "cost_window_id",
        "attempt_id",
        "generation_id",
    }
    for candidate in candidates:
        if names <= set(candidate):
            return {name: candidate[name] for name in names}
    raise SchedulerError("attempt reservation lacks scheduler binding")


def _snapshot_from_audit_state(
    *,
    event: Mapping[str, Any],
    genesis: Mapping[str, Any],
    lanes: Mapping[str, Mapping[str, object]],
    budget: BudgetState,
    auxiliary_assignments: Sequence[AuxiliaryAssignmentState] = (),
    complexity_rounds: Sequence[ComplexityRoundState] = (),
    sidecar_requests: Sequence[SidecarRequestEvidence] = (),
) -> CampaignSnapshot:
    frontiers: list[Frontier] = []
    for game, lane in lanes.items():
        raw_wip = lane["wip"]
        wip = _wip_from_dict(raw_wip) if raw_wip is not None else None
        evidence = selection_evidence(
            parent_source_path=str(lane["parent_source_path"]),
            parent_source_tree_sha256=str(
                lane["parent_source_tree_sha256"]
            ),
            candidate_source_path=(
                wip.solver_source_path if wip is not None else None
            ),
            candidate_source_tree_sha256=(
                wip.solver_source_tree_sha256
                if wip is not None else None
            ),
        )
        public_receipts = tuple(
            sorted(
                set(
                    lane.get(
                        "public_observation_receipt_sha256s", ()
                    )
                )
            )
        )
        frontiers.append(
            Frontier(
                game=game,
                target=int(lane["target"]),
                reached=int(lane["reached"]),
                no_progress=int(lane["no_progress"]),
                last_dispatch_sequence=int(
                    lane["last_dispatch_sequence"]
                ),
                parent_checkpoint_sha256=str(
                    lane["parent_checkpoint_sha256"]
                ),
                parent_source_path=str(lane["parent_source_path"]),
                parent_source_tree_sha256=str(
                    lane["parent_source_tree_sha256"]
                ),
                frontier_sha256=str(lane["frontier_sha256"]),
                active_attempt_id=(
                    str(lane["active_attempt_id"])
                    if lane["active_attempt_id"] is not None
                    else None
                ),
                draining=bool(lane["draining"]),
                blocked_reason=(
                    str(lane["blocked_reason"])
                    if lane["blocked_reason"] is not None
                    else None
                ),
                wip=wip,
                evidence=evidence,
                public_observation_receipt_sha256s=public_receipts,
                observation_ledger_sha256=(
                    public_observation_ledger_sha256(
                        game=game,
                        frontier_sha256=str(
                            lane["frontier_sha256"]
                        ),
                        parent_checkpoint_sha256=str(
                            lane["parent_checkpoint_sha256"]
                        ),
                        receipt_sha256s=public_receipts,
                    )
                ),
            )
        )
    return CampaignSnapshot(
        campaign_id=str(genesis["campaign_id"]),
        journal_head_sequence=int(event["sequence"]) - 1,
        journal_head_digest=str(event["previous_digest"]),
        inventory=tuple(validate_inventory(genesis["inventory"]).items()),
        max_lanes=int(genesis["max_lanes"]),
        frontiers=tuple(frontiers),
        budget=budget,
        clean_proposer_settlements=tuple(
            settlement
            for lane in lanes.values()
            for settlement in lane.get(
                "clean_proposer_settlements", []
            )
            if isinstance(settlement, CleanProposerSettlement)
        ),
        complexity_rounds=tuple(complexity_rounds),
        auxiliary_assignments=tuple(auxiliary_assignments),
        sidecar_requests=tuple(sidecar_requests),
    )


def _validate_decision_against_state(
    decision: SchedulerDecision,
    *,
    event: Mapping[str, Any],
    genesis: Mapping[str, Any],
    lanes: dict[str, dict[str, object]],
    budget: BudgetState,
    auxiliary_assignments: Sequence[AuxiliaryAssignmentState] = (),
    complexity_rounds: Sequence[ComplexityRoundState] = (),
    sidecar_requests: Sequence[SidecarRequestEvidence] = (),
) -> None:
    if (
        decision.campaign_id != genesis["campaign_id"]
        or decision.journal_head_sequence != event["sequence"] - 1
        or decision.journal_head_digest != event["previous_digest"]
        or decision.inventory_sha256
        != inventory_sha256(genesis["inventory"])
        or decision.max_lanes != genesis["max_lanes"]
        or decision.cost_window_id != budget.cost_window_id
        or decision.limit_units != budget.limit_units
        or decision.settled_units != budget.settled_units
        or decision.live_reservation_units
        != sum(item.units for item in budget.live_reservations)
    ):
        raise SchedulerError("scheduler decision is stale against journal state")
    frontiers: list[Frontier] = []
    for game, lane in lanes.items():
        raw_wip = lane["wip"]
        wip = _wip_from_dict(raw_wip) if raw_wip is not None else None
        evidence = selection_evidence(
            parent_source_path=str(lane["parent_source_path"]),
            parent_source_tree_sha256=str(
                lane["parent_source_tree_sha256"]
            ),
            candidate_source_path=(
                wip.solver_source_path if wip is not None else None
            ),
            candidate_source_tree_sha256=(
                wip.solver_source_tree_sha256
                if wip is not None else None
            ),
        )
        public_receipts = tuple(
            sorted(
                set(
                    lane.get(
                        "public_observation_receipt_sha256s", ()
                    )
                )
            )
        )
        frontiers.append(
            Frontier(
                game=game,
                target=int(lane["target"]),
                reached=int(lane["reached"]),
                no_progress=int(lane["no_progress"]),
                last_dispatch_sequence=int(
                    lane["last_dispatch_sequence"]
                ),
                parent_checkpoint_sha256=str(
                    lane["parent_checkpoint_sha256"]
                ),
                parent_source_path=str(lane["parent_source_path"]),
                parent_source_tree_sha256=str(
                    lane["parent_source_tree_sha256"]
                ),
                frontier_sha256=str(lane["frontier_sha256"]),
                active_attempt_id=(
                    str(lane["active_attempt_id"])
                    if lane["active_attempt_id"] is not None
                    else None
                ),
                draining=bool(lane["draining"]),
                blocked_reason=(
                    str(lane["blocked_reason"])
                    if lane["blocked_reason"] is not None
                    else None
                ),
                wip=wip,
                evidence=evidence,
                public_observation_receipt_sha256s=public_receipts,
                observation_ledger_sha256=(
                    public_observation_ledger_sha256(
                        game=game,
                        frontier_sha256=str(
                            lane["frontier_sha256"]
                        ),
                        parent_checkpoint_sha256=str(
                            lane["parent_checkpoint_sha256"]
                        ),
                        receipt_sha256s=public_receipts,
                    )
                ),
            )
        )
    reconstructed = CampaignSnapshot(
        campaign_id=str(genesis["campaign_id"]),
        journal_head_sequence=int(event["sequence"]) - 1,
        journal_head_digest=str(event["previous_digest"]),
        inventory=tuple(validate_inventory(genesis["inventory"]).items()),
        max_lanes=int(genesis["max_lanes"]),
        frontiers=tuple(frontiers),
        budget=budget,
        clean_proposer_settlements=tuple(
            settlement
            for lane in lanes.values()
            for settlement in lane.get(
                "clean_proposer_settlements", []
            )
            if isinstance(settlement, CleanProposerSettlement)
        ),
        complexity_rounds=tuple(complexity_rounds),
        auxiliary_assignments=tuple(auxiliary_assignments),
        sidecar_requests=tuple(sidecar_requests),
    )
    expected = build_decision(
        reconstructed,
        decision_id=decision.decision_id,
        attempt_id=decision.attempt_id,
        generation_id=decision.generation_id,
        reservation_id=decision.reservation_id,
    )
    if expected is None or expected != decision:
        raise SchedulerError(
            "scheduler decision is not the exact canonical reconstruction"
        )


_LIFECYCLE_TRANSITIONS: dict[str, tuple[frozenset[str], str | None]] = {
    "ATTEMPT_PREPARED": (frozenset({"RESERVED"}), "PREPARED"),
    "ATTEMPT_RETRY": (
        frozenset({
            "RESERVED",
            "PREPARED",
            "BACKEND_PREPARED",
            "RUNNING",
            "DRAINING",
            "EXITED",
            "COLLECTED",
            "COLLECTION_REJECTED",
            "PROMOTING",
        }),
        None,
    ),
    "BACKEND_PREPARED": (
        frozenset({"PREPARED"}),
        "BACKEND_PREPARED",
    ),
    "ATTEMPT_LAUNCHED": (
        frozenset({"BACKEND_PREPARED"}),
        "RUNNING",
    ),
    "ATTEMPT_OBSERVED": (
        frozenset({"RUNNING", "DRAINING"}),
        None,
    ),
    "ATTEMPT_EXITED": (
        frozenset({"RUNNING", "DRAINING"}),
        "EXITED",
    ),
    "ATTEMPT_COLLECTED": (
        frozenset({"EXITED"}),
        "COLLECTED",
    ),
    "ATTEMPT_COLLECTION_REJECTED": (
        frozenset({"EXITED"}),
        "COLLECTION_REJECTED",
    ),
    "ATTEMPT_TORN_DOWN": (
        frozenset({"COLLECTED", "COLLECTION_REJECTED"}),
        "TORN_DOWN",
    ),
}


def _verify_exact_auxiliary_receipt(
    path_value: object,
    digest_value: object,
    *,
    expected: Mapping[str, object],
    label: str,
) -> None:
    receipt = _reopen_json_receipt(
        path_value, digest_value, label=label
    )
    if receipt != dict(expected):
        raise SchedulerError(
            f"{label} is not the exact host-bound auxiliary receipt"
        )


def _verify_auxiliary_admission_gates(
    payload: Mapping[str, object],
    *,
    assignment: AuxiliaryAssignmentState,
    admitted_evidence_sha256: str,
    admission_contract_sha256: str,
    admission_kind: str,
) -> None:
    if assignment.output is None:
        raise SchedulerError("auxiliary admission has no quarantined output")
    output_sha = assignment.output.output_manifest_sha256
    common = {
        "schema": 1,
        "assignment_id": assignment.assignment_id,
        "frontier_sha256": assignment.frontier_sha256,
        "parent_checkpoint_sha256":
            assignment.parent_checkpoint_sha256,
        "output_manifest_sha256": output_sha,
    }
    replay_expected = {
        **common,
        "kind": "auxiliary_fresh_public_replay",
        "status": "PASS",
    }
    taint_expected = {
        **common,
        "kind": "auxiliary_taint_scan",
        "status": "CLEAN",
    }
    provenance_expected = {
        **common,
        "kind": "auxiliary_provenance_scan",
        "status": "PASS",
    }
    for prefix, expected in (
        ("fresh_replay", replay_expected),
        ("taint", taint_expected),
        ("provenance", provenance_expected),
    ):
        _verify_exact_auxiliary_receipt(
            payload.get(f"{prefix}_receipt_path"),
            payload.get(f"{prefix}_receipt_sha256"),
            expected=expected,
            label=f"auxiliary {prefix.replace('_', ' ')} receipt",
        )
    admission_expected = {
        **common,
        "kind": admission_kind,
        "authority": "host_only",
        "admission_contract_sha256": admission_contract_sha256,
        "fresh_replay_receipt_sha256":
            payload["fresh_replay_receipt_sha256"],
        "taint_receipt_sha256": payload["taint_receipt_sha256"],
        "provenance_receipt_sha256":
            payload["provenance_receipt_sha256"],
        "admitted_evidence_sha256": admitted_evidence_sha256,
        "verdict": "ADMITTED",
    }
    _verify_exact_auxiliary_receipt(
        payload.get("admission_receipt_path"),
        payload.get("admission_receipt_sha256"),
        expected=admission_expected,
        label="auxiliary host admission receipt",
    )


def _audit_events(
    events: Sequence[Mapping[str, Any]],
    *,
    allow_pending_decision: bool = False,
) -> dict[str, object]:
    genesis_event = events[0]
    if genesis_event["kind"] != "GENESIS":
        raise SchedulerError("journal does not begin with GENESIS")
    genesis = genesis_event["payload"]
    inventory = validate_inventory(genesis.get("inventory"))
    if (
        genesis.get("inventory_sha256") != inventory_sha256(inventory)
        or not _is_identifier(genesis.get("campaign_id"))
        or not _is_int(genesis.get("max_lanes"), minimum=1)
        or genesis["max_lanes"] > MAX_LANES
        or not isinstance(genesis.get("zero_checkpoints"), dict)
        or not isinstance(genesis.get("zero_sources"), dict)
        or set(genesis["zero_checkpoints"]) != set(inventory)
        or set(genesis["zero_sources"]) != set(inventory)
    ):
        raise SchedulerError("GENESIS scheduler contract is invalid")
    cost_window_id = genesis.get("cost_window_id")
    if not _is_identifier(cost_window_id):
        raise SchedulerError("GENESIS lacks an immutable cost_window_id")
    expected_limit_units = limit_to_units(genesis.get("limit"))
    if (
        genesis.get("limit_units") != expected_limit_units
        or (
            genesis.get("limit") is None
            and genesis.get("limit_units") is not None
        )
    ):
        raise SchedulerError("GENESIS budget normalization is invalid")
    budget = BudgetState(
        cost_window_id=cost_window_id,
        limit_units=expected_limit_units,
        settled_units=0,
        live_reservations=(),
    )
    raw_auxiliary_configuration = genesis.get(
        "auxiliary_launch_configuration"
    )
    auxiliary_configuration = (
        disabled_auxiliary_launch_configuration()
        if raw_auxiliary_configuration is None
        else auxiliary_launch_configuration_from_dict(
            raw_auxiliary_configuration
        )
    )
    lanes: dict[str, dict[str, object]] = {}
    for game, target in inventory.items():
        checkpoint_path, checkpoint_sha = _zero_entry(
            genesis["zero_checkpoints"][game], "zero checkpoint"
        )
        source_path, source_sha = _zero_entry(
            genesis["zero_sources"][game], "zero source"
        )
        checkpoint = _verify_regular_sha256(
            checkpoint_path,
            checkpoint_sha,
            label=f"{game} zero checkpoint",
            maximum=16 * 1024 * 1024,
        )
        try:
            import arc_agi3_contiguous_supervisor as Supervisor

            parsed_zero = Supervisor.load_trusted_checkpoint(
                checkpoint,
                expected_game=game,
                authoritative_target=target,
            )
        except Exception as exc:
            raise SchedulerError(
                f"{game} zero checkpoint schema is invalid"
            ) from exc
        if (
            parsed_zero.reached != 0
            or parsed_zero.total_marginal_C != 0
            or parsed_zero.records
            or parsed_zero.final_path
            or parsed_zero.validated
        ):
            raise SchedulerError(
                f"{game} zero checkpoint is not the canonical L0 anchor"
            )
        _source_tree(Path(source_path), source_sha)
        lanes[game] = {
            "game": game,
            "target": target,
            "reached": 0,
            "no_progress": 0,
            "last_dispatch_sequence": 0,
            "parent_checkpoint_sha256": checkpoint_sha,
            "parent_checkpoint_path": checkpoint_path,
            "parent_source_path": source_path,
            "parent_source_tree_sha256": source_sha,
            "frontier_sha256": _frontier_digest(game, 0, checkpoint_sha),
            "active_attempt_id": None,
            "draining": False,
            "blocked_reason": None,
            "wip": None,
            "clean_proposer_settlements": [],
            "public_observation_receipt_sha256s": [],
        }
    pending: tuple[SchedulerDecision, Mapping[str, Any]] | None = None
    pending_auxiliary: tuple[
        AuxiliaryDecision, Mapping[str, Any]
    ] | None = None
    attempts: dict[str, dict[str, object]] = {}
    auxiliary_assignments: dict[str, AuxiliaryAssignmentState] = {}
    complexity_rounds: list[ComplexityRoundState] = []
    sidecar_requests: list[SidecarRequestEvidence] = []
    used_decision_ids: set[str] = set()
    used_attempt_ids: set[str] = set()
    used_generation_ids: set[str] = set()
    used_reservation_ids: set[str] = set()
    used_expert_ids: set[str] = set()
    used_thread_ids: set[str] = set()
    failure_operation_circuits: dict[
        str, dict[str, object]
    ] = {}
    failure_domain_circuits: dict[
        str, dict[str, object]
    ] = {}
    operator_incident: dict[str, object] | None = None
    substrate_incident: dict[str, object] | None = None
    storage_incident: dict[str, object] | None = None
    storage_quiescence: dict[str, object] | None = None
    decisions = reservations = settlements = promotions = 0
    auxiliary_decisions = auxiliary_reservations = 0
    auxiliary_settlements = auxiliary_admissions = 0
    for event in events[1:]:
        kind = event["kind"]
        payload = event["payload"]
        if (
            pending is not None
            and kind not in {
                "ATTEMPT_RESERVED",
                "JOURNAL_OR_STORAGE_EXHAUSTED",
                "STORAGE_EMERGENCY_QUIESCED",
            }
        ):
            raise SchedulerError(
                "SCHEDULER_DECISION is not immediately consumed by reservation"
            )
        if (
            pending_auxiliary is not None
            and kind not in {
                "AUXILIARY_RESERVED",
                "JOURNAL_OR_STORAGE_EXHAUSTED",
                "STORAGE_EMERGENCY_QUIESCED",
            }
        ):
            raise SchedulerError(
                "AUXILIARY_DECISION is not immediately consumed by reservation"
            )
        if kind == "JOURNAL_OR_STORAGE_EXHAUSTED":
            incident = _strict_keys(
                payload,
                {
                    "reason_code",
                    "failed_event_id",
                    "failed_event_kind",
                    "failure_stage",
                    "error_code",
                    "storage_snapshot",
                    "solver_authority",
                    "wip_authority",
                    "cost_authority",
                    "promotion_authority",
                    "status",
                },
                kind,
            )
            snapshot = incident["storage_snapshot"]
            if (
                storage_incident is not None
                or incident["reason_code"]
                != "journal_or_storage_exhausted"
                or not _is_identifier(
                    incident["failed_event_id"]
                )
                or not _is_identifier(
                    incident["failed_event_kind"]
                )
                or not _is_identifier(incident["failure_stage"])
                or not _is_identifier(incident["error_code"])
                or not isinstance(snapshot, dict)
                or any(
                    incident[name] is not False
                    for name in (
                        "solver_authority",
                        "wip_authority",
                        "cost_authority",
                        "promotion_authority",
                    )
                )
                or incident["status"] != "OPERATOR_INCIDENT"
            ):
                raise SchedulerError(
                    "journal/storage incident is malformed"
                )
            if snapshot and (
                set(snapshot)
                != {
                    "schema", "kind", "filesystem_device",
                    "available_bytes", "available_inodes",
                    "required_event_bytes", "minimum_free_bytes",
                    "minimum_free_inodes", "byte_admitted",
                    "inode_admitted",
                }
                or snapshot.get("schema") != 1
                or snapshot.get("kind")
                != "contiguous_journal_filesystem_admission"
                or not all(
                    _is_int(snapshot.get(name), minimum=0)
                    for name in (
                        "filesystem_device",
                        "available_bytes",
                        "available_inodes",
                        "required_event_bytes",
                        "minimum_free_bytes",
                        "minimum_free_inodes",
                    )
                )
                or not isinstance(
                    snapshot.get("byte_admitted"), bool
                )
                or not isinstance(
                    snapshot.get("inode_admitted"), bool
                )
            ):
                raise SchedulerError(
                    "journal/storage filesystem snapshot is malformed"
                )
            storage_incident = {
                **dict(incident),
                "incident_event_sequence": event["sequence"],
                "incident_event_digest": event["digest"],
            }
            continue
        if kind == "STORAGE_EMERGENCY_QUIESCED":
            quiescence = _strict_keys(
                payload,
                {
                    "storage_incident_event_sequence",
                    "storage_incident_event_digest",
                    "primary_containments",
                    "auxiliary_aborts",
                    "promotion_quarantines",
                    "all_primary_children_absent",
                    "all_auxiliary_children_absent",
                    "all_promotions_non_authoritative",
                    "solver_authority",
                    "wip_authority",
                    "cost_authority",
                    "promotion_authority",
                    "status",
                },
                kind,
            )
            primary = quiescence["primary_containments"]
            auxiliary = quiescence["auxiliary_aborts"]
            promotions = quiescence["promotion_quarantines"]
            if (
                storage_incident is None
                or storage_quiescence is not None
                or quiescence["storage_incident_event_sequence"]
                != storage_incident["incident_event_sequence"]
                or quiescence["storage_incident_event_digest"]
                != storage_incident["incident_event_digest"]
                or not isinstance(primary, list)
                or not isinstance(auxiliary, list)
                or not isinstance(promotions, list)
                or any(
                    quiescence[name] is not True
                    for name in (
                        "all_primary_children_absent",
                        "all_auxiliary_children_absent",
                        "all_promotions_non_authoritative",
                    )
                )
                or any(
                    quiescence[name] is not False
                    for name in (
                        "solver_authority",
                        "wip_authority",
                        "cost_authority",
                        "promotion_authority",
                    )
                )
                or quiescence["status"] != "QUIESCED"
            ):
                raise SchedulerError(
                    "storage emergency quiescence is malformed"
                )
            expected_primary = {
                attempt_id
                for attempt_id, attempt in attempts.items()
                if attempt["phase"]
                in {
                    "PREPARED",
                    "BACKEND_PREPARED",
                    "RUNNING",
                    "DRAINING",
                    "EXITED",
                    "COLLECTED",
                    "COLLECTION_REJECTED",
                }
            }
            observed_primary: set[str] = set()
            containment_fields = {
                "containment_receipt_path",
                "containment_receipt_sha256",
                "launched_container_id",
                "attempt_container_absent",
                "controller_roles_absent",
                "arena_resources_absent",
                "rpc_endpoints_absent",
                "workspace_probe_containers_absent",
                "host_process_groups_absent",
                "containment_canaries_absent",
                "no_descendants",
            }
            for item in primary:
                if (
                    not isinstance(item, dict)
                    or set(item)
                    != {"attempt_id", "prior_phase", "containment"}
                    or item["attempt_id"] in observed_primary
                    or item["attempt_id"] not in expected_primary
                    or item["prior_phase"]
                    != attempts[item["attempt_id"]]["phase"]
                    or not isinstance(item["containment"], dict)
                    or set(item["containment"]) != containment_fields
                    or not isinstance(
                        item["containment"][
                            "containment_receipt_path"
                        ],
                        str,
                    )
                    or not _is_sha256(
                        item["containment"][
                            "containment_receipt_sha256"
                        ]
                    )
                    or (
                        item["containment"]["launched_container_id"]
                        is not None
                        and not isinstance(
                            item["containment"][
                                "launched_container_id"
                            ],
                            str,
                        )
                    )
                    or any(
                        item["containment"][name] is not True
                        for name in (
                            "attempt_container_absent",
                            "controller_roles_absent",
                            "arena_resources_absent",
                            "rpc_endpoints_absent",
                            "workspace_probe_containers_absent",
                            "host_process_groups_absent",
                            "containment_canaries_absent",
                            "no_descendants",
                        )
                    )
                ):
                    raise SchedulerError(
                        "storage primary containment is malformed"
                    )
                observed_primary.add(str(item["attempt_id"]))
            if observed_primary != expected_primary:
                raise SchedulerError(
                    "storage primary containment coverage is incomplete"
                )
            expected_auxiliary = {
                assignment_id
                for assignment_id, assignment
                in auxiliary_assignments.items()
                if assignment.phase in AUXILIARY_ACTIVE_PHASES
            }
            observed_auxiliary: set[str] = set()
            for item in auxiliary:
                if (
                    not isinstance(item, dict)
                    or set(item)
                    != {
                        "assignment_id",
                        "prior_phase",
                        "teardown_receipt_path",
                        "teardown_receipt_sha256",
                        "no_descendants",
                        "cost_authority",
                    }
                    or item["assignment_id"] in observed_auxiliary
                    or item["assignment_id"] not in expected_auxiliary
                    or item["prior_phase"]
                    != auxiliary_assignments[
                        item["assignment_id"]
                    ].phase
                    or item["no_descendants"] is not True
                    or item["cost_authority"] is not False
                    or (
                        item["prior_phase"] == "RUNNING"
                        and (
                            not isinstance(
                                item["teardown_receipt_path"], str
                            )
                            or not _is_sha256(
                                item["teardown_receipt_sha256"]
                            )
                        )
                    )
                    or (
                        item["prior_phase"] != "RUNNING"
                        and (
                            item["teardown_receipt_path"] is not None
                            or item["teardown_receipt_sha256"] is not None
                        )
                    )
                ):
                    raise SchedulerError(
                        "storage auxiliary containment is malformed"
                    )
                observed_auxiliary.add(str(item["assignment_id"]))
            if observed_auxiliary != expected_auxiliary:
                raise SchedulerError(
                    "storage auxiliary containment coverage is incomplete"
                )
            expected_promotions = {
                attempt_id
                for attempt_id, attempt in attempts.items()
                if attempt["phase"] == "PROMOTING"
            }
            observed_promotions: set[str] = set()
            for item in promotions:
                if (
                    not isinstance(item, dict)
                    or set(item)
                    != {
                        "attempt_id",
                        "external_commit_observed",
                        "external_commit_sha256",
                        "promotion_authority",
                    }
                    or item["attempt_id"] in observed_promotions
                    or item["attempt_id"] not in expected_promotions
                    or not isinstance(
                        item["external_commit_observed"], bool
                    )
                    or (
                        item["external_commit_observed"]
                        and not _is_sha256(
                            item["external_commit_sha256"]
                        )
                    )
                    or (
                        not item["external_commit_observed"]
                        and item["external_commit_sha256"] is not None
                    )
                    or item["promotion_authority"] is not False
                ):
                    raise SchedulerError(
                        "storage promotion quarantine is malformed"
                    )
                observed_promotions.add(str(item["attempt_id"]))
            if observed_promotions != expected_promotions:
                raise SchedulerError(
                    "storage promotion quarantine coverage is incomplete"
                )
            storage_quiescence = dict(quiescence)
            continue
        if kind == "FAILURE_CIRCUIT_FAILURE":
            circuit = _strict_keys(
                payload,
                {
                    "attempt_id",
                    "operation",
                    "fault_domain",
                    "operation_consecutive",
                    "operation_failure_index",
                    "domain_consecutive",
                    "domain_failure_index",
                    "backoff_seconds",
                    "retry_not_before",
                },
                "FAILURE_CIRCUIT_FAILURE",
            )
            operation = circuit["operation"]
            domain = circuit["fault_domain"]
            circuit_attempt_id = circuit["attempt_id"]
            if (
                not _is_identifier(operation)
                or domain not in FAILURE_FAULT_DOMAINS
                or (
                    circuit_attempt_id is not None
                    and not _is_identifier(circuit_attempt_id)
                )
            ):
                raise SchedulerError(
                    "failure circuit identity is malformed"
                )
            if operation == "substrate_health_reprobe" and (
                substrate_incident is None
                or substrate_incident[
                    "circuit_failure_recorded"
                ]
                is not False
                or circuit_attempt_id
                != substrate_incident["attempt_id"]
                or domain != "controller_substrate"
            ):
                raise SchedulerError(
                    "substrate circuit failure is duplicated or "
                    "unbound"
                )
            if operation == "backend_terminal":
                terminal_attempt = attempts.get(
                    str(circuit_attempt_id)
                )
                if (
                    terminal_attempt is None
                    or terminal_attempt["phase"] != "TORN_DOWN"
                    or terminal_attempt[
                        "terminal_failure_circuit_recorded"
                    ]
                ):
                    raise SchedulerError(
                        "terminal failure circuit is duplicated or "
                        "out of lifecycle order"
                    )
                terminal_attempt[
                    "terminal_failure_circuit_recorded"
                ] = True
            operation_key = f"{operation}:{domain}"
            operation_state = failure_operation_circuits.get(
                operation_key,
                {
                    "consecutive": 0,
                    "failure_index": 0,
                    "retry_not_before": None,
                },
            )
            domain_state = failure_domain_circuits.get(
                str(domain),
                {
                    "consecutive": 0,
                    "failure_index": 0,
                    "retry_not_before": None,
                    "last_operation": None,
                },
            )
            operation_consecutive = (
                int(operation_state["consecutive"]) + 1
            )
            domain_consecutive = (
                int(domain_state["consecutive"]) + 1
            )
            backoff_schedule = (
                SUBSTRATE_HEALTH_REPROBE_BACKOFF_SECONDS
                if operation == "substrate_health_reprobe"
                else OPERATION_RETRY_BACKOFF_SECONDS
            )
            expected_backoff = backoff_schedule[
                min(
                    max(
                        operation_consecutive,
                        domain_consecutive,
                    ),
                    len(backoff_schedule),
                )
                - 1
            ]
            if (
                circuit["operation_consecutive"]
                != operation_consecutive
                or circuit["operation_failure_index"]
                != int(operation_state["failure_index"]) + 1
                or circuit["domain_consecutive"]
                != domain_consecutive
                or circuit["domain_failure_index"]
                != int(domain_state["failure_index"]) + 1
                or circuit["backoff_seconds"] != expected_backoff
                or circuit["retry_not_before"]
                != float(event["recorded_at"]) + expected_backoff
            ):
                raise SchedulerError(
                    "failure circuit count/backoff is noncanonical"
                )
            failure_operation_circuits[operation_key] = {
                "consecutive": operation_consecutive,
                "failure_index":
                    circuit["operation_failure_index"],
                "retry_not_before": circuit["retry_not_before"],
            }
            failure_domain_circuits[str(domain)] = {
                "consecutive": domain_consecutive,
                "failure_index": circuit["domain_failure_index"],
                "retry_not_before": circuit["retry_not_before"],
                "last_operation": operation,
            }
            if operation == "substrate_health_reprobe":
                substrate_incident[
                    "circuit_failure_recorded"
                ] = True
            continue
        if kind == "FAILURE_CIRCUIT_RESET":
            reset = _strict_keys(
                payload,
                {
                    "attempt_id",
                    "operation",
                    "fault_domain",
                    "operation_consecutive",
                    "domain_consecutive",
                    "evidence_kind",
                    "reset_operation",
                    "reset_domain",
                },
                "FAILURE_CIRCUIT_RESET",
            )
            operation = reset["operation"]
            domain = reset["fault_domain"]
            circuit_attempt_id = reset["attempt_id"]
            operation_key = f"{operation}:{domain}"
            operation_state = failure_operation_circuits.get(
                operation_key
            )
            domain_state = failure_domain_circuits.get(str(domain))
            evidence_index = int(event["sequence"]) - 2
            while (
                evidence_index >= 0
                and events[evidence_index]["kind"]
                == "FAILURE_CIRCUIT_RESET"
            ):
                evidence_index -= 1
            evidence_event = (
                events[evidence_index]
                if evidence_index >= 0
                else None
            )
            evidence_contract = {
                "attempt_prepared": (
                    {"ATTEMPT_PREPARED"},
                    "attempt_id",
                    {"input_materialize"},
                ),
                "backend_prepared": (
                    {"BACKEND_PREPARED"},
                    "attempt_id",
                    {"backend_prepare"},
                ),
                "attempt_launched": (
                    {"ATTEMPT_LAUNCHED"},
                    "attempt_id",
                    {"backend_launch"},
                ),
                "backend_poll_observation": (
                    {"ATTEMPT_OBSERVED", "ATTEMPT_EXITED"},
                    "attempt_id",
                    {"backend_poll"},
                ),
                "attempt_collected": (
                    {"ATTEMPT_COLLECTED"},
                    "attempt_id",
                    {"backend_collect"},
                ),
                "attempt_torn_down": (
                    {"ATTEMPT_TORN_DOWN"},
                    "attempt_id",
                    {"backend_teardown"},
                ),
                "promotion_committed": (
                    {"PROMOTION_COMMITTED"},
                    "attempt_id",
                    {"promotion_commit", "promotion_recover"},
                ),
                "substrate_health_restored": (
                    {"SUBSTRATE_HEALTH_RESTORED"},
                    "attempt_id",
                    {"substrate_health_reprobe"},
                ),
                "auxiliary_input_prepared": (
                    {"AUXILIARY_INPUT_PREPARED"},
                    "assignment_id",
                    {"auxiliary_prepare"},
                ),
                "auxiliary_launched": (
                    {"AUXILIARY_LAUNCHED"},
                    "assignment_id",
                    {"auxiliary_launch"},
                ),
                "auxiliary_result_quarantined": (
                    {"AUXILIARY_RESULT_QUARANTINED"},
                    "assignment_id",
                    {"auxiliary_collect", "auxiliary_teardown"},
                ),
                "auxiliary_output_rejected": (
                    {"AUXILIARY_OUTPUT_REJECTED"},
                    "assignment_id",
                    {"auxiliary_admit"},
                ),
                "auxiliary_output_admitted": (
                    {
                        "AUXILIARY_OUTPUT_ADMITTED",
                        "AUXILIARY_PROFILE_ADMITTED",
                    },
                    "assignment_id",
                    {"auxiliary_admit"},
                ),
            }
            evidence_rule = evidence_contract.get(
                reset["evidence_kind"]
            )
            if (
                not _is_identifier(operation)
                or domain not in FAILURE_FAULT_DOMAINS
                or (
                    circuit_attempt_id is not None
                    and not _is_identifier(circuit_attempt_id)
                )
                or not _is_identifier(reset["evidence_kind"])
                or evidence_event is None
                or evidence_rule is None
                or evidence_event["kind"] not in evidence_rule[0]
                or evidence_event["payload"].get(
                    evidence_rule[1]
                )
                != circuit_attempt_id
                or operation not in evidence_rule[2]
                or not isinstance(reset["reset_operation"], bool)
                or not isinstance(reset["reset_domain"], bool)
                or not (
                    reset["reset_operation"]
                    or reset["reset_domain"]
                )
                or operation_state is None
                or domain_state is None
                or (
                    reset["reset_operation"]
                    and int(operation_state["consecutive"]) == 0
                )
                or (
                    reset["reset_domain"]
                    and int(domain_state["consecutive"]) == 0
                )
                or (
                    reset["reset_domain"]
                    and domain_state.get("last_operation")
                    != operation
                )
                or reset["operation_consecutive"]
                != operation_state["consecutive"]
                or reset["domain_consecutive"]
                != domain_state["consecutive"]
            ):
                raise SchedulerError(
                    "failure circuit reset lacks matching success"
                )
            failure_operation_circuits[operation_key] = {
                **operation_state,
                "consecutive": (
                    0
                    if reset["reset_operation"]
                    else operation_state["consecutive"]
                ),
                "retry_not_before": (
                    None
                    if reset["reset_operation"]
                    else operation_state["retry_not_before"]
                ),
            }
            failure_domain_circuits[str(domain)] = {
                **domain_state,
                "consecutive": (
                    0
                    if reset["reset_domain"]
                    else domain_state["consecutive"]
                ),
                "retry_not_before": (
                    None
                    if reset["reset_domain"]
                    else domain_state["retry_not_before"]
                ),
                "last_operation": (
                    None
                    if reset["reset_domain"]
                    else domain_state["last_operation"]
                ),
            }
            continue
        if kind == "OPERATOR_INCIDENT":
            incident = _strict_keys(
                payload,
                {
                    "attempt_id",
                    "operation",
                    "fault_domain",
                    "operation_consecutive",
                    "domain_consecutive",
                    "threshold",
                    "reason_code",
                },
                "OPERATOR_INCIDENT",
            )
            operation = incident["operation"]
            domain = incident["fault_domain"]
            operation_state = failure_operation_circuits.get(
                f"{operation}:{domain}"
            )
            domain_state = failure_domain_circuits.get(str(domain))
            deterministic_substrate_incident = (
                operation == "substrate_health_reprobe"
                and domain == "controller_substrate"
                and incident["threshold"] == 2
                and incident["reason_code"]
                == "deterministic_substrate_configuration_repeated"
                and substrate_incident is not None
                and incident["attempt_id"]
                == substrate_incident["attempt_id"]
                and substrate_incident["failure_class"]
                == "DETERMINISTIC_CONFIGURATION"
                and substrate_incident["health_probe_count"] == 1
                and isinstance(
                    substrate_incident["last_health_probe"], dict
                )
                and substrate_incident[
                    "last_health_probe"
                ].get("failure_class")
                == "DETERMINISTIC_CONFIGURATION"
                and substrate_incident[
                    "last_health_probe"
                ].get("failure_code")
                == substrate_incident["failure_code"]
                and incident["operation_consecutive"] == 2
                and incident["domain_consecutive"] == 2
            )
            if (
                operator_incident is not None
                or not _is_identifier(operation)
                or domain not in FAILURE_FAULT_DOMAINS
                or (
                    incident["attempt_id"] is not None
                    and not _is_identifier(incident["attempt_id"])
                )
                or (
                    not deterministic_substrate_incident
                    and (
                        incident["threshold"]
                        != FAILURE_CIRCUIT_THRESHOLD
                        or incident["reason_code"]
                        != "failure_circuit_exhausted"
                        or operation_state is None
                        or domain_state is None
                        or incident["operation_consecutive"]
                        != operation_state["consecutive"]
                        or incident["domain_consecutive"]
                        != domain_state["consecutive"]
                        or max(
                            int(operation_state["consecutive"]),
                            int(domain_state["consecutive"]),
                        )
                        < FAILURE_CIRCUIT_THRESHOLD
                    )
                )
            ):
                raise SchedulerError(
                    "operator incident lacks exhausted circuit"
                )
            operator_incident = dict(incident)
            continue
        if kind == "SCHEDULER_DECISION":
            if (
                pending_auxiliary is not None
                or substrate_incident is not None
            ):
                raise SchedulerError(
                    "proposer decision overlaps pending work or an "
                    "unhealthy substrate"
                )
            decision_payload = _strict_keys(
                payload, {"decision"}, "SCHEDULER_DECISION"
            )
            decision = decision_from_dict(decision_payload["decision"])
            if (
                decision.decision_id in used_decision_ids
                or decision.attempt_id in used_attempt_ids
                or decision.generation_id in used_generation_ids
                or decision.reservation_id in used_reservation_ids
            ):
                raise SchedulerError(
                    "scheduler reused a decision/attempt/generation/"
                    "reservation identity"
                )
            _validate_decision_against_state(
                decision,
                event=event,
                genesis=genesis,
                lanes=lanes,
                budget=budget,
                auxiliary_assignments=tuple(
                    auxiliary_assignments.values()
                ),
                complexity_rounds=complexity_rounds,
                sidecar_requests=sidecar_requests,
            )
            used_decision_ids.add(decision.decision_id)
            used_attempt_ids.add(decision.attempt_id)
            used_generation_ids.add(decision.generation_id)
            used_reservation_ids.add(decision.reservation_id)
            pending = (decision, event)
            decisions += 1
            continue
        if kind == "AUXILIARY_DECISION":
            if (
                pending is not None
                or pending_auxiliary is not None
                or substrate_incident is not None
            ):
                raise SchedulerError(
                    "auxiliary decision overlaps a pending decision"
                )
            decision_payload = _strict_keys(
                payload, {"decision"}, "AUXILIARY_DECISION"
            )
            auxiliary_decision = auxiliary_decision_from_dict(
                decision_payload["decision"]
            )
            identity_values = {
                auxiliary_decision.decision_id,
                auxiliary_decision.assignment_id,
                auxiliary_decision.reservation_id,
                auxiliary_decision.expert_id,
            }
            if (
                identity_values
                & (
                    used_decision_ids
                    | used_attempt_ids
                    | used_generation_ids
                    | used_reservation_ids
                    | used_expert_ids
                )
                or auxiliary_decision.thread_id in used_thread_ids
            ):
                raise SchedulerError(
                    "auxiliary decision reuses an identity or expert thread"
                )
            snapshot = _snapshot_from_audit_state(
                event=event,
                genesis=genesis,
                lanes=lanes,
                budget=budget,
                auxiliary_assignments=tuple(
                    auxiliary_assignments.values()
                ),
                complexity_rounds=complexity_rounds,
                sidecar_requests=sidecar_requests,
            )
            verify_auxiliary_decision(
                snapshot,
                auxiliary_decision,
                launch_configuration=auxiliary_configuration,
            )
            used_decision_ids.add(auxiliary_decision.decision_id)
            used_attempt_ids.add(auxiliary_decision.assignment_id)
            used_reservation_ids.add(
                auxiliary_decision.reservation_id
            )
            used_expert_ids.add(auxiliary_decision.expert_id)
            used_thread_ids.add(auxiliary_decision.thread_id)
            pending_auxiliary = (auxiliary_decision, event)
            auxiliary_decisions += 1
            continue
        if kind == "AUXILIARY_RESERVED":
            if pending_auxiliary is None:
                raise SchedulerError(
                    "AUXILIARY_RESERVED has no preceding auxiliary decision"
                )
            auxiliary_decision, _ = pending_auxiliary
            reserved_payload = _strict_keys(
                payload,
                {"assignment_id", "reservation"},
                "AUXILIARY_RESERVED",
            )
            if (
                reserved_payload["assignment_id"]
                != auxiliary_decision.assignment_id
                or reserved_payload["reservation"]
                != auxiliary_reservation_projection(auxiliary_decision)
            ):
                raise SchedulerError(
                    "auxiliary reservation does not consume the exact decision"
                )
            lane = lanes[auxiliary_decision.game]
            if (
                lane["active_attempt_id"]
                != auxiliary_decision.active_proposer_attempt_id
                or lane["frontier_sha256"]
                != auxiliary_decision.frontier_sha256
            ):
                raise SchedulerError(
                    "auxiliary reservation no longer has its live max proposer"
                )
            budget = reserve_budget(
                budget,
                reservation_id=auxiliary_decision.reservation_id,
                attempt_id=auxiliary_decision.assignment_id,
                units=auxiliary_decision.reservation_units,
            )
            assignment = AuxiliaryAssignmentState(
                schema=1,
                assignment_id=auxiliary_decision.assignment_id,
                decision_id=auxiliary_decision.decision_id,
                reservation_id=auxiliary_decision.reservation_id,
                game=auxiliary_decision.game,
                frontier_sha256=auxiliary_decision.frontier_sha256,
                parent_checkpoint_sha256=(
                    auxiliary_decision.parent_checkpoint_sha256
                ),
                trigger_no_progress=auxiliary_decision.no_progress,
                trigger_history_sha256=(
                    auxiliary_decision.trigger_history_sha256
                ),
                profile_id=auxiliary_decision.profile_id,
                round_index=auxiliary_decision.round_index,
                specialization=auxiliary_decision.specialization,
                expert_id=auxiliary_decision.expert_id,
                thread_id=auxiliary_decision.thread_id,
                active_proposer_attempt_id=(
                    auxiliary_decision.active_proposer_attempt_id
                ),
                input_manifest=auxiliary_decision.input_manifest,
                input_manifest_sha256=(
                    auxiliary_decision.input_manifest_sha256
                ),
                observation_ledger_sha256=(
                    auxiliary_decision.observation_ledger_sha256
                ),
                model=auxiliary_decision.model,
                reasoning_effort=(
                    auxiliary_decision.reasoning_effort
                ),
                role=auxiliary_decision.role,
                context_limit_tokens=(
                    auxiliary_decision.context_limit_tokens
                ),
                role_max_concurrency=(
                    auxiliary_decision.role_max_concurrency
                ),
                supervisory_launch_configuration_sha256=(
                    auxiliary_decision
                    .supervisory_launch_configuration_sha256
                ),
                sidecar_request=auxiliary_decision.sidecar_request,
                sidecar_request_sha256=(
                    auxiliary_decision.sidecar_request_sha256
                ),
                phase="RESERVED",
            )
            validate_auxiliary_assignment(assignment)
            auxiliary_assignments[assignment.assignment_id] = assignment
            active_total = sum(
                item["active_attempt_id"] is not None
                for item in lanes.values()
            ) + sum(
                item.phase in AUXILIARY_ACTIVE_PHASES
                for item in auxiliary_assignments.values()
            )
            if active_total > genesis["max_lanes"]:
                raise SchedulerError(
                    "auxiliary reservation exceeds total lane capacity"
                )
            pending_auxiliary = None
            auxiliary_reservations += 1
            continue
        if kind == "ATTEMPT_RESERVED":
            if pending is None:
                raise SchedulerError(
                    "ATTEMPT_RESERVED has no preceding scheduler decision"
                )
            decision, _ = pending
            binding = _reservation_scheduler_binding(payload)
            if binding != reservation_binding(decision):
                raise SchedulerError(
                    "reservation does not consume the exact scheduler decision"
                )
            reservation = payload.get("reservation")
            if not isinstance(reservation, dict):
                raise SchedulerError("attempt reservation body is missing")
            lane = lanes[decision.choice.game]
            expected_cost_remaining = (
                None
                if decision.choice.reservation_units is None
                else (
                    Decimal(decision.choice.reservation_units)
                    / Decimal(COST_SCALE)
                )
            )
            raw_cost_remaining = reservation.get(
                "cost_limit_remaining"
            )
            if expected_cost_remaining is None:
                cost_binding_valid = raw_cost_remaining is None
            else:
                try:
                    cost_binding_valid = (
                        limit_to_units(raw_cost_remaining)
                        == decision.choice.reservation_units
                        and charge_to_units(raw_cost_remaining)
                        == decision.choice.reservation_units
                    )
                except SchedulerError:
                    cost_binding_valid = False
            selected_wip = decision.choice.selected_wip
            reservation_wip = reservation.get("wip")
            if selected_wip is None:
                wip_binding_valid = (
                    reservation_wip is None
                    and reservation.get("resume_thread_id") is None
                    and reservation.get(
                        "resume_thread_binding_sha256"
                    ) is None
                )
            else:
                try:
                    projected_wip = _wip_projection_from_result(
                        reservation_wip
                    )
                except SchedulerError:
                    projected_wip = None
                wip_binding_valid = (
                    projected_wip == asdict(selected_wip)
                    and reservation.get("resume_thread_id")
                    == selected_wip.codex_thread_id
                    and reservation.get(
                        "resume_thread_binding_sha256"
                    )
                    == selected_wip.final_thread_binding_sha256
                )
            if (
                payload.get("attempt_id") != decision.attempt_id
                or reservation.get("campaign_id")
                != decision.campaign_id
                or reservation.get("attempt_id")
                != decision.attempt_id
                or reservation.get("generation_id")
                != decision.generation_id
                or reservation.get("game") != decision.choice.game
                or reservation.get("target_level")
                != decision.choice.target_level
                or reservation.get("authoritative_target")
                != decision.choice.authoritative_target
                or reservation.get("effort") != decision.choice.effort
                or reservation.get("soft_allocation_seconds")
                != decision.choice.soft_allocation_seconds
                or reservation.get("wip_mode")
                != decision.choice.effective_wip_mode
                or reservation.get("thread_mode")
                != decision.choice.thread_mode
                or reservation.get("parent_checkpoint_path")
                != lane["parent_checkpoint_path"]
                or reservation.get("parent_checkpoint_sha256")
                != lane["parent_checkpoint_sha256"]
                or reservation.get("frontier_sha256")
                != lane["frontier_sha256"]
                or reservation.get("parent_source_path")
                != lane["parent_source_path"]
                or reservation.get("parent_source_tree_sha256")
                != lane["parent_source_tree_sha256"]
                or not cost_binding_valid
                or not wip_binding_valid
            ):
                raise SchedulerError(
                    "attempt reservation differs from scheduler choice"
                )
            if lane["active_attempt_id"] is not None:
                raise SchedulerError("same game has overlapping attempts")
            budget = reserve_budget(
                budget,
                reservation_id=decision.reservation_id,
                attempt_id=decision.attempt_id,
                units=decision.choice.reservation_units,
            )
            lane["active_attempt_id"] = decision.attempt_id
            lane["last_dispatch_sequence"] = event["sequence"]
            if sum(
                item["active_attempt_id"] is not None
                for item in lanes.values()
            ) + sum(
                item.phase in AUXILIARY_ACTIVE_PHASES
                for item in auxiliary_assignments.values()
            ) > genesis["max_lanes"]:
                raise SchedulerError("reservation exceeds lane capacity")
            attempts[decision.attempt_id] = {
                "attempt_id": decision.attempt_id,
                "generation_id": decision.generation_id,
                "game": decision.choice.game,
                "target_level": decision.choice.target_level,
                "authoritative_target":
                    decision.choice.authoritative_target,
                "parent_checkpoint_sha256":
                    lane["parent_checkpoint_sha256"],
                "reservation_id": decision.reservation_id,
                "decision_id": decision.decision_id,
                "frontier_sha256": lane["frontier_sha256"],
                "generation_dir": reservation.get("generation_dir"),
                "host_transcript_path":
                    reservation.get("host_transcript_path"),
                "no_progress_before": decision.choice.no_progress,
                "effort": decision.choice.effort,
                "soft_allocation_seconds":
                    decision.choice.soft_allocation_seconds,
                "requested_wip_mode":
                    decision.choice.requested_wip_mode,
                "supervisory_handoff_sha256": (
                    decision.choice.selected_supervisory_handoff
                    .supervisory_handoff_sha256
                    if decision.choice.selected_supervisory_handoff
                    is not None
                    else None
                ),
                "candidate": False,
                "candidate_evidence": None,
                "native_sidecar_request_draft": None,
                "public_observation_transition": None,
                "public_observation_receipt_sha256s": (),
                "collection_result_kind": None,
                "terminal_status": None,
                "structured_provider_outcome": None,
                "typed_teardown": False,
                "retry_count": 0,
                "operation_retry_counts": {},
                "operation_retry_not_before": {},
                "terminal_failure_circuit_recorded": False,
                "promoted": False,
                "settled": False,
                "phase": "RESERVED",
            }
            pending = None
            reservations += 1
            continue
        if kind == "AUXILIARY_INPUT_PREPARED":
            prepared_payload = _strict_keys(
                payload,
                {
                    "assignment_id",
                    "input_manifest_path",
                    "input_manifest_sha256",
                    "input_bundle_receipt_path",
                    "input_bundle_receipt_sha256",
                },
                "AUXILIARY_INPUT_PREPARED",
            )
            assignment_id = prepared_payload["assignment_id"]
            assignment = auxiliary_assignments.get(str(assignment_id))
            if assignment is None or assignment.phase != "RESERVED":
                raise SchedulerError(
                    "auxiliary input preparation targets the wrong phase"
                )
            _verify_exact_auxiliary_receipt(
                prepared_payload["input_manifest_path"],
                prepared_payload["input_manifest_sha256"],
                expected=asdict(assignment.input_manifest),
                label="materialized auxiliary input manifest",
            )
            if (
                prepared_payload["input_manifest_sha256"]
                != assignment.input_manifest_sha256
            ):
                raise SchedulerError(
                    "materialized auxiliary input manifest does not match "
                    "the pre-reservation decision commitment"
                )
            expected_receipt = {
                "schema": 1,
                "kind": "auxiliary_private_input_bundle",
                "assignment_id": assignment.assignment_id,
                "frontier_sha256": assignment.frontier_sha256,
                "parent_checkpoint_sha256":
                    assignment.parent_checkpoint_sha256,
                "input_manifest_sha256":
                    assignment.input_manifest_sha256,
                "observation_ledger_sha256":
                    assignment.observation_ledger_sha256,
                "input_bundle_contract_sha256":
                    auxiliary_configuration
                    .input_bundle_contract_sha256,
                "immutable_inputs": True,
                "live_lineage_mounted": False,
                "public_observations_only": True,
            }
            _verify_exact_auxiliary_receipt(
                prepared_payload["input_bundle_receipt_path"],
                prepared_payload["input_bundle_receipt_sha256"],
                expected=expected_receipt,
                label="auxiliary private input bundle receipt",
            )
            auxiliary_assignments[assignment.assignment_id] = replace(
                assignment, phase="INPUT_PREPARED"
            )
            continue
        if kind == "AUXILIARY_LAUNCHED":
            launched_payload = _strict_keys(
                payload,
                {
                    "assignment_id",
                    "launch_receipt_path",
                    "launch_receipt_sha256",
                },
                "AUXILIARY_LAUNCHED",
            )
            assignment_id = launched_payload["assignment_id"]
            assignment = auxiliary_assignments.get(str(assignment_id))
            if assignment is None or assignment.phase != "INPUT_PREPARED":
                raise SchedulerError(
                    "auxiliary launch targets the wrong phase"
                )
            expected_receipt = {
                "schema": 1,
                "kind": "auxiliary_backend_launch",
                "assignment_id": assignment.assignment_id,
                "backend_contract_sha256":
                    auxiliary_configuration.backend_contract_sha256,
                "expert_id": assignment.expert_id,
                "thread_id": assignment.thread_id,
                "model": assignment.model,
                "reasoning_effort": assignment.reasoning_effort,
                "fresh_context": True,
                "live_lineage_write_authority": False,
            }
            _verify_exact_auxiliary_receipt(
                launched_payload["launch_receipt_path"],
                launched_payload["launch_receipt_sha256"],
                expected=expected_receipt,
                label="auxiliary backend launch receipt",
            )
            auxiliary_assignments[assignment.assignment_id] = replace(
                assignment, phase="RUNNING"
            )
            continue
        if kind == "AUXILIARY_ABORTED":
            aborted_payload = _strict_keys(
                payload,
                {
                    "assignment_id",
                    "prior_phase",
                    "reason",
                    "cost_used",
                    "authenticated_cost_units",
                    "budget_reservation_id",
                    "auxiliary_decision_id",
                    "abort_receipt_path",
                    "abort_receipt_sha256",
                    "teardown_receipt_path",
                    "teardown_receipt_sha256",
                },
                "AUXILIARY_ABORTED",
            )
            assignment_id = aborted_payload["assignment_id"]
            assignment = auxiliary_assignments.get(str(assignment_id))
            if (
                assignment is None
                or assignment.phase not in AUXILIARY_ACTIVE_PHASES
                or aborted_payload["prior_phase"] != assignment.phase
            ):
                raise SchedulerError(
                    "auxiliary abort targets the wrong assignment phase"
                )
            reason = _require_bounded_text(
                aborted_payload["reason"], "auxiliary abort reason"
            )
            if assignment.invalidated and reason != "frontier_promoted":
                raise SchedulerError(
                    "invalidated auxiliary assignment must record promotion "
                    "as its abort cause"
                )
            charged_units = aborted_payload["authenticated_cost_units"]
            try:
                derived_units = charge_to_units(
                    aborted_payload["cost_used"]
                )
            except SchedulerError:
                derived_units = None
            if (
                not _is_int(charged_units)
                or charged_units != derived_units
                or aborted_payload["budget_reservation_id"]
                != assignment.reservation_id
                or aborted_payload["auxiliary_decision_id"]
                != assignment.decision_id
            ):
                raise SchedulerError(
                    "auxiliary abort lacks exact usage settlement"
                )
            teardown_expected = {
                "schema": 1,
                "kind": "auxiliary_backend_abort_teardown",
                "assignment_id": assignment.assignment_id,
                "backend_contract_sha256":
                    auxiliary_configuration.backend_contract_sha256,
                "prior_phase": assignment.phase,
                "descendants_absent": True,
                "live_lineage_mutated": False,
            }
            if assignment.phase == "RUNNING":
                _verify_exact_auxiliary_receipt(
                    aborted_payload["teardown_receipt_path"],
                    aborted_payload["teardown_receipt_sha256"],
                    expected=teardown_expected,
                    label="auxiliary abort teardown receipt",
                )
            elif (
                aborted_payload["teardown_receipt_path"] is not None
                or aborted_payload["teardown_receipt_sha256"] is not None
            ):
                raise SchedulerError(
                    "unlaunched auxiliary abort claims a teardown receipt"
                )
            abort_expected = {
                "schema": 1,
                "kind": "auxiliary_assignment_abort",
                "authority": "host_only",
                "assignment_id": assignment.assignment_id,
                "frontier_sha256": assignment.frontier_sha256,
                "parent_checkpoint_sha256":
                    assignment.parent_checkpoint_sha256,
                "prior_phase": assignment.phase,
                "reason": reason,
                "invalidated": assignment.invalidated,
                "backend_contract_sha256":
                    auxiliary_configuration.backend_contract_sha256,
                "teardown_receipt_sha256": (
                    aborted_payload["teardown_receipt_sha256"]
                    if assignment.phase == "RUNNING"
                    else None
                ),
                "verdict": "ABORTED",
            }
            _verify_exact_auxiliary_receipt(
                aborted_payload["abort_receipt_path"],
                aborted_payload["abort_receipt_sha256"],
                expected=abort_expected,
                label="auxiliary host abort receipt",
            )
            budget = settle_budget(
                budget,
                reservation_id=assignment.reservation_id,
                attempt_id=assignment.assignment_id,
                charged_units=int(charged_units),
            )
            auxiliary_assignments[assignment.assignment_id] = replace(
                assignment, phase="ABORTED"
            )
            auxiliary_settlements += 1
            continue
        if kind == "AUXILIARY_RESULT_QUARANTINED":
            result_payload = _strict_keys(
                payload,
                {
                    "assignment_id",
                    "output",
                    "cost_used",
                    "authenticated_cost_units",
                    "budget_reservation_id",
                    "auxiliary_decision_id",
                    "teardown_receipt_path",
                    "teardown_receipt_sha256",
                },
                "AUXILIARY_RESULT_QUARANTINED",
            )
            assignment_id = result_payload["assignment_id"]
            assignment = auxiliary_assignments.get(str(assignment_id))
            if assignment is None or assignment.phase != "RUNNING":
                raise SchedulerError(
                    "auxiliary result targets the wrong phase"
                )
            output = auxiliary_output_from_dict(
                result_payload["output"], assignment=assignment
            )
            charged_units = result_payload["authenticated_cost_units"]
            try:
                derived_units = charge_to_units(
                    result_payload["cost_used"]
                )
            except SchedulerError:
                derived_units = None
            if (
                not _is_int(charged_units)
                or charged_units != derived_units
                or result_payload["budget_reservation_id"]
                != assignment.reservation_id
                or result_payload["auxiliary_decision_id"]
                != assignment.decision_id
            ):
                raise SchedulerError(
                    "auxiliary result lacks exact usage settlement"
                )
            teardown_expected = {
                "schema": 1,
                "kind": "auxiliary_backend_teardown",
                "assignment_id": assignment.assignment_id,
                "backend_contract_sha256":
                    auxiliary_configuration.backend_contract_sha256,
                "output_manifest_sha256":
                    output.output_manifest_sha256,
                "descendants_absent": True,
                "live_lineage_mutated": False,
            }
            _verify_exact_auxiliary_receipt(
                result_payload["teardown_receipt_path"],
                result_payload["teardown_receipt_sha256"],
                expected=teardown_expected,
                label="auxiliary backend teardown receipt",
            )
            budget = settle_budget(
                budget,
                reservation_id=assignment.reservation_id,
                attempt_id=assignment.assignment_id,
                charged_units=int(charged_units),
            )
            auxiliary_assignments[assignment.assignment_id] = replace(
                assignment, phase="QUARANTINED", output=output
            )
            auxiliary_settlements += 1
            continue
        if kind == "AUXILIARY_PROFILE_ADMITTED":
            admission_payload = _strict_keys(
                payload,
                {
                    "assignment_id",
                    "profile",
                    "admitted_evidence_sha256",
                    "fresh_replay_receipt_path",
                    "fresh_replay_receipt_sha256",
                    "taint_receipt_path",
                    "taint_receipt_sha256",
                    "provenance_receipt_path",
                    "provenance_receipt_sha256",
                    "admission_receipt_path",
                    "admission_receipt_sha256",
                },
                "AUXILIARY_PROFILE_ADMITTED",
            )
            assignment_id = admission_payload["assignment_id"]
            assignment = auxiliary_assignments.get(str(assignment_id))
            if (
                assignment is None
                or assignment.phase != "QUARANTINED"
                or assignment.specialization != "complexity_diagnosis"
                or assignment.invalidated
                or assignment.output is None
            ):
                raise SchedulerError(
                    "complexity profile does not consume a current "
                    "quarantined diagnosis"
                )
            lane = lanes[assignment.game]
            if (
                lane["frontier_sha256"] != assignment.frontier_sha256
                or lane["parent_checkpoint_sha256"]
                != assignment.parent_checkpoint_sha256
            ):
                raise SchedulerError(
                    "complexity profile is stale after frontier change"
                )
            profile = _complexity_profile_from_dict(
                admission_payload["profile"]
            )
            if (
                profile.frontier_sha256 != assignment.frontier_sha256
                or profile.round_index != assignment.round_index
                or profile.observation_receipt_sha256
                not in assignment.output
                .public_observation_receipt_sha256s
                or profile.taint_scan_receipt_sha256
                != admission_payload["taint_receipt_sha256"]
            ):
                raise SchedulerError(
                    "complexity profile substitutes its exact round"
                )
            admitted_profile_sha = sha256_json(asdict(profile))
            if (
                admission_payload["admitted_evidence_sha256"]
                != admitted_profile_sha
            ):
                raise SchedulerError(
                    "profile admission evidence hash is not the profile"
                )
            _verify_auxiliary_admission_gates(
                admission_payload,
                assignment=assignment,
                admitted_evidence_sha256=admitted_profile_sha,
                admission_contract_sha256=str(
                    auxiliary_configuration.admission_contract_sha256
                ),
                admission_kind="auxiliary_profile_admission",
            )
            prior_rounds = [
                item
                for item in complexity_rounds
                if item.game == assignment.game
                and item.frontier_sha256 == assignment.frontier_sha256
                and not item.invalidated
            ]
            if (
                assignment.round_index != len(prior_rounds)
                or (
                    prior_rounds
                    and assignment.trigger_no_progress
                    < prior_rounds[-1].trigger_no_progress + 2
                )
            ):
                raise SchedulerError(
                    "complexity re-diagnosis lacks a fresh clean pair"
                )
            if prior_rounds:
                previous = prior_rounds[-1]
                completed = {
                    item.specialization
                    for item in auxiliary_assignments.values()
                    if item.profile_id
                    == previous.profile.profile_id
                    and item.phase
                    in {"QUARANTINED", "ADMITTED", "REJECTED"}
                }
                if completed != set(previous.profile.priorities):
                    raise SchedulerError(
                        "complexity re-diagnosis precedes obligation "
                        "exhaustion"
                    )
            round_state = ComplexityRoundState(
                schema=1,
                game=assignment.game,
                frontier_sha256=assignment.frontier_sha256,
                parent_checkpoint_sha256=(
                    assignment.parent_checkpoint_sha256
                ),
                parent_source_tree_sha256=str(
                    lane["parent_source_tree_sha256"]
                ),
                round_index=assignment.round_index,
                profile=profile,
                diagnosis_assignment_id=assignment.assignment_id,
                trigger_no_progress=assignment.trigger_no_progress,
                trigger_history_sha256=(
                    assignment.trigger_history_sha256
                ),
                input_manifest_sha256=(
                    assignment.input_manifest_sha256
                ),
                observation_ledger_sha256=(
                    assignment.observation_ledger_sha256
                ),
                admission_receipt_path=str(
                    admission_payload["admission_receipt_path"]
                ),
                admission_receipt_sha256=str(
                    admission_payload["admission_receipt_sha256"]
                ),
                admitted_sequence=int(event["sequence"]),
                admitted_event_digest=str(event["digest"]),
            )
            validate_complexity_round(round_state)
            complexity_rounds.append(round_state)
            auxiliary_assignments[assignment.assignment_id] = replace(
                assignment,
                phase="ADMITTED",
                admission_receipt_path=str(
                    admission_payload["admission_receipt_path"]
                ),
                admission_receipt_sha256=str(
                    admission_payload["admission_receipt_sha256"]
                ),
                admitted_sequence=int(event["sequence"]),
                admitted_event_digest=str(event["digest"]),
            )
            auxiliary_admissions += 1
            continue
        if kind == "AUXILIARY_OUTPUT_ADMITTED":
            admission_payload = _strict_keys(
                payload,
                {
                    "assignment_id",
                    "admitted_evidence_sha256",
                    "fresh_replay_receipt_path",
                    "fresh_replay_receipt_sha256",
                    "taint_receipt_path",
                    "taint_receipt_sha256",
                    "provenance_receipt_path",
                    "provenance_receipt_sha256",
                    "admission_receipt_path",
                    "admission_receipt_sha256",
                },
                "AUXILIARY_OUTPUT_ADMITTED",
            )
            assignment_id = admission_payload["assignment_id"]
            assignment = auxiliary_assignments.get(str(assignment_id))
            if (
                assignment is None
                or assignment.phase != "QUARANTINED"
                or assignment.specialization == "complexity_diagnosis"
                or assignment.invalidated
                or assignment.output is None
                or lanes[assignment.game]["frontier_sha256"]
                != assignment.frontier_sha256
            ):
                raise SchedulerError(
                    "auxiliary admission is stale or lacks quarantined output"
                )
            admitted_sha = admission_payload[
                "admitted_evidence_sha256"
            ]
            if (
                not _is_sha256(admitted_sha)
                or admitted_sha != sha256_json(asdict(assignment.output))
            ):
                raise SchedulerError(
                    "admitted auxiliary evidence hash is not the exact "
                    "quarantined output"
                )
            _verify_auxiliary_admission_gates(
                admission_payload,
                assignment=assignment,
                admitted_evidence_sha256=str(admitted_sha),
                admission_contract_sha256=str(
                    auxiliary_configuration.admission_contract_sha256
                ),
                admission_kind="auxiliary_output_admission",
            )
            auxiliary_assignments[assignment.assignment_id] = replace(
                assignment,
                phase="ADMITTED",
                admission_receipt_path=str(
                    admission_payload["admission_receipt_path"]
                ),
                admission_receipt_sha256=str(
                    admission_payload["admission_receipt_sha256"]
                ),
                admitted_sequence=int(event["sequence"]),
                admitted_event_digest=str(event["digest"]),
            )
            auxiliary_admissions += 1
            continue
        if kind == "AUXILIARY_OUTPUT_REJECTED":
            rejection_payload = _strict_keys(
                payload,
                {
                    "assignment_id",
                    "reason",
                    "admission_receipt_path",
                    "admission_receipt_sha256",
                },
                "AUXILIARY_OUTPUT_REJECTED",
            )
            assignment_id = rejection_payload["assignment_id"]
            assignment = auxiliary_assignments.get(str(assignment_id))
            if (
                assignment is None
                or assignment.phase != "QUARANTINED"
                or assignment.output is None
            ):
                raise SchedulerError(
                    "auxiliary rejection lacks quarantined output"
                )
            reason = _require_bounded_text(
                rejection_payload["reason"],
                "auxiliary rejection reason",
            )
            expected_rejection = {
                "schema": 1,
                "kind": "auxiliary_output_rejection",
                "authority": "host_only",
                "assignment_id": assignment.assignment_id,
                "frontier_sha256": assignment.frontier_sha256,
                "parent_checkpoint_sha256":
                    assignment.parent_checkpoint_sha256,
                "output_manifest_sha256":
                    assignment.output.output_manifest_sha256,
                "admission_contract_sha256":
                    auxiliary_configuration.admission_contract_sha256,
                "reason": reason,
                "verdict": "REJECTED",
            }
            _verify_exact_auxiliary_receipt(
                rejection_payload["admission_receipt_path"],
                rejection_payload["admission_receipt_sha256"],
                expected=expected_rejection,
                label="auxiliary host rejection receipt",
            )
            auxiliary_assignments[assignment.assignment_id] = replace(
                assignment, phase="REJECTED"
            )
            auxiliary_admissions += 1
            continue
        if kind == "NATIVE_SIDECAR_REQUEST_ADMITTED":
            request_payload = _strict_keys(
                payload,
                {"attempt_id", "draft", "request"},
                "NATIVE_SIDECAR_REQUEST_ADMITTED",
            )
            draft = native_sidecar_request_draft_from_dict(
                request_payload["draft"]
            )
            request = sidecar_request_from_dict(
                request_payload["request"]
            )
            if request_payload["attempt_id"] != draft.native_attempt_id:
                raise SchedulerError(
                    "native sidecar request event changes its attempt"
                )
            origin_attempt = attempts.get(draft.native_attempt_id)
            if (
                origin_attempt is None
                or not origin_attempt["settled"]
                or origin_attempt["phase"] != "FINISHED"
                or origin_attempt["game"] != draft.game
                or origin_attempt[
                    "native_sidecar_request_draft"
                ]
                != draft
            ):
                raise SchedulerError(
                    "native sidecar request lacks a settled clean attempt"
                )
            lane = lanes[draft.game]
            settlement = next(
                (
                    item
                    for item in lane[
                        "clean_proposer_settlements"
                    ]
                    if isinstance(item, CleanProposerSettlement)
                    and item.attempt_id == draft.native_attempt_id
                ),
                None,
            )
            expected = (
                None
                if settlement is None
                else native_sidecar_request_from_draft(
                    draft, settlement=settlement
                )
            )
            if (
                expected is None
                or request != expected
                or lane["frontier_sha256"]
                != request.frontier_sha256
                or not set(
                    request
                    .cited_public_observation_receipt_sha256s
                ).issubset(
                    lane[
                        "public_observation_receipt_sha256s"
                    ]
                )
                or any(
                    prior.request_id == request.request_id
                    or prior.request_sha256
                    == request.request_sha256
                    or (
                        prior.authority == "native_proposer"
                        and prior.native_attempt_id
                        == request.native_attempt_id
                    )
                    for prior in sidecar_requests
                )
            ):
                raise SchedulerError(
                    "native sidecar request is forged, stale, or repeated"
                )
            sidecar_requests.append(request)
            continue
        if kind == "SUPERVISORY_SIDECAR_REQUEST_ADMITTED":
            request_payload = _strict_keys(
                payload,
                {"assignment_id", "request"},
                "SUPERVISORY_SIDECAR_REQUEST_ADMITTED",
            )
            assignment = auxiliary_assignments.get(
                str(request_payload["assignment_id"])
            )
            request = sidecar_request_from_dict(
                request_payload["request"]
            )
            expected = (
                None
                if assignment is None
                else supervisory_sidecar_request_from_assignment(
                    assignment
                )
            )
            if (
                expected is None
                or request != expected
                or any(
                    prior.request_id == request.request_id
                    or prior.request_sha256
                    == request.request_sha256
                    or (
                        prior.authority
                        == "admitted_supervisory_proposer"
                        and prior.supervisory_assignment_id
                        == request.supervisory_assignment_id
                    )
                    for prior in sidecar_requests
                )
            ):
                raise SchedulerError(
                    "supervisory sidecar request is forged or repeated"
                )
            sidecar_requests.append(request)
            continue
        attempt_id = payload.get("attempt_id")
        attempt = attempts.get(str(attempt_id))
        if kind == "ATTEMPT_SUBSTRATE_INFRASTRUCTURE":
            failure_payload = _strict_keys(
                payload,
                {
                    "attempt_id",
                    "substrate_identity_sha256",
                    "failure_receipt_path",
                    "failure_receipt_sha256",
                    "result",
                    "authenticated_cost_units",
                    "budget_reservation_id",
                    "scheduler_decision_id",
                },
                "ATTEMPT_SUBSTRATE_INFRASTRUCTURE",
            )
            result = failure_payload["result"]
            if (
                attempt is None
                or attempt["phase"] != "BACKEND_PREPARED"
                or attempt["settled"]
                or substrate_incident is not None
                or not _is_sha256(
                    failure_payload[
                        "substrate_identity_sha256"
                    ]
                )
                or failure_payload["authenticated_cost_units"] != 0
                or failure_payload["budget_reservation_id"]
                != attempt["reservation_id"]
                or failure_payload["scheduler_decision_id"]
                != attempt["decision_id"]
                or not isinstance(result, dict)
                or set(result)
                != {
                    "kind",
                    "cost_used",
                    "reason",
                    "candidate",
                    "wip",
                    "blocker",
                    "native_sidecar_request_draft",
                }
                or result.get("kind") != "infrastructure"
                or result.get("cost_used") != 0.0
                or result.get("reason")
                != "codex_substrate_preflight_failed"
                or any(
                    result.get(name) is not None
                    for name in (
                        "candidate",
                        "wip",
                        "blocker",
                        "native_sidecar_request_draft",
                    )
                )
            ):
                raise SchedulerError(
                    "substrate infrastructure settlement is malformed"
                )
            receipt_path = failure_payload[
                "failure_receipt_path"
            ]
            receipt = _reopen_json_receipt(
                receipt_path,
                failure_payload["failure_receipt_sha256"],
                label="substrate preflight failure receipt",
            )
            expected_substrate_failure_keys = {
                "schema", "kind", "campaign_id", "generation_id",
                "attempt_id", "attempt_spec_sha256",
                "substrate_identity_sha256",
                "substrate_preflight_intent_path",
                "substrate_preflight_intent_sha256",
                "preflight_root", "state_root", "failure_stage",
                "error_type", "failure_class", "failure_code",
                "partial_scan_receipt_path",
                "partial_scan_receipt_sha256", "purge_receipt_path",
                "purge_receipt_sha256",
                "post_failure_state_tree_sha256",
                "state_root_empty", "preflight_root_absent",
                "prior_clean_wip_tree_sha256",
                "post_purge_clean_wip_tree_sha256",
                "backend_launch_failure_tombstone_path",
                "backend_launch_failure_tombstone_sha256",
                "proposer_container_started", "bridge_connected",
                "thread_started", "turn_started",
                "candidate_authority", "wip_authority",
                "promotion_authority", "cost_used", "status",
            }
            if (
                not isinstance(receipt_path, str)
                or set(receipt) != expected_substrate_failure_keys
                or Path(receipt_path).name
                != "substrate_preflight_failure_receipt.json"
                or receipt.get("schema") != 1
                or receipt.get("kind")
                != "contiguous_substrate_preflight_failure"
                or receipt.get("campaign_id")
                != genesis["campaign_id"]
                or receipt.get("generation_id")
                != attempt["generation_id"]
                or receipt.get("attempt_id") != attempt_id
                or receipt.get("substrate_identity_sha256")
                != failure_payload[
                    "substrate_identity_sha256"
                ]
                or receipt.get("state_root_empty") is not True
                or receipt.get("preflight_root_absent")
                is not True
                or receipt.get("prior_clean_wip_tree_sha256")
                != receipt.get(
                    "post_purge_clean_wip_tree_sha256"
                )
                or any(
                    receipt.get(name) is not False
                    for name in (
                        "proposer_container_started",
                        "bridge_connected",
                        "thread_started",
                        "turn_started",
                        "candidate_authority",
                        "wip_authority",
                        "promotion_authority",
                    )
                )
                or receipt.get("cost_used") != 0.0
                or receipt.get("status") != "INFRASTRUCTURE"
                or receipt.get("failure_class") not in {
                    "DETERMINISTIC_CONFIGURATION",
                    "TRANSIENT_INFRASTRUCTURE",
                }
                or not _is_identifier(receipt.get("failure_code"))
                or not _is_sha256(
                    receipt.get("partial_scan_receipt_sha256")
                )
                or not _is_sha256(
                    receipt.get("purge_receipt_sha256")
                )
            ):
                raise SchedulerError(
                    "substrate failure receipt grants authority or "
                    "retains failed state"
                )
            lane = lanes[str(attempt["game"])]
            if (
                lane["active_attempt_id"] != attempt_id
                or lane["no_progress"]
                != attempt["no_progress_before"]
            ):
                raise SchedulerError(
                    "substrate failure changed its frontier coordinate"
                )
            budget = settle_budget(
                budget,
                reservation_id=str(attempt["reservation_id"]),
                attempt_id=str(attempt_id),
                charged_units=0,
            )
            lane["active_attempt_id"] = None
            lane["draining"] = False
            attempt["settled"] = True
            attempt["phase"] = "FINISHED"
            attempt["collection_result_kind"] = "infrastructure"
            substrate_incident = {
                "attempt_id": attempt_id,
                "game": attempt["game"],
                "frontier_sha256": attempt["frontier_sha256"],
                "substrate_identity_sha256":
                    failure_payload[
                        "substrate_identity_sha256"
                    ],
                "failure_receipt_path": receipt_path,
                "failure_receipt_sha256":
                    failure_payload["failure_receipt_sha256"],
                "reason_code":
                    "codex_substrate_preflight_failed",
                "failure_class": receipt["failure_class"],
                "failure_code": receipt["failure_code"],
                "incident_event_sequence": event["sequence"],
                "incident_event_digest": event["digest"],
                "incident_identity_sha256":
                    _substrate_incident_identity_sha256(
                        campaign_id=genesis["campaign_id"],
                        attempt_id=str(attempt_id),
                        game=str(attempt["game"]),
                        frontier_sha256=str(
                            attempt["frontier_sha256"]
                        ),
                        substrate_identity_sha256=str(
                            failure_payload[
                                "substrate_identity_sha256"
                            ]
                        ),
                        failure_receipt_sha256=str(
                            failure_payload[
                                "failure_receipt_sha256"
                            ]
                        ),
                        failure_class=str(
                            receipt["failure_class"]
                        ),
                        failure_code=str(
                            receipt["failure_code"]
                        ),
                    ),
                "health_probe_count": 0,
                "pending_reprobe": None,
                "attempted_remediation_epochs": [],
                "last_health_probe": None,
                "circuit_failure_recorded": False,
                "meta_recovery_invocation_count": 0,
                "meta_recovery": None,
            }
            settlements += 1
            continue
        if kind == "META_SUBSTRATE_RECOVERY_AUTHORIZED":
            authorization_payload = _strict_keys(
                payload,
                {
                    "authorization_id",
                    "attempt_id",
                    "substrate_identity_sha256",
                    "incident_failure_receipt_sha256",
                    "incident_event_sequence",
                    "incident_event_digest",
                    "incident_identity_sha256",
                    "meta_request_sha256",
                    "meta_response_sha256",
                    "meta_terminal_sha256",
                    "recommendation",
                    "operator_configuration_sha256",
                    "authorization_receipt_path",
                    "authorization_receipt_sha256",
                    "authorization_authentication_sha256",
                    "invocation_index",
                },
                kind,
            )
            if (
                substrate_incident is None
                or operator_incident is None
                or substrate_incident["pending_reprobe"] is not None
                or substrate_incident["meta_recovery"] is not None
                or substrate_incident[
                    "meta_recovery_invocation_count"
                ]
                != 0
                or attempt_id != substrate_incident["attempt_id"]
                or authorization_payload[
                    "substrate_identity_sha256"
                ]
                != substrate_incident[
                    "substrate_identity_sha256"
                ]
                or authorization_payload[
                    "incident_failure_receipt_sha256"
                ]
                != substrate_incident["failure_receipt_sha256"]
                or authorization_payload[
                    "incident_event_sequence"
                ]
                != substrate_incident["incident_event_sequence"]
                or authorization_payload[
                    "incident_event_digest"
                ]
                != substrate_incident["incident_event_digest"]
                or authorization_payload[
                    "incident_identity_sha256"
                ]
                != substrate_incident[
                    "incident_identity_sha256"
                ]
                or authorization_payload["recommendation"]
                != META_SUBSTRATE_RECOVERY_RECOMMENDATION
                or authorization_payload[
                    "operator_configuration_sha256"
                ]
                != genesis.get("operator_configuration_sha256")
                or authorization_payload["invocation_index"] != 1
                or not _is_canonical_uuid(
                    authorization_payload["authorization_id"]
                )
                or any(
                    not _is_sha256(authorization_payload[name])
                    for name in (
                        "meta_request_sha256",
                        "meta_response_sha256",
                        "meta_terminal_sha256",
                        "incident_event_digest",
                        "incident_identity_sha256",
                        "authorization_receipt_sha256",
                        "authorization_authentication_sha256",
                    )
                )
            ):
                raise SchedulerError(
                    "meta substrate recovery authorization is stale"
                )
            authorization = _reopen_json_receipt(
                authorization_payload[
                    "authorization_receipt_path"
                ],
                authorization_payload[
                    "authorization_receipt_sha256"
                ],
                label="meta substrate recovery authorization",
            )
            expected_keys = {
                "schema", "kind", "campaign_id",
                "authorization_id", "attempt_id",
                "substrate_identity_sha256",
                "incident_failure_receipt_sha256",
                "incident_event_sequence",
                "incident_event_digest",
                "incident_identity_sha256",
                "meta_request_sha256", "meta_response_sha256",
                "meta_terminal_sha256", "recommendation",
                "operator_configuration_sha256",
                "invocation_index", "single_use",
                "solver_authority", "wip_authority",
                "cost_authority", "promotion_authority",
                "authorization_authentication_sha256",
            }
            unsigned = dict(authorization)
            observed_authentication = unsigned.pop(
                "authorization_authentication_sha256", None
            )
            if (
                set(authorization) != expected_keys
                or authorization.get("schema") != 1
                or authorization.get("kind")
                != (
                    "contiguous_meta_substrate_"
                    "recovery_authorization"
                )
                or authorization.get("campaign_id")
                != genesis["campaign_id"]
                or any(
                    authorization.get(name)
                    != authorization_payload[name]
                    for name in (
                        "authorization_id",
                        "attempt_id",
                        "substrate_identity_sha256",
                        "incident_failure_receipt_sha256",
                        "incident_event_sequence",
                        "incident_event_digest",
                        "incident_identity_sha256",
                        "meta_request_sha256",
                        "meta_response_sha256",
                        "meta_terminal_sha256",
                        "recommendation",
                        "operator_configuration_sha256",
                        "invocation_index",
                    )
                )
                or authorization.get("single_use") is not True
                or any(
                    authorization.get(name) is not False
                    for name in (
                        "solver_authority",
                        "wip_authority",
                        "cost_authority",
                        "promotion_authority",
                    )
                )
                or observed_authentication
                != authorization_payload[
                    "authorization_authentication_sha256"
                ]
                or observed_authentication
                != _meta_substrate_recovery_authentication_sha256(
                    unsigned,
                    operator_configuration_sha256=(
                        authorization_payload[
                            "operator_configuration_sha256"
                        ]
                    ),
                )
            ):
                raise SchedulerError(
                    "meta substrate recovery authorization is "
                    "malformed"
                )
            substrate_incident[
                "meta_recovery_invocation_count"
            ] = 1
            substrate_incident["meta_recovery"] = {
                **dict(authorization_payload),
                "probe_index":
                    int(substrate_incident["health_probe_count"]) + 1,
                "phase": "AUTHORIZED",
                "result": None,
            }
            substrate_incident["pending_reprobe"] = {
                "authorization_id":
                    authorization_payload["authorization_id"],
                "attempt_id": attempt_id,
                "substrate_identity_sha256":
                    authorization_payload[
                        "substrate_identity_sha256"
                    ],
                "incident_failure_receipt_sha256":
                    authorization_payload[
                        "incident_failure_receipt_sha256"
                    ],
                "probe_index":
                    int(substrate_incident["health_probe_count"]) + 1,
                "authorization_receipt_sha256":
                    authorization_payload[
                        "authorization_receipt_sha256"
                    ],
                "meta_recovery": True,
            }
            continue
        if kind == "SUBSTRATE_HEALTH_REPROBE_AUTHORIZED":
            authorization_payload = _strict_keys(
                payload,
                {
                    "authorization_id",
                    "attempt_id",
                    "substrate_identity_sha256",
                    "incident_failure_receipt_sha256",
                    "probe_index",
                    "reason_code",
                    "authorization_mode",
                    "retry_not_before",
                    "authorization_receipt_path",
                    "authorization_receipt_sha256",
                },
                "SUBSTRATE_HEALTH_REPROBE_AUTHORIZED",
            )
            authorization_id = authorization_payload[
                "authorization_id"
            ]
            if (
                substrate_incident is None
                or substrate_incident["pending_reprobe"]
                is not None
                or attempt_id != substrate_incident["attempt_id"]
                or not _is_canonical_uuid(authorization_id)
                or authorization_payload[
                    "substrate_identity_sha256"
                ]
                != substrate_incident[
                    "substrate_identity_sha256"
                ]
                or authorization_payload[
                    "incident_failure_receipt_sha256"
                ]
                != substrate_incident[
                    "failure_receipt_sha256"
                ]
                or authorization_payload[
                    "probe_index"
                ]
                != int(substrate_incident["health_probe_count"]) + 1
                or not _is_identifier(
                    authorization_payload["reason_code"]
                )
                or authorization_payload[
                    "authorization_mode"
                ] not in {
                    "sealed_autonomous_circuit",
                    "trusted_operator_early_override",
                }
                or not _is_finite_number(
                    authorization_payload["retry_not_before"]
                )
                or not isinstance(
                    failure_operation_circuits.get(
                        "substrate_health_reprobe:"
                        "controller_substrate"
                    ),
                    dict,
                )
                or authorization_payload["retry_not_before"]
                != failure_operation_circuits[
                    "substrate_health_reprobe:"
                    "controller_substrate"
                ]["retry_not_before"]
            ):
                raise SchedulerError(
                    "substrate reprobe authorization is stale or "
                    "duplicated"
                )
            authorization = _reopen_json_receipt(
                authorization_payload[
                    "authorization_receipt_path"
                ],
                authorization_payload[
                    "authorization_receipt_sha256"
                ],
                label="substrate reprobe authorization",
            )
            expected_authorization_keys = {
                "schema", "kind", "campaign_id",
                "authorization_id", "attempt_id",
                "substrate_identity_sha256",
                "incident_failure_receipt_sha256", "probe_index",
                "reason_code", "authorization_mode",
                "operator_configuration_sha256",
                "retry_not_before", "issued_at", "single_use",
                "sealed_supervisor_authority",
                "trusted_operator_authority",
                "game_scheduler_authority",
                "meta_scheduler_authority",
                "authorization_binding_sha256",
            }
            if (
                set(authorization) != expected_authorization_keys
                or authorization.get("schema") != 2
                or authorization.get("kind")
                != "contiguous_substrate_health_reprobe_authorization"
                or authorization.get("campaign_id")
                != genesis["campaign_id"]
                or authorization.get("authorization_id")
                != authorization_id
                or authorization.get("attempt_id") != attempt_id
                or any(
                    authorization.get(name)
                    != authorization_payload[name]
                    for name in (
                        "authorization_id",
                        "attempt_id",
                        "substrate_identity_sha256",
                        "incident_failure_receipt_sha256",
                        "probe_index",
                        "reason_code",
                        "authorization_mode",
                        "retry_not_before",
                    )
                )
                or authorization.get(
                    "operator_configuration_sha256"
                )
                != genesis.get("operator_configuration_sha256")
                or not _is_finite_number(
                    authorization.get("issued_at")
                )
                or authorization.get("issued_at")
                != float(event["recorded_at"])
                or authorization.get("single_use") is not True
                or authorization.get("game_scheduler_authority")
                is not False
                or authorization.get("meta_scheduler_authority")
                is not False
                or authorization.get("sealed_supervisor_authority")
                is not (
                    authorization.get("authorization_mode")
                    == "sealed_autonomous_circuit"
                )
                or authorization.get("trusted_operator_authority")
                is not (
                    authorization.get("authorization_mode")
                    == "trusted_operator_early_override"
                )
                or (
                    authorization.get("authorization_mode")
                    == "sealed_autonomous_circuit"
                    and authorization.get("issued_at")
                    < authorization.get("retry_not_before")
                )
            ):
                raise SchedulerError(
                    "substrate reprobe authorization receipt is "
                    "malformed"
                )
            unsigned_authorization = dict(authorization)
            observed_binding = unsigned_authorization.pop(
                "authorization_binding_sha256"
            )
            if (
                observed_binding
                != sha256_json(unsigned_authorization)
            ):
                raise SchedulerError(
                    "substrate reprobe authorization binding changed"
                )
            substrate_incident["pending_reprobe"] = dict(
                authorization_payload
            )
            continue
        if kind in {
            "SUBSTRATE_HEALTH_REPROBE_FAILED",
            "SUBSTRATE_HEALTH_RESTORED",
            "META_SUBSTRATE_RECOVERY_FAILED",
            "META_SUBSTRATE_HEALTH_RESTORED",
        }:
            health_result_fields = {
                "authorization_id",
                "attempt_id",
                "substrate_identity_sha256",
                "incident_failure_receipt_sha256",
                "probe_index",
                "remediation_epoch_sha256",
                "healthy_substrate_identity_sha256",
                "failure_class",
                "failure_code",
                "health_receipt_path",
                "health_receipt_sha256",
                "status",
            }
            meta_result_fields = {
                "incident_event_sequence",
                "incident_event_digest",
                "incident_identity_sha256",
                "meta_request_sha256",
                "meta_response_sha256",
                "meta_terminal_sha256",
                "recommendation",
                "authorization_receipt_sha256",
                "authorization_authentication_sha256",
                "rematerialization_evidence_path",
                "rematerialization_evidence_sha256",
                "invocation_index",
            }
            is_meta_result = kind.startswith("META_")
            health_payload = _strict_keys(
                payload,
                (
                    health_result_fields | meta_result_fields
                    if is_meta_result
                    else health_result_fields
                ),
                kind,
            )
            pending_reprobe = (
                None
                if substrate_incident is None
                else substrate_incident["pending_reprobe"]
            )
            expected_status = (
                "PASS"
                if kind in {
                    "SUBSTRATE_HEALTH_RESTORED",
                    "META_SUBSTRATE_HEALTH_RESTORED",
                }
                else "FAILED"
            )
            meta_recovery = (
                None
                if substrate_incident is None
                else substrate_incident["meta_recovery"]
            )
            if (
                pending_reprobe is None
                or health_payload["status"] != expected_status
                or any(
                    health_payload[name]
                    != pending_reprobe[name]
                    for name in (
                        "authorization_id",
                        "attempt_id",
                        "substrate_identity_sha256",
                        "incident_failure_receipt_sha256",
                        "probe_index",
                    )
                )
                or not _is_sha256(
                    health_payload["remediation_epoch_sha256"]
                )
                or health_payload["remediation_epoch_sha256"]
                in substrate_incident[
                    "attempted_remediation_epochs"
                ]
                or (
                    expected_status == "PASS"
                    and (
                        not _is_sha256(
                            health_payload[
                                "healthy_substrate_identity_sha256"
                            ]
                        )
                        or health_payload["failure_class"] is not None
                        or health_payload["failure_code"] is not None
                    )
                )
                or (
                    expected_status == "FAILED"
                    and (
                        health_payload[
                            "healthy_substrate_identity_sha256"
                        ]
                        is not None
                        or health_payload["failure_class"] not in {
                            "DETERMINISTIC_CONFIGURATION",
                            "TRANSIENT_INFRASTRUCTURE",
                        }
                        or not _is_identifier(
                            health_payload["failure_code"]
                        )
                    )
                )
                or (
                    is_meta_result
                    and (
                        not isinstance(meta_recovery, dict)
                        or meta_recovery.get("phase")
                        != "AUTHORIZED"
                        or pending_reprobe.get("meta_recovery")
                        is not True
                        or health_payload["invocation_index"] != 1
                        or any(
                            health_payload[name]
                            != meta_recovery[name]
                            for name in (
                                "meta_request_sha256",
                                "meta_response_sha256",
                                "meta_terminal_sha256",
                                "incident_event_sequence",
                                "incident_event_digest",
                                "incident_identity_sha256",
                                "recommendation",
                                "authorization_receipt_sha256",
                                "authorization_authentication_sha256",
                                "invocation_index",
                            )
                        )
                    )
                )
                or (
                    not is_meta_result
                    and pending_reprobe.get("meta_recovery")
                    is not None
                )
            ):
                raise SchedulerError(
                    "substrate health result lacks a single-use "
                    "authorization"
                )
            health = _reopen_json_receipt(
                health_payload["health_receipt_path"],
                health_payload["health_receipt_sha256"],
                label="substrate health reprobe receipt",
            )
            expected_health_keys = {
                "schema", "kind", "campaign_id", "generation_id",
                "attempt_id", "attempt_spec_sha256",
                "authorization_id", "authorization_receipt_sha256",
                "probe_index", "failed_substrate_identity_sha256",
                "healthy_substrate_identity_sha256",
                "incident_failure_receipt_sha256",
                "remediation_epoch_sha256",
                "rematerialization_evidence_path",
                "rematerialization_evidence_sha256",
                "fresh_state_root_created", "health_state_root",
                "health_runtime_root", "preflight_receipt_path",
                "preflight_receipt_sha256",
                "guardian_state_root_write_probe_status",
                "scan_receipt_path", "scan_receipt_sha256",
                "purge_receipt_path", "purge_receipt_sha256",
                "failure_class", "failure_code",
                "health_state_root_absent",
                "health_runtime_root_absent",
                "proposer_container_started", "bridge_connected",
                "thread_started", "turn_started",
                "candidate_authority", "wip_authority",
                "promotion_authority", "cost_used", "status",
            }
            health_root = Path(
                health_payload["health_receipt_path"]
            ).parent
            remediation = _reopen_json_receipt(
                health.get("rematerialization_evidence_path"),
                health.get("rematerialization_evidence_sha256"),
                label="substrate rematerialization evidence",
            )
            scan = _reopen_json_receipt(
                health.get("scan_receipt_path"),
                health.get("scan_receipt_sha256"),
                label="substrate health scan receipt",
            )
            purge = _reopen_json_receipt(
                health.get("purge_receipt_path"),
                health.get("purge_receipt_sha256"),
                label="substrate health purge receipt",
            )
            observed_remediation_epoch = remediation.pop(
                "remediation_epoch_sha256", None
            )
            expected_healthy_identity = (
                sha256_json({
                    "schema": 1,
                    "kind": "healthy_controller_substrate_identity",
                    "failed_substrate_identity_sha256":
                        health_payload["substrate_identity_sha256"],
                    "remediation_epoch_sha256":
                        health_payload["remediation_epoch_sha256"],
                    "preflight_receipt_sha256":
                        health.get("preflight_receipt_sha256"),
                    "guardian_state_root_write_probe_status":
                        health.get(
                            "guardian_state_root_write_probe_status"
                        ),
                    "status": "PASS",
                })
                if expected_status == "PASS"
                else None
            )
            if (
                set(health) != expected_health_keys
                or health.get("schema") != 1
                or health.get("kind")
                != "contiguous_substrate_health_reprobe"
                or health.get("campaign_id")
                != genesis["campaign_id"]
                or health.get("attempt_id") != attempt_id
                or health.get("authorization_id")
                != health_payload["authorization_id"]
                or health.get("probe_index")
                != health_payload["probe_index"]
                or health.get("authorization_receipt_sha256")
                != pending_reprobe[
                    "authorization_receipt_sha256"
                ]
                or health.get("remediation_epoch_sha256")
                != health_payload["remediation_epoch_sha256"]
                or health.get("failed_substrate_identity_sha256")
                != health_payload["substrate_identity_sha256"]
                or health.get(
                    "incident_failure_receipt_sha256"
                )
                != health_payload[
                    "incident_failure_receipt_sha256"
                ]
                or health.get(
                    "healthy_substrate_identity_sha256"
                )
                != health_payload[
                    "healthy_substrate_identity_sha256"
                ]
                or health.get("failure_class")
                != health_payload["failure_class"]
                or health.get("failure_code")
                != health_payload["failure_code"]
                or health.get("status") != expected_status
                or health.get("fresh_state_root_created") is not True
                or observed_remediation_epoch
                != health_payload["remediation_epoch_sha256"]
                or sha256_json(remediation)
                != health_payload["remediation_epoch_sha256"]
                or health.get(
                    "healthy_substrate_identity_sha256"
                )
                != expected_healthy_identity
                or (
                    expected_status == "PASS"
                    and (
                        health.get(
                            "guardian_state_root_write_probe_status"
                        )
                        != "PASS"
                        or not _is_sha256(
                            health.get("preflight_receipt_sha256")
                        )
                        or health.get("preflight_receipt_path")
                        != str(
                            health_root
                            / "substrate_preflight_receipt.json"
                        )
                    )
                )
                or health.get("scan_receipt_path")
                != str(health_root / "scan.json")
                or set(scan)
                != {
                    "schema", "kind", "campaign_id",
                    "generation_id", "attempt_id",
                    "attempt_spec_sha256", "authorization_id",
                    "probe_index", "source_scan_receipt_path",
                    "source_scan_receipt_sha256",
                    "state_inventory_before_purge", "status",
                }
                or scan.get("kind")
                != "contiguous_substrate_health_reprobe_scan"
                or scan.get("campaign_id") != genesis["campaign_id"]
                or scan.get("attempt_id") != attempt_id
                or scan.get("authorization_id")
                != health_payload["authorization_id"]
                or scan.get("probe_index")
                != health_payload["probe_index"]
                or scan.get("status") != "COMPLETE"
                or health.get("purge_receipt_path")
                != str(health_root / "purge.json")
                or set(purge)
                != {
                    "schema", "kind", "campaign_id",
                    "generation_id", "attempt_id",
                    "attempt_spec_sha256", "authorization_id",
                    "probe_index", "scan_receipt_sha256",
                    "health_state_root_absent",
                    "health_runtime_root_absent",
                    "prior_clean_wip_tree_sha256",
                    "post_clean_wip_tree_sha256", "status",
                }
                or purge.get("kind")
                != "contiguous_substrate_health_reprobe_purge"
                or purge.get("campaign_id") != genesis["campaign_id"]
                or purge.get("attempt_id") != attempt_id
                or purge.get("authorization_id")
                != health_payload["authorization_id"]
                or purge.get("probe_index")
                != health_payload["probe_index"]
                or purge.get("scan_receipt_sha256")
                != health.get("scan_receipt_sha256")
                or purge.get("health_state_root_absent") is not True
                or purge.get("health_runtime_root_absent") is not True
                or purge.get("prior_clean_wip_tree_sha256")
                != purge.get("post_clean_wip_tree_sha256")
                or purge.get("status") != "PASS"
                or not _is_sha256(
                    health.get(
                        "rematerialization_evidence_sha256"
                    )
                )
                or any(
                    health.get(name) is not False
                    for name in (
                        "proposer_container_started",
                        "bridge_connected",
                        "thread_started",
                        "turn_started",
                        "candidate_authority",
                        "wip_authority",
                        "promotion_authority",
                    )
                )
                or health.get("health_state_root_absent")
                is not True
                or health.get("health_runtime_root_absent")
                is not True
                or health.get("cost_used") != 0.0
                or (
                    is_meta_result
                    and (
                        health_payload[
                            "rematerialization_evidence_path"
                        ]
                        != health.get(
                            "rematerialization_evidence_path"
                        )
                        or health_payload[
                            "rematerialization_evidence_sha256"
                        ]
                        != health.get(
                            "rematerialization_evidence_sha256"
                        )
                    )
                )
            ):
                raise SchedulerError(
                    "substrate health receipt grants solver authority"
                )
            substrate_incident[
                "attempted_remediation_epochs"
            ].append(health_payload["remediation_epoch_sha256"])
            substrate_incident["health_probe_count"] = (
                int(substrate_incident["health_probe_count"]) + 1
            )
            substrate_incident["last_health_probe"] = {
                "authorization_id":
                    health_payload["authorization_id"],
                "probe_index": health_payload["probe_index"],
                "remediation_epoch_sha256":
                    health_payload["remediation_epoch_sha256"],
                "health_receipt_path":
                    health_payload["health_receipt_path"],
                "health_receipt_sha256":
                    health_payload["health_receipt_sha256"],
                "status": expected_status,
                "healthy_substrate_identity_sha256":
                    health_payload[
                        "healthy_substrate_identity_sha256"
                    ],
                "failure_class": health_payload["failure_class"],
                "failure_code": health_payload["failure_code"],
            }
            substrate_incident["pending_reprobe"] = None
            if kind == "SUBSTRATE_HEALTH_RESTORED":
                substrate_incident = None
            elif kind == "META_SUBSTRATE_HEALTH_RESTORED":
                substrate_incident["meta_recovery"] = {
                    **meta_recovery,
                    "phase": "HEALTH_RESTORED",
                    "result": {
                        "recovery_result_event_sequence":
                            event["sequence"],
                        "recovery_result_event_digest":
                            event["digest"],
                        "health_receipt_path":
                            health_payload["health_receipt_path"],
                        "health_receipt_sha256":
                            health_payload["health_receipt_sha256"],
                        "rematerialization_evidence_path":
                            health_payload[
                                "rematerialization_evidence_path"
                            ],
                        "rematerialization_evidence_sha256":
                            health_payload[
                                "rematerialization_evidence_sha256"
                            ],
                        "remediation_epoch_sha256":
                            health_payload[
                                "remediation_epoch_sha256"
                            ],
                        "healthy_substrate_identity_sha256":
                            health_payload[
                                "healthy_substrate_identity_sha256"
                            ],
                    },
                }
            elif kind == "META_SUBSTRATE_RECOVERY_FAILED":
                substrate_incident["meta_recovery"] = {
                    **meta_recovery,
                    "phase": "FAILED",
                    "result": {
                        "recovery_result_event_sequence":
                            event["sequence"],
                        "recovery_result_event_digest":
                            event["digest"],
                        "health_receipt_path":
                            health_payload["health_receipt_path"],
                        "health_receipt_sha256":
                            health_payload["health_receipt_sha256"],
                        "failure_class":
                            health_payload["failure_class"],
                        "failure_code":
                            health_payload["failure_code"],
                    },
                }
            else:
                substrate_incident[
                    "circuit_failure_recorded"
                ] = False
            continue
        if kind == "META_SUBSTRATE_RESUME_AUTHORIZED":
            resume_payload = _strict_keys(
                payload,
                {
                    "authorization_id",
                    "attempt_id",
                    "incident_event_sequence",
                    "incident_event_digest",
                    "incident_identity_sha256",
                    "recovery_result_event_sequence",
                    "recovery_result_event_digest",
                    "health_receipt_sha256",
                    "rematerialization_evidence_sha256",
                    "healthy_substrate_identity_sha256",
                    "resume_receipt_path",
                    "resume_receipt_sha256",
                    "resume_authentication_sha256",
                    "invocation_index",
                },
                kind,
            )
            meta_recovery = (
                None
                if substrate_incident is None
                else substrate_incident["meta_recovery"]
            )
            result = (
                None
                if not isinstance(meta_recovery, dict)
                else meta_recovery.get("result")
            )
            if (
                substrate_incident is None
                or operator_incident is None
                or not isinstance(meta_recovery, dict)
                or meta_recovery.get("phase") != "HEALTH_RESTORED"
                or not isinstance(result, dict)
                or resume_payload["authorization_id"]
                != meta_recovery["authorization_id"]
                or attempt_id != substrate_incident["attempt_id"]
                or resume_payload["incident_event_sequence"]
                != substrate_incident["incident_event_sequence"]
                or resume_payload["incident_event_digest"]
                != substrate_incident["incident_event_digest"]
                or resume_payload["incident_identity_sha256"]
                != substrate_incident[
                    "incident_identity_sha256"
                ]
                or resume_payload["invocation_index"] != 1
                or any(
                    resume_payload[name] != result[name]
                    for name in (
                        "recovery_result_event_sequence",
                        "recovery_result_event_digest",
                        "health_receipt_sha256",
                        "rematerialization_evidence_sha256",
                        "healthy_substrate_identity_sha256",
                    )
                )
                or any(
                    not _is_sha256(resume_payload[name])
                    for name in (
                        "recovery_result_event_digest",
                        "incident_event_digest",
                        "incident_identity_sha256",
                        "health_receipt_sha256",
                        "rematerialization_evidence_sha256",
                        "healthy_substrate_identity_sha256",
                        "resume_receipt_sha256",
                        "resume_authentication_sha256",
                    )
                )
            ):
                raise SchedulerError(
                    "meta substrate resume lacks healthy evidence"
                )
            resume = _reopen_json_receipt(
                resume_payload["resume_receipt_path"],
                resume_payload["resume_receipt_sha256"],
                label="meta substrate resume authorization",
            )
            expected_keys = {
                "schema", "kind", "campaign_id",
                "authorization_id", "attempt_id",
                "substrate_identity_sha256",
                "incident_failure_receipt_sha256",
                "incident_event_sequence",
                "incident_event_digest",
                "incident_identity_sha256",
                "meta_request_sha256", "meta_response_sha256",
                "meta_terminal_sha256", "recommendation",
                "operator_configuration_sha256",
                "recovery_result_event_sequence",
                "recovery_result_event_digest",
                "health_receipt_sha256",
                "rematerialization_evidence_sha256",
                "remediation_epoch_sha256",
                "healthy_substrate_identity_sha256",
                "invocation_index", "single_use",
                "solver_authority", "wip_authority",
                "cost_authority", "promotion_authority",
                "resume_authentication_sha256",
            }
            unsigned = dict(resume)
            observed_authentication = unsigned.pop(
                "resume_authentication_sha256", None
            )
            if (
                set(resume) != expected_keys
                or resume.get("schema") != 1
                or resume.get("kind")
                != (
                    "contiguous_meta_substrate_"
                    "resume_authorization"
                )
                or resume.get("campaign_id")
                != genesis["campaign_id"]
                or any(
                    resume.get(name) != meta_recovery[name]
                    for name in (
                        "authorization_id",
                        "attempt_id",
                        "substrate_identity_sha256",
                        "incident_failure_receipt_sha256",
                        "incident_event_sequence",
                        "incident_event_digest",
                        "incident_identity_sha256",
                        "meta_request_sha256",
                        "meta_response_sha256",
                        "meta_terminal_sha256",
                        "recommendation",
                        "operator_configuration_sha256",
                        "invocation_index",
                    )
                )
                or any(
                    resume.get(name) != result[name]
                    for name in (
                        "recovery_result_event_sequence",
                        "recovery_result_event_digest",
                        "health_receipt_sha256",
                        "rematerialization_evidence_sha256",
                        "remediation_epoch_sha256",
                        "healthy_substrate_identity_sha256",
                    )
                )
                or resume.get("single_use") is not True
                or any(
                    resume.get(name) is not False
                    for name in (
                        "solver_authority",
                        "wip_authority",
                        "cost_authority",
                        "promotion_authority",
                    )
                )
                or observed_authentication
                != resume_payload["resume_authentication_sha256"]
                or observed_authentication
                != _meta_substrate_resume_authentication_sha256(
                    unsigned,
                    operator_configuration_sha256=(
                        meta_recovery[
                            "operator_configuration_sha256"
                        ]
                    ),
                )
            ):
                raise SchedulerError(
                    "meta substrate resume authorization is malformed"
                )
            operation_key = (
                "substrate_health_reprobe:controller_substrate"
            )
            operation_state = failure_operation_circuits.get(
                operation_key
            )
            domain_state = failure_domain_circuits.get(
                "controller_substrate"
            )
            if (
                not isinstance(operation_state, dict)
                or not isinstance(domain_state, dict)
                or operator_incident.get("attempt_id")
                != substrate_incident["attempt_id"]
                or operator_incident.get("operation")
                != "substrate_health_reprobe"
                or operator_incident.get("fault_domain")
                != "controller_substrate"
            ):
                raise SchedulerError(
                    "meta substrate resume incident is not exact"
                )
            failure_operation_circuits[operation_key] = {
                **operation_state,
                "consecutive": 0,
                "retry_not_before": None,
            }
            failure_domain_circuits["controller_substrate"] = {
                **domain_state,
                "consecutive": 0,
                "retry_not_before": None,
                "last_operation": None,
            }
            operator_incident = None
            substrate_incident = None
            continue
        if kind == "SUBSTRATE_HEALTH_REPROBE_ABORTED":
            abort_payload = _strict_keys(
                payload,
                {
                    "authorization_id",
                    "attempt_id",
                    "substrate_identity_sha256",
                    "incident_failure_receipt_sha256",
                    "probe_index",
                    "error_type",
                    "status",
                },
                "SUBSTRATE_HEALTH_REPROBE_ABORTED",
            )
            pending_reprobe = (
                None
                if substrate_incident is None
                else substrate_incident["pending_reprobe"]
            )
            if (
                pending_reprobe is None
                or abort_payload["status"] != "ABORTED"
                or not _is_identifier(abort_payload["error_type"])
                or any(
                    abort_payload[name] != pending_reprobe[name]
                    for name in (
                        "authorization_id",
                        "attempt_id",
                        "substrate_identity_sha256",
                        "incident_failure_receipt_sha256",
                        "probe_index",
                    )
                )
            ):
                raise SchedulerError(
                    "substrate health abort is malformed"
                )
            substrate_incident["health_probe_count"] = (
                int(substrate_incident["health_probe_count"]) + 1
            )
            substrate_incident["last_health_probe"] = {
                "authorization_id":
                    abort_payload["authorization_id"],
                "probe_index": abort_payload["probe_index"],
                "remediation_epoch_sha256": None,
                "health_receipt_path": None,
                "health_receipt_sha256": None,
                "status": "ABORTED",
                "error_type": abort_payload["error_type"],
            }
            substrate_incident["pending_reprobe"] = None
            substrate_incident["circuit_failure_recorded"] = False
            continue
        if kind == "ATTEMPT_RETRY":
            retry_payload = _strict_keys(
                payload,
                {
                    "attempt_id",
                    "retry_index",
                    "operation",
                    "operation_retry_index",
                    "error_type",
                    "backoff_seconds",
                    "retry_not_before",
                },
                "ATTEMPT_RETRY",
            )
            operation = retry_payload["operation"]
            allowed_phases = {
                "input_materialize": {"RESERVED"},
                "backend_prepare": {"PREPARED"},
                "backend_launch": {"BACKEND_PREPARED"},
                "backend_poll": {"RUNNING", "DRAINING"},
                "backend_collect": {"EXITED"},
                "backend_teardown": {
                    "COLLECTED", "COLLECTION_REJECTED"
                },
                "promotion_commit": {"PROMOTING"},
                "promotion_recover": {"PROMOTING"},
            }
            if (
                attempt is None
                or not isinstance(operation, str)
                or operation not in allowed_phases
                or attempt["phase"] not in allowed_phases[operation]
                or retry_payload["retry_index"]
                != int(attempt["retry_count"]) + 1
                or retry_payload["operation_retry_index"]
                != attempt["operation_retry_counts"].get(operation, 0) + 1
                or not _is_identifier(retry_payload["error_type"])
            ):
                raise SchedulerError(
                    "attempt retry schema/order/phase is invalid"
                )
            operation_retry_index = int(
                retry_payload["operation_retry_index"]
            )
            expected_backoff = OPERATION_RETRY_BACKOFF_SECONDS[
                min(
                    operation_retry_index,
                    len(OPERATION_RETRY_BACKOFF_SECONDS),
                ) - 1
            ]
            if (
                retry_payload["backoff_seconds"] != expected_backoff
                or retry_payload["retry_not_before"]
                != float(event["recorded_at"]) + expected_backoff
            ):
                raise SchedulerError(
                    "attempt retry backoff is not canonical"
                )
            attempt["retry_count"] = retry_payload["retry_index"]
            attempt["operation_retry_counts"][
                operation
            ] = operation_retry_index
            attempt["operation_retry_not_before"][
                operation
            ] = retry_payload["retry_not_before"]
            continue
        if kind == "ATTEMPT_PUBLIC_OBSERVATIONS_STAGING":
            staging_payload = _strict_keys(
                payload,
                {"attempt_id", "transition"},
                "ATTEMPT_PUBLIC_OBSERVATIONS_STAGING",
            )
            if (
                attempt is None
                or attempt["phase"] != "EXITED"
                or attempt["public_observation_transition"] is not None
            ):
                raise SchedulerError(
                    "public observation staging targets the wrong attempt"
                )
            generation_dir = attempt.get("generation_dir")
            host_transcript_path = attempt.get(
                "host_transcript_path"
            )
            if (
                not isinstance(generation_dir, str)
                or not isinstance(host_transcript_path, str)
                or not Path(generation_dir).is_absolute()
                or Path(generation_dir).name
                != attempt["generation_id"]
                or Path(host_transcript_path)
                != Path(generation_dir) / "host" / "backend.jsonl"
            ):
                raise SchedulerError(
                    "public observation staging lacks its exact attempt root"
                )
            validate_public_observation_transition(
                staging_payload["transition"],
                attempt_id=str(attempt_id),
                generation_id=str(attempt["generation_id"]),
                game=str(attempt["game"]),
                frontier_sha256=str(attempt["frontier_sha256"]),
                parent_checkpoint_sha256=str(
                    attempt["parent_checkpoint_sha256"]
                ),
                host_transcript_path=host_transcript_path,
                reopen_receipts=False,
            )
            attempt["public_observation_transition"] = dict(
                staging_payload["transition"]
            )
            continue
        if kind == "ATTEMPT_EXITED" and "terminal" in payload:
            exited_payload = _strict_keys(
                payload,
                {"attempt_id", "terminal"},
                "ATTEMPT_EXITED",
            )
            terminal = exited_payload["terminal"]
            if (
                attempt is None
                or attempt["phase"] not in {"RUNNING", "DRAINING"}
                or not isinstance(terminal, dict)
                or set(terminal)
                != {"status", "observation_sha256", "exit_code"}
                or terminal.get("status")
                not in {"exited", "containment_fault"}
                or not _is_sha256(
                    terminal.get("observation_sha256")
                )
                or (
                    terminal.get("exit_code") is not None
                    and (
                        not isinstance(
                            terminal.get("exit_code"), int
                        )
                        or isinstance(
                            terminal.get("exit_code"), bool
                        )
                    )
                )
            ):
                raise SchedulerError(
                    "typed terminal observation is malformed"
                )
            attempt["terminal_status"] = terminal["status"]
            attempt["phase"] = "EXITED"
            continue
        if kind == "ATTEMPT_PUBLIC_ACTION_PROTOCOL_INVALID":
            invalid_payload = _strict_keys(
                payload,
                {
                    "attempt_id",
                    "protocol_invalid_receipt_path",
                    "protocol_invalid_receipt_sha256",
                    "terminal_evidence",
                    "result",
                },
                "ATTEMPT_PUBLIC_ACTION_PROTOCOL_INVALID",
            )
            result = invalid_payload["result"]
            receipt_path = invalid_payload[
                "protocol_invalid_receipt_path"
            ]
            receipt_sha256 = invalid_payload[
                "protocol_invalid_receipt_sha256"
            ]
            terminal_evidence = invalid_payload["terminal_evidence"]
            try:
                protocol_invalid_cost_units = charge_to_units(
                    result.get("cost_used")
                    if isinstance(result, dict)
                    else None
                )
            except SchedulerError:
                protocol_invalid_cost_units = None
            if (
                attempt is None
                or attempt["phase"] != "EXITED"
                or not isinstance(result, dict)
                or set(result)
                != {
                    "kind",
                    "cost_used",
                    "reason",
                    "candidate",
                    "wip",
                    "blocker",
                    "native_sidecar_request_draft",
                }
                or result.get("kind") != "protocol_invalid"
                or result.get("reason")
                != "public_action_protocol_invalid"
                or any(
                    result.get(name) is not None
                    for name in (
                        "candidate",
                        "wip",
                        "blocker",
                        "native_sidecar_request_draft",
                    )
                )
                or protocol_invalid_cost_units is None
                or not isinstance(receipt_path, str)
                or Path(receipt_path).name
                != "arena_public_action_protocol_invalid_receipt.json"
                or not isinstance(terminal_evidence, dict)
            ):
                raise SchedulerError(
                    "protocol-invalid event is malformed or retains authority"
                )
            receipt = _reopen_json_receipt(
                receipt_path,
                receipt_sha256,
                label="public-action protocol-invalid receipt",
            )
            violation = receipt.get("protocol_violation")
            if (
                set(receipt)
                != {
                    "schema",
                    "kind",
                    "campaign_id",
                    "generation_id",
                    "attempt_id",
                    "attempt_spec_sha256",
                    "protocol_violation",
                    "protocol_violation_sha256",
                    "proposer_containment_sha256",
                    "controller_absence_receipt_sha256",
                    "controller_state_scan_receipt_path",
                    "controller_state_scan_receipt_sha256",
                    "retained_canary_scan_receipt_path",
                    "retained_canary_scan_receipt_sha256",
                    "partial_taint_scan_receipt_path",
                    "partial_taint_scan_receipt_sha256",
                    "partial_taint_status",
                    "partial_usage_receipt_path",
                    "partial_usage_receipt_sha256",
                    "usage_accounting_complete",
                    "cost_used",
                    "cost_authority",
                    "candidate_admissible",
                    "wip_admissible",
                    "public_observation_admissible",
                    "sidecar_request_admissible",
                    "supervisory_handoff_admissible",
                    "promotion_admissible",
                    "restart_restoration_admissible",
                    "status",
                }
                or receipt.get("schema") != 1
                or receipt.get("kind")
                != "contiguous_arena_public_action_protocol_invalid"
                or receipt.get("campaign_id") != genesis["campaign_id"]
                or receipt.get("generation_id")
                != attempt["generation_id"]
                or receipt.get("attempt_id") != attempt_id
                or not _is_sha256(
                    receipt.get("attempt_spec_sha256")
                )
                or not isinstance(violation, dict)
                or receipt.get("protocol_violation_sha256")
                != sha256_json(violation)
                or not _is_sha256(
                    receipt.get("proposer_containment_sha256")
                )
                or not _is_sha256(
                    receipt.get("controller_absence_receipt_sha256")
                )
                or receipt.get("cost_used") != result["cost_used"]
                or receipt.get("cost_authority")
                not in {
                    "full_finite_reservation",
                    "explicit_unlimited_no_local_charge",
                }
                or any(
                    receipt.get(name) is not False
                    for name in (
                        "candidate_admissible",
                        "wip_admissible",
                        "public_observation_admissible",
                        "sidecar_request_admissible",
                        "supervisory_handoff_admissible",
                        "promotion_admissible",
                        "restart_restoration_admissible",
                    )
                )
                or receipt.get("status") != "PROTOCOL_INVALID"
            ):
                raise SchedulerError(
                    "protocol-invalid receipt is stale or grants authority"
                )
            terminal_receipts = {
                "controller_state_scan": (
                    "controller_state_scan_receipt_path",
                    "controller_state_scan_receipt_sha256",
                    "controller_state_scan_receipt.json",
                    "contiguous_controller_state_scan",
                ),
                "retained_canary_scan": (
                    "retained_canary_scan_receipt_path",
                    "retained_canary_scan_receipt_sha256",
                    "retained_canary_scan_receipt.json",
                    "contiguous_retained_canary_scan",
                ),
                "partial_taint_scan": (
                    "partial_taint_scan_receipt_path",
                    "partial_taint_scan_receipt_sha256",
                    "protocol_invalid_partial_taint_scan_receipt.json",
                    "contiguous_protocol_invalid_partial_taint_scan",
                ),
                "partial_usage": (
                    "partial_usage_receipt_path",
                    "partial_usage_receipt_sha256",
                    "protocol_invalid_partial_usage_receipt.json",
                    "contiguous_protocol_invalid_partial_usage",
                ),
            }
            expected_terminal_keys = {
                field
                for path_field, digest_field, _name, _kind
                in terminal_receipts.values()
                for field in (path_field, digest_field)
            }
            reopened_terminal: dict[str, dict[str, Any]] = {}
            if set(terminal_evidence) != expected_terminal_keys:
                raise SchedulerError(
                    "protocol-invalid terminal evidence schema is malformed"
                )
            for label, (
                path_field,
                digest_field,
                expected_name,
                expected_kind,
            ) in terminal_receipts.items():
                path_value = terminal_evidence.get(path_field)
                digest_value = terminal_evidence.get(digest_field)
                if (
                    not isinstance(path_value, str)
                    or Path(path_value).name != expected_name
                    or receipt.get(path_field) != path_value
                    or receipt.get(digest_field) != digest_value
                ):
                    raise SchedulerError(
                        "protocol-invalid terminal evidence is unbound"
                    )
                terminal_receipt = _reopen_json_receipt(
                    path_value,
                    digest_value,
                    label=f"protocol-invalid {label}",
                )
                if (
                    terminal_receipt.get("schema") != 1
                    or terminal_receipt.get("kind") != expected_kind
                    or terminal_receipt.get("campaign_id")
                    != genesis["campaign_id"]
                    or terminal_receipt.get("generation_id")
                    != attempt["generation_id"]
                    or terminal_receipt.get("attempt_id") != attempt_id
                ):
                    raise SchedulerError(
                        "protocol-invalid terminal receipt crosses identity"
                    )
                reopened_terminal[label] = terminal_receipt
            partial_taint = reopened_terminal["partial_taint_scan"]
            partial_usage = reopened_terminal["partial_usage"]
            if (
                partial_taint.get("classification_authority")
                != "source_environment_taint_only"
                or partial_taint.get("status") not in {"CLEAN", "TAINT"}
                or receipt.get("partial_taint_status")
                != partial_taint.get("status")
                or not isinstance(
                    partial_usage.get("accounting_complete"), bool
                )
                or partial_usage.get("unknown_token_usage")
                is not (
                    not partial_usage["accounting_complete"]
                )
                or partial_usage.get("cost_settlement_authority")
                is not False
                or receipt.get("usage_accounting_complete")
                is not partial_usage["accounting_complete"]
            ):
                raise SchedulerError(
                    "protocol-invalid taint/accounting classification changed"
                )
            if (
                budget.limit_units is None
                and (
                    receipt["cost_authority"]
                    != "explicit_unlimited_no_local_charge"
                    or result["cost_used"] != 0.0
                )
            ):
                raise SchedulerError(
                    "unlimited protocol-invalid cost is not conservative"
                )
            attempt["phase"] = "COLLECTION_REJECTED"
            attempt["protocol_invalid_result"] = dict(result)
            attempt["collection_result_kind"] = "protocol_invalid"
            continue
        if kind == "ATTEMPT_COLLECTED" and "collection" in payload:
            collected_payload = _strict_keys(
                payload,
                {
                    "attempt_id",
                    "collection",
                    "public_observation_transition_sha256",
                },
                "ATTEMPT_COLLECTED",
            )
            transition = (
                None
                if attempt is None
                else attempt.get("public_observation_transition")
            )
            collection = collected_payload["collection"]
            if (
                attempt is None
                or attempt["phase"] != "EXITED"
                or not isinstance(transition, dict)
                or collected_payload[
                    "public_observation_transition_sha256"
                ] != sha256_json(transition)
                or not isinstance(collection, dict)
            ):
                raise SchedulerError(
                    "collected observations lack their write-ahead transition"
                )
            result = collection.get("result")
            native_receipts = collection.get(
                "native_public_observation_receipt_sha256s"
            )
            host_transcript_path = collection.get(
                "host_transcript_path"
            )
            provider_outcome = collection.get(
                "structured_provider_outcome"
            )
            if (
                not isinstance(result, dict)
                or result.get("kind") not in TERMINAL_RESULT_KINDS
                or not isinstance(native_receipts, list)
                or host_transcript_path
                != attempt["host_transcript_path"]
                or provider_outcome
                not in {
                    "completed",
                    "capacity",
                    "rate_limit",
                    "provider_failure",
                    "containment_fault",
                }
            ):
                raise SchedulerError(
                    "collected observation projection is malformed"
                )
            authoritative_receipts = (
                validate_public_observation_transition(
                    transition,
                    attempt_id=str(attempt_id),
                    generation_id=str(attempt["generation_id"]),
                    game=str(attempt["game"]),
                    frontier_sha256=str(
                        attempt["frontier_sha256"]
                    ),
                    parent_checkpoint_sha256=str(
                        attempt["parent_checkpoint_sha256"]
                    ),
                    host_transcript_path=str(
                        attempt["host_transcript_path"]
                    ),
                    result_kind=str(result["kind"]),
                    receipt_sha256s=tuple(native_receipts),
                    reopen_receipts=True,
                )
            )
            attempt["public_observation_receipt_sha256s"] = (
                authoritative_receipts
            )
            attempt["collection_result_kind"] = result["kind"]
            attempt["structured_provider_outcome"] = provider_outcome
            attempt["phase"] = "COLLECTED"
            continue
        if kind == "ATTEMPT_COLLECTION_REJECTED" and "result" in payload:
            rejected_payload = _strict_keys(
                payload,
                {"attempt_id", "reason", "result"},
                "ATTEMPT_COLLECTION_REJECTED",
            )
            result = rejected_payload["result"]
            if (
                attempt is None
                or attempt["phase"] != "EXITED"
                or not isinstance(rejected_payload["reason"], str)
                or not isinstance(result, dict)
                or result.get("kind") != "infrastructure"
                or any(
                    result.get(name) is not None
                    for name in (
                        "candidate",
                        "wip",
                        "blocker",
                        "native_sidecar_request_draft",
                    )
                )
            ):
                raise SchedulerError(
                    "typed collection rejection is malformed"
                )
            attempt["collection_result_kind"] = "infrastructure"
            attempt["phase"] = "COLLECTION_REJECTED"
            continue
        if kind == "ATTEMPT_TORN_DOWN" and "teardown" in payload:
            torn_down_payload = _strict_keys(
                payload,
                {"attempt_id", "teardown"},
                "ATTEMPT_TORN_DOWN",
            )
            teardown = torn_down_payload["teardown"]
            if (
                attempt is None
                or attempt["phase"]
                not in {"COLLECTED", "COLLECTION_REJECTED"}
                or not isinstance(teardown, dict)
                or not _is_sha256(teardown.get("proof_sha256"))
                or teardown.get("container_inspect_absent") is not True
                or teardown.get("container_top_absent") is not True
                or teardown.get("identity_query_empty") is not True
                or teardown.get("no_descendants") is not True
                or teardown.get("app_server_process_absent") is not True
                or teardown.get(
                    "app_server_process_group_absent"
                ) is not True
            ):
                raise SchedulerError(
                    "typed teardown proof is malformed or incomplete"
                )
            attempt["typed_teardown"] = True
            attempt["phase"] = "TORN_DOWN"
            continue
        if kind == "ATTEMPT_DRAINING":
            if attempt is None:
                raise SchedulerError("draining event targets unknown attempt")
            lane = lanes[str(attempt["game"])]
            if (
                lane["active_attempt_id"] != attempt_id
                or lane["draining"]
                or attempt["phase"] != "RUNNING"
            ):
                raise SchedulerError("draining transition is invalid")
            lane["draining"] = True
            attempt["phase"] = "DRAINING"
            continue
        if kind in _LIFECYCLE_TRANSITIONS:
            if attempt is None:
                raise SchedulerError(
                    f"{kind} targets an unknown scheduler attempt"
                )
            allowed, next_phase = _LIFECYCLE_TRANSITIONS[kind]
            if attempt["phase"] not in allowed:
                raise SchedulerError(
                    f"{kind} violates the attempt lifecycle"
                )
            if next_phase is not None:
                attempt["phase"] = next_phase
            continue
        if "SIGNAL" in kind or "INTERRUPT" in kind or "KILL" in kind:
            raise SchedulerError(
                "journal records a forbidden scheduler soft-deadline signal"
            )
        if kind == "ATTEMPT_RESULT":
            if (
                attempt is None
                or attempt["settled"]
                or attempt["phase"] != "TORN_DOWN"
            ):
                raise SchedulerError("attempt result is missing or duplicated")
            result_kind = payload.get("kind")
            charged_units = payload.get("authenticated_cost_units")
            try:
                derived_charged_units = charge_to_units(
                    payload.get("cost_used")
                )
            except SchedulerError:
                derived_charged_units = None
            if (
                result_kind not in TERMINAL_RESULT_KINDS
                or not _is_int(charged_units)
                or charged_units != derived_charged_units
                or payload.get("budget_reservation_id")
                != attempt["reservation_id"]
                or payload.get("scheduler_decision_id")
                != attempt["decision_id"]
                or not isinstance(payload.get("reason"), str)
                or len(str(payload.get("reason"))) > 4096
                or "\x00" in str(payload.get("reason"))
                or (
                    result_kind != "blocker"
                    and payload.get("blocker") is not None
                )
            ):
                raise SchedulerError(
                    "attempt result lacks authenticated scheduler settlement"
                )
            if (
                attempt.get("terminal_status")
                not in {"exited", "containment_fault"}
                or attempt.get("collection_result_kind") is None
                or attempt.get("typed_teardown") is not True
            ):
                raise SchedulerError(
                    "attempt settlement lacks typed terminal, collection, "
                    "or teardown evidence"
                )
            collection_result_kind = attempt.get(
                "collection_result_kind"
            )
            authoritative_observations = tuple(
                attempt.get(
                    "public_observation_receipt_sha256s", ()
                )
            )
            if (
                collection_result_kind is not None
                and collection_result_kind != result_kind
            ):
                raise SchedulerError(
                    "attempt result differs from its collected observation "
                    "transition"
                )
            if (
                result_kind
                not in PUBLIC_OBSERVATION_AUTHORITY_RESULT_KINDS
                and authoritative_observations
            ):
                raise SchedulerError(
                    "non-authoritative result gained observation authority"
                )
            protocol_invalid_result = attempt.get(
                "protocol_invalid_result"
            )
            if (
                protocol_invalid_result is not None
                and any(
                    payload.get(name)
                    != protocol_invalid_result.get(name)
                    for name in (
                        "kind",
                        "cost_used",
                        "reason",
                        "candidate",
                        "wip",
                        "blocker",
                        "native_sidecar_request_draft",
                    )
                )
            ):
                raise SchedulerError(
                    "settled protocol-invalid result changed after teardown"
                )
            blocker_code: str | None = None
            if result_kind == "blocker":
                blocker_code = _verify_host_blocker_result(
                    payload,
                    attempt=attempt,
                    campaign_id=str(genesis["campaign_id"]),
                )
            raw_native_request = payload.get(
                "native_sidecar_request_draft"
            )
            native_request_draft = (
                None
                if raw_native_request is None
                else native_sidecar_request_draft_from_dict(
                    raw_native_request
                )
            )
            if (
                native_request_draft is not None
                and (
                    result_kind != "clean_no_progress"
                    or native_request_draft.native_attempt_id
                    != attempt_id
                    or native_request_draft.game
                    != attempt["game"]
                    or native_request_draft.frontier_sha256
                    != attempt["frontier_sha256"]
                    or native_request_draft.parent_checkpoint_sha256
                    != attempt["parent_checkpoint_sha256"]
                )
            ):
                raise SchedulerError(
                    "native sidecar request draft is not its clean result"
                )
            budget = settle_budget(
                budget,
                reservation_id=str(attempt["reservation_id"]),
                attempt_id=str(attempt_id),
                charged_units=int(charged_units),
            )
            attempt["settled"] = True
            attempt["native_sidecar_request_draft"] = (
                native_request_draft
            )
            settlements += 1
            lane = lanes[str(attempt["game"])]
            if (
                result_kind
                in PUBLIC_OBSERVATION_AUTHORITY_RESULT_KINDS
                and collection_result_kind is not None
            ):
                lane[
                    "public_observation_receipt_sha256s"
                ] = sorted({
                    *lane[
                        "public_observation_receipt_sha256s"
                    ],
                    *authoritative_observations,
                })
            transition = terminal_policy_transition(result_kind)
            exposure_detected = (
                attempt.get("terminal_status")
                == "containment_fault"
                or attempt.get("structured_provider_outcome")
                == "containment_fault"
            )
            current_attempt_wip = (
                _validate_terminal_wip(
                    value=payload.get("wip"),
                    lane=lane,
                    attempt=attempt,
                    campaign_id=str(genesis["campaign_id"]),
                    cost_used=payload.get("cost_used"),
                )
                if result_kind == "clean_no_progress"
                else payload.get("wip")
            )
            lane["wip"] = reduce_terminal_wip(
                transition=transition,
                prior_wip=lane["wip"],
                current_attempt_wip=current_attempt_wip,
                exposure_detected=exposure_detected,
            )
            if transition.next_lane_phase == "PROMOTING":
                candidate = payload.get("candidate")
                if not isinstance(candidate, dict):
                    raise SchedulerError(
                        "candidate result lacks exact candidate evidence"
                    )
                manifest_path = candidate.get(
                    "candidate_manifest_path"
                )
                manifest_sha = candidate.get(
                    "candidate_manifest_sha256"
                )
                if (
                    candidate.get("game") != attempt["game"]
                    or candidate.get("from_level")
                    != int(attempt["target_level"]) - 1
                    or candidate.get("to_level")
                    != attempt["target_level"]
                    or candidate.get("parent_checkpoint_sha256")
                    != attempt["parent_checkpoint_sha256"]
                ):
                    raise SchedulerError(
                        "candidate result is not the reserved exact edge"
                    )
                _verify_regular_sha256(
                    manifest_path,
                    manifest_sha,
                    label="candidate manifest",
                    maximum=16 * 1024 * 1024,
                )
                attempt["candidate"] = True
                attempt["candidate_evidence"] = {
                    "candidate_manifest_path": manifest_path,
                    "candidate_manifest_sha256": manifest_sha,
                }
                if payload.get("wip") is not None:
                    raise SchedulerError("candidate result carries mutable WIP")
                attempt["phase"] = "PROMOTING"
            else:
                lane["active_attempt_id"] = None
                lane["draining"] = False
                if transition.retry_coordinate_delta == 1:
                    if (
                        lane["frontier_sha256"]
                        != attempt["frontier_sha256"]
                        or lane["no_progress"]
                        != attempt["no_progress_before"]
                    ):
                        raise SchedulerError(
                            "clean result is not the next exact-frontier "
                            "retry coordinate"
                        )
                    settlement = CleanProposerSettlement(
                        schema=1,
                        game=str(attempt["game"]),
                        frontier_sha256=str(
                            attempt["frontier_sha256"]
                        ),
                        parent_checkpoint_sha256=str(
                            attempt["parent_checkpoint_sha256"]
                        ),
                        attempt_id=str(attempt_id),
                        scheduler_decision_id=str(
                            attempt["decision_id"]
                        ),
                        no_progress_before=int(
                            attempt["no_progress_before"]
                        ),
                        effort=attempt["effort"],  # type: ignore[arg-type]
                        soft_allocation_seconds=int(
                            attempt["soft_allocation_seconds"]
                        ),
                        requested_wip_mode=attempt[
                            "requested_wip_mode"
                        ],  # type: ignore[arg-type]
                        supervisory_handoff_sha256=attempt[
                            "supervisory_handoff_sha256"
                        ],  # type: ignore[arg-type]
                        result_sequence=int(event["sequence"]),
                        result_digest=str(event["digest"]),
                    )
                    validate_clean_proposer_settlement(settlement)
                    clean_rows = lane[
                        "clean_proposer_settlements"
                    ]
                    if not isinstance(clean_rows, list):
                        raise SchedulerError(
                            "clean settlement reducer state is malformed"
                        )
                    clean_rows.append(settlement)
                    lane["no_progress"] = int(lane["no_progress"]) + 1
                elif payload.get("wip") is not None:
                    raise SchedulerError(
                        "non-clean terminal result carries WIP"
                    )
                if transition.next_lane_phase == "BLOCKED":
                    if blocker_code is None:
                        raise SchedulerError(
                            "blocked transition lacks host blocker authority"
                        )
                    lane["blocked_reason"] = (
                        HOST_BLOCKER_REASON_PREFIX + blocker_code
                    )
                attempt["phase"] = "FINISHED"
            continue
        if kind == "PROMOTION_COMMITTED":
            if (
                attempt is None
                or not attempt["candidate"]
                or attempt["promoted"]
                or attempt["phase"] != "PROMOTING"
            ):
                raise SchedulerError("promotion has no exact candidate result")
            lane = lanes[str(attempt["game"])]
            from_level = payload.get("from_level")
            to_level = payload.get("to_level")
            checkpoint_sha = payload.get("checkpoint_sha256")
            source_sha = payload.get("source_tree_sha256")
            source_path = payload.get("source_path")
            checkpoint_path = payload.get("checkpoint_path")
            candidate_evidence = attempt["candidate_evidence"]
            if not isinstance(candidate_evidence, dict):
                raise SchedulerError(
                    "promotion candidate evidence was not retained"
                )
            if (
                from_level != int(attempt["target_level"]) - 1
                or to_level != attempt["target_level"]
                or from_level != lane["reached"]
                or to_level > lane["target"]
                or payload.get("parent_checkpoint_sha256")
                != attempt["parent_checkpoint_sha256"]
                or payload.get("candidate_manifest_sha256")
                != candidate_evidence["candidate_manifest_sha256"]
                or not _is_sha256(checkpoint_sha)
                or not _is_sha256(source_sha)
                or not isinstance(source_path, str)
                or not isinstance(checkpoint_path, str)
            ):
                raise SchedulerError("promotion is not an exact K→K+1 edge")
            _validate_promoted_artifacts(
                game=str(attempt["game"]),
                authoritative_target=int(
                    attempt["authoritative_target"]
                ),
                to_level=int(to_level),
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                source_path=source_path,
                source_tree_sha256=source_sha,
            )
            old_frontier_sha256 = str(lane["frontier_sha256"])
            complexity_rounds[:] = [
                replace(item, invalidated=True)
                if (
                    item.game == attempt["game"]
                    and item.frontier_sha256 == old_frontier_sha256
                )
                else item
                for item in complexity_rounds
            ]
            for assignment_id, auxiliary in tuple(
                auxiliary_assignments.items()
            ):
                if (
                    auxiliary.game == attempt["game"]
                    and auxiliary.frontier_sha256
                    == old_frontier_sha256
                ):
                    auxiliary_assignments[assignment_id] = replace(
                        auxiliary, invalidated=True
                    )
            # Preserve request admission in the immutable journal, but remove
            # stale-frontier briefs from every future scheduler snapshot.
            sidecar_requests[:] = [
                request
                for request in sidecar_requests
                if not (
                    request.game == attempt["game"]
                    and request.frontier_sha256
                    == old_frontier_sha256
                )
            ]
            lane.update(
                reached=to_level,
                no_progress=0,
                parent_checkpoint_sha256=checkpoint_sha,
                parent_checkpoint_path=checkpoint_path,
                parent_source_path=source_path,
                parent_source_tree_sha256=source_sha,
                frontier_sha256=_frontier_digest(
                    str(attempt["game"]), int(to_level), str(checkpoint_sha)
                ),
                active_attempt_id=None,
                draining=False,
                blocked_reason=None,
                wip=None,
                clean_proposer_settlements=[],
                public_observation_receipt_sha256s=[],
            )
            attempt["candidate"] = False
            attempt["promoted"] = True
            attempt["phase"] = "FINISHED"
            promotions += 1
            continue
        if kind == "PROMOTION_FAILED":
            if (
                attempt is None
                or not attempt["candidate"]
                or attempt["phase"] != "PROMOTING"
                or set(payload) != {"attempt_id", "code"}
                or payload.get("code") not in PROMOTION_FAILURE_CODES
            ):
                raise SchedulerError("promotion failure has no candidate")
            lane = lanes[str(attempt["game"])]
            transition = promotion_failure_policy_transition()
            if (
                transition.next_lane_phase != "READY"
                or transition.retry_coordinate_delta != 0
                or transition.blocker_authority is not False
            ):
                raise SchedulerError(
                    "promotion failure policy is not retryable"
                )
            lane["active_attempt_id"] = None
            lane["draining"] = False
            attempt["candidate"] = False
            attempt["promoted"] = True
            attempt["phase"] = "FINISHED"
            continue
        raise SchedulerError(f"unknown scheduler journal event: {kind}")
    if (
        pending is not None
        and not allow_pending_decision
        and storage_incident is None
    ):
        raise SchedulerError(
            "campaign ends with an unconsumed scheduler decision"
        )
    if (
        pending_auxiliary is not None
        and not allow_pending_decision
        and storage_incident is None
    ):
        raise SchedulerError(
            "campaign ends with an unconsumed auxiliary decision"
        )
    validate_budget_state(budget)
    if budget.limit_units is not None:
        expected_live_ids = {
            str(attempt_id)
            for attempt_id, attempt in attempts.items()
            if not bool(attempt["settled"])
        } | {
            assignment.assignment_id
            for assignment in auxiliary_assignments.values()
            if assignment.phase in AUXILIARY_ACTIVE_PHASES
        }
        actual_live_ids = {
            item.attempt_id for item in budget.live_reservations
        }
        if actual_live_ids != expected_live_ids:
            raise SchedulerError(
                "live budget reservations do not match proposer/auxiliary "
                "occupancy"
            )
    return {
        "decisions": decisions,
        "reservations": reservations,
        "settlements": settlements,
        "promotions": promotions,
        "auxiliary_decisions": auxiliary_decisions,
        "auxiliary_reservations": auxiliary_reservations,
        "auxiliary_settlements": auxiliary_settlements,
        "auxiliary_admissions": auxiliary_admissions,
        "active_auxiliary_assignments": sorted(
            assignment.assignment_id
            for assignment in auxiliary_assignments.values()
            if assignment.phase in AUXILIARY_ACTIVE_PHASES
        ),
        "quarantined_auxiliary_assignments": sorted(
            assignment.assignment_id
            for assignment in auxiliary_assignments.values()
            if assignment.phase == "QUARANTINED"
        ),
        "complexity_profiles": sorted(
            item.profile.profile_id
            for item in complexity_rounds
            if not item.invalidated
        ),
        "pending_decision": None,
        "settled_cost_units": budget.settled_units,
        "live_reservation_units": sum(
            item.units for item in budget.live_reservations
        ),
        "limit_units": budget.limit_units,
        "cost_control_enabled": budget.limit_units is not None,
        "failure_operation_circuits":
            failure_operation_circuits,
        "failure_domain_circuits": failure_domain_circuits,
        "operator_incident": operator_incident,
        "substrate_incident": substrate_incident,
        "storage_incident": storage_incident,
        "storage_quiescence": storage_quiescence,
        # This count is reconstructed from policy journal transitions only.
        # It is deliberately not named ``solved_levels``: replay, taint,
        # schema-v2 boundary, and promotion-receipt verification belong to the
        # full runner/unified audit before a level may be called solved.
        "policy_promoted_levels": sum(
            int(lane["reached"]) for lane in lanes.values()
        ),
        "total_levels": sum(inventory.values()),
    }


def validate_journal_event_sequence(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Validate an already reopened immutable scheduler journal sequence."""

    if not events:
        raise SchedulerError("scheduler journal is empty")
    return _audit_events(events, allow_pending_decision=True)


_CONTROL_FILES = (
    "arc/crack_lab/ARC_AGI3_CAMPAIGN_PLAN.md",
    "arc/crack_lab/arc_agi3_contiguous_scheduler.py",
    "arc/crack_lab/test_arc_agi3_contiguous_scheduler.py",
    "arc/crack_lab/arc_agi3_contiguous_runner.py",
    "arc/crack_lab/arc_agi3_contiguous_orchestrator.py",
    "arc/crack_lab/arc_agi3_contiguous_supervisor.py",
    "arc/crack_lab/arc_agi3_contiguous_taint.py",
    "arc/crack_lab/arc_agi3_proposer_boundary.py",
    "arc/crack_lab/arc_agi3_source_schema.py",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _control_hashes() -> dict[str, str]:
    root = _repository_root()
    result: dict[str, str] = {}
    for relative in _CONTROL_FILES:
        path = root / relative
        result[relative] = hashlib.sha256(
            _read_regular(path, maximum=64 * 1024 * 1024)
        ).hexdigest()
    return result


def audit_campaign(campaign_root: Path) -> dict[str, object]:
    """Derive a scheduler receipt bound to the full runner lifecycle."""

    root = Path(campaign_root).resolve()
    try:
        events = read_journal(root)
        summary = _audit_events(events)
        summary["journal_prefix"] = journal_prefix_status(root)
        try:
            import arc_agi3_contiguous_runner as Runner

            runner_receipt = (
                Runner.audit_runner_state_read_only(root)
            )
        except Exception as exc:
            raise SchedulerError(
                "full runner lifecycle audit failed: "
                f"{type(exc).__name__}"
            ) from exc
        if (
            runner_receipt.get("status") != "PASS"
            or runner_receipt.get("journal_event_count")
            != len(events)
            or runner_receipt.get("journal_head_sequence")
            != events[-1]["sequence"]
            or runner_receipt.get("journal_head_digest")
            != events[-1]["digest"]
            or not _is_sha256(
                runner_receipt.get("receipt_sha256")
            )
        ):
            raise SchedulerError(
                "runner lifecycle audit is not a same-head PASS"
            )
        runner_binding = {
            "kind": runner_receipt["kind"],
            "status": runner_receipt["status"],
            "receipt_sha256":
                runner_receipt["receipt_sha256"],
            "journal_event_count":
                runner_receipt["journal_event_count"],
            "journal_head_sequence":
                runner_receipt["journal_head_sequence"],
            "journal_head_digest":
                runner_receipt["journal_head_digest"],
            "state_sha256": runner_receipt["state_sha256"],
            "solved_levels": runner_receipt["solved_levels"],
            "total_levels": runner_receipt["total_levels"],
            "complete": runner_receipt["complete"],
        }
        result = {
            "schema": AUDIT_SCHEMA,
            "kind": "ARC_AGI3_CONTIGUOUS_SCHEDULER_AUDIT",
            "verdict": "PASS",
            "campaign_root": str(root),
            "policy_name": POLICY_NAME,
            "policy_sha256": SCHEDULER_POLICY_SHA256,
            "proposer_policy_sha256": PROPOSER_POLICY_SHA256,
            "journal_events": len(events),
            "journal_head_sequence": events[-1]["sequence"],
            "journal_head_digest": events[-1]["digest"],
            "control_files": _control_hashes(),
            "runner_lifecycle": runner_binding,
            "summary": summary,
            "findings": [],
        }
    except (SchedulerError, OSError) as exc:
        result = {
            "schema": AUDIT_SCHEMA,
            "kind": "ARC_AGI3_CONTIGUOUS_SCHEDULER_AUDIT",
            "verdict": "FAIL",
            "campaign_root": str(root),
            "policy_name": POLICY_NAME,
            "policy_sha256": SCHEDULER_POLICY_SHA256,
            "proposer_policy_sha256": PROPOSER_POLICY_SHA256,
            "journal_events": None,
            "journal_head_sequence": None,
            "journal_head_digest": None,
            "control_files": {},
            "runner_lifecycle": {},
            "summary": {},
            "findings": [f"{type(exc).__name__}: {exc}"],
        }
    result["receipt_sha256"] = sha256_json(result)
    return result


def _write_new_receipt(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o400)
    except OSError as exc:
        raise SchedulerError(
            f"audit output must be a new unaliased file: {path}"
        ) from exc
    payload = canonical_json(value) + b"\n"
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SchedulerError("short scheduler-audit receipt write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_audit_receipt(
    campaign_root: Path, receipt_path: Path
) -> dict[str, object]:
    retained = _read_json_regular(
        Path(receipt_path), maximum=16 * 1024 * 1024
    )
    receipt_sha = retained.get("receipt_sha256")
    body = {key: retained[key] for key in retained if key != "receipt_sha256"}
    if not _is_sha256(receipt_sha) or receipt_sha != sha256_json(body):
        raise SchedulerError("scheduler audit receipt hash is invalid")
    regenerated = audit_campaign(campaign_root)
    if regenerated != retained:
        raise SchedulerError(
            "scheduler audit receipt no longer matches campaign/control bytes"
        )
    if regenerated.get("verdict") != "PASS":
        raise SchedulerError("scheduler audit receipt is not PASS")
    return regenerated


def verify_pre_retention_audit_receipt(
    campaign_root: Path,
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
) -> dict[str, object]:
    """Verify a terminally bound pre-cleanup scheduler PASS.

    The ordinary verifier deliberately reopens transient WIP trees.  Once the
    separately audited terminal-retention transaction has compacted admitted
    receipts and removed all attempt generations, those trees must not be
    recreated merely to make an audit pass.  This verifier is therefore valid
    only when its exact receipt hash is supplied from that retention binding.
    It reauthenticates the unchanged journal head/prefix and current control
    bytes, while the runner and retention audits own reducer replay and compact
    evidence completeness respectively.
    """

    if not _is_sha256(expected_receipt_sha256):
        raise SchedulerError(
            "pre-retention scheduler binding hash is malformed"
        )
    retained = _read_json_regular(
        Path(receipt_path), maximum=16 * 1024 * 1024
    )
    receipt_sha = retained.get("receipt_sha256")
    body = {
        key: retained[key]
        for key in retained
        if key != "receipt_sha256"
    }
    root = Path(campaign_root).resolve()
    events = read_journal(root)
    if not events:
        raise SchedulerError(
            "pre-retention scheduler audit found no journal"
        )
    summary = retained.get("summary")
    runner_lifecycle = retained.get("runner_lifecycle")
    if (
        receipt_sha != expected_receipt_sha256
        or not _is_sha256(receipt_sha)
        or receipt_sha != sha256_json(body)
        or retained.get("verdict") != "PASS"
        or retained.get("campaign_root") != str(root)
        or retained.get("policy_name") != POLICY_NAME
        or retained.get("policy_sha256") != SCHEDULER_POLICY_SHA256
        or retained.get("proposer_policy_sha256")
        != PROPOSER_POLICY_SHA256
        or retained.get("journal_events") != len(events)
        or retained.get("journal_head_sequence")
        != events[-1]["sequence"]
        or retained.get("journal_head_digest") != events[-1]["digest"]
        or retained.get("control_files") != _control_hashes()
        or not isinstance(runner_lifecycle, dict)
        or runner_lifecycle.get("status") != "PASS"
        or not _is_sha256(
            runner_lifecycle.get("receipt_sha256")
        )
        or runner_lifecycle.get("journal_event_count")
        != len(events)
        or runner_lifecycle.get("journal_head_sequence")
        != events[-1]["sequence"]
        or runner_lifecycle.get("journal_head_digest")
        != events[-1]["digest"]
        or not _is_sha256(
            runner_lifecycle.get("state_sha256")
        )
        or not isinstance(summary, dict)
        or summary.get("journal_prefix")
        != journal_prefix_status(root)
        or retained.get("findings") != []
    ):
        raise SchedulerError(
            "pre-retention scheduler audit no longer matches its "
            "terminal journal/control binding"
        )
    return retained


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or verify the automatic ARC-AGI-3 contiguous scheduler"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "policy",
        help="print the canonical game-independent scheduler policy",
    )
    audit = subparsers.add_parser("audit")
    audit.add_argument("--campaign-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--campaign-root", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "policy":
        print(
            json.dumps(
                {
                    "policy": policy_projection(),
                    "policy_sha256": SCHEDULER_POLICY_SHA256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "audit":
        result = audit_campaign(args.campaign_root)
        _write_new_receipt(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["verdict"] == "PASS" else 1
    try:
        result = verify_audit_receipt(
            args.campaign_root, args.receipt
        )
    except SchedulerError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
