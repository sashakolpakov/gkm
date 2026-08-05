#!/usr/bin/env python3
"""Canonical conformance suite for the ARC-AGI-3 contiguous campaign.

This is the single aggregate release-control entry point.  Focused unit tests
remain useful, but launch admission consumes only this suite's exact,
machine-readable PASS artifact.  The aggregate does not trust pytest's exit
code alone: it requires an exact one-to-one registry/collection match, records
every setup/call/teardown outcome, and fails on skips, duplicates, missing
cases, or unexpected cases.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence


SCHEMA = 3
KIND = "arc_agi3_contiguous_conformance"
EXPECTED_GAMES = 25
EXPECTED_LEVELS = 183
STATUS_VALUES = {"PASS", "FAIL", "SKIP", "MISSING", "DUPLICATE"}
COMPONENT_STATUS_VALUES = {
    "PASS",
    "FAIL",
    "SKIP",
    "XFAIL",
    "XPASS",
    "MISSING",
}
MAX_COMPONENT_TEST_FILES = 64
MAX_COMPONENT_TEST_CASES = 4096
MAX_COMPONENT_NODEID_BYTES = 2048
MAX_COMPONENT_INVENTORY_BYTES = 2 * 1024 * 1024
MAX_LOADED_CONTROL_MODULES = 512
MAX_WORKSPACE_ROOT_ENTRIES = 4096
MAX_WORKSPACE_ENTRY_NAME_BYTES = 255
MAX_SCENARIO_DRIVER_OUTPUT_BYTES = 1024 * 1024
SCENARIO_DRIVER_CONTROL_PATH = (
    "arc/crack_lab/arc_agi3_contiguous_scenario_driver.py"
)
EXPECTED_PRODUCTION_SCENARIO_IDS = tuple(
    f"S{number:02d}" for number in range(1, 13)
)


class ConformanceError(RuntimeError):
    """The canonical registry, execution, or artifact failed closed."""


@dataclass(frozen=True)
class Invariant:
    invariant_id: str
    component: str
    nodeid: str
    claim: str


# Each release-level invariant has exactly one owner test.  Focused suites may
# exercise related behavior too; only this registry defines canonical release
# representation, preventing a large green aggregate from double-counting one
# behavior while silently omitting another.
INVARIANTS = (
    Invariant(
        "conformance_rejects_mid_suite_control_mutation",
        "conformance",
        "arc/crack_lab/test_arc_agi3_contiguous_conformance.py::"
        "test_mid_suite_control_mutation_forces_fail",
        "The canonical suite binds one immutable loaded/start/end control snapshot and fails on TOCTOU mutation.",
    ),
    Invariant(
        "workspace_root_pre_post_is_stable",
        "conformance",
        "arc/crack_lab/test_arc_agi3_contiguous_conformance.py::"
        "test_workspace_root_leak_forces_fail",
        "The canonical suite records a bounded workspace-root inventory before and after execution and fails if any test leaves, removes, or substitutes a root entry.",
    ),
    Invariant(
        "suite_scratch_is_empty_and_confined",
        "conformance",
        "arc/crack_lab/test_arc_agi3_contiguous_conformance.py::"
        "test_suite_scratch_gate_rejects_unrelated_entry_without_removing_it",
        "The canonical suite begins and ends with empty private scratch; an unrelated entry fails closed and is not silently erased.",
    ),
    Invariant(
        "inventory_metadata_is_exact_and_sealed",
        "conformance",
        "arc/crack_lab/test_arc_agi3_contiguous_conformance.py::"
        "test_authoritative_inventory_metadata_is_exact_public_25_183_input",
        "The 25-game/183-level inventory is derived only from the literal public metadata allowlist sealed into the descriptor-rooted control snapshot.",
    ),
    Invariant(
        "terminal_launch_authority_binds_release_and_image",
        "conformance",
        "arc/crack_lab/test_arc_agi3_contiguous_conformance.py::"
        "test_terminal_launch_authority_binds_release_and_image",
        "Launch authority remains closed until the sealed driver reopens exact machine-observed PASS receipts for S01-S12, then also reopens the exact 183-boundary release and binds the immutable image.",
    ),
    Invariant(
        "scenario_driver_never_synthesizes_pass",
        "conformance",
        "arc/crack_lab/test_arc_agi3_contiguous_scenario_driver.py::"
        "test_missing_production_observers_emit_typed_blocked_receipts",
        "The production S01-S12 run/verify driver leaves unavailable observers typed BLOCKED and cannot derive PASS from caller-supplied status metadata.",
    ),
    Invariant(
        "formal_operator_rejects_cli_before_mutation",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_formal_operator_rejects_cli_and_preflight_before_mutation",
        "The formal operator rejects unknown/duplicate CLI input and completes terminal preflight before campaign mutation.",
    ),
    Invariant(
        "formal_operator_process_lease_covers_mutable_path",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_formal_operator_holds_one_process_lease_across_mutable_path",
        "After immutable preflight, the formal operator holds one kernel-enforced, PID/start-bound, host-authenticated heartbeat lease across its mutable execution path and rejects a live second owner.",
    ),
    Invariant(
        "production_host_children_use_campaign_ledger",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_production_host_child_ledger_is_campaign_bound",
        "Every production Docker, auxiliary, and meta driver child is launched through the sole managed command runner whose authenticated recovery ledger is fixed beneath the exact campaign root.",
    ),
    Invariant(
        "host_child_ledger_is_terminally_reopened",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_host_child_ledger_audit_requires_quiescent_accounting",
        "Startup and terminal authority reopen the authenticated complete host-child ledger and reject any active, pending, unaccounted, dead, reused, or cleanup-unverified invocation.",
    ),
    Invariant(
        "storage_incident_reaches_exact_operator_terminal",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_storage_incident_has_reachable_exact_terminal_projection",
        "An authenticated journal, inode, free-space, or ENOSPC latch reaches the exact non-authoritative JOURNAL_OR_STORAGE_EXHAUSTED operator terminal instead of sleeping or redispatching.",
    ),
    Invariant(
        "watchdog_exact_operator_restart_is_reachable",
        "watchdog",
        "arc/crack_lab/test_arc_agi3_contiguous_watchdog.py::"
        "test_watchdog_restarts_crashed_operator_and_returns_exact_terminal",
        "The independent watchdog invokes the exact sealed operator command, restarts a crashed process, and stops only after reopening exact terminal authority.",
    ),
    Invariant(
        "watchdog_restart_circuit_exhausts_to_incident",
        "watchdog",
        "arc/crack_lab/test_arc_agi3_contiguous_watchdog.py::"
        "test_watchdog_restart_exhaustion_is_durable_human_intervention",
        "The watchdog restart circuit is finite and durably converts exhaustion into OPERATOR_INCIDENT with a fixed surfaced human-intervention request.",
    ),
    Invariant(
        "watchdog_long_run_heartbeat_phase_is_monotone",
        "watchdog",
        "arc/crack_lab/test_arc_agi3_contiguous_watchdog.py::"
        "test_watchdog_does_not_reapply_startup_deadline_after_active_lease",
        "After observing its exact ACTIVE operator lease, the watchdog never reapplies the expired startup deadline: transient reads use heartbeat staleness and RELEASED receives a bounded terminal-output grace period.",
    ),
    Invariant(
        "watchdog_long_child_collects_delayed_terminal",
        "watchdog",
        "arc/crack_lab/test_arc_agi3_contiguous_watchdog.py::"
        "test_watchdog_real_child_survives_expired_startup_and_delayed_terminal",
        "A real long-lived child survives beyond the retired startup deadline through injected authenticated-lease read loss and RELEASED reconciliation, then has its delayed exact terminal stdout collected.",
    ),
    Invariant(
        "post_incident_meta_is_once_and_quarantine_only",
        "supervisor",
        "arc/crack_lab/test_arc_agi3_contiguous_supervisor.py::"
        "test_post_incident_meta_diagnostic_is_once_and_quarantine_only",
        "A durable substrate incident admits at most one bounded sealed meta-proposer invocation, whose result has no scheduler, solver, WIP, cost, retry, dispatch, or promotion authority.",
    ),
    Invariant(
        "post_incident_meta_formal_path_is_reachable",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_latched_substrate_meta_path_is_reachable_and_no_authority",
        "The formal operator reaches the sealed meta-proposer only from an authenticated controller-substrate incident and rejects mutation of campaign authority.",
    ),
    Invariant(
        "post_incident_meta_resume_requires_fresh_probe_chain",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_meta_recovery_requires_fresh_pass_journal_before_resume",
        "A quarantine-only meta recommendation can resume work only through the trusted runner's exact recovery-authorized, fresh-health-PASS, and one-shot-resume journal chain without changing solver, WIP, cost, lane, attempt, or promotion authority.",
    ),
    Invariant(
        "post_incident_meta_identity_survives_recovery_events",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_meta_projection_is_stable_after_recovery_journal_events",
        "The meta episode binds the exact OPERATOR_INCIDENT event rather than a moving journal head, so authorization/probe crash recovery cannot manufacture a second invocation or invalidate its first request.",
    ),
    Invariant(
        "post_incident_meta_is_once_per_distinct_incident",
        "supervisor",
        "arc/crack_lab/test_arc_agi3_contiguous_supervisor.py::"
        "test_post_incident_meta_allows_one_episode_per_distinct_incident",
        "Each authenticated substrate incident has one immutable diagnostic episode while a later distinct incident can autonomously receive its own bounded episode under the finite campaign cap.",
    ),
    Invariant(
        "pilot_manifest_executes_exact_empty_root_order",
        "pilot",
        "arc/crack_lab/test_arc_agi3_contiguous_pilot.py::"
        "test_frozen_pilot_executor_runs_exact_order_and_unlocks_gate",
        "The isolated pilot mode executes exact ft09 then lp85 full-game runs from four empty roots each and emits one authenticated noncanonical launch-gate receipt.",
    ),
    Invariant(
        "pilot_gate_requires_real_meta_handoff",
        "pilot",
        "arc/crack_lab/test_arc_agi3_contiguous_pilot.py::"
        "test_pilot_gate_rejects_no_real_meta_handoff",
        "Two otherwise passing pilot runs cannot unlock full launch without one real sealed quarantine-only meta-proposer handoff.",
    ),
    Invariant(
        "pilot_completed_runs_recover_without_reexecution",
        "pilot",
        "arc/crack_lab/test_arc_agi3_contiguous_pilot.py::"
        "test_pilot_controller_recovers_existing_runs_without_reexecution",
        "A controller restart reopens authenticated completed pilot runs and regenerates only the missing gate; it cannot rerun an already completed lineage or require operator steering.",
    ),
    Invariant(
        "full_launch_reopens_ordered_pilot_gate",
        "supervisor",
        "arc/crack_lab/test_arc_agi3_contiguous_supervisor.py::"
        "test_full_launch_reopens_exact_ordered_pilot_gate",
        "Full 25-game launch reopens and binds the exact live ft09-to-lp85 pilot gate, image, control contract, production-stack attestation, and real meta handoff.",
    ),
    Invariant(
        "inventory_exact_25_183",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_authoritative_inventory_six_disjoint_bound_lanes",
        "The scheduler admits the authoritative 25-game/183-level inventory.",
    ),
    Invariant(
        "source_schema_and_blank_scaffold_are_exact",
        "source_schema",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_production_l1_uses_only_control_hashed_blank_scaffold",
        "L1 starts from only the control-hashed source-schema-valid blank scaffold.",
    ),
    Invariant(
        "provider_usage_is_typed_and_rotation_closed",
        "app_server_transport",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_provider_usage_rejects_stale_rotation_and_mixed_units",
        "Authenticated provider windows reject stale, rotated, or mixed-unit settlement.",
    ),
    Invariant(
        "scheduler_retry_policy_is_monotone",
        "scheduler",
        "arc/crack_lab/test_arc_agi3_contiguous_scheduler.py::"
        "test_retry_ladder_is_monotone_and_enters_long_coherence",
        "The canonical retry ladder is monotone and enters bounded long coherence.",
    ),
    Invariant(
        "supervision_is_game_agnostic_receipt_reduction",
        "scheduler",
        "arc/crack_lab/test_arc_agi3_contiguous_scheduler.py::"
        "test_supervision_loop_is_a_game_agnostic_receipt_reducer",
        "Terminal classification, escalation, admission, promotion, and refill are deterministic receipt transitions with no semantic or operator steering.",
    ),
    Invariant(
        "scheduler_decision_binds_snapshot_and_budget",
        "scheduler",
        "arc/crack_lab/test_arc_agi3_contiguous_scheduler.py::"
        "test_decision_binds_snapshot_and_reservation",
        "Each durable scheduler decision binds its exact snapshot and reservation.",
    ),
    Invariant(
        "auxiliary_round_obligation_is_unique",
        "scheduler",
        "arc/crack_lab/test_arc_agi3_contiguous_scheduler.py::"
        "test_next_round_diagnosis_cannot_be_dispatched_twice",
        "Only one nonterminal assignment may occupy a frontier/round/specialization obligation, including profile-less next-round diagnosis.",
    ),
    Invariant(
        "formal_auxiliary_backend_is_policy_selected_and_quarantine_only",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_formal_auxiliary_backend_executes_only_policy_selected_quarantine_path",
        "The formal operator executes the n>=5 scheduler-selected sidecar through one attested fixed-argv driver; no caller selects its game, effort, or specialization. Driver paths are component-confined beneath a pinned assignment descriptor, canonical receipts are verified from one stable read and rebound before every recovered-phase call, streams and canonical responses have exact durable byte bindings, and Socratic output remains quarantine-only until host admission.",
    ),
    Invariant(
        "scheduler_audit_reopens_promotion_artifacts",
        "scheduler",
        "arc/crack_lab/test_arc_agi3_contiguous_scheduler.py::"
        "test_audit_reopens_promoted_artifacts_and_rejects_missing_path",
        "Scheduler audit reopens promoted paths and rejects hash-only evidence.",
    ),
    Invariant(
        "terminal_wip_reopens_complete_evidence",
        "scheduler",
        "arc/crack_lab/test_arc_agi3_contiguous_scheduler.py::"
        "test_complete_terminal_wip_audit_reopens_every_bound_artifact",
        "The authenticated scheduler journal phase reducer reopens reusable terminal WIP source, state, thread, taint, token, and typed provider evidence, and each independently mutated artifact forces phase validation to fail.",
    ),
    Invariant(
        "terminal_wip_rejects_nonexistent_tree",
        "scheduler",
        "arc/crack_lab/test_arc_agi3_contiguous_scheduler.py::"
        "test_audit_rejects_nonexistent_terminal_wip_tree",
        "A hash-only terminal WIP whose retained tree does not exist is inadmissible.",
    ),
    Invariant(
        "terminal_wip_rejects_provider_settlement_mutation",
        "scheduler",
        "arc/crack_lab/test_arc_agi3_contiguous_scheduler.py::"
        "test_audit_rejects_forged_terminal_provider_settlement",
        "A terminal WIP cannot retain a mutated provider settlement behind a recomputed file hash.",
    ),
    Invariant(
        "unlimited_means_no_cost_cutoff",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_limit_none_disables_uniform_cost_cutoff",
        "limit=None removes proposer cost cutoff uniformly.",
    ),
    Invariant(
        "soft_deadline_drains_without_interrupt",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_soft_deadline_drains_lane_locally_and_refills_unrelated_lane",
        "Soft expiry drains and starts no new turn without signalling the active one.",
    ),
    Invariant(
        "real_cycle_rejects_stage_reordering",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_real_cycle_rejects_frozen_supervision_stage_reordering",
        "A real supervision cycle rejects any departure from the independently frozen stage order before campaign mutation.",
    ),
    Invariant(
        "containment_fault_has_infrastructure_precedence",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_containment_fault_precedes_blocker_candidate_and_no_progress",
        "Authenticated containment failure preserves taint separately but supersedes blocker, candidate, and clean-no-progress labels as noncounting infrastructure.",
    ),
    Invariant(
        "public_action_protocol_invalid_is_terminal_no_authority",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_public_action_protocol_invalid_revokes_all_lineage_authority_and_restart",
        "A rejected public action is a distinct protocol-invalid terminal outcome; containment, teardown, reducer replay, and scheduler audit preserve accounting while admitting no candidate, WIP, observation, sidecar, handoff, restart, or promotion authority.",
    ),
    Invariant(
        "blocker_claim_cross_product_is_noncounting",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_blocker_claim_negative_cross_product_is_noncounting_infrastructure",
        "Missing, malformed, unsigned, replayed, unknown, or wrongly bound blocker claims remain retryable noncounting infrastructure.",
    ),
    Invariant(
        "authenticated_blocker_recovery_is_idempotent",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_authenticated_blocker_recovers_idempotently_and_is_revalidated_closed",
        "A genuine host-authenticated blocker survives exact recovery once and its external lane-stopping receipt is revalidated while closed.",
    ),
    Invariant(
        "orphan_materialization_recovers",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_crash_after_input_materialization_recovers_reserved_identity",
        "A crash during input materialization recovers the durable identity.",
    ),
    Invariant(
        "runner_exact_lifecycle_reaches_boundary",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_exact_lifecycle_reaches_one_authoritative_game_boundary",
        "The unified runner lifecycle reaches a boundary only through collection, teardown, result, and promotion.",
    ),
    Invariant(
        "runner_journal_audit_is_full_and_read_only",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_read_only_runner_audit_reuses_full_reducer_without_writes",
        "Independent runner audit reuses the full reducer without constructing locks or mutating evidence.",
    ),
    Invariant(
        "runner_supervision_history_is_bounded",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_escalation_is_frontier_bound_and_missing_wip_falls_back_to_exclude",
        "Repeated supervision cycles verify each durable decision exactly once, reuse no stale WIP, and cached reads add no authenticated work.",
    ),
    Invariant(
        "runner_source_cache_revalidates_exact_pointers",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_lane_source_cache_revalidates_replacement_and_mtime_spoof",
        "Cached source evidence is invalidated by replacement, inode/ctime change, or mtime spoofing.",
    ),
    Invariant(
        "runner_reducer_cache_is_return_isolated",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_reducer_checkpoint_is_not_mutable_through_returned_state",
        "A caller cannot mutate a returned state view to poison the authenticated reducer checkpoint.",
    ),
    Invariant(
        "runner_journal_cache_is_fail_closed_and_return_isolated",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_journal_cache_rejects_mutation_replacement_and_truncation",
        "Journal caching rejects mutation, replacement, and truncation, while public read and append values cannot alias cached authority.",
    ),
    Invariant(
        "runner_journal_event_read_is_descriptor_anchored",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_journal_event_read_is_descriptor_anchored_against_valid_race",
        "A valid in-place A-to-B rewrite cannot cache A's parsed bytes under B's later signature; suffix append and public-copy isolation remain exact.",
    ),
    Invariant(
        "runner_journal_pointer_signature_is_descriptor_anchored",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_journal_file_signature_rejects_regular_pointer_swap",
        "Cache-prefix signature revalidation binds one regular directory entry to one open event descriptor and rejects an A-to-B pointer swap.",
    ),
    Invariant(
        "runner_full_scale_journal_reads_only_suffix_bytes",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_full_scale_journal_authentication_reads_only_appended_suffix_bytes",
        "At the configured full journal-byte scale, an authenticated cached prefix is metadata-revalidated while event-byte parsing remains bounded to the new suffix.",
    ),
    Invariant(
        "runner_exact_authority_failure_stops_later_effects",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_exact_authority_failure_stops_all_later_lanes_and_effects",
        "An exact authority-gate failure aborts the complete cycle before any later lane poll, promotion, or filesystem effect.",
    ),
    Invariant(
        "runner_campaign_lock_serializes_threads_and_processes",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_foreign_thread_and_process_block_then_serialize_campaign_lock",
        "Same-thread recursive acquisition is rejected before flock, while foreign native threads and forked processes block and serialize on one campaign lock.",
    ),
    Invariant(
        "runner_bound_receipt_read_is_descriptor_stable",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_bound_receipt_hash_and_parse_share_one_stable_descriptor",
        "Bound receipt hashing and parsing use one stable descriptor and reject an in-read directory-entry replacement.",
    ),
    Invariant(
        "runner_auxiliary_effects_reopen_full_journal_prefix",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_auxiliary_prepare_launch_poll_admit_and_abort_use_full_prefix_gate",
        "Auxiliary prepare, launch, poll, admission, and abort each reopen the complete authenticated journal prefix before effect.",
    ),
    Invariant(
        "runner_promotion_effect_reauthenticates_full_journal_prefix",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_promotion_effect_gate_reauthenticates_complete_journal_prefix",
        "Promotion reopens the complete journal prefix immediately before the gate call, so prior-prefix mutation prevents gate invocation.",
    ),
    Invariant(
        "runner_rejects_wip_provider_path_substitution",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_runner_rejects_terminal_wip_provider_path_substitution",
        "WIP reuse requires the provider receipt at the exact canonical originating-generation path.",
    ),
    Invariant(
        "runner_rejects_wip_attempt_spec_substitution",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_runner_rejects_terminal_wip_attempt_spec_substitution",
        "WIP reuse binds every retained receipt to the true journaled originating attempt spec.",
    ),
    Invariant(
        "unified_audit_requires_runner_and_promotion_evidence",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_unified_audit_requires_runner_and_promotion_evidence",
        "A scheduler-only solved claim cannot pass without the full runner reducer and selected promotion evidence.",
    ),
    Invariant(
        "terminal_attempt_retention_is_compact_and_crash_recoverable",
        "runner",
        "arc/crack_lab/test_arc_agi3_terminal_retention.py::"
        "test_terminal_retention_crash_recovery_is_copy_before_purge",
        "Only after complete coverage, every admitted compact receipt is sealed before attempt generations are purged; an interrupted purge resumes exactly and no scratch, workspace, cache, transcript, stdout, or stderr survives.",
    ),
    Invariant(
        "terminal_retention_plan_covers_real_attempt_receipts",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_terminal_retention_plan_covers_real_compact_attempt_evidence",
        "The retention plan derived from a real completed attempt includes its input, launch, taint, hash, usage, binding, canary, and worker receipts while excluding raw model, app-server, stdout, and stderr streams.",
    ),
    Invariant(
        "pre_retention_scheduler_pass_is_exactly_terminal_bound",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_pre_retention_scheduler_pass_reopens_exact_terminal_binding",
        "Given the exact receipt hash supplied by terminal retention, a real same-head runner lifecycle yields a verifiable pre-cleanup scheduler PASS after transient generations are removed, while independent receipt-binding, journal-prefix, or journal-head mutations fail.",
    ),
    Invariant(
        "terminal_cross_audit_recovery_never_reopens_deleted_wip",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_terminal_scheduler_audit_resumes_partial_retention_without_wip",
        "A restarted terminal operator recognizes the durable retention intent, resumes from its exact pre-cleanup scheduler PASS, and never reruns a WIP-reopening audit against a partially purged generation set.",
    ),
    Invariant(
        "terminal_unified_audit_binds_pre_cleanup_scheduler_and_replay",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_complete_unified_audit_requires_retention_bound_scheduler_pass",
        "The complete unified audit requires the retention intent to bind both the exact pre-cleanup scheduler PASS and the exact independently reopened promotion/replay inventory before accepting post-purge evidence.",
    ),
    Invariant(
        "fresh_adapter_restart_recovers",
        "container_backend",
        "arc/crack_lab/test_arc_agi3_container_backend.py::"
        "test_adapter_rehydrates_fresh_instance_after_supervisor_crash",
        "A fresh backend adapter reconciles a journaled external lifecycle.",
    ),
    Invariant(
        "ambiguous_or_tampered_recovery_fails_closed",
        "container_backend",
        "arc/crack_lab/test_arc_agi3_container_backend.py::"
        "test_adapter_rehydration_rejects_ambiguous_or_tampered_identity",
        "Ambiguous labels or tampered recovery evidence cannot be rehydrated.",
    ),
    Invariant(
        "clean_typed_container_lifecycle",
        "container_backend",
        "arc/crack_lab/test_arc_agi3_container_backend.py::"
        "test_typed_adapter_full_mocked_lifecycle_and_token_secrecy",
        "The typed clean lifecycle binds image, compatibility closure, exact "
        "per-turn RPC/container receipt, output, transcript, and token "
        "secrecy; launch reopens the non-authoritative receipt and fails "
        "closed on drift.",
    ),
    Invariant(
        "container_backend_uses_canonical_terminal_precedence",
        "container_backend",
        "arc/crack_lab/test_arc_agi3_container_backend.py::"
        "test_backend_uses_runner_terminal_precedence_byte_exactly",
        "The production backend delegates every terminal/result cross-product byte-exactly to the runner's single containment-precedence transition.",
    ),
    Invariant(
        "host_terminal_parent_issues_authenticated_blocker",
        "container_backend",
        "arc/crack_lab/test_arc_agi3_container_backend.py::"
        "test_host_terminal_parent_issues_idempotent_authenticated_blocker",
        "The exact S03 owner replays the parent through the real one-client Arena RPC open/close boundary, derives the finite byte-idempotent HMAC receipt only from the trusted host result, and rejects synthetic, replayed, or mutated observations.",
    ),
    Invariant(
        "controller_container_recipe_is_pinned",
        "container_backend",
        "arc/crack_lab/test_arc_agi3_container_backend.py::"
        "test_formal_container_recipe_pins_base_and_nonroot_user",
        "The controller/container recipe pins its base and runs as the reviewed nonroot identity.",
    ),
    Invariant(
        "native_proposer_git_root_is_local_and_content_exclusive",
        "container_backend",
        "arc/crack_lab/test_arc_agi3_controller_guardian.py::"
        "test_native_proposer_workspace_is_its_only_git_root_and_broad_git_is_local",
        "The native proposer starts in one deterministic zero-source Git "
        "root equal to its private tmpfs workspace; Git discovery is ceiling-"
        "bound there, broad status/diff/log stay local, and campaign plans, "
        "sidecar/quarantine outputs, manuscript/comparator/benchmark files, "
        "parent-repository metadata, symlinks, hardlinks, alternates, and "
        "path escapes are absent.",
    ),
    Invariant(
        "controller_supply_chain_manifest_is_canonical",
        "container_backend",
        "arc/crack_lab/test_arc_agi3_controller_supply_chain.py::"
        "test_controller_supply_chain_manifest_and_recipe_are_exact_and_exclusive",
        "The controller image derives one exclusive canonical supply-chain manifest from descriptor-stable in-image bytes without a builder-specific parser extension.",
    ),
    Invariant(
        "controller_egress_default_deny_precedes_controller",
        "container_backend",
        "arc/crack_lab/test_arc_agi3_container_backend.py::"
        "test_controller_egress_probe_precedes_controller_creation_and_fails_closed",
        "The default-deny egress namespace must emit bound readiness and pass live allow-and-deny probes before the model controller can be created.",
    ),
    Invariant(
        "observed_container_isolation",
        "container_backend",
        "arc/crack_lab/test_arc_agi3_container_backend.py::"
        "test_create_command_has_exact_fail_closed_isolation",
        "The effective OCI configuration has the exact fail-closed isolation boundary.",
    ),
    Invariant(
        "probe_stderr_traceback_is_host_only",
        "app_server_transport",
        "arc/crack_lab/test_arc_agi3_codex_app_server_transport.py::"
        "test_probe_stderr_traceback_is_host_only_and_visibility_receipt_is_exact",
        "Raw Python/harness traceback bytes remain immutable host audit evidence while proposer-visible stderr is one fixed clean line with a hash-bound visibility classification.",
    ),
    Invariant(
        "teardown_proves_no_descendants",
        "container_backend",
        "arc/crack_lab/test_arc_agi3_container_backend.py::"
        "test_teardown_is_exactly_scoped_and_proves_no_descendants",
        "Teardown removes the exact identity and proves no descendants survive.",
    ),
    Invariant(
        "arena_rpc_is_only_solver_engine_boundary",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_arena_rpc.py::"
        "test_pinned_worker_uses_only_default_rpc_clone",
        "The pinned worker reaches the engine only through authenticated Arena RPC.",
    ),
    Invariant(
        "compatibility_arena_client_source_is_purified",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_compatibility_arena_closure.py::"
        "test_client_closure_is_exact_and_has_no_host_read_capability",
        "The canonical compatibility RPC client has no repository, engine, "
        "environment-file, parent-path, private-state, raw-Arena, or host-"
        "filesystem capability.",
    ),
    Invariant(
        "compatibility_arena_closure_custody_is_descriptor_safe",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_compatibility_arena_closure.py::"
        "test_root_swap_during_inventory_fails_final_descriptor_recheck",
        "The deterministic compatibility content manifest and instance "
        "custody receipt are read through one held directory descriptor, "
        "reopened at decision, and reject path, inode, mode, or inventory "
        "substitution.",
    ),
    Invariant(
        "compatibility_arena_closure_publication_is_crash_atomic",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_compatibility_arena_closure.py::"
        "test_crash_atomic_publication_fault_matrix_is_behavioral",
        "Closure files and their directory are fsynced under one deterministic "
        "non-authoritative sibling, exclusively atomically renamed into an "
        "absent destination, and parent-fsynced. Ordinary failures may clean "
        "only the current invocation's still-held staging inode and recorded "
        "child inodes; every pre-existing, racing, or post-crash staging inode "
        "is preserved fail-closed. Only the typed pre-existing-staging signal "
        "permits zero-cost quarantine and a fresh scheduler root.",
    ),
    Invariant(
        "compatibility_arena_closure_receipt_is_non_authoritative",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_compatibility_arena_closure.py::"
        "test_compatibility_closure_has_no_standalone_launch_authority",
        "The compatibility closure may be consumed by the existing backend, "
        "but its receipt alone explicitly grants no launch, scheduling, "
        "mutation, or promotion authority.",
    ),
    Invariant(
        "retained_prepare_crash_is_quarantined_and_fresh_generation_is_bounded",
        "runner",
        "arc/crack_lab/test_arc_agi3_contiguous_runner.py::"
        "test_retained_prepare_crash_uses_fresh_generation_and_bounds_repeats",
        "A process death that leaves typed compatibility staging closes the "
        "old PREPARED identity at zero authority, preserves a bounded "
        "descriptor observation, dispatches only a distinct fresh generation "
        "after circuit policy permits it, and gates repeated quarantines.",
    ),
    Invariant(
        "arena_named_volume_relay_is_byte_exact",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_arena_volume_relay.py::"
        "test_relay_client_to_host_stream_is_byte_exact",
        "The Colima transport's networkless named-volume relay preserves client bytes exactly; live Docker isolation remains a machine-observed launch prerequisite.",
    ),
    Invariant(
        "arena_attached_relay_is_full_duplex_and_bounded",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_arena_volume_transport.py::"
        "test_attached_relay_is_full_duplex_and_emits_exact_receipt",
        "The host-owned Docker-attach bridge preserves both Arena directions "
        "byte exactly, drains bounded stderr, joins every relay thread, and "
        "emits a hash-bound terminal receipt.",
    ),
    Invariant(
        "sibling_clone_leak_selects_fresh_process",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_arena_rpc.py::"
        "test_sibling_clone_leak_selects_authenticated_fresh_process",
        "A deterministic host-only sibling canary admits clone probing only "
        "after disjoint inspectable mutable graphs and matching isolated "
        "trajectories; leakage or inconclusive "
        "evidence selects a separately authenticated fresh process for every "
        "candidate branch, and solver text cannot select the mode.",
    ),
    Invariant(
        "arena_rpc_rejects_out_of_range_protocol_invalid",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_arena_rpc.py::"
        "test_out_of_range_client_call_is_host_logged_protocol_invalid",
        "An out-of-range ACTION6 reaches the trusted host, mutates no engine state, is durably rejected, and makes a clean-close result impossible.",
    ),
    Invariant(
        "public_action_grammar_is_cross_layer_exact",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_action_protocol.py::"
        "test_all_layers_accept_exactly_the_same_public_json_action_tokens",
        "Acquisition, replay, supervision, proposer publication, release, trusted RPC, and retained-evidence audit accept the same public JSON action grammar; ACTION6 always carries in-frame coordinates.",
    ),
    Invariant(
        "collector_requires_clean_arena_close_without_candidate",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_collector_requires_clean_arena_close_even_without_candidate",
        "Missing clean-close Arena authority is infrastructure with no WIP even when the attempt publishes no candidate.",
    ),
    Invariant(
        "worker_outcome_is_non_authoritative",
        "container_worker",
        "arc/crack_lab/test_arc_agi3_container_worker.py::"
        "test_worker_executes_solver_and_marks_outcome_non_authoritative",
        "Container worker output cannot claim promotion authority.",
    ),
    Invariant(
        "target_boundary_freezes_before_debrief",
        "proposer_worker",
        "arc/crack_lab/test_arc_agi3_proposer_worker.py::"
        "test_target_step_freezes_workspace_without_next_level_observation",
        "The first exact target boundary freezes source before debrief or next-level observation.",
    ),
    Invariant(
        "reward_boundary_is_absorbing_before_action7",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_arena_rpc.py::"
        "test_reward_boundary_is_absorbing_before_action7_undo",
        "The trusted Arena session seals the first exact K→K+1 reward: "
        "ACTION7, reset, and observation cannot mutate or continue it; only "
        "clean close remains.",
    ),
    Invariant(
        "action7_context_mismatch_reconstructs_and_invalidates",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_arena_rpc.py::"
        "test_action7_frame_mismatch_invalidates_and_reconstructs_branch",
        "Pre-reward ACTION7 is context-specific: frame, level, and terminal "
        "must exactly restore the immediate branch point; a mismatch is "
        "reconstructed from the authenticated seed and remains inadmissible.",
    ),
    Invariant(
        "action7_exact_context_restoration_is_admissible",
        "arena_rpc",
        "arc/crack_lab/test_arc_agi3_arena_rpc.py::"
        "test_action7_requires_exact_context_restoration",
        "A pre-reward ACTION7 branch may continue only when its public frame, "
        "level, and terminal state exactly match the captured branch point.",
    ),
    Invariant(
        "schema1_reached_before_debrief_binds_source",
        "boundary_certifier",
        "arc/crack_lab/test_arc_agi3_boundary_certifier.py::"
        "test_reached_before_debrief_outranks_and_binds_schema1_source",
        "A schema-1 reached-before-debrief boundary outranks later output and binds the exact contemporaneous winning source.",
    ),
    Invariant(
        "schema1_raw_transcript_failure_is_not_laundered",
        "boundary_certifier",
        "arc/crack_lab/test_arc_agi3_boundary_certifier.py::"
        "test_schema1_raw_transcript_failure_is_not_laundered",
        "A failed schema-1 raw transcript boundary cannot be replaced by a more favorable derived interpretation.",
    ),
    Invariant(
        "exact_path_reconstruction_is_last_resort",
        "boundary_certifier",
        "arc/crack_lab/test_arc_agi3_boundary_certifier.py::"
        "test_exact_path_reconstruction_is_last_resort",
        "Exact-path reconstruction is used only after stronger contemporaneous boundary evidence is unavailable.",
    ),
    Invariant(
        "app_server_has_zero_ambient_configuration",
        "app_server_transport",
        "arc/crack_lab/test_arc_agi3_codex_app_server_transport.py::"
        "test_config_projection_is_zero_ambient_and_fail_closed",
        "The authenticated controller observes no ambient project capability.",
    ),
    Invariant(
        "complete_controller_lifecycle_is_taint_checked",
        "taint",
        "arc/crack_lab/test_arc_agi3_contiguous_taint.py::"
        "test_complete_exact_lifecycle_and_tool_pairing_passes",
        "The exact app-server lifecycle and dynamic-tool pairs pass the strict taint parser.",
    ),
    Invariant(
        "filesystem_boundary_is_prewrite_and_scheduler_visible",
        "taint",
        "arc/crack_lab/test_arc_agi3_container_backend.py::"
        "test_prewrite_boundary_rejection_is_backend_taint_and_noncounting_scheduler_transition",
        "The production BridgeClient and container backend, exercised with "
        "authenticated controller/Docker doubles, carry a prewrite-denied "
        "filesystem-boundary workspace_write from its immutable transcript "
        "through runner validation into candidate/WIP-free taint, scheduler "
        "lineage revocation, and a zero retry delta. Live Docker execution "
        "remains a pilot/conformance-gate obligation.",
    ),
    Invariant(
        "general_taint_dependency_is_control_bound_and_syntax_aware",
        "taint",
        "arc/crack_lab/test_arc_agi3_contiguous_taint.py::"
        "test_general_taint_dependency_is_control_bound_and_syntax_aware",
        "The general scanner imported by contiguous taint is control-bound and distinguishes Python/AWK data from literal host-process commands.",
    ),
    Invariant(
        "exact_k_to_k_plus_one_promotion",
        "supervisor",
        "arc/crack_lab/test_arc_agi3_contiguous_supervisor.py::"
        "test_promotion_requires_path_and_source_replay_from_zero",
        "Private-probe and clone-derived candidate claims remain "
        "non-authoritative: promotion requires the exact K→K+1 path and "
        "winning source independently replayed from public zero, and rejects "
        "a divergent candidate path or candidate-authored host receipt.",
    ),
    Invariant(
        "exact_bundle_manifest_is_atomic_and_fail_closed",
        "supervisor",
        "arc/crack_lab/test_arc_agi3_contiguous_supervisor.py::"
        "test_exact_bundle_manifests_are_atomic_and_fail_closed",
        "Archived evidence and live candidate exports reject stale, changed, "
        "missing, or extra files; manifest publication is atomic.",
    ),
    Invariant(
        "promotion_rejects_post_boundary_suffix",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_promotion_rejects_actions_after_first_exact_boundary",
        "Promotion rejects any action suffix after the first exact K→K+1 boundary.",
    ),
    Invariant(
        "promotion_rejects_action7_after_reward",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_promotion_never_uses_action7_after_reward_as_validation",
        "ACTION7 after reward is never treated as replay validation or "
        "continued acquisition; promotion starts from the sealed exact "
        "boundary and an independent fresh replay.",
    ),
    Invariant(
        "retained_journal_is_full_lifecycle_prefix",
        "orchestrator",
        "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py::"
        "test_retained_journal_requires_full_genesis_to_result_prefix",
        "Retained evidence is the complete GENESIS-to-selected-result journal prefix.",
    ),
    Invariant(
        "transcript_taint_and_hashes_are_bound",
        "release_gate",
        "arc/crack_lab/test_arc_agi3_release_gate.py::"
        "test_reverification_fails_after_each_evidence_mutation_or_deletion",
        "Transcript, taint, replay, and hash evidence remain mutation-detecting.",
    ),
    Invariant(
        "full_inventory_release_receipt_admitted",
        "release_gate",
        "arc/crack_lab/test_arc_agi3_release_gate.py::"
        "test_issues_and_reverifies_content_addressed_183_receipt",
        "One content-addressed receipt admits all exact 183 boundaries.",
    ),
)

LAUNCH_REQUIREMENTS_CONTROL_PATH = (
    "arc/crack_lab/arc_agi3_contiguous_launch_requirements.json"
)
# This literal, reviewed allowlist is the second half of the canonical release
# suite.  The one-owner registry above proves each named launch invariant
# exactly once; every test in every file below must also collect and pass, so a
# stale component suite cannot hide behind a green owner aggregate.
COMPONENT_TEST_FILES = (
    "arc/crack_lab/test_arc_agi3_action_protocol.py",
    "arc/crack_lab/test_arc_agi3_arena_rpc.py",
    "arc/crack_lab/test_arc_agi3_arena_volume_relay.py",
    "arc/crack_lab/test_arc_agi3_arena_volume_transport.py",
    "arc/crack_lab/test_arc_agi3_boundary_certifier.py",
    "arc/crack_lab/test_arc_agi3_codex_app_server_transport.py",
    "arc/crack_lab/test_arc_agi3_compatibility_arena_closure.py",
    "arc/crack_lab/test_arc_agi3_container_backend.py",
    "arc/crack_lab/test_arc_agi3_container_worker.py",
    "arc/crack_lab/test_arc_agi3_containment_canary_operator.py",
    "arc/crack_lab/test_arc_agi3_contiguous_conformance.py",
    "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py",
    "arc/crack_lab/test_arc_agi3_contiguous_pilot.py",
    "arc/crack_lab/test_arc_agi3_contiguous_runner.py",
    "arc/crack_lab/test_arc_agi3_contiguous_scenario_driver.py",
    "arc/crack_lab/test_arc_agi3_contiguous_scheduler.py",
    "arc/crack_lab/test_arc_agi3_contiguous_supervisor.py",
    "arc/crack_lab/test_arc_agi3_contiguous_taint.py",
    "arc/crack_lab/test_arc_agi3_contiguous_watchdog.py",
    "arc/crack_lab/test_arc_agi3_controller_egress_guardian.py",
    "arc/crack_lab/test_arc_agi3_controller_guardian.py",
    "arc/crack_lab/test_arc_agi3_controller_supply_chain.py",
    "arc/crack_lab/test_arc_agi3_leaderboard_v3_gate.py",
    "arc/crack_lab/test_arc_agi3_proposer_worker.py",
    "arc/crack_lab/test_arc_agi3_python_runtime_manifest.py",
    "arc/crack_lab/test_arc_agi3_release_gate.py",
    "arc/crack_lab/test_arc_agi3_terminal_retention.py",
)
# These are the only environment-tree inputs admitted to the controller
# snapshot. They are public toolkit metadata, not game implementations or
# proposer context. Keeping the reviewed paths literal makes inventory drift
# (including a 26th metadata file) a control-contract failure instead of an
# implicit scheduler target change.
AUTHORITATIVE_INVENTORY_METADATA_FILES = (
    "environment_files/ar25/0c556536/metadata.json",
    "environment_files/bp35/0a0ad940/metadata.json",
    "environment_files/cd82/fb555c5d/metadata.json",
    "environment_files/cn04/2fe56bfb/metadata.json",
    "environment_files/dc22/fdcac232/metadata.json",
    "environment_files/ft09/0d8bbf25/metadata.json",
    "environment_files/g50t/5849a774/metadata.json",
    "environment_files/ka59/38d34dbb/metadata.json",
    "environment_files/lf52/271a04aa/metadata.json",
    "environment_files/lp85/305b61c3/metadata.json",
    "environment_files/ls20/9607627b/metadata.json",
    "environment_files/m0r0/492f87ba/metadata.json",
    "environment_files/r11l/495a7899/metadata.json",
    "environment_files/re86/8af5384d/metadata.json",
    "environment_files/s5i5/18d95033/metadata.json",
    "environment_files/sb26/7fbdac44/metadata.json",
    "environment_files/sc25/635fd71a/metadata.json",
    "environment_files/sk48/d8078629/metadata.json",
    "environment_files/sp80/589a99af/metadata.json",
    "environment_files/su15/1944f8ab/metadata.json",
    "environment_files/tn36/ef4dde99/metadata.json",
    "environment_files/tr87/cd924810/metadata.json",
    "environment_files/tu93/0768757b/metadata.json",
    "environment_files/vc33/5430563c/metadata.json",
    "environment_files/wa30/ee6fef47/metadata.json",
)
AUTHORITATIVE_INVENTORY_METADATA_KEYS = {
    "baseline_actions",
    "date_downloaded",
    "default_fps",
    "game_id",
    "local_dir",
    "tags",
    "title",
}
CONTROL_CONTRACT_FILES = (
    "arc/crack_lab/ARC_AGI3_CAMPAIGN_PLAN.md",
    LAUNCH_REQUIREMENTS_CONTROL_PATH,
    "arc/crack_lab/arc_agi3_contiguous_conformance.py",
    "arc/crack_lab/arc_agi3_contiguous_scenario_driver.py",
    "arc/crack_lab/arc_agi3_contiguous_supervisor.py",
    "arc/crack_lab/arc_agi3_exact_bundle.py",
    "arc/crack_lab/arc_agi3_contiguous_scheduler.py",
    "arc/crack_lab/arc_agi3_contiguous_runner.py",
    "arc/crack_lab/arc_agi3_contiguous_orchestrator.py",
    "arc/crack_lab/arc_agi3_contiguous_pilot.py",
    "arc/crack_lab/arc_agi3_contiguous_watchdog.py",
    "arc/crack_lab/arc_agi3_container_backend.py",
    "arc/crack_lab/arc_agi3_containment_canary_operator.py",
    "arc/crack_lab/arc_agi3_python_runtime_manifest.py",
    "arc/crack_lab/arc_agi3_codex_app_server_transport.py",
    "arc/audit_submission_taint.py",
    "arc/audit_action_protocol.py",
    "arc/crack_lab/arc_agi3_contiguous_taint.py",
    "arc/crack_lab/arc_agi3_proposer_boundary.py",
    "arc/crack_lab/arc_agi3_source_schema.py",
    "arc/crack_lab/arc_agi3_arena_rpc.py",
    "arc/crack_lab/arc_agi3_arena_rpc_client.py",
    "arc/crack_lab/arc_agi3_compatibility_arena_closure.py",
    "arc/crack_lab/arc_agi3_arena_volume_relay.py",
    "arc/crack_lab/arc_agi3_arena_volume_transport.py",
    "arc/crack_lab/arc_agi3_proposer_worker.py",
    "arc/crack_lab/arc_agi3_container_worker.py",
    "arc/crack_lab/arc_agi3_controller_guardian.py",
    "arc/crack_lab/arc_agi3_controller_supply_chain.py",
    "arc/crack_lab/arc_agi3_controller_egress_guardian.py",
    "arc/crack_lab/arc_agi3_release_gate.py",
    "arc/crack_lab/arc_agi3_boundary_certifier.py",
    "arc/arc_agi3_adapter.py",
    "arc/crack_lab/codex_campaign_status.py",
    "arc/crack_lab/claude_usage_guard.py",
    "arc/crack_lab/codex_usage_guard.py",
    "arc/crack_lab/gkm_arena.py",
    "arc/crack_lab/gkm_api_agent.py",
    "arc/crack_lab/gkm_legs.py",
    "arc/crack_lab/gkm_solve_agent.py",
    "arc/crack_lab/lab.py",
    "arc/crack_lab/llm_binder.py",
    "arc/crack_lab/priors.py",
    "arc/crack_lab/proposer.py",
    "arc/crack_lab/replay_scorecard.py",
    "arc/crack_lab/verify_frozen_release.py",
    "cone/cone_foraging.py",
    "cone/cone_foraging_bound.py",
    "arc/crack_lab/container/Containerfile.arc-agi3-contiguous",
    "arc/crack_lab/container/arc_agi3_solver_requirements.lock",
    "arc/crack_lab/container/Containerfile.arc-agi3-controller",
    "arc/crack_lab/container/Containerfile.arc-agi3-controller-egress",
    "arc/crack_lab/container/Containerfile.arc-agi3-arena-volume-relay",
    "arc/crack_lab/container/Containerfile.arc-agi3-workspace-probe",
    "arc/crack_lab/contiguous_blank_scaffold/legs.py",
    "arc/crack_lab/contiguous_blank_scaffold/players.py",
    "arc/crack_lab/contiguous_blank_scaffold/solve.py",
    "arc/crack_lab/test_arc_agi3_contiguous_conformance.py",
    "arc/crack_lab/test_arc_agi3_contiguous_scenario_driver.py",
    "arc/crack_lab/test_arc_agi3_contiguous_supervisor.py",
    "arc/crack_lab/test_arc_agi3_contiguous_scheduler.py",
    "arc/crack_lab/test_arc_agi3_contiguous_runner.py",
    "arc/crack_lab/test_arc_agi3_contiguous_orchestrator.py",
    "arc/crack_lab/test_arc_agi3_contiguous_pilot.py",
    "arc/crack_lab/test_arc_agi3_terminal_retention.py",
    "arc/crack_lab/test_arc_agi3_container_backend.py",
    "arc/crack_lab/test_arc_agi3_containment_canary_operator.py",
    "arc/crack_lab/test_arc_agi3_python_runtime_manifest.py",
    "arc/crack_lab/test_arc_agi3_codex_app_server_transport.py",
    "arc/crack_lab/test_arc_agi3_contiguous_taint.py",
    "arc/crack_lab/test_arc_agi3_contiguous_watchdog.py",
    "arc/crack_lab/test_arc_agi3_action_protocol.py",
    "arc/crack_lab/test_arc_agi3_compatibility_arena_closure.py",
    "arc/crack_lab/test_arc_agi3_arena_rpc.py",
    "arc/crack_lab/test_arc_agi3_arena_volume_relay.py",
    "arc/crack_lab/test_arc_agi3_arena_volume_transport.py",
    "arc/crack_lab/test_arc_agi3_proposer_worker.py",
    "arc/crack_lab/test_arc_agi3_container_worker.py",
    "arc/crack_lab/test_arc_agi3_controller_supply_chain.py",
    "arc/crack_lab/test_arc_agi3_controller_guardian.py",
    "arc/crack_lab/test_arc_agi3_controller_egress_guardian.py",
    "arc/crack_lab/test_arc_agi3_leaderboard_v3_gate.py",
    "arc/crack_lab/test_arc_agi3_release_gate.py",
    "arc/crack_lab/test_arc_agi3_boundary_certifier.py",
    "arc/crack_lab/test_verify_frozen_release.py",
    "arc/test_audit_action_protocol.py",
    *AUTHORITATIVE_INVENTORY_METADATA_FILES,
)

ENTRY_COMMAND = (
    "<absolute-digest-bound-python>",
    "-I",
    "-E",
    "-s",
    "-S",
    "-B",
    "-c",
    "<control-bound-hermetic-suite-bootstrap>",
    "<manifest-bound-site-packages>",
    "<immutable-control-root>/arc/crack_lab/"
    "arc_agi3_contiguous_conformance.py",
    "<absolute-python-runtime-manifest>",
    "<python-runtime-manifest-sha256>",
)
SUITE_EXECUTION_POLICY = {
    "interpreter_flags": ["-I", "-E", "-s", "-S", "-B"],
    "cwd": "immutable_control_root/.neutral",
    "home": "immutable_control_root/.neutral",
    "locale": "C",
    "pytest_plugin_autoload": False,
    "scratch":
        "private_system_temp_0700_empty_after_and_removed",
    "process_containment":
        "exact_pid_start_identity_plus_process_group_and_session",
    "workspace_root_inventory": "bounded_pre_post_exact",
}
_LOADED_SUITE_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()
SUITE_CONTROL_PATH = (
    "arc/crack_lab/arc_agi3_contiguous_conformance.py"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _regular_file_bytes(path: Path) -> bytes:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise ConformanceError(f"control file is unavailable: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ConformanceError(
                f"control file is aliased or nonregular: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _rooted_regular_file_bytes(root: Path, relative: str) -> bytes:
    """Read one root-relative file without following any path component."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ConformanceError(
            "descriptor-rooted control reads require O_NOFOLLOW/O_DIRECTORY"
        )
    logical = Path(relative)
    if (
        logical.is_absolute()
        or not logical.parts
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise ConformanceError(
            f"invalid rooted control path: {relative!r}"
        )
    root_path = Path(os.path.abspath(root))
    try:
        root_named = os.lstat(root_path)
    except OSError as exc:
        raise ConformanceError(
            f"control root is unavailable: {root_path}"
        ) from exc
    if not stat.S_ISDIR(root_named.st_mode):
        raise ConformanceError(
            f"control root is aliased or non-directory: {root_path}"
        )
    descriptors: list[int] = []
    bindings: list[tuple[int, str, tuple[int, ...]]] = []
    try:
        root_descriptor = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        descriptors.append(root_descriptor)
        if (
            _metadata_identity(os.fstat(root_descriptor))
            != _metadata_identity(root_named)
        ):
            raise ConformanceError(
                "control root changed while opened"
            )
        parent_descriptor = root_descriptor
        for component in logical.parts[:-1]:
            descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(descriptor)
                raise ConformanceError(
                    "control path ancestor is not a directory"
                )
            descriptors.append(descriptor)
            bindings.append((
                parent_descriptor,
                component,
                _metadata_identity(metadata),
            ))
            parent_descriptor = descriptor
        final_name = logical.parts[-1]
        file_descriptor = os.open(
            final_name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise ConformanceError(
                f"control file is aliased or nonregular: {relative}"
            )
        bindings.append((
            parent_descriptor,
            final_name,
            _metadata_identity(before),
        ))
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        if _metadata_identity(os.fstat(file_descriptor)) != (
            _metadata_identity(before)
        ):
            raise ConformanceError(
                f"control file changed while read: {relative}"
            )
        for parent, component, identity in bindings:
            try:
                named = os.stat(
                    component,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ConformanceError(
                    f"control path changed while read: {relative}"
                ) from exc
            if _metadata_identity(named) != identity:
                raise ConformanceError(
                    f"control path changed while read: {relative}"
                )
        if _metadata_identity(os.lstat(root_path)) != (
            _metadata_identity(root_named)
        ):
            raise ConformanceError(
                "control root changed while read"
            )
        return b"".join(chunks)
    except OSError as exc:
        raise ConformanceError(
            f"control path is unavailable or symlinked: {relative}"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _authoritative_inventory_metadata_paths(
    root: Path,
) -> tuple[str, ...]:
    """Enumerate only the fixed two-level metadata tree by directory FDs."""

    expected = tuple(AUTHORITATIVE_INVENTORY_METADATA_FILES)
    structure = {
        Path(relative).parts[1]: Path(relative).parts[2]
        for relative in expected
    }
    root_path = Path(os.path.abspath(root))
    try:
        root_named = os.lstat(root_path)
    except OSError as exc:
        raise ConformanceError(
            "authoritative inventory root is unavailable"
        ) from exc
    if not stat.S_ISDIR(root_named.st_mode):
        raise ConformanceError(
            "authoritative inventory root is symlinked or non-directory"
        )
    root_descriptor = -1
    environment_descriptor = -1
    try:
        root_descriptor = os.open(
            root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        if (
            _metadata_identity(os.fstat(root_descriptor))
            != _metadata_identity(root_named)
        ):
            raise ConformanceError(
                "authoritative inventory root changed while opened"
            )
        environment_descriptor = os.open(
            "environment_files",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_descriptor,
        )
        environment_identity = _metadata_identity(
            os.fstat(environment_descriptor)
        )
        game_names = tuple(sorted(os.listdir(environment_descriptor)))
        if game_names != tuple(sorted(structure)):
            raise ConformanceError(
                "authoritative inventory metadata is missing or unexpected"
            )
        observed: list[str] = []
        for game in game_names:
            game_descriptor = os.open(
                game,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=environment_descriptor,
            )
            try:
                game_identity = _metadata_identity(
                    os.fstat(game_descriptor)
                )
                versions = tuple(sorted(os.listdir(game_descriptor)))
                if versions != (structure[game],):
                    raise ConformanceError(
                        "authoritative inventory metadata is missing or "
                        "unexpected"
                    )
                version = versions[0]
                version_descriptor = os.open(
                    version,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW,
                    dir_fd=game_descriptor,
                )
                try:
                    version_identity = _metadata_identity(
                        os.fstat(version_descriptor)
                    )
                    entries = set(os.listdir(version_descriptor))
                    if "metadata.json" not in entries:
                        raise ConformanceError(
                            "authoritative inventory metadata is missing or "
                            "unexpected"
                        )
                    metadata = os.stat(
                        "metadata.json",
                        dir_fd=version_descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_nlink != 1
                    ):
                        raise ConformanceError(
                            "authoritative inventory metadata is aliased or "
                            "nonregular"
                        )
                    observed.append(
                        f"environment_files/{game}/{version}/metadata.json"
                    )
                    if _metadata_identity(os.stat(
                        version,
                        dir_fd=game_descriptor,
                        follow_symlinks=False,
                    )) != version_identity:
                        raise ConformanceError(
                            "authoritative inventory metadata changed while "
                            "enumerated"
                        )
                finally:
                    os.close(version_descriptor)
                if _metadata_identity(os.stat(
                    game,
                    dir_fd=environment_descriptor,
                    follow_symlinks=False,
                )) != game_identity:
                    raise ConformanceError(
                        "authoritative inventory metadata changed while "
                        "enumerated"
                    )
            finally:
                os.close(game_descriptor)
        if _metadata_identity(os.stat(
            "environment_files",
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )) != environment_identity:
            raise ConformanceError(
                "authoritative inventory metadata changed while enumerated"
            )
        if _metadata_identity(os.lstat(root_path)) != (
            _metadata_identity(root_named)
        ):
            raise ConformanceError(
                "authoritative inventory root changed while enumerated"
            )
        return tuple(observed)
    except OSError as exc:
        raise ConformanceError(
            "authoritative inventory metadata has a symlinked ancestor"
        ) from exc
    finally:
        if environment_descriptor >= 0:
            os.close(environment_descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)


def authoritative_inventory_metadata_snapshot(
    repository: Path | None = None,
) -> dict[str, Any]:
    """Validate the exact public metadata inputs behind the 25/183 map."""

    root = (
        _repository_root() if repository is None else Path(repository)
    )
    expected = tuple(AUTHORITATIVE_INVENTORY_METADATA_FILES)
    if _authoritative_inventory_metadata_paths(root) != tuple(
        sorted(expected)
    ):
        raise ConformanceError(
            "authoritative inventory metadata is missing or unexpected"
        )
    records: list[dict[str, Any]] = []
    games: set[str] = set()
    levels = 0
    for relative in expected:
        raw = _rooted_regular_file_bytes(root, relative)
        if len(raw) > 64 * 1024:
            raise ConformanceError(
                "authoritative inventory metadata is oversized"
            )
        try:
            value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ConformanceError(
                "authoritative inventory metadata is invalid JSON"
            ) from exc
        parts = Path(relative).parts
        game = parts[1]
        version = parts[2]
        actions = (
            value.get("baseline_actions")
            if isinstance(value, dict)
            else None
        )
        if (
            not isinstance(value, dict)
            or set(value) != AUTHORITATIVE_INVENTORY_METADATA_KEYS
            or value.get("game_id") != f"{game}-{version}"
            or value.get("title") != game.upper()
            or not isinstance(value.get("default_fps"), int)
            or isinstance(value.get("default_fps"), bool)
            or value["default_fps"] <= 0
            or not isinstance(value.get("tags"), list)
            or any(
                not isinstance(tag, str) or not tag
                for tag in value.get("tags", [])
            )
            or not isinstance(actions, list)
            or not actions
            or any(
                not isinstance(action, int)
                or isinstance(action, bool)
                or action <= 0
                for action in actions
            )
            or not isinstance(value.get("local_dir"), str)
            or not value["local_dir"]
            or not isinstance(value.get("date_downloaded"), str)
            or not value["date_downloaded"]
            or game in games
        ):
            raise ConformanceError(
                "authoritative inventory metadata schema mismatch"
            )
        games.add(game)
        levels += len(actions)
        records.append({
            "path": relative,
            "sha256": _sha256(raw),
            "game": game,
            "levels": len(actions),
        })
    if len(games) != EXPECTED_GAMES or levels != EXPECTED_LEVELS:
        raise ConformanceError(
            "authoritative inventory metadata must be exactly 25/183"
        )
    if _authoritative_inventory_metadata_paths(root) != tuple(
        sorted(expected)
    ):
        raise ConformanceError(
            "authoritative inventory metadata changed while read"
        )
    return {
        "files": records,
        "games": len(games),
        "levels": levels,
        "sha256": _sha256(_canonical_json(records)),
    }


def _unique_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ConformanceError(
                f"launch requirements duplicate JSON key: {key}"
            )
        value[key] = item
    return value


def launch_requirements_snapshot(
    repository: Path | None = None,
    *,
    body: object | None = None,
) -> dict[str, Any]:
    """Validate the independent, frozen S01--S12 launch ledger.

    The ledger is deliberately not generated from :data:`INVARIANTS`.
    Supplying ``body`` exists only for adversarial structural tests; normal
    registration and result verification reopen the control-hashed file.
    """

    raw: bytes | None = None
    if body is None:
        root = (
            _repository_root()
            if repository is None
            else Path(repository)
        )
        raw = _rooted_regular_file_bytes(
            root, LAUNCH_REQUIREMENTS_CONTROL_PATH
        )
        try:
            body = json.loads(
                raw,
                object_pairs_hook=_unique_json_object,
            )
        except (
            UnicodeError,
            json.JSONDecodeError,
            ConformanceError,
        ) as exc:
            raise ConformanceError(
                "launch requirements are not unique-key JSON"
            ) from exc
    if not isinstance(body, dict):
        raise ConformanceError(
            "launch requirements must be a JSON object"
        )
    required_keys = {
        "schema",
        "kind",
        "scenario_owners",
        "supervision_cycle_stages",
        "requirements",
    }
    scenarios = body.get("scenario_owners")
    stages = body.get("supervision_cycle_stages")
    requirements = body.get("requirements")
    expected_scenarios = tuple(
        f"S{index:02d}" for index in range(1, 13)
    )
    if (
        set(body) != required_keys
        or body.get("schema") != 1
        or isinstance(body.get("schema"), bool)
        or body.get("kind")
        != "arc_agi3_contiguous_launch_requirements"
        or not isinstance(scenarios, list)
        or not isinstance(stages, list)
        or not isinstance(requirements, list)
        or not stages
        or any(
            not isinstance(stage, str) or not stage
            for stage in stages
        )
        or len(stages) != len(set(stages))
    ):
        raise ConformanceError(
            "launch requirements schema is malformed"
        )
    scenario_ids: list[str] = []
    scenario_owner_ids: list[str] = []
    scenario_versions = {
        scenario_id: (2 if scenario_id == "S09" else 1)
        for scenario_id in expected_scenarios
    }
    for expected_id, scenario in zip(
        expected_scenarios, scenarios, strict=False
    ):
        expected_version = scenario_versions[expected_id]
        if (
            not isinstance(scenario, dict)
            or set(scenario) != {"scenario_id", "owner", "version"}
            or scenario.get("scenario_id") != expected_id
            or not isinstance(scenario.get("owner"), str)
            or scenario["owner"]
            != (
                f"arc_agi3_contiguous_{expected_id.lower()}_v"
                f"{expected_version}"
            )
            or scenario.get("version") != expected_version
            or isinstance(scenario.get("version"), bool)
        ):
            raise ConformanceError(
                "launch requirements scenario owner is malformed"
            )
        scenario_ids.append(scenario["scenario_id"])
        scenario_owner_ids.append(scenario["owner"])
    if (
        tuple(scenario_ids) != expected_scenarios
        or len(scenarios) != len(expected_scenarios)
        or len(set(scenario_owner_ids)) != len(scenario_owner_ids)
    ):
        raise ConformanceError(
            "launch requirements do not own exact S01-S12"
        )
    requirement_ids: list[str] = []
    owner_nodeids: list[str] = []
    represented_scenarios: set[str] = set()
    for requirement in requirements:
        if (
            not isinstance(requirement, dict)
            or set(requirement)
            != {
                "invariant_id",
                "component",
                "owner_nodeid",
                "scenario_id",
            }
            or not all(
                isinstance(requirement.get(key), str)
                and bool(requirement[key])
                for key in (
                    "invariant_id",
                    "component",
                    "owner_nodeid",
                    "scenario_id",
                )
            )
            or requirement["scenario_id"] not in expected_scenarios
        ):
            raise ConformanceError(
                "launch requirement entry is malformed"
            )
        requirement_ids.append(requirement["invariant_id"])
        owner_nodeids.append(requirement["owner_nodeid"])
        represented_scenarios.add(requirement["scenario_id"])
    if (
        not requirements
        or len(requirement_ids) != len(set(requirement_ids))
        or len(owner_nodeids) != len(set(owner_nodeids))
        or represented_scenarios != set(expected_scenarios)
    ):
        raise ConformanceError(
            "launch requirements are missing, duplicated, or owner-ambiguous"
        )
    normalized = json.loads(_canonical_json(body))
    return {
        "body": normalized,
        "sha256": _sha256(
            raw if raw is not None else _canonical_json(normalized)
        ),
    }


def validate_registry(
    invariants: Iterable[Invariant] = INVARIANTS,
    *,
    repository: Path | None = None,
    launch_requirements: object | None = None,
) -> tuple[Invariant, ...]:
    values = tuple(invariants)
    ids = [value.invariant_id for value in values]
    nodeids = [value.nodeid for value in values]
    launch = launch_requirements_snapshot(
        repository=repository,
        body=launch_requirements,
    )
    required = launch["body"]["requirements"]
    required_by_id = {
        item["invariant_id"]: item for item in required
    }
    if (
        not values
        or any(
            not isinstance(value, Invariant)
            or not value.invariant_id
            or not value.component
            or not value.nodeid
            or not value.claim
            for value in values
        )
        or len(ids) != len(set(ids))
        or len(nodeids) != len(set(nodeids))
        or set(ids) != set(required_by_id)
        or len(values) != len(required)
        or any(
            required_by_id[value.invariant_id]["component"]
            != value.component
            or required_by_id[value.invariant_id]["owner_nodeid"]
            != value.nodeid
            for value in values
            if value.invariant_id in required_by_id
        )
    ):
        raise ConformanceError(
            "canonical invariant registry is missing, duplicated, or malformed"
        )
    return values


def registry_sha256(
    invariants: Iterable[Invariant] = INVARIANTS,
    *,
    repository: Path | None = None,
    launch_requirements: object | None = None,
) -> str:
    launch = launch_requirements_snapshot(
        repository=repository,
        body=launch_requirements,
    )
    values = validate_registry(
        invariants,
        repository=repository,
        launch_requirements=launch["body"],
    )
    return _sha256(_canonical_json({
        "invariants": [asdict(value) for value in values],
        "launch_requirements_sha256": launch["sha256"],
    }))


def control_contract_snapshot(
    control_files: Mapping[str, Path] | None = None,
    *,
    repository: Path | None = None,
    expected_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return the one canonical ordered control-file snapshot.

    ``control_files=None`` selects the launch contract and therefore requires
    exactly :data:`CONTROL_CONTRACT_FILES` in its reviewed order.  Explicit
    mappings support release-gate fixture contracts; they are deterministically
    ordered unless an exact independent order is supplied.
    """
    if control_files is None:
        root = _repository_root() if repository is None else Path(repository)
        authoritative_inventory_metadata_snapshot(root)
        order = tuple(CONTROL_CONTRACT_FILES)
        selected = {relative: root / relative for relative in order}
    else:
        if repository is not None:
            raise ConformanceError(
                "repository cannot accompany explicit control files"
            )
        if not isinstance(control_files, Mapping) or not control_files:
            raise ConformanceError(
                "control contract must contain regular files"
            )
        selected = {
            str(logical): Path(path)
            for logical, path in control_files.items()
        }
        order = (
            tuple(str(item) for item in expected_order)
            if expected_order is not None
            else tuple(sorted(selected))
        )
        if set(order) != set(selected) or len(order) != len(selected):
            raise ConformanceError(
                "control-contract order is missing, duplicated, or unexpected"
            )
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for logical_name in order:
        if (
            not logical_name
            or logical_name.startswith("/")
            or ".." in Path(logical_name).parts
        ):
            raise ConformanceError(
                f"invalid control-contract logical path: {logical_name!r}"
            )
        raw = (
            _rooted_regular_file_bytes(root, logical_name)
            if control_files is None
            else _regular_file_bytes(selected[logical_name])
        )
        digest = _sha256(raw)
        records.append({"path": logical_name, "sha256": digest})
        hashes[logical_name] = digest
    return {
        "sha256": _sha256(_canonical_json(records)),
        "files_sha256": hashes,
    }


def control_contract_files_sha256(
    repository: Path | None = None,
) -> dict[str, str]:
    return control_contract_snapshot(
        repository=repository
    )["files_sha256"]


def control_contract_sha256(
    repository: Path | None = None,
) -> str:
    return control_contract_snapshot(repository=repository)["sha256"]


def validate_immutable_control_snapshot(
    root: Path,
    *,
    expected_sha256: str | None = None,
    _require_content_addressed_name: bool = True,
) -> dict[str, Any]:
    """Reopen one sealed, exact, content-addressed control snapshot."""

    selected = Path(root)
    if (
        not selected.is_absolute()
        or selected.is_symlink()
        or not selected.is_dir()
    ):
        raise ConformanceError(
            "immutable control snapshot root is unsafe"
        )
    snapshot = control_contract_snapshot(repository=selected)
    if (
        (
            _require_content_addressed_name
            and selected.name != snapshot["sha256"]
        )
        or (
            expected_sha256 is not None
            and snapshot["sha256"] != expected_sha256
        )
    ):
        raise ConformanceError(
            "immutable control snapshot is not content-addressed"
        )
    allowed_files = set(CONTROL_CONTRACT_FILES)
    allowed_directories = {".neutral"}
    for relative in CONTROL_CONTRACT_FILES:
        parent = Path(relative).parent
        while str(parent) not in {"", "."}:
            allowed_directories.add(parent.as_posix())
            parent = parent.parent
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in selected.rglob("*"):
        relative = path.relative_to(selected).as_posix()
        metadata = os.lstat(path)
        if stat.S_ISREG(metadata.st_mode):
            observed_files.add(relative)
            if (
                metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) & 0o222
            ):
                raise ConformanceError(
                    "immutable control snapshot has a mutable file"
                )
        elif stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(relative)
            if (
                metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) & 0o222
            ):
                raise ConformanceError(
                    "immutable control snapshot has a mutable directory"
                )
        else:
            raise ConformanceError(
                "immutable control snapshot has an aliased/nonregular entry"
            )
    root_metadata = os.stat(selected, follow_symlinks=False)
    if (
        root_metadata.st_uid != os.getuid()
        or stat.S_IMODE(root_metadata.st_mode) & 0o222
        or observed_files != allowed_files
        or observed_directories != allowed_directories
    ):
        raise ConformanceError(
            "immutable control snapshot tree is incomplete or unexpected"
        )
    return snapshot


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_owned_snapshot_staging(path: Path, *, prefix: str) -> None:
    """Remove only one same-parent, owner-private staging directory."""

    metadata = os.lstat(path)
    if (
        path.name.startswith(prefix) is False
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ConformanceError(
            "control snapshot staging entry is unsafe to recover"
        )
    for entry in sorted(
        path.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        entry_metadata = os.lstat(entry)
        if stat.S_ISDIR(entry_metadata.st_mode):
            if entry_metadata.st_uid != os.getuid():
                raise ConformanceError(
                    "control snapshot staging directory changed owner"
                )
            os.chmod(entry, 0o700, follow_symlinks=False)
        elif (
            not stat.S_ISREG(entry_metadata.st_mode)
            or entry_metadata.st_uid != os.getuid()
            or entry_metadata.st_nlink != 1
        ):
            raise ConformanceError(
                "control snapshot staging contains an unsafe entry"
            )
    os.chmod(path, 0o700, follow_symlinks=False)
    shutil.rmtree(path)


def materialize_immutable_control_snapshot(
    source_repository: Path,
    destination: Path,
) -> dict[str, Any]:
    """Descriptor-copy, fsync, seal, and reverify the exact live manifest."""

    source = Path(source_repository)
    target = Path(destination)
    start = control_contract_snapshot(repository=source)
    if (
        not target.is_absolute()
        or target.name != start["sha256"]
        or target.parent.is_symlink()
        or not target.parent.is_dir()
    ):
        raise ConformanceError(
            "control snapshot destination must be an explicit digest path"
        )
    parent_metadata = os.stat(target.parent, follow_symlinks=False)
    if (
        parent_metadata.st_uid != os.getuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise ConformanceError(
            "control snapshot parent is not host-owned and private"
        )
    staging_prefix = f".{start['sha256']}.staging."
    lock_path = target.parent / f".{start['sha256']}.materialize.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    lock_metadata = os.fstat(lock_descriptor)
    try:
        if (
            not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_uid != os.getuid()
            or lock_metadata.st_nlink != 1
            or stat.S_IMODE(lock_metadata.st_mode) != 0o600
        ):
            raise ConformanceError(
                "control snapshot materialization lock is unsafe"
            )
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if target.exists() or target.is_symlink():
            validated = validate_immutable_control_snapshot(
                target, expected_sha256=start["sha256"]
            )
            if control_contract_snapshot(repository=source) != start:
                raise ConformanceError(
                    "live control tree changed while reopening snapshot"
                )
            return validated
        for stale in target.parent.iterdir():
            if stale.name.startswith(staging_prefix):
                _remove_owned_snapshot_staging(
                    stale, prefix=staging_prefix
                )
        staging = Path(
            tempfile.mkdtemp(
                prefix=staging_prefix,
                dir=target.parent,
            )
        )
        try:
            for relative in CONTROL_CONTRACT_FILES:
                raw = _rooted_regular_file_bytes(source, relative)
                if _sha256(raw) != start["files_sha256"][relative]:
                    raise ConformanceError(
                        "live control file changed while snapshotting"
                    )
                output = staging / relative
                output.parent.mkdir(
                    parents=True, exist_ok=True, mode=0o700
                )
                descriptor = os.open(
                    output,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o400,
                )
                try:
                    view = memoryview(raw)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise ConformanceError(
                                "short immutable control snapshot write"
                            )
                        view = view[written:]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            neutral = staging / ".neutral"
            neutral.mkdir(mode=0o500)
            copied = control_contract_snapshot(repository=staging)
            if copied != start:
                raise ConformanceError(
                    "copied control snapshot differs from live start bytes"
                )
            directories = sorted(
                (
                    path
                    for path in staging.rglob("*")
                    if path.is_dir()
                ),
                key=lambda item: len(item.parts),
                reverse=True,
            )
            for directory in directories:
                _fsync_directory(directory)
                os.chmod(directory, 0o500, follow_symlinks=False)
                _fsync_directory(directory)
            _fsync_directory(staging)
            os.chmod(staging, 0o500, follow_symlinks=False)
            _fsync_directory(staging)
            validate_immutable_control_snapshot(
                staging,
                expected_sha256=start["sha256"],
                _require_content_addressed_name=False,
            )
            try:
                os.rename(staging, target)
                staging = None
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                validate_immutable_control_snapshot(
                    target, expected_sha256=start["sha256"]
                )
                _remove_owned_snapshot_staging(
                    staging, prefix=staging_prefix
                )
                staging = None
            _fsync_directory(target.parent)
            validated = validate_immutable_control_snapshot(
                target, expected_sha256=start["sha256"]
            )
            if control_contract_snapshot(repository=source) != start:
                raise ConformanceError(
                    "live control tree changed while materializing snapshot"
                )
            return validated
        except Exception:
            if staging is not None and staging.exists():
                _remove_owned_snapshot_staging(
                    staging, prefix=staging_prefix
                )
            raise
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def component_test_files_snapshot(
    repository: Path | None = None,
    *,
    control_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reopen the exact component-test allowlist from one control root."""

    root = (
        _repository_root() if repository is None else Path(repository)
    )
    expected = tuple(COMPONENT_TEST_FILES)
    if (
        not expected
        or len(expected) > MAX_COMPONENT_TEST_FILES
        or expected != tuple(sorted(expected))
        or len(expected) != len(set(expected))
        or any(
            not path.startswith("arc/crack_lab/test_arc_agi3_")
            or not path.endswith(".py")
            or path not in CONTROL_CONTRACT_FILES
            for path in expected
        )
    ):
        raise ConformanceError(
            "component test-file allowlist is malformed or unsealed"
        )
    test_root = root / "arc" / "crack_lab"
    try:
        observed = tuple(sorted(
            path.relative_to(root).as_posix()
            for path in test_root.glob("test_arc_agi3_*.py")
        ))
    except (OSError, ValueError) as exc:
        raise ConformanceError(
            "component test-file inventory cannot be enumerated"
        ) from exc
    if observed != expected:
        raise ConformanceError(
            "component test-file inventory has a missing or unknown file"
        )
    selected_control = (
        control_contract_snapshot(repository=root)
        if control_snapshot is None
        else dict(control_snapshot)
    )
    try:
        control_hashes = selected_control["files_sha256"]
    except (KeyError, TypeError) as exc:
        raise ConformanceError(
            "control snapshot omits component test hashes"
        ) from exc
    records: list[dict[str, Any]] = []
    for relative in expected:
        raw = _rooted_regular_file_bytes(root, relative)
        digest = _sha256(raw)
        if control_hashes.get(relative) != digest:
            raise ConformanceError(
                "component test bytes differ from the control snapshot"
            )
        records.append({"path": relative, "sha256": digest})
    return {
        "files": records,
        "sha256": _sha256(_canonical_json(records)),
    }


def workspace_root_inventory(
    repository: Path,
) -> dict[str, Any]:
    """Return one bounded, non-recursive root-entry inventory."""

    selected = Path(repository)
    if selected.is_symlink():
        raise ConformanceError(
            "workspace inventory root is not a canonical directory"
        )
    root = selected.resolve()
    if not root.is_dir():
        raise ConformanceError(
            "workspace inventory root is not a canonical directory"
        )
    records: list[dict[str, Any]] = []
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        raise ConformanceError(
            "workspace root cannot be inventoried"
        ) from exc
    if len(entries) > MAX_WORKSPACE_ROOT_ENTRIES:
        raise ConformanceError(
            "workspace root entry inventory exceeds its bound"
        )
    for entry in entries:
        name = entry.name
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or len(name.encode("utf-8"))
            > MAX_WORKSPACE_ENTRY_NAME_BYTES
        ):
            raise ConformanceError(
                "workspace root contains an invalid entry name"
            )
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise ConformanceError(
                "workspace root entry changed during inventory"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
        else:
            kind = "other"
        records.append({
            "name": name,
            "kind": kind,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
            "nlink": metadata.st_nlink,
            "uid": metadata.st_uid,
        })
    records.sort(key=lambda record: record["name"])
    forbidden = [
        record["name"]
        for record in records
        if (
            record["kind"] == "directory"
            and (
                record["name"].startswith((
                    ".a3cb_",
                    ".a3vr_",
                    ".a3sock_",
                    ".a3s_",
                ))
                or re.fullmatch(
                    r"a[0-9a-f]{2}", record["name"]
                )
                is not None
            )
        )
    ]
    body = {
        "root": str(root),
        "entries": records,
        "forbidden_entries": forbidden,
    }
    return {
        **body,
        "sha256": _sha256(_canonical_json(body)),
    }


def loaded_control_modules_snapshot(
    repository: Path | None = None,
    *,
    control_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind loaded local modules to exact files in the selected control tree."""

    root = (
        _repository_root() if repository is None else Path(repository)
    ).resolve()
    selected_control = (
        control_contract_snapshot(repository=root)
        if control_snapshot is None
        else dict(control_snapshot)
    )
    try:
        control_hashes = selected_control["files_sha256"]
    except (KeyError, TypeError) as exc:
        raise ConformanceError(
            "control snapshot omits loaded-module hashes"
        ) from exc
    python_controls = {
        relative: digest
        for relative, digest in control_hashes.items()
        if relative.endswith(".py")
    }
    expected_by_name: dict[str, str] = {}
    expected_by_module_leaf: dict[str, str] = {}
    for relative in python_controls:
        control_path = Path(relative)
        basename = control_path.name
        if basename in expected_by_name:
            raise ConformanceError(
                "control Python basenames are ambiguous"
            )
        expected_by_name[basename] = relative
        module_leaf = control_path.stem
        if module_leaf in expected_by_module_leaf:
            raise ConformanceError(
                "control Python module names are ambiguous"
            )
        expected_by_module_leaf[module_leaf] = relative
    records: list[dict[str, str]] = []
    conflicts: set[str] = set()
    unsealed_local: set[str] = set()
    represented: set[str] = set()
    for module_name, module in sorted(sys.modules.items()):
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str) or not raw_origin:
            continue
        raw_path = Path(raw_origin)
        expected_from_origin = expected_by_name.get(raw_path.name)
        expected_from_module = expected_by_module_leaf.get(
            module_name.rsplit(".", 1)[-1]
        )
        if (
            expected_from_origin is not None
            and expected_from_module is not None
            and expected_from_origin != expected_from_module
        ):
            conflicts.add(module_name)
            continue
        expected_relative = (
            expected_from_module
            if expected_from_module is not None
            else expected_from_origin
        )
        if raw_origin.startswith("<") and raw_origin.endswith(">"):
            if expected_relative is not None:
                conflicts.add(module_name)
            continue
        absolute_origin = Path(os.path.abspath(raw_path))
        try:
            lexical_relative = absolute_origin.relative_to(
                root
            ).as_posix()
        except ValueError:
            lexical_relative = None
        try:
            origin = absolute_origin.resolve(strict=True)
        except OSError:
            if (
                expected_relative is not None
                or lexical_relative is not None
            ):
                conflicts.add(module_name)
            continue
        try:
            relative = origin.relative_to(root).as_posix()
        except ValueError:
            relative = None
        if (
            expected_relative is None
            and lexical_relative is None
            and relative is None
        ):
            continue
        if absolute_origin != origin:
            conflicts.add(module_name)
            continue
        if expected_relative is not None:
            expected_origin = root / expected_relative
            if (
                absolute_origin != expected_origin
                or origin != expected_origin
            ):
                conflicts.add(module_name)
                continue
        if relative is None:
            conflicts.add(module_name)
            continue
        expected_digest = python_controls.get(relative)
        if expected_digest is None:
            unsealed_local.add(relative)
            continue
        digest = _sha256(_rooted_regular_file_bytes(root, relative))
        if digest != expected_digest:
            conflicts.add(module_name)
            continue
        records.append({
            "module": module_name,
            "path": relative,
            "sha256": digest,
        })
        represented.add(relative)
    if len(records) > MAX_LOADED_CONTROL_MODULES:
        raise ConformanceError(
            "loaded control module inventory exceeds its bound"
        )
    required = {SUITE_CONTROL_PATH, *COMPONENT_TEST_FILES}
    missing_required = sorted(required - represented)
    normalized = sorted(
        records,
        key=lambda record: (record["module"], record["path"]),
    )
    body = {
        "records": normalized,
        "required_paths": sorted(required),
    }
    return {
        "complete": not (
            missing_required or conflicts or unsealed_local
        ),
        "records": normalized,
        "sha256": _sha256(_canonical_json(body)),
        "summary": {
            "required": len(required),
            "represented": len(required) - len(missing_required),
            "records": len(normalized),
            "missing_required": len(missing_required),
            "conflicting_origins": len(conflicts),
            "unsealed_local_modules": len(unsealed_local),
        },
    }


def _bounded_component_nodeids(
    value: object, *, label: str
) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > MAX_COMPONENT_TEST_CASES
        or any(
            not isinstance(nodeid, str)
            or not nodeid
            or len(nodeid.encode("utf-8")) > MAX_COMPONENT_NODEID_BYTES
            for nodeid in value
        )
    ):
        raise ConformanceError(
            f"{label} is malformed or exceeds the receipt bound"
        )
    normalized = list(value)
    if len(_canonical_json(normalized)) > MAX_COMPONENT_INVENTORY_BYTES:
        raise ConformanceError(
            f"{label} exceeds the deterministic receipt byte bound"
        )
    return normalized


def _component_collection_facts(
    nodeids: list[str],
) -> dict[str, Any]:
    allowed = set(COMPONENT_TEST_FILES)
    counts = Counter(nodeids)
    duplicate_nodeids = sorted(
        nodeid for nodeid, count in counts.items() if count != 1
    )
    unknown_nodeids = sorted(
        nodeid
        for nodeid in set(nodeids)
        if nodeid.partition("::")[0] not in allowed
    )
    represented = {
        nodeid.partition("::")[0]
        for nodeid in nodeids
        if nodeid.partition("::")[0] in allowed
    }
    missing_files = sorted(allowed - represented)
    return {
        "duplicate_nodeids": duplicate_nodeids,
        "unknown_nodeids": unknown_nodeids,
        "missing_files": missing_files,
    }


class _PytestRecorder:
    def __init__(self) -> None:
        self.collected: list[str] = []
        self.outcomes: dict[str, str] = {}

    def pytest_collection_finish(self, session: Any) -> None:
        self.collected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report: Any) -> None:
        nodeid = report.nodeid
        prior = self.outcomes.get(nodeid)
        if getattr(report, "wasxfail", None) is not None:
            self.outcomes[nodeid] = (
                "XFAIL" if report.skipped else "XPASS"
            )
        elif report.skipped:
            self.outcomes[nodeid] = "SKIP"
        elif report.failed:
            self.outcomes[nodeid] = "FAIL"
        elif report.when == "call" and report.passed and prior is None:
            self.outcomes[nodeid] = "PASS"


def _run_registered_pytest(
    invariants: tuple[Invariant, ...],
) -> tuple[int, list[str], dict[str, str], str]:
    try:
        import pytest
    except ImportError as exc:
        raise ConformanceError("pytest is required by conformance") from exc
    recorder = _PytestRecorder()
    output = StringIO()
    arguments = [
        "-q",
        "--disable-warnings",
        "-p",
        "no:cacheprovider",
        *[invariant.nodeid for invariant in invariants],
    ]
    with redirect_stdout(output), redirect_stderr(output):
        exit_code = int(pytest.main(arguments, plugins=[recorder]))
    return (
        exit_code,
        recorder.collected,
        dict(recorder.outcomes),
        output.getvalue(),
    )


def _collect_registered_pytest(
    invariants: tuple[Invariant, ...],
) -> tuple[int, list[str], str]:
    """Resolve the exact registry before any release test is executed."""

    try:
        import pytest
    except ImportError as exc:
        raise ConformanceError("pytest is required by conformance") from exc
    recorder = _PytestRecorder()
    output = StringIO()
    arguments = [
        "--collect-only",
        "-q",
        "--disable-warnings",
        "-p",
        "no:cacheprovider",
        *[invariant.nodeid for invariant in invariants],
    ]
    with redirect_stdout(output), redirect_stderr(output):
        exit_code = int(pytest.main(arguments, plugins=[recorder]))
    return exit_code, recorder.collected, output.getvalue()


def _component_pytest_arguments(
    *,
    collect_only: bool,
    basetemp: Path | None = None,
) -> list[str]:
    arguments = [
        "-q",
        "--disable-warnings",
        "--strict-config",
        "--strict-markers",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
    ]
    if basetemp is not None:
        arguments.extend(("--basetemp", str(basetemp)))
    if collect_only:
        arguments.insert(0, "--collect-only")
    arguments.extend(COMPONENT_TEST_FILES)
    return arguments


def _new_component_pytest_basetemp(
) -> tuple[Path, Path, tuple[int, ...]]:
    raw = os.environ.get("TMPDIR")
    if not isinstance(raw, str):
        raise ConformanceError(
            "component pytest requires explicit private TMPDIR"
        )
    parent = Path(raw)
    repository = _repository_root().resolve()
    if not parent.is_absolute():
        raise ConformanceError(
            "component pytest TMPDIR must be absolute"
        )
    try:
        metadata = os.lstat(parent)
        canonical_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise ConformanceError(
            "component pytest TMPDIR is unavailable"
        ) from exc
    if (
        canonical_parent != parent
        or parent.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or canonical_parent == repository
        or canonical_parent.is_relative_to(repository)
    ):
        raise ConformanceError(
            "component pytest TMPDIR must be canonical, checkout-external, "
            "host-owned mode 0700"
        )
    session_name = "p"
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    created_session = False
    parent_identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )
    try:
        opened_parent = os.fstat(parent_descriptor)
        if (
            opened_parent.st_dev,
            opened_parent.st_ino,
            opened_parent.st_mode,
            opened_parent.st_uid,
            opened_parent.st_gid,
        ) != parent_identity:
            raise ConformanceError(
                "component pytest TMPDIR changed while opened"
            )
        os.mkdir(session_name, mode=0o700, dir_fd=parent_descriptor)
        created_session = True
        created = os.stat(
            session_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        reopened_parent = os.lstat(parent)
        reopened_parent_identity = (
            reopened_parent.st_dev,
            reopened_parent.st_ino,
            reopened_parent.st_mode,
            reopened_parent.st_uid,
            reopened_parent.st_gid,
        )
        if (
            not stat.S_ISDIR(created.st_mode)
            or created.st_uid != os.getuid()
            or stat.S_IMODE(created.st_mode) != 0o700
            or reopened_parent_identity != parent_identity
        ):
            raise ConformanceError(
                "component pytest session creation changed identity"
            )
    except BaseException as exc:
        if created_session:
            os.rmdir(session_name, dir_fd=parent_descriptor)
        if isinstance(exc, ConformanceError):
            raise
        raise ConformanceError(
            "component pytest session creation failed closed"
        ) from exc
    finally:
        os.close(parent_descriptor)
    session = parent / session_name
    session_identity = (
        created.st_dev,
        created.st_ino,
        created.st_uid,
        created.st_gid,
    )
    return session, session / "t", session_identity


def _require_empty_suite_scratch(*, phase: str) -> None:
    raw = os.environ.get("TMPDIR")
    if not isinstance(raw, str):
        raise ConformanceError(
            f"suite scratch is unavailable at {phase}"
        )
    root = Path(raw)
    try:
        named = os.lstat(root)
        descriptor = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise ConformanceError(
            f"suite scratch is unsafe at {phase}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(named.st_mode)
            or named.st_uid != os.getuid()
            or stat.S_IMODE(named.st_mode) != 0o700
            or _metadata_identity(opened) != _metadata_identity(named)
        ):
            raise ConformanceError(
                f"suite scratch changed identity at {phase}"
            )
        entries = sorted(os.listdir(descriptor))
        reopened = os.lstat(root)
        if _metadata_identity(reopened) != _metadata_identity(named):
            raise ConformanceError(
                f"suite scratch changed while inspected at {phase}"
            )
        if entries:
            raise ConformanceError(
                f"suite scratch is not empty at {phase}: {entries}"
            )
    finally:
        os.close(descriptor)


def _remove_owned_private_tree(
    root: Path,
    *,
    expected_identity: tuple[int, ...],
    label: str,
) -> None:
    try:
        metadata = os.lstat(root)
    except OSError as exc:
        raise ConformanceError(
            f"{label} root disappeared"
        ) from exc
    if (
        root.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_gid,
        ) != expected_identity
    ):
        raise ConformanceError(
            f"{label} root changed identity"
        )
    if not shutil.rmtree.avoids_symlink_attacks:
        raise ConformanceError(
            f"{label} cleanup requires fd-safe rmtree"
        )
    parent_descriptor = os.open(
        root.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    session_descriptor = -1
    try:
        session_descriptor = os.open(
            root.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(session_descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_uid,
            opened.st_gid,
        ) != expected_identity:
            raise ConformanceError(
                f"{label} root changed before cleanup"
            )
        for _current, directories, _files, descriptor in os.fwalk(
            ".",
            topdown=True,
            follow_symlinks=False,
            dir_fd=session_descriptor,
        ):
            current_metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(current_metadata.st_mode)
                or current_metadata.st_uid != os.getuid()
            ):
                raise ConformanceError(
                    f"{label} contains an unsafe directory"
                )
            os.fchmod(descriptor, 0o700)
            for directory in directories:
                child_metadata = os.stat(
                    directory,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(child_metadata.st_mode):
                    continue
                if (
                    not stat.S_ISDIR(child_metadata.st_mode)
                    or child_metadata.st_uid != os.getuid()
                ):
                    raise ConformanceError(
                        f"{label} contains an unsafe directory"
                    )
                child_descriptor = os.open(
                    directory,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                try:
                    os.fchmod(child_descriptor, 0o700)
                finally:
                    os.close(child_descriptor)
        reopened = os.stat(
            root.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            reopened.st_dev,
            reopened.st_ino,
            reopened.st_uid,
            reopened.st_gid,
        ) != expected_identity:
            raise ConformanceError(
                f"{label} root changed during cleanup"
            )
        os.close(session_descriptor)
        session_descriptor = -1
        shutil.rmtree(root.name, dir_fd=parent_descriptor)
        try:
            os.stat(
                root.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ConformanceError(
                f"{label} root was not removed"
            )
    except OSError as exc:
        raise ConformanceError(
            f"{label} cleanup failed closed"
        ) from exc
    finally:
        if session_descriptor >= 0:
            os.close(session_descriptor)
        os.close(parent_descriptor)


def _remove_component_pytest_basetemp(
    session: Path,
    *,
    expected_identity: tuple[int, ...],
) -> None:
    _remove_owned_private_tree(
        session,
        expected_identity=expected_identity,
        label="component pytest session",
    )


def _run_component_pytest(
) -> tuple[int, list[str], dict[str, str], str]:
    """Run every test from the sealed component-file allowlist."""

    try:
        import pytest
    except ImportError as exc:
        raise ConformanceError("pytest is required by conformance") from exc
    recorder = _PytestRecorder()
    output = StringIO()
    session, basetemp, session_identity = (
        _new_component_pytest_basetemp()
    )
    try:
        with redirect_stdout(output), redirect_stderr(output):
            exit_code = int(pytest.main(
                _component_pytest_arguments(
                    collect_only=False,
                    basetemp=basetemp,
                ),
                plugins=[recorder],
            ))
    finally:
        _remove_component_pytest_basetemp(
            session,
            expected_identity=session_identity,
        )
    return (
        exit_code,
        recorder.collected,
        dict(recorder.outcomes),
        output.getvalue(),
    )


def _collect_component_pytest() -> tuple[int, list[str], str]:
    """Resolve the complete sealed component inventory before execution."""

    try:
        import pytest
    except ImportError as exc:
        raise ConformanceError("pytest is required by conformance") from exc
    recorder = _PytestRecorder()
    output = StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        exit_code = int(pytest.main(
            _component_pytest_arguments(collect_only=True),
            plugins=[recorder],
        ))
    return exit_code, recorder.collected, output.getvalue()


def _build_component_suite_evidence(
    *,
    repository: Path,
    control_snapshot: Mapping[str, Any],
    collect_exit_code: int,
    collected_nodeids: list[str] | None,
    pytest_exit_code: int,
    run_collected_nodeids: list[str] | None,
    outcomes: dict[str, str] | None,
    pytest_output: str,
) -> dict[str, Any]:
    file_snapshot = component_test_files_snapshot(
        repository,
        control_snapshot=control_snapshot,
    )
    preflight = _bounded_component_nodeids(
        [] if collected_nodeids is None else collected_nodeids,
        label="component preflight inventory",
    )
    executed = _bounded_component_nodeids(
        (
            []
            if run_collected_nodeids is None
            else run_collected_nodeids
        ),
        label="component execution inventory",
    )
    raw_outcomes: object = {} if outcomes is None else outcomes
    if (
        not isinstance(raw_outcomes, dict)
        or len(raw_outcomes) > MAX_COMPONENT_TEST_CASES
        or any(
            not isinstance(nodeid, str)
            or not nodeid
            or len(nodeid.encode("utf-8"))
            > MAX_COMPONENT_NODEID_BYTES
            or not isinstance(status, str)
            for nodeid, status in raw_outcomes.items()
        )
    ):
        raise ConformanceError(
            "component outcome map is malformed or exceeds its bound"
        )
    normalized_outcomes = dict(raw_outcomes)
    outcome_only = sorted(
        set(normalized_outcomes) - set(executed)
    )
    outcome_order = [*executed, *outcome_only]
    if len(outcome_order) > MAX_COMPONENT_TEST_CASES:
        raise ConformanceError(
            "component outcomes exceed the deterministic receipt bound"
        )
    outcome_entries = [
        {
            "nodeid": nodeid,
            "status": (
                normalized_outcomes.get(nodeid, "MISSING")
                if normalized_outcomes.get(nodeid, "MISSING")
                in COMPONENT_STATUS_VALUES
                else "FAIL"
            ),
        }
        for nodeid in outcome_order
    ]
    if (
        len(_canonical_json(outcome_entries))
        > MAX_COMPONENT_INVENTORY_BYTES
    ):
        raise ConformanceError(
            "component outcomes exceed the receipt byte bound"
        )
    preflight_facts = _component_collection_facts(preflight)
    execution_facts = _component_collection_facts(executed)
    collection_stable = preflight == executed
    statuses = [entry["status"] for entry in outcome_entries]
    expected_outcome_keys = (
        len(executed) == len(set(executed))
        and set(normalized_outcomes) == set(executed)
    )
    passed = statuses.count("PASS")
    failed = statuses.count("FAIL")
    skipped = statuses.count("SKIP")
    xfailed = statuses.count("XFAIL")
    xpassed = statuses.count("XPASS")
    missing = statuses.count("MISSING")
    inventory_body = {
        "component_test_files_sha256": file_snapshot["sha256"],
        "nodeids": preflight,
    }
    run_inventory_body = {
        "component_test_files_sha256": file_snapshot["sha256"],
        "nodeids": executed,
    }
    inventory_sha256 = _sha256(_canonical_json(inventory_body))
    run_inventory_sha256 = _sha256(
        _canonical_json(run_inventory_body)
    )
    outcomes_body = {
        "run_inventory_sha256": run_inventory_sha256,
        "outcomes": outcome_entries,
    }
    status: Literal["PASS", "FAIL"] = (
        "PASS"
        if (
            collect_exit_code == 0
            and pytest_exit_code == 0
            and collection_stable
            and not preflight_facts["duplicate_nodeids"]
            and not preflight_facts["unknown_nodeids"]
            and not preflight_facts["missing_files"]
            and not execution_facts["duplicate_nodeids"]
            and not execution_facts["unknown_nodeids"]
            and not execution_facts["missing_files"]
            and expected_outcome_keys
            and passed == len(executed)
            and failed == skipped == xfailed == xpassed == missing == 0
        )
        else "FAIL"
    )
    return {
        "component_suite_status": status,
        "component_test_files": file_snapshot["files"],
        "component_test_files_sha256": file_snapshot["sha256"],
        "component_suite_collect_exit_code": collect_exit_code,
        "component_suite_pytest_exit_code": pytest_exit_code,
        "component_suite_collection_stable": collection_stable,
        "component_suite_inventory": preflight,
        "component_suite_inventory_sha256": inventory_sha256,
        "component_suite_run_inventory_sha256":
            run_inventory_sha256,
        "component_suite_outcomes": outcome_entries,
        "component_suite_outcomes_sha256":
            _sha256(_canonical_json(outcomes_body)),
        "component_suite_output_sha256": _sha256(
            pytest_output.encode("utf-8", errors="replace")
        ),
        "component_suite_summary": {
            "files_required": len(COMPONENT_TEST_FILES),
            "files_represented": (
                len(COMPONENT_TEST_FILES)
                - len(preflight_facts["missing_files"])
            ),
            "preflight_collected": len(preflight),
            "run_collected": len(executed),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "xfailed": xfailed,
            "xpassed": xpassed,
            "missing": missing,
            "preflight_duplicate_nodeids":
                len(preflight_facts["duplicate_nodeids"]),
            "run_duplicate_nodeids":
                len(execution_facts["duplicate_nodeids"]),
            "preflight_unknown_nodeids":
                len(preflight_facts["unknown_nodeids"]),
            "run_unknown_nodeids":
                len(execution_facts["unknown_nodeids"]),
            "missing_files":
                len(preflight_facts["missing_files"]),
            "unexpected_outcome_nodeids": len(outcome_only),
        },
    }


def build_result(
    *,
    pytest_exit_code: int,
    collected_nodeids: list[str],
    outcomes: dict[str, str],
    pytest_output: str,
    component_collect_exit_code: int = 4,
    component_collected_nodeids: list[str] | None = None,
    component_pytest_exit_code: int = 4,
    component_run_collected_nodeids: list[str] | None = None,
    component_outcomes: dict[str, str] | None = None,
    component_pytest_output: str = "",
    repository: Path | None = None,
    invariants: Iterable[Invariant] = INVARIANTS,
    control_contract_start: Mapping[str, Any] | None = None,
    control_contract_end: Mapping[str, Any] | None = None,
    suite_source_loaded_sha256: str | None = None,
    started_at_ns: int = 0,
    ended_at_ns: int = 0,
    suite_interpreter_path: str | None = None,
    suite_interpreter_sha256: str | None = None,
    suite_runtime_manifest_path: str | None = None,
    suite_runtime_manifest_sha256: str | None = None,
    execution_control_root: str | None = None,
    execution_control_snapshot_immutable: bool | None = None,
    workspace_inventory_start: Mapping[str, Any] | None = None,
    workspace_inventory_end: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    launch_requirements = launch_requirements_snapshot(
        repository=repository
    )
    values = validate_registry(
        invariants, repository=repository
    )
    start_snapshot = (
        control_contract_snapshot(repository=repository)
        if control_contract_start is None
        else dict(control_contract_start)
    )
    end_snapshot = (
        control_contract_snapshot(repository=repository)
        if control_contract_end is None
        else dict(control_contract_end)
    )
    loaded_suite_sha256 = (
        _LOADED_SUITE_SOURCE_SHA256
        if suite_source_loaded_sha256 is None
        else suite_source_loaded_sha256
    )
    try:
        start_suite_sha256 = start_snapshot[
            "files_sha256"
        ][SUITE_CONTROL_PATH]
        end_suite_sha256 = end_snapshot[
            "files_sha256"
        ][SUITE_CONTROL_PATH]
    except (KeyError, TypeError) as exc:
        raise ConformanceError(
            "control snapshots omit the canonical suite source"
        ) from exc
    control_contract_stable = (
        start_snapshot == end_snapshot
        and loaded_suite_sha256 == start_suite_sha256
        and loaded_suite_sha256 == end_suite_sha256
    )
    selected_root = (
        _repository_root() if repository is None else Path(repository)
    ).resolve()
    interpreter = Path(
        sys.executable
        if suite_interpreter_path is None
        else suite_interpreter_path
    ).resolve()
    interpreter_sha256 = (
        _sha256(_regular_file_bytes(interpreter))
        if suite_interpreter_sha256 is None
        else suite_interpreter_sha256
    )
    if (
        (suite_runtime_manifest_path is None)
        != (suite_runtime_manifest_sha256 is None)
    ):
        raise ConformanceError(
            "runtime manifest path and digest must be supplied together"
        )
    runtime_manifest_path = suite_runtime_manifest_path
    runtime_manifest_sha256 = suite_runtime_manifest_sha256
    execution_root = (
        str(selected_root)
        if execution_control_root is None
        else execution_control_root
    )
    if execution_control_snapshot_immutable is None:
        try:
            validate_immutable_control_snapshot(
                Path(execution_root),
                expected_sha256=start_snapshot["sha256"],
            )
            immutable_snapshot = True
        except (ConformanceError, OSError):
            immutable_snapshot = False
    else:
        immutable_snapshot = execution_control_snapshot_immutable
    workspace_start = (
        workspace_root_inventory(selected_root)
        if workspace_inventory_start is None
        else dict(workspace_inventory_start)
    )
    workspace_end = (
        workspace_root_inventory(selected_root)
        if workspace_inventory_end is None
        else dict(workspace_inventory_end)
    )
    workspace_stable = workspace_start == workspace_end
    component_evidence = _build_component_suite_evidence(
        repository=selected_root,
        control_snapshot=start_snapshot,
        collect_exit_code=component_collect_exit_code,
        collected_nodeids=component_collected_nodeids,
        pytest_exit_code=component_pytest_exit_code,
        run_collected_nodeids=component_run_collected_nodeids,
        outcomes=component_outcomes,
        pytest_output=component_pytest_output,
    )
    loaded_modules = loaded_control_modules_snapshot(
        selected_root,
        control_snapshot=start_snapshot,
    )
    expected = [value.nodeid for value in values]
    counts = Counter(collected_nodeids)
    duplicate_nodeids = sorted(
        nodeid for nodeid, count in counts.items() if count != 1
    )
    missing_nodeids = sorted(set(expected) - set(collected_nodeids))
    unexpected_nodeids = sorted(set(collected_nodeids) - set(expected))
    cases = []
    for invariant in values:
        if counts[invariant.nodeid] == 0:
            status = "MISSING"
        elif counts[invariant.nodeid] != 1:
            status = "DUPLICATE"
        else:
            status = outcomes.get(invariant.nodeid, "MISSING")
        if status not in STATUS_VALUES:
            status = "FAIL"
        case_body = {
            **asdict(invariant),
            "status": status,
            "launch_requirements_sha256":
                launch_requirements["sha256"],
            "suite_execution_policy_sha256":
                _sha256(_canonical_json(SUITE_EXECUTION_POLICY)),
            "control_contract_start_sha256":
                start_snapshot["sha256"],
            "suite_source_loaded_sha256": loaded_suite_sha256,
            "suite_interpreter_path": str(interpreter),
            "suite_interpreter_sha256": interpreter_sha256,
            "suite_runtime_manifest_path": runtime_manifest_path,
            "suite_runtime_manifest_sha256": runtime_manifest_sha256,
            "execution_control_root": execution_root,
            "execution_control_snapshot_sha256":
                start_snapshot["sha256"],
            "execution_control_snapshot_immutable":
                immutable_snapshot,
            "workspace_root_inventory_start_sha256":
                workspace_start.get("sha256"),
            "workspace_root_inventory_end_sha256":
                workspace_end.get("sha256"),
            "component_suite_inventory_sha256":
                component_evidence[
                    "component_suite_inventory_sha256"
                ],
            "component_suite_outcomes_sha256":
                component_evidence[
                    "component_suite_outcomes_sha256"
                ],
            "suite_loaded_control_modules_sha256":
                loaded_modules["sha256"],
        }
        cases.append({
            **asdict(invariant),
            "status": status,
            "scenario_receipt_sha256":
                _sha256(_canonical_json(case_body)),
        })
    passed = sum(case["status"] == "PASS" for case in cases)
    failed = sum(case["status"] == "FAIL" for case in cases)
    skipped = sum(case["status"] == "SKIP" for case in cases)
    missing = sum(case["status"] == "MISSING" for case in cases)
    duplicated = sum(case["status"] == "DUPLICATE" for case in cases)
    exact = not (
        duplicate_nodeids or missing_nodeids or unexpected_nodeids
    )
    status: Literal["PASS", "FAIL"] = (
        "PASS"
        if (
            pytest_exit_code == 0
            and component_evidence["component_suite_status"] == "PASS"
            and (
                not immutable_snapshot
                or loaded_modules["complete"] is True
            )
            and control_contract_stable
            and workspace_stable
            and (
                not immutable_snapshot
                or not workspace_start.get("forbidden_entries")
            )
            and exact
            and passed == len(values)
            and failed == skipped == missing == duplicated == 0
        )
        else "FAIL"
    )
    # Lazy import avoids a module-import cycle with supervisor preflight.
    import arc_agi3_contiguous_supervisor as supervisor

    inventory = supervisor.authoritative_inventory()
    supervisor.validate_inventory(inventory)
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "status": status,
        "entry_command": list(ENTRY_COMMAND),
        "suite_execution_policy":
            dict(SUITE_EXECUTION_POLICY),
        "suite_execution_policy_sha256":
            _sha256(_canonical_json(SUITE_EXECUTION_POLICY)),
        "registry_sha256": registry_sha256(
            values, repository=repository
        ),
        "launch_requirements_sha256":
            launch_requirements["sha256"],
        "control_contract_sha256": end_snapshot["sha256"],
        "control_contract_files_sha256":
            end_snapshot["files_sha256"],
        "control_contract_start_sha256": start_snapshot["sha256"],
        "control_contract_end_sha256": end_snapshot["sha256"],
        "control_contract_stable": control_contract_stable,
        "suite_source_loaded_sha256": loaded_suite_sha256,
        "suite_source_start_sha256": start_suite_sha256,
        "suite_source_end_sha256": end_suite_sha256,
        "suite_interpreter_path": str(interpreter),
        "suite_interpreter_sha256": interpreter_sha256,
        "suite_runtime_manifest_path": runtime_manifest_path,
        "suite_runtime_manifest_sha256": runtime_manifest_sha256,
        "execution_control_root": execution_root,
        "execution_control_snapshot_sha256":
            start_snapshot["sha256"],
        "execution_control_snapshot_immutable": immutable_snapshot,
        "workspace_root_inventory_start": workspace_start,
        "workspace_root_inventory_end": workspace_end,
        "workspace_root_inventory_start_sha256":
            workspace_start.get("sha256"),
        "workspace_root_inventory_end_sha256":
            workspace_end.get("sha256"),
        "workspace_root_inventory_stable": workspace_stable,
        "started_at_ns": started_at_ns,
        "ended_at_ns": ended_at_ns,
        "observed_exit_status": pytest_exit_code,
        "scenario_receipts_sha256": {
            case["invariant_id"]: case["scenario_receipt_sha256"]
            for case in cases
        },
        "launch_authority": False,
        "container_image_digest": None,
        "frozen_release_receipt_path": None,
        "frozen_release_receipt_sha256": None,
        "frozen_release_levels": None,
        "production_scenario_driver_receipt_path": None,
        "production_scenario_driver_receipt_sha256": None,
        "production_scenario_receipts_sha256": None,
        "production_scenario_verification_environment_sha256": None,
        "terminal_evidence_sha256": None,
        "inventory_sha256":
            supervisor.authoritative_inventory_sha256(inventory),
        "games": len(inventory),
        "levels": sum(inventory.values()),
        "pytest_exit_code": pytest_exit_code,
        "pytest_output_sha256": _sha256(
            pytest_output.encode("utf-8", errors="replace")
        ),
        "suite_loaded_control_modules_complete":
            loaded_modules["complete"],
        "suite_loaded_control_modules":
            loaded_modules["records"],
        "suite_loaded_control_modules_sha256":
            loaded_modules["sha256"],
        "suite_loaded_control_modules_summary":
            loaded_modules["summary"],
        **component_evidence,
        "cases": cases,
        "summary": {
            "required": len(values),
            "collected": len(collected_nodeids),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "missing": missing,
            "duplicated": duplicated,
            "unexpected": len(unexpected_nodeids),
            "duplicate_nodeids": duplicate_nodeids,
            "missing_nodeids": missing_nodeids,
            "unexpected_nodeids": unexpected_nodeids,
        },
    }


def _validate_pass_component_suite(
    value: Mapping[str, Any],
    *,
    repository: Path | None,
) -> None:
    selected_root = (
        _repository_root() if repository is None else Path(repository)
    )
    expected_files = component_test_files_snapshot(
        selected_root,
        control_snapshot={
            "files_sha256": value["control_contract_files_sha256"]
        },
    )
    if (
        value["component_suite_status"] != "PASS"
        or value["component_test_files"] != expected_files["files"]
        or value["component_test_files_sha256"]
        != expected_files["sha256"]
        or value["component_suite_collect_exit_code"] != 0
        or value["component_suite_pytest_exit_code"] != 0
        or value["component_suite_collection_stable"] is not True
        or re.fullmatch(
            r"[0-9a-f]{64}",
            value["component_suite_output_sha256"],
        )
        is None
    ):
        raise ConformanceError(
            "full component suite is stale, incomplete, or not PASS"
        )
    inventory = _bounded_component_nodeids(
        value["component_suite_inventory"],
        label="component suite receipt inventory",
    )
    facts = _component_collection_facts(inventory)
    if (
        not inventory
        or facts["duplicate_nodeids"]
        or facts["unknown_nodeids"]
        or facts["missing_files"]
    ):
        raise ConformanceError(
            "full component suite inventory is incomplete or unexpected"
        )
    inventory_body = {
        "component_test_files_sha256": expected_files["sha256"],
        "nodeids": inventory,
    }
    inventory_sha256 = _sha256(_canonical_json(inventory_body))
    if (
        value["component_suite_inventory_sha256"]
        != inventory_sha256
        or value["component_suite_run_inventory_sha256"]
        != inventory_sha256
    ):
        raise ConformanceError(
            "full component suite collection drifted or has a stale digest"
        )
    outcome_entries = value["component_suite_outcomes"]
    if (
        not isinstance(outcome_entries, list)
        or len(outcome_entries) != len(inventory)
        or len(_canonical_json(outcome_entries))
        > MAX_COMPONENT_INVENTORY_BYTES
        or any(
            not isinstance(entry, dict)
            or set(entry) != {"nodeid", "status"}
            or entry.get("nodeid") != nodeid
            or entry.get("status") != "PASS"
            for nodeid, entry in zip(
                inventory, outcome_entries, strict=False
            )
        )
    ):
        raise ConformanceError(
            "full component suite masks a failed, skipped, or missing test"
        )
    outcomes_body = {
        "run_inventory_sha256": inventory_sha256,
        "outcomes": outcome_entries,
    }
    if value["component_suite_outcomes_sha256"] != _sha256(
        _canonical_json(outcomes_body)
    ):
        raise ConformanceError(
            "full component suite outcome digest is stale or forged"
        )
    expected_summary = {
        "files_required": len(COMPONENT_TEST_FILES),
        "files_represented": len(COMPONENT_TEST_FILES),
        "preflight_collected": len(inventory),
        "run_collected": len(inventory),
        "passed": len(inventory),
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "missing": 0,
        "preflight_duplicate_nodeids": 0,
        "run_duplicate_nodeids": 0,
        "preflight_unknown_nodeids": 0,
        "run_unknown_nodeids": 0,
        "missing_files": 0,
        "unexpected_outcome_nodeids": 0,
    }
    if value["component_suite_summary"] != expected_summary:
        raise ConformanceError(
            "full component suite summary masks a non-PASS outcome"
        )


def _validate_loaded_control_module_receipt(
    value: Mapping[str, Any],
) -> None:
    records = value["suite_loaded_control_modules"]
    summary = value["suite_loaded_control_modules_summary"]
    files = value["control_contract_files_sha256"]
    if (
        not isinstance(
            value["suite_loaded_control_modules_complete"], bool
        )
        or not isinstance(records, list)
        or len(records) > MAX_LOADED_CONTROL_MODULES
        or not isinstance(summary, dict)
        or set(summary)
        != {
            "required",
            "represented",
            "records",
            "missing_required",
            "conflicting_origins",
            "unsealed_local_modules",
        }
    ):
        raise ConformanceError(
            "loaded control module receipt is malformed"
        )
    seen: set[tuple[str, str]] = set()
    represented: set[str] = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"module", "path", "sha256"}
            or not isinstance(record["module"], str)
            or not record["module"]
            or not isinstance(record["path"], str)
            or record["path"] not in files
            or not record["path"].endswith(".py")
            or record["sha256"] != files[record["path"]]
            or (record["module"], record["path"]) in seen
        ):
            raise ConformanceError(
                "loaded control module record is stale or ambiguous"
            )
        seen.add((record["module"], record["path"]))
        represented.add(record["path"])
    if records != sorted(
        records,
        key=lambda record: (record["module"], record["path"]),
    ):
        raise ConformanceError(
            "loaded control module records are not canonical"
        )
    required = {SUITE_CONTROL_PATH, *COMPONENT_TEST_FILES}
    missing = required - represented
    body = {
        "records": records,
        "required_paths": sorted(required),
    }
    if value["suite_loaded_control_modules_sha256"] != _sha256(
        _canonical_json(body)
    ):
        raise ConformanceError(
            "loaded control module digest is stale or forged"
        )
    if (
        not all(
            isinstance(summary[field], int)
            and not isinstance(summary[field], bool)
            and summary[field] >= 0
            for field in summary
        )
        or summary["required"] != len(required)
        or summary["represented"] != len(required) - len(missing)
        or summary["records"] != len(records)
        or summary["missing_required"] != len(missing)
    ):
        raise ConformanceError(
            "loaded control module summary is inconsistent"
        )
    if value["execution_control_snapshot_immutable"]:
        if (
            value["suite_loaded_control_modules_complete"] is not True
            or missing
            or summary["conflicting_origins"] != 0
            or summary["unsealed_local_modules"] != 0
        ):
            raise ConformanceError(
                "immutable suite used missing, live, or unsealed modules"
            )


def _validate_workspace_inventory_receipt(
    value: object,
    *,
    expected_root: Path,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {"root", "entries", "forbidden_entries", "sha256"}
        or value["root"] != str(expected_root.resolve())
        or not isinstance(value["entries"], list)
        or len(value["entries"]) > MAX_WORKSPACE_ROOT_ENTRIES
        or not isinstance(value["forbidden_entries"], list)
        or not isinstance(value["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
    ):
        raise ConformanceError(
            "workspace root inventory receipt is malformed"
        )
    records = value["entries"]
    if (
        any(
            not isinstance(record, dict)
            or set(record)
            != {
                "name",
                "kind",
                "device",
                "inode",
                "mode",
                "nlink",
                "uid",
            }
            or not isinstance(record["name"], str)
            or not record["name"]
            or len(record["name"].encode("utf-8"))
            > MAX_WORKSPACE_ENTRY_NAME_BYTES
            or record["kind"]
            not in {"directory", "file", "symlink", "other"}
            or any(
                not isinstance(record[field], int)
                or isinstance(record[field], bool)
                or record[field] < 0
                for field in (
                    "device",
                    "inode",
                    "mode",
                    "nlink",
                    "uid",
                )
            )
            for record in records
        )
        or records
        != sorted(records, key=lambda record: record["name"])
        or len({record["name"] for record in records}) != len(records)
    ):
        raise ConformanceError(
            "workspace root inventory entries are not exact"
        )
    expected_forbidden = [
        record["name"]
        for record in records
        if (
            record["kind"] == "directory"
            and (
                record["name"].startswith((
                    ".a3cb_",
                    ".a3vr_",
                    ".a3sock_",
                    ".a3s_",
                ))
                or re.fullmatch(
                    r"a[0-9a-f]{2}", record["name"]
                )
                is not None
            )
        )
    ]
    if value["forbidden_entries"] != expected_forbidden:
        raise ConformanceError(
            "workspace junk inventory is stale or forged"
        )
    body = {
        "root": value["root"],
        "entries": records,
        "forbidden_entries": expected_forbidden,
    }
    if value["sha256"] != _sha256(_canonical_json(body)):
        raise ConformanceError(
            "workspace root inventory hash is stale or forged"
        )
    return dict(value)


TERMINAL_EVIDENCE_FIELDS = (
    "container_image_digest",
    "frozen_release_receipt_path",
    "frozen_release_receipt_sha256",
    "frozen_release_levels",
    "production_scenario_driver_receipt_path",
    "production_scenario_driver_receipt_sha256",
    "production_scenario_receipts_sha256",
    "production_scenario_verification_environment_sha256",
    "control_contract_sha256",
    "inventory_sha256",
    "registry_sha256",
    "launch_requirements_sha256",
    "suite_execution_policy_sha256",
    "scenario_receipts_sha256",
    "component_suite_inventory_sha256",
    "component_suite_outcomes_sha256",
    "suite_loaded_control_modules_sha256",
    "suite_source_loaded_sha256",
    "suite_interpreter_path",
    "suite_interpreter_sha256",
    "suite_runtime_manifest_path",
    "suite_runtime_manifest_sha256",
    "execution_control_root",
    "execution_control_snapshot_sha256",
    "execution_control_snapshot_immutable",
    "workspace_root_inventory_start_sha256",
    "workspace_root_inventory_end_sha256",
    "started_at_ns",
    "ended_at_ns",
)


def _terminal_evidence_body(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return {
            field: value[field]
            for field in TERMINAL_EVIDENCE_FIELDS
        }
    except KeyError as exc:
        raise ConformanceError(
            "terminal evidence omits a bound field"
        ) from exc


def validate_result(
    value: object, *, repository: Path | None = None
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConformanceError("conformance result must be an object")
    required = {
        "schema",
        "kind",
        "status",
        "entry_command",
        "suite_execution_policy",
        "suite_execution_policy_sha256",
        "registry_sha256",
        "launch_requirements_sha256",
        "control_contract_sha256",
        "control_contract_files_sha256",
        "control_contract_start_sha256",
        "control_contract_end_sha256",
        "control_contract_stable",
        "suite_source_loaded_sha256",
        "suite_source_start_sha256",
        "suite_source_end_sha256",
        "suite_interpreter_path",
        "suite_interpreter_sha256",
        "suite_runtime_manifest_path",
        "suite_runtime_manifest_sha256",
        "execution_control_root",
        "execution_control_snapshot_sha256",
        "execution_control_snapshot_immutable",
        "workspace_root_inventory_start",
        "workspace_root_inventory_end",
        "workspace_root_inventory_start_sha256",
        "workspace_root_inventory_end_sha256",
        "workspace_root_inventory_stable",
        "started_at_ns",
        "ended_at_ns",
        "observed_exit_status",
        "scenario_receipts_sha256",
        "launch_authority",
        "container_image_digest",
        "frozen_release_receipt_path",
        "frozen_release_receipt_sha256",
        "frozen_release_levels",
        "production_scenario_driver_receipt_path",
        "production_scenario_driver_receipt_sha256",
        "production_scenario_receipts_sha256",
        "production_scenario_verification_environment_sha256",
        "terminal_evidence_sha256",
        "inventory_sha256",
        "games",
        "levels",
        "pytest_exit_code",
        "pytest_output_sha256",
        "component_suite_status",
        "component_test_files",
        "component_test_files_sha256",
        "component_suite_collect_exit_code",
        "component_suite_pytest_exit_code",
        "component_suite_collection_stable",
        "component_suite_inventory",
        "component_suite_inventory_sha256",
        "component_suite_run_inventory_sha256",
        "component_suite_outcomes",
        "component_suite_outcomes_sha256",
        "component_suite_output_sha256",
        "component_suite_summary",
        "suite_loaded_control_modules_complete",
        "suite_loaded_control_modules",
        "suite_loaded_control_modules_sha256",
        "suite_loaded_control_modules_summary",
        "cases",
        "summary",
    }
    launch_requirements = launch_requirements_snapshot(
        repository=repository
    )
    values = validate_registry(repository=repository)
    files = control_contract_files_sha256(repository)
    if (
        set(value) != required
        or value["schema"] != SCHEMA
        or isinstance(value["schema"], bool)
        or value["kind"] != KIND
        or value["status"] != "PASS"
        or value["entry_command"] != list(ENTRY_COMMAND)
        or value["suite_execution_policy"]
        != SUITE_EXECUTION_POLICY
        or value["suite_execution_policy_sha256"]
        != _sha256(_canonical_json(SUITE_EXECUTION_POLICY))
        or value["registry_sha256"]
        != registry_sha256(repository=repository)
        or value["launch_requirements_sha256"]
        != launch_requirements["sha256"]
        or value["launch_requirements_sha256"]
        != files[LAUNCH_REQUIREMENTS_CONTROL_PATH]
        or value["control_contract_files_sha256"] != files
        or value["control_contract_sha256"]
        != control_contract_sha256(repository)
        or value["control_contract_start_sha256"]
        != value["control_contract_sha256"]
        or value["control_contract_end_sha256"]
        != value["control_contract_sha256"]
        or value["control_contract_stable"] is not True
        or value["suite_source_loaded_sha256"]
        != files[SUITE_CONTROL_PATH]
        or value["suite_source_start_sha256"]
        != value["suite_source_loaded_sha256"]
        or value["suite_source_end_sha256"]
        != value["suite_source_loaded_sha256"]
        or not isinstance(value["suite_interpreter_path"], str)
        or not Path(value["suite_interpreter_path"]).is_absolute()
        or re.fullmatch(
            r"[0-9a-f]{64}",
            value["suite_interpreter_sha256"],
        )
        is None
        or (
            (
                value["suite_runtime_manifest_path"] is None
                and value["suite_runtime_manifest_sha256"] is not None
            )
            or (
                value["suite_runtime_manifest_path"] is not None
                and (
                    not isinstance(
                        value["suite_runtime_manifest_path"], str
                    )
                    or not Path(
                        value["suite_runtime_manifest_path"]
                    ).is_absolute()
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        value["suite_runtime_manifest_sha256"],
                    )
                    is None
                )
            )
        )
        or not isinstance(value["execution_control_root"], str)
        or not Path(value["execution_control_root"]).is_absolute()
        or value["execution_control_snapshot_sha256"]
        != value["control_contract_sha256"]
        or not isinstance(
            value["execution_control_snapshot_immutable"], bool
        )
        or value["workspace_root_inventory_stable"] is not True
        or not isinstance(value["started_at_ns"], int)
        or isinstance(value["started_at_ns"], bool)
        or value["started_at_ns"] < 0
        or not isinstance(value["ended_at_ns"], int)
        or isinstance(value["ended_at_ns"], bool)
        or value["ended_at_ns"] < value["started_at_ns"]
        or value["observed_exit_status"] != value["pytest_exit_code"]
        or not isinstance(value["scenario_receipts_sha256"], dict)
        or not isinstance(value["launch_authority"], bool)
        or value["games"] != EXPECTED_GAMES
        or value["levels"] != EXPECTED_LEVELS
        or value["pytest_exit_code"] != 0
        or not isinstance(value["pytest_output_sha256"], str)
        or len(value["pytest_output_sha256"]) != 64
        or not isinstance(value["cases"], list)
        or len(value["cases"]) != len(values)
        or not isinstance(value["summary"], dict)
    ):
        raise ConformanceError(
            "conformance result is stale, incomplete, or not PASS"
        )
    execution_root = Path(value["execution_control_root"])
    workspace_start = _validate_workspace_inventory_receipt(
        value["workspace_root_inventory_start"],
        expected_root=execution_root,
    )
    workspace_end = _validate_workspace_inventory_receipt(
        value["workspace_root_inventory_end"],
        expected_root=execution_root,
    )
    if (
        workspace_start != workspace_end
        or value["workspace_root_inventory_start_sha256"]
        != workspace_start["sha256"]
        or value["workspace_root_inventory_end_sha256"]
        != workspace_end["sha256"]
        or workspace_root_inventory(execution_root)
        != workspace_end
        or (
            value["execution_control_snapshot_immutable"]
            and workspace_end["forbidden_entries"]
        )
    ):
        raise ConformanceError(
            "workspace root changed, leaked junk, or has stale evidence"
        )
    _validate_pass_component_suite(
        value, repository=repository
    )
    _validate_loaded_control_module_receipt(value)
    try:
        interpreter_sha256 = _sha256(
            _regular_file_bytes(
                Path(value["suite_interpreter_path"])
            )
        )
    except (ConformanceError, OSError) as exc:
        raise ConformanceError(
            "suite interpreter evidence cannot be reopened"
        ) from exc
    if interpreter_sha256 != value["suite_interpreter_sha256"]:
        raise ConformanceError(
            "suite interpreter bytes changed"
        )
    if value["suite_runtime_manifest_path"] is not None:
        try:
            runtime_manifest_sha256 = _sha256(
                _regular_file_bytes(
                    Path(value["suite_runtime_manifest_path"])
                )
            )
        except (ConformanceError, OSError) as exc:
            raise ConformanceError(
                "suite runtime manifest evidence cannot be reopened"
            ) from exc
        if (
            runtime_manifest_sha256
            != value["suite_runtime_manifest_sha256"]
        ):
            raise ConformanceError(
                "suite runtime manifest bytes changed"
            )
    if value["execution_control_snapshot_immutable"]:
        validate_immutable_control_snapshot(
            Path(value["execution_control_root"]),
            expected_sha256=value[
                "execution_control_snapshot_sha256"
            ],
        )
    expected_cases = [asdict(invariant) for invariant in values]
    seen_ids: set[str] = set()
    seen_nodes: set[str] = set()
    for expected, case in zip(expected_cases, value["cases"], strict=True):
        case_body = {
            **expected,
            "status": "PASS",
            "launch_requirements_sha256":
                value["launch_requirements_sha256"],
            "suite_execution_policy_sha256":
                value["suite_execution_policy_sha256"],
            "control_contract_start_sha256":
                value["control_contract_start_sha256"],
            "suite_source_loaded_sha256":
                value["suite_source_loaded_sha256"],
            "suite_interpreter_path":
                value["suite_interpreter_path"],
            "suite_interpreter_sha256":
                value["suite_interpreter_sha256"],
            "suite_runtime_manifest_path":
                value["suite_runtime_manifest_path"],
            "suite_runtime_manifest_sha256":
                value["suite_runtime_manifest_sha256"],
            "execution_control_root":
                value["execution_control_root"],
            "execution_control_snapshot_sha256":
                value["execution_control_snapshot_sha256"],
            "execution_control_snapshot_immutable":
                value["execution_control_snapshot_immutable"],
            "workspace_root_inventory_start_sha256":
                value["workspace_root_inventory_start_sha256"],
            "workspace_root_inventory_end_sha256":
                value["workspace_root_inventory_end_sha256"],
            "component_suite_inventory_sha256":
                value["component_suite_inventory_sha256"],
            "component_suite_outcomes_sha256":
                value["component_suite_outcomes_sha256"],
            "suite_loaded_control_modules_sha256":
                value["suite_loaded_control_modules_sha256"],
        }
        expected_scenario_sha256 = _sha256(
            _canonical_json(case_body)
        )
        if (
            not isinstance(case, dict)
            or set(case)
            != {*expected, "status", "scenario_receipt_sha256"}
            or any(case[key] != expected[key] for key in expected)
            or case["status"] != "PASS"
            or case["scenario_receipt_sha256"]
            != expected_scenario_sha256
            or case["invariant_id"] in seen_ids
            or case["nodeid"] in seen_nodes
        ):
            raise ConformanceError(
                "conformance case registry is skipped, duplicated, or altered"
            )
        seen_ids.add(case["invariant_id"])
        seen_nodes.add(case["nodeid"])
    if value["scenario_receipts_sha256"] != {
        case["invariant_id"]: case["scenario_receipt_sha256"]
        for case in value["cases"]
    }:
        raise ConformanceError(
            "scenario receipt hash map is missing or substituted"
        )
    expected_summary = {
        "required": len(values),
        "collected": len(values),
        "passed": len(values),
        "failed": 0,
        "skipped": 0,
        "missing": 0,
        "duplicated": 0,
        "unexpected": 0,
        "duplicate_nodeids": [],
        "missing_nodeids": [],
        "unexpected_nodeids": [],
    }
    if value["summary"] != expected_summary:
        raise ConformanceError(
            "green aggregate masks a skipped, duplicated, or unexpected case"
        )
    import arc_agi3_contiguous_supervisor as supervisor

    if value["inventory_sha256"] != (
        supervisor.authoritative_inventory_sha256()
    ):
        raise ConformanceError(
            "conformance result targets another inventory"
        )
    terminal_values = (
        value["container_image_digest"],
        value["frozen_release_receipt_path"],
        value["frozen_release_receipt_sha256"],
        value["frozen_release_levels"],
        value["production_scenario_driver_receipt_path"],
        value["production_scenario_driver_receipt_sha256"],
        value["production_scenario_receipts_sha256"],
        value[
            "production_scenario_verification_environment_sha256"
        ],
        value["terminal_evidence_sha256"],
    )
    if value["launch_authority"] is False:
        if any(item is not None for item in terminal_values):
            raise ConformanceError(
                "prelaunch conformance carries unauthenticated terminal evidence"
            )
    elif (
        value["execution_control_snapshot_immutable"] is not True
        or
        not isinstance(value["container_image_digest"], str)
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            value["container_image_digest"],
        )
        is None
        or not isinstance(value["frozen_release_receipt_path"], str)
        or not Path(value["frozen_release_receipt_path"]).is_absolute()
        or not isinstance(
            value["frozen_release_receipt_sha256"], str
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            value["frozen_release_receipt_sha256"],
        )
        is None
        or value["frozen_release_levels"] != EXPECTED_LEVELS
        or not isinstance(
            value["production_scenario_driver_receipt_path"],
            str,
        )
        or not Path(
            value["production_scenario_driver_receipt_path"]
        ).is_absolute()
        or not isinstance(
            value["production_scenario_driver_receipt_sha256"],
            str,
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            value["production_scenario_driver_receipt_sha256"],
        )
        is None
        or not isinstance(
            value["production_scenario_receipts_sha256"],
            str,
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            value["production_scenario_receipts_sha256"],
        )
        is None
        or not isinstance(
            value[
                "production_scenario_verification_environment_sha256"
            ],
            str,
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            value[
                "production_scenario_verification_environment_sha256"
            ],
        )
        is None
        or not isinstance(value["terminal_evidence_sha256"], str)
        or re.fullmatch(
            r"[0-9a-f]{64}", value["terminal_evidence_sha256"]
        )
        is None
    ):
        raise ConformanceError(
            "terminal launch authority fields are malformed"
        )
    else:
        terminal_body = _terminal_evidence_body(value)
        if value["terminal_evidence_sha256"] != _sha256(
            _canonical_json(terminal_body)
        ):
            raise ConformanceError(
                "terminal launch evidence hash is stale or forged"
            )
    return value


def _verify_production_scenario_authority(
    base: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    """Reverify S01--S12 with the sealed driver in a fresh interpreter."""

    root = Path(str(base["execution_control_root"])).resolve()
    driver = root / SCENARIO_DRIVER_CONTROL_PATH
    interpreter = Path(str(base["suite_interpreter_path"])).resolve()
    receipt = Path(receipt_path)
    if not receipt.is_absolute():
        raise ConformanceError(
            "production scenario receipt path must be absolute"
        )
    if base["suite_runtime_manifest_path"] is None:
        raise ConformanceError(
            "production scenario authority requires a runtime manifest"
        )
    if (
        base["execution_control_snapshot_immutable"] is not True
        or not driver.is_file()
        or driver.is_symlink()
    ):
        raise ConformanceError(
            "production scenario authority requires the sealed driver"
        )
    neutral = root / ".neutral"
    command = [
        str(interpreter),
        "-I",
        "-E",
        "-s",
        "-S",
        "-B",
        "-c",
        (
            "import runpy,sys;"
            "root=sys.argv.pop(1);"
            "script=sys.argv.pop(1);"
            "sys.path.insert(0,root);"
            "sys.argv=[script,*sys.argv[1:]];"
            "runpy.run_path(script,run_name='__main__')"
        ),
        str(driver.parent),
        str(driver),
        "verify",
        "--repository",
        str(root),
        "--receipt",
        str(receipt),
    ]
    try:
        import arc_agi3_contiguous_supervisor as supervisor

        scratch = supervisor._private_system_scratch()
        scratch_metadata = os.stat(
            scratch, follow_symlinks=False
        )
    except Exception as exc:
        raise ConformanceError(
            "private scenario verifier scratch is unavailable"
        ) from exc
    scratch_identity = (
        scratch_metadata.st_dev,
        scratch_metadata.st_ino,
        scratch_metadata.st_uid,
        scratch_metadata.st_gid,
    )
    environment = {
        "HOME": str(neutral),
        "LANG": "C",
        "LC_ALL": "C",
        "TMPDIR": str(scratch),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    environment_body = {
        "interpreter_flags": ["-I", "-E", "-s", "-S", "-B"],
        "cwd": str(neutral),
        "home": str(neutral),
        "locale": "C",
        "scratch_parent": str(scratch.parent),
        "scratch_policy":
            "private_0700_empty_after_and_removed",
        "process_containment":
            "exact_pid_start_identity_plus_process_group_and_session",
        "stdin": "devnull",
    }
    completed: Any | None = None
    scenario_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        completed = supervisor._run_bounded_process_group(
            tuple(command),
            cwd=neutral,
            environment=environment,
            timeout_seconds=120,
            scratch_root=scratch,
        )
        descriptor = os.open(
            scratch,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            reopened = os.fstat(descriptor)
            if (
                reopened.st_dev,
                reopened.st_ino,
                reopened.st_uid,
                reopened.st_gid,
            ) != scratch_identity:
                raise ConformanceError(
                    "sealed production scenario scratch changed identity"
                )
            scratch_entries = sorted(os.listdir(descriptor))
        finally:
            os.close(descriptor)
        if scratch_entries:
            raise ConformanceError(
                "sealed production scenario verifier leaked scratch files"
            )
    except BaseException as exc:
        scenario_error = exc
    finally:
        try:
            _remove_owned_private_tree(
                scratch,
                expected_identity=scratch_identity,
                label="production scenario verifier scratch",
            )
        except BaseException as exc:
            cleanup_error = exc
    if cleanup_error is not None:
        raise ConformanceError(
            "sealed production scenario scratch cleanup failed closed"
        ) from cleanup_error
    if scenario_error is not None:
        if isinstance(scenario_error, (KeyboardInterrupt, SystemExit)):
            raise scenario_error
        raise ConformanceError(
            "sealed production scenario verification did not complete"
        ) from scenario_error
    assert completed is not None
    stdout = completed.stdout.encode("utf-8")
    stderr = completed.stderr.encode("utf-8")
    if (
        completed.timed_out
        or completed.captured_descendants_absent is not True
        or len(stdout) > MAX_SCENARIO_DRIVER_OUTPUT_BYTES
        or len(stderr) > MAX_SCENARIO_DRIVER_OUTPUT_BYTES
    ):
        raise ConformanceError(
            "sealed production scenario verification exceeded its bound"
        )
    try:
        verified = json.loads(
            stdout,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConformanceError(
            "sealed production scenario verification was malformed"
        ) from exc
    required = {
        "schema",
        "kind",
        "mode",
        "status",
        "launch_authority",
        "receipt_path",
        "receipt_sha256",
        "control_contract_sha256",
        "runtime_manifest_path",
        "runtime_manifest_sha256",
        "scenario_ids",
        "scenario_statuses",
        "scenario_receipts",
        "scenario_receipts_sha256",
    }
    rows = (
        verified.get("scenario_receipts")
        if isinstance(verified, dict)
        else None
    )
    expected_ids = list(EXPECTED_PRODUCTION_SCENARIO_IDS)
    if (
        completed.returncode != 0
        or not isinstance(verified, dict)
        or set(verified) != required
        or stdout != _canonical_json(verified) + b"\n"
        or verified["schema"] != 1
        or isinstance(verified["schema"], bool)
        or verified["kind"]
        != "arc_agi3_contiguous_scenario_driver"
        or verified["mode"] != "verify"
        or verified["status"] != "PASS"
        or verified["launch_authority"] is not True
        or not isinstance(verified["receipt_path"], str)
        or verified["receipt_path"] != str(receipt)
        or not isinstance(verified["receipt_sha256"], str)
        or re.fullmatch(
            r"[0-9a-f]{64}", verified["receipt_sha256"]
        )
        is None
        or verified["control_contract_sha256"]
        != base["control_contract_sha256"]
        or verified["runtime_manifest_path"]
        != base["suite_runtime_manifest_path"]
        or verified["runtime_manifest_sha256"]
        != base["suite_runtime_manifest_sha256"]
        or verified["scenario_ids"] != expected_ids
        or verified["scenario_statuses"]
        != ["PASS"] * len(expected_ids)
        or not isinstance(rows, list)
        or len(rows) != len(expected_ids)
        or [
            row.get("scenario_id")
            if isinstance(row, dict)
            else None
            for row in rows
        ]
        != expected_ids
        or any(
            not isinstance(row, dict)
            or set(row)
            != {"scenario_id", "owner", "path", "sha256", "status"}
            or row["status"] != "PASS"
            or not isinstance(row["owner"], str)
            or not row["owner"]
            or not isinstance(row["path"], str)
            or not Path(row["path"]).is_absolute()
            or not isinstance(row["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
            for row in rows
        )
        or not isinstance(
            verified["scenario_receipts_sha256"], str
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            verified["scenario_receipts_sha256"],
        )
        is None
    ):
        raise ConformanceError(
            "production S01--S12 observations are not exact PASS"
        )
    return {
        **verified,
        "verification_environment_sha256":
            _sha256(_canonical_json(environment_body)),
    }


def bind_terminal_launch_authority(
    result: object,
    *,
    container_image_digest: str,
    release_receipt_path: Path,
    scenario_driver_receipt_path: Path | None = None,
    canonical_root: Path,
    environments_root: Path,
    repository: Path | None = None,
) -> dict[str, Any]:
    """Bind exact production S01--S12, image, and frozen 183 evidence."""

    base = validate_result(result, repository=repository)
    if base["launch_authority"] is not False:
        raise ConformanceError(
            "terminal authority can only bind a prelaunch result once"
        )
    if base["execution_control_snapshot_immutable"] is not True:
        raise ConformanceError(
            "terminal authority requires immutable control execution"
        )
    if re.fullmatch(
        r"sha256:[0-9a-f]{64}", container_image_digest
    ) is None:
        raise ConformanceError(
            "terminal authority requires an immutable image digest"
        )
    if scenario_driver_receipt_path is None:
        raise ConformanceError(
            "terminal authority requires production S01--S12 evidence"
        )
    scenario_verified = _verify_production_scenario_authority(
        base,
        scenario_driver_receipt_path,
    )
    try:
        import arc_agi3_release_gate as release_gate

        verified = release_gate.verify_release_receipt(
            receipt_path=release_receipt_path,
            canonical_root=canonical_root,
            environments_root=environments_root,
        )
    except Exception as exc:
        raise ConformanceError(
            "frozen release receipt failed live 183-boundary verification"
        ) from exc
    release_path = verified.path.resolve()
    if (
        verified.body.get("canonical_game_count") != EXPECTED_GAMES
        or verified.body.get("authoritative_level_count")
        != EXPECTED_LEVELS
        or verified.body.get("inventory_sha256")
        != base["inventory_sha256"]
        or verified.body.get("control_contract", {}).get("sha256")
        != base["control_contract_sha256"]
    ):
        raise ConformanceError(
            "frozen release differs from conformance inventory/control"
        )
    terminal_fields = {
        "container_image_digest": container_image_digest,
        "frozen_release_receipt_path": str(release_path),
        "frozen_release_receipt_sha256": verified.sha256,
        "frozen_release_levels": EXPECTED_LEVELS,
        "production_scenario_driver_receipt_path":
            scenario_verified["receipt_path"],
        "production_scenario_driver_receipt_sha256":
            scenario_verified["receipt_sha256"],
        "production_scenario_receipts_sha256":
            scenario_verified["scenario_receipts_sha256"],
        "production_scenario_verification_environment_sha256":
            scenario_verified["verification_environment_sha256"],
    }
    terminal_body = _terminal_evidence_body({
        **base,
        **terminal_fields,
    })
    terminal = {
        **base,
        **terminal_fields,
        "launch_authority": True,
        "terminal_evidence_sha256":
            _sha256(_canonical_json(terminal_body)),
    }
    return validate_result(terminal, repository=repository)


def validate_launch_authority_result(
    value: object,
    *,
    canonical_root: Path,
    environments_root: Path,
    repository: Path | None = None,
) -> dict[str, Any]:
    """Require terminal fields and reverify the live frozen release bytes."""

    result = validate_result(value, repository=repository)
    if result["launch_authority"] is not True:
        raise ConformanceError(
            "prelaunch conformance is not launch authority"
        )
    rebound = bind_terminal_launch_authority(
        {
            **result,
            "launch_authority": False,
            "container_image_digest": None,
            "frozen_release_receipt_path": None,
            "frozen_release_receipt_sha256": None,
            "frozen_release_levels": None,
            "production_scenario_driver_receipt_path": None,
            "production_scenario_driver_receipt_sha256": None,
            "production_scenario_receipts_sha256": None,
            "production_scenario_verification_environment_sha256":
                None,
            "terminal_evidence_sha256": None,
        },
        container_image_digest=result["container_image_digest"],
        release_receipt_path=Path(
            result["frozen_release_receipt_path"]
        ),
        scenario_driver_receipt_path=Path(
            result["production_scenario_driver_receipt_path"]
        ),
        canonical_root=canonical_root,
        environments_root=environments_root,
        repository=repository,
    )
    if rebound != result:
        raise ConformanceError(
            "terminal launch authority no longer matches live evidence"
        )
    return result


def load_result(
    path: Path, *, repository: Path | None = None
) -> dict[str, Any]:
    raw = _regular_file_bytes(Path(path))
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConformanceError("invalid conformance JSON") from exc
    if raw != _canonical_json(value) + b"\n":
        raise ConformanceError(
            "conformance artifact is not canonical JSON"
        )
    return validate_result(value, repository=repository)


def _run_once(
    repository: Path | None = None,
    *,
    runtime_manifest_path: Path | None = None,
    runtime_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    root = _repository_root() if repository is None else Path(repository)
    started_at_ns = time.time_ns()
    start_snapshot = control_contract_snapshot(repository=root)
    workspace_start = workspace_root_inventory(root)
    component_test_files_snapshot(
        root, control_snapshot=start_snapshot
    )
    values = validate_registry(repository=root)
    component_collected: list[str] = []
    component_exit_code = 4
    component_outcomes: dict[str, str] = {}
    component_output = ""
    previous = Path.cwd()
    try:
        os.chdir(root)
        if (
            start_snapshot["files_sha256"].get(
                SUITE_CONTROL_PATH
            )
            != _LOADED_SUITE_SOURCE_SHA256
        ):
            exit_code = 4
            collected: list[str] = []
            outcomes: dict[str, str] = {}
            output = (
                "loaded conformance source differs from immutable "
                "start snapshot"
            )
        else:
            (
                component_exit_code,
                component_collected,
                component_outcomes,
                component_output,
            ) = _run_component_pytest()
            bounded_component_collected = _bounded_component_nodeids(
                component_collected,
                label="component execution inventory",
            )
            component_facts = _component_collection_facts(
                bounded_component_collected
            )
            expected = [value.nodeid for value in values]
            expected_set = set(expected)
            collected = [
                nodeid
                for nodeid in component_collected
                if nodeid in expected_set
            ]
            outcomes = {
                nodeid: component_outcomes[nodeid]
                for nodeid in expected
                if nodeid in component_outcomes
            }
            if (
                Counter(collected) != Counter(expected)
                or len(collected) != len(expected)
            ):
                exit_code = component_exit_code or 4
            else:
                exit_code = component_exit_code
            output = component_output
            component_collect_exit = (
                0
                if (
                    component_collected
                    and not component_facts["duplicate_nodeids"]
                    and not component_facts["unknown_nodeids"]
                    and not component_facts["missing_files"]
                )
                else (component_exit_code or 4)
            )
        if start_snapshot["files_sha256"].get(
            SUITE_CONTROL_PATH
        ) != _LOADED_SUITE_SOURCE_SHA256:
            component_collect_exit = 4
    finally:
        os.chdir(previous)
        workspace_end = workspace_root_inventory(root)
    ended_at_ns = time.time_ns()
    end_snapshot = control_contract_snapshot(repository=root)
    return build_result(
        pytest_exit_code=exit_code,
        collected_nodeids=collected,
        outcomes=outcomes,
        pytest_output=output,
        component_collect_exit_code=component_collect_exit,
        component_collected_nodeids=component_collected,
        component_pytest_exit_code=component_exit_code,
        component_run_collected_nodeids=component_collected,
        component_outcomes=component_outcomes,
        component_pytest_output=component_output,
        repository=root,
        invariants=values,
        control_contract_start=start_snapshot,
        control_contract_end=end_snapshot,
        suite_source_loaded_sha256=_LOADED_SUITE_SOURCE_SHA256,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
        suite_runtime_manifest_path=(
            None
            if runtime_manifest_path is None
            else str(runtime_manifest_path)
        ),
        suite_runtime_manifest_sha256=runtime_manifest_sha256,
        workspace_inventory_start=workspace_start,
        workspace_inventory_end=workspace_end,
    )


_ACTIVE_RUN_ENVIRONMENT_KEY = "ARC_AGI3_CONTIGUOUS_CONFORMANCE_ACTIVE"


def _require_authoritative_fresh_process(root: Path) -> None:
    selected = Path(root).resolve()
    neutral = selected / ".neutral"
    raw_tmp = os.environ.get("TMPDIR")
    try:
        tmp_root = (
            Path(raw_tmp)
            if isinstance(raw_tmp, str)
            else Path("")
        )
        tmp_metadata = tmp_root.lstat()
    except OSError:
        tmp_metadata = None
    flags = sys.flags
    if (
        flags.isolated != 1
        or flags.ignore_environment != 1
        or flags.no_user_site != 1
        or flags.no_site != 1
        or flags.dont_write_bytecode != 1
        or os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") != "1"
        or Path.cwd().resolve() != neutral
        or os.environ.get("HOME") != str(neutral)
        or os.environ.get("LANG") != "C"
        or os.environ.get("LC_ALL") != "C"
        or not isinstance(raw_tmp, str)
        or not tmp_root.is_absolute()
        or tmp_root.is_symlink()
        or tmp_metadata is None
        or not stat.S_ISDIR(tmp_metadata.st_mode)
        or tmp_metadata.st_uid != os.getuid()
        or stat.S_IMODE(tmp_metadata.st_mode) != 0o700
        or tmp_root.resolve() == selected
        or tmp_root.resolve().is_relative_to(selected)
        or tmp_root.resolve() == _repository_root().resolve()
        or tmp_root.resolve().is_relative_to(
            _repository_root().resolve()
        )
        or Path(__file__).resolve()
        != (selected / SUITE_CONTROL_PATH).resolve()
    ):
        raise ConformanceError(
            "authoritative conformance requires the fresh isolated "
            "snapshot-rooted interpreter contract"
        )
    validate_immutable_control_snapshot(selected)


def _install_sealed_import_roots(root: Path) -> None:
    selected = Path(root).resolve()
    ordered = (
        selected / "arc" / "crack_lab",
        selected / "arc",
        selected / "cone",
        selected,
    )
    if any(
        not path.is_dir()
        or path.is_symlink()
        for path in ordered
    ):
        raise ConformanceError(
            "sealed conformance import roots are incomplete"
        )
    normalized = [str(path) for path in ordered]
    sys.path[:] = [
        *normalized,
        *[entry for entry in sys.path if entry not in normalized],
    ]


def run(
    repository: Path | None = None,
    *,
    runtime_manifest_path: Path | None = None,
    runtime_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one non-recursive owner-plus-full-component release suite."""

    if os.environ.get(_ACTIVE_RUN_ENVIRONMENT_KEY) is not None:
        raise ConformanceError(
            "recursive contiguous conformance execution is forbidden"
        )
    root = _repository_root() if repository is None else Path(repository)
    _require_authoritative_fresh_process(root)
    _install_sealed_import_roots(root)
    token = f"{os.getpid()}:{time.time_ns()}"
    os.environ[_ACTIVE_RUN_ENVIRONMENT_KEY] = token
    try:
        _require_empty_suite_scratch(phase="suite start")
        result = _run_once(
            repository=repository,
            runtime_manifest_path=runtime_manifest_path,
            runtime_manifest_sha256=runtime_manifest_sha256,
        )
        _require_empty_suite_scratch(phase="suite end")
        return result
    finally:
        if os.environ.get(_ACTIVE_RUN_ENVIRONMENT_KEY) == token:
            del os.environ[_ACTIVE_RUN_ENVIRONMENT_KEY]


def _write_new_result(path: Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value) + b"\n"
    descriptor = os.open(
        target,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--stdout", action="store_true")
    destination.add_argument("--output", type=Path)
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--runtime-manifest-sha256",
        required=True,
    )
    args = parser.parse_args(argv)
    result = run(
        runtime_manifest_path=args.runtime_manifest,
        runtime_manifest_sha256=args.runtime_manifest_sha256,
    )
    if args.stdout:
        sys.stdout.buffer.write(_canonical_json(result) + b"\n")
    else:
        _write_new_result(args.output, result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
