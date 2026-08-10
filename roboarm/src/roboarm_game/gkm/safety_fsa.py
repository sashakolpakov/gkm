"""Deterministic finite-state safety gate for untrusted RoboArm scenarios.

This is the only campaign component allowed to turn a proposer-authored action
sequence into a connector commit permit.  The proposer supplies no verdicts.
Every candidate is first executed in an isolated authoritative digital twin;
unsafe, incomplete, or non-goal preflights remain observed learning evidence
but cannot reach the committed transition.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from .arena import ConnectorViolation, RoboArmConnector
from .scenario import (
    ProposalBundle,
    ScenarioProposal,
    canonical_sha256,
)


class SafetyState(str, Enum):
    RECEIVED = "received"
    CONTRACT_VALIDATED = "contract_validated"
    PREFLIGHTING = "preflighting"
    PREFLIGHT_OBSERVED = "preflight_observed"
    COMMIT_DEFERRED = "commit_deferred"
    COMMIT_AUTHORIZED = "commit_authorized"
    COMMITTING = "committing"
    OBSERVED = "observed"
    VERIFIED = "verified"
    REJECTED = "rejected"
    SEALED = "sealed"


ALLOWED_TRANSITIONS: dict[SafetyState, frozenset[SafetyState]] = {
    SafetyState.RECEIVED: frozenset(
        {SafetyState.CONTRACT_VALIDATED, SafetyState.REJECTED}
    ),
    SafetyState.CONTRACT_VALIDATED: frozenset(
        {SafetyState.PREFLIGHTING, SafetyState.REJECTED}
    ),
    SafetyState.PREFLIGHTING: frozenset(
        {SafetyState.PREFLIGHT_OBSERVED, SafetyState.REJECTED}
    ),
    SafetyState.PREFLIGHT_OBSERVED: frozenset(
        {
            SafetyState.COMMIT_DEFERRED,
            SafetyState.COMMIT_AUTHORIZED,
            SafetyState.VERIFIED,
            SafetyState.REJECTED,
        }
    ),
    SafetyState.COMMIT_DEFERRED: frozenset({SafetyState.VERIFIED}),
    SafetyState.COMMIT_AUTHORIZED: frozenset(
        {SafetyState.COMMITTING, SafetyState.REJECTED}
    ),
    SafetyState.COMMITTING: frozenset(
        {SafetyState.OBSERVED, SafetyState.REJECTED}
    ),
    SafetyState.OBSERVED: frozenset(
        {SafetyState.VERIFIED, SafetyState.REJECTED}
    ),
    SafetyState.VERIFIED: frozenset({SafetyState.SEALED}),
    SafetyState.REJECTED: frozenset({SafetyState.SEALED}),
    SafetyState.SEALED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    """Generic actuator safety and campaign-admission bounds."""

    max_contact_load: float = 0.95
    require_complete_candidate_preflight: bool = True

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.max_contact_load)
            or not 0.0 < self.max_contact_load <= 1.0
        ):
            raise ValueError("max_contact_load must be finite in (0, 1]")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "max_contact_load": self.max_contact_load,
            "require_complete_candidate_preflight":
                self.require_complete_candidate_preflight,
            "candidate_requires_goal_preflight": True,
            "experiment_commit_allowed": False,
            "stepwise_commit_interlock": True,
        }


class _Machine:
    def __init__(self) -> None:
        self.state = SafetyState.RECEIVED
        self.transitions: list[dict[str, object]] = [
            {
                "sequence": 0,
                "from": None,
                "to": self.state.value,
                "reason": "scenario_received",
            }
        ]

    def move(self, target: SafetyState, reason: str) -> None:
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(
                f"illegal safety-FSA transition "
                f"{self.state.value!r} -> {target.value!r}"
            )
        prior = self.state
        self.state = target
        self.transitions.append(
            {
                "sequence": len(self.transitions),
                "from": prior.value,
                "to": target.value,
                "reason": reason,
            }
        )


def _event_kinds(snapshot: Mapping[str, object]) -> tuple[str, ...]:
    raw_events = snapshot.get("events")
    if not isinstance(raw_events, list):
        return ()
    result: list[str] = []
    for event in raw_events:
        if isinstance(event, Mapping) and isinstance(event.get("kind"), str):
            result.append(str(event["kind"]))
    return tuple(result)


def _safety_findings(
    trace: Mapping[str, object],
    policy: SafetyPolicy,
    *,
    require_complete: bool,
) -> tuple[str, ...]:
    """Inspect trusted private simulator telemetry; fail closed."""

    findings: list[str] = []
    actions_requested = trace.get("actions_requested")
    actions_executed = trace.get("actions")
    steps = trace.get("steps")
    if (
        not isinstance(actions_requested, list)
        or not isinstance(actions_executed, list)
        or not isinstance(steps, list)
        or len(actions_executed) != len(steps)
    ):
        return ("malformed_preflight_trace",)
    if require_complete and actions_executed != actions_requested:
        findings.append("incomplete_preflight")

    for step in steps:
        if not isinstance(step, Mapping):
            findings.append("malformed_preflight_step")
            continue
        snapshot = step.get("visual_state")
        if not isinstance(snapshot, Mapping):
            findings.append("missing_safety_telemetry")
            continue
        robot = snapshot.get("robot")
        if not isinstance(robot, Mapping):
            findings.append("missing_robot_safety_telemetry")
            continue
        contact_load = robot.get("contactLoad")
        if (
            isinstance(contact_load, bool)
            or not isinstance(contact_load, (int, float))
            or not math.isfinite(float(contact_load))
        ):
            findings.append("invalid_contact_load")
        elif float(contact_load) > policy.max_contact_load:
            findings.append("contact_load_limit")
        if robot.get("rejected") is True:
            reason = robot.get("rejectionReason")
            if isinstance(reason, str) and "collision" in reason:
                findings.append("collision_interlock")
            else:
                findings.append("motion_interlock")
        kinds = _event_kinds(snapshot)
        if "motion_rejected" in kinds:
            findings.append("motion_interlock")
        if "action_budget_exhausted" in kinds:
            findings.append("action_budget_terminal")

    return tuple(dict.fromkeys(findings))


def _observed_failure_evidence(
    trace: Mapping[str, object] | None,
    *,
    proposal_kind: str,
    disposition: str,
) -> tuple[str, ...]:
    """Classify concrete host-observed operational failures.

    A useful probe that merely leaves sparse reward at zero is not a failed
    attempt.  The campaign gate requires an empty grasp, rejected motion,
    collision, or an unsuccessful candidate.
    """

    evidence: list[str] = []
    if isinstance(trace, Mapping):
        steps = trace.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, Mapping):
                    continue
                snapshot = step.get("visual_state")
                if not isinstance(snapshot, Mapping):
                    continue
                kinds = _event_kinds(snapshot)
                if "gripper_closed_empty" in kinds:
                    evidence.append("empty_grasp")
                robot = snapshot.get("robot")
                if not isinstance(robot, Mapping):
                    continue
                if robot.get("rejected") is True or "motion_rejected" in kinds:
                    reason = robot.get("rejectionReason")
                    if isinstance(reason, str) and "collision" in reason:
                        evidence.append("collision_rejection")
                    else:
                        evidence.append("motion_rejection")

    if proposal_kind == "candidate":
        if disposition == "candidate_failed_preflight":
            evidence.append("candidate_goal_not_observed")
        elif disposition == "candidate_rejected_by_fsa":
            evidence.append("candidate_safety_rejection")
        elif disposition in {
            "commit_interlock_rejected",
            "commit_verification_rejected",
        }:
            evidence.append("candidate_commit_rejection")
    return tuple(dict.fromkeys(evidence))


def _trace_public_projection(trace: Mapping[str, object]) -> dict[str, object]:
    """Project only frame/action/reward facts into the next proposer payload."""

    steps: list[dict[str, object]] = []
    for value in trace.get("steps", []):
        if not isinstance(value, Mapping):
            continue
        steps.append(
            {
                "turn": value.get("turn"),
                "role": value.get("role"),
                "action": value.get("action"),
                "before_frame_sha256": value.get("before_frame_sha256"),
                "before_telemetry_sha256": value.get(
                    "before_telemetry_sha256"
                ),
                "frame_sha256": value.get("frame_sha256"),
                "frame_b64": value.get("frame_b64"),
                "telemetry_sha256": value.get("telemetry_sha256"),
                "telemetry": copy.deepcopy(value.get("telemetry")),
                "levels_completed": value.get("levels_completed"),
                "terminal": value.get("terminal"),
            }
        )
    return {
        "schema_version": trace.get("schema_version"),
        "attempt_id": trace.get("attempt_id"),
        "role": trace.get("role"),
        "sensor_contract_id": trace.get("sensor_contract_id"),
        "frame_encoding": trace.get("frame_encoding"),
        "frame_shape": copy.deepcopy(trace.get("frame_shape")),
        "camera_model": copy.deepcopy(trace.get("camera_model")),
        "initial_frame_sha256": trace.get("initial_frame_sha256"),
        "initial_frame_b64": trace.get("initial_frame_b64"),
        "initial_telemetry_sha256": trace.get(
            "initial_telemetry_sha256"
        ),
        "initial_telemetry": copy.deepcopy(
            trace.get("initial_telemetry")
        ),
        "actions_requested": copy.deepcopy(trace.get("actions_requested")),
        "actions": copy.deepcopy(trace.get("actions")),
        "levels_completed": trace.get("levels_completed"),
        "terminal": trace.get("terminal"),
        "steps": steps,
        "connector_receipt_sha256": trace.get("receipt_sha256"),
    }


def public_attempt_projection(
    attempt: Mapping[str, object],
) -> dict[str, object]:
    """Return the host-sealed public evidence supplied to later generations."""

    proposal = attempt.get("proposal")
    preflight = attempt.get("preflight")
    commit = attempt.get("commit")
    result: dict[str, object] = {
        "schema_version": 1,
        "attempt_id": attempt.get("attempt_id"),
        "generation": attempt.get("generation"),
        "scenario_id": (
            proposal.get("scenario_id")
            if isinstance(proposal, Mapping)
            else None
        ),
        "proposal_kind": (
            proposal.get("kind") if isinstance(proposal, Mapping) else None
        ),
        "hypothesis": (
            proposal.get("hypothesis")
            if isinstance(proposal, Mapping)
            else None
        ),
        "expected_observation": (
            proposal.get("expected_observation")
            if isinstance(proposal, Mapping)
            else None
        ),
        "disposition": attempt.get("disposition"),
        "observed_failure_evidence": copy.deepcopy(
            attempt.get("observed_failure_evidence")
        ),
        "authorized_for_commit": attempt.get("authorized_for_commit"),
        "fsa_state": attempt.get("fsa_state"),
        "fsa_public_code": attempt.get("fsa_public_code"),
        "preflight": (
            _trace_public_projection(preflight)
            if isinstance(preflight, Mapping)
            else None
        ),
        "commit": (
            _trace_public_projection(commit)
            if isinstance(commit, Mapping)
            else None
        ),
        "host_receipt_sha256": attempt.get("receipt_sha256"),
    }
    result["public_receipt_sha256"] = canonical_sha256(result)
    return result


def _traces_match(
    preflight: Mapping[str, object],
    commit: Mapping[str, object],
) -> bool:
    if (
        preflight.get("actions") != commit.get("actions")
        or preflight.get("levels_completed") != commit.get("levels_completed")
        or preflight.get("terminal") != commit.get("terminal")
    ):
        return False
    first = preflight.get("steps")
    second = commit.get("steps")
    if not isinstance(first, list) or not isinstance(second, list):
        return False
    if len(first) != len(second):
        return False
    return all(
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("action") == right.get("action")
        and left.get("frame_sha256") == right.get("frame_sha256")
        and left.get("telemetry_sha256")
        == right.get("telemetry_sha256")
        and left.get("levels_completed") == right.get("levels_completed")
        and left.get("terminal") == right.get("terminal")
        for left, right in zip(first, second, strict=True)
    )


class SafetyAutomaton:
    """Validate, preflight, optionally commit, verify, and seal scenarios."""

    def __init__(
        self,
        connector: RoboArmConnector,
        *,
        policy: SafetyPolicy = SafetyPolicy(),
    ) -> None:
        self.connector = connector
        self.policy = policy

    def run_scenario(
        self,
        proposal: ScenarioProposal,
        *,
        generation: int,
        sequence: int,
        commit_enabled: bool,
    ) -> dict[str, object]:
        machine = _Machine()
        attempt_id = (
            f"g{generation:03d}-s{sequence:03d}-{proposal.scenario_id}"
        )
        machine.move(
            SafetyState.CONTRACT_VALIDATED,
            "closed proposal schema validated",
        )
        machine.move(
            SafetyState.PREFLIGHTING,
            "isolated digital-twin preflight required",
        )
        preflight: dict[str, object] | None = None
        commit: dict[str, object] | None = None
        findings: tuple[str, ...] = ()
        authorized = False
        disposition = "rejected"
        public_code = "connector_failure"

        try:
            preflight = self.connector.preflight(
                proposal.actions,
                attempt_id=attempt_id + "-preflight",
            )
        except ConnectorViolation as error:
            machine.move(
                SafetyState.REJECTED,
                f"connector_preflight_failure:{type(error).__name__}",
            )
            public_code = "preflight_unavailable"
        else:
            machine.move(
                SafetyState.PREFLIGHT_OBSERVED,
                "authoritative preflight facts recorded",
            )
            require_complete = (
                proposal.kind == "candidate"
                and self.policy.require_complete_candidate_preflight
            )
            findings = _safety_findings(
                preflight,
                self.policy,
                require_complete=require_complete,
            )
            completed = int(preflight.get("levels_completed", 0)) >= 1

            if proposal.kind == "experiment":
                disposition = (
                    "probe_success_uncommitted"
                    if completed
                    else "probe_observed"
                )
                public_code = "experiment_isolated"
                machine.move(
                    SafetyState.VERIFIED,
                    "experiment observations accepted; commit forbidden",
                )
            elif findings:
                disposition = "candidate_rejected_by_fsa"
                public_code = "preflight_not_commit_safe"
                machine.move(
                    SafetyState.REJECTED,
                    "candidate preflight violated deterministic safety policy",
                )
            elif not completed:
                disposition = "candidate_failed_preflight"
                public_code = "candidate_goal_not_observed"
                machine.move(
                    SafetyState.VERIFIED,
                    "safe preflight did not reach sparse goal; no commit",
                )
            elif not commit_enabled:
                disposition = "candidate_commit_deferred"
                public_code = "discovery_feedback_required"
                machine.move(
                    SafetyState.COMMIT_DEFERRED,
                    "campaign requires an earlier observed failed hypothesis",
                )
                machine.move(
                    SafetyState.VERIFIED,
                    "clone-only success retained as evidence, not promotion",
                )
            else:
                safety_receipt = canonical_sha256(
                    {
                        "attempt_id": attempt_id,
                        "proposal_sha256": proposal.sha256,
                        "preflight_receipt_sha256":
                            preflight["receipt_sha256"],
                        "policy": self.policy.as_dict(),
                        "findings": [],
                        "decision": "commit_authorized",
                    }
                )
                permit = self.connector._mint_permit(
                    actions=preflight["actions"],
                    preflight=preflight,
                    safety_receipt_sha256=safety_receipt,
                )
                authorized = True
                machine.move(
                    SafetyState.COMMIT_AUTHORIZED,
                    "safe goal preflight and campaign gate satisfied",
                )
                machine.move(
                    SafetyState.COMMITTING,
                    "single-use in-memory permit consumed",
                )
                try:
                    commit = self.connector._commit_authorized(
                        permit,
                        attempt_id=attempt_id + "-commit",
                    )
                except ConnectorViolation as error:
                    machine.move(
                        SafetyState.REJECTED,
                        f"commit_interlock_failure:{type(error).__name__}",
                    )
                    disposition = "commit_interlock_rejected"
                    public_code = "commit_interlock_rejected"
                else:
                    machine.move(
                        SafetyState.OBSERVED,
                        "committed connector facts recorded",
                    )
                    commit_findings = _safety_findings(
                        commit,
                        self.policy,
                        require_complete=True,
                    )
                    if (
                        commit_findings
                        or int(commit.get("levels_completed", 0)) < 1
                        or not _traces_match(preflight, commit)
                    ):
                        findings = tuple(
                            dict.fromkeys((*findings, *commit_findings))
                        )
                        machine.move(
                            SafetyState.REJECTED,
                            "committed trace failed deterministic verification",
                        )
                        disposition = "commit_verification_rejected"
                        public_code = "commit_verification_rejected"
                    else:
                        machine.move(
                            SafetyState.VERIFIED,
                            "commit matched preflight and sparse goal",
                        )
                        disposition = "committed_success"
                        public_code = "committed_verified"

        if machine.state not in {
            SafetyState.VERIFIED,
            SafetyState.REJECTED,
        }:
            raise RuntimeError(
                f"safety FSA stopped in nonterminal state {machine.state}"
            )
        machine.move(SafetyState.SEALED, "immutable host receipt emitted")
        result: dict[str, object] = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "generation": generation,
            "sequence": sequence,
            "proposal": proposal.as_dict(),
            "proposal_sha256": proposal.sha256,
            "policy": self.policy.as_dict(),
            "transitions": machine.transitions,
            "preflight": preflight,
            "commit": commit,
            "safety_findings": list(findings),
            "authorized_for_commit": authorized,
            "disposition": disposition,
            "observed_failure_evidence": list(
                _observed_failure_evidence(
                    preflight,
                    proposal_kind=proposal.kind,
                    disposition=disposition,
                )
            ),
            "fsa_public_code": public_code,
            "fsa_state": machine.state.value,
        }
        result["receipt_sha256"] = canonical_sha256(result)
        return result

    def run_bundle(
        self,
        bundle: ProposalBundle,
        *,
        commit_enabled: bool,
    ) -> dict[str, object]:
        attempts: list[dict[str, object]] = []
        success: dict[str, object] | None = None
        for sequence, proposal in enumerate(bundle.scenarios, 1):
            attempt = self.run_scenario(
                proposal,
                generation=bundle.generation,
                sequence=sequence,
                commit_enabled=commit_enabled,
            )
            attempts.append(attempt)
            if attempt["disposition"] == "committed_success":
                success = attempt
                break
        result: dict[str, object] = {
            "schema_version": 1,
            "kind": "roboarm_safety_fsa_generation",
            "generation": bundle.generation,
            "proposal_bundle_sha256": bundle.sha256,
            "commit_enabled_at_generation_start": commit_enabled,
            "attempts": attempts,
            "successful_attempt_id": (
                success.get("attempt_id") if success is not None else None
            ),
        }
        result["receipt_sha256"] = canonical_sha256(result)
        return result


def first_success(
    attempts: Iterable[Mapping[str, object]],
) -> Mapping[str, object] | None:
    return next(
        (
            attempt
            for attempt in attempts
            if attempt.get("disposition") == "committed_success"
        ),
        None,
    )


__all__ = [
    "ALLOWED_TRANSITIONS",
    "SafetyAutomaton",
    "SafetyPolicy",
    "SafetyState",
    "first_success",
    "public_attempt_projection",
]
