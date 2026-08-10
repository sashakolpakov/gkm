"""Strict untrusted scenario-proposal schema for RoboArm GKM.

The coding proposer may describe a hypothesis and a bounded sequence of public
actions.  It may not supply observations, rewards, authorization, safety
verdicts, or pass/fail fields.  Those are produced only by the trusted host.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..interface import ACTIONS

SCHEMA_VERSION = 1
PROPOSAL_KIND = "roboarm_scenario_proposals"
ROUND_ID = "rb01-round-1"
SCENARIO_KINDS = frozenset({"experiment", "candidate"})
SCENARIO_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")

BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "game_id",
        "round_id",
        "generation",
        "scenarios",
    }
)
SCENARIO_FIELDS = frozenset(
    {
        "scenario_id",
        "kind",
        "hypothesis",
        "expected_observation",
        "actions",
    }
)
FORBIDDEN_OUTCOME_FIELDS = frozenset(
    {
        "admitted",
        "authorization",
        "authorized",
        "events",
        "frame",
        "frame_b64",
        "levels_completed",
        "observed",
        "observedStatus",
        "passed",
        "reward",
        "safety",
        "success",
        "telemetry",
        "telemetry_sha256",
        "terminal",
        "verdict",
    }
)


class ScenarioContractError(ValueError):
    """An untrusted proposal failed the closed scenario contract."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScenarioContractError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: str | bytes) -> Any:
    """Parse JSON while rejecting duplicate keys and non-finite numbers."""

    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ScenarioContractError(f"non-finite JSON number: {value}")
            ),
        )
    except ScenarioContractError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ScenarioContractError("malformed scenario JSON") from error


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _bounded_text(value: object, *, field: str, limit: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > limit
    ):
        raise ScenarioContractError(f"{field} must be nonempty bounded text")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ScenarioProposal:
    """One validated but still untrusted experiment or candidate."""

    scenario_id: str
    kind: str
    hypothesis: str
    expected_observation: str
    actions: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "kind": self.kind,
            "hypothesis": self.hypothesis,
            "expected_observation": self.expected_observation,
            "actions": list(self.actions),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True, slots=True)
class ProposalBundle:
    """A generation-bound collection of validated untrusted proposals."""

    generation: int
    scenarios: tuple[ScenarioProposal, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PROPOSAL_KIND,
            "game_id": "rb01-v1",
            "round_id": ROUND_ID,
            "generation": self.generation,
            "scenarios": [
                scenario.as_dict() for scenario in self.scenarios
            ],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


def _scenario(
    value: object,
    *,
    max_actions: int,
) -> ScenarioProposal:
    if not isinstance(value, Mapping):
        raise ScenarioContractError("scenario must be an object")
    keys = frozenset(value)
    if keys != SCENARIO_FIELDS:
        forbidden = sorted(keys & FORBIDDEN_OUTCOME_FIELDS)
        if forbidden:
            raise ScenarioContractError(
                "proposer scenarios cannot contain host-owned outcome fields: "
                + ", ".join(forbidden)
            )
        raise ScenarioContractError(
            "scenario fields differ from the closed contract: "
            f"missing={sorted(SCENARIO_FIELDS - keys)!r}, "
            f"extra={sorted(keys - SCENARIO_FIELDS)!r}"
        )
    scenario_id = value["scenario_id"]
    if (
        not isinstance(scenario_id, str)
        or SCENARIO_ID_RE.fullmatch(scenario_id) is None
    ):
        raise ScenarioContractError("scenario_id is invalid")
    kind = value["kind"]
    if kind not in SCENARIO_KINDS:
        raise ScenarioContractError(
            f"scenario kind must be one of {sorted(SCENARIO_KINDS)}"
        )
    hypothesis = _bounded_text(
        value["hypothesis"],
        field="hypothesis",
        limit=2_000,
    )
    expected = _bounded_text(
        value["expected_observation"],
        field="expected_observation",
        limit=2_000,
    )
    raw_actions = value["actions"]
    if (
        not isinstance(raw_actions, list)
        or not raw_actions
        or len(raw_actions) > max_actions
    ):
        raise ScenarioContractError(
            f"actions must contain 1..{max_actions} public actions"
        )
    actions: list[int] = []
    for action in raw_actions:
        if (
            isinstance(action, bool)
            or not isinstance(action, int)
            or int(action) not in ACTIONS
        ):
            raise ScenarioContractError(
                f"invalid public action {action!r}; expected one of {ACTIONS}"
            )
        actions.append(int(action))
    return ScenarioProposal(
        scenario_id=scenario_id,
        kind=str(kind),
        hypothesis=hypothesis,
        expected_observation=expected,
        actions=tuple(actions),
    )


def validate_proposal_bundle(
    value: object,
    *,
    expected_generation: int,
    max_scenarios: int,
    max_actions: int,
) -> ProposalBundle:
    """Validate structure only; this does not authorize or execute actions."""

    if (
        isinstance(expected_generation, bool)
        or not isinstance(expected_generation, int)
        or expected_generation <= 0
    ):
        raise ValueError("expected_generation must be a positive integer")
    if max_scenarios <= 0 or max_actions <= 0:
        raise ValueError("scenario and action bounds must be positive")
    if not isinstance(value, Mapping) or frozenset(value) != BUNDLE_FIELDS:
        raise ScenarioContractError(
            "proposal bundle fields differ from the closed contract"
        )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != PROPOSAL_KIND
        or value["game_id"] != "rb01-v1"
        or value["round_id"] != ROUND_ID
        or value["generation"] != expected_generation
    ):
        raise ScenarioContractError(
            "proposal bundle identity does not match the active generation"
        )
    raw_scenarios = value["scenarios"]
    if (
        not isinstance(raw_scenarios, list)
        or not raw_scenarios
        or len(raw_scenarios) > max_scenarios
    ):
        raise ScenarioContractError(
            f"bundle must contain 1..{max_scenarios} scenarios"
        )
    scenarios = tuple(
        _scenario(item, max_actions=max_actions)
        for item in raw_scenarios
    )
    identifiers = [item.scenario_id for item in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise ScenarioContractError("scenario_id values must be unique")
    return ProposalBundle(
        generation=expected_generation,
        scenarios=scenarios,
    )


__all__ = [
    "FORBIDDEN_OUTCOME_FIELDS",
    "PROPOSAL_KIND",
    "ProposalBundle",
    "ROUND_ID",
    "SCENARIO_KINDS",
    "SCHEMA_VERSION",
    "ScenarioContractError",
    "ScenarioProposal",
    "canonical_json",
    "canonical_sha256",
    "strict_json_loads",
    "validate_proposal_bundle",
]
