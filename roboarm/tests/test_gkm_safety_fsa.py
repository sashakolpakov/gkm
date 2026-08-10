from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboarm_game.gkm.arena import RoboArmConnector
from roboarm_game.gkm.safety_fsa import (
    SafetyAutomaton,
    SafetyState,
    public_attempt_projection,
)
from roboarm_game.gkm.scenario import (
    ScenarioContractError,
    validate_proposal_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _value(
    *,
    actions: list[int],
    kind: str = "candidate",
    extras: dict[str, object] | None = None,
) -> dict[str, object]:
    scenario: dict[str, object] = {
        "scenario_id": "proposal-1",
        "kind": kind,
        "hypothesis": "this sequence tests clearance",
        "expected_observation": "the public rejection HUD may activate",
        "actions": actions,
    }
    scenario.update(extras or {})
    return {
        "schema_version": 1,
        "kind": "roboarm_scenario_proposals",
        "game_id": "rb01-v1",
        "round_id": "rb01-round-1",
        "generation": 1,
        "scenarios": [scenario],
    }


@pytest.mark.parametrize(
    "field",
    [
        "passed",
        "observedStatus",
        "levels_completed",
        "safety",
        "authorization",
        "reward",
    ],
)
def test_proposer_cannot_supply_host_owned_outcome_fields(field: str):
    with pytest.raises(
        ScenarioContractError,
        match="host-owned outcome fields",
    ):
        validate_proposal_bundle(
            _value(actions=[6], extras={field: True}),
            expected_generation=1,
            max_scenarios=8,
            max_actions=160,
        )


def test_proposal_schema_rejects_illegal_actions_before_execution():
    with pytest.raises(ScenarioContractError, match="invalid public action"):
        validate_proposal_bundle(
            _value(actions=[7]),
            expected_generation=1,
            max_scenarios=8,
            max_actions=160,
        )


def test_collision_attempt_is_real_preflight_evidence_but_never_commits():
    fixture = json.loads(
        (
            PROJECT_ROOT
            / "web"
            / "public"
            / "mechanics-test"
            / "collision_attempt.json"
        ).read_text(encoding="utf-8")
    )
    bundle = validate_proposal_bundle(
        _value(actions=fixture["actions"]),
        expected_generation=1,
        max_scenarios=8,
        max_actions=160,
    )
    connector = RoboArmConnector(
        max_committed_actions=200,
        max_preflight_actions=200,
    )
    result = SafetyAutomaton(connector).run_bundle(
        bundle,
        commit_enabled=True,
    )
    attempt = result["attempts"][0]

    assert attempt["disposition"] == "candidate_rejected_by_fsa"
    assert "collision_interlock" in attempt["safety_findings"]
    assert attempt["preflight"]["actions"] == fixture["actions"]
    assert attempt["preflight"]["steps"][-1]["visual_state"]["robot"][
        "rejected"
    ] is True
    assert attempt["commit"] is None
    assert connector.committed_actions == 0
    states = [transition["to"] for transition in attempt["transitions"]]
    assert states == [
        SafetyState.RECEIVED.value,
        SafetyState.CONTRACT_VALIDATED.value,
        SafetyState.PREFLIGHTING.value,
        SafetyState.PREFLIGHT_OBSERVED.value,
        SafetyState.REJECTED.value,
        SafetyState.SEALED.value,
    ]


def test_public_projection_strips_private_visual_state_and_findings():
    bundle = validate_proposal_bundle(
        _value(actions=[6], kind="experiment"),
        expected_generation=1,
        max_scenarios=8,
        max_actions=160,
    )
    attempt = SafetyAutomaton(RoboArmConnector()).run_bundle(
        bundle,
        commit_enabled=False,
    )["attempts"][0]
    projection = public_attempt_projection(attempt)

    assert "safety_findings" not in projection
    assert "transitions" not in projection
    assert "visual_state" not in projection["preflight"]
    assert all(
        "visual_state" not in step
        for step in projection["preflight"]["steps"]
    )
    assert projection["preflight"]["steps"][0]["frame_b64"]
    assert projection["public_receipt_sha256"]
