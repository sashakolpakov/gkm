from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from roboarm_game.canonical import CANONICAL_PICK_PLACE_ACTIONS
from roboarm_game.gkm.arena import ConnectorViolation, RoboArmConnector
from roboarm_game.gkm.runner import proposer_prompt
from roboarm_game.gkm.replay import run_proposal_source
from roboarm_game.gkm.safety_fsa import SafetyAutomaton
from roboarm_game.gkm.scenario import (
    ProposalBundle,
    ScenarioProposal,
)
from roboarm_game.gkm.workspace import materialize_workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAFE_PICK_PLACE_ACTIONS = (
    *CANONICAL_PICK_PLACE_ACTIONS[:56],
    *CANONICAL_PICK_PLACE_ACTIONS[58:],
)


def _test_root(label: str) -> Path:
    root = (
        PROJECT_ROOT
        / "artifacts"
        / "gkm-tests"
        / f"{label}-{uuid.uuid4().hex}"
    )
    root.mkdir(parents=True)
    return root


def _evidence(connector: RoboArmConnector) -> dict[str, object]:
    value = {
        "schema_version": 2,
        "kind": "roboarm_host_sealed_public_evidence",
        "game_id": "rb01-v1",
        "round_id": "rb01-round-1",
        "seed": 0,
        "generation": 1,
        "initial_observation": connector.initial_observation(),
        "attempts": [],
        "host_feedback": [],
        "authority_boundary": {
            "connector_visible_to_proposer": False,
        },
    }
    value["receipt_sha256"] = "test-fixture"
    return value


def _bundle(
    kind: str,
    actions: tuple[int, ...],
    *,
    scenario_id: str = "test",
) -> ProposalBundle:
    return ProposalBundle(
        generation=1,
        scenarios=(
            ScenarioProposal(
                scenario_id=scenario_id,
                kind=kind,
                hypothesis="a falsifiable test hypothesis",
                expected_observation="the public frame should change",
                actions=actions,
            ),
        ),
    )


def test_proposer_workspace_has_no_connector_or_actuation_capability():
    root = _test_root("proposal-workspace")
    connector = RoboArmConnector(
        max_committed_actions=20,
        max_preflight_actions=20,
    )
    workspace = materialize_workspace(
        root / "workspace",
        write_root=root,
        public_evidence=_evidence(connector),
        generation=1,
    ).root

    names = {path.name for path in workspace.iterdir()}
    assert {
        "README.md",
        "ROUND.md",
        "evidence.json",
        "gkm_propose.py",
        "interface.py",
        "perception.py",
        "protocol.py",
        "scenario_contract.py",
        "legs.py",
        "players.py",
        "solve.py",
    } <= names
    assert "arena.py" not in names
    assert ".arena.json" not in names
    assert not {
        "canonical.py",
        "dynamics.py",
        "environment.py",
        "geometry.py",
        "operational.py",
        "oracle.py",
        "world_state.py",
    } & names
    all_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in workspace.iterdir()
        if path.is_file()
    )
    assert "ROBOARM_ARENA_CONFIG" not in all_text
    assert "socket path" not in all_text.lower()
    assert "There is deliberately no `Arena`" in (
        workspace / "solver_index.md"
    ).read_text(encoding="utf-8")

    prompt = proposer_prompt()
    assert "1/2 decrease/increase the selected coordinate" in prompt
    assert "Objects persist and collide" in prompt
    assert "You do not have an Arena" in prompt
    assert "model is never the oracle" in prompt
    assert "object coordinates" not in prompt.lower()


def test_experiment_is_observed_but_can_never_commit():
    connector = RoboArmConnector(
        max_committed_actions=20,
        max_preflight_actions=20,
    )
    result = SafetyAutomaton(connector).run_bundle(
        _bundle("experiment", (6,), scenario_id="empty-close"),
        commit_enabled=True,
    )
    attempt = result["attempts"][0]

    assert attempt["disposition"] == "probe_observed"
    assert attempt["authorized_for_commit"] is False
    assert attempt["commit"] is None
    assert attempt["observed_failure_evidence"] == ["empty_grasp"]
    assert connector.preflight_actions == 1
    assert connector.committed_actions == 0
    assert attempt["preflight"]["steps"][0]["visual_state"] is not None


def test_successful_clearance_probe_is_not_mislabeled_as_failure():
    connector = RoboArmConnector(
        max_committed_actions=20,
        max_preflight_actions=20,
    )
    result = SafetyAutomaton(connector).run_bundle(
        _bundle("experiment", (1,), scenario_id="clearance"),
        commit_enabled=False,
    )
    attempt = result["attempts"][0]

    assert attempt["disposition"] == "probe_observed"
    assert attempt["observed_failure_evidence"] == []


def test_clone_only_success_is_deferred_until_campaign_gate():
    connector = RoboArmConnector(
        max_committed_actions=100,
        max_preflight_actions=100,
    )
    result = SafetyAutomaton(connector).run_bundle(
        _bundle(
            "candidate",
            tuple(SAFE_PICK_PLACE_ACTIONS),
            scenario_id="candidate",
        ),
        commit_enabled=False,
    )
    attempt = result["attempts"][0]

    assert attempt["disposition"] == "candidate_commit_deferred"
    assert attempt["preflight"]["levels_completed"] == 1
    assert attempt["authorized_for_commit"] is False
    assert connector.committed_actions == 0


def test_fsa_permit_is_required_and_success_replays_stepwise():
    connector = RoboArmConnector(
        max_committed_actions=100,
        max_preflight_actions=100,
    )
    with pytest.raises(ConnectorViolation, match="safety permit"):
        connector._commit_authorized(
            object(),  # type: ignore[arg-type]
            attempt_id="forged",
        )

    result = SafetyAutomaton(connector).run_bundle(
        _bundle(
            "candidate",
            tuple(SAFE_PICK_PLACE_ACTIONS),
            scenario_id="candidate",
        ),
        commit_enabled=True,
    )
    attempt = result["attempts"][0]

    assert attempt["disposition"] == "committed_success"
    assert attempt["authorized_for_commit"] is True
    assert attempt["commit"]["levels_completed"] == 1
    assert (
        [step["frame_sha256"] for step in attempt["preflight"]["steps"]]
        == [step["frame_sha256"] for step in attempt["commit"]["steps"]]
    )
    assert (
        [
            step["telemetry_sha256"]
            for step in attempt["preflight"]["steps"]
        ]
        == [
            step["telemetry_sha256"]
            for step in attempt["commit"]["steps"]
        ]
    )
    assert (
        attempt["preflight"]["initial_telemetry_sha256"]
        == attempt["commit"]["initial_telemetry_sha256"]
    )
    assert all(
        step["before_telemetry_sha256"]
        for step in attempt["commit"]["steps"]
    )
    assert connector.committed_actions == len(
        SAFE_PICK_PLACE_ACTIONS
    )
    assert connector.evidence()["live_permits"] == 0


def test_workspace_evidence_is_regular_json_and_immutable_input():
    root = _test_root("evidence")
    connector = RoboArmConnector()
    workspace = materialize_workspace(
        root / "workspace",
        write_root=root,
        public_evidence=_evidence(connector),
        generation=1,
    ).root
    value = json.loads((workspace / "evidence.json").read_text())

    assert value["generation"] == 1
    assert value["initial_observation"]["frame_sha256"]


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_proposal_output_refuses_alias_to_sealed_evidence(alias_kind: str):
    root = _test_root(f"output-{alias_kind}")
    connector = RoboArmConnector()
    workspace = materialize_workspace(
        root / "workspace",
        write_root=root,
        public_evidence=_evidence(connector),
        generation=1,
    ).root
    (workspace / "legs.py").write_text(
        """\
from scenario_contract import scenario


def proposal(evidence):
    return [
        scenario(
            "safe-output-test",
            "experiment",
            "test the output boundary",
            "the host should reject the alias",
            [6],
        )
    ]
""",
        encoding="utf-8",
    )
    (workspace / "players.py").write_text(
        """\
from legs import *


def propose_level_1(evidence):
    return proposal(evidence)
""",
        encoding="utf-8",
    )
    evidence_path = workspace / "evidence.json"
    before = evidence_path.read_bytes()
    output = workspace / "scenario_proposals.json"
    if alias_kind == "symlink":
        output.symlink_to("evidence.json")
    else:
        os.link(evidence_path, output)

    result = run_proposal_source(workspace)

    assert result.returncode != 0
    assert result.result is None
    assert evidence_path.read_bytes() == before
