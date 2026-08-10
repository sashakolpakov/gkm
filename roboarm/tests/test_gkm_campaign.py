from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from roboarm_game.canonical import CANONICAL_PICK_PLACE_ACTIONS
from roboarm_game.gkm.replay import run_proposal_source
from roboarm_game.gkm.report import write_campaign_report
from roboarm_game.gkm.runner import (
    CampaignConfig,
    _representative_failed_attempts,
    run_campaign,
)
from roboarm_game.gkm.viewer_export import export_campaign_viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAFE_PICK_PLACE_ACTIONS = (
    *CANONICAL_PICK_PLACE_ACTIONS[:56],
    *CANONICAL_PICK_PLACE_ACTIONS[58:],
)


def _fake_regression_proposer(
    workspace: Path,
    _prompt: str,
    transcript: Path,
    stderr_path: Path,
    _config: CampaignConfig,
) -> int:
    """Injected harness test only; explicitly not discovery evidence."""

    evidence = json.loads(
        (workspace / "evidence.json").read_text(encoding="utf-8")
    )
    generation = int(evidence["generation"])
    if generation == 1:
        (workspace / "legs.py").write_text(
            """\
from scenario_contract import scenario


def test_empty_close(evidence):
    return [
        scenario(
            "empty-close",
            "experiment",
            "closing at the initial pose may enclose the object",
            "the frame should show attachment or an empty close",
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
    return test_empty_close(evidence)
""",
            encoding="utf-8",
        )
    else:
        proposed_actions = (
            CANONICAL_PICK_PLACE_ACTIONS
            if generation == 2
            else SAFE_PICK_PLACE_ACTIONS
        )
        actions = ", ".join(
            str(value) for value in proposed_actions
        )
        (workspace / "legs.py").write_text(
            f"""\
from scenario_contract import scenario


def regression_candidate(evidence):
    return [
        scenario(
            "regression-candidate-{generation}",
            "candidate",
            "the injected test fixture should complete the mechanics round",
            "sparse levels_completed should advance to one",
            [{actions}],
        )
    ]
""",
            encoding="utf-8",
        )
        (workspace / "players.py").write_text(
            """\
from legs import *


def propose_level_1(evidence):
    return regression_candidate(evidence)
""",
            encoding="utf-8",
        )

    proposal_run = run_proposal_source(workspace)
    assert proposal_run.returncode == 0
    event = {
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "python3 gkm_propose.py",
        },
    }
    transcript.write_text(json.dumps(event) + "\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return 0


def test_representative_failures_prefer_distinct_operational_signatures():
    def attempt(
        attempt_id: str,
        evidence: list[str],
        rejection_reason: str = "",
    ) -> dict[str, object]:
        return {
            "attempt_id": attempt_id,
            "observed_failure_evidence": evidence,
            "preflight": {
                "steps": [
                    {
                        "telemetry": {
                            "motion": {
                                "rejected": bool(rejection_reason),
                                "reason": rejection_reason,
                            }
                        }
                    }
                ]
            },
        }

    selected = _representative_failed_attempts(
        [
            attempt(
                "early-barrier-and-empty",
                ["collision_rejection", "empty_grasp"],
                "gripper_barrier_collision",
            ),
            attempt(
                "redundant-early-barrier",
                ["collision_rejection", "empty_grasp"],
                "gripper_barrier_collision",
            ),
            attempt("empty-only", ["empty_grasp"]),
            attempt(
                "held-object-collision",
                ["collision_rejection"],
                "held_object_barrier_collision",
            ),
        ]
    )

    assert [value["attempt_id"] for value in selected] == [
        "early-barrier-and-empty",
        "empty-only",
        "held-object-collision",
    ]


def test_injected_campaign_exercises_feedback_fsa_replay_and_promotion():
    artifact_root = (
        PROJECT_ROOT
        / "artifacts"
        / "gkm-tests"
        / f"campaign-{uuid.uuid4().hex}"
    )
    result = run_campaign(
        CampaignConfig(
            artifact_root=artifact_root,
            campaign_id="injected-regression",
            proposer_timeout_seconds=60,
            max_generations=3,
        ),
        proposer=_fake_regression_proposer,
    )

    assert result.promoted
    assert result.genuine_failed_attempt
    assert result.revised_after_failure
    assert result.source_verified
    assert result.path_replayed
    assert result.proposer_generations == 3
    assert result.proposed_scenarios == 3
    assert result.fsa_rejections == 1
    assert result.exact_actions == len(SAFE_PICK_PLACE_ACTIONS)
    assert result.committed_actions == len(SAFE_PICK_PLACE_ACTIONS)
    assert result.clone_actions == (
        1
        + len(CANONICAL_PICK_PLACE_ACTIONS)
        + len(SAFE_PICK_PLACE_ACTIONS)
    )
    assert result.literal_action_cost >= len(
        SAFE_PICK_PLACE_ACTIONS
    )

    root = Path(result.root)
    payload = json.loads(
        (
            root
            / "evidence"
            / "proposer_payload_manifest_001.json"
        ).read_text()
    )
    visible_names = {entry["path"] for entry in payload["files"]}
    assert visible_names == {
        "README.md",
        "ROUND.md",
        "evidence.json",
        "gkm_propose.py",
        "interface.py",
        "legs.py",
        "perception.py",
        "players.py",
        "protocol.py",
        "scenario_contract.py",
        "solve.py",
        "solver_index.md",
    }
    assert "arena.py" not in visible_names
    assert ".arena.json" not in visible_names
    assert "canonical.py" not in visible_names
    assert "environment.py" not in visible_names
    assert payload["authority"]["can_actuate_connector"] is False
    assert payload["authority"]["can_write_observed_facts"] is False
    assert (
        "connector object, socket, token, and live environment handle"
        in payload["explicitly_excluded"]
    )
    assert payload["prompt"]["bytes"] > 0
    assert len(payload["prompt"]["sha256"]) == 64

    observed = json.loads(
        (root / "evidence" / "observed_attempt_ledger.json").read_text()
    )
    first, second, third = observed["attempts"]
    assert first["proposal"]["kind"] == "experiment"
    assert first["commit"] is None
    assert first["preflight"]["levels_completed"] == 0
    assert first["observed_failure_evidence"] == ["empty_grasp"]
    assert second["proposal"]["kind"] == "candidate"
    assert second["disposition"] == "candidate_rejected_by_fsa"
    assert second["authorized_for_commit"] is False
    assert second["commit"] is None
    assert third["proposal"]["kind"] == "candidate"
    assert third["authorized_for_commit"] is True
    assert third["commit"]["levels_completed"] == 1
    public = json.loads(
        (root / "evidence" / "public_feedback_ledger.json").read_text()
    )
    assert "visual_state" not in public["attempts"][0]["preflight"]
    assert "safety_findings" not in public["attempts"][0]
    assert public["attempts"][0]["observed_failure_evidence"] == [
        "empty_grasp"
    ]

    promotion = json.loads(
        (root / "promotions" / "level_01" / "promotion.json").read_text()
    )
    assert promotion["replay_validated"] is True
    assert promotion["genuine_failed_attempt_observed"] is True
    assert len(promotion["qualifying_failure_attempts"]) == 2
    assert promotion["qualifying_failure_attempts"][0][
        "observed_failure_evidence"
    ] == ["empty_grasp"]
    assert promotion["revised_after_failure"] is True
    assert promotion["proposer_had_actuation_authority"] is False
    assert promotion["discovery_fsa_receipt_sha256"]
    assert promotion["verification_fsa_receipt_sha256"]
    failed = json.loads((root / "browser" / "failed_attempt.json").read_text())
    successful = json.loads(
        (root / "browser" / "successful_attempt.json").read_text()
    )
    successful_commit = json.loads(
        (root / "browser" / "successful_commit.json").read_text()
    )
    assert failed["initial_visual_state"]["turn"] == 0
    assert failed["observed_failure_evidence"] == ["empty_grasp"]
    assert successful_commit["replay_stage"] == "discovery_commit"
    assert successful["initial_visual_state"]["turn"] == 0

    viewer = (
        PROJECT_ROOT
        / "artifacts"
        / "gkm-tests"
        / f"viewer-{uuid.uuid4().hex}"
    )
    manifest = export_campaign_viewer(root, viewer)
    assert manifest["export_kind"] == "replay-validated-gkm-evidence"
    assert manifest["failure_replays"] == 2
    assert manifest["success_replays"] == 2
    assert len(manifest["attempts"]) == 4
    assert manifest["lineage_profile"] == "lineage_profile.json"
    assert len(manifest["lineage_profile_receipt_sha256"]) == 64
    exported_lineage = json.loads(
        (viewer / "lineage_profile.json").read_text()
    )
    assert exported_lineage["profile_kind"] == (
        "campaign-construction-lineage"
    )
    assert exported_lineage["interpretation"][
        "solved_level_sawtooth_claim"
    ] is False
    assert len(exported_lineage["generations"]) == 3
    exported_failure = json.loads(
        (viewer / "failed_attempt.json").read_text()
    )
    exported_collision = json.loads(
        (viewer / "failed_attempt_002.json").read_text()
    )
    exported_commit = json.loads(
        (viewer / "successful_commit.json").read_text()
    )
    exported_success = json.loads(
        (viewer / "successful_attempt.json").read_text()
    )
    assert exported_failure["actions"] == [6]
    assert exported_failure["observed_failure_evidence"] == ["empty_grasp"]
    assert exported_failure["replay_receipt_sha256"]
    assert "collision_rejection" in exported_collision[
        "observed_failure_evidence"
    ]
    assert exported_commit["replay_stage"] == "discovery_commit"
    assert exported_commit["steps"][-1]["levels_completed"] == 1
    assert exported_success["actions"] == list(SAFE_PICK_PLACE_ACTIONS)
    assert exported_success["steps"][-1]["levels_completed"] == 1

    report = write_campaign_report(root)
    assert report["scientific_disposition"] == "replay_gated_promotion"
    assert report["canonical_fixture_counted"] is False
    assert report["browser_counted_as_solver"] is False
    assert report["authority"]["proposer_actuation"] is False
    assert report["interaction"]["proposer_attempts"] == 3
    assert report["interaction"]["failed_preflight_attempts"] == 1
    assert report["interaction"]["committed_successes"] == 1
    assert (
        report["interaction"]["verification_actions"]
        == len(SAFE_PICK_PLACE_ACTIONS)
    )
    assert (
        report["interaction"]["exact_replay_actions"]
        == len(SAFE_PICK_PLACE_ACTIONS)
    )
    assert (
        report["interaction"]["committed_actions"]
        == len(SAFE_PICK_PLACE_ACTIONS)
    )
    assert report["interaction"]["fsa_rejections"] == 1
    assert report["proposer_usage"]["commands"] == 3
    assert report["wall_time"]["campaign_total_seconds"] >= 0
    assert (
        report["wall_time"]["proposer_generation_001_seconds"] >= 0
    )
    assert (root / "reports" / "campaign_report.json").is_file()
    assert report["program_lineage"]["interpretation"][
        "construction_profile_only"
    ] is True
    assert (root / "reports" / "lineage_profile.json").is_file()
    lineage_markdown = (
        root / "reports" / "lineage_profile.md"
    ).read_text(encoding="utf-8")
    assert "campaign construction profile" in lineage_markdown
    assert "not a solved-level sawtooth" in lineage_markdown
    report_markdown = (
        root / "reports" / "campaign_report.md"
    ).read_text(encoding="utf-8")
    assert "canonical mechanics fixture" in report_markdown
    assert "browser playback" in report_markdown
    assert "no connector, socket, token" in report_markdown

    with pytest.raises(FileExistsError):
        export_campaign_viewer(root, viewer)
