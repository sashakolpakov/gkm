"""Report one safety-gated Godel-Kolmogorov machine campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .lineage import campaign_lineage_profile, lineage_markdown
from .replay import write_json
from .runner import PROJECT_ROOT


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _transcript_metrics(path: Path) -> dict[str, object]:
    metrics: dict[str, object] = {
        "events": 0,
        "turns_completed": 0,
        "model_turns": 0,
        "tool_results": 0,
        "commands": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "usage_reported": False,
    }
    if not path.is_file() or path.is_symlink():
        return metrics
    for line in path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        metrics["events"] = int(metrics["events"]) + 1
        event_type = event.get("type")
        item = event.get("item")
        if event_type == "turn.completed":
            metrics["turns_completed"] = (
                int(metrics["turns_completed"]) + 1
            )
        if event_type == "model_turn" or (
            event_type == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "agent_message"
        ):
            metrics["model_turns"] = int(metrics["model_turns"]) + 1
        if event_type == "tool_result":
            metrics["tool_results"] = int(metrics["tool_results"]) + 1
        if (
            event_type == "item.completed"
            and isinstance(item, dict)
            and item.get("type") == "command_execution"
        ):
            metrics["commands"] = int(metrics["commands"]) + 1
        usage = event.get("usage")
        if event_type == "turn.completed" and isinstance(usage, dict):
            for field in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
            ):
                value = usage.get(field)
                if isinstance(value, int) and value >= 0:
                    metrics[field] = int(metrics[field]) + value
            metrics["usage_reported"] = True
    return metrics


def _all_transcript_metrics(evidence: Path) -> dict[str, object]:
    total = _transcript_metrics(Path("/nonexistent"))
    files = sorted(evidence.glob("proposer_generation_*.jsonl"))
    for path in files:
        current = _transcript_metrics(path)
        for field in (
            "events",
            "turns_completed",
            "model_turns",
            "tool_results",
            "commands",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        ):
            total[field] = int(total[field]) + int(current[field])
        total["usage_reported"] = bool(
            total["usage_reported"] or current["usage_reported"]
        )
    total["generations"] = len(files)
    return total


def _attempt_metrics(ledger: dict[str, Any]) -> dict[str, int]:
    result = {
        "observed_attempts": 0,
        "failed_preflight_attempts": 0,
        "fsa_rejections": 0,
        "commit_deferred": 0,
        "committed_successes": 0,
    }
    attempts = ledger.get("attempts")
    if not isinstance(attempts, list):
        return result
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        result["observed_attempts"] += 1
        preflight = attempt.get("preflight")
        if (
            isinstance(preflight, dict)
            and isinstance(preflight.get("actions"), list)
            and preflight["actions"]
            and int(preflight.get("levels_completed", 0)) < 1
        ):
            result["failed_preflight_attempts"] += 1
        disposition = attempt.get("disposition")
        if disposition in {
            "candidate_rejected_by_fsa",
            "commit_interlock_rejected",
            "commit_verification_rejected",
        }:
            result["fsa_rejections"] += 1
        if disposition == "candidate_commit_deferred":
            result["commit_deferred"] += 1
        if disposition == "committed_success":
            result["committed_successes"] += 1
    return result


def campaign_report(
    campaign_root: Path,
) -> tuple[dict[str, object], str]:
    project_root = PROJECT_ROOT.resolve(strict=True)
    root = campaign_root.resolve(strict=True)
    if not root.is_relative_to(project_root / "artifacts"):
        raise ValueError(
            "campaign report input must stay below roboarm/artifacts"
        )

    evidence = root / "evidence"
    result = _read_object(root / "campaign_result.json")
    config = _read_object(evidence / "campaign_config.json")
    exploration = _read_object(
        evidence / "exploration_connector.json"
    )
    accounting = _read_object(evidence / "source_accounting.json")
    timing = _read_object(evidence / "campaign_timing.json")
    verification = _read_object(
        evidence / "verification_connector.json"
    )
    exact_replay = _read_object(evidence / "exact_path_replay.json")
    observed = _read_object(
        evidence / "observed_attempt_ledger.json"
    )
    transcript = _all_transcript_metrics(evidence)
    attempts = _attempt_metrics(observed)
    admissions = [
        _read_object(path)
        for path in sorted(
            evidence.glob("generation_admission_*.json")
        )
    ]
    containments = [
        _read_object(path)
        for path in sorted(
            evidence.glob(
                "proposer_generation_*.containment.json"
            )
        )
    ]
    lineage = campaign_lineage_profile(root)

    promoted = result.get("promoted") is True
    scientific_disposition = (
        "replay_gated_promotion"
        if promoted
        else "unpromoted_generation"
    )
    report: dict[str, object] = {
        "schema_version": 2,
        "campaign_id":
            result.get("campaign_id") or config.get("campaign_id"),
        "scientific_disposition": scientific_disposition,
        "canonical_fixture_counted": False,
        "browser_counted_as_solver": False,
        "authority": {
            "proposer_actuation": False,
            "proposer_connector_handle": False,
            "proposer_observation_verdicts": False,
            "host_safety_fsa": True,
            "single_use_commit_permit": True,
        },
        "config": config,
        "result": result,
        "admissions": admissions,
        "containments": containments,
        "wall_time": timing or None,
        "proposer_usage": transcript,
        "interaction": {
            "proposer_attempts": result.get(
                "proposer_generations",
                transcript.get("generations", 0),
            ),
            "proposed_scenarios": result.get(
                "proposed_scenarios",
                0,
            ),
            "committed_actions": exploration.get(
                "committed_actions",
                0,
            ),
            "clone_actions": exploration.get(
                "preflight_actions",
                0,
            ),
            "verification_actions": verification.get(
                "committed_actions",
                0,
            ),
            "verification_clone_actions": verification.get(
                "preflight_actions",
                0,
            ),
            "exact_replay_actions": (
                len(exact_replay.get("steps", []))
                if isinstance(exact_replay.get("steps"), list)
                else 0
            ),
            **attempts,
        },
        "source_accounting": accounting or None,
        "program_lineage": lineage,
        "evidence_paths": {
            "prompts": "evidence/proposer_prompt_*.md",
            "payload_manifests":
                "evidence/proposer_payload_manifest_*.json",
            "transcripts":
                "evidence/proposer_generation_*.jsonl",
            "proposals":
                "evidence/proposed_scenarios_*.json",
            "safety_fsa":
                "evidence/safety_fsa_generation_*.json",
            "observed_ledger":
                "evidence/observed_attempt_ledger.json",
            "public_feedback":
                "evidence/public_feedback_ledger.json",
            "exploration": "evidence/exploration_connector.json",
            "verification":
                "evidence/verification_safety_fsa.json",
            "exact_replay": "evidence/exact_path_replay.json",
            "timing": "evidence/campaign_timing.json",
            "promotion": "promotions/level_01/promotion.json",
            "lineage_profile": "reports/lineage_profile.json",
        },
    }

    result_reason = result.get("failure_reason") or "none"
    markdown = f"""\
# RoboArm Godel-Kolmogorov machine campaign report

- Campaign: `{report["campaign_id"]}`
- Scientific disposition: `{scientific_disposition}`
- Provider/model: `{config.get("provider", "unknown")}` / `{config.get("model", "unknown")}`
- Promoted: `{promoted}`
- Clean generations: `{result.get("clean_generation", False)}`
- Genuine failure then source revision: `{result.get("genuine_failed_attempt", False)}`
- Failure reason: `{result_reason}`
- Campaign wall time: `{timing.get("campaign_total_seconds", "unavailable")} s`

## Authority boundary

The headless-Codex proposer had no connector, socket, token, live environment
handle, or action method. It authored program structure and declarative
scenarios only. A host-owned deterministic FSA validated each closed schema,
ran isolated digital-twin preflight, rejected unsafe or non-goal candidates,
and minted a single-use in-memory permit only for an admitted commit. The model
did not write observed facts, sparse reward, `passed`, or a safety verdict.

## Interaction accounting

- Proposer generations: `{result.get("proposer_generations", 0)}`
- Proposed scenarios: `{result.get("proposed_scenarios", 0)}`
- Isolated preflight actions: `{exploration.get("preflight_actions", 0)}`
- FSA-authorized committed actions: `{exploration.get("committed_actions", 0)}`
- Failed preflight attempts: `{attempts["failed_preflight_attempts"]}`
- FSA rejections: `{attempts["fsa_rejections"]}`
- Deferred clone-only successes: `{attempts["commit_deferred"]}`
- Verified committed successes: `{attempts["committed_successes"]}`
- Fresh-source verification actions: `{verification.get("committed_actions", 0)}`
- Independent exact-replay actions: `{len(exact_replay.get("steps", [])) if isinstance(exact_replay.get("steps"), list) else 0}`
- Exact promoted replay actions: `{result.get("exact_actions", 0)}`

## Proposer compute

- Headless-Codex generations with transcripts: `{transcript.get("generations", 0)}`
- Completed model turns: `{transcript.get("model_turns", 0)}`
- Completed offline tool commands: `{transcript.get("commands", 0)}`
- Input tokens: `{transcript.get("input_tokens", 0)}`
- Cached input tokens: `{transcript.get("cached_input_tokens", 0)}`
- Output tokens: `{transcript.get("output_tokens", 0)}`
- Reasoning output tokens: `{transcript.get("reasoning_output_tokens", 0)}`
- Token usage reported by provider: `{transcript.get("usage_reported", False)}`

## Program accounting

- Marginal retained description: `{result.get("marginal_description", 0)}`
- Literal action-container charge: `{result.get("literal_action_cost", 0)}`
- Free energy: `{result.get("free_energy")}`
- Source digest: `{accounting.get("source_tree_sha256", "unavailable")}`

## Retained-leg reuse and construction profile

- Historical net-growth trace: `{[row["historical_net_growth"] for row in lineage["generations"]]}`
- Conditional-AST novelty trace: `{[row["conditional_ast_zlib_bytes"] for row in lineage["generations"]]}` bytes
- Net-growth direction changes: `{lineage["interpretation"]["historical_net_growth_direction_changes"]}`
- Generations with transitively invoked unchanged legs: `{lineage["interpretation"]["transitive_reuse_generations"]}`
- Generations with a direct unchanged-leg call: `{lineage["interpretation"]["direct_reuse_generations"]}`
- Sharp-drop/direct-call coupled witnesses: `{lineage["interpretation"]["sharp_direct_coupled_witnesses"]}`

This is a campaign construction profile, not a solved-level sawtooth: `rb01`
has one promoted round. The exact per-generation source and call-graph evidence
is in `reports/lineage_profile.json` and `reports/lineage_profile.md`.

## Evidence interpretation

The canonical mechanics fixture and browser playback are excluded from the
campaign result. Browser playback is downstream illustration only. A browser
export is admissible after the host independently replays every recorded
action and matches its frame, sparse reward, terminal state, FSA receipt, and
promotion receipt.
"""
    return report, markdown


def write_campaign_report(
    campaign_root: Path,
) -> dict[str, object]:
    report, markdown = campaign_report(campaign_root)
    destination = campaign_root.resolve(strict=True) / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "campaign_report.json", report)
    lineage = report["program_lineage"]
    if not isinstance(lineage, dict):
        raise ValueError("campaign report lineage profile is invalid")
    write_json(destination / "lineage_profile.json", lineage)
    lineage_path = destination / "lineage_profile.md"
    if lineage_path.is_symlink():
        raise ValueError("lineage report destination is linked")
    lineage_path.write_text(lineage_markdown(lineage), encoding="utf-8")
    path = destination / "campaign_report.md"
    if path.is_symlink():
        raise ValueError("campaign report destination is linked")
    path.write_text(markdown, encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one evidence-linked safety-gated "
            "Godel-Kolmogorov machine report"
        )
    )
    parser.add_argument("campaign_root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = write_campaign_report(arguments.campaign_root)
    print(json.dumps(report, sort_keys=True))
    return 0


__all__ = ["campaign_report", "main", "write_campaign_report"]
