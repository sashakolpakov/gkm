from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import signal
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
import fcntl

import pytest

import codex_campaign_runner as R


def _item():
    binding = R.Status.validate_frontier_binding({
        "frontier_binding_schema": R.Status.FRONTIER_BINDING_SCHEMA,
        "game": "ar25",
        "reached": 0,
        "target_level": 1,
        "parent_action_count": 0,
        "parent_checkpoint_sha256": R.Status.ZERO_SHA256,
        "parent_source_tree_sha256": R.Status.ZERO_SHA256,
        "frontier_sha256": R.Status._sha256_json({
            "game": "ar25",
            "reached": 0,
            "parent_checkpoint_sha256": R.Status.ZERO_SHA256,
        }),
    })
    return {
        **binding,
        "game": "ar25",
        "target_level": 1,
        "effort": "medium",
        "minutes": 15,
        "retry_complexity_n": 0,
        "dispatch_mode": "fresh_frontier",
        "recommended_auxiliary_parallelism": 0,
        "cost_control_enabled": True,
        "max_campaign_runs": 60,
        "max_campaign_tokens": 32_000_000,
        "required_headroom_percent": 6,
        "warm_wip_available": False,
        "seed_mode": "zero_seed",
        "wip_mode": "exclude",
        "lineage_input_mode": "zero_seed+exclude",
        "argv": [
            "python3", "-u", "arc/crack_lab/gkm_legs.py",
            "--game=ar25", "--max-level=1",
            "--proposer=codex", "--model=gpt-5.6-sol",
            "--minutes=15", "--codex-effort=medium",
            "--codex-allocation-policy=drain",
            "--debrief-policy=never", "--transient-retries=0",
            "--codex-weekly-reserve=25", "--codex-weekly-headroom=6",
            "--codex-max-campaign-runs=60",
            "--codex-max-campaign-tokens=32000000",
            "--seed-mode=zero_seed", "--wip-mode=exclude",
            "--expected-parent-reached=0",
            "--expected-parent-action-count=0",
            (
                "--expected-parent-checkpoint-sha256="
                f"{R.Status.ZERO_SHA256}"
            ),
            (
                "--expected-parent-source-tree-sha256="
                f"{R.Status.ZERO_SHA256}"
            ),
            f"--expected-frontier-sha256={binding['frontier_sha256']}",
        ],
    }


def test_validate_item_rejects_arbitrary_commands():
    with pytest.raises(R.CampaignPlanError, match="non-GKM"):
        R.validate_item({"argv": ["sh", "-c", "anything"]})
    assert R.validate_item(_item())[0] == "python3"


def test_validate_item_requires_budget_arguments_to_match_plan():
    with pytest.raises(R.CampaignPlanError, match="reserve does not match"):
        R.validate_item(
            _item(),
            {"reserve_percent": 20, "cost_control_enabled": True},
        )
    item = _item()
    reserve_index = item["argv"].index("--codex-weekly-reserve=25")
    item["argv"][reserve_index] = "--codex-weekly-reserve=20"
    assert R.validate_item(
        item,
        {"reserve_percent": 20, "cost_control_enabled": True},
    )[0] == "python3"


def test_item_admission_requires_reset_epoch_and_headroom():
    plan = {"not_before_epoch": 100, "reserve_percent": 25}
    allowance = SimpleNamespace(remaining_percent=100)
    ok, reason = R.item_is_admissible(plan, _item(), now=99, allowance=allowance)
    assert not ok and "held until" in reason
    ok, reason = R.item_is_admissible(plan, _item(), now=101, allowance=allowance)
    assert ok and reason == "admissible"
    allowance = SimpleNamespace(remaining_percent=30)
    ok, reason = R.item_is_admissible(plan, _item(), now=101, allowance=allowance)
    assert not ok and "requires 6%" in reason


def test_item_admission_ignores_cost_controls_for_unlimited_pool():
    allowance = SimpleNamespace(
        remaining_percent=0, window_name="unlimited"
    )
    item = _item()
    item["cost_control_enabled"] = False
    item["max_campaign_runs"] = -1
    item["max_campaign_tokens"] = -1
    ok, reason = R.item_is_admissible(
        {"reserve_percent": 99}, item, now=101, allowance=allowance
    )
    assert ok
    assert "unlimited" in reason


def test_item_admission_never_disables_cost_controls_for_finite_pool():
    item = _item()
    item["cost_control_enabled"] = False
    item["max_campaign_runs"] = -1
    item["max_campaign_tokens"] = -1
    allowance = SimpleNamespace(
        remaining_percent=100, window_name="weekly"
    )
    ok, reason = R.item_is_admissible(
        {"reserve_percent": 20}, item, now=101, allowance=allowance
    )
    assert not ok
    assert "requires cost controls" in reason


def test_validate_item_rejects_unlimited_mode_with_local_cost_cutoff():
    item = _item()
    item["cost_control_enabled"] = False
    with pytest.raises(R.CampaignPlanError, match="retains a local"):
        R.validate_item(item)


def test_validate_item_rejects_unconsumed_or_tampered_frontier_binding():
    item = _item()
    item["argv"] = [
        argument
        for argument in item["argv"]
        if not argument.startswith("--expected-frontier-sha256=")
    ]
    with pytest.raises(
        R.CampaignPlanError, match="does not consume its exact-frontier"
    ):
        R.validate_item(item)

    item = _item()
    item["frontier_sha256"] = "f" * 64
    with pytest.raises(
        R.CampaignPlanError, match="invalid exact-frontier binding"
    ):
        R.validate_item(item)


def _infrastructure_recovery_item():
    item = copy.deepcopy(_item())
    item.update({
        "dispatch_mode": "recover_clean_infrastructure_wip",
        "policy_dispatch_mode": "fresh_frontier",
        "warm_wip_available": True,
        "warm_wip_phase": "infrastructure_failure_transport",
        "warm_wip_recovery_required": True,
        "expected_wip_attempt": "infrastructure_failure_transport_abc123",
        "wip_mode": "restore_clean_same_frontier",
        "lineage_input_mode": "zero_seed+restore_clean_same_frontier",
    })
    index = item["argv"].index("--wip-mode=exclude")
    item["argv"][index] = "--wip-mode=restore_clean_same_frontier"
    item["argv"].append(
        "--expected-wip-attempt=infrastructure_failure_transport_abc123"
    )
    return item


def test_validate_item_admits_only_sealed_infrastructure_recovery_override():
    item = _infrastructure_recovery_item()
    assert R.validate_item(item)[0] == "python3"

    item["warm_wip_phase"] = "not_reached"
    with pytest.raises(
        R.CampaignPlanError, match="sealed exact-frontier capsule"
    ):
        R.validate_item(item)

    item = _infrastructure_recovery_item()
    item["argv"].pop()
    with pytest.raises(
        R.CampaignPlanError, match="does not pin one"
    ):
        R.validate_item(item)


def test_runner_projects_policy_rejected_legacy_wip_to_clean_reset():
    policy = R.Status.retry_policy(2)
    item = {
        "warm_wip_available": False,
        "warm_wip_recovery_required": False,
        "warm_wip_validation": (
            "rejected:filesystem_boundary_policy_binding"
        ),
    }

    assert R._effective_retry_inputs(item, policy) == (
        "exclude", "filesystem_boundary_clean_reset"
    )


def test_live_policy_reopens_infrastructure_recovery_capsule(monkeypatch):
    item = _infrastructure_recovery_item()
    monkeypatch.setattr(
        R.Status,
        "campaign_report",
        lambda **kwargs: {
            "frontiers": [{
                **{
                    key: item[key]
                    for key in (
                        *R.Status.FRONTIER_BINDING_FIELDS,
                        "reached",
                        "parent_action_count",
                    )
                },
                "game": "ar25",
                "next_level": 1,
                "retry_complexity_n": 0,
                "warm_wip_available": True,
                "warm_wip_attempt": item["expected_wip_attempt"],
                "warm_wip_phase": item["warm_wip_phase"],
                "warm_wip_recovery_required": True,
            }]
        },
    )
    R.validate_live_policy_item(item)


def test_unattended_admission_reopens_canonical_filesystem_boundary(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "agent_solutions" / "ar25_legs"
    artifact.mkdir(parents=True)
    monkeypatch.setattr(R, "HERE", tmp_path)
    seen = []

    def boundary(path):
        seen.append(path)
        return "parent_path in legs.py:1"

    monkeypatch.setattr(R.Legs, "promoted_artifact_taint_reason", boundary)
    with pytest.raises(
        R.CampaignPlanError, match="canonical parent fails"
    ):
        R.validate_live_policy_item(_item())
    assert seen == [str(artifact)]


def test_live_policy_reset_does_not_consume_unused_wip_capsule(monkeypatch):
    item = _item()
    item.update({
        "warm_wip_available": True,
        "warm_wip_phase": "not_reached",
        "warm_wip_recovery_required": False,
        "expected_wip_attempt": None,
    })
    monkeypatch.setattr(
        R.Status,
        "campaign_report",
        lambda **kwargs: {
            "frontiers": [{
                **{
                    key: item[key]
                    for key in (
                        *R.Status.FRONTIER_BINDING_FIELDS,
                        "reached",
                        "parent_action_count",
                    )
                },
                "game": "ar25",
                "next_level": 1,
                "retry_complexity_n": 0,
                "warm_wip_available": True,
                "warm_wip_attempt": "not_reached_unused_capsule",
                "warm_wip_phase": "not_reached",
                "warm_wip_recovery_required": False,
            }]
        },
    )
    R.validate_live_policy_item(item)


def test_live_policy_restore_rejects_changed_wip_capsule(monkeypatch):
    item = _infrastructure_recovery_item()
    monkeypatch.setattr(
        R.Status,
        "campaign_report",
        lambda **kwargs: {
            "frontiers": [{
                **{
                    key: item[key]
                    for key in (
                        *R.Status.FRONTIER_BINDING_FIELDS,
                        "reached",
                        "parent_action_count",
                    )
                },
                "game": "ar25",
                "next_level": 1,
                "retry_complexity_n": 0,
                "warm_wip_available": True,
                "warm_wip_attempt": "infrastructure_failure_transport_new",
                "warm_wip_phase": item["warm_wip_phase"],
                "warm_wip_recovery_required": True,
            }]
        },
    )
    with pytest.raises(
        R.CampaignPlanError, match="warm_wip_attempt is stale"
    ):
        R.validate_live_policy_item(item)


def test_active_workspace_lock_detects_other_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "HERE", tmp_path)
    scratch = tmp_path / "runs" / "scratch"
    monkeypatch.setattr(R.Legs, "SCRATCH", str(scratch))
    workspace = scratch / "gkm_legs_ws_sk48_other"
    workspace.mkdir(parents=True)
    path = workspace / ".orchestrate.lock"
    lock = path.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert R.active_workspace_lock("sk48") == path
        assert R.active_workspace_lock("bp35") is None
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def test_inventory_gate_rejects_nonexistent_and_nonsequential_levels():
    targets = {"re86": 8, "bp35": 9}
    with pytest.raises(R.CampaignPlanError, match="nonexistent level"):
        R.validate_inventory_item(
            {"game": "re86", "target_level": 9}, targets, reached=8
        )
    with pytest.raises(R.CampaignPlanError, match="nonsequential"):
        R.validate_inventory_item(
            {"game": "bp35", "target_level": 8}, targets, reached=6
        )
    R.validate_inventory_item(
        {
            "game": "bp35", "target_level": 7,
            "reached": 6,
            "seed_mode": "verified_parent", "wip_mode": "exclude",
            "warm_wip_available": False,
        },
        targets, reached=6
    )
    for invalid in (True, 0, -1):
        with pytest.raises(R.CampaignPlanError, match="invalid game or target"):
            R.validate_inventory_item(
                {"game": "bp35", "target_level": invalid},
                targets,
                reached=6,
            )


def test_inventory_gate_rejects_lineage_input_mismatch():
    targets = {"bp35": 9}
    with pytest.raises(R.CampaignPlanError, match="lineage seed mismatch"):
        R.validate_inventory_item(
            {
                "game": "bp35", "target_level": 7,
                "reached": 6,
                "seed_mode": "zero_seed", "wip_mode": "exclude",
                "warm_wip_available": False,
            },
            targets, reached=6,
        )
    with pytest.raises(R.CampaignPlanError, match="without a recorded clean"):
        R.validate_inventory_item(
            {
                "game": "bp35", "target_level": 7,
                "reached": 6,
                "seed_mode": "verified_parent",
                "wip_mode": "restore_clean_same_frontier",
                "warm_wip_available": False,
            },
            targets, reached=6,
        )


def test_checkpoint_read_fails_closed_on_path_only_state(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "HERE", tmp_path)
    monkeypatch.setattr(R, "_authoritative_targets", lambda: {"ar25": 8})
    checkpoint = (
        tmp_path / "agent_solutions" / "ar25_legs" / "checkpoint.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        '{"game":"ar25","final_path":[1],"validated":true}',
        encoding="utf-8",
    )
    with pytest.raises(R.CampaignPlanError, match="untrusted checkpoint"):
        R._checkpoint_reached("ar25")


def test_live_policy_rejects_old_max_spec_after_promotion_reset(monkeypatch):
    item = copy.deepcopy(_item())
    item.update({
        "effort": "max",
        "minutes": 60,
        "retry_complexity_n": 4,
        "dispatch_mode": "first_max",
        "recommended_auxiliary_parallelism": 0,
        "wip_mode": "restore_clean_same_frontier",
        "lineage_input_mode": "zero_seed+restore_clean_same_frontier",
        "warm_wip_available": True,
    })
    monkeypatch.setattr(
        R.Status,
        "campaign_report",
        lambda **kwargs: {
            "frontiers": [{
                **{
                    key: item[key]
                    for key in (
                        *R.Status.FRONTIER_BINDING_FIELDS,
                        "reached",
                        "parent_action_count",
                    )
                },
                "game": "ar25",
                "next_level": 1,
                "retry_complexity_n": 0,
                "warm_wip_available": False,
            }]
        },
    )
    with pytest.raises(R.CampaignPlanError, match="retry coordinate is stale"):
        R.validate_live_policy_item(item)


def test_dry_run_rejects_stale_live_retry_coordinate(tmp_path, monkeypatch):
    plan_path = tmp_path / "queue.json"
    plan_path.write_text(json.dumps({
        "reserve_percent": 25,
        "cost_control_enabled": True,
        "initial_queue": [_item()],
    }))
    monkeypatch.setattr(R, "_authoritative_targets", lambda: {"ar25": 8})
    monkeypatch.setattr(R, "_checkpoint_reached", lambda game: 0)

    def reject_stale(item):
        raise R.CampaignPlanError("plan item retry coordinate is stale")

    monkeypatch.setattr(R, "validate_live_policy_item", reject_stale)
    monkeypatch.setattr(
        sys, "argv", ["codex_campaign_runner.py", "--plan", str(plan_path)]
    )
    with pytest.raises(R.CampaignPlanError, match="retry coordinate is stale"):
        R.main()


def test_run_item_turns_expected_headroom_failure_into_reserve_stop(monkeypatch):
    monkeypatch.setattr(R, "_checkpoint_reached", lambda game: 0)
    monkeypatch.setattr(
        R, "_authoritative_targets", lambda: {"ar25": 8}
    )
    monkeypatch.setattr(R, "validate_live_policy_item", lambda item: None)
    plan = {
        "not_before_epoch": 100,
        "reserve_percent": 25,
        "cost_control_enabled": True,
    }
    allowance = SimpleNamespace(remaining_percent=30)
    result = R._run_item(plan, _item(), allowance=allowance)
    assert result["result"] == "reserve_stop"
    assert "requires 6%" in result["reason"]


def _taint_dispatch_fixture(
    tmp_path, monkeypatch, *, duplicate_exec=False, child_mutation=None
):
    monkeypatch.setattr(R, "HERE", tmp_path)
    (tmp_path / "agent_solutions" / "ar25_legs").mkdir(parents=True)
    scratch = tmp_path / "scratch"
    protected_root = scratch / ".proposer_transcripts"
    protected_root.mkdir(parents=True)
    lock_root = scratch / ".workspace_locks"
    lock_root.mkdir()
    tag = "arc_agi3_n0_fresh_frontier"
    workspace_name = f"gkm_legs_ws_ar25_{tag}_deadbeef"
    workspace = scratch / workspace_name
    workspace.mkdir()
    protected = protected_root / workspace_name
    protected.mkdir()
    transcript_name = "codex_turn_20260805T000000000000Z_ar25_L1_propose.jsonl"
    diagnostics_name = (
        "codex_turn_20260805T000000000000Z_ar25_L1_propose.stderr.log"
    )
    transcript = (
        json.dumps({"type": "thread.started", "thread_id": "tainted-thread"})
        + "\n"
        + json.dumps({
            "type": "item.completed",
            "item": {
                "id": "process-query",
                "type": "command_execution",
                "command": "/bin/zsh -lc 'ps -axo pid,command'",
                "aggregated_output": "operation not permitted",
            },
        })
        + "\n"
        + json.dumps({"type": "turn.completed", "usage": {}})
        + "\n"
    ).encode()
    diagnostics = b""
    (protected / transcript_name).write_bytes(transcript)
    (protected / diagnostics_name).write_bytes(diagnostics)

    sibling = scratch / f"gkm_legs_ws_ar25_{tag}_sibling"
    sibling.mkdir()
    sibling_protected = protected_root / sibling.name
    sibling_protected.mkdir()
    (sibling / "keep.txt").write_text("keep")
    (sibling_protected / "keep.txt").write_text("keep")
    exact_lock = R.Legs._workspace_lock_path(str(workspace))
    exact_lock.write_text("")
    sibling_lock = R.Legs._workspace_lock_path(str(sibling))
    sibling_lock.write_text("")

    ledger = tmp_path / "usage.jsonl"
    item = copy.deepcopy(_item())
    item["argv"].extend([
        f"--tag={tag}",
        f"--codex-ledger={ledger}",
    ])
    record = {
        "event": "codex_exec",
        "started_at": "2026-08-05T00:00:00+00:00",
        "thread_id": "tainted-thread",
        "transcript": transcript_name,
        "diagnostics": diagnostics_name,
        "workspace": workspace_name,
        "game": "ar25",
        "target_level": 1,
        "run_label": "ar25:L1:propose",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "minutes_limit": 15,
        "allocation_policy": "drain",
        "reached": 0,
        "parent_action_count": 0,
        **{
            field: item[field]
            for field in R.Status.FRONTIER_BINDING_FIELDS
        },
        "returncode": 0,
        "failure_class": None,
        "protected_transcript_status": "sealed",
        "protected_transcript_sha256": hashlib.sha256(transcript).hexdigest(),
        "protected_diagnostics_status": "sealed",
        "protected_diagnostics_sha256": hashlib.sha256(diagnostics).hexdigest(),
        "observed_tokens": 123,
    }

    monkeypatch.setattr(R.Legs, "SCRATCH", str(scratch))
    monkeypatch.setattr(
        R.Legs, "_compatibility_arena_control_reason", lambda: None
    )
    monkeypatch.setattr(R, "_checkpoint_reached", lambda game: 0)
    monkeypatch.setattr(R, "_authoritative_targets", lambda: {"ar25": 8})
    monkeypatch.setattr(R, "validate_live_policy_item", lambda selected: None)
    expected_binding = R.Status.validate_frontier_binding({
        field: item[field]
        for field in (
            *R.Status.FRONTIER_BINDING_FIELDS,
            "game",
            "reached",
            "target_level",
            "parent_action_count",
        )
    })
    monkeypatch.setattr(
        R.Status,
        "exact_frontier_binding",
        lambda *args, **kwargs: expected_binding,
    )
    monkeypatch.setattr(R, "_taint_gate", lambda: None)

    def failed_child(*args, **kwargs):
        if child_mutation is not None:
            child_mutation(tmp_path)
        R.Guard.append_ledger(record, ledger)
        if duplicate_exec:
            R.Guard.append_ledger(record, ledger)
        return R.GuardedChildResult(
            returncode=1, process_tree_quiesced=True
        )

    monkeypatch.setattr(R, "_run_guarded_child", failed_child)
    plan = {
        "not_before_epoch": 0,
        "reserve_percent": 25,
        "cost_control_enabled": True,
    }
    allowance = SimpleNamespace(
        remaining_percent=100, window_name="weekly"
    )
    return {
        "item": item,
        "plan": plan,
        "allowance": allowance,
        "ledger": ledger,
        "workspace": workspace,
        "protected": protected,
        "sibling": sibling,
        "sibling_protected": sibling_protected,
        "exact_lock": exact_lock,
        "sibling_lock": sibling_lock,
        "record": record,
    }


def _append_clean_dispatch_ledger(fixture, *, reached_after=0):
    record = fixture["record"]
    item = fixture["item"]
    R.Guard.append_ledger(record, fixture["ledger"])
    R.Guard.append_ledger({
        "event": "codex_level_outcome",
        "codex_exec_transcript": record["transcript"],
        "thread_id": record["thread_id"],
        "game": item["game"],
        "target_level": item["target_level"],
        "run_label": record["run_label"],
        "model": record["model"],
        "reasoning_effort": item["effort"],
        "reached": item["reached"],
        "reached_before": item["reached"],
        "reached_after": reached_after,
        "solved_target": reached_after >= item["target_level"],
        "taint_verdict": "clean",
        **{
            field: item[field]
            for field in R.Status.FRONTIER_BINDING_FIELDS
        },
    }, fixture["ledger"])


def _zero_ledger_child_result(fixture):
    return R.GuardedChildResult(
        returncode=-15,
        taint_reason="host process introspection: synthetic boundary stop",
        workspace=fixture["workspace"].name,
        transcript=fixture["record"]["transcript"],
        workspace_identity=(
            fixture["workspace"].stat().st_dev,
            fixture["workspace"].stat().st_ino,
        ),
        protected_identity=(
            fixture["protected"].stat().st_dev,
            fixture["protected"].stat().st_ino,
        ),
        process_tree_quiesced=True,
        detached_processes_proven_absent=True,
    )


def test_quiesced_zero_exec_is_infrastructure_noncounting_and_exactly_cleaned(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    canonical = tmp_path / "agent_solutions" / "ar25_legs" / "solver.py"
    canonical.write_bytes(b"sealed canonical baseline\n")
    level = R._target_wip_level(fixture["item"])
    old_attempt = level / "old_attempt"
    old_attempt.mkdir(parents=True)
    (old_attempt / "sealed.txt").write_bytes(b"sealed WIP baseline\n")
    latest = level / "latest.json"
    old_latest = b'{"attempt":"old_attempt","status":"clean"}\n'
    latest.write_bytes(old_latest)
    transcript_payload = (
        fixture["protected"] / fixture["record"]["transcript"]
    ).read_bytes()

    def zero_child(*_args, **_kwargs):
        canonical.write_bytes(b"late unpublished promotion\n")
        _write_authenticated_tainted_attempt(
            fixture["item"],
            level,
            attempt="interrupted_zero_ledger",
            transcript_name=fixture["record"]["transcript"],
            transcript_payload=transcript_payload,
            update_latest=True,
        )
        return _zero_ledger_child_result(fixture)

    monkeypatch.setattr(R, "_run_guarded_child", zero_child)

    result = R._run_item(
        fixture["plan"], fixture["item"], allowance=fixture["allowance"]
    )

    assert result["result"] == "infrastructure_noncounting"
    assert result["retry_complexity_n"] == 0
    assert result["reached"] == 0
    assert not fixture["workspace"].exists()
    assert not fixture["protected"].exists()
    assert not fixture["exact_lock"].exists()
    assert canonical.read_bytes() == b"sealed canonical baseline\n"
    assert latest.read_bytes() == old_latest
    assert (old_attempt / "sealed.txt").read_bytes() == b"sealed WIP baseline\n"
    assert not (level / "interrupted_zero_ledger").exists()
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        R.ZERO_LEDGER_EVENT,
        "codex_dispatch_release_authorized",
    ]
    event = rows[0]
    assert event["failure_class"] == "infrastructure"
    assert event["retry_increment"] == 0
    assert event["codex_exec_appended"] is False
    assert R.Status.joined_turns(rows) == []
    assert R.Status.infrastructure_noncounting_events(rows) == [event]


def test_zero_exec_recovery_requires_an_exact_empty_dispatch_suffix(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)

    def child_with_unrelated_row(*_args, **_kwargs):
        R.Guard.append_ledger(
            {"event": "rate_limit_snapshot", "allowance": {}},
            fixture["ledger"],
        )
        return _zero_ledger_child_result(fixture)

    monkeypatch.setattr(R, "_run_guarded_child", child_with_unrelated_row)
    with pytest.raises(
        R.CampaignPlanError, match="did not append the bound Codex exec"
    ):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert all(
        row.get("event") != R.ZERO_LEDGER_EVENT
        for row in R.Guard.read_ledger(fixture["ledger"])
    )
    marker = (
        tmp_path / "agent_solutions" / ".campaign_quarantine" / "ar25.jsonl"
    )
    marker_rows = [
        json.loads(line)
        for line in marker.read_text(encoding="utf-8").splitlines()
    ]
    failed = marker_rows[-1]
    assert failed["event"] == "dispatch_failed"
    assert failed["workspace"] == fixture["workspace"].name
    assert failed["transcript"] == fixture["record"]["transcript"]
    assert failed["workspace_identity"] == [
        fixture["workspace"].stat().st_dev,
        fixture["workspace"].stat().st_ino,
    ]
    assert failed["protected_identity"] == [
        fixture["protected"].stat().st_dev,
        fixture["protected"].stat().st_ino,
    ]


def test_zero_exec_durable_marker_replays_without_second_child(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    child_calls = 0

    def zero_child(*_args, **_kwargs):
        nonlocal child_calls
        child_calls += 1
        return _zero_ledger_child_result(fixture)

    real_complete = R._complete_zero_ledger_recovery
    injected = False

    def crash_after_marker(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            raise RuntimeError("injected crash after zero-ledger marker")
        return real_complete(*args, **kwargs)

    monkeypatch.setattr(R, "_run_guarded_child", zero_child)
    monkeypatch.setattr(R, "_complete_zero_ledger_recovery", crash_after_marker)
    with pytest.raises(RuntimeError, match="injected crash"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    marker = (
        tmp_path / "agent_solutions" / ".campaign_quarantine" / "ar25.jsonl"
    )
    parsed = R.RebootRecovery.parse_dispatch_marker(
        marker.read_bytes(), require_recovery_arm=False
    )
    assert parsed.unquiesced["event"] == "dispatch_zero_ledger_quarantined"

    result = R._run_item(
        fixture["plan"], fixture["item"], allowance=fixture["allowance"]
    )

    assert result["result"] == "infrastructure_noncounting"
    assert result["zero_ledger_replayed"] is True
    assert child_calls == 1
    assert not marker.exists()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == [R.ZERO_LEDGER_EVENT, "codex_dispatch_release_authorized"]


def test_zero_exec_replay_rejects_protected_evidence_mutation(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        R,
        "_run_guarded_child",
        lambda *_args, **_kwargs: _zero_ledger_child_result(fixture),
    )
    real_complete = R._complete_zero_ledger_recovery
    injected = False

    def crash_once(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            raise RuntimeError("injected crash after zero-ledger marker")
        return real_complete(*args, **kwargs)

    monkeypatch.setattr(R, "_complete_zero_ledger_recovery", crash_once)
    with pytest.raises(RuntimeError, match="injected crash"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )
    transcript = fixture["protected"] / fixture["record"]["transcript"]
    transcript.write_bytes(b"replaced protected evidence\n")

    with pytest.raises(R.CampaignPlanError, match="hash does not match"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    marker = (
        tmp_path / "agent_solutions" / ".campaign_quarantine" / "ar25.jsonl"
    )
    assert marker.is_file()
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert transcript.read_bytes() == b"replaced protected evidence\n"
    assert R.Guard.read_ledger(fixture["ledger"]) == []


def _leave_zero_ledger_marker(fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(
        R,
        "_run_guarded_child",
        lambda *_args, **_kwargs: _zero_ledger_child_result(fixture),
    )
    with monkeypatch.context() as fault:
        fault.setattr(
            R,
            "_complete_zero_ledger_recovery",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected crash after zero-ledger marker")
            ),
        )
        with pytest.raises(RuntimeError, match="injected crash"):
            R._run_item(
                fixture["plan"],
                fixture["item"],
                allowance=fixture["allowance"],
            )
    marker = (
        tmp_path / "agent_solutions" / ".campaign_quarantine" / "ar25.jsonl"
    )
    parsed = R.RebootRecovery.parse_dispatch_marker(
        marker.read_bytes(), require_recovery_arm=False
    )
    assert parsed.unquiesced["event"] == "dispatch_zero_ledger_quarantined"
    return marker, parsed


@pytest.mark.parametrize("forgery", ("unexpected_field", "workspace_identity"))
def test_forged_zero_ledger_event_cannot_authorize_two_row_marker_release(
    tmp_path, monkeypatch, forgery
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    marker_path, parsed = _leave_zero_ledger_marker(
        fixture, tmp_path, monkeypatch
    )
    marker, opened = R._read_existing_dispatch_quarantine(
        fixture["item"], require_recovery_arm=False
    )
    try:
        forged = R._build_zero_ledger_event(fixture["item"], opened)
        if forgery == "unexpected_field":
            forged["forged_release_authority"] = True
        else:
            forged["workspace_identity"] = [
                parsed.unquiesced["workspace_identity"][0],
                parsed.unquiesced["workspace_identity"][1] + 1,
            ]
        with pytest.raises(
            R.CampaignPlanError, match="zero-ledger infrastructure event"
        ):
            R._validate_zero_ledger_event(fixture["item"], opened, forged)
        R.Guard.append_ledger(forged, fixture["ledger"])
        result = R._zero_ledger_result(
            fixture["item"], opened, replayed=True
        )
        with pytest.raises(
            R.CampaignPlanError, match="zero-ledger infrastructure event"
        ):
            R._build_dispatch_release_authority(
                fixture["item"],
                marker,
                fixture["ledger"],
                result,
                kind="ordinary_safe_terminal_v1",
            )
    finally:
        R._close_dispatch_quarantine(marker)
    assert marker_path.is_file()
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()


def test_zero_ledger_phase_intent_crash_replays_once_and_retires_all_residue(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        R,
        "_run_guarded_child",
        lambda *_args, **_kwargs: _zero_ledger_child_result(fixture),
    )
    real_append = R._append_zero_ledger_event_cas

    def append_then_crash(**kwargs):
        def crash():
            raise RuntimeError("injected crash after zero-ledger phase intent")

        return real_append(**kwargs, after_intent=crash)

    with monkeypatch.context() as fault:
        fault.setattr(R, "_append_zero_ledger_event_cas", append_then_crash)
        with pytest.raises(RuntimeError, match="after zero-ledger phase intent"):
            R._run_item(
                fixture["plan"],
                fixture["item"],
                allowance=fixture["allowance"],
            )

    marker = (
        tmp_path / "agent_solutions" / ".campaign_quarantine" / "ar25.jsonl"
    )
    parsed = R.RebootRecovery.parse_dispatch_marker(
        marker.read_bytes(), require_recovery_arm=False
    )
    capsule_name = parsed.armed["wip_rollback_capsule_name"]
    phase_name = R._recovery_phase_intent_names(
        fixture["ledger"], parsed.dispatch_id
    )[R.ZERO_LEDGER_EVENT]
    phase_intent = marker.parent / phase_name
    assert phase_intent.is_file()
    assert R.Guard.read_ledger(fixture["ledger"]) == []

    result = R._run_item(
        fixture["plan"], fixture["item"], allowance=fixture["allowance"]
    )

    assert result["zero_ledger_replayed"] is True
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        R.ZERO_LEDGER_EVENT,
        "codex_dispatch_release_authorized",
    ]
    assert not marker.exists()
    assert not (marker.parent / capsule_name).exists()
    assert not phase_intent.exists()
    assert not fixture["workspace"].exists()
    assert not fixture["protected"].exists()
    assert not fixture["exact_lock"].exists()


def test_zero_ledger_final_wip_mutation_blocks_release(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        R,
        "_run_guarded_child",
        lambda *_args, **_kwargs: _zero_ledger_child_result(fixture),
    )
    real_cleanup = R._resume_zero_ledger_generation_cleanup
    level = R._target_wip_level(fixture["item"])

    def cleanup_then_mutate(**kwargs):
        real_cleanup(**kwargs)
        level.mkdir(parents=True, exist_ok=True)
        (level / "injected-after-cleanup").write_bytes(b"must block release\n")

    monkeypatch.setattr(
        R, "_resume_zero_ledger_generation_cleanup", cleanup_then_mutate
    )
    with pytest.raises(R.CampaignPlanError, match="WIP"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    marker = (
        tmp_path / "agent_solutions" / ".campaign_quarantine" / "ar25.jsonl"
    )
    assert marker.is_file()
    assert (level / "injected-after-cleanup").read_bytes() == (
        b"must block release\n"
    )
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        R.ZERO_LEDGER_EVENT,
    ]


def test_zero_ledger_release_authority_crash_reconciles_before_next_dispatch(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    child_calls = 0

    def zero_child(*_args, **_kwargs):
        nonlocal child_calls
        child_calls += 1
        return _zero_ledger_child_result(fixture)

    monkeypatch.setattr(R, "_run_guarded_child", zero_child)
    real_finish = R._finish_dispatch_release_intent

    def crash_after_authority(*args, **kwargs):
        assert [
            row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
        ] == [R.ZERO_LEDGER_EVENT, "codex_dispatch_release_authorized"]
        raise RuntimeError("injected crash after release authority")

    with monkeypatch.context() as fault:
        fault.setattr(R, "_finish_dispatch_release_intent", crash_after_authority)
        with pytest.raises(RuntimeError, match="after release authority"):
            R._run_item(
                fixture["plan"],
                fixture["item"],
                allowance=fixture["allowance"],
            )

    marker = (
        tmp_path / "agent_solutions" / ".campaign_quarantine" / "ar25.jsonl"
    )
    parsed = R.RebootRecovery.parse_dispatch_marker(
        marker.read_bytes(), require_recovery_arm=False
    )
    capsule_name = parsed.armed["wip_rollback_capsule_name"]
    intent_name, preparing_name = R._dispatch_release_intent_names(marker.name)
    assert (marker.parent / intent_name).is_file()

    monkeypatch.setattr(R, "_finish_dispatch_release_intent", real_finish)
    stopped = R._run_item(
        fixture["plan"],
        fixture["item"],
        allowance=SimpleNamespace(remaining_percent=0, window_name="weekly"),
    )

    assert stopped["result"] == "reserve_stop"
    assert child_calls == 1
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        R.ZERO_LEDGER_EVENT,
        "codex_dispatch_release_authorized",
    ]
    for residue in (
        marker,
        marker.parent / capsule_name,
        marker.parent / intent_name,
        marker.parent / preparing_name,
    ):
        assert not residue.exists()


def test_nonzero_confirmed_taint_is_noncounting_and_exactly_cleaned(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)

    result = R._run_item(
        fixture["plan"],
        fixture["item"],
        allowance=fixture["allowance"],
    )

    assert result["result"] == "tainted_noncounting"
    assert result["retry_complexity_n"] == 0
    assert "host process introspection" in result["reason"]
    assert not fixture["workspace"].exists()
    assert not fixture["protected"].exists()
    assert not fixture["exact_lock"].exists()
    assert fixture["sibling_lock"].is_file()
    assert (fixture["sibling"] / "keep.txt").read_text() == "keep"
    assert (fixture["sibling_protected"] / "keep.txt").read_text() == "keep"
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_dispatch_release_authorized",
    ]
    correction = rows[1]
    assert correction["failure_class"] == "taint"
    assert correction["failure_detail_class"] == "host_process_introspection"
    assert correction["taint_verdict"] == "tainted"
    assert correction["solved_target"] is None
    assert correction["retry_increment"] == 0
    assert rows[0]["observed_tokens"] == 123


def test_confirmed_taint_without_scoped_tree_proof_preserves_quarantine(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)

    def unproved_child(*_args, **_kwargs):
        R.Guard.append_ledger(fixture["record"], fixture["ledger"])
        return R.GuardedChildResult(returncode=1)

    monkeypatch.setattr(R, "_run_guarded_child", unproved_child)
    with pytest.raises(
        R.UnquiescedChildError, match="quiescence proof"
    ):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    marker = (
        tmp_path
        / "agent_solutions"
        / ".campaign_quarantine"
        / "ar25.jsonl"
    )
    assert marker.is_file()


def test_nonzero_taint_recovery_fails_closed_on_ambiguous_exec_records(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(
        tmp_path, monkeypatch, duplicate_exec=True
    )

    with pytest.raises(R.CampaignPlanError, match="ambiguous.*ledger suffix"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert all(
        row["event"] == "codex_exec"
        for row in R.Guard.read_ledger(fixture["ledger"])
    )


@pytest.mark.parametrize("suffix", (b"[]\n", b"null\n", b"\n", b"{bad}\n"))
def test_dispatch_ledger_suffix_rejects_every_nonobject_or_malformed_row(
    tmp_path, monkeypatch, suffix
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)

    def malformed_suffix_child(*_args, **_kwargs):
        R.Guard.append_ledger(fixture["record"], fixture["ledger"])
        with fixture["ledger"].open("ab") as stream:
            stream.write(suffix)
        return R.GuardedChildResult(
            returncode=1, process_tree_quiesced=True
        )

    monkeypatch.setattr(R, "_run_guarded_child", malformed_suffix_child)

    with pytest.raises(R.CampaignPlanError, match="ledger dispatch suffix"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    marker = (
        tmp_path
        / "agent_solutions"
        / ".campaign_quarantine"
        / "ar25.jsonl"
    )
    assert marker.is_file()


def test_clean_ledger_outcome_must_bind_the_exact_exec(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)

    def mismatched_outcome_child(*_args, **_kwargs):
        _append_clean_dispatch_ledger(fixture)
        rows = R.Guard.read_ledger(fixture["ledger"])
        rows[-1]["codex_exec_transcript"] = "codex_turn_other.jsonl"
        fixture["ledger"].write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return R.GuardedChildResult(
            returncode=0,
            workspace=fixture["workspace"].name,
            transcript=fixture["record"]["transcript"],
            workspace_identity=(
                fixture["workspace"].stat().st_dev,
                fixture["workspace"].stat().st_ino,
            ),
            protected_identity=(
                fixture["protected"].stat().st_dev,
                fixture["protected"].stat().st_ino,
            ),
            process_tree_quiesced=True,
            detached_processes_proven_absent=True,
        )

    monkeypatch.setattr(R, "_run_guarded_child", mismatched_outcome_child)

    with pytest.raises(R.CampaignPlanError, match="outcome does not bind"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )


def test_nonzero_taint_recovery_fails_closed_on_hash_mismatch(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    fixture["record"]["protected_transcript_sha256"] = "f" * 64

    with pytest.raises(R.CampaignPlanError, match="hash does not match"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
    )

    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert len(R.Guard.read_ledger(fixture["ledger"])) == 1


def test_nonzero_child_without_independent_taint_confirmation_is_not_cleaned(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        R.Legs, "_workspace_or_protected_taint_reason", lambda workspace: None
    )

    with pytest.raises(R.CampaignPlanError, match="no independently confirmed"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert len(R.Guard.read_ledger(fixture["ledger"])) == 1


def test_nonzero_taint_recovery_rejects_unsafe_workspace_record(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    fixture["record"]["workspace"] = "../tainted-workspace"

    with pytest.raises(R.CampaignPlanError, match="unsafe workspace"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert len(R.Guard.read_ledger(fixture["ledger"])) == 1


def test_nonzero_taint_recovery_refuses_active_exact_workspace(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "_workspace_lock_is_active", lambda workspace: True)

    with pytest.raises(R.CampaignPlanError, match="remains active"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert len(R.Guard.read_ledger(fixture["ledger"])) == 1


def test_nonzero_taint_recovery_refuses_changed_canonical_frontier(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    changed = {
        **{
            field: fixture["item"][field]
            for field in (
                *R.Status.FRONTIER_BINDING_FIELDS,
                "game",
                "reached",
                "target_level",
                "parent_action_count",
            )
        },
        "parent_source_tree_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        R.Status, "exact_frontier_binding", lambda *args, **kwargs: changed
    )

    with pytest.raises(R.CampaignPlanError, match="canonical exact frontier"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        "codex_exec",
        "codex_exec_classification_correction",
    ]


def test_terminal_watchdog_exception_rolls_back_late_canonical_promotion(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    canonical = tmp_path / "agent_solutions" / "ar25_legs"
    source = canonical / "ar25_legs.py"
    source.write_bytes(b"sealed clean source\n")

    def terminal_revalidation_failure(*args, **kwargs):
        source.write_bytes(b"late tainted promotion\n")
        extra = canonical / "promotion_evidence" / "level_01"
        extra.mkdir(parents=True)
        (extra / "receipt.json").write_bytes(b'{"tainted":true}\n')
        raise R.CampaignPlanError("terminal historical control drift")

    monkeypatch.setattr(
        R, "_run_guarded_child", terminal_revalidation_failure
    )

    with pytest.raises(
        R.CampaignPlanError, match="terminal historical control drift"
    ):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert source.read_bytes() == b"sealed clean source\n"
    assert not (canonical / "promotion_evidence").exists()
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()


def test_unquiesced_child_exception_preserves_all_evidence_without_rollback(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    canonical = tmp_path / "agent_solutions" / "ar25_legs"
    source = canonical / "ar25_legs.py"
    source.write_bytes(b"sealed clean source\n")

    def unquiesced_failure(*args, **kwargs):
        source.write_bytes(b"possibly still mutating\n")
        raise R.UnquiescedChildError("descendant quiescence is unproven")

    monkeypatch.setattr(R, "_run_guarded_child", unquiesced_failure)

    with pytest.raises(
        R.UnquiescedChildError, match="quiescence is unproven"
    ):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert source.read_bytes() == b"possibly still mutating\n"
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()


def test_terminal_process_escape_is_cleaned_after_scoped_quiescence_proof(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    canonical = tmp_path / "agent_solutions" / "ar25_legs" / "ar25_legs.py"
    canonical.write_bytes(b"sealed canonical source\n")
    wip = (
        tmp_path
        / "agent_solutions"
        / "ar25_legs"
        / "wip_context"
        / "level_01"
    )
    wip.mkdir(parents=True)
    latest = wip / "latest.json"
    latest.write_bytes(b'{"attempt":"sealed"}\n')
    workspace_sentinel = fixture["workspace"] / "tainted.txt"
    workspace_sentinel.write_bytes(b"workspace evidence\n")

    def nominally_clean_child(*_args, **_kwargs):
        _append_clean_dispatch_ledger(fixture)
        return R.GuardedChildResult(
            returncode=0,
            workspace=fixture["workspace"].name,
            transcript=fixture["record"]["transcript"],
            workspace_identity=(
                fixture["workspace"].stat().st_dev,
                fixture["workspace"].stat().st_ino,
            ),
            protected_identity=(
                fixture["protected"].stat().st_dev,
                fixture["protected"].stat().st_ino,
            ),
            process_tree_quiesced=True,
            detached_processes_proven_absent=True,
        )

    terminal_scans = []

    def terminal_exact_scan(_item, record, *, require_taint=True):
        terminal_scans.append(require_taint)
        return (
            fixture["workspace"],
            fixture["protected"],
            "detached_process_escape: terminal-only process capability",
            record["protected_transcript_sha256"],
            record["protected_diagnostics_sha256"],
            True,
        )

    monkeypatch.setattr(R, "_run_guarded_child", nominally_clean_child)
    monkeypatch.setattr(R, "_exact_tainted_generation", terminal_exact_scan)

    result = R._run_item(
        fixture["plan"],
        fixture["item"],
        allowance=fixture["allowance"],
    )

    assert result["result"] == "tainted_noncounting"
    assert terminal_scans == [False, True]
    assert canonical.read_bytes() == b"sealed canonical source\n"
    assert latest.read_bytes() == b'{"attempt":"sealed"}\n'
    assert not workspace_sentinel.exists()
    assert not fixture["workspace"].exists()
    assert not fixture["protected"].exists()
    assert [
        row["event"] for row in R.Guard.read_ledger(fixture["ledger"])
    ] == [
        "codex_exec",
        "codex_level_outcome",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
        "codex_dispatch_release_authorized",
    ]
    quarantine = (
        tmp_path
        / "agent_solutions"
        / ".campaign_quarantine"
        / "ar25.jsonl"
    )
    assert not quarantine.exists()


def test_exact_scan_tracks_process_capability_beyond_first_taint(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    findings = (
        R.Boundary.BoundaryFinding(
            "host_path_escape",
            "candidate.py",
            1,
            "first non-process taint",
        ),
        R.Boundary.BoundaryFinding(
            "dynamic_execution",
            "candidate.py",
            2,
            "later process-capable taint",
        ),
    )
    monkeypatch.setattr(
        R.Boundary, "scan_workspace", lambda *_args, **_kwargs: findings
    )
    monkeypatch.setattr(
        R.Boundary,
        "scan_codex_transcript",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(R, "_historical_tester_scaffolds", lambda *_a: {})
    monkeypatch.setattr(
        R.Legs,
        "_workspace_or_protected_taint_reason",
        lambda _workspace: "first non-process taint",
    )

    *_, descendant_unproven = R._exact_tainted_generation(
        fixture["item"], fixture["record"], require_taint=False
    )
    assert descendant_unproven is True


@pytest.mark.parametrize(
    "marker_kind", ("valid", "malformed", "symlink", "hardlink")
)
def test_preexisting_dispatch_quarantine_blocks_before_child(
    tmp_path, monkeypatch, marker_kind
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root = tmp_path / "agent_solutions" / ".campaign_quarantine"
    root.mkdir(mode=0o700)
    os.chmod(root, 0o700)
    marker = root / "ar25.jsonl"
    outside = tmp_path / "outside-quarantine"
    outside.write_bytes(b"outside must survive\n")
    if marker_kind == "valid":
        marker.write_text(
            json.dumps({
                "schema": R.DISPATCH_QUARANTINE_SCHEMA,
                "event": "dispatch_armed",
                "dispatch_id": "a" * 32,
            }) + "\n",
            encoding="utf-8",
        )
    elif marker_kind == "malformed":
        marker.write_bytes(b"not json\n")
    elif marker_kind == "symlink":
        marker.symlink_to(outside)
    else:
        os.link(outside, marker)
    calls = 0

    def forbidden_child(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("quarantined dispatch reached the proposer")

    monkeypatch.setattr(R, "_run_guarded_child", forbidden_child)

    with pytest.raises(R.CampaignPlanError, match="explicit operator release"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert calls == 0
    assert os.path.lexists(marker)
    assert outside.read_bytes() == b"outside must survive\n"


def test_dispatch_quarantine_is_armed_before_child_and_released_after_cleanup(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    failed_child = R._run_guarded_child
    observed = []
    fsynced_identities = []
    real_fsync = R.os.fsync
    marker = (
        tmp_path
        / "agent_solutions"
        / ".campaign_quarantine"
        / "ar25.jsonl"
    )

    def observe_armed(*args, **kwargs):
        rows = [
            json.loads(line)
            for line in marker.read_text(encoding="utf-8").splitlines()
        ]
        observed.extend(rows)
        return failed_child(*args, **kwargs)

    def record_fsync(descriptor):
        metadata = os.fstat(descriptor)
        fsynced_identities.append((metadata.st_dev, metadata.st_ino))
        return real_fsync(descriptor)

    monkeypatch.setattr(R, "_run_guarded_child", observe_armed)
    monkeypatch.setattr(R.os, "fsync", record_fsync)

    result = R._run_item(
        fixture["plan"],
        fixture["item"],
        allowance=fixture["allowance"],
    )

    assert result["result"] == "tainted_noncounting"
    assert [row["event"] for row in observed] == ["dispatch_armed"]
    assert observed[0]["schema"] == (
        R.RebootRecovery.DISPATCH_QUARANTINE_SCHEMA_V2
    )
    assert observed[0]["armed_schema"] == (
        R.RebootRecovery.DISPATCH_ARMED_SCHEMA_V2
    )
    assert observed[0]["wip_rollback_capsule_sha256"]
    assert observed[0]["canonical_digest"]
    assert observed[0]["ledger_prefix_sha256"]
    artifact_root = tmp_path / "agent_solutions"
    assert (
        artifact_root.stat().st_dev,
        artifact_root.stat().st_ino,
    ) in fsynced_identities
    assert not marker.exists()


@pytest.mark.parametrize(
    "fault", ("short_write", "marker_fsync", "root_fsync")
)
def test_failed_dispatch_arm_removes_exact_marker_and_allows_retry(
    tmp_path, monkeypatch, fault
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    marker = (
        tmp_path
        / "agent_solutions"
        / ".campaign_quarantine"
        / "ar25.jsonl"
    )
    child = R._run_guarded_child
    child_calls = 0

    def counted_child(*args, **kwargs):
        nonlocal child_calls
        child_calls += 1
        return child(*args, **kwargs)

    monkeypatch.setattr(R, "_run_guarded_child", counted_child)
    if fault == "short_write":
        real_operation = R.os.write
        calls = 0

        def fail_after_short_write(descriptor, payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_operation(descriptor, payload[:17])
            if calls == 2:
                raise OSError(errno.EIO, "injected marker write failure")
            return real_operation(descriptor, payload)

        monkeypatch.setattr(R.os, "write", fail_after_short_write)
        expected = "could not durably install the WIP rollback capsule"
    else:
        real_operation = R.os.fsync
        calls = 0
        injected = False

        def fail_first_fsync(descriptor):
            nonlocal calls, injected
            calls += 1
            descriptor_metadata = os.fstat(descriptor)
            marker_metadata = (
                marker.stat(follow_symlinks=False)
                if marker.exists()
                else None
            )
            marker_descriptor = (
                marker_metadata is not None
                and (
                    descriptor_metadata.st_dev,
                    descriptor_metadata.st_ino,
                )
                == (marker_metadata.st_dev, marker_metadata.st_ino)
            )
            root_metadata = (
                marker.parent.stat(follow_symlinks=False)
                if marker.parent.exists()
                else None
            )
            root_descriptor = (
                marker.exists()
                and root_metadata is not None
                and (
                    descriptor_metadata.st_dev,
                    descriptor_metadata.st_ino,
                )
                == (root_metadata.st_dev, root_metadata.st_ino)
            )
            selected = (
                marker_descriptor
                if fault == "marker_fsync"
                else root_descriptor
            )
            if selected and not injected:
                injected = True
                raise OSError(errno.EIO, "injected marker fsync failure")
            return real_operation(descriptor)

        monkeypatch.setattr(R.os, "fsync", fail_first_fsync)
        expected = (
            "could not seal the dispatch quarantine receipt"
            if fault == "marker_fsync"
            else "could not durably install the WIP rollback capsule"
        )

    with pytest.raises(R.CampaignPlanError, match=expected):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert child_calls == 0
    assert not marker.exists()

    if fault == "short_write":
        monkeypatch.setattr(R.os, "write", real_operation)
    else:
        monkeypatch.setattr(R.os, "fsync", real_operation)
    result = R._run_item(
        fixture["plan"],
        fixture["item"],
        allowance=fixture["allowance"],
    )
    assert result["result"] == "tainted_noncounting"
    assert child_calls == 1
    assert not marker.exists()


def test_failed_dispatch_arm_preserves_replacement_marker(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    marker = (
        tmp_path
        / "agent_solutions"
        / ".campaign_quarantine"
        / "ar25.jsonl"
    )
    replacement = b"replacement quarantine must survive\n"
    real_write = R.os.write
    injected = False
    child_calls = 0

    def counted_child(*_args, **_kwargs):
        nonlocal child_calls
        child_calls += 1
        raise AssertionError("failed arm reached exact child")

    def replace_before_write(_descriptor, _payload):
        nonlocal injected
        assert injected is False
        injected = True
        marker.unlink()
        marker.write_bytes(replacement)
        raise OSError(errno.EIO, "injected post-replacement write failure")

    monkeypatch.setattr(R, "_run_guarded_child", counted_child)
    monkeypatch.setattr(R.os, "write", replace_before_write)
    with pytest.raises(
        R.CampaignPlanError,
        match="could not prove quarantine cleanup",
    ):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    monkeypatch.setattr(R.os, "write", real_write)
    assert child_calls == 0
    assert marker.read_bytes() == replacement
    with pytest.raises(R.CampaignPlanError, match="explicit operator release"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )
    assert child_calls == 0
    assert marker.read_bytes() == replacement


def test_failure_immediately_after_arm_is_durably_recorded(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    marker = (
        tmp_path
        / "agent_solutions"
        / ".campaign_quarantine"
        / "ar25.jsonl"
    )

    real_arm = R._arm_dispatch_quarantine
    child_calls = 0

    def fail_at_ownership_handoff(*args, **kwargs):
        real_arm(*args, **kwargs)
        raise RuntimeError("injected immediate post-arm failure")

    def forbidden_child(*_args, **_kwargs):
        nonlocal child_calls
        child_calls += 1
        raise AssertionError("ownership-handoff failure reached child")

    monkeypatch.setattr(
        R, "_arm_dispatch_quarantine", fail_at_ownership_handoff
    )
    monkeypatch.setattr(R, "_run_guarded_child", forbidden_child)
    with pytest.raises(RuntimeError, match="immediate post-arm failure"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    rows = [
        json.loads(line)
        for line in marker.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "dispatch_armed",
        "dispatch_failed",
    ]
    assert child_calls == 0


def test_pending_sigint_during_arm_is_delivered_after_durable_handoff(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    marker = (
        tmp_path
        / "agent_solutions"
        / ".campaign_quarantine"
        / "ar25.jsonl"
    )
    real_open = R.os.open
    injected = False
    child_calls = 0

    def interrupt_after_marker_open(path, flags, *args, **kwargs):
        nonlocal injected
        descriptor = real_open(path, flags, *args, **kwargs)
        if (
            not injected
            and path == "ar25.jsonl"
            and flags & os.O_EXCL
        ):
            injected = True
            signal.raise_signal(signal.SIGINT)
        return descriptor

    def forbidden_child(*_args, **_kwargs):
        nonlocal child_calls
        child_calls += 1
        raise AssertionError("pending arm signal reached child")

    monkeypatch.setattr(R.os, "open", interrupt_after_marker_open)
    monkeypatch.setattr(R, "_run_guarded_child", forbidden_child)
    with pytest.raises(KeyboardInterrupt):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    rows = [
        json.loads(line)
        for line in marker.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "dispatch_armed",
        "dispatch_failed",
    ]
    assert injected is True
    assert child_calls == 0


def test_unquiesced_dispatch_marker_blocks_a_second_scheduler_process(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    calls = 0

    def unquiesced(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise R.UnquiescedChildError("detached child remains unproven")

    monkeypatch.setattr(R, "_run_guarded_child", unquiesced)
    with pytest.raises(R.UnquiescedChildError, match="remains unproven"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    marker = (
        tmp_path
        / "agent_solutions"
        / ".campaign_quarantine"
        / "ar25.jsonl"
    )
    assert marker.is_file()
    rows = [
        json.loads(line)
        for line in marker.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "dispatch_armed",
        "dispatch_failed",
    ]
    with pytest.raises(R.CampaignPlanError, match="explicit operator release"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )
    assert calls == 1


def test_complete_unquiesced_generation_writes_recoverable_marker(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)

    def complete_unquiesced_child(*_args, **_kwargs):
        return R.GuardedChildResult(
            returncode=1,
            workspace=fixture["workspace"].name,
            transcript=fixture["record"]["transcript"],
            workspace_identity=(
                fixture["workspace"].stat().st_dev,
                fixture["workspace"].stat().st_ino,
            ),
            protected_identity=(
                fixture["protected"].stat().st_dev,
                fixture["protected"].stat().st_ino,
            ),
            descendant_quiescence_unproven=True,
            process_tree_quiesced=True,
        )

    monkeypatch.setattr(
        R, "_run_guarded_child", complete_unquiesced_child
    )
    with pytest.raises(R.UnquiescedChildError, match="quiescence proof"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    marker = (
        tmp_path
        / "agent_solutions"
        / ".campaign_quarantine"
        / "ar25.jsonl"
    )
    parsed = R.RebootRecovery.parse_dispatch_marker(
        marker.read_bytes(), require_recovery_arm=False
    )
    assert parsed.unquiesced["event"] == "dispatch_unquiesced"
    assert parsed.unquiesced["workspace"] == fixture["workspace"].name


def test_dispatch_quarantine_root_replacement_refuses_safe_release(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    failed_child = R._run_guarded_child
    root = tmp_path / "agent_solutions" / ".campaign_quarantine"
    displaced = tmp_path / "displaced-quarantine"

    def replace_quarantine_root(*args, **kwargs):
        root.rename(displaced)
        root.mkdir(mode=0o700)
        os.chmod(root, 0o700)
        (root / "ar25.jsonl").write_bytes(b"replacement must survive\n")
        return failed_child(*args, **kwargs)

    monkeypatch.setattr(
        R, "_run_guarded_child", replace_quarantine_root
    )

    with pytest.raises(R.CampaignPlanError, match="identity changed"):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert (displaced / "ar25.jsonl").is_file()
    assert (root / "ar25.jsonl").read_bytes() == b"replacement must survive\n"


def _release_test_paths(tmp_path):
    root = tmp_path / "agent_solutions" / ".campaign_quarantine"
    marker = root / "ar25.jsonl"
    intent_name, preparing_name = R._dispatch_release_intent_names(
        marker.name
    )
    return root, marker, root / intent_name, root / preparing_name


def _release_test_capsule(marker):
    armed = json.loads(marker.read_bytes().splitlines()[0])
    name = armed["wip_rollback_capsule_name"]
    assert isinstance(name, str) and Path(name).name == name
    return marker.parent / name


def _release_test_file_state(path):
    metadata = path.stat(follow_symlinks=False)
    return (metadata.st_dev, metadata.st_ino, path.read_bytes())


def _release_test_residue(root):
    if not root.exists():
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if (
            name == "ar25.jsonl"
            or name.startswith(".ar25.jsonl.release_")
            or (
                name.startswith(".ar25.jsonl.")
                and name.endswith(".wip_rollback_capsule")
            )
        )
    )


def _leave_complete_release_preparing(fixture, tmp_path, monkeypatch):
    root, marker, intent, preparing = _release_test_paths(tmp_path)
    real_replace = R.os.replace
    injected = False

    def fail_before_intent_install(source, target, *args, **kwargs):
        nonlocal injected
        if (
            os.fspath(source) == preparing.name
            and os.fspath(target) == intent.name
        ):
            injected = True
            raise OSError(errno.EIO, "synthetic release-intent rename crash")
        return real_replace(source, target, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(R.os, "replace", fail_before_intent_install)
        with pytest.raises(R.CampaignPlanError) as failure:
            R._run_item(
                fixture["plan"],
                fixture["item"],
                allowance=fixture["allowance"],
            )

    assert injected is True, repr(failure.value)
    assert marker.is_file()
    assert preparing.is_file()
    assert not intent.exists()
    capsule = _release_test_capsule(marker)
    assert capsule.is_file()
    return root, marker, intent, preparing, capsule


def _release_test_crash_after_unlink(
    fixture, monkeypatch, *, target
):
    real_unlink = R.os.unlink
    real_fsync = R.os.fsync
    state = {"awaiting_fsync": False, "injected": False}

    def selected(name):
        return {
            "capsule": name.endswith(".wip_rollback_capsule"),
            "marker": name == "ar25.jsonl",
            "intent": name == ".ar25.jsonl.release_intent",
        }[target]

    def observe_unlink(path, *args, **kwargs):
        result = real_unlink(path, *args, **kwargs)
        if selected(os.fspath(path)):
            state["awaiting_fsync"] = True
        return result

    def fail_root_fsync(descriptor):
        if state["awaiting_fsync"] and not state["injected"]:
            state["injected"] = True
            state["awaiting_fsync"] = False
            raise OSError(
                errno.EIO,
                f"synthetic {target} unlink-before-root-fsync crash",
            )
        return real_fsync(descriptor)

    with monkeypatch.context() as fault:
        fault.setattr(R.os, "unlink", observe_unlink)
        fault.setattr(R.os, "fsync", fail_root_fsync)
        with pytest.raises(R.CampaignPlanError) as failure:
            R._run_item(
                fixture["plan"],
                fixture["item"],
                allowance=fixture["allowance"],
            )
    assert state["injected"] is True, repr(failure.value)


def _configure_clean_release(
    fixture, monkeypatch, *, reached_after
):
    state = {"reached": 0, "child_calls": 0}

    def checkpoint(_game):
        return state["reached"]

    def clean_child(*_args, **_kwargs):
        state["child_calls"] += 1
        _append_clean_dispatch_ledger(
            fixture, reached_after=reached_after
        )
        state["reached"] = reached_after
        return R.GuardedChildResult(
            returncode=0,
            process_tree_quiesced=True,
            detached_processes_proven_absent=True,
        )

    monkeypatch.setattr(R, "_checkpoint_reached", checkpoint)
    monkeypatch.setattr(R, "_run_guarded_child", clean_child)
    monkeypatch.setattr(
        R, "_authenticate_clean_generation", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(R, "_refresh_solver_audits", lambda: None)
    return state


def test_complete_release_preparing_without_authority_preserves_quarantine(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, _intent, _preparing, capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    marker_before = _release_test_file_state(marker)
    capsule_before = _release_test_file_state(capsule)
    assert not any(
        row.get("event") == "codex_dispatch_release_authorized"
        for row in R.Guard.read_ledger(fixture["ledger"])
    )

    with pytest.raises(
        R.CampaignPlanError, match="explicit operator release"
    ):
        R._assert_no_dispatch_quarantine(fixture["item"])

    assert _release_test_file_state(marker) == marker_before
    assert _release_test_file_state(capsule) == capsule_before
    assert marker.name in _release_test_residue(root)
    assert capsule.name in _release_test_residue(root)


def test_safe_release_empty_tail_arm_reboot_recover_and_markerless_replay(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, _intent, preparing, capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    armed_row = json.loads(marker.read_bytes().splitlines()[0])
    dispatch_id = armed_row["dispatch_id"]
    before_arm_ledger = fixture["ledger"].read_bytes()
    marker_before = _release_test_file_state(marker)
    preparing_before = _release_test_file_state(preparing)
    capsule_before = _release_test_file_state(capsule)
    armed_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "11111111-1111-4111-8111-111111111111",
    )
    current_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "22222222-2222-4222-8222-222222222222",
    )

    armed = R._arm_post_reboot_recovery(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        boot_identity_provider=lambda: armed_boot,
    )
    assert armed["result"] == "post_reboot_safe_release_armed"
    assert fixture["ledger"].read_bytes() == before_arm_ledger
    assert _release_test_file_state(marker) == marker_before
    assert _release_test_file_state(preparing) == preparing_before
    assert _release_test_file_state(capsule) == capsule_before

    outcome = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: current_boot,
    )
    assert outcome["operator_recovery"] == (
        "post_reboot_safe_release_authenticated"
    )
    assert not marker.exists()
    assert not preparing.exists()
    assert not capsule.exists()
    arm_name = R._safe_release_recovery_arm_name(marker.name)
    receipt_name = R._safe_release_recovery_receipt_name(
        marker.name, dispatch_id
    )
    assert not (root / arm_name).exists()
    assert (root / receipt_name).is_file()
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert sum(
        row.get("event") == "codex_dispatch_release_authorized"
        for row in rows
    ) == 1
    sealed = fixture["ledger"].read_bytes()

    repeated = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: current_boot,
    )
    assert repeated["operator_recovery"] == (
        "post_reboot_safe_release_already_completed"
    )
    assert fixture["ledger"].read_bytes() == sealed


def test_marker_absence_cannot_retire_pre_authority_release_preparing(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, _intent, preparing, capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    preparing_before = _release_test_file_state(preparing)
    capsule_before = _release_test_file_state(capsule)
    os.unlink(marker)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)

    with pytest.raises(R.CampaignPlanError):
        R._assert_no_dispatch_quarantine(fixture["item"])

    assert _release_test_file_state(preparing) == preparing_before
    assert _release_test_file_state(capsule) == capsule_before


def test_release_preparing_rejects_later_unquiesced_marker_row(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    workspace_identity = (
        fixture["workspace"].stat().st_dev,
        fixture["workspace"].stat().st_ino,
    )
    protected_identity = (
        fixture["protected"].stat().st_dev,
        fixture["protected"].stat().st_ino,
    )
    _root, marker, _intent, _preparing, capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    armed = json.loads(marker.read_bytes().splitlines()[0])
    row = {
        "schema": armed["schema"],
        "dispatch_id": armed["dispatch_id"],
        "event": "dispatch_unquiesced",
        "recorded_at": armed["recorded_at"],
        "exception_type": "UnquiescedChildError",
        "reason": "synthetic later unquiesced terminal row",
        "child_returncode": 1,
        "workspace": fixture["workspace"].name,
        "protected": fixture["protected"].name,
        "transcript": fixture["record"]["transcript"],
        "workspace_identity": list(workspace_identity),
        "protected_identity": list(protected_identity),
    }
    payload = R.RebootRecovery.canonical_json_line(row)
    descriptor = os.open(
        marker,
        os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        assert (opened.st_dev, opened.st_ino) == (
            marker.stat(follow_symlinks=False).st_dev,
            marker.stat(follow_symlinks=False).st_ino,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            assert written > 0
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    root_fd = os.open(
        marker.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    marker_before = _release_test_file_state(marker)
    capsule_before = _release_test_file_state(capsule)

    with pytest.raises(R.CampaignPlanError):
        R._assert_no_dispatch_quarantine(fixture["item"])

    assert _release_test_file_state(marker) == marker_before
    assert _release_test_file_state(capsule) == capsule_before
    parsed = R.RebootRecovery.parse_dispatch_marker(
        marker.read_bytes(), require_recovery_arm=False
    )
    assert parsed.unquiesced["event"] == "dispatch_unquiesced"


def _leave_installed_release_authority_tail(
    fixture, tmp_path, monkeypatch, *, conflicting=False
):
    root, marker, intent, preparing = _release_test_paths(tmp_path)
    captured = {}

    def crash_during_authority_append(
        item,
        root_fd,
        record,
        intent_identity,
        *,
        allow_new_authority_append=False,
    ):
        del item, root_fd, intent_identity, allow_new_authority_append
        authority = record["release_authority"]
        line = R.RebootRecovery.canonical_json_line(
            authority["authority_record"]
        )
        ledger = Path(authority["ledger"])
        prefix = ledger.read_bytes()
        fragment = (
            b"!conflicting-release-authority-tail"
            if conflicting else line[: max(1, len(line) // 2)]
        )
        with ledger.open("ab", buffering=0) as stream:
            stream.write(fragment)
            os.fsync(stream.fileno())
        captured.update(prefix=prefix, line=line, fragment=fragment)
        raise OSError(errno.EIO, "synthetic authority append crash")

    with monkeypatch.context() as fault:
        fault.setattr(
            R,
            "_ensure_dispatch_release_authority_row",
            crash_during_authority_append,
        )
        with pytest.raises(R.CampaignPlanError):
            R._run_item(
                fixture["plan"],
                fixture["item"],
                allowance=fixture["allowance"],
            )
    assert captured and marker.is_file() and intent.is_file()
    assert not preparing.exists()
    return root, marker, intent, _release_test_capsule(marker), captured


def test_safe_release_partial_tail_arm_reboot_recover_and_markerless_replay(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, intent, capsule, captured = (
        _leave_installed_release_authority_tail(
            fixture, tmp_path, monkeypatch
        )
    )
    dispatch_id = json.loads(marker.read_bytes().splitlines()[0])["dispatch_id"]
    before_arm_ledger = fixture["ledger"].read_bytes()
    armed_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "11111111-1111-4111-8111-111111111111",
    )
    current_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "22222222-2222-4222-8222-222222222222",
    )

    armed = R._arm_post_reboot_recovery(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        boot_identity_provider=lambda: armed_boot,
    )
    assert fixture["ledger"].read_bytes() == before_arm_ledger
    assert before_arm_ledger == captured["prefix"] + captured["fragment"]
    assert marker.is_file() and intent.is_file() and capsule.is_file()

    outcome = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: current_boot,
    )
    assert outcome["operator_recovery"] == (
        "post_reboot_safe_release_authenticated"
    )
    assert not marker.exists() and not intent.exists() and not capsule.exists()
    assert fixture["ledger"].read_bytes() != before_arm_ledger
    assert sum(
        row.get("event") == "codex_dispatch_release_authorized"
        for row in R.Guard.read_ledger(fixture["ledger"])
    ) == 1
    sealed = fixture["ledger"].read_bytes()
    repeated = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: current_boot,
    )
    assert repeated["operator_recovery"] == (
        "post_reboot_safe_release_already_completed"
    )
    assert fixture["ledger"].read_bytes() == sealed
    assert (root / R._safe_release_recovery_receipt_name(
        marker.name, dispatch_id
    )).is_file()


@pytest.mark.parametrize(
    "boundary",
    (
        "after_old_wal_retire",
        "after_receipt",
        "after_authority",
        "after_release_before_arm_retire",
    ),
)
def test_safe_release_recovery_crash_boundaries_are_idempotent(
    tmp_path, monkeypatch, boundary
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, _intent, _preparing, _capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    dispatch_id = json.loads(marker.read_bytes().splitlines()[0])["dispatch_id"]
    armed_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "11111111-1111-4111-8111-111111111111",
    )
    current_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "22222222-2222-4222-8222-222222222222",
    )
    armed = R._arm_post_reboot_recovery(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        boot_identity_provider=lambda: armed_boot,
    )
    arm_name = R._safe_release_recovery_arm_name(marker.name)
    receipt_name = R._safe_release_recovery_receipt_name(
        marker.name, dispatch_id
    )

    with monkeypatch.context() as fault:
        if boundary == "after_old_wal_retire":
            original = R._build_dispatch_release_authority
            injected = False

            def crash_after_old_wal(*args, **kwargs):
                nonlocal injected
                if not injected:
                    injected = True
                    raise R.CampaignPlanError(
                        "synthetic crash after old WAL retirement"
                    )
                return original(*args, **kwargs)

            fault.setattr(R, "_build_dispatch_release_authority", crash_after_old_wal)
            pattern = "after old WAL retirement"
        elif boundary == "after_receipt":
            def crash_after_receipt(*_args, **_kwargs):
                assert (root / receipt_name).is_file()
                raise R.CampaignPlanError("synthetic crash after receipt")

            fault.setattr(
                R, "_ensure_dispatch_release_authority_row", crash_after_receipt
            )
            pattern = "after receipt"
        elif boundary == "after_authority":
            def crash_after_authority(*_args, **_kwargs):
                assert any(
                    row.get("event") == "codex_dispatch_release_authorized"
                    for row in R.Guard.read_ledger(fixture["ledger"])
                )
                raise R.CampaignPlanError("synthetic crash after authority")

            fault.setattr(
                R, "_finish_dispatch_release_intent", crash_after_authority
            )
            pattern = "after authority"
        else:
            original_unlink = R.os.unlink

            def crash_before_arm_retire(path, *args, **kwargs):
                if os.fspath(path) == arm_name:
                    raise OSError(
                        errno.EIO, "synthetic crash before arm retirement"
                    )
                return original_unlink(path, *args, **kwargs)

            fault.setattr(R.os, "unlink", crash_before_arm_retire)
            pattern = "before arm retirement"
        with pytest.raises((R.CampaignPlanError, OSError), match=pattern):
            R._recover_post_reboot_quarantine(
                fixture["item"],
                confirm_dispatch_id=dispatch_id,
                confirm_recovery_nonce=armed["recovery_nonce"],
                boot_identity_provider=lambda: current_boot,
            )

    recovered = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: current_boot,
    )
    assert recovered["result"] == "tainted_noncounting"
    assert not marker.exists()
    assert not (root / arm_name).exists()
    assert (root / receipt_name).is_file()
    assert sum(
        row.get("event") == "codex_dispatch_release_authorized"
        for row in R.Guard.read_ledger(fixture["ledger"])
    ) == 1
    sealed = fixture["ledger"].read_bytes()
    repeated = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: current_boot,
    )
    assert repeated["operator_recovery"] == (
        "post_reboot_safe_release_already_completed"
    )
    assert fixture["ledger"].read_bytes() == sealed


def test_safe_release_fresh_intent_partial_write_retries_after_old_wal_retire(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, intent, old_preparing, capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    dispatch_id = json.loads(marker.read_bytes().splitlines()[0])["dispatch_id"]
    armed_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "11111111-1111-4111-8111-111111111111",
    )
    current_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "22222222-2222-4222-8222-222222222222",
    )
    armed = R._arm_post_reboot_recovery(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        boot_identity_provider=lambda: armed_boot,
    )
    old_identity = (
        old_preparing.stat(follow_symlinks=False).st_dev,
        old_preparing.stat(follow_symlinks=False).st_ino,
    )
    receipt_name = R._safe_release_recovery_receipt_name(
        marker.name, dispatch_id
    )
    real_write = R.os.write
    target_identity = None

    def partial_fresh_intent_write(descriptor, payload):
        nonlocal target_identity
        if target_identity is not None:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) == target_identity:
                raise OSError(
                    errno.EIO,
                    "synthetic fresh release-intent partial-write crash",
                )
        if R.DISPATCH_RELEASE_INTENT_SCHEMA.encode("utf-8") in payload:
            opened = os.fstat(descriptor)
            target_identity = (opened.st_dev, opened.st_ino)
            written = real_write(descriptor, payload[:17])
            assert written == 17
            return written
        return real_write(descriptor, payload)

    with monkeypatch.context() as fault:
        fault.setattr(R.os, "write", partial_fresh_intent_write)
        with pytest.raises(
            R.CampaignPlanError,
            match="durably install the dispatch release intent",
        ):
            R._recover_post_reboot_quarantine(
                fixture["item"],
                confirm_dispatch_id=dispatch_id,
                confirm_recovery_nonce=armed["recovery_nonce"],
                boot_identity_provider=lambda: current_boot,
            )

    assert target_identity is not None and target_identity != old_identity
    assert marker.is_file() and capsule.is_file()
    assert old_preparing.is_file()
    assert len(old_preparing.read_bytes()) == 17
    assert not intent.exists()
    assert not (root / receipt_name).exists()

    recovered = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: current_boot,
    )
    assert recovered["result"] == "tainted_noncounting"
    assert not marker.exists() and not old_preparing.exists() and not capsule.exists()
    assert (root / receipt_name).is_file()
    assert sum(
        row.get("event") == "codex_dispatch_release_authorized"
        for row in R.Guard.read_ledger(fixture["ledger"])
    ) == 1
    sealed = fixture["ledger"].read_bytes()
    replay = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: current_boot,
    )
    assert replay["operator_recovery"] == (
        "post_reboot_safe_release_already_completed"
    )
    assert fixture["ledger"].read_bytes() == sealed


def _leave_malformed_fresh_release_preparing(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, intent, preparing, capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    dispatch_id = json.loads(marker.read_bytes().splitlines()[0])["dispatch_id"]
    armed_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "11111111-1111-4111-8111-111111111111",
    )
    current_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "22222222-2222-4222-8222-222222222222",
    )
    armed = R._arm_post_reboot_recovery(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        boot_identity_provider=lambda: armed_boot,
    )
    old_identity = (
        preparing.stat(follow_symlinks=False).st_dev,
        preparing.stat(follow_symlinks=False).st_ino,
    )
    real_write = R.os.write
    target_identity = None

    def partial_fresh_intent_write(descriptor, payload):
        nonlocal target_identity
        if target_identity is not None:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) == target_identity:
                raise OSError(
                    errno.EIO,
                    "synthetic malformed fresh release staging",
                )
        if R.DISPATCH_RELEASE_INTENT_SCHEMA.encode("utf-8") in payload:
            opened = os.fstat(descriptor)
            target_identity = (opened.st_dev, opened.st_ino)
            written = real_write(descriptor, payload[:17])
            assert written == 17
            return written
        return real_write(descriptor, payload)

    with monkeypatch.context() as fault:
        fault.setattr(R.os, "write", partial_fresh_intent_write)
        with pytest.raises(R.CampaignPlanError):
            R._recover_post_reboot_quarantine(
                fixture["item"],
                confirm_dispatch_id=dispatch_id,
                confirm_recovery_nonce=armed["recovery_nonce"],
                boot_identity_provider=lambda: current_boot,
            )
    assert target_identity is not None and target_identity != old_identity
    assert preparing.is_file() and len(preparing.read_bytes()) == 17
    assert not intent.exists()
    return {
        "fixture": fixture,
        "root": root,
        "marker": marker,
        "intent": intent,
        "preparing": preparing,
        "capsule": capsule,
        "dispatch_id": dispatch_id,
        "armed": armed,
        "current_boot": current_boot,
    }


@pytest.mark.parametrize(
    "surface",
    (
        "marker",
        "ledger",
        "capsule",
        "receipt",
        "hardlink",
        "mode",
        "malformed_final",
    ),
)
def test_malformed_fresh_release_staging_preserved_on_authority_drift(
    tmp_path, monkeypatch, surface
):
    state = _leave_malformed_fresh_release_preparing(tmp_path, monkeypatch)
    fixture = state["fixture"]
    root = state["root"]
    marker = state["marker"]
    intent = state["intent"]
    preparing = state["preparing"]
    capsule = state["capsule"]
    receipt = root / R._safe_release_recovery_receipt_name(
        marker.name, state["dispatch_id"]
    )
    target = preparing

    if surface == "marker":
        with marker.open("ab", buffering=0) as stream:
            stream.write(b"x")
            os.fsync(stream.fileno())
    elif surface == "ledger":
        with fixture["ledger"].open("ab", buffering=0) as stream:
            stream.write(b"x")
            os.fsync(stream.fileno())
    elif surface == "capsule":
        with capsule.open("ab", buffering=0) as stream:
            stream.write(b"x")
            os.fsync(stream.fileno())
    elif surface == "receipt":
        receipt.write_bytes(b"conflicting receipt residue\n")
        os.chmod(receipt, 0o600)
    elif surface == "hardlink":
        os.link(preparing, root / "fresh-release-staging-alias")
    elif surface == "mode":
        os.chmod(preparing, 0o640)
    else:
        os.rename(preparing, intent)
        target = intent
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    target_before = _release_test_file_state(target)

    with pytest.raises(R.CampaignPlanError):
        R._recover_post_reboot_quarantine(
            fixture["item"],
            confirm_dispatch_id=state["dispatch_id"],
            confirm_recovery_nonce=state["armed"]["recovery_nonce"],
            boot_identity_provider=lambda: state["current_boot"],
        )

    assert _release_test_file_state(target) == target_before
    assert marker.is_file() and capsule.is_file()


def test_malformed_arm_bound_release_wal_is_never_retired(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, _intent, preparing, capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    dispatch_id = json.loads(marker.read_bytes().splitlines()[0])["dispatch_id"]
    armed_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "11111111-1111-4111-8111-111111111111",
    )
    current_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "22222222-2222-4222-8222-222222222222",
    )
    armed = R._arm_post_reboot_recovery(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        boot_identity_provider=lambda: armed_boot,
    )
    descriptor = os.open(
        preparing,
        os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.write(descriptor, b"{")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    preparing_before = _release_test_file_state(preparing)

    with pytest.raises(R.CampaignPlanError):
        R._recover_post_reboot_quarantine(
            fixture["item"],
            confirm_dispatch_id=dispatch_id,
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: current_boot,
        )

    assert _release_test_file_state(preparing) == preparing_before
    assert marker.is_file() and capsule.is_file()


def _inject_safe_release_sidecar_boundary(
    fault,
    *,
    root,
    final_name,
    schema,
    boundary,
    preparing_name=None,
):
    """Inject one exact durable-sidecar boundary without touching fixtures."""

    real_write = R.os.write
    real_fsync = R.os.fsync
    real_replace = R.os.replace
    real_pread = R.os.pread
    root_metadata = root.stat(follow_symlinks=False)
    root_identity = (root_metadata.st_dev, root_metadata.st_ino)
    if preparing_name is None:
        preparing_name = R._durable_recovery_record_preparing_name(final_name)
    token = schema.encode("utf-8")
    state = {
        "target_seen": False,
        "target_identity": None,
        "renamed": False,
        "injected": False,
    }

    def target_fd(descriptor):
        if state["target_identity"] is None:
            return False
        opened = os.fstat(descriptor)
        return (opened.st_dev, opened.st_ino) == state["target_identity"]

    def write_boundary(descriptor, payload):
        if not state["target_seen"] and token in payload:
            state["target_seen"] = True
            opened = os.fstat(descriptor)
            state["target_identity"] = (opened.st_dev, opened.st_ino)
            if boundary in {"short_write", "partial_write"}:
                count = max(1, len(payload) // 2)
                written = real_write(descriptor, payload[:count])
                assert written == count
                if boundary == "partial_write":
                    state["injected"] = True
                    raise OSError(
                        errno.EIO,
                        "synthetic partial safe-release sidecar write",
                    )
                return written
        return real_write(descriptor, payload)

    def replace_boundary(source, target, *args, **kwargs):
        result = real_replace(source, target, *args, **kwargs)
        if (
            os.fspath(source) == preparing_name
            and os.fspath(target) == final_name
        ):
            state["renamed"] = True
            if boundary == "rename" and not state["injected"]:
                state["injected"] = True
                raise OSError(
                    errno.EIO,
                    "synthetic safe-release sidecar rename report failure",
                )
        return result

    def fsync_boundary(descriptor):
        if (
            boundary == "file_fsync"
            and state["target_seen"]
            and target_fd(descriptor)
            and not state["injected"]
        ):
            result = real_fsync(descriptor)
            state["injected"] = True
            raise OSError(
                errno.EIO,
                "synthetic safe-release sidecar file-fsync failure",
            )
        opened = os.fstat(descriptor)
        if (
            boundary == "root_fsync_after_rename"
            and state["renamed"]
            and (opened.st_dev, opened.st_ino) == root_identity
            and not state["injected"]
        ):
            result = real_fsync(descriptor)
            state["injected"] = True
            raise OSError(
                errno.EIO,
                "synthetic safe-release sidecar root-fsync failure",
            )
        return real_fsync(descriptor)

    def pread_boundary(descriptor, count, offset):
        if (
            boundary == "strict_reread"
            and state["renamed"]
            and target_fd(descriptor)
            and not state["injected"]
        ):
            state["injected"] = True
            raise OSError(
                errno.EIO,
                "synthetic safe-release sidecar strict-reread failure",
            )
        return real_pread(descriptor, count, offset)

    fault.setattr(R.os, "write", write_boundary)
    fault.setattr(R.os, "replace", replace_boundary)
    fault.setattr(R.os, "fsync", fsync_boundary)
    fault.setattr(R.os, "pread", pread_boundary)
    return state


@pytest.mark.parametrize(
    "boundary",
    (
        "short_write",
        "partial_write",
        "file_fsync",
        "rename",
        "root_fsync_after_rename",
        "strict_reread",
    ),
)
def test_safe_release_arm_sidecar_boundaries_retry_exactly(
    tmp_path, monkeypatch, boundary
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, _intent, preparing, capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    dispatch_id = json.loads(marker.read_bytes().splitlines()[0])["dispatch_id"]
    marker_before = _release_test_file_state(marker)
    wal_before = _release_test_file_state(preparing)
    capsule_before = _release_test_file_state(capsule)
    ledger_before = fixture["ledger"].read_bytes()
    boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "11111111-1111-4111-8111-111111111111",
    )
    arm_name = R._safe_release_recovery_arm_name(marker.name)

    with monkeypatch.context() as fault:
        state = _inject_safe_release_sidecar_boundary(
            fault,
            root=root,
            final_name=arm_name,
            schema=R.SAFE_RELEASE_RECOVERY_ARM_SCHEMA,
            boundary=boundary,
        )
        if boundary == "short_write":
            armed = R._arm_post_reboot_recovery(
                fixture["item"],
                confirm_dispatch_id=dispatch_id,
                boot_identity_provider=lambda: boot,
            )
        else:
            with pytest.raises((R.CampaignPlanError, OSError)):
                R._arm_post_reboot_recovery(
                    fixture["item"],
                    confirm_dispatch_id=dispatch_id,
                    boot_identity_provider=lambda: boot,
                )
            assert state["injected"] is True
            armed = R._arm_post_reboot_recovery(
                fixture["item"],
                confirm_dispatch_id=dispatch_id,
                boot_identity_provider=lambda: boot,
            )

    assert state["target_seen"] is True
    assert armed["result"] in {
        "post_reboot_safe_release_armed",
        "post_reboot_safe_release_already_armed",
    }
    assert _release_test_file_state(marker) == marker_before
    assert _release_test_file_state(preparing) == wal_before
    assert _release_test_file_state(capsule) == capsule_before
    assert fixture["ledger"].read_bytes() == ledger_before
    assert (root / arm_name).is_file()
    assert not (
        root / R._durable_recovery_record_preparing_name(arm_name)
    ).exists()


@pytest.mark.parametrize(
    "boundary",
    (
        "short_write",
        "partial_write",
        "file_fsync",
        "rename",
        "root_fsync_after_rename",
        "strict_reread",
    ),
)
def test_safe_release_receipt_sidecar_boundaries_retry_exactly(
    tmp_path, monkeypatch, boundary
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, _intent, _preparing, _capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    dispatch_id = json.loads(marker.read_bytes().splitlines()[0])["dispatch_id"]
    armed_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "11111111-1111-4111-8111-111111111111",
    )
    current_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "22222222-2222-4222-8222-222222222222",
    )
    armed = R._arm_post_reboot_recovery(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        boot_identity_provider=lambda: armed_boot,
    )
    receipt_name = R._safe_release_recovery_receipt_name(
        marker.name, dispatch_id
    )

    with monkeypatch.context() as fault:
        state = _inject_safe_release_sidecar_boundary(
            fault,
            root=root,
            final_name=receipt_name,
            schema=R.SAFE_RELEASE_RECOVERY_RECEIPT_SCHEMA,
            boundary=boundary,
        )
        if boundary == "short_write":
            outcome = R._recover_post_reboot_quarantine(
                fixture["item"],
                confirm_dispatch_id=dispatch_id,
                confirm_recovery_nonce=armed["recovery_nonce"],
                boot_identity_provider=lambda: current_boot,
            )
        else:
            with pytest.raises((R.CampaignPlanError, OSError)):
                R._recover_post_reboot_quarantine(
                    fixture["item"],
                    confirm_dispatch_id=dispatch_id,
                    confirm_recovery_nonce=armed["recovery_nonce"],
                    boot_identity_provider=lambda: current_boot,
                )
            assert state["injected"] is True
            outcome = R._recover_post_reboot_quarantine(
                fixture["item"],
                confirm_dispatch_id=dispatch_id,
                confirm_recovery_nonce=armed["recovery_nonce"],
                boot_identity_provider=lambda: current_boot,
            )

    assert state["target_seen"] is True
    assert outcome["result"] == "tainted_noncounting"
    assert not marker.exists()
    assert (root / receipt_name).is_file()
    assert not (
        root / R._durable_recovery_record_preparing_name(receipt_name)
    ).exists()
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert sum(
        row.get("event") == "codex_dispatch_release_authorized"
        for row in rows
    ) == 1
    sealed = fixture["ledger"].read_bytes()
    replay = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: current_boot,
    )
    assert replay["operator_recovery"] == (
        "post_reboot_safe_release_already_completed"
    )
    assert fixture["ledger"].read_bytes() == sealed


@pytest.mark.parametrize(
    "boundary",
    (
        "short_write",
        "partial_write",
        "file_fsync",
        "rename",
        "root_fsync_after_rename",
        "strict_reread",
    ),
)
def test_safe_release_fresh_intent_boundaries_retry_after_old_wal_retire(
    tmp_path, monkeypatch, boundary
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, intent, preparing, capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    dispatch_id = json.loads(marker.read_bytes().splitlines()[0])["dispatch_id"]
    armed_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "11111111-1111-4111-8111-111111111111",
    )
    current_boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "22222222-2222-4222-8222-222222222222",
    )
    armed = R._arm_post_reboot_recovery(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        boot_identity_provider=lambda: armed_boot,
    )
    receipt_name = R._safe_release_recovery_receipt_name(
        marker.name, dispatch_id
    )

    with monkeypatch.context() as fault:
        state = _inject_safe_release_sidecar_boundary(
            fault,
            root=root,
            final_name=intent.name,
            schema=R.DISPATCH_RELEASE_INTENT_SCHEMA,
            boundary=boundary,
            preparing_name=preparing.name,
        )
        if boundary == "short_write":
            outcome = R._recover_post_reboot_quarantine(
                fixture["item"],
                confirm_dispatch_id=dispatch_id,
                confirm_recovery_nonce=armed["recovery_nonce"],
                boot_identity_provider=lambda: current_boot,
            )
        else:
            try:
                outcome = R._recover_post_reboot_quarantine(
                    fixture["item"],
                    confirm_dispatch_id=dispatch_id,
                    confirm_recovery_nonce=armed["recovery_nonce"],
                    boot_identity_provider=lambda: current_boot,
                )
            except (R.CampaignPlanError, OSError):
                assert state["injected"] is True
                assert marker.is_file() and capsule.is_file()
                outcome = R._recover_post_reboot_quarantine(
                    fixture["item"],
                    confirm_dispatch_id=dispatch_id,
                    confirm_recovery_nonce=armed["recovery_nonce"],
                    boot_identity_provider=lambda: current_boot,
                )
            else:
                # Some reported-failure boundaries are reconciled by the
                # encompassing authenticated recovery call itself.  They are
                # still required to have executed the injected boundary.
                assert state["injected"] is True

    assert state["target_seen"] is True
    assert outcome["result"] == "tainted_noncounting"
    assert not marker.exists() and not intent.exists()
    assert not preparing.exists() and not capsule.exists()
    assert (root / receipt_name).is_file()
    assert sum(
        row.get("event") == "codex_dispatch_release_authorized"
        for row in R.Guard.read_ledger(fixture["ledger"])
    ) == 1
    sealed = fixture["ledger"].read_bytes()
    replay = R._recover_post_reboot_quarantine(
        fixture["item"],
        confirm_dispatch_id=dispatch_id,
        confirm_recovery_nonce=armed["recovery_nonce"],
        boot_identity_provider=lambda: current_boot,
    )
    assert replay["operator_recovery"] == (
        "post_reboot_safe_release_already_completed"
    )
    assert fixture["ledger"].read_bytes() == sealed


def _safe_release_arm_fixture(tmp_path, monkeypatch):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, _intent, preparing, capsule = (
        _leave_complete_release_preparing(fixture, tmp_path, monkeypatch)
    )
    dispatch_id = json.loads(marker.read_bytes().splitlines()[0])["dispatch_id"]
    boot = R.RebootRecovery.BootIdentity(
        "linux_proc_boot_id",
        "11111111-1111-4111-8111-111111111111",
    )
    arm_name = R._safe_release_recovery_arm_name(marker.name)
    return {
        "fixture": fixture,
        "root": root,
        "marker": marker,
        "wal": preparing,
        "capsule": capsule,
        "dispatch_id": dispatch_id,
        "boot": boot,
        "arm_name": arm_name,
    }


def _arm_safe_release(sidecar):
    return R._arm_post_reboot_recovery(
        sidecar["fixture"]["item"],
        confirm_dispatch_id=sidecar["dispatch_id"],
        boot_identity_provider=lambda: sidecar["boot"],
    )


def test_safe_release_partial_preparing_cleanup_preserves_dispatch(
    tmp_path, monkeypatch
):
    sidecar = _safe_release_arm_fixture(tmp_path, monkeypatch)
    root = sidecar["root"]
    preparing_name = R._durable_recovery_record_preparing_name(
        sidecar["arm_name"]
    )
    preparing = root / preparing_name
    preparing.write_bytes(b'{"partial":')
    os.chmod(preparing, 0o600)
    partial_identity = (
        preparing.stat().st_dev,
        preparing.stat().st_ino,
    )
    stable = (
        _release_test_file_state(sidecar["marker"]),
        _release_test_file_state(sidecar["wal"]),
        _release_test_file_state(sidecar["capsule"]),
        sidecar["fixture"]["ledger"].read_bytes(),
    )

    assert _arm_safe_release(sidecar)["result"] == (
        "post_reboot_safe_release_armed"
    )

    assert not preparing.exists()
    final = root / sidecar["arm_name"]
    assert final.is_file()
    assert (final.stat().st_dev, final.stat().st_ino) != partial_identity
    assert (
        _release_test_file_state(sidecar["marker"]),
        _release_test_file_state(sidecar["wal"]),
        _release_test_file_state(sidecar["capsule"]),
        sidecar["fixture"]["ledger"].read_bytes(),
    ) == stable


def test_safe_release_malformed_final_is_preserved_fail_closed(
    tmp_path, monkeypatch
):
    sidecar = _safe_release_arm_fixture(tmp_path, monkeypatch)
    final = sidecar["root"] / sidecar["arm_name"]
    final.write_bytes(b'{"malformed":true}\ntrailing')
    os.chmod(final, 0o600)
    final_before = _release_test_file_state(final)
    stable = (
        _release_test_file_state(sidecar["marker"]),
        _release_test_file_state(sidecar["wal"]),
        _release_test_file_state(sidecar["capsule"]),
        sidecar["fixture"]["ledger"].read_bytes(),
    )

    with pytest.raises(R.CampaignPlanError):
        _arm_safe_release(sidecar)

    assert _release_test_file_state(final) == final_before
    assert (
        _release_test_file_state(sidecar["marker"]),
        _release_test_file_state(sidecar["wal"]),
        _release_test_file_state(sidecar["capsule"]),
        sidecar["fixture"]["ledger"].read_bytes(),
    ) == stable


@pytest.mark.parametrize("custody", ("symlink", "hardlink", "mode"))
def test_safe_release_unsafe_preparing_is_preserved_fail_closed(
    tmp_path, monkeypatch, custody
):
    sidecar = _safe_release_arm_fixture(tmp_path, monkeypatch)
    preparing = sidecar["root"] / (
        R._durable_recovery_record_preparing_name(sidecar["arm_name"])
    )
    outside = tmp_path / f"unsafe-{custody}"
    outside.write_bytes(b'{"untrusted":true}\n')
    os.chmod(outside, 0o600)
    if custody == "symlink":
        preparing.symlink_to(outside)
    elif custody == "hardlink":
        os.link(outside, preparing)
    else:
        preparing.write_bytes(b'{"untrusted":true}\n')
        os.chmod(preparing, 0o644)
    outside_before = outside.read_bytes()

    with pytest.raises(R.CampaignPlanError):
        _arm_safe_release(sidecar)

    assert os.path.lexists(preparing)
    assert outside.read_bytes() == outside_before


@pytest.mark.parametrize("record_kind", ("arm", "receipt"))
def test_safe_release_installed_record_refsyncs_then_strictly_rereads(
    tmp_path, monkeypatch, record_kind
):
    sidecar = _safe_release_arm_fixture(tmp_path, monkeypatch)
    armed = _arm_safe_release(sidecar)
    if record_kind == "arm":
        final_name = sidecar["arm_name"]
        action = lambda: _arm_safe_release(sidecar)
    else:
        current_boot = R.RebootRecovery.BootIdentity(
            "linux_proc_boot_id",
            "22222222-2222-4222-8222-222222222222",
        )
        R._recover_post_reboot_quarantine(
            sidecar["fixture"]["item"],
            confirm_dispatch_id=sidecar["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: current_boot,
        )
        final_name = R._safe_release_recovery_receipt_name(
            sidecar["marker"].name, sidecar["dispatch_id"]
        )
        action = lambda: R._recover_post_reboot_quarantine(
            sidecar["fixture"]["item"],
            confirm_dispatch_id=sidecar["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: current_boot,
        )
    final = sidecar["root"] / final_name
    final_stat = final.stat(follow_symlinks=False)
    final_identity = (final_stat.st_dev, final_stat.st_ino)
    root_stat = sidecar["root"].stat(follow_symlinks=False)
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    real_fsync = R.os.fsync
    real_pread = R.os.pread
    events = []

    def observe_fsync(descriptor):
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        if identity == final_identity:
            events.append("file_fsync")
        elif identity == root_identity:
            events.append("root_fsync")
        return real_fsync(descriptor)

    def observe_pread(descriptor, count, offset):
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) == final_identity:
            events.append("strict_read")
        return real_pread(descriptor, count, offset)

    with monkeypatch.context() as observation:
        observation.setattr(R.os, "fsync", observe_fsync)
        observation.setattr(R.os, "pread", observe_pread)
        action()

    first_file = events.index("file_fsync")
    first_root = events.index("root_fsync", first_file + 1)
    first_read = events.index("strict_read", first_root + 1)
    assert first_file < first_root < first_read


def test_safe_release_installed_arm_rejects_interposed_inode_replacement(
    tmp_path, monkeypatch
):
    sidecar = _safe_release_arm_fixture(tmp_path, monkeypatch)
    _arm_safe_release(sidecar)
    final = sidecar["root"] / sidecar["arm_name"]
    original = final.stat(follow_symlinks=False)
    original_identity = (original.st_dev, original.st_ino)
    payload = final.read_bytes()
    real_pread = R.os.pread
    injected = False

    def replace_during_read(descriptor, count, offset):
        nonlocal injected
        opened = os.fstat(descriptor)
        if not injected and (opened.st_dev, opened.st_ino) == original_identity:
            replacement = sidecar["root"] / ".replacement-arm"
            replacement.write_bytes(payload)
            os.chmod(replacement, 0o600)
            os.replace(replacement, final)
            injected = True
        return real_pread(descriptor, count, offset)

    with monkeypatch.context() as fault:
        fault.setattr(R.os, "pread", replace_during_read)
        with pytest.raises(R.CampaignPlanError, match="changed during"):
            _arm_safe_release(sidecar)
    assert injected
    assert final.read_bytes() == payload
    assert (final.stat().st_dev, final.stat().st_ino) != original_identity


@pytest.mark.parametrize("record_kind", ("arm", "receipt"))
def test_safe_release_installed_record_rejects_quarantine_root_rebind(
    tmp_path, monkeypatch, record_kind
):
    sidecar = _safe_release_arm_fixture(tmp_path, monkeypatch)
    armed = _arm_safe_release(sidecar)
    root = sidecar["root"]
    if record_kind == "arm":
        final_name = sidecar["arm_name"]
        action = lambda: _arm_safe_release(sidecar)
    else:
        current_boot = R.RebootRecovery.BootIdentity(
            "linux_proc_boot_id",
            "22222222-2222-4222-8222-222222222222",
        )
        R._recover_post_reboot_quarantine(
            sidecar["fixture"]["item"],
            confirm_dispatch_id=sidecar["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: current_boot,
        )
        final_name = R._safe_release_recovery_receipt_name(
            sidecar["marker"].name, sidecar["dispatch_id"]
        )
        action = lambda: R._recover_post_reboot_quarantine(
            sidecar["fixture"]["item"],
            confirm_dispatch_id=sidecar["dispatch_id"],
            confirm_recovery_nonce=armed["recovery_nonce"],
            boot_identity_provider=lambda: current_boot,
        )
    final = root / final_name
    final_stat = final.stat(follow_symlinks=False)
    final_identity = (final_stat.st_dev, final_stat.st_ino)
    displaced = tmp_path / "displaced-safe-release-quarantine"
    real_pread = R.os.pread
    injected = False

    def rebind_root_during_read(descriptor, count, offset):
        nonlocal injected
        opened = os.fstat(descriptor)
        if not injected and (opened.st_dev, opened.st_ino) == final_identity:
            root.rename(displaced)
            root.mkdir(mode=0o700)
            os.chmod(root, 0o700)
            injected = True
        return real_pread(descriptor, count, offset)

    with monkeypatch.context() as fault:
        fault.setattr(R.os, "pread", rebind_root_during_read)
        with pytest.raises(R.CampaignPlanError, match="root identity changed"):
            action()

    assert injected
    assert (displaced / final_name).is_file()
    if record_kind == "arm":
        assert sidecar["marker"].name in os.listdir(displaced)


def test_safe_release_conflicting_tail_cannot_be_armed(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    _root, marker, intent, capsule, _captured = (
        _leave_installed_release_authority_tail(
            fixture, tmp_path, monkeypatch, conflicting=True
        )
    )
    dispatch_id = json.loads(marker.read_bytes().splitlines()[0])["dispatch_id"]
    states = tuple(
        _release_test_file_state(path) for path in (marker, intent, capsule)
    )
    ledger_before = fixture["ledger"].read_bytes()
    with pytest.raises(R.CampaignPlanError, match="conflicting"):
        R._arm_post_reboot_recovery(
            fixture["item"],
            confirm_dispatch_id=dispatch_id,
            boot_identity_provider=lambda: R.RebootRecovery.BootIdentity(
                "linux_proc_boot_id",
                "11111111-1111-4111-8111-111111111111",
            ),
        )
    assert fixture["ledger"].read_bytes() == ledger_before
    assert tuple(
        _release_test_file_state(path) for path in (marker, intent, capsule)
    ) == states


def test_forged_full_release_row_cannot_retire_unquiesced_one_exec_marker(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)

    def unquiesced_child(*_args, **_kwargs):
        R.Guard.append_ledger(fixture["record"], fixture["ledger"])
        return R.GuardedChildResult(
            returncode=1,
            workspace=fixture["workspace"].name,
            transcript=fixture["record"]["transcript"],
            workspace_identity=(
                fixture["workspace"].stat().st_dev,
                fixture["workspace"].stat().st_ino,
            ),
            protected_identity=(
                fixture["protected"].stat().st_dev,
                fixture["protected"].stat().st_ino,
            ),
            descendant_quiescence_unproven=True,
            process_tree_quiesced=True,
        )

    monkeypatch.setattr(R, "_run_guarded_child", unquiesced_child)
    with pytest.raises(R.UnquiescedChildError):
        R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )
    root, marker, intent, _preparing = _release_test_paths(tmp_path)
    marker_payload = marker.read_bytes()
    armed = json.loads(marker_payload.splitlines()[0])
    assert len(marker_payload.splitlines()) == 2
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        "codex_exec"
    ]
    capsule = _release_test_capsule(marker)
    ledger_raw = fixture["ledger"].read_bytes()
    ledger_metadata = fixture["ledger"].stat(follow_symlinks=False)
    terminal_result = {
        "game": fixture["item"]["game"],
        "target_level": fixture["item"]["target_level"],
        "reached": fixture["item"]["reached"],
        "result": "not_solved",
        "retry_complexity_n": fixture["item"]["retry_complexity_n"],
    }
    base_authority = {
        "schema": "scheduler_dispatch_release_authority_v1",
        "kind": "ordinary_safe_terminal_v1",
        "projected_item_sha256": armed["projected_item_sha256"],
        "game": fixture["item"]["game"],
        "target_level": fixture["item"]["target_level"],
        "retry_complexity_n": fixture["item"]["retry_complexity_n"],
        **{
            field: fixture["item"][field]
            for field in (
                *R.Status.FRONTIER_BINDING_FIELDS,
                "reached",
                "parent_action_count",
            )
        },
        "ledger": str(fixture["ledger"]),
        "ledger_parent_identity": [
            fixture["ledger"].parent.stat().st_dev,
            fixture["ledger"].parent.stat().st_ino,
        ],
        "ledger_file_identity": [
            ledger_metadata.st_dev,
            ledger_metadata.st_ino,
        ],
        "ledger_prefix_bytes": len(ledger_raw),
        "ledger_prefix_sha256": hashlib.sha256(ledger_raw).hexdigest(),
        "dispatch_ledger_prefix_bytes": armed["ledger_prefix_bytes"],
        "dispatch_ledger_prefix_sha256": armed["ledger_prefix_sha256"],
        "terminal_event": "codex_exec",
        "terminal_record_sha256": R._recovery_record_sha256(
            fixture["record"]
        ),
        "terminal_result": terminal_result,
        "terminal_result_sha256": hashlib.sha256(json.dumps(
            terminal_result,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()).hexdigest(),
    }
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    descriptor = os.open(
        intent.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=root_fd,
    )
    try:
        os.fchmod(descriptor, 0o600)
        intent_metadata = os.fstat(descriptor)
        capsule_metadata = capsule.stat(follow_symlinks=False)
        capsule_payload = capsule.read_bytes()
        record = {
            "schema": R.DISPATCH_RELEASE_INTENT_SCHEMA,
            "event": "dispatch_release_intent",
            "dispatch_id": armed["dispatch_id"],
            "intent_name": intent.name,
            "intent_identity": [intent_metadata.st_dev, intent_metadata.st_ino],
            "quarantine_root_identity": [
                root.stat().st_dev,
                root.stat().st_ino,
            ],
            "marker_name": marker.name,
            "marker_identity": [marker.stat().st_dev, marker.stat().st_ino],
            "marker_bytes": len(marker_payload),
            "marker_sha256": hashlib.sha256(marker_payload).hexdigest(),
            "capsule_name": capsule.name,
            "capsule_identity": [capsule_metadata.st_dev, capsule_metadata.st_ino],
            "capsule_present_at_intent": True,
            "capsule_bytes": len(capsule_payload),
            "capsule_sha256": hashlib.sha256(capsule_payload).hexdigest(),
            "release_authority": dict(base_authority),
        }
        release_nonce = "a" * 64
        core = R._dispatch_release_intent_core_sha256(record, base_authority)
        authority_row = {
            "event": "codex_dispatch_release_authorized",
            "schema": "scheduler_dispatch_release_authorized_v1",
            "recorded_at": "2026-08-07T00:00:00+00:00",
            "dispatch_id": armed["dispatch_id"],
            "release_nonce": release_nonce,
            "intent_name": intent.name,
            "intent_identity": record["intent_identity"],
            "intent_core_sha256": core,
            "projected_item_sha256": base_authority["projected_item_sha256"],
            "game": base_authority["game"],
            "target_level": base_authority["target_level"],
            "retry_complexity_n": base_authority["retry_complexity_n"],
            "reached": base_authority["reached"],
            "parent_action_count": base_authority["parent_action_count"],
            "terminal_kind": base_authority["kind"],
            "terminal_event": base_authority["terminal_event"],
            "terminal_record_sha256": base_authority["terminal_record_sha256"],
            "ledger": base_authority["ledger"],
            "ledger_parent_identity": base_authority["ledger_parent_identity"],
            "ledger_file_identity": base_authority["ledger_file_identity"],
            "ledger_prefix_bytes": base_authority["ledger_prefix_bytes"],
            "ledger_prefix_sha256": base_authority["ledger_prefix_sha256"],
            **{
                field: base_authority[field]
                for field in R.Status.FRONTIER_BINDING_FIELDS
            },
        }
        record["release_authority"] = {
            **base_authority,
            "release_nonce": release_nonce,
            "intent_core_sha256": core,
            "authority_record": authority_row,
        }
        R._validate_dispatch_release_intent_record(
            record, marker_name=marker.name
        )
        payload = R.RebootRecovery.canonical_json_line(record)
        assert os.write(descriptor, payload) == len(payload)
        os.fsync(descriptor)
        os.fsync(root_fd)
    finally:
        os.close(descriptor)
        os.close(root_fd)
    with fixture["ledger"].open("ab", buffering=0) as stream:
        stream.write(R.RebootRecovery.canonical_json_line(authority_row))
        os.fsync(stream.fileno())
    states = tuple(
        _release_test_file_state(path) for path in (marker, capsule, intent)
    )
    ledger_before = fixture["ledger"].read_bytes()

    with pytest.raises(R.CampaignPlanError):
        R._assert_no_dispatch_quarantine(fixture["item"])

    assert fixture["ledger"].read_bytes() == ledger_before
    assert tuple(
        _release_test_file_state(path) for path in (marker, capsule, intent)
    ) == states


@pytest.mark.parametrize("tail", ("exact_prefix", "conflict"))
def test_release_authority_incomplete_tail_always_fails_closed(
    tmp_path, monkeypatch, tail
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, marker, intent, _preparing = _release_test_paths(tmp_path)
    captured = {}

    def crash_during_authority_append(
        item,
        root_fd,
        record,
        intent_identity,
        *,
        allow_new_authority_append=False,
    ):
        del (
            item,
            root_fd,
            intent_identity,
            allow_new_authority_append,
        )
        authority = record["release_authority"]
        line = R.RebootRecovery.canonical_json_line(
            authority["authority_record"]
        )
        ledger = Path(authority["ledger"])
        prefix = ledger.read_bytes()
        assert len(prefix) == authority["ledger_prefix_bytes"]
        fragment = (
            line[: max(1, len(line) // 2)]
            if tail == "exact_prefix"
            else b"!conflicting-release-authority-tail"
        )
        with ledger.open("ab", buffering=0) as stream:
            stream.write(fragment)
            os.fsync(stream.fileno())
        captured.update({
            "prefix": prefix,
            "line": line,
            "fragment": fragment,
        })
        raise OSError(errno.EIO, "synthetic authority append crash")

    with monkeypatch.context() as fault:
        fault.setattr(
            R,
            "_ensure_dispatch_release_authority_row",
            crash_during_authority_append,
        )
        with pytest.raises(R.CampaignPlanError) as failure:
            R._run_item(
                fixture["plan"],
                fixture["item"],
                allowance=fixture["allowance"],
            )

    assert captured, repr(failure.value)
    assert marker.is_file()
    assert intent.is_file()
    capsule = _release_test_capsule(marker)
    marker_before = _release_test_file_state(marker)
    capsule_before = _release_test_file_state(capsule)
    intent_before = _release_test_file_state(intent)
    with pytest.raises(R.CampaignPlanError):
        R._assert_no_dispatch_quarantine(fixture["item"])
    assert fixture["ledger"].read_bytes() == (
        captured["prefix"] + captured["fragment"]
    )
    assert _release_test_file_state(marker) == marker_before
    assert _release_test_file_state(capsule) == capsule_before
    assert _release_test_file_state(intent) == intent_before


@pytest.mark.parametrize("target", ("capsule", "marker", "intent"))
def test_release_unlink_before_root_fsync_reconciles_idempotently(
    tmp_path, monkeypatch, target
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root, _marker, _intent, _preparing = _release_test_paths(tmp_path)
    _release_test_crash_after_unlink(
        fixture, monkeypatch, target=target
    )

    R._assert_no_dispatch_quarantine(fixture["item"])

    assert _release_test_residue(root) == []
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [
        row["event"] for row in rows
    ].count("codex_dispatch_release_authorized") == 1


def test_not_solved_retry_advance_reconciles_prior_release(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    state = _configure_clean_release(
        fixture, monkeypatch, reached_after=0
    )
    root, _marker, _intent, _preparing = _release_test_paths(tmp_path)
    _release_test_crash_after_unlink(
        fixture, monkeypatch, target="marker"
    )
    assert state["child_calls"] == 1
    next_item = copy.deepcopy(fixture["item"])
    next_item["retry_complexity_n"] = 1

    def forbidden_child(*_args, **_kwargs):
        raise AssertionError("next dispatch started before release reconcile")

    with monkeypatch.context() as next_dispatch:
        next_dispatch.setattr(
            R, "validate_item", lambda item, plan=None: tuple(item["argv"])
        )
        next_dispatch.setattr(
            R, "validate_inventory_item", lambda *_args, **_kwargs: None
        )
        next_dispatch.setattr(R, "active_workspace_lock", lambda _game: None)
        next_dispatch.setattr(R, "_run_guarded_child", forbidden_child)
        outcome = R._run_item(
            fixture["plan"],
            next_item,
            allowance=SimpleNamespace(
                remaining_percent=0, window_name="weekly"
            ),
        )

    assert outcome["result"] == "reserve_stop"
    assert _release_test_residue(root) == []
    assert state["child_calls"] == 1


def test_solved_rerun_reconciles_release_before_already_solved(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    state = _configure_clean_release(
        fixture, monkeypatch, reached_after=1
    )
    root, _marker, _intent, _preparing = _release_test_paths(tmp_path)
    _release_test_crash_after_unlink(
        fixture, monkeypatch, target="marker"
    )
    assert state == {"reached": 1, "child_calls": 1}

    with monkeypatch.context() as rerun:
        rerun.setattr(
            R, "validate_inventory_item", lambda *_args, **_kwargs: None
        )
        outcome = R._run_item(
            fixture["plan"],
            fixture["item"],
            allowance=fixture["allowance"],
        )

    assert outcome["result"] == "already_solved"
    assert _release_test_residue(root) == []
    assert state == {"reached": 1, "child_calls": 1}


@pytest.mark.parametrize("category", ["successful_candidate_wip", "discarded_wip"])
def test_taint_gate_rejects_wip_hits_even_when_canonical_verdict_passes(
    tmp_path, monkeypatch, category
):
    monkeypatch.setattr(R, "HERE", tmp_path)
    report = {
        "automated_verdict": "PASS",
        "successful_candidate_wip": {"hits": []},
        "discarded_wip": {"hits": []},
    }
    report[category]["hits"] = [{"attempt": "forensic"}]
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(report)
        ),
    )

    with pytest.raises(R.CampaignPlanError, match="forensic WIP taint"):
        R._taint_gate()


def _historical_single_transcript_fixture(tmp_path, monkeypatch, *, tainted=True):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    diagnostics = fixture["record"].pop("diagnostics")
    fixture["record"].pop("protected_diagnostics_status")
    fixture["record"].pop("protected_diagnostics_sha256")
    (fixture["protected"] / diagnostics).unlink()
    fixture["exact_lock"].unlink()
    in_workspace_lock = fixture["workspace"] / ".orchestrate.lock"
    in_workspace_lock.write_text("")
    (fixture["workspace"] / "gkm_try.py").write_text("pass\n")
    worktree = tmp_path / "historical"
    module_root = worktree / "arc" / "crack_lab"
    module_root.mkdir(parents=True)
    historical_source = b"TESTER = 'pass\\n'\n"
    (module_root / "gkm_legs.py").write_bytes(historical_source)
    fixture["item"]["historical_runner"] = {
        "evidence_schema": "sealed_transcript_only_v1",
        "lock_schema": "in_workspace_v1",
        "worktree": str(worktree),
        "source_sha256": hashlib.sha256(historical_source).hexdigest(),
    }
    if not tainted:
        clean = (
            json.dumps({
                "type": "thread.started", "thread_id": "tainted-thread",
            })
            + "\n"
            + json.dumps({"type": "turn.completed", "usage": {}})
            + "\n"
        ).encode()
        transcript = fixture["protected"] / fixture["record"]["transcript"]
        transcript.write_bytes(clean)
        fixture["record"]["protected_transcript_sha256"] = hashlib.sha256(
            clean
        ).hexdigest()
    fixture["in_workspace_lock"] = in_workspace_lock
    return fixture


def test_historical_single_transcript_taint_is_cleaned_without_hash_lock(
    tmp_path, monkeypatch
):
    fixture = _historical_single_transcript_fixture(
        tmp_path, monkeypatch, tainted=True
    )
    before = R.Guard.read_ledger(fixture["ledger"])
    R.Guard.append_ledger(fixture["record"], fixture["ledger"])

    result = R._recover_confirmed_taint(
        fixture["item"],
        ledger=fixture["ledger"],
        ledger_before=before,
        reached_before=0,
        wip_rollback_before=R._capture_wip_rollback(fixture["item"]),
        child_returncode=1,
        process_tree_quiesced=True,
    )

    assert result["result"] == "tainted_noncounting"
    assert result["retry_complexity_n"] == 0
    assert not fixture["workspace"].exists()
    assert not fixture["protected"].exists()
    assert fixture["sibling"].is_dir()
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        "codex_exec",
        "codex_exec_classification_correction",
        "codex_taint_cleanup_completed",
    ]
    assert "diagnostics" not in rows[1]
    assert "diagnostics" not in rows[2]


def test_clean_historical_nonzero_is_not_misclassified_by_old_scaffold(
    tmp_path, monkeypatch
):
    fixture = _historical_single_transcript_fixture(
        tmp_path, monkeypatch, tainted=False
    )
    before = R.Guard.read_ledger(fixture["ledger"])
    R.Guard.append_ledger(fixture["record"], fixture["ledger"])

    with pytest.raises(
        R.CampaignPlanError, match="no independently confirmed"
    ):
        R._recover_confirmed_taint(
            fixture["item"],
            ledger=fixture["ledger"],
            ledger_before=before,
            reached_before=0,
            wip_rollback_before=R._capture_wip_rollback(fixture["item"]),
            child_returncode=1,
            process_tree_quiesced=True,
        )

    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert R.Guard.read_ledger(fixture["ledger"]) == [fixture["record"]]


def test_plan_runner_receipt_projects_initial_and_adaptive_commands(
    tmp_path, monkeypatch
):
    worktree = tmp_path / "submitted"
    worktree.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    artifacts = tmp_path / "agent_solutions"
    artifacts.mkdir()
    ledger = tmp_path / "usage.jsonl"
    source_sha = "b" * 64
    head = "a" * 40
    monkeypatch.setattr(R, "HERE", tmp_path)
    monkeypatch.setattr(R.Legs, "SCRATCH", str(scratch))
    monkeypatch.setattr(R.Guard, "DEFAULT_LEDGER", ledger)
    monkeypatch.setattr(R, "PINNED_HISTORICAL_RUNNERS", {
        source_sha: {
            "head_commit": head,
            "evidence_schema": "sealed_transcript_only_v1",
            "lock_schema": "in_workspace_v1",
        }
    })
    receipt = {
        "schema": R.RUNNER_RECEIPT_SCHEMA,
        "worktree": str(worktree),
        "cwd": str(worktree),
        "interpreter": str(Path(sys.executable).absolute()),
        "head_commit": head,
        "source_sha256": source_sha,
        "artifacts_root": str(artifacts),
        "scratch_root": str(scratch),
        "ledger": str(ledger),
        "evidence_schema": "sealed_transcript_only_v1",
        "lock_schema": "in_workspace_v1",
    }
    plan = {"runner_receipt": receipt}

    initial = R._project_runner_receipt(plan, _item())
    adaptive = R._project_runner_receipt(plan, copy.deepcopy(_item()))

    for projected in (initial, adaptive):
        assert projected["historical_runner"] == receipt
        assert projected["argv"][:3] == [
            receipt["interpreter"],
            "-u",
            str(worktree / "arc" / "crack_lab" / "gkm_legs.py"),
        ]
        assert f"--artifacts-root={artifacts}" in projected["argv"]
        assert f"--codex-ledger={ledger}" in projected["argv"]


def test_pre_dispatch_taint_gate_stops_before_child(tmp_path, monkeypatch):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    child_called = False

    def forbidden_child(*_args, **_kwargs):
        nonlocal child_called
        child_called = True
        raise AssertionError("child launched after failed pre-dispatch gate")

    monkeypatch.setattr(R, "_run_guarded_child", forbidden_child)
    monkeypatch.setattr(
        R,
        "_taint_gate",
        lambda: (_ for _ in ()).throw(
            R.CampaignPlanError("pre-dispatch forensic WIP taint")
        ),
    )

    with pytest.raises(R.CampaignPlanError, match="pre-dispatch"):
        R._run_item(
            fixture["plan"], fixture["item"], allowance=fixture["allowance"]
        )
    assert child_called is False
    assert R.Guard.read_ledger(fixture["ledger"]) == []


def test_nested_wip_mutation_blocks_taint_cleanup(tmp_path, monkeypatch):
    nested = (
        tmp_path / "agent_solutions" / "ar25_legs" / "wip_context"
        / "level_01" / "old_attempt" / "files" / "probe.py"
    )

    def mutate(_root):
        nested.write_text("changed\n")

    fixture = _taint_dispatch_fixture(
        tmp_path, monkeypatch, child_mutation=mutate
    )
    nested.parent.mkdir(parents=True)
    nested.write_text("original\n")

    with pytest.raises(R.CampaignPlanError, match="preexisting WIP evidence"):
        R._run_item(
            fixture["plan"], fixture["item"], allowance=fixture["allowance"]
        )
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        "codex_exec",
        "codex_exec_classification_correction",
    ]


def test_incomplete_prior_taint_cleanup_blocks_new_dispatch(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    R.Guard.append_ledger({
        "event": "codex_exec_classification_correction",
        "classification_authority": "scheduler_exact_generation_taint_scan_v1",
        "thread_id": "old-thread",
        "transcript": "codex_turn_old.jsonl",
        "workspace": "gkm_legs_ws_ar25_old_deadbeef",
        "failure_class": "taint",
    }, fixture["ledger"])

    with pytest.raises(R.CampaignPlanError, match="lacks cleanup completion"):
        R._run_item(
            fixture["plan"], fixture["item"], allowance=fixture["allowance"]
        )
    assert fixture["workspace"].is_dir()
    assert len(R.Guard.read_ledger(fixture["ledger"])) == 1


def _historical_watchdog_fixture(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    protected_root = scratch / ".proposer_transcripts"
    protected_root.mkdir(parents=True)
    item = copy.deepcopy(_item())
    tag = "arc_agi3_n0_fresh_frontier"
    item["argv"].append(f"--tag={tag}")
    worktree = tmp_path / "submitted"
    item["historical_runner"] = {
        "evidence_schema": "sealed_transcript_only_v1",
        "worktree": str(worktree),
    }
    name = f"gkm_legs_ws_ar25_{tag}_cafefeed"
    workspace = scratch / name
    protected = protected_root / name
    transcript = protected / "codex_turn_live_ar25_L1_propose.jsonl"

    monkeypatch.setattr(R.Legs, "SCRATCH", str(scratch))
    monkeypatch.setattr(R, "_workspace_lock_is_active", lambda _path: True)
    monkeypatch.setattr(R, "_historical_tester_scaffolds", lambda *_args: {})
    return item, workspace, protected, transcript


def test_guarded_child_contains_launch_return_handoff_failure(
    tmp_path, monkeypatch
):
    item, _workspace, _protected, _transcript = (
        _historical_watchdog_fixture(tmp_path, monkeypatch)
    )
    trees = []

    class HandoffTree:
        pid = 12344

        def __init__(self):
            self.sealed = False
            self.seal_calls = []
            trees.append(self)

        def seal(self, *, stop_requested, grace_seconds):
            self.seal_calls.append((stop_requested, grace_seconds))
            self.sealed = True
            return SimpleNamespace(
                returncode=-9,
                detached_processes_proven_absent=True,
            )

    def fail_after_tree_publish(
        _argv, *, cwd, environment, ownership=None
    ):
        del cwd, environment
        tree = HandoffTree()
        assert ownership is not None
        ownership[0] = tree
        raise KeyboardInterrupt("injected tree-return handoff failure")

    monkeypatch.setattr(
        R.Contiguous.ScopedProcessTree,
        "launch",
        fail_after_tree_publish,
    )
    with pytest.raises(KeyboardInterrupt, match="handoff failure"):
        R._run_guarded_child(
            item, ["exact-legacy-child"], cwd=tmp_path, env={}
        )

    assert len(trees) == 1
    assert trees[0].sealed is True
    assert trees[0].seal_calls == [
        (True, R.EXACT_CHILD_TERMINATE_SECONDS)
    ]


def test_guarded_child_accepts_legacy_banner_and_split_clean_json_growth(
    tmp_path, monkeypatch
):
    item, workspace, protected, transcript = _historical_watchdog_fixture(
        tmp_path, monkeypatch
    )
    phase = {"sleeps": 0}
    spawned = []

    class FakeScopedTree:
        pid = 12345

        def __init__(self):
            self.sealed = False
            self.seal_calls = []
            workspace.mkdir()
            protected.mkdir()
            transcript.write_bytes(
                R.Boundary.HISTORICAL_STDIN_DIAGNOSTIC
                + b'{"type":"thread.st'
            )
            spawned.append(self)

        def observe_exit(self):
            return phase["sleeps"] >= 2

        def seal(self, *, stop_requested, grace_seconds):
            self.seal_calls.append((stop_requested, grace_seconds))
            self.sealed = True
            return SimpleNamespace(returncode=0)

    def advance(_seconds):
        if phase["sleeps"] == 0:
            with transcript.open("ab") as stream:
                stream.write(
                    b'arted","thread_id":"clean-thread"}\n'
                    b'{"type":"turn.com'
                )
        elif phase["sleeps"] == 1:
            with transcript.open("ab") as stream:
                stream.write(b'pleted","usage":{}}\n')
        phase["sleeps"] += 1

    monkeypatch.setattr(
        R, "_launch_exact_child", lambda *_args, **_kwargs: FakeScopedTree()
    )
    monkeypatch.setattr(R.time, "sleep", advance)

    result = R._run_guarded_child(
        item,
        ["exact-legacy-child"],
        cwd=tmp_path,
        env={"GKM_SCRATCH": str(tmp_path / "scratch")},
    )

    assert result == R.GuardedChildResult(
        returncode=0,
        taint_reason=None,
        workspace=workspace.name,
        transcript=transcript.name,
        workspace_identity=(workspace.stat().st_dev, workspace.stat().st_ino),
        protected_identity=(protected.stat().st_dev, protected.stat().st_ino),
        process_tree_quiesced=True,
    )
    assert phase["sleeps"] == 2
    assert len(spawned) == 1
    assert spawned[0].seal_calls == [(False, 0)]


def test_guarded_child_terminates_exact_child_on_current_policy_taint_even_rc0(
    tmp_path, monkeypatch
):
    item, workspace, protected, transcript = _historical_watchdog_fixture(
        tmp_path, monkeypatch
    )
    spawned = []
    appended = False

    class FakeScopedTree:
        pid = 12346

        def __init__(self):
            self.sealed = False
            self.seal_calls = []
            workspace.mkdir()
            protected.mkdir()
            transcript.write_bytes(
                R.Boundary.HISTORICAL_STDIN_DIAGNOSTIC
                + json.dumps({
                    "type": "thread.started", "thread_id": "tainted-thread",
                }).encode()
                + b"\n"
            )
            spawned.append(self)

        def observe_exit(self):
            return False

        def seal(self, *, stop_requested, grace_seconds):
            self.seal_calls.append((stop_requested, grace_seconds))
            self.sealed = True
            return SimpleNamespace(returncode=0)

    def append_forbidden(_seconds):
        nonlocal appended
        assert appended is False
        event = {
            "type": "item.completed",
            "item": {
                "id": "forbidden-find",
                "type": "command_execution",
                "command": "find . -type f",
                "aggregated_output": "",
            },
        }
        with transcript.open("ab") as stream:
            stream.write(json.dumps(event).encode() + b"\n")
        appended = True

    monkeypatch.setattr(
        R, "_launch_exact_child", lambda *_args, **_kwargs: FakeScopedTree()
    )
    monkeypatch.setattr(R.time, "sleep", append_forbidden)

    result = R._run_guarded_child(
        item, ["exact-legacy-child"], cwd=tmp_path, env={}
    )

    assert result.returncode == 0
    assert result.taint_reason is not None
    assert "shell_or_host_filesystem_escape" in result.taint_reason
    assert len(spawned) == 1
    assert result.process_tree_quiesced is True
    assert spawned[0].seal_calls == [
        (True, R.EXACT_CHILD_TERMINATE_SECONDS)
    ]


def test_guarded_child_preserves_first_live_taint_reason_at_terminal_scan(
    tmp_path, monkeypatch
):
    item, workspace, protected, transcript = _historical_watchdog_fixture(
        tmp_path, monkeypatch
    )

    class FakeScopedTree:
        pid = 12349

        def __init__(self):
            self.sealed = False
            workspace.mkdir()
            protected.mkdir()
            transcript.write_bytes(b"sealed synthetic transcript\n")

        def observe_exit(self):
            return False

        def seal(self, *, stop_requested, grace_seconds):
            assert stop_requested is True
            assert grace_seconds == R.EXACT_CHILD_TERMINATE_SECONDS
            self.sealed = True
            return SimpleNamespace(
                returncode=1,
                detached_processes_proven_absent=True,
            )

    class FakeFinding:
        code = "shell_or_host_filesystem_escape"

        def __init__(self, description):
            self.description = description

        def describe(self):
            return self.description

    class FakeMonitor:
        trusted_host_scaffolds = {}

        def __init__(self, *_args, **_kwargs):
            self.transcript_scans = 0

        def scan_workspace(self):
            return []

        def scan_transcript(self, _path, *, final):
            self.transcript_scans += 1
            if not final:
                return [FakeFinding("first live taint")]
            return [FakeFinding("later terminal taint")]

    monkeypatch.setattr(
        R, "_launch_exact_child", lambda *_args, **_kwargs: FakeScopedTree()
    )
    monkeypatch.setattr(R.Boundary, "LiveBoundaryMonitor", FakeMonitor)
    result = R._run_guarded_child(
        item, ["exact-legacy-child"], cwd=tmp_path, env={}
    )

    assert result.taint_reason == "first live taint"
    assert result.process_tree_quiesced is True
    assert result.detached_processes_proven_absent is True


def test_guarded_child_keeps_original_control_error_after_handled_sigterm(
    tmp_path, monkeypatch
):
    item, workspace, _protected, _transcript = _historical_watchdog_fixture(
        tmp_path, monkeypatch
    )
    sibling = workspace.with_name(f"{workspace.name}_second")

    class FakeScopedTree:
        pid = 12347

        def __init__(self):
            self.sealed = False
            workspace.mkdir()
            sibling.mkdir()

        def observe_exit(self):
            return False

        def seal(self, *, stop_requested, grace_seconds):
            assert stop_requested is True
            assert grace_seconds == R.EXACT_CHILD_TERMINATE_SECONDS
            self.sealed = True
            return SimpleNamespace(returncode=-2)

    monkeypatch.setattr(
        R, "_launch_exact_child", lambda *_args, **_kwargs: FakeScopedTree()
    )

    with pytest.raises(
        R.CampaignPlanError, match="multiple candidate workspaces"
    ):
        R._run_guarded_child(
            item, ["exact-legacy-child"], cwd=tmp_path, env={}
        )


def test_guarded_child_fails_closed_when_scoped_tree_cannot_be_sealed(
    tmp_path, monkeypatch
):
    item, workspace, _protected, _transcript = _historical_watchdog_fixture(
        tmp_path, monkeypatch
    )

    class UncontainedTree:
        pid = 12348
        sealed = False

        def __init__(self):
            workspace.mkdir()
            workspace.with_name(f"{workspace.name}_second").mkdir()

        def observe_exit(self):
            return False

        def seal(self, *, stop_requested, grace_seconds):
            del stop_requested, grace_seconds
            raise R.Contiguous.ScopedProcessContainmentError(
                "synthetic containment failure"
            )

    monkeypatch.setattr(
        R, "_launch_exact_child", lambda *_args, **_kwargs: UncontainedTree()
    )
    with pytest.raises(
        R.UnquiescedChildError, match="containment unproven"
    ):
        R._run_guarded_child(
            item, ["exact-legacy-child"], cwd=tmp_path, env={}
        )


def _wip_rollback_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "HERE", tmp_path)
    item = copy.deepcopy(_item())
    level = R._target_wip_level(item)
    old_attempt = level / "old_attempt"
    old_attempt.mkdir(parents=True)
    (old_attempt / "sealed.txt").write_text("untouched\n")
    latest = level / "latest.json"
    old_latest = b'{"attempt":"old_attempt","status":"clean"}\n'
    latest.write_bytes(old_latest)
    state = R._capture_wip_rollback(item)
    return item, level, latest, old_latest, state


def _write_authenticated_tainted_attempt(
    item, level, *, attempt, transcript_name, transcript_payload, update_latest
):
    attempt_dir = level / attempt
    files = attempt_dir / "files"
    files.mkdir(parents=True)
    (files / transcript_name).write_bytes(transcript_payload)
    metadata = {
        "attempt": attempt,
        "game": item["game"],
        "level": item["target_level"],
        "phase": "interrupted",
        "files": [transcript_name],
    }
    (attempt_dir / "metadata.json").write_text(json.dumps(metadata))
    if update_latest:
        (level / "latest.json").write_text(json.dumps({
            "attempt": attempt,
            "metadata": metadata,
        }))
    return attempt_dir


def test_tainted_wip_rollback_authenticates_one_attempt_and_restores_latest(
    tmp_path, monkeypatch
):
    item, level, latest, old_latest, state = _wip_rollback_fixture(
        tmp_path, monkeypatch
    )
    baseline = state.baseline_snapshot
    latest_inode = latest.stat().st_ino
    transcript_name = "codex_turn_tainted.jsonl"
    transcript_payload = b'{"type":"turn.completed","usage":{}}\n'
    attempt_dir = _write_authenticated_tainted_attempt(
        item,
        level,
        attempt="interrupted_deadbeef",
        transcript_name=transcript_name,
        transcript_payload=transcript_payload,
        update_latest=True,
    )

    R._rollback_tainted_wip(
        item,
        state,
        {"transcript": transcript_name},
        hashlib.sha256(transcript_payload).hexdigest(),
    )

    assert not attempt_dir.exists()
    assert latest.read_bytes() == old_latest
    assert latest.stat().st_ino == latest_inode
    assert (level / "old_attempt" / "sealed.txt").read_text() == "untouched\n"
    assert R._target_wip_snapshot(item) == baseline


def test_tainted_wip_rollback_restores_root_and_latest_metadata(
    tmp_path, monkeypatch
):
    item, level, latest, _old_latest, _initial_state = _wip_rollback_fixture(
        tmp_path, monkeypatch
    )
    root_attribute = "user.gkm_wip_root"
    latest_attribute = "user.gkm_wip_latest"
    R._portable_setxattr(level, root_attribute, b"sealed-root")
    R._portable_setxattr(latest, latest_attribute, b"sealed-latest")
    state = R._capture_wip_rollback(item)
    baseline_level = level.stat(follow_symlinks=False)
    baseline_latest = latest.stat(follow_symlinks=False)
    transcript_name = "codex_turn_tainted.jsonl"
    transcript_payload = b'{"type":"turn.completed","usage":{}}\n'
    _write_authenticated_tainted_attempt(
        item,
        level,
        attempt="interrupted_metadata",
        transcript_name=transcript_name,
        transcript_payload=transcript_payload,
        update_latest=True,
    )
    R._portable_setxattr(level, root_attribute, b"changed-root")
    R._portable_setxattr(level, "user.gkm_wip_extra", b"tainted")
    R._portable_setxattr(latest, latest_attribute, b"changed-latest")
    R._portable_setxattr(latest, "user.gkm_latest_extra", b"tainted")
    os.chmod(level, 0o700)
    os.chmod(latest, 0o600)

    R._rollback_tainted_wip(
        item,
        state,
        {"transcript": transcript_name},
        hashlib.sha256(transcript_payload).hexdigest(),
    )

    restored_level = level.stat(follow_symlinks=False)
    restored_latest = latest.stat(follow_symlinks=False)
    assert restored_level.st_mode == baseline_level.st_mode
    assert restored_level.st_uid == baseline_level.st_uid
    assert restored_level.st_gid == baseline_level.st_gid
    assert restored_level.st_mtime_ns == baseline_level.st_mtime_ns
    assert R._portable_getxattr(level, root_attribute) == b"sealed-root"
    assert "user.gkm_wip_extra" not in R._portable_listxattr(level)
    assert restored_latest.st_mode == baseline_latest.st_mode
    assert restored_latest.st_uid == baseline_latest.st_uid
    assert restored_latest.st_gid == baseline_latest.st_gid
    assert restored_latest.st_mtime_ns == baseline_latest.st_mtime_ns
    assert R._portable_getxattr(latest, latest_attribute) == b"sealed-latest"
    assert "user.gkm_latest_extra" not in R._portable_listxattr(latest)


def test_tainted_wip_rollback_rejects_ambiguous_multiple_new_attempts(
    tmp_path, monkeypatch
):
    item, level, latest, _old_latest, state = _wip_rollback_fixture(
        tmp_path, monkeypatch
    )
    transcript_name = "codex_turn_tainted.jsonl"
    transcript_payload = b'{"type":"turn.completed","usage":{}}\n'
    first = _write_authenticated_tainted_attempt(
        item,
        level,
        attempt="interrupted_first",
        transcript_name=transcript_name,
        transcript_payload=transcript_payload,
        update_latest=True,
    )
    second = _write_authenticated_tainted_attempt(
        item,
        level,
        attempt="interrupted_second",
        transcript_name=transcript_name,
        transcript_payload=transcript_payload,
        update_latest=False,
    )

    with pytest.raises(R.CampaignPlanError, match="exactly one isolated"):
        R._rollback_tainted_wip(
            item,
            state,
            {"transcript": transcript_name},
            hashlib.sha256(transcript_payload).hexdigest(),
        )

    assert first.is_dir()
    assert second.is_dir()
    assert json.loads(latest.read_text())["attempt"] == "interrupted_first"


@pytest.mark.parametrize("preexisting_parent", (True, False))
def test_tainted_wip_rollback_restores_absent_namespace_custody(
    tmp_path, monkeypatch, preexisting_parent
):
    monkeypatch.setattr(R, "HERE", tmp_path)
    item = copy.deepcopy(_item())
    level = R._target_wip_level(item)
    canonical_root = level.parents[1]
    canonical_root.mkdir(parents=True)
    if preexisting_parent:
        level.parent.mkdir()
    custody_path = level.parent if preexisting_parent else canonical_root
    baseline_metadata = custody_path.stat(follow_symlinks=False)
    state = R._capture_wip_rollback(item)
    assert not state.existed
    assert state.absence_custody is not None
    assert state.absence_custody.parent == custody_path

    level.mkdir(parents=True)
    transcript_name = "codex_turn_tainted.jsonl"
    transcript_payload = b'{"type":"turn.completed","usage":{}}\n'
    _write_authenticated_tainted_attempt(
        item,
        level,
        attempt="interrupted_absent_root",
        transcript_name=transcript_name,
        transcript_payload=transcript_payload,
        update_latest=True,
    )
    os.chmod(custody_path, 0o700)

    R._rollback_tainted_wip(
        item,
        state,
        {"transcript": transcript_name},
        hashlib.sha256(transcript_payload).hexdigest(),
    )

    assert not level.exists()
    if not preexisting_parent:
        assert not level.parent.exists()
    restored_metadata = custody_path.stat(follow_symlinks=False)
    assert (restored_metadata.st_dev, restored_metadata.st_ino) == (
        baseline_metadata.st_dev,
        baseline_metadata.st_ino,
    )
    assert restored_metadata.st_mode == baseline_metadata.st_mode
    assert restored_metadata.st_uid == baseline_metadata.st_uid
    assert restored_metadata.st_gid == baseline_metadata.st_gid
    assert restored_metadata.st_mtime_ns == baseline_metadata.st_mtime_ns
    assert R._wip_logical_restore_state_sha256(
        R._capture_wip_rollback(item)
    ) == R._wip_logical_restore_state_sha256(state)


def test_tainted_wip_absence_custody_replays_after_parent_fsync_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(R, "HERE", tmp_path)
    item = copy.deepcopy(_item())
    level = R._target_wip_level(item)
    level.parent.mkdir(parents=True)
    state = R._capture_wip_rollback(item)
    assert state.absence_custody is not None
    parent = state.absence_custody.parent
    parent_identity = state.absence_custody.parent_identity
    level.mkdir()
    transcript_name = "codex_turn_tainted.jsonl"
    transcript_payload = b'{"type":"turn.completed","usage":{}}\n'
    _write_authenticated_tainted_attempt(
        item,
        level,
        attempt="interrupted_fsync",
        transcript_name=transcript_name,
        transcript_payload=transcript_payload,
        update_latest=True,
    )
    real_fsync = R.os.fsync
    injected = False

    def fail_parent_fsync_once(descriptor):
        nonlocal injected
        metadata = os.fstat(descriptor)
        if not injected and (metadata.st_dev, metadata.st_ino) == parent_identity:
            injected = True
            raise OSError(errno.EIO, "synthetic absence-parent fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(R.os, "fsync", fail_parent_fsync_once)
    with pytest.raises(R.CampaignPlanError, match="durably restore"):
        R._rollback_tainted_wip(
            item,
            state,
            {"transcript": transcript_name},
            hashlib.sha256(transcript_payload).hexdigest(),
        )
    assert injected
    assert not level.exists()

    monkeypatch.setattr(R.os, "fsync", real_fsync)
    R._durably_restore_wip_absence_custody(
        level, state.absence_custody
    )
    restored = parent.stat(follow_symlinks=False)
    assert restored.st_mtime_ns == state.absence_custody.parent_mtime_ns
    assert R._wip_logical_restore_state_sha256(
        R._capture_wip_rollback(item)
    ) == R._wip_logical_restore_state_sha256(state)


def _canonical_rollback_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "HERE", tmp_path)
    item = copy.deepcopy(_item())
    root = tmp_path / "agent_solutions" / "ar25_legs"
    evidence = root / "promotion_evidence" / "level_01"
    evidence.mkdir(parents=True)
    (root / "ar25_legs.py").write_bytes(b"canonical combined source\n")
    (root / "ar25_manifest.json").write_bytes(b'{"reached":1}\n')
    (evidence / "receipt.json").write_bytes(b'{"level":1,"clean":true}\n')

    wip = root / "wip_context" / "level_01"
    wip.mkdir(parents=True)
    (wip / "latest.json").write_bytes(b'{"attempt":"clean_attempt"}\n')
    (wip / "sealed_context.txt").write_bytes(b"must remain untouched\n")
    other_wip = root / "wip_context" / "level_02" / "clean_attempt"
    other_wip.mkdir(parents=True)
    (other_wip / "sealed.txt").write_bytes(b"other level must restore\n")
    state = R._capture_canonical_rollback(item)
    return item, root, wip, state


def test_canonical_rollback_excludes_wip_and_restores_tainted_promotion(
    tmp_path, monkeypatch
):
    item, root, wip, state = _canonical_rollback_fixture(
        tmp_path, monkeypatch
    )
    assert "wip_context" in state.entries
    assert not any(
        name == "wip_context/level_01"
        or name.startswith("wip_context/level_01/")
        for name in state.entries
    )
    wip_snapshot = R._target_wip_snapshot(item)
    wip_payloads = {
        path.relative_to(wip).as_posix(): path.read_bytes()
        for path in sorted(wip.iterdir())
        if path.is_file()
    }

    (root / "ar25_legs.py").write_bytes(b"tainted promoted source\n")
    (root / "ar25_manifest.json").unlink()
    (root / "promotion_evidence" / "level_01" / "receipt.json").write_bytes(
        b'{"level":1,"clean":false}\n'
    )
    promoted_evidence = root / "promotion_evidence" / "level_02"
    promoted_evidence.mkdir()
    (promoted_evidence / "receipt.json").write_bytes(
        b'{"level":2,"tainted":true}\n'
    )
    arbitrary_extra = root / "tainted_extra" / "nested"
    arbitrary_extra.mkdir(parents=True)
    (arbitrary_extra / "payload.bin").write_bytes(b"tainted extra\n")
    outside = tmp_path / "outside-sentinel.txt"
    outside.write_bytes(b"outside must survive\n")
    extra_link = root / "tainted_symlink"
    extra_link.symlink_to(outside)
    other_wip = root / "wip_context" / "level_02" / "clean_attempt"
    (other_wip / "sealed.txt").write_bytes(b"poisoned other level\n")
    (other_wip / "extra.py").write_bytes(b"raise SystemExit('poison')\n")

    assert not R._canonical_matches(state)
    R._rollback_tainted_canonical(state)

    assert R._canonical_matches(state)
    assert R._canonical_inventory(
        root, excluded_prefixes=state.excluded_prefixes
    )[1] == state.digest
    assert (root / "ar25_legs.py").read_bytes() == b"canonical combined source\n"
    assert (root / "ar25_manifest.json").read_bytes() == b'{"reached":1}\n'
    assert (
        root / "promotion_evidence" / "level_01" / "receipt.json"
    ).read_bytes() == b'{"level":1,"clean":true}\n'
    assert not promoted_evidence.exists()
    assert not (root / "tainted_extra").exists()
    assert not extra_link.exists()
    assert outside.read_bytes() == b"outside must survive\n"
    assert (other_wip / "sealed.txt").read_bytes() == (
        b"other level must restore\n"
    )
    assert not (other_wip / "extra.py").exists()
    assert R._target_wip_snapshot(item) == wip_snapshot
    assert {
        path.relative_to(wip).as_posix(): path.read_bytes()
        for path in sorted(wip.iterdir())
        if path.is_file()
    } == wip_payloads


def test_canonical_rollback_seals_shared_excluded_ancestor_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(R, "HERE", tmp_path)
    item = copy.deepcopy(_item())
    root = tmp_path / "agent_solutions" / "ar25_legs"
    shared = root / "wip_context"
    other = shared / "level_02" / "sealed"
    other.mkdir(parents=True)
    (root / "ar25_legs.py").write_bytes(b"canonical source\n")
    (other / "keep.txt").write_bytes(b"other frontier\n")
    attribute = "user.gkm_shared_wip_parent"
    R._portable_setxattr(shared, attribute, b"sealed")
    state = R._capture_canonical_rollback(item)
    baseline = shared.stat(follow_symlinks=False)

    target = shared / "level_01" / "ambiguous"
    target.mkdir(parents=True)
    (target / "preserve.txt").write_bytes(b"target evidence\n")
    R._portable_setxattr(shared, attribute, b"changed")
    R._portable_setxattr(shared, "user.gkm_extra", b"tainted")
    os.chmod(shared, 0o700)
    os.utime(shared, ns=(baseline.st_atime_ns, baseline.st_mtime_ns + 1))

    assert "wip_context" in state.entries
    assert not R._canonical_matches(state)
    R._rollback_tainted_canonical(state)

    restored = shared.stat(follow_symlinks=False)
    assert R._canonical_matches(state)
    assert stat.S_IMODE(restored.st_mode) == stat.S_IMODE(baseline.st_mode)
    assert restored.st_mtime_ns == baseline.st_mtime_ns
    assert R._portable_getxattr(shared, attribute) == b"sealed"
    assert "user.gkm_extra" not in R._portable_listxattr(shared)
    assert (target / "preserve.txt").read_bytes() == b"target evidence\n"


def test_canonical_rollback_fails_closed_if_root_identity_changes(
    tmp_path, monkeypatch
):
    _item_value, root, _wip, state = _canonical_rollback_fixture(
        tmp_path, monkeypatch
    )
    displaced = tmp_path / "displaced-canonical-root"
    root.rename(displaced)
    root.mkdir()
    (root / "replacement.txt").write_bytes(b"unrelated replacement\n")

    with pytest.raises(R.CampaignPlanError, match="root changed identity"):
        R._rollback_tainted_canonical(state)

    assert (displaced / "ar25_legs.py").read_bytes() == (
        b"canonical combined source\n"
    )
    assert (root / "replacement.txt").read_bytes() == (
        b"unrelated replacement\n"
    )


def test_canonical_rollback_detects_and_restores_root_mode_only_change(
    tmp_path, monkeypatch
):
    _item_value, root, _wip, state = _canonical_rollback_fixture(
        tmp_path, monkeypatch
    )
    baseline_mode = stat.S_IMODE(state.root_mode)
    changed_mode = 0o700 if baseline_mode != 0o700 else 0o755
    os.chmod(root, changed_mode)

    assert not R._canonical_matches(state)
    R._rollback_tainted_canonical(state)

    assert R._canonical_matches(state)
    assert stat.S_IMODE(root.stat().st_mode) == baseline_mode


def test_canonical_rollback_restores_extended_attributes(
    tmp_path, monkeypatch
):
    item, root, _wip, _initial_state = _canonical_rollback_fixture(
        tmp_path, monkeypatch
    )
    source = root / "ar25_legs.py"
    attribute = "user.gkm_scheduler_test"
    extra = "user.gkm_scheduler_extra"
    R._portable_setxattr(source, attribute, b"sealed")
    state = R._capture_canonical_rollback(item)

    R._portable_setxattr(source, attribute, b"changed")
    R._portable_setxattr(source, extra, b"tainted")

    assert not R._canonical_matches(state)
    R._rollback_tainted_canonical(state)

    assert R._canonical_matches(state)
    assert R._portable_getxattr(source, attribute) == b"sealed"
    assert extra not in R._portable_listxattr(source)


def test_clean_generation_auth_failure_restores_canonical_and_preserves_evidence(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(tmp_path, monkeypatch)
    root = tmp_path / "agent_solutions" / "ar25_legs"
    canonical = root / "ar25_legs.py"
    canonical.write_bytes(b"sealed canonical baseline\n")
    baseline = R._capture_canonical_rollback(fixture["item"])
    ambiguous_wip = (
        root / "wip_context" / "level_01" / "ambiguous_after_clean_exit"
    )

    def nominally_clean_child(*_args, **_kwargs):
        canonical.write_bytes(b"unauthenticated promoted bytes\n")
        promoted = root / "promotion_evidence" / "level_01"
        promoted.mkdir(parents=True)
        (promoted / "receipt.json").write_bytes(b'{"authenticated":false}\n')
        ambiguous_wip.mkdir(parents=True)
        (ambiguous_wip / "preserve.txt").write_bytes(
            b"generation identity remains ambiguous\n"
        )
        _append_clean_dispatch_ledger(fixture)
        return R.GuardedChildResult(
            returncode=0,
            workspace=fixture["workspace"].name,
            transcript=fixture["record"]["transcript"],
            workspace_identity=(
                fixture["workspace"].stat().st_dev,
                fixture["workspace"].stat().st_ino,
            ),
            protected_identity=(
                fixture["protected"].stat().st_dev,
                fixture["protected"].stat().st_ino,
            ),
            process_tree_quiesced=True,
        )

    monkeypatch.setattr(R, "_run_guarded_child", nominally_clean_child)

    with pytest.raises(
        R.CampaignPlanError,
        match="tainted WIP receipt is unavailable",
    ):
        R._run_item(
            fixture["plan"], fixture["item"], allowance=fixture["allowance"]
        )

    assert R._canonical_matches(baseline)
    assert canonical.read_bytes() == b"sealed canonical baseline\n"
    assert not (root / "promotion_evidence").exists()
    assert (ambiguous_wip / "preserve.txt").read_bytes() == (
        b"generation identity remains ambiguous\n"
    )
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    rows = R.Guard.read_ledger(fixture["ledger"])
    assert [row["event"] for row in rows] == [
        "codex_exec",
        "codex_level_outcome",
        "codex_exec_classification_correction",
    ]
    assert rows[2]["failure_class"] == "taint"
    assert rows[2]["retry_increment"] == 0


def test_taint_generation_auth_failure_after_exit_restores_canonical(
    tmp_path, monkeypatch
):
    root = tmp_path / "agent_solutions" / "ar25_legs"
    canonical = root / "ar25_legs.py"

    def mutate_canonical(_tmp_root):
        canonical.write_bytes(b"tainted promoted bytes\n")
        extra = root / "tainted_promotion" / "nested"
        extra.mkdir(parents=True)
        (extra / "payload.bin").write_bytes(b"must be rolled back\n")

    fixture = _taint_dispatch_fixture(
        tmp_path,
        monkeypatch,
        duplicate_exec=True,
        child_mutation=mutate_canonical,
    )
    canonical.write_bytes(b"sealed canonical baseline\n")
    baseline = R._capture_canonical_rollback(fixture["item"])

    with pytest.raises(R.CampaignPlanError, match="ambiguous.*ledger suffix"):
        R._run_item(
            fixture["plan"], fixture["item"], allowance=fixture["allowance"]
        )

    assert R._canonical_matches(baseline)
    assert canonical.read_bytes() == b"sealed canonical baseline\n"
    assert not (root / "tainted_promotion").exists()
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert R.Guard.read_ledger(fixture["ledger"]) == [
        fixture["record"], fixture["record"],
    ]
