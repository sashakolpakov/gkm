from __future__ import annotations

import copy
import hashlib
import json
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
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(R.subprocess, "run", failed_child)
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
    ]
    correction = rows[1]
    assert correction["failure_class"] == "taint"
    assert correction["failure_detail_class"] == "host_process_introspection"
    assert correction["taint_verdict"] == "tainted"
    assert correction["solved_target"] is None
    assert correction["retry_increment"] == 0
    assert rows[0]["observed_tokens"] == 123


def test_nonzero_taint_recovery_fails_closed_on_ambiguous_exec_records(
    tmp_path, monkeypatch
):
    fixture = _taint_dispatch_fixture(
        tmp_path, monkeypatch, duplicate_exec=True
    )

    with pytest.raises(R.CampaignPlanError, match="exact-dispatch"):
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
    assert len(R.Guard.read_ledger(fixture["ledger"])) == 1


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
    (fixture["workspace"] / "gkm_try.py").write_text(
        "import sys\nsys.path.insert(0, '/receipt-bound/historical/arena')\n"
    )
    fixture["item"]["historical_runner"] = {
        "evidence_schema": "sealed_transcript_only_v1",
        "lock_schema": "in_workspace_v1",
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
        wip_snapshot_before=R._target_wip_snapshot(fixture["item"]),
        child_returncode=1,
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
            wip_snapshot_before=R._target_wip_snapshot(fixture["item"]),
            child_returncode=1,
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

    monkeypatch.setattr(R.subprocess, "run", forbidden_child)
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

    with pytest.raises(R.CampaignPlanError, match="WIP inventory"):
        R._run_item(
            fixture["plan"], fixture["item"], allowance=fixture["allowance"]
        )
    assert fixture["workspace"].is_dir()
    assert fixture["protected"].is_dir()
    assert [row["event"] for row in R.Guard.read_ledger(fixture["ledger"])] == [
        "codex_exec"
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
