from __future__ import annotations

import copy
import json
import sys
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
    workspace = tmp_path / "runs" / "scratch" / "gkm_legs_ws_sk48_other"
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
