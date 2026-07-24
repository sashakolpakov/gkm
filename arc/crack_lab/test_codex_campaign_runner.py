from __future__ import annotations

from types import SimpleNamespace

import pytest

import codex_campaign_runner as R


def _item():
    return {
        "game": "ar25",
        "target_level": 1,
        "required_headroom_percent": 6,
        "argv": [
            "python3", "-u", "arc/crack_lab/gkm_legs.py",
            "--game=ar25", "--proposer=codex", "--model=gpt-5.6-sol",
            "--debrief-policy=never", "--transient-retries=0",
            "--codex-weekly-reserve=25", "--codex-weekly-headroom=6",
        ],
    }


def test_validate_item_rejects_arbitrary_commands():
    with pytest.raises(R.CampaignPlanError, match="non-GKM"):
        R.validate_item({"argv": ["sh", "-c", "anything"]})
    assert R.validate_item(_item())[0] == "python3"


def test_validate_item_requires_budget_arguments_to_match_plan():
    with pytest.raises(R.CampaignPlanError, match="reserve does not match"):
        R.validate_item(_item(), {"reserve_percent": 20})
    item = _item()
    item["argv"][-2] = "--codex-weekly-reserve=20"
    assert R.validate_item(item, {"reserve_percent": 20})[0] == "python3"


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


def test_run_item_turns_expected_headroom_failure_into_reserve_stop(monkeypatch):
    monkeypatch.setattr(R, "_checkpoint_reached", lambda game: 0)
    plan = {"not_before_epoch": 100, "reserve_percent": 25}
    allowance = SimpleNamespace(remaining_percent=30)
    result = R._run_item(plan, _item(), allowance=allowance)
    assert result["result"] == "reserve_stop"
    assert "requires 6%" in result["reason"]
