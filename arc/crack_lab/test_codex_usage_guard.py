"""Offline tests for the headless Codex weekly-allowance guard."""

import json
from pathlib import Path

import pytest

import codex_usage_guard as G


def _snapshot(*, weekly_used=6, weekly_reset=1_800_000_000):
    return {
        "rateLimitsByLimitId": {
            "codex": {
                "planType": "plus",
                "primary": {
                    "usedPercent": 1,
                    "resetsAt": weekly_reset - 10_000,
                    "windowDurationMins": 300,
                },
                "secondary": {
                    "usedPercent": weekly_used,
                    "resetsAt": weekly_reset,
                    "windowDurationMins": 10_080,
                },
            }
        },
        "rateLimitResetCredits": {"availableCount": 1},
    }


def test_weekly_allowance_ignores_short_primary_window():
    allowance = G.weekly_allowance(_snapshot(weekly_used=6))
    assert allowance.window_name == "secondary"
    assert allowance.used_percent == 6
    assert allowance.remaining_percent == 94
    assert allowance.window_duration_mins == G.WEEK_MINUTES
    assert allowance.reset_credits_available == 1


def test_weekly_allowance_refuses_to_guess_from_short_window():
    snapshot = _snapshot()
    snapshot["rateLimitsByLimitId"]["codex"].pop("secondary")
    with pytest.raises(G.CodexUsageGuardError, match="no seven-day window"):
        G.weekly_allowance(snapshot)


def test_preflight_enforces_reserve_and_current_window_local_caps(tmp_path):
    ledger = tmp_path / "usage.jsonl"
    reset = 1_800_000_000
    records = [
        {
            "event": "codex_exec",
            "weekly_resets_at": reset,
            "observed_tokens": 120,
            "input_tokens": 100,
            "cached_input_tokens": 70,
            "output_tokens": 20,
            "reasoning_output_tokens": 12,
        },
        {
            "event": "codex_exec",
            "weekly_resets_at": reset - 604_800,
            "observed_tokens": 999_999,
        },
    ]
    ledger.write_text("".join(json.dumps(record) + "\n" for record in records))

    status = G.preflight(
        reserve_percent=90,
        max_campaign_tokens=200,
        max_campaign_runs=2,
        ledger_path=ledger,
        snapshot=_snapshot(weekly_used=6, weekly_reset=reset),
    )
    assert status["allowance"]["remaining_percent"] == 94
    assert status["local_window"]["runs"] == 1
    assert status["local_window"]["observed_tokens"] == 120

    jittered = G.WeeklyAllowance(
        limit_id="codex", window_name="secondary", used_percent=6,
        remaining_percent=94, resets_at=reset + 1,
        window_duration_mins=G.WEEK_MINUTES, plan_type="plus",
        reset_credits_available=1,
    )
    assert len(G.current_window_records(G.read_ledger(ledger), jittered)) == 1

    with pytest.raises(G.CodexUsageGuardError, match="at or below the configured"):
        G.preflight(
            reserve_percent=95,
            max_campaign_tokens=200,
            max_campaign_runs=2,
            ledger_path=ledger,
            snapshot=_snapshot(weekly_used=6, weekly_reset=reset),
        )
    with pytest.raises(G.CodexUsageGuardError, match="only 3% headroom"):
        G.preflight(
            reserve_percent=91,
            minimum_headroom_percent=4,
            max_campaign_tokens=200,
            max_campaign_runs=2,
            ledger_path=ledger,
            snapshot=_snapshot(weekly_used=6, weekly_reset=reset),
        )
    with pytest.raises(G.CodexUsageGuardError, match="at or below the configured"):
        G.preflight(
            reserve_percent=94,
            max_campaign_tokens=200,
            max_campaign_runs=2,
            ledger_path=ledger,
            snapshot=_snapshot(weekly_used=6, weekly_reset=reset),
        )
    with pytest.raises(G.CodexUsageGuardError, match="run cap"):
        G.preflight(
            reserve_percent=90,
            max_campaign_tokens=200,
            max_campaign_runs=1,
            ledger_path=ledger,
            snapshot=_snapshot(weekly_used=6, weekly_reset=reset),
        )
    with pytest.raises(G.CodexUsageGuardError, match="token cap"):
        G.preflight(
            reserve_percent=90,
            max_campaign_tokens=120,
            max_campaign_runs=2,
            ledger_path=ledger,
            snapshot=_snapshot(weekly_used=6, weekly_reset=reset),
        )


def test_campaign_lock_rejects_concurrent_holder(tmp_path):
    ledger = tmp_path / "usage.jsonl"
    with G.campaign_lock(ledger):
        with pytest.raises(G.CodexUsageGuardError, match="concurrent run"):
            with G.campaign_lock(ledger):
                pass


def test_append_ledger_is_valid_and_durable(tmp_path):
    ledger = tmp_path / "usage.jsonl"
    with G.campaign_lock(ledger):
        G.append_ledger({"event": "codex_exec", "observed_tokens": 17}, ledger)
    assert G.read_ledger(ledger) == [
        {"event": "codex_exec", "observed_tokens": 17}
    ]
