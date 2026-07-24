"""Offline regression tests for the local Claude proposer budget guard.

No `claude` turn is ever launched: parsing, windowing, and cap enforcement run on
synthetic records, and the integration test proves that a hit cap refuses the turn
*before* any subprocess spawns, so no subscription allowance is spent to test it.
"""
import time

import pytest

import gkm_legs
import claude_usage_guard as C
from claude_usage_guard import WindowCaps, ClaudeUsageGuardError


def _rec(started, dur_s, out_tokens=None, cost=None):
    row = {"event": "claude_exec", "started_at": started, "duration_seconds": dur_s}
    if out_tokens is not None:
        row["output_tokens"] = out_tokens
    if cost is not None:
        row["total_cost_usd"] = cost
    return row


# --- parse_claude_json_usage ----------------------------------------------

def test_parse_success_sums_cache_tiers():
    s = ('{"type":"result","result":"ok","usage":{"input_tokens":100,'
         '"cache_read_input_tokens":40,"cache_creation_input_tokens":10,'
         '"output_tokens":55},"total_cost_usd":0.07,"num_turns":4}')
    u = C.parse_claude_json_usage(s)
    assert u["result_text"] == "ok"
    assert u["input_tokens"] == 150          # base + both cache tiers
    assert u["output_tokens"] == 55
    assert u["total_cost_usd"] == 0.07
    assert u["num_turns"] == 4
    assert u["usage_reported"] is True


def test_parse_timeout_partial_is_text_only():
    u = C.parse_claude_json_usage("partial not-json output")
    assert u["result_text"].startswith("partial")
    assert u["output_tokens"] is None
    assert u["usage_reported"] is False


def test_parse_empty_and_usageless():
    assert C.parse_claude_json_usage("")["result_text"] == ""
    u = C.parse_claude_json_usage('{"type":"result","result":"done"}')
    assert u["result_text"] == "done"
    assert u["output_tokens"] is None
    assert u["usage_reported"] is False


# --- window selection + totals --------------------------------------------

def test_window_excludes_records_older_than_window():
    now = 1_000_000.0
    recs = [_rec(now - 3600, 60), _rec(now - 6 * 3600, 60)]
    assert len(C.window_records(recs, window_hours=5, now=now)) == 1


def test_window_includes_untimestamped_conservatively():
    now = 1_000_000.0
    recs = [{"event": "claude_exec", "duration_seconds": 60}]
    assert len(C.window_records(recs, window_hours=5, now=now)) == 1


def test_window_ignores_non_claude_events():
    now = 1_000_000.0
    recs = [{"event": "rate_limit_snapshot", "started_at": now}]
    assert C.window_records(recs, window_hours=5, now=now) == []


def test_window_totals_sums_turns_tokens_walltime_cost():
    now = 1_000_000.0
    recs = [_rec(now - 100, 300, out_tokens=1000, cost=0.5),
            _rec(now - 50, 120, out_tokens=2000, cost=0.3)]
    t = C.window_totals(recs)
    assert t["turns"] == 2
    assert t["output_tokens"] == 3000
    assert t["wall_minutes"] == 7.0
    assert t["cost_usd"] == 0.8
    assert t["turns_missing_token_usage"] == 0


def test_window_totals_counts_missing_token_usage():
    assert C.window_totals([_rec(0, 60)])["turns_missing_token_usage"] == 1


# --- preflight cap enforcement --------------------------------------------

def test_preflight_passes_with_headroom(tmp_path):
    led = tmp_path / "l.jsonl"
    C.append_ledger(_rec(time.time(), 120, out_tokens=500), led)
    out = C.preflight(caps=WindowCaps(max_turns=5, max_wall_minutes=120), ledger_path=led)
    assert out["window_totals"]["turns"] == 1
    assert out["remaining"]["turns"] == 4


def test_preflight_raises_on_turn_cap(tmp_path):
    led = tmp_path / "l.jsonl"
    for _ in range(2):
        C.append_ledger(_rec(time.time(), 60), led)
    with pytest.raises(ClaudeUsageGuardError):
        C.preflight(caps=WindowCaps(max_turns=2), ledger_path=led)


def test_preflight_raises_on_wall_minute_cap(tmp_path):
    led = tmp_path / "l.jsonl"
    C.append_ledger(_rec(time.time(), 130 * 60), led)  # 130 wall minutes
    with pytest.raises(ClaudeUsageGuardError):
        C.preflight(caps=WindowCaps(max_wall_minutes=120), ledger_path=led)


def test_preflight_none_caps_never_raise(tmp_path):
    led = tmp_path / "l.jsonl"
    for _ in range(50):
        C.append_ledger(_rec(time.time(), 600), led)
    out = C.preflight(caps=WindowCaps(), ledger_path=led)  # every cap None
    assert out["remaining"]["turns"] is None


def test_preflight_old_records_do_not_count(tmp_path):
    led = tmp_path / "l.jsonl"
    C.append_ledger(_rec(time.time() - 6 * 3600, 60), led)  # outside the 5h window
    out = C.preflight(caps=WindowCaps(max_turns=1), window_hours=5, ledger_path=led)
    assert out["window_totals"]["turns"] == 0


# --- integration: the guard refuses without spending a Claude turn ---------

def test_claude_agent_guard_refuses_at_cap_without_spawning(tmp_path):
    led = tmp_path / "claude.jsonl"
    for _ in range(2):
        C.append_ledger(_rec(time.time(), 60), led)
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(gkm_legs.CreditOut):
        gkm_legs._claude_agent(
            str(ws), "task", None, 5, guard=True,
            ledger_path=str(led), max_turns=2, window_hours=5,
        )
    # Preflight fired before any subprocess: no proposer transcript was written,
    # proving no `claude` turn (and no allowance) was spent to hit the cap.
    assert not (ws / "proposer_last.log").exists()
