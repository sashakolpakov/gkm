#!/usr/bin/env python3
"""Local budget control for the Claude Code headless proposer.

Unlike Codex -- whose ``codex app-server`` exposes ``account/rateLimits/read`` so a
live seven-day allowance can be gated before every turn -- the ``claude`` CLI exposes
**no** readable remaining allowance (it has 5-hour and weekly subscription limits, but
no command reads them; `claude --help` offers only `agents/doctor/install/mcp/
setup-token`). Its limits are known only reactively, when a turn reports credit-out.

Control here is therefore a LOCAL budget, not a provider read: every Claude proposer
turn appends its OBSERVED cost (wall time, and -- via ``claude -p --output-format json``
-- input/output tokens and dollar cost) to a durable ledger, and admission enforces
cumulative caps within a rolling window (turns, output tokens, wall minutes, dollars).
Reactive credit-out remains handled by the orchestrator's ``CreditOut`` path.

This module has no operation that reads or mutates provider allowance state.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_LEDGER = Path(__file__).resolve().parent / "runs" / "claude_campaign_usage.jsonl"
# Claude's shorter subscription bucket is five hours; the operator can widen the
# rolling window to the weekly horizon (168h) for a second, longer-horizon guard.
DEFAULT_WINDOW_HOURS = 5.0


class ClaudeUsageGuardError(RuntimeError):
    """A configured local Claude budget is exhausted for the current window."""


def _iso(epoch: Optional[float]) -> Optional[str]:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _epoch_of(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def read_ledger(path: Path | str = DEFAULT_LEDGER) -> list[Dict[str, Any]]:
    ledger = Path(path)
    if not ledger.exists():
        return []
    records = []
    for line_number, raw in enumerate(ledger.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ClaudeUsageGuardError(
                f"invalid JSON in {ledger} line {line_number}: {exc}"
            ) from exc
        if isinstance(value, dict):
            records.append(value)
    return records


@contextmanager
def campaign_lock(path: Path | str = DEFAULT_LEDGER):
    """Serialize a paid Claude turn against its own ledger (independent of Codex)."""
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{ledger}.lock")
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ClaudeUsageGuardError(
                f"another Claude campaign turn holds {lock_path}; refusing a concurrent run"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def append_ledger(record: Dict[str, Any], path: Path | str = DEFAULT_LEDGER) -> None:
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def parse_claude_json_usage(stdout: str) -> Dict[str, Any]:
    """Pull observed usage from a ``claude -p --output-format json`` result.

    Returns the result text plus token and dollar counters.  A timed-out or
    malformed stream yields text-only with null counters, so metering degrades to
    wall-time and turn-count rather than crashing the proposer.
    """
    blank = {
        "result_text": stdout,
        "input_tokens": None,
        "output_tokens": None,
        "total_cost_usd": None,
        "num_turns": None,
        "usage_reported": False,
    }
    stripped = (stdout or "").strip()
    if not stripped:
        return blank
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return blank
    if not isinstance(payload, dict):
        return blank
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    output_tokens = usage.get("output_tokens")
    # Claude usage reports cache tiers separately; sum what is present for input.
    input_tokens = usage.get("input_tokens")
    if isinstance(input_tokens, int):
        for extra in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            value = usage.get(extra)
            if isinstance(value, int):
                input_tokens += value
    return {
        "result_text": payload.get("result", stdout)
        if isinstance(payload.get("result"), str) else stdout,
        "input_tokens": input_tokens if isinstance(input_tokens, int) else None,
        "output_tokens": output_tokens if isinstance(output_tokens, int) else None,
        "total_cost_usd": (
            float(payload["total_cost_usd"])
            if isinstance(payload.get("total_cost_usd"), (int, float)) else None
        ),
        "num_turns": payload.get("num_turns")
        if isinstance(payload.get("num_turns"), int) else None,
        "usage_reported": isinstance(output_tokens, int),
    }


@dataclass(frozen=True)
class WindowCaps:
    max_turns: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_wall_minutes: Optional[float] = None
    max_cost_usd: Optional[float] = None


def window_records(records: Iterable[Dict[str, Any]], *, window_hours: float,
                   now: Optional[float] = None) -> list[Dict[str, Any]]:
    """Claude proposer turns whose start falls within the rolling window."""
    now = time.time() if now is None else now
    cutoff = now - window_hours * 3600.0
    result = []
    for record in records:
        if record.get("event") != "claude_exec":
            continue
        started = _epoch_of(record.get("started_at"))
        if started is None or started >= cutoff:
            result.append(record)
    return result


def window_totals(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    return {
        "turns": len(records),
        "output_tokens": sum(
            int(r.get("output_tokens") or 0) for r in records
        ),
        "input_tokens": sum(int(r.get("input_tokens") or 0) for r in records),
        "wall_minutes": round(sum(
            float(r.get("duration_seconds") or 0.0) for r in records
        ) / 60.0, 3),
        "cost_usd": round(sum(
            float(r.get("total_cost_usd") or 0.0) for r in records
        ), 4),
        "turns_missing_token_usage": sum(
            1 for r in records if not isinstance(r.get("output_tokens"), int)
        ),
    }


def preflight(*, caps: WindowCaps, window_hours: float = DEFAULT_WINDOW_HOURS,
              ledger_path: Path | str = DEFAULT_LEDGER,
              now: Optional[float] = None) -> Dict[str, Any]:
    """Admit a Claude turn only if every configured local cap has headroom.

    There is no provider allowance read; this sums the durable ledger over the
    rolling window and refuses once a cap is reached.  It is a spend ceiling, not
    a remaining-allowance gate, because the subscription allowance is unreadable.
    """
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    current = window_records(read_ledger(ledger_path), window_hours=window_hours, now=now)
    totals = window_totals(current)
    if caps.max_turns is not None and totals["turns"] >= caps.max_turns:
        raise ClaudeUsageGuardError(
            f"local Claude turn cap reached ({totals['turns']}/{caps.max_turns} "
            f"in the last {window_hours}h)"
        )
    if (caps.max_output_tokens is not None
            and totals["output_tokens"] >= caps.max_output_tokens):
        raise ClaudeUsageGuardError(
            f"local Claude output-token cap reached "
            f"({totals['output_tokens']}/{caps.max_output_tokens} in the last {window_hours}h)"
        )
    if (caps.max_wall_minutes is not None
            and totals["wall_minutes"] >= caps.max_wall_minutes):
        raise ClaudeUsageGuardError(
            f"local Claude wall-time cap reached "
            f"({totals['wall_minutes']}/{caps.max_wall_minutes} min in the last {window_hours}h)"
        )
    if caps.max_cost_usd is not None and totals["cost_usd"] >= caps.max_cost_usd:
        raise ClaudeUsageGuardError(
            f"local Claude dollar cap reached "
            f"(${totals['cost_usd']}/${caps.max_cost_usd} in the last {window_hours}h)"
        )

    def remaining(cap: Optional[float], used: float) -> Optional[float]:
        return None if cap is None else max(0.0, cap - used)

    return {
        "window_hours": window_hours,
        "window_totals": totals,
        "remaining": {
            "turns": remaining(caps.max_turns, totals["turns"]),
            "output_tokens": remaining(caps.max_output_tokens, totals["output_tokens"]),
            "wall_minutes": remaining(caps.max_wall_minutes, totals["wall_minutes"]),
            "cost_usd": remaining(caps.max_cost_usd, totals["cost_usd"]),
        },
        "note": "local spend ceiling; the Claude subscription allowance is not readable",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--window-hours", type=float, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--max-wall-minutes", type=float, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    caps = WindowCaps(
        max_turns=args.max_turns,
        max_output_tokens=args.max_output_tokens,
        max_wall_minutes=args.max_wall_minutes,
        max_cost_usd=args.max_cost_usd,
    )
    current = window_records(
        read_ledger(args.ledger), window_hours=args.window_hours
    )
    status = {
        "generated_at": _iso(time.time()),
        "ledger": os.fspath(args.ledger),
        "window_hours": args.window_hours,
        "window_totals": window_totals(current),
        "note": "local spend meter; Claude subscription allowance is not provider-readable",
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
