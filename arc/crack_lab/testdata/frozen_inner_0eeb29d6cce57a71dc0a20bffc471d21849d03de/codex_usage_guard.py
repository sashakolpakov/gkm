#!/usr/bin/env python3
"""Read Codex rate limits and enforce a local campaign budget.

The guard talks to ``codex app-server`` over its JSONL protocol.  The
``account/rateLimits/read`` request does not start a model turn, so it can be used
before every headless proposer invocation.  Per-turn token usage still comes from
``codex exec --json`` and is appended to a local JSONL ledger by ``gkm_legs``.

This module intentionally has no operation that consumes a rate-limit reset credit.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import stat
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


WEEK_MINUTES = 7 * 24 * 60
RESET_EPOCH_TOLERANCE_SECONDS = 120
DEFAULT_LEDGER = Path(__file__).resolve().parent / "runs" / "codex_campaign_usage.jsonl"


class CodexUsageGuardError(RuntimeError):
    """The live allowance cannot be read or a configured budget is exhausted."""


def _open_unaliased_regular(
    path: Path, *, flags: int, mode: int = 0o600
):
    """Open a host-owned accounting file without following inode aliases."""
    try:
        descriptor = os.open(
            path,
            flags | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
    except OSError as exc:
        raise CodexUsageGuardError(
            f"accounting path must be an unaliased regular file: {path}"
        ) from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise CodexUsageGuardError(
            f"accounting path must be an unaliased regular file: {path}"
        )
    return descriptor


@dataclass(frozen=True)
class WeeklyAllowance:
    limit_id: str
    window_name: str
    used_percent: int
    remaining_percent: int
    resets_at: Optional[int]
    window_duration_mins: int
    plan_type: Optional[str]
    reset_credits_available: Optional[int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "limit_id": self.limit_id,
            "window_name": self.window_name,
            "used_percent": self.used_percent,
            "remaining_percent": self.remaining_percent,
            "resets_at": self.resets_at,
            "resets_at_iso": _timestamp_iso(self.resets_at),
            "window_duration_mins": self.window_duration_mins,
            "plan_type": self.plan_type,
            "reset_credits_available": self.reset_credits_available,
        }


def _timestamp_iso(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _jsonl_reader(stream, output: queue.Queue) -> None:
    try:
        for line in stream:
            output.put(line)
    finally:
        output.put(None)


def _send(proc: subprocess.Popen, message: Dict[str, Any]) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
    proc.stdin.flush()


def _wait_response(lines: queue.Queue, request_id: int, deadline: float,
                   diagnostics: list[str]) -> Dict[str, Any]:
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=min(0.5, max(0.01, deadline - time.monotonic())))
        except queue.Empty:
            continue
        if line is None:
            break
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            diagnostics.append(line.strip())
            continue
        if message.get("id") != request_id:
            continue
        if message.get("error"):
            raise CodexUsageGuardError(
                f"codex app-server request {request_id} failed: {message['error']}"
            )
        result = message.get("result")
        if not isinstance(result, dict):
            raise CodexUsageGuardError(
                f"codex app-server request {request_id} returned no object result"
            )
        return result
    detail = "; ".join(x for x in diagnostics[-3:] if x)
    suffix = f" ({detail})" if detail else ""
    raise CodexUsageGuardError(
        f"timed out waiting for codex app-server request {request_id}{suffix}"
    )


def query_rate_limits(codex_bin: str = "codex", timeout: float = 20.0) -> Dict[str, Any]:
    """Return the authenticated account's current rate-limit snapshot.

    This performs only initialization and ``account/rateLimits/read``; it never
    starts a thread or model turn and never requests a reset-credit redemption.
    """
    proc = subprocess.Popen(
        [codex_bin, "app-server", "--listen", "stdio://"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_jsonl_reader, args=(proc.stdout, lines), daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    diagnostics: list[str] = []
    try:
        _send(proc, {
            "method": "initialize",
            "id": 0,
            "params": {
                "clientInfo": {
                    "name": "gkm_usage_guard",
                    "title": "GKM Usage Guard",
                    "version": "0.1",
                }
            },
        })
        _wait_response(lines, 0, deadline, diagnostics)
        _send(proc, {"method": "initialized", "params": {}})
        _send(proc, {"method": "account/rateLimits/read", "id": 1})
        return _wait_response(lines, 1, deadline, diagnostics)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)


def weekly_allowance(snapshot: Dict[str, Any]) -> WeeklyAllowance:
    """Select the longest seven-day-or-greater window from a rate snapshot."""
    buckets = snapshot.get("rateLimitsByLimitId")
    if not isinstance(buckets, dict) or not buckets:
        buckets = {"default": snapshot.get("rateLimits")}
    reset_summary = snapshot.get("rateLimitResetCredits")
    reset_count = None
    if isinstance(reset_summary, dict):
        value = reset_summary.get("availableCount")
        if isinstance(value, int):
            reset_count = value

    candidates = []
    for limit_id, rate_limit in buckets.items():
        if not isinstance(rate_limit, dict):
            continue
        for window_name in ("primary", "secondary"):
            window = rate_limit.get(window_name)
            if not isinstance(window, dict):
                continue
            duration = window.get("windowDurationMins")
            used = window.get("usedPercent")
            if not isinstance(duration, int) or not isinstance(used, int):
                continue
            if duration < WEEK_MINUTES:
                continue
            candidates.append((
                duration,
                str(limit_id),
                window_name,
                used,
                window.get("resetsAt") if isinstance(window.get("resetsAt"), int) else None,
                rate_limit.get("planType") if isinstance(rate_limit.get("planType"), str) else None,
            ))
    if not candidates:
        # Business workspaces can report an explicitly unlimited Codex pool
        # without percentage windows.  This is distinct from a malformed or
        # shorter-window-only response: the provider positively asserts both
        # unlimited credits and that no spend control has been reached.
        for limit_id, rate_limit in buckets.items():
            if not isinstance(rate_limit, dict):
                continue
            credits = rate_limit.get("credits")
            if (
                isinstance(credits, dict)
                and credits.get("unlimited") is True
                and rate_limit.get("spendControlReached") is False
            ):
                plan_type = rate_limit.get("planType")
                return WeeklyAllowance(
                    limit_id=str(limit_id),
                    window_name="unlimited",
                    used_percent=0,
                    remaining_percent=100,
                    resets_at=None,
                    window_duration_mins=WEEK_MINUTES,
                    plan_type=plan_type if isinstance(plan_type, str) else None,
                    reset_credits_available=reset_count,
                )
        raise CodexUsageGuardError(
            "the Codex rate-limit response contained no seven-day window; "
            "refusing to guess from a shorter bucket"
        )
    duration, limit_id, window_name, used, resets_at, plan_type = max(candidates)
    return WeeklyAllowance(
        limit_id=limit_id,
        window_name=window_name,
        used_percent=used,
        remaining_percent=max(0, 100 - used),
        resets_at=resets_at,
        window_duration_mins=duration,
        plan_type=plan_type,
        reset_credits_available=reset_count,
    )


def read_ledger(path: Path | str = DEFAULT_LEDGER) -> list[Dict[str, Any]]:
    ledger = Path(path)
    if ledger.parent.is_symlink() or (
        ledger.parent.exists() and not ledger.parent.is_dir()
    ):
        raise CodexUsageGuardError(
            f"accounting root must be a regular host directory: {ledger.parent}"
        )
    if ledger.is_symlink():
        raise CodexUsageGuardError(
            f"accounting path must be an unaliased regular file: {ledger}"
        )
    if not ledger.exists():
        return []
    descriptor = _open_unaliased_regular(ledger, flags=os.O_RDONLY)
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    records = []
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CodexUsageGuardError(
                f"invalid JSON in {ledger} line {line_number}: {exc}"
            ) from exc
        if isinstance(value, dict):
            records.append(value)
    return records


@contextmanager
def campaign_lock(path: Path | str = DEFAULT_LEDGER,
                  wait_seconds: float = 0.0):
    """Fail closed if another paid campaign turn is already in flight.

    The lock covers live preflight, ``codex exec``, postflight, and ledger append.
    This prevents two independent per-game orchestrators from admitting turns
    against the same stale allowance and local-budget snapshot.
    """
    if wait_seconds < 0:
        raise ValueError("wait_seconds must be nonnegative")
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if ledger.parent.is_symlink() or not ledger.parent.is_dir():
        raise CodexUsageGuardError(
            f"accounting root must be a regular host directory: {ledger.parent}"
        )
    lock_path = Path(f"{ledger}.lock")
    descriptor = _open_unaliased_regular(
        lock_path, flags=os.O_RDWR | os.O_CREAT
    )
    handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    deadline = time.monotonic() + wait_seconds
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise CodexUsageGuardError(
                        f"another Codex campaign turn holds {lock_path}; "
                        "refusing a concurrent run"
                    ) from exc
                time.sleep(min(0.1, max(0.01, deadline - time.monotonic())))
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def ledger_append_lock(path: Path | str = DEFAULT_LEDGER):
    """Serialize only the short durable append operation.

    Unlimited campaigns may run model turns concurrently because they have no
    shared cost-admission decision.  Their JSONL records still need a small
    critical section so concurrent appends cannot interleave.
    """
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if ledger.parent.is_symlink() or not ledger.parent.is_dir():
        raise CodexUsageGuardError(
            f"accounting root must be a regular host directory: {ledger.parent}"
        )
    lock_path = Path(f"{ledger}.append.lock")
    descriptor = _open_unaliased_regular(
        lock_path, flags=os.O_RDWR | os.O_CREAT
    )
    handle = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def append_ledger(record: Dict[str, Any],
                  path: Path | str = DEFAULT_LEDGER) -> None:
    """Append one durable JSON record.

    A short append-only lock protects JSONL integrity for concurrent unlimited
    turns.  Finite-budget callers additionally hold :func:`campaign_lock`
    across their whole admission/model/record transaction.
    """
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with ledger_append_lock(ledger):
        descriptor = _open_unaliased_regular(
            ledger, flags=os.O_WRONLY | os.O_APPEND | os.O_CREAT
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def current_window_records(records: Iterable[Dict[str, Any]],
                           allowance: WeeklyAllowance) -> list[Dict[str, Any]]:
    """Select this window's turns despite small provider reset-epoch jitter."""
    if allowance.resets_at is None and allowance.window_name == "unlimited":
        cutoff = datetime.now(timezone.utc).timestamp() - WEEK_MINUTES * 60
        selected = []
        for record in records:
            if record.get("event") != "codex_exec":
                continue
            value = record.get("started_at")
            if not isinstance(value, str):
                continue
            try:
                started = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
            if started.timestamp() >= cutoff:
                selected.append(record)
        return selected

    def same_window(value: Any) -> bool:
        return bool(
            isinstance(value, int)
            and isinstance(allowance.resets_at, int)
            and abs(value - allowance.resets_at) <= RESET_EPOCH_TOLERANCE_SECONDS
        )

    return [
        record for record in records
        if record.get("event") == "codex_exec"
        and same_window(record.get("weekly_resets_at"))
    ]


def local_window_totals(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    records = list(records)
    return {
        "runs": len(records),
        "observed_tokens": sum(int(record.get("observed_tokens") or 0) for record in records),
        "input_tokens": sum(int(record.get("input_tokens") or 0) for record in records),
        "cached_input_tokens": sum(
            int(record.get("cached_input_tokens") or 0) for record in records
        ),
        "output_tokens": sum(int(record.get("output_tokens") or 0) for record in records),
        "reasoning_output_tokens": sum(
            int(record.get("reasoning_output_tokens") or 0) for record in records
        ),
    }


def preflight(*, reserve_percent: int, max_campaign_tokens: int,
              max_campaign_runs: int, minimum_headroom_percent: int = 1,
              ledger_path: Path | str = DEFAULT_LEDGER,
              snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    allowance = weekly_allowance(snapshot or query_rate_limits())
    records = current_window_records(read_ledger(ledger_path), allowance)
    totals = local_window_totals(records)
    if allowance.window_name == "unlimited":
        return {
            "allowance": allowance.as_dict(),
            "local_window": totals,
            "cost_control_enabled": False,
            "note": "provider reports limit=None/unlimited; cost controls bypassed",
        }
    if not 0 <= reserve_percent <= 100:
        raise ValueError("reserve_percent must be between 0 and 100")
    if minimum_headroom_percent < 1:
        raise ValueError("minimum_headroom_percent must be positive")
    if allowance.remaining_percent <= reserve_percent:
        raise CodexUsageGuardError(
            f"weekly Codex allowance is {allowance.remaining_percent}% remaining, "
            f"at or below the configured {reserve_percent}% reserve"
        )
    headroom = allowance.remaining_percent - reserve_percent
    if headroom < minimum_headroom_percent:
        raise CodexUsageGuardError(
            f"weekly Codex allowance has only {headroom}% headroom above the "
            f"{reserve_percent}% reserve; this turn requires "
            f"{minimum_headroom_percent}%"
        )
    if max_campaign_runs >= 0 and totals["runs"] >= max_campaign_runs:
        raise CodexUsageGuardError(
            f"local campaign run cap reached ({totals['runs']}/{max_campaign_runs})"
        )
    if max_campaign_tokens >= 0 and totals["observed_tokens"] >= max_campaign_tokens:
        raise CodexUsageGuardError(
            "local campaign token cap reached "
            f"({totals['observed_tokens']}/{max_campaign_tokens})"
        )
    return {
        "allowance": allowance.as_dict(),
        "local_window": totals,
        "cost_control_enabled": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve-percent", type=int, default=80)
    parser.add_argument("--minimum-headroom-percent", type=int, default=1)
    parser.add_argument("--max-campaign-tokens", type=int, default=2_000_000)
    parser.add_argument("--max-campaign-runs", type=int, default=12)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    return parser


def main() -> None:
    args = _parser().parse_args()
    status = preflight(
        reserve_percent=args.reserve_percent,
        minimum_headroom_percent=args.minimum_headroom_percent,
        max_campaign_tokens=args.max_campaign_tokens,
        max_campaign_runs=args.max_campaign_runs,
        ledger_path=args.ledger,
    )
    status["checked_at"] = datetime.now(timezone.utc).isoformat()
    status["ledger"] = os.fspath(args.ledger)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
