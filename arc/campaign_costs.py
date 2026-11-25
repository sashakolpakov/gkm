"""Audit recorded Codex campaign tokens and compute explicit API-equivalent costs.

This reads accounting evidence only. It never imports or executes solver code.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path


TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
PRICING_URL = "https://developers.openai.com/api/docs/models/gpt-6.1-sol"


def execution_rows(raw):
    rows = []
    seen = set()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "codex_exec":
            continue  # Outcomes/corrections classify a turn; they are not new charges.
        identity = event["thread_id"]
        if not identity or identity in seen:
            raise ValueError(f"missing or duplicate execution identity at line {number}")
        seen.add(identity)
        if event["model"] != "gpt-6.1-sol":
            raise ValueError(f"unpriced model: {event['model']}")
        for field in TOKEN_FIELDS:
            value = event.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid {field} at line {number}")
        if event["cached_input_tokens"] > event["input_tokens"]:
            raise ValueError("cached input exceeds total input")
        if event["reasoning_output_tokens"] > event["output_tokens"]:
            raise ValueError("reasoning output exceeds total output")
        if type(event.get("usage_reported")) is not bool:
            raise ValueError("usage_reported must explicitly distinguish unknown usage")
        rows.append(event)
    return rows


def capture_usage(rows, root):
    captures = []
    for row in rows:
        workspace, transcript = row["workspace"], row["transcript"]
        if Path(workspace).name != workspace or Path(transcript).name != transcript:
            raise ValueError("capture names must be single path components")
        path = root / workspace / transcript
        digest = hashlib.sha256()
        usage = None
        for line in path.open("rb"):
            digest.update(line)
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and item.get("type") == "turn.completed":
                usage = item.get("usage")
        if digest.hexdigest() != row["protected_transcript_sha256"]:
            raise ValueError(f"capture hash mismatch: {workspace}")
        if row["usage_reported"]:
            if not isinstance(usage, dict):
                raise ValueError(f"reported usage lacks terminal evidence: {workspace}")
            for field in TOKEN_FIELDS:
                if usage.get(field) != row[field]:
                    raise ValueError(f"terminal {field} mismatch: {workspace}")
        captures.append({"thread_id": row["thread_id"], "workspace": workspace,
                         "transcript": transcript, "sha256": digest.hexdigest(), "usage": usage})
    return captures


def totals(rows, usage_by_thread):
    known = [r for r in rows if r["usage_reported"]]
    result = {field: sum(r[field] for r in known) for field in TOKEN_FIELDS}
    result.update(turns=len(rows), turns_with_reported_usage=len(known),
                  turns_without_reported_usage=len(rows) - len(known))
    writes = sum((usage_by_thread.get(r["thread_id"]) or {}).get("cache_write_input_tokens", 0)
                 for r in known)
    result["cache_write_input_tokens"] = writes
    result["uncached_read_input_tokens"] = result["input_tokens"] - result["cached_input_tokens"] - writes
    if result["uncached_read_input_tokens"] < 0:
        raise ValueError("cache reads plus writes exceed total input")
    result["nonreasoning_output_tokens"] = result["output_tokens"] - result["reasoning_output_tokens"]
    result["input_plus_output_tokens"] = result["input_tokens"] + result["output_tokens"]
    u, c, w, o = (result[k] for k in ["uncached_read_input_tokens", "cached_input_tokens",
                                     "cache_write_input_tokens", "output_tokens"])
    million = Decimal(1000000)
    result["usd_standard_reported_cache"] = str((2*u + Decimal('.10')*c + Decimal('2.50')*w + 10*o) / million)
    result["usd_if_all_requests_long_context"] = str((4*u + Decimal('.20')*c + 5*w + 15*o) / million)
    result["usd_without_cache_discount"] = str((2*result["input_tokens"] + 10*o) / million)
    result["unpriced_turn_ids"] = [r["thread_id"] for r in rows if not r["usage_reported"]]
    return result


def build_report(raw, cutoff_thread, captures):
    rows = execution_rows(raw)
    cuts = [i for i, r in enumerate(rows) if r["thread_id"] == cutoff_thread]
    if len(cuts) != 1:
        raise ValueError("cutoff must identify exactly one recorded execution")
    evidence = {item["thread_id"]: item.get("usage") for item in captures}
    if len(captures) != len(rows) or set(evidence) != {r["thread_id"] for r in rows}:
        raise ValueError("capture evidence must cover every execution")
    hashes = {item['thread_id']: item['sha256'] for item in captures}
    for row in rows:
        if hashes[row['thread_id']] != row['protected_transcript_sha256']:
            raise ValueError("capture evidence differs from ledger hash")
        if row['usage_reported']:
            usage = evidence[row['thread_id']]
            if not isinstance(usage, dict) or any(usage.get(k) != row[k] for k in TOKEN_FIELDS):
                raise ValueError("capture counters differ from ledger")
            writes = usage.get('cache_write_input_tokens', 0)
            if type(writes) is not int or writes < 0 or writes + row['cached_input_tokens'] > row['input_tokens']:
                raise ValueError("invalid cache-write input count")
    cut = cuts[0] + 1
    phases = {"through_cutoff": rows[:cut], "after_cutoff": rows[cut:], "all_recorded": rows}
    grouped = {}
    for phase, items in phases.items():
        games = defaultdict(list)
        for row in items:
            games[row["game"]].append(row)
        grouped[phase] = {game: totals(values, evidence) for game, values in sorted(games.items())}
    return {
        "schema": 1, "model": "gpt-6.1-sol", "ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": {"thread_id": cutoff_thread, "workspace": rows[cut-1]["workspace"],
                   "started_at": rows[cut-1]["started_at"], "execution_ordinal": cut,
                   "selection": "retrospective reporting cutoff; later work is accounted separately"},
        "summary": {phase: totals(items, evidence) for phase, items in phases.items()},
        "by_game": grouped,
        "pricing": {"checked_on": "2026-11-25", "url": PRICING_URL,
                    "currency": "USD", "rates_per_million": {"input": "2.00", "cached_input": "0.10",
                    "cache_write": "2.50", "output_including_reasoning": "10.00"},
                    "long_context": "For requests over 272K input tokens, input/cache rates double and output is 1.5 times the standard rate.",
                    "scope": "API-equivalent valuation of reported tokens, not an invoice. Per-request context sizes are not in the CLI capture; both standard and all-long-context scenarios are shown. Cache writes use the reported counter. Unknown usage is not valued at zero."},
        "accounting": "Each codex_exec records cumulative usage for one proposer turn, potentially spanning many API requests. Count that execution once, including failed, discarded and infrastructure work; corrections/outcomes are not extra executions. Cached input is a subset of input and reasoning is a subset of output; neither is added twice. Host orchestration, manuscript editing, local compute and later API action replay are outside proposer-token accounting.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--transcript-root", type=Path, required=True)
    parser.add_argument("--cutoff-thread", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.ledger.read_bytes()
    captures = capture_usage(execution_rows(raw), args.transcript_root)
    report = build_report(raw, args.cutoff_thread, captures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "usage.jsonl").write_bytes(raw)
    (args.output_dir / "capture_usage.json").write_text(json.dumps(captures, indent=2) + "\n")
    (args.output_dir / "token_costs.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Recorded proposer tokens and API-equivalent cost", "",
             "The reporting cutoff is execution " + str(report['cutoff']['execution_ordinal']) + " (`" + report['cutoff']['workspace'] + "`). It is retrospective; subsequent plateau attempts remain in the accounting.", "",
             "| Scope | Turns | Input (includes cached) | Cached input | Output (includes reasoning) | Reasoning | Standard USD | All-long-context USD | Missing usage turns |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for phase, values in report["summary"].items():
        lines.append("| " + " | ".join([phase, *[f"{values[k]:,}" for k in ["turns", *TOKEN_FIELDS]],
                     f"{Decimal(values['usd_standard_reported_cache']):.2f}",
                     f"{Decimal(values['usd_if_all_requests_long_context']):.2f}",
                     str(values['turns_without_reported_usage'])]) + " |")
    lines += ["", report["accounting"], "", report["pricing"]["scope"], "",
              str(report['summary']['all_recorded']['turns_without_reported_usage']) + " historical captures have no terminal usage report. Their counters are unknown, not zero-cost executions. The scenarios price only reported usage; they are not bounds on the missing tokens.", "",
              "[Official GPT-6.1 Sol prices](" + PRICING_URL + ") were checked on 2026-11-25. Rates per million tokens: input $2, cached input $0.10, cache writes $2.50, output $10. Reasoning is already included in output. No Batch/Flex/Fast discount or premium is assumed.", ""]
    (args.output_dir / "README.md").write_text("\n".join(lines))
    with (args.output_dir / "per_game.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=["scope", "game", "turns", *TOKEN_FIELDS,
                               "usd_standard_reported_cache", "turns_without_reported_usage"])
        writer.writeheader()
        for phase, games in report["by_game"].items():
            for game, values in games.items():
                writer.writerow({key: ({"scope": phase, "game": game, **values})[key] for key in writer.fieldnames})
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
