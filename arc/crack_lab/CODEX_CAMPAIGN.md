# Headless Codex campaign policy

This document records the safety and accounting rules for extending the
Gödel–Kolmogorov Machine with `gpt-5.6-sol`. It separates provider allowance,
executable success, and solver structure; token count alone is not treated as a
price meter for ChatGPT-plan use.

## Invariants

1. Query the live seven-day bucket before every model turn. Never infer it from a
   shorter bucket and never redeem a reset credit automatically.
2. Serialize preflight, execution, postflight, and durable JSONL accounting with one
   campaign lock.
3. Pin model and reasoning effort. Disable web search and shell network access, strip
   API-key environment variables, and never expand sandbox approval inside the agent.
4. Preserve an immutable transcript. Source access, prior-solution access,
   agent-authored private-runtime introspection, or external-network use invalidates
   the attempt before verification or promotion.
5. Admit one level and one paid proposal per transaction. Debrief is a distinct paid
   turn; the cheap campaign disables it.
6. Promote only after fresh replay validates the complete path to the new depth.
   Failed and interrupted work remains WIP and never becomes a solve checkpoint.
7. Attempt incumbent legs before invention. Generate `solver_index.md` so the agent
   can inspect signatures and direct calls without repeatedly reading the full solver.
8. Disable nested transient retries. Infrastructure failure returns to the outer
   scheduler for a new live admission decision.
9. Keep a 20% weekly reserve, with 60-turn and 32-million-observed-token secondary
   caps.
10. Start every fresh continuation on medium. Permit one bounded high rescue after a
    clean medium failure, then quarantine the frontier if high also fails.

The structural JSONL taint scanner evaluates agent-authored command, web-search, and
file-change surfaces rather than tool output. This distinction is necessary because a
traceback from the permitted `env.clone()` API can print internal harness field names.
Agent-authored workspace and promotion-evidence files are scanned separately, so an
actual `env._game` access still fails.

## Completed reset-window campaign

The campaign began with a reset allowance of 100%. Paid solving stopped in the
protected tail of the window; no solver turn was launched below the 20% reserve.
The campaign executed 49 bounded proposal turns and promoted 23 new levels. The
complete local frontier is now 67 levels across 23 games.

| Phase | Effort | Attempts | Clears | Displayed points | Points/clear |
|---|---|---:|---:|---:|---:|
| cold L1 | medium | 4 | 3 | 4 | 1.333 |
| cold L1 | high | 15 | 12 | 18 | 1.500 |
| continuation L2+ | medium | 23 | 7 | 39 | 5.571 |
| continuation L2+ | high | 7 | 1 | 14 | 14.000 |

The phase split is essential. High continuation turns were escalations of already
failed medium frontiers and therefore cannot identify a causal effort penalty. High
was effective on cold entry and cleared `re86` L4 after medium failed, but the campaign
does not support the hypothesis that high generally buys smaller solvers. Medium
produced all four new exact literal-reuse wins and both new sharp coupled witnesses.

The pooled table is descriptive, not the decision rule for the next window. The next
campaign evaluates high as an escalation: a qualifying high attempt must follow a
medium failure on the same game and target level. Its total displayed-point charge,
replay-validated rescue count, and points per rescue are reported directly. This is
not a randomized estimate—medium may improve the retained WIP—but it answers whether
paying for the additional high turn is useful under the actual sequential policy.
Applied retrospectively, the ledger contains six qualifying high-after-medium turns,
one replay-validated rescue (`re86` L4), and 12 displayed points charged: a 1/6 rescue
rate and 12 points per rescue.

The final exact audit contains 63 winning sources for 67 clears and 39 exact adjacent
transitions. Twenty-one conditional normalized-AST marginals decrease, six decrease by
at least half, 14 winning entry points directly call unchanged legs, and four sharp
drops coincide with such direct calls.

## Admission headroom

Displayed allowance is integer-valued and coarse. For a requested effort and wall
limit, the policy projects the worst completed-turn point/minute rate and adds one
displayed point, subject to conservative floors of four points for medium and six for
high. Timed-out full-budget turns remain valid rate observations.

An operator-interrupted duplicate turn remains charged in campaign efficiency but is
excluded from the rate projection. Extrapolating its two rounded points over 53 seconds
had produced a false 20-point high requirement. The regression test now proves that
an ordinary two-point/eight-minute high observation yields the six-point floor.

No turn is admissible while the live allowance is at or below the 20% reserve plus
the empirical per-turn headroom. Paid work remains frozen until reset. The saved
post-reset plan contains only the first medium item (`re86` L5); after that, the
scheduler re-ranks from fresh artifacts and a fresh allowance read.

Two versioned cold-start interventions are now eligible after `re86`: `tn36` L1 uses
a time-isolation scaffold, and `bp35` L1 uses symbolic phase/macro search. Their older
failures remain in cost accounting but no longer quarantine the changed intervention.
Each resumed clean room receives a generated `frontier_brief.md` containing only prior
agent progress messages and a compact probe index; bulky command output is excluded.

## Evidence locations

- Usage ledger: `arc/crack_lab/runs/codex_campaign_usage.jsonl`
- Promoted artifacts: `arc/crack_lab/agent_solutions/*_legs/`
- Exact checkpoint audit: `arc/audit_results/gkm-solved-checkpoints.json`
- Marginal/reuse audit: `arc/audit_results/marginal-literal-reuse.json`
- Operating strategy: `arc/crack_lab/CHEAP_CAMPAIGN.md`

Local status is non-metered:

```sh
python3 arc/crack_lab/codex_campaign_status.py
```

Use `--live` only when a fresh provider read is required. It consumes no model turn
but durably records the allowance snapshot.
