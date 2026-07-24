# Cost-aware GPT-5.6 solver campaign

This is the operating record for extending the Gödel–Kolmogorov Machine with
`gpt-5.6-sol` while protecting the shared seven-day Codex allowance. Displayed
weekly percentage is the hard cost signal: timed-out JSONL streams can omit final
token usage, so reported token totals are a diagnostic lower bound rather than an
admission meter.

## Current verified frontier

The promoted artifacts contain **67 replay-valid clears across 23 of 25 games**:

| Depth | Games |
|---:|---|
| L9 | `wa30` |
| L7 | `ls20` |
| L6 | `ft09`, `r11l`, `tr87` |
| L5 | `g50t` |
| L4 | `ar25`, `re86`, `sp80` |
| L2 | `cd82`, `m0r0` |
| L1 | `cn04`, `dc22`, `ka59`, `lf52`, `lp85`, `s5i5`, `sb26`, `sc25`, `sk48`, `su15`, `tu93`, `vc33` |
| L0 | `bp35`, `tn36` |

The reset-window campaign added 23 clears: 15 cold L1 entries and eight
continuations. The final bounded high escalation promoted `re86` L4. The last
post-solve reading entered the protected tail of the window. The solver dispatcher
launched nothing below its 20% reserve; paid work remains frozen until reset.

## What medium versus high actually showed

Failures are charged to the arm that produced them. Cold entry and retained-solver
continuation are separated because their difficulty and context sizes differ.

| Phase | Effort | Attempts | Clears | Displayed points | Points/clear |
|---|---|---:|---:|---:|---:|
| cold L1 | medium | 4 | 3 | 4 | 1.333 |
| cold L1 | high | 15 | 12 | 18 | 1.500 |
| continuation L2+ | medium | 23 | 7 | 39 | 5.571 |
| continuation L2+ | high | 7 | 1 | 14 | 14.000 |

This is descriptive, not a causal effort comparison. High continuation turns were
selected after medium had already failed, so they faced a harder cohort. The evidence
supports three narrower conclusions:

1. High was effective and inexpensive on cold L1 acquisition, but medium was not
   meaningfully worse in the small matched subset.
2. Medium produced all four new literal-reuse wins and both new sharp
   reuse/drop couplings (`ar25` L2 and `m0r0` L2). This window does not show that
   high writes more compressed or more reusable continuations.
3. Bounded high escalation remains useful: `re86` L4 succeeded for two displayed
   points after the medium attempt failed. It should not be the default arm.

The current-window exact solver-quality medians are confounded by frontier. Medium's
ten clears have median conditional normalized-AST marginal 521 bytes and four literal
reuse wins; high's 13 clears have median 1131 bytes and no literal reuse win. Most high
clears are cold L1 entries, so these medians are evidence about the realized campaign,
not an intrinsic model property.

## Admission policy

- Read the live seven-day bucket before every paid turn.
- Maintain a 20% terminal reserve. No reset credit is redeemed automatically.
- Serialize turns with the campaign lock.
- Admit one proposal per item. Paid debriefs and automatic transient retries remain
  disabled.
- Use medium first on an unattempted continuation. Escalate once to high only after a
  clean medium failure and only when empirical full-turn headroom remains.
- Quarantine a frontier after medium and high both fail, or after two high failures.
- Resume clean WIP, but never restore a snapshot with an actual taint finding.
- Refresh replay, exact-checkpoint, literal-reuse, and taint audits after every win.

Headroom is the worst observed displayed-point burn rate projected to the requested
wall time, plus one point, subject to conservative floors. Operator-interrupted turns
remain real costs in points-per-clear accounting but are excluded from burn-rate
extrapolation: integer rounding on a 53-second interruption had incorrectly inflated
an eight-minute high bound to 20 points. The tested corrected high bound is six points.

## Next-window strategy

At the next reset, continue in this order:

1. Try `re86` L5 with medium: it is a fresh continuation on the deepest newly
   advancing artifact.
2. Re-rank immediately. The current redesigned cold-start candidates are `tn36` L1
   and `bp35` L1, both on medium. Their versioned scaffolds supersede—without erasing
   the cost of—the earlier failed policy: `tn36` isolates time/turn dynamics before
   coordinate search, while `bp35` searches symbolic phase macros instead of raw
   frames.
3. Use bounded high escalation on a medium-failed frontier only when its live
   priority remains above those materially changed cold starts.
4. Re-rank after every clear. A newly verified level becomes the only checkpoint;
   interim commits and same-level edits never enter the complexity denominator.

The saved plan freezes only this first item. Every later choice is regenerated from
the live allowance, the new verified frontier, and charged failures. High is evaluated
as a rescue policy: count only high turns made after a medium failure on the same
game/level, and report their replay-validated rescue rate and total displayed points.
This answers the operational cost question without pretending that pooled medium and
high cohorts were randomized. Retrospectively, the current ledger contains six such
high-after-medium attempts, one replay-validated rescue (`re86` L4), and 12 displayed
points charged in total: a 1/6 rescue rate and 12 points per rescue. This remains
observational because the medium turn may have improved retained WIP.

Before a paid turn, the clean room now generates `frontier_brief.md` from the latest
clean JSONL agent messages and a file-size index of preserved probes. It excludes raw
command output, source dumps, and pixel traces. Observations remain labelled
unverified and must be reproduced, but the next proposer no longer spends its opening
minutes rediscovering which probes exist or rereading a 50–70 kB transcript.

The generic perception seed also exposes `bounded_replay_bfs`: its queue stores
compact paths and reconstructs nodes from a root clone. This directly addresses the
observed `bp35` failure mode in which recursively deep-copying thousands of evolved
Arena clones dominated for more than two minutes without producing a frontier report.

The current run caps are 60 turns and 32 million observed tokens per window. They are
secondary safeguards because timeout token usage is incomplete; live reserve and
per-turn headroom remain authoritative.

## Adaptive wall-time sizing

Per-turn wall time is no longer a static per-phase constant. It is sized from each
arm's own replay-validated solve-time distribution so that no historically successful
solve would have been truncated (`codex_campaign_status.recommend_minutes`, exercised
by `test_codex_campaign_timeout.py`). Measured over the 59 replay-validated proposal
turns in the ledger:

| Phase | Effort | Validated solves | Slowest solve | Static cap | Adaptive cap |
|---|---|---:|---:|---:|---:|
| cold L1 | medium | 3 | 4.0 min | 6 | **5** |
| cold L1 | high | 12 | 6.0 min (at cap) | 6 | **7** |
| continuation L2+ | medium | 8 | 7.5 min | 8 | **9** |
| continuation L2+ | high | 7 | 12.0 min (at cap) | 8 | **14** |

The rule is `ceil(slowest_validated_solve × 1.15)`, floored at five minutes for medium
and six for high, capped at a fifteen-minute secondary ceiling, and never below the
slowest observed solve. An arm with fewer than three validated solves keeps its static
cap. Every frontier row records the basis, the solve-sample size, and the slowest solve
so the number is auditable rather than asserted.

The dominant change is continuation high, 8 → 14 minutes. Two of the seven high
continuation solves used the full twelve minutes; the former eight-minute high
recommendation would have truncated them, converting a cheap solve into a total-loss
timeout that charges displayed points for zero levels. An under-sized cap is the most
expensive outcome, so the correction is to size up. The only reduction is cold medium,
6 → 5 minutes: its slowest solve was four minutes and its single failure still timed out
at six without solving, so five loses no observed solve.

This policy does not shorten turns to conserve allowance; it stops wasting whole turns
on truncated solves. Net displayed-point impact is therefore not a simple saving —
continuation high and the two censoring-margin arms cost slightly more per admitted
turn — but the per-turn headroom formula still gates admission, so a longer high turn is
admitted only when the live allowance is ample.

Honest uncertainty. The samples are small (3 / 12 / 8 / 7 solves), so cold medium is the
thinnest override. The high arms are right-censored: their solves cluster against the
historical caps, so the true solve-time tail is unknown and may exceed what was observed;
the 1.15 margin decensors only modestly, which is a further reason the rule sizes censored
arms up rather than down. These are per-arm wall-time sizings, not a causal medium-versus-
high comparison. Failures almost always run to the full cap with no early self-abort, and
clean WIP is resumed on the next turn; an early-abort or shorter-cap-plus-resume lever is
possible future work and is not evaluated here.

## Integrity status

The final audit reports:

- 138 canonical promoted files, zero taint hits;
- 2 future frontier scaffolds, zero taint hits;
- 23 promotion chains, zero taint or integrity failures;
- 63 exact winning sources for 67 clears;
- 39 exact adjacent transitions, 21 conditional-AST marginal decreases, six sharp
  drops, 14 direct literal-reuse wins, and four sharp coupled witnesses.

The transcript scanner parses Codex JSONL structurally. It scans requested commands,
web-search items, changed paths, and separately preserved file contents; it does not
misclassify a public `env.clone()` traceback merely because command output mentions
the harness's internal `_game` field. Actual agent-authored private-runtime commands
remain fatal.

Current machine-readable status:

```sh
python3 arc/crack_lab/codex_campaign_status.py
python3 arc/crack_lab/codex_campaign_policy.py
```
