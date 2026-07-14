# GKM vs OPINE `wa30` Marginal-Complexity Audit

This note is artifact-scoped. The OPINE archive contains `wa30` but not `ls20`,
so the direct game overlap with the current GKM artifacts is `wa30`. GKM `ls20`
is included only as a positive-control trace for what explicit marginal
reuse/novelty accounting looks like.

## Artifact Scope

- OPINE source archive: `arc/opine-world`
- OPINE extracted `wa30` run: `arc/opine-artifacts/wa30/`
- OPINE `wa30` summary: `arc/opine-artifacts/wa30/summary.json`
- GKM `ls20` ledger: `arc/crack_lab/agent_solutions/ls20_legs/run.log`
- GKM `wa30` ledger: `arc/crack_lab/agent_solutions/wa30_legs/README.md`
- Figure: `arc/manuscript/figures/gkm_opine_marginal_complexity.png`

The OPINE results archive top-level game set is:
`ar25 bp35 cd82 ft09 g50t lp85 m0r0 r11l re86 sb26 sc25 sk48 sp80 su15 tn36 tr87 vc33 wa30`.
There is no `ls20` directory in the downloaded index.

## Main Finding

OPINE `wa30` does not publish a marginal-complexity ledger, but its archive is
complete enough to measure one directly. Diffing all 65 recoverable
`game_engine.py` snapshots in synthesis order and attributing added tokens to
the level active at each run gives:

| Level | Added tokens | Removed tokens | Synthesis runs |
|---:|---:|---:|---:|
| 1 | 3,243 | 665 | 3 |
| 2 | 3,978 | 1,344 | 7 |
| 3 | 1,712 | 310 | 5 |
| 4 | 2,053 | 638 | 6 |
| 5 | 1,246 | 439 | 8 |
| 6 | 1,136 | 59 | 5 |
| 7 | 266 | 2 | 4 |
| 8 | 3,310 | 465 | 17 |
| 9 | 1,444 | 305 | 10 |

Total measured marginal novelty is 18,388 added tokens against 4,227 removed.
Two diagnostics follow:

- **No reuse tooth.** The cheapest level (L7) still charges 266 new tokens, and
  every level ships a verbatim 8,464-byte entry pickle loaded by the final
  transition function. Nothing is ever near-free the way GKM `ls20` L2 (2) and
  L4 (3) are.
- **No compression tooth.** Retained source shrinks in only 8 of 64 revision
  steps, and the largest shrink is 222 tokens (1.6% of the 14,161-token final
  model). No refactor ever compresses accumulated structure into abstractions.

Honesty note: in relative terms OPINE's `wa30` floor (266/3,978 = 6.7% of peak)
is comparable to GKM `wa30`'s (30/405 = 7.4%) because `wa30`'s level structure
keeps demanding novelty from any solver. That limited caveat does not change the
artifact-level result: the OPINE archive exposes no compression tooth, no charged
reuse ledger, and uncompressed per-level literal caches. On the published artifacts,
the measurable object is accumulated replay machinery, not a compressed world model.

Supporting artifact-scoped measurements:

- 65 logged synthesis revisions on `wa30`.
- All 65 `game_engine.py` snapshots are recoverable from the archive.
- Retained normalized Python source grows from 1,882 tokens at synthesis run 1
  to 14,161 tokens at run 65, a 7.52x increase.
- The final revision uses nine level-entry pickle caches, 8,464 raw bytes each,
  for 76,176 bytes of final cached level-entry data.
- Across all `wa30` synthesis revisions, the archive contains 389 level-entry
  pickle files totaling 3,292,496 raw bytes.
- The final transition function explicitly loads `l%d_initial.pkl` on RESET and
  on goal-completing level advance. That is valid replay support for an observed
  environment, but it is not a compressed generative account of unseen level
  boards.

This does not prove "no reuse" inside OPINE. It shows that the published archive
does not expose a charged reuse ledger. The measurable published object is a
cumulative simulator program plus cached level-entry frames, not a sawtooth
marginal-C account.

## GKM Ledgers

`ls20` positive-control trace:

| Level | Marginal C |
|---:|---:|
| 1 | 43 |
| 2 | 2 |
| 3 | 45 |
| 4 | 3 |
| 5 | 72 |
| 6 | 130 |
| 7 | 67 |

`ls20` total marginal C is 362. The sawtooth is explicit: L2 reuses the existing
generic search leg, L3 adds the noise-mask leg, and L4 reuses the L1 sliding-tile
mechanic plus the L3 noise-mask structure.

`wa30` ledger:

| Level | Marginal C |
|---:|---:|
| 1 | 112 |
| 2 | 78 |
| 3 | 95 |
| 4 | 47 |
| 5 | 405 |
| 6 | 225 |
| 7 | 145 |
| 8 | 204 |
| 9 | 147 |

The complete published `wa30` ledger totals 1458. Its clean-source provenance and
hashes are indexed in `artifact_history/wa30/manifest.json`; the root checkpoint's
1243 total is the unchanged operational resume record, not the complete manuscript
ledger. The late `wa30` debrief records actual leg
reuse/refactor provenance: L6/L7/L8 share the `neutralize_then_deliver` shape,
and the recovered L9 suffix is decoded into repeated
`grab_carry_release`/`ferry_each` structure instead of being left as an opaque
61-action replay.

## OPINE `wa30` Measurements

The OPINE summary records `game_won=true`, `total_reward=9.0`,
`total_transitions=1465`, and `synthesis_runs=65`.

| Level | Synthesis runs | Run range | Source tokens, first -> last |
|---:|---:|---:|---:|
| 1 | 3 | 1-3 | 1,882 -> 2,578 |
| 2 | 7 | 4-10 | 2,647 -> 5,212 |
| 3 | 5 | 11-15 | 5,217 -> 6,614 |
| 4 | 6 | 16-21 | 6,612 -> 8,029 |
| 5 | 8 | 22-29 | 8,029 -> 8,836 |
| 6 | 5 | 30-34 | 8,836 -> 9,913 |
| 7 | 4 | 35-38 | 9,913 -> 10,177 |
| 8 | 17 | 39-55 | 10,604 -> 13,022 |
| 9 | 10 | 56-65 | 12,800 -> 14,161 |

Level advances in the run log occur at steps 48, 217, 297, 368, 613, 667, 720,
and 1160. The heavy retry levels are L8 and L9, with 17 and 10 synthesis
revisions respectively.

The OPINE source code contains machinery for `synthesis_curve.jsonl` rows with
per-round token usage, but the recovered `wa30` archive index does not include a
published `synthesis_curve.jsonl` file. Direct API cost reconstruction should
therefore be kept out of the public audit until token-usage rows or preserved
Claude/Codex chat logs are available.

The OPINE README's own cost section states that the reported sweep ran across
four Claude Max accounts at $200 per account per month, about $800 total, and
that direct API use is possible but more expensive. That is the artifact-backed
cost statement currently available.

## Interpretation

For GKM, the central artifact is a replay-validated solver-growth ledger:
reused legs are free, new legs are charged once, per-level glue is charged, and
literal paths remain charged until they are compressed by a later refactor. That
is why `ls20` exposes a sawtooth trace and `wa30` exposes explicit ferry and
neutralize/deliver provenance.

For OPINE `wa30`, the central published artifact is a cumulative executable
simulator. It is useful, but the audit question is different: how much new
description length was admitted at each level, and which structure was actually
reused rather than merely carried forward in a growing file and level-entry
cache set? The archive does not answer that question. It shows cumulative code
growth and per-level retry load; it does not show marginal-C transfer.

The defensible criticism is therefore:

> OPINE `wa30` may be an executable replay/planning artifact, but the published
> archive does not substantiate a marginally compressed world-model claim. A
> proper comparison needs a charged code/data delta per level and provenance for
> which abstractions were reused. GKM publishes that ledger; OPINE publishes a
> growing simulator plus cached level-entry frames.
