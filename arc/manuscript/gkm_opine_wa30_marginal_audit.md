# The Gödel–Kolmogorov Machine vs OPINE `wa30`: Solved-Checkpoint Audit

This note supersedes the former comparison over all 65 OPINE synthesis
revisions. Interim repairs are search history, not level-solved checkpoints.
The Gödel–Kolmogorov Machine is the verifier-gated solver-growth architecture
being audited. After naming the Gödel–Kolmogorov Machine in full, this note uses
**GKM** as its abbreviation.

## Comparable boundary rule

- GKM: use the exact `legs.py`, `players.py`, and `solve.py` source present at
  a replay-validated level clear, before any debrief rewrite.
- OPINE: use the last `game_engine.py` present before the positive-reward
  action, together with its runtime `l*.pkl` files.
- Do not credit a synthesis emitted after the reward with solving that level.
- Keep cumulative executable description separate from conditional novelty and
  from an operational reuse witness.

## OPINE `wa30`

The nine pre-solve engines are runs 3, 10, 15, 21, 29, 34, 38, 55, and 65.
The compressed engine-plus-runtime-data bundle is:

| Level | Bundle zlib bytes | Change from preceding solve | Winning-plan source |
| ---: | ---: | ---: | --- |
| 1 | 5,137 | — | analyzer |
| 2 | 8,819 | +3,682 | analyzer |
| 3 | 11,675 | +2,856 | analyzer |
| 4 | 15,595 | +3,920 | analyzer |
| 5 | 17,932 | +2,337 | analyzer |
| 6 | 20,980 | +3,048 | analyzer |
| 7 | 22,264 | +1,284 | analyzer |
| 8 | 30,854 | +8,590 | analyzer |
| 9 | 35,968 | +5,114 | analyzer |

The cumulative bundle expands at every level-to-level transition. That does not
show absence of semantic reuse: many top-level definitions are retained. It
does show that `wa30` supplies no cumulative executable contraction under this
description. Because every winning plan was analyzer-generated, the complete
winning-policy description is not present in the engine bundle.

## GKM `wa30`

The exact pre-debrief WIP archive covers levels 4–9; winning sources for the
first three levels are absent from that tree and are reported as coverage gaps
rather than reconstructed from the final promoted source. Every measurable cumulative
`legs.py + players.py + solve.py` transition expands. The complete historical
marginal ledger remains:

| Level | Historical marginal charge |
| ---: | ---: |
| 1 | 112 |
| 2 | 78 |
| 3 | 95 |
| 4 | 47 |
| 5 | 405 |
| 6 | 225 |
| 7 | 145 |
| 8 | 204 |
| 9 | 147 |

This sequence is a harness-native acquisition-cost profile, not the derivative
of the compressed cumulative bundle and not the exact checkpoint-conditioned
AST marginal. On the exact sources available for L4–L9, the conditional AST
marginals are 4237, 3510, 3033, 3804, and 808 bytes for transitions ending at
L5–L9. L9 is a sharp drop, but its winning player directly calls no unchanged
leg literal from the L8 winning checkpoint. Under the joint rule it is not a
reuse witness. The later `grab_carry_release`/`ferry_each` factorization is a
post-win debrief result, not retroactive evidence about the L9 winning program.

## Conclusion

Neither `wa30` trajectory supplies a coupled solved-checkpoint
marginal-drop/literal-reuse witness. OPINE’s nine winners are analyzer policies;
GKM’s sharp L9 marginal drop has no unchanged direct leg call. This game
therefore contributes no hard reuse event to the cross-system count.
