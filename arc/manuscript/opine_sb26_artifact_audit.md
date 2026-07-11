# OPINE-World `sb26` Artifact Audit

This is a first artifact-level audit, not a claim about all OPINE-World games.
The published archive contains 20 retained synthesis snapshots for `sb26`.
Snapshot ordering and level transitions are taken from its `run_log.txt`.

## What The Artifact Shows

`sb26` is not a clean example of an entirely fresh solver per level. Its final
`game_engine.py` implements one frame-derived, recursive splice-traversal rule
which its own comments say covers reward-confirmed levels 1--7. That is real
within-game structural reuse and must be acknowledged.

It is also not a complete learned world model over level transitions. The final
model identifies the current level by matching against `l<N>_initial.pkl` files;
on RESET and a solved ACTION5, it returns the corresponding cached entry frame.
All eight encountered level-entry frames are retained as 8,464-byte pickle
artifacts. Those bytes are level-specific description cost and cannot be treated
as free transfer.

The model itself states that its L8 planner returns `None` pending an observed
reward board. Thus exact replay on levels 1--7 does not establish predictive
generalization to the next unseen level.

## Marginal Code Ledger

The source-token counts exclude comments, whitespace, and Python formatting.
Additions/removals are token diffs between consecutive synthesis snapshots.
They are an implementation-level MDL proxy, not machine-independent Kolmogorov
complexity.

| Synthesis run | Current level after/at run | Model tokens | Added | Removed |
| --- | --- | ---: | ---: | ---: |
| 1 | L1 | 1,760 | 1,760 | 0 |
| 2 | L1 | 1,796 | 58 | 22 |
| 3 | L2 | 2,990 | 1,714 | 520 |
| 4 | L2 | 3,012 | 42 | 20 |
| 5 | L2 | 3,460 | 452 | 4 |
| 6 | L2 | 3,768 | 359 | 51 |
| 7 | L3 | 3,772 | 5 | 1 |
| 8 | L3 | 3,867 | 119 | 24 |
| 9--10 | L3--L4 | 3,867 | 0 | 0 |
| 11 | L4 | 4,315 | 544 | 96 |
| 12--13 | L4--L5 | 3,976 | 13 | 352 |
| 14--15 | L5--L6 | 4,093 | 156 | 39 |
| 16 | L6 | 4,366 | 608 | 335 |
| 17 | L6 | 4,528 | 182 | 20 |
| 18 | L7 | 4,528 | 2 | 2 |
| 19--20 | L8 | 4,643 | 176 | 61 |

The retained code grows from 1,760 to 4,643 tokens (2.64x). A sawtooth is not
reported by OPINE and cannot be inferred from final solved count. There are
zero-cost transitions at L3->L4 and L4->L5, but they are intermixed with large
within-level repair costs; the artifact therefore does not yet demonstrate a
general monotone transfer curve or a compression ledger.

## Required Controls

1. Charge every source addition, cache byte, planner patch, and LLM synthesis
   call to the level that first requires it.
2. Test the pre-level model on the next level's transitions before allowing any
   new cache, transition, or planner revision. Report both transition accuracy
   and planning success.
3. Plot per-level marginal code/data cost beside exploratory actions and model
   rewrites. Genuine reuse should reduce these measures after a mechanic is
   acquired.
4. Separate in-level dynamics from level-entry/reset dynamics. A cached frame
   is valid replay data, but it is not a predicted transition.

Until these controls are supplied across the public archive, the appropriate
claim is replay-verified, cumulative program synthesis with some demonstrated
within-game mechanism reuse. Calling it a general world model goes beyond the
evidence in this artifact.
