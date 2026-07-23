# OPINE-World `sb26` Solved-Checkpoint Audit

This note supersedes the earlier synthesis-revision table. Only the last
executable engine present *before* each positive-reward action is measured.
Post-win synthesis is not credited with solving the preceding level.

## Boundary reconstruction

The released `run_log.txt` contains seven positive-reward events. Their
pre-solve engines are runs 2, 6, 9, 12, 14, 17, and 18:

| Solved level | Engine run | Engine zlib bytes | Engine + runtime-data zlib bytes | Winning-plan source |
| ---: | ---: | ---: | ---: | --- |
| 1 | 2 | 3,329 | 3,572 | analyzer |
| 2 | 6 | 7,216 | 7,685 | analyzer |
| 3 | 9 | 7,743 | 8,436 | analyzer |
| 4 | 12 | 8,015 | 8,880 | analyzer |
| 5 | 14 | 9,017 | 10,072 | analyzer |
| 6 | 17 | 9,827 | 11,124 | analyzer |
| 7 | 18 | 10,253 | 11,765 | analyzer |

The summary claims eight rewards after a repair from another engine log, but
the released main run log contains no positive-reward event for level 8. No L8
solve checkpoint is imputed.

## Retained structure versus winning reuse

The later `game_engine.py` contains a recursive splice-traversal rule rather
than one independent transition table per level. That is literal structural
retention in the engine. It is not a winning reuse witness: none of the seven
logged positive-reward action batches came from the synthesized planner.

The conditional AST marginal has sharp drops at L3 (7814 to 2416 compressed
novelty bytes) and L7 (4461 to 2142). Both winning policies came from the
transient analyzer. The released artifacts therefore contain no direct call
from a winning executable entry point to the retained splice routine.

The artifact also retains one `l<N>_initial.pkl` entry-state file per observed
level. These files are legitimate runtime conditioning data, but they belong in
the executable description whenever the engine loads them.

## What is not established

Consequently:

- the retained world model’s recursive structure is observable;
- the complexity of the complete winning policy is not, because the transient
  analyzer policy is absent from `game_engine.py`; and
- the monotonically growing runtime bundle is not a cumulative solver
  contraction.

The hard result for `sb26` is therefore **no executable literal-reuse witness**.
OPINE’s positive executable reuse cases occur instead at `lp85` L3/L4/L6 and
`tu93` L3.
