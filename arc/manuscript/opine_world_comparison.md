# Solved-Checkpoint Comparison: The Gödel–Kolmogorov Machine, OPINE-World, baseline1, and Retrodict

Sources: [OPINE-World](https://github.com/david-courtis/opine-world),
[baseline1](https://github.com/astroseger/arc-3-agents-baseline1), and
[Retrodict](https://github.com/ryanbbrown/Retrodict). Measurements below come
from the released run artifacts, not from leaderboard totals.

The Gödel–Kolmogorov Machine is the verifier-gated solver-growth architecture
studied in the accompanying manuscript. The name Gödel–Kolmogorov Machine is
kept explicit here before **GKM** is used as its abbreviation in the table and
subsequent comparison.

## Rule of comparison

The checkpoint for a level is the retained state that actually cleared that
level. Interim synthesis revisions, repeated same-level Git commits, and
within-level notebook edits are excluded. Four quantities are kept separate:

1. cumulative executable description at the solved checkpoint;
2. conditional or marginal novelty since the preceding solved checkpoint;
3. an operational witness that a later solve invoked earlier machinery; and
4. descriptive memory such as a playbook or transcript.

A drop in one quantity is not a drop in the others. In particular, action
length, log length, context length, and notebook length are not substitutes for
executable solver complexity.

For a uniform marginal screen, normalize every top-level AST statement at
winning checkpoint \(P_k\), remove the literal statements already present in
\(P_{k-1}\), and compress the remainder with zlib-9. This conditional AST
marginal charges a same-size rewrite and gives unchanged code zero cost. A
“sharp” drop means that this marginal is at most half its preceding level’s
value. The drop is attributed to reuse only when the winning entry point
directly calls a named definition whose normalized AST is identical at the
preceding winning checkpoint.

## Result

| System | Exact solved-checkpoint coverage | Conditional AST marginal | Direct literal world-model reuse |
| --- | --- | --- | --- |
| GKM | 40 winning sources across 44 claimed clears; 31 exact adjacent transitions | 16 decreases among 29 comparable level-to-level marginals; 4 are half-or-more drops | 10 winning players directly call unchanged leg definitions. The sharp-drop/reuse intersections are `g50t` L4, \(2238\to168\), calling unchanged `solve_unlock_macro`, and `ls20` L7, \(682\to222\), calling unchanged `execute_path`. |
| OPINE-World | 153 positive-reward events; 146 with a pre-solve engine and 121 adjacent transitions | 49 decreases among 115 comparable marginals; 14 are half-or-more drops | 4 synthesized-planner wins directly call unchanged engine definitions. Two coincide with sharp drops: `lp85` L4, \(5818\to2550\), and `tu93` L3, \(7091\to2608\). |
| baseline1 GPT-5.5 xHigh | 160 post-solve retained snapshots for 174 clears; 50 exact winning sources and 18 exact adjacent transitions | 5 decreases among 8 comparable marginals; 0 half-or-more drops | 0. Every exact adjacent winning command is a fresh literal action program: 4 direct-action commands, 6 inline plans, and 8 plans passed to `plan_executor.py`. |
| Retrodict | 170 solved memory checkpoints; 145 adjacent memory transitions | No executable marginal is identifiable | 0 executable witnesses: the release contains playbook memory and limited scratch Python, not winning executable entry points. |

The conclusion of the exact winning-entry-point test is deliberately asymmetric:
OPINE has hard level-to-level executable reuse; baseline1 does not, and Retrodict
does not release an executable winning entry point on which to run the test. GKM
has the strongest literal-leg evidence in the measured exact set.

The baseline1 repositories still contain four exact contractions in their
retained authored-Python bundle: `ar25` L4→L5, `cd82` L5→L6, `lp85` L6→L7,
and `sb26` L6→L7. All four survive normalized-AST compression. They are not,
however, evidence of solver reuse under the winning-entry-point rule. The real
winning commands invoke no retained world-model definition. They execute fresh
action literals directly, through inline loops, or through the generic
`plan_executor.py`. Retaining 18–23 unchanged core definitions elsewhere in the
workspace does not make those definitions part of the winning program.

The 153 OPINE events are the coverage of the released main logs, not an attempt
to replace the reported 160-level headline. The archive lacks an `s5i5`
summary, the repaired `sb26` summary contains one more reward than its main
log, and `tr87` has the converse mismatch. Missing boundary programs are not
imputed.

GKM’s cumulative solver grows at every exact adjacent transition, but the
checkpoint-conditioned AST test supplies the direct evidence. Ten winning
players call unchanged leg literals. At `g50t` L4 the conditional marginal
collapses by 92.5% while the winning one-call player invokes unchanged
`solve_unlock_macro`; at `ls20` L7 it falls by 67.4% while the player invokes
unchanged `execute_path`. The other two sharp GKM drops, `ft09` L2 and `wa30`
L9, have no unchanged direct leg call and are not classified as reuse. This is
why the marginal screen and literal-call test must be coupled.

The new `ft09` L6 checkpoint is the tenth direct witness: its winning player
calls the literally unchanged `solve_coupled_key_board` acquired at L5. Its
conditional AST marginal decreases from 5008 to 3730 bytes, so it is reuse but
not a half-or-more drop under the stated sharpness rule. This distinction also
prevents the historical two-unit `marginal_C` charge from being substituted for
the normalized-AST comparator.

## OPINE-World refutes the all-levels-new hypothesis

OPINE’s released log distinguishes synthesized-planner plans from transient
analyzer plans. That distinction leaves four executable planner wins, and all
four directly call engine definitions that are literal matches to the preceding
winning checkpoint. Two also meet the sharp-drop criterion. At `lp85` L4, the
same normalized `planner` literal that won L3 directly calls unchanged
`_cross_components`, `_cursor_pairs`, and `_square_blocks`; the conditional AST
marginal falls from 5818 to 2550 compressed bytes. At `tu93` L3, the winning
planner directly calls unchanged `_find_player`, `_goal_topleft`, and
`transition_function`; the marginal falls from 7091 to 2608.

Thus the categorical statement “OPINE solves every level as a new task” is
false. The positive evidence is sparse—four executable reuse wins among 121
adjacent engine transitions—but literal and operational. The 14 sharp OPINE
drops are not all reuse: 12 occur on analyzer-solved levels whose transient
winning policy is absent. `tr87` L6 and `tu93` L8 even have zero retained-engine
novelty, yet their analyzer policies prevent a literal executable reuse
certificate.

## Retrodict is a memory result, not an executable sawtooth

Retrodict’s `playbook.md` is deliberately rewritten and survives context
resets. Replaying its write/edit traces reconstructs 76 between-level memory
contractions. Twenty-three of the 25 selected runs have no substantive scratch
Python at any solved checkpoint. `cn04` and `r11l` account for the four
transitions that do; their scratch trajectory has three expansions and zero
contractions. There is no released executable winning entry point, hence no
executable marginal and no literal code-reuse witness.

## Reproducible outputs

- `arc/audit_results/gkm-solved-checkpoints.json`
- `arc/audit_results/baseline1_gpt55_xhigh_solved_checkpoints.json`
- `arc/audit_results/opine-solved-checkpoints.json`
- `arc/audit_results/retrodict-solved-checkpoint-memory.json`
- `arc/audit_results/marginal-literal-reuse.json`

The source analyzers are `arc/audit_gkm_solved_checkpoints.py`,
`arc/audit_baseline1_artifacts.py`,
`arc/audit_opine_solved_checkpoints.py`,
`arc/audit_retrodict_artifacts.py`, and
`arc/audit_marginal_literal_reuse.py`.
