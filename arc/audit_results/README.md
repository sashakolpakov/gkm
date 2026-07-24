# Solved-checkpoint audit outputs

These files contain only level-solving checkpoints. Interim synthesis
revisions, repeated same-level commits, and within-level notebook states are
not treated as solver checkpoints.

The Gödel–Kolmogorov Machine is the local solver-growth system represented by
the `gkm-*` filenames. The name Gödel–Kolmogorov Machine is kept in prose before
**GKM** is used as its abbreviation below; filenames remain unchanged for
reproducibility.

## Files

- `gkm-solved-checkpoints.{csv,json}` measures the exact winning
  `legs.py + players.py + solve.py` bundle. Ordinary clears use the preserved
  pre-debrief source; four auto-solve sources are deterministically
  reconstructed from the prior retained source plus the harness's one-call
  player stub. It reports 40 exact sources across 44 claimed clears, 31
  adjacent transitions, and four missing per-level sources.
- `baseline1_gpt55_xhigh_solved_checkpoints.{csv,json}` measures the three
  core world-model modules and the complete Python authored relative to the
  fixed scaffold. It distinguishes post-solve retained snapshots from exact
  winning sources by scanning each real winning command through the end of its
  Codex turn. Of 11 retained-snapshot contractions, four have exact adjacent
  endpoints.
- `opine-solved-checkpoints.{csv,json}` selects the last synthesis before each
  positive-reward action and distinguishes synthesized-planner plans from
  analyzer plans. The executable bundle includes `game_engine.py` and runtime
  `l*.pkl` files.
- `retrodict-solved-checkpoint-memory.{csv,json}` reconstructs retained
  `playbook.md` and scratch Python. It is explicitly a memory trajectory, not
  an executable-solver trajectory.
- `marginal-literal-reuse.json` applies one cross-system test at exact winning
  checkpoints. Its conditional AST marginal is the compressed normalized
  top-level AST in the current winning program that is not a literal member of
  the preceding winning program. A reuse witness requires the winning entry
  point to call an unchanged named definition directly. GKM has ten such
  transitions, OPINE has four, baseline1 has zero among its 18 exact adjacent
  transitions, and Retrodict releases no executable winning entry point.

The joint sharp-drop/reuse findings are:

- GKM `g50t` L4: 2238 to 168 compressed AST-novelty bytes, with a direct call
  to unchanged `solve_unlock_macro`;
- GKM `ls20` L7: 682 to 222, with a direct call to unchanged `execute_path`;
- OPINE `lp85` L4: 5818 to 2550, with the identical winning planner directly
  calling three unchanged engine definitions; and
- OPINE `tu93` L3: 7091 to 2608, with the winning planner directly calling
  three unchanged engine definitions.

baseline1's four exact cumulative source/AST contractions do not pass the
literal winning-entry-point test. All 18 exact adjacent winning commands are
fresh action programs—four direct commands, six inline plans, and eight plans
passed to the generic executor—and none invokes a retained world-model
definition.

All compressed lengths use zlib level 9 and are computable description-length
upper bounds, not estimates of machine-independent Kolmogorov complexity.
Normalized-AST fields remove comments and formatting. A contraction is
interpreted only between adjacent solved levels.

The source analyzers are:

- `arc/audit_gkm_solved_checkpoints.py`
- `arc/audit_baseline1_artifacts.py`
- `arc/audit_opine_solved_checkpoints.py`
- `arc/audit_retrodict_artifacts.py`
- `arc/audit_marginal_literal_reuse.py`
