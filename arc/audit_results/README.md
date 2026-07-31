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
  player stub. This raw audit can advance while the campaign is running.
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
  point to call an unchanged named definition directly. The tracked file is the
  active-campaign snapshot frozen at 2026-07-26 21:02 CEST: GKM has 126 exact
  winning checkpoints, 99 exact adjacent transitions, 38 hard direct-call
  witnesses, and nine sharp/direct-call intersections. OPINE has four hard
  witnesses and two sharp intersections; baseline1 has zero among its 18 exact
  adjacent transitions; Retrodict releases no executable winning entry point.

The eleven joint sharp-drop/reuse findings are:

- GKM `ar25` L2: 622 to 175, calling unchanged `repeat_action`;
- GKM `g50t` L4: 2238 to 168 compressed AST-novelty bytes, with a direct call
  to unchanged `solve_unlock_macro`;
- GKM `ka59` L2: 810 to 301, calling unchanged `move_steps` and `select_at`;
- GKM `lp85` L2: 698 to 189, calling unchanged `repeat_click`;
- GKM `ls20` L7: 682 to 222, with a direct call to unchanged `execute_path`;
- GKM `m0r0` L2: 673 to 265, calling unchanged
  `follow_action_sequence`;
- GKM `sc25` L2: 1131 to 279, calling unchanged
  `move_until_level_progress` and `select_grid_cells_of_color`;
- GKM `tu93` L5: 1302 to 186, calling unchanged
  `drive_dynamic_maze_via_color`;
- GKM `tu93` L7: 1440 to 212, calling unchanged
  `drive_dynamic_directional_waypoints`;
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

## Reproduction

The frozen uniform comparison is self-contained in
`marginal-literal-reuse.json` (SHA-256
`a2b6248dff6e4ed31de0299c9882120788745986ff3e699c3888025d4fb84775`).
To recompute GKM against the mutable campaign while reusing the pinned
large-artifact rows:

```bash
python3 arc/audit_marginal_literal_reuse.py \
  --gkm-root arc/crack_lab/agent_solutions \
  --reuse-non-gkm-from-json arc/audit_results/marginal-literal-reuse.json \
  --retrodict-audit-json arc/audit_results/retrodict-solved-checkpoint-memory.json \
  --json /tmp/marginal-literal-reuse-current.json
```

The command is a drift check: after another promotion, its GKM row should differ
from the frozen snapshot. Raw external-artifact commands and pinned input hashes
are documented in `arc/manuscript/opine_world_comparison.md`.

The source analyzers are:

- `arc/audit_gkm_solved_checkpoints.py`
- `arc/audit_baseline1_artifacts.py`
- `arc/audit_opine_solved_checkpoints.py`
- `arc/audit_retrodict_artifacts.py`
- `arc/audit_marginal_literal_reuse.py`
